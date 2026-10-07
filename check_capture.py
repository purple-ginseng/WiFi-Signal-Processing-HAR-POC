"""Quick quality check for a BFM capture before using it for training.

Usage:
    python check_capture.py <file> [<file> ...] [--split 60] [--labels standing,walking] [--window 5]

<file> is a session CSV from bfm_real_imag_csv/ (written by main_gui.py / main_gui3.py),
or a .pcap from bfm_pcap/ / live_bfm_pcap/ (adds RSSI and channel, needs tshark).

For each file it reports:
  * integrity  - packet rate, gaps, incomplete rows, duplicate/out-of-order timestamps, transmitters,
                 SU vs MU feedback frames
  * per segment (the two activities either side of --split):
      phi_jump   mean |packet-to-packet phi change| in codebook steps, over contiguous runs
      coherence  correlation of that change between neighbouring subcarriers
                 (body motion moves neighbouring subcarriers together; quantisation noise doesn't)
      levels     distinct phi values per subcarrier
      motion     mean per-packet change of (cos phi, sin phi, psi) across all subcarriers
  * separation - within-session AUC of the --window rolling motion index, second vs first segment

Reference points (nofoil training data, SU frames, 2025-10): phi_jump ~0.5 standing / ~1.6 walking,
coherence ~0.5 standing / ~0.9 walking.
"""
import argparse
import subprocess
import sys

import numpy as np
import pandas as pd

TSHARK_PATH = "/usr/bin/tshark"

# Must match decompress_bfm_from_angles in bfmtool/preprocessor.py, which decodes every frame as (9, 7).
DECODER_PHI_BITS, DECODER_PSI_BITS = 9, 7

# Per-frame codebook actually sent: SU codebook 1 -> (6, 4), MU codebook 1 -> (9, 7).
SU_PHI_BITS, SU_PSI_BITS = 6, 4

GAP_S = 0.5          # inter-packet gap reported as a dropout
CONTIGUOUS_S = 0.15  # packets closer than this count as one unbroken run
BOUNDARY_GUARD_S = 2.0


def angle_indices(z):
    """Recover integer (phi, psi) codebook indices from the stored complex ratio h1/h2.

    ratio = exp(j*phi) * cot(psi), so phi = arg(ratio) and psi = arctan(1 / |ratio|).
    """
    phi = np.mod(np.angle(z), 2 * np.pi)
    psi = np.arctan2(1.0, np.abs(z))
    k_phi = (phi / np.pi - 1 / 2 ** DECODER_PHI_BITS) * 2 ** (DECODER_PHI_BITS - 1)
    k_psi = (psi / np.pi - 1 / 2 ** (DECODER_PSI_BITS + 2)) * 2 ** (DECODER_PSI_BITS + 1)
    residual = np.nanmax(np.abs(np.r_[k_phi.ravel() - np.rint(k_phi.ravel()),
                                      k_psi.ravel() - np.rint(k_psi.ravel())]))
    return np.rint(k_phi), np.rint(k_psi), residual


def load_csv(path):
    df = pd.read_csv(path)
    real = sorted((c for c in df.columns if c.endswith("_Ratio_Real")),
                  key=lambda c: int(c.split("_")[1]))
    if not real:
        raise ValueError("no SCIDX_*_Ratio_Real columns (expected a bfm_real_imag_csv/ file)")
    imag = [c.replace("_Real", "_Imag") for c in real]
    t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(float)
    z = df[real].to_numpy(float) + 1j * df[imag].to_numpy(float)
    tx = df["transmitter_address"].dropna().unique().tolist() if "transmitter_address" in df else []
    return t, z, tx


def integrity(t, z):
    """Row-level problems, and the cleaned (sorted, de-duplicated, complete) row order."""
    incomplete = np.isnan(z).any(axis=1)
    order = np.argsort(t, kind="stable")
    nonmono = int((np.diff(t) < 0).sum())
    keep = order[~incomplete[order]]
    ts = t[keep]
    first = np.r_[True, np.diff(ts) > 0]
    dup_ts = int((~first).sum())
    keep = keep[first]
    tk = t[keep]
    d = np.diff(tk)
    info = dict(rows=len(t), incomplete=int(incomplete.sum()), nonmono=nonmono,
                dup_ts=dup_ts,
                duration=float(tk[-1] - tk[0]) if len(tk) > 1 else 0.0,
                rate=len(tk) / max(tk[-1] - tk[0], 1e-9) if len(tk) > 1 else 0.0,
                max_gap=float(d.max()) if len(d) else 0.0, gaps=int((d > GAP_S).sum()))
    return keep, info


def segment_stats(t, kp, ks, modulus):
    """Motion statistics for one segment; t relative seconds, kp/ks integer indices."""
    d = np.diff(t)
    run = (d > 0) & (d < CONTIGUOUS_S)
    dp = ((np.diff(kp, axis=0) + modulus // 2) % modulus) - modulus // 2
    dp = dp[run]
    if len(dp) < 2:
        return None
    jump = np.abs(dp).mean()
    a, b = dp[:, :-1].ravel(), dp[:, 1:].ravel()
    coherence = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0.0
    levels = np.median([len(np.unique(kp[:, k])) for k in range(kp.shape[1])])
    return dict(n=len(t), phi_jump=jump, coherence=coherence, levels=levels,
                static=float((kp.std(0) == 0).mean()))


def motion_index(kp, ks, modulus, psi_levels):
    ph = kp * 2 * np.pi / modulus
    x = np.hstack([np.cos(ph), np.sin(ph), ks / (psi_levels - 1)])
    return np.r_[0.0, np.abs(np.diff(x, axis=0)).mean(1)]


def rolling_auc(t, mi, split, window):
    """AUC of the trailing-window mean motion index, segment 2 vs segment 1 (Mann-Whitney)."""
    score = np.array([mi[(t > ti - window) & (t <= ti)].mean() for ti in t])
    ok = (t >= window) & ((t < split - BOUNDARY_GUARD_S) | (t > split + BOUNDARY_GUARD_S + window))
    s, y = score[ok], t[ok] > split
    if y.all() or not y.any():
        return None
    ranks = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pcap_summary(path, split, labels):
    """RSSI/channel/feedback type of the BFM frames in a pcap (piped via stdin to dodge AppArmor)."""
    fields = ["frame.time_epoch", "radiotap.dbm_antsignal", "wlan_radio.channel",
              "wlan.vht.mimo_control.feedbacktype", "wlan.ta"]
    cmd = [TSHARK_PATH, "-r", "-", "-Y", "wlan.fixed.category_code == 21", "-T", "fields"]
    for f in fields:
        cmd += ["-e", f]
    with open(path, "rb") as fh:
        out = subprocess.run(cmd, stdin=fh, capture_output=True, text=True).stdout
    rows = [line.split("\t") for line in out.splitlines() if line.strip()]
    if not rows:
        print("  no BFM report frames found")
        return
    df = pd.DataFrame(rows, columns=fields)
    t = pd.to_numeric(df["frame.time_epoch"], errors="coerce").to_numpy()
    t = t - t[0]
    rssi = pd.to_numeric(df["radiotap.dbm_antsignal"].str.split(",").str[0], errors="coerce").to_numpy()
    mu = (df["wlan.vht.mimo_control.feedbacktype"].str.strip() == "0x000001").mean()
    print(f"  frames {len(df)}  channel {', '.join(df['wlan_radio.channel'].unique())}  "
          f"transmitters {', '.join(df['wlan.ta'].unique())}  MU frames {mu:.1%}")
    segs = segments(t, split)
    for name, (_, m) in zip(labels if len(segs) == 2 else ["all"], segs):
        r = rssi[m]
        print(f"  RSSI {name:>10s}: mean {np.nanmean(r):6.1f} dBm  std {np.nanstd(r):4.2f} dB")


def segments(t, split):
    if split is None or split >= t[-1]:
        return [("all", np.ones(len(t), bool))]
    return [("seg1", t < split), ("seg2", t >= split)]


def check_csv(path, split, labels, window):
    t, z, tx = load_csv(path)
    keep, info = integrity(t, z)
    print(f"  rows {info['rows']}  duration {info['duration']:.1f}s  rate {info['rate']:.2f} Hz  "
          f"max gap {info['max_gap']:.2f}s  gaps>{GAP_S}s {info['gaps']}")
    names = {"incomplete": "incomplete rows", "nonmono": "out-of-order", "dup_ts": "duplicate timestamps"}
    problems = [f"{info[k]} {v}" for k, v in names.items() if info[k]]
    if len(tx) > 1:
        problems.append(f"{len(tx)} transmitters {tx}")
    print(f"  transmitter {', '.join(tx) or '?'}  |  issues: {', '.join(problems) or 'none'}")

    t, z = t[keep], z[keep]
    t = t - t[0]
    kp, ks, residual = angle_indices(z)
    if residual > 0.05:
        print(f"  WARNING: ratios don't map to integer codebook indices (residual {residual:.2f}); "
              "has bfmtool/preprocessor.py's decoder changed? Motion numbers below are unreliable.")
    mu = (kp.max(1) >= 2 ** SU_PHI_BITS) | (ks.max(1) >= 2 ** SU_PSI_BITS)
    su = ~mu if (~mu).mean() >= 0.5 else mu
    modulus = 2 ** SU_PHI_BITS if (~mu).mean() >= 0.5 else 2 ** DECODER_PHI_BITS
    psi_levels = 2 ** SU_PSI_BITS if (~mu).mean() >= 0.5 else 2 ** DECODER_PSI_BITS
    print(f"  feedback frames: {(~mu).mean():.1%} SU (6/4-bit), {mu.mean():.1%} MU (9/7-bit); "
          f"stats use the {'SU' if modulus == 64 else 'MU'} frames")
    t, kp, ks = t[su], kp[su], ks[su]

    segs = segments(t, split)
    names = labels if len(segs) == 2 else ["all"]
    stats = []
    print(f"  {'segment':>10s} {'packets':>7s} {'phi_jump':>8s} {'coherence':>9s} {'levels':>6s} {'static':>6s}")
    for name, (_, m) in zip(names, segs):
        s = segment_stats(t[m], kp[m], ks[m], modulus)
        stats.append(s)
        if s:
            print(f"  {name:>10s} {s['n']:7d} {s['phi_jump']:8.2f} {s['coherence']:+9.2f} "
                  f"{s['levels']:6.0f} {s['static']:6.0%}")
    if len(segs) == 2 and all(stats):
        mi = motion_index(kp, ks, modulus, psi_levels)
        auc = rolling_auc(t, mi, split, window)
        ratio = stats[1]["phi_jump"] / max(stats[0]["phi_jump"], 1e-9)
        print(f"  separation: phi_jump {names[1]}/{names[0]} = {ratio:.2f}x  |  "
              f"{window:g}s motion-index AUC ({names[1]} > {names[0]}) = "
              f"{'n/a' if auc is None else f'{auc:.2f}'}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("files", nargs="+")
    p.add_argument("--split", default="60",
                   help="seconds where the second activity starts, or 'none' (default 60)")
    p.add_argument("--labels", default="standing,walking", help="names of the two segments")
    p.add_argument("--window", type=float, default=5.0, help="motion-index window in seconds")
    a = p.parse_args(argv)
    split = None if a.split.lower() == "none" else float(a.split)
    labels = a.labels.split(",")
    if len(labels) != 2:
        p.error("--labels needs exactly two comma-separated names")

    for path in a.files:
        print(f"== {path}")
        try:
            if path.endswith(".pcap"):
                pcap_summary(path, split, labels)
            else:
                check_csv(path, split, labels, a.window)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
