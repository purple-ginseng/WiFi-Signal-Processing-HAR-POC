"""
Plot the BFM beamforming-feedback angle vs subcarrier index in the classic
"CSI amplitude across subcarriers" paper format.

The raw BFM CSVs store, per subcarrier, a quantized angle pair
(phi11, psi21). There is no direct "CSI amplitude", so each selected CSV
file is treated as one "Channel" and we plot, per subcarrier index, the
mean psi21 angle (in degrees) across all packets in that file.

psi21 is 4-bit quantized over [0, 90) degrees, so - unlike the 6-bit phi11
which wraps at 0/360 - it gives smooth, bounded, amplitude-like curves that
match the reference figure. The 234 (80 MHz) subcarriers are binned down to
N_STEPS points along the x-axis (default 60) to mirror the reference layout.

Usage:
    python plot_bfm_subcarrier_channels.py [csv_folder] [n_channels] [n_steps]
"""

import sys
import glob
import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 802.11 quantization levels: phi is 6-bit over [0,2*pi), psi is 4-bit over [0,pi/2).
PHI_LEVELS = 64   # 0..63  -> [0, 360) degrees (wraps)
PSI_LEVELS = 16   # 0..15  -> [0, 90)  degrees (no wrap)
N_STEPS = 60      # number of subcarrier bins along the x-axis


def subcarrier_index_from_col(col):
    """Extract the signed subcarrier index from a 'SCIDX_<n>_psi21' column."""
    m = re.match(r"SCIDX_(-?\d+)_psi21", col)
    return int(m.group(1)) if m else None


def bin_to_steps(curve, n_steps):
    """Average a per-subcarrier curve down to n_steps bins along the x-axis."""
    edges = np.linspace(0, len(curve), n_steps + 1).astype(int)
    return np.array([np.nanmean(curve[edges[i]:edges[i + 1]])
                     for i in range(n_steps)])


def channel_curve(csv_path, n_steps=N_STEPS):
    """Return the mean psi21 angle (deg) binned to n_steps for one CSV file."""
    df = pd.read_csv(csv_path)
    psi_cols = [c for c in df.columns if c.startswith("SCIDX_") and c.endswith("_psi21")]
    # keep natural (signed) subcarrier ordering
    psi_cols.sort(key=subcarrier_index_from_col)

    vals = df[psi_cols].apply(pd.to_numeric, errors="coerce")
    # psi bit-depth varies between capture sessions (4-bit 0..15 vs 7-bit 0..127),
    # so infer the level count per file: next power of two above the observed max.
    vmax = np.nanmax(vals.values)
    levels = 1 << int(np.ceil(np.log2(vmax + 1))) if vmax > 0 else PSI_LEVELS
    # per-subcarrier mean across packets, mapped from quantized level to degrees
    per_sc = vals.mean(axis=0).values / levels * 90.0
    return bin_to_steps(per_sc, n_steps)


def pretty_label(path, i):
    """Short human label from the filename, falling back to 'Channel i'."""
    base = os.path.basename(path)
    m = re.match(r"bfm_data_([a-zA-Z]+)_([a-zA-Z]+)", base)
    if m:
        return f"Ch{i}: {m.group(1)}/{m.group(2)}"
    return f"Channel {i}"


def main():
    csv_folder = sys.argv[1] if len(sys.argv) > 1 else "bfm_raw_csv"
    n_channels = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else N_STEPS

    files = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))
    if not files:
        raise SystemExit(f"No CSVs found in {csv_folder}")

    # psi bit-depth varies between capture sessions; mixing 4-bit and 7-bit files
    # puts channels in different angle bands. Group files by inferred level count
    # (from a cheap partial read) and keep only the largest homogeneous group so
    # all channels share a comparable scale, like the reference figure.
    def infer_levels(path):
        head = pd.read_csv(path, nrows=200)
        psi = [c for c in head.columns if c.endswith("_psi21")]
        vmax = np.nanmax(head[psi].apply(pd.to_numeric, errors="coerce").values)
        return 1 << int(np.ceil(np.log2(vmax + 1))) if vmax > 0 else PSI_LEVELS

    groups = {}
    for f in files:
        groups.setdefault(infer_levels(f), []).append(f)
    files = max(groups.values(), key=len)  # largest homogeneous group

    # Evenly sample n_channels files across that group for variety
    if len(files) > n_channels:
        pick = np.linspace(0, len(files) - 1, n_channels).round().astype(int)
        files = [files[i] for i in pick]

    # Reference-style line aesthetics: distinct color + linestyle + marker
    styles = [
        dict(color="black",   ls="-",  marker=None),
        dict(color="red",     ls="--", marker=None),
        dict(color="blue",    ls="-",  marker=None),
        dict(color="#1f77b4", ls="-",  marker=None),
        dict(color="#d62728", ls="-",  marker="x", markevery=3, ms=4),
        dict(color="orange",  ls="-",  marker=None),
        dict(color="purple",  ls="-",  marker="s", markevery=3, ms=3, mfc="none"),
        dict(color="green",   ls="-",  marker="v", markevery=3, ms=4, mfc="none"),
        dict(color="cyan",    ls="-",  marker="o", markevery=3, ms=4, mfc="none"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for i, f in enumerate(files, start=1):
        curve = channel_curve(f, n_steps)
        x = np.arange(1, len(curve) + 1)  # subcarrier index 1..n_steps
        st = styles[(i - 1) % len(styles)]
        ax.plot(x, curve, linewidth=1.3, label=f"Channel {i}", **st)

    ax.set_xlabel("Subcarrier index", fontsize=12)
    ax.set_ylabel("BFM angle $\\psi_{21}$ (deg)", fontsize=12)
    ax.set_xlim(0, n_steps)
    ax.grid(False)
    ax.legend(loc="center right", fontsize=9, ncol=1, framealpha=1.0)
    ax.tick_params(labelsize=10)
    fig.tight_layout()

    out_png = "bfm_subcarrier_channels.png"
    out_pdf = "bfm_subcarrier_channels.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved {out_png} and {out_pdf}  ({len(files)} channels)")


if __name__ == "__main__":
    main()
