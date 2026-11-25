"""Offline BFM analysis pipeline: QC -> 10 Hz grid -> per-day subcarrier mask -> rtbfm features -> XGB evaluation.

Usage:
    python bfm_pipeline.py refs                                  # reference rooms + placement check per day
    python bfm_pipeline.py features --out feats.npz              # build feature cache (masks none, lvl25)
    python bfm_pipeline.py evaluate feats.npz                    # leave-one-subject-out / leave-one-day-out
    python bfm_pipeline.py evaluate feats.npz --pooled           # also leave-one-session-out (slow)

Session files: bfm_real_imag_csv/bfm_ri_data_<Subject>_<Activity>_<Desc>_<YYYYMMDD>_<HHMMSS>.csv
Official subjects are S01, S02, ...; empty rooms are Nobody_Empty_*; "Valid" sessions are skipped.
Only days from --since (default 20261104, first official ch149 day) are used.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

import rtbfm
from bfmtool.utils import bfm_row_widths_mhz

DATA_DIR = "bfm_real_imag_csv"
OFFICIAL_START = "20261104"   # Oct 23 S01/S02 were ch36 pilots, not the official set
FS = 10.0            # resampling grid, Hz
WIN_S = 4.0          # window length, seconds
HOP_S = 0.5          # window hop, seconds
MAX_GAP = 0.15       # skip windows with more gap slots than this share
GAP_TOL = 2 / FS     # grid slot is a gap if no report within 0.2 s
JITTER_K = 5         # moving-average length (0.5 s) for the jitter high-pass
CLASSES = ["Walking", "Waving", "Bending", "Standing", "Sitting"]
NAME_RE = re.compile(r"^bfm_ri_data_(?P<subject>[^_]+)_(?P<activity>[^_]+)_(?P<desc>.+)_(?P<date>\d{8})_(?P<time>\d{6})\.csv$")


# ---------------------------------------------------------------- step 1: find sessions

def parse_name(path):
    """Split a session filename into subject/activity/desc/date/time; None if it doesn't match."""
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        return None
    d = m.groupdict()
    d["path"], d["name"] = path, os.path.basename(path)[len("bfm_ri_data_"):-4]
    d["official"] = bool(re.fullmatch(r"S\d\d", d["subject"])) and "Valid" not in d["desc"]
    d["empty"] = d["activity"] == "Empty"
    return d


def list_sessions(root=".", since=OFFICIAL_START):
    """All parseable sessions on or after `since`, sorted by date and time."""
    rows = [parse_name(p) for p in glob.glob(os.path.join(root, DATA_DIR, "bfm_ri_data_*.csv"))]
    return sorted((r for r in rows if r and r["date"] >= since), key=lambda r: (r["date"], r["time"]))


def choose_reference_rooms(sessions):
    """Per day: the last empty room before that day's first official session (the final placement)."""
    refs = {}
    for date in sorted({s["date"] for s in sessions}):
        day = [s for s in sessions if s["date"] == date]
        first = next((s["time"] for s in day if s["official"]), None)
        if first is None:
            continue
        before = [s for s in day if s["empty"] and s["time"] < first]
        if before:
            refs[date] = before[-1]
    return refs


# ---------------------------------------------------------------- steps 2-3: load, QC, 10 Hz grid

def resample_nearest(t, z, fs=FS, gap_tol=GAP_TOL):
    """Nearest-report resampling onto a uniform grid; gap flags slots with no report within gap_tol."""
    order = np.argsort(t, kind="stable")
    t, z = t[order], z[order]
    keep = np.r_[True, np.diff(t) > 0]          # drop duplicate timestamps
    t, z = t[keep] - t[keep][0], z[keep]
    grid = np.arange(0, t[-1] + 1e-9, 1 / fs)
    j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    pick = np.where(np.abs(t[j - 1] - grid) <= np.abs(t[j] - grid), j - 1, j)
    gap = np.abs(t[pick] - grid) > gap_tol
    return z[pick], gap


def load_session(path):
    """80 MHz rows only -> complex ratio r on a 10 Hz grid. Returns |r|, angle(r), gap flags, subcarrier ids."""
    df = pd.read_csv(path)
    df = df[bfm_row_widths_mhz(df) == 80]       # drop 20/40 MHz blips; widths can't be pooled
    real = sorted((c for c in df.columns if c.endswith("_Ratio_Real") and df[c].notna().all()),
                  key=lambda c: int(c.split("_")[1]))
    sc = np.array([int(c.split("_")[1]) for c in real])
    z = df[real].to_numpy(float) + 1j * df[[c.replace("_Real", "_Imag") for c in real]].to_numpy(float)
    t = pd.to_numeric(df["timestamp"]).to_numpy(float)
    z, gap = resample_nearest(t, z)
    return np.abs(z), np.angle(z).astype(np.float32), gap, sc


# ---------------------------------------------------------------- step 4: per-day subcarrier mask

def static_level(mag):
    """Static level per subcarrier: log of time-mean |r|. Magnitude only; one value per subcarrier."""
    return np.log(mag.mean(0))


def empty_jitter(mag, k=JITTER_K):
    """Per-subcarrier jitter: std of relative magnitude after removing a 0.5 s moving average."""
    rel = mag / mag.mean(0)
    smooth = np.apply_along_axis(lambda x: np.convolve(x, np.ones(k) / k, mode="same"), 0, rel)
    return (rel - smooth)[k:-k].std(0)


def build_masks(mag):
    """Keep-masks (True = keep) from a reference empty room. lvl25 is the standard one."""
    lvl, jit = static_level(mag), empty_jitter(mag)
    return {
        "none": np.ones(mag.shape[1], bool),
        "lvl25": lvl <= np.percentile(lvl, 75),      # drop largest-|r| quarter (ill-conditioned ratio)
        "noise25": jit <= np.percentile(jit, 75),    # drop noisiest quarter (comparison only)
    }


def placement_report(mag):
    """Setup check numbers: level spread p10-p90 in dB and jitter by static-level quartile (x1000)."""
    lvl, jit = static_level(mag), empty_jitter(mag)
    q = np.digitize(lvl, np.percentile(lvl, [25, 50, 75]))
    return dict(spread_db=20 / np.log(10) * (np.percentile(lvl, 90) - np.percentile(lvl, 10)),
                jitter=1000 * jit.mean(),
                **{f"jit_q{i + 1}": 1000 * jit[q == i].mean() for i in range(4)})


# ---------------------------------------------------------------- steps 5-6: windows + rtbfm features

def session_features(path, masks, sc_ref):
    """rtbfm features for every 4 s window that passes the gap check, once per mask."""
    mag, phase, gap, sc = load_session(path)
    if not np.array_equal(sc, sc_ref):
        raise ValueError(f"{path}: subcarrier set differs from the day's reference room")
    logmag = np.log(np.clip(mag, 1e-6, None)).astype(np.float32)
    w, hop = int(round(WIN_S * FS)), int(round(HOP_S * FS))
    starts = [a for a in range(0, len(logmag) - w + 1, hop) if gap[a:a + w].mean() <= MAX_GAP]
    return {name: np.array([rtbfm.window_features(logmag[a:a + w][:, keep], phase[a:a + w][:, keep], FS)
                            for a in starts], np.float32).reshape(len(starts), -1)
            for name, keep in masks.items()}


def cmd_refs(args):
    sessions = list_sessions(args.root, args.since)
    rows = []
    for date, ref in choose_reference_rooms(sessions).items():
        mag, _, _, sc = load_session(ref["path"])
        rep = placement_report(mag)
        rows.append(dict(date=date, ref=ref["name"][-40:], n_sc=len(sc),
                         dropped_lvl25=int((~build_masks(mag)["lvl25"]).sum()), **rep))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    print("\nTargets: spread_db <~ 10, jit_q4 <~ 30 (both from the 10-11 Nov diagnosis).")


def cmd_features(args):
    from joblib import Parallel, delayed
    sessions = list_sessions(args.root, args.since)
    refs = choose_reference_rooms(sessions)
    mask_names = args.masks.split(",")
    day_masks = {}
    for date, ref in refs.items():
        mag, _, _, sc = load_session(ref["path"])
        day_masks[date] = ({k: v for k, v in build_masks(mag).items() if k in mask_names}, sc)
        print(f"{date}: reference {ref['name']}", flush=True)
    # official sessions, plus empty rooms from the reference onward, on days that have a reference
    todo = [s for s in sessions if s["date"] in refs
            and (s["official"] or (s["empty"] and s["time"] >= refs[s["date"]]["time"]))]
    print(f"{len(todo)} sessions", flush=True)
    res = Parallel(n_jobs=args.jobs)(delayed(session_features)(s["path"], *day_masks[s["date"]]) for s in todo)
    meta = pd.DataFrame([{k: s[k] for k in ("name", "subject", "activity", "desc", "date", "time")}
                         | {"n_windows": len(r["none" if "none" in r else mask_names[0]])}
                         for s, r in zip(todo, res)])
    arr = {f"X_{m}": np.concatenate([r[m] for r in res]) for m in mask_names}
    arr["sess"] = np.concatenate([[i] * len(r[mask_names[0]]) for i, r in enumerate(res)])
    np.savez_compressed(args.out, **arr)
    meta.to_pickle(args.out.replace(".npz", "_meta.pkl"))
    print(meta[meta.activity != "Empty"].groupby(["subject", "activity"]).size().unstack(fill_value=0).to_string())
    print(f"saved {args.out}")


# ---------------------------------------------------------------- steps 7-9: model, vote, evaluation

def make_model(jobs):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=250, learning_rate=0.08, max_depth=4, reg_lambda=1.0,
                         random_state=0, eval_metric="mlogloss", n_jobs=jobs)


def session_votes(P, sess, y):
    """Session label = argmax of mean window probability. Returns (correct, total)."""
    ids = np.unique(sess)
    return sum(P[sess == k].mean(0).argmax() == y[sess == k][0] for k in ids), len(ids)


def hold_out(X, y, groups, jobs):
    """Train on every group but one, predict that group; repeat for all groups."""
    P = np.zeros((len(y), len(CLASSES)))
    for g in np.unique(groups):
        te = groups == g
        P[te] = make_model(jobs).fit(X[~te], y[~te]).predict_proba(X[te])
    return P


def cmd_evaluate(args):
    d = np.load(args.cache)
    meta = pd.read_pickle(args.cache.replace(".npz", "_meta.pkl"))
    sess = d["sess"]
    act, subj, date = (meta[c].to_numpy() for c in ("activity", "subject", "date"))
    on = (act != "Empty")[sess]
    sg = sess[on]
    y = np.array([CLASSES.index(a) for a in act[sg]])
    splits = [("new person", subj[sg]), ("new day", date[sg])] + ([("pooled LOSO", sg)] if args.pooled else [])
    masks = [k[2:] for k in d.files if k.startswith("X_")]
    rows, empties = [], []
    for m in masks:
        X = d[f"X_{m}"][on]
        for split, groups in splits:
            P = hold_out(X, y, groups, args.jobs)
            ok, n = session_votes(P, sg, y)
            row = dict(mask=m, split=split, window_acc=100 * (P.argmax(1) == y).mean(), sessions=f"{ok}/{n}")
            row |= {s: 100 * (P[subj[sg] == s].argmax(1) == y[subj[sg] == s]).mean() for s in np.unique(subj[sg])}
            rows.append(row)
            print(f"{m:8s} {split:12s} {row['window_acc']:.1f}% {row['sessions']}", flush=True)
        # each empty room through a model that never saw its day
        day_models = {}
        for k in np.where(act == "Empty")[0]:
            if date[k] not in day_models:
                tr = date[sg] != date[k]
                day_models[date[k]] = make_model(args.jobs).fit(X[tr], y[tr])
            pr = day_models[date[k]].predict(d[f"X_{m}"][sess == k])
            empties.append(dict(mask=m, room=f"{date[k]} {meta.desc[k][:25]}",
                                **{c[:5]: round(100 * (pr == i).mean()) for i, c in enumerate(CLASSES)}))
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\nWindow accuracy (%) per held-out subject column; sessions = session-vote correct/total")
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    print("\nEmpty rooms, % of windows per predicted class (expect mostly Sitting):")
    print(pd.DataFrame(empties).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repo root holding bfm_real_imag_csv/")
    ap.add_argument("--since", default=OFFICIAL_START, help="first date (YYYYMMDD) to include")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("refs", help="show each day's reference empty room and placement check")
    f = sub.add_parser("features", help="build the feature cache")
    f.add_argument("--out", default="bfm_features.npz")
    f.add_argument("--masks", default="none,lvl25", help="comma list of none,lvl25,noise25")
    f.add_argument("--jobs", type=int, default=4, help="parallel loaders; keep low, memory is tight")
    e = sub.add_parser("evaluate", help="evaluate a feature cache")
    e.add_argument("cache")
    e.add_argument("--pooled", action="store_true", help="also leave-one-session-out (one fit per session)")
    e.add_argument("--jobs", type=int, default=6, help="XGB threads; 4-6 if a GUI is running")
    args = ap.parse_args(argv)
    {"refs": cmd_refs, "features": cmd_features, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
