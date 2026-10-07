"""Train the real-time standing/walking detector bundle loaded by main_gui3.py.

Usage:
    python train_rt_bfm.py [--window 5] [--hop 0.5] [--out bfm_rt_model_5s_xgb.joblib]

Reads the session CSVs in bfm_real_imag_csv/ written by main_gui.py / main_gui3.py:
  * single-activity sessions: filename contains "_Standing_" or "_Walking_"
  * mixed sessions: "Standing + Walking" or "S + W" -> standing before --split s, walking after

Each window goes through the same transform as main_gui3._bfm_rt_window
(nearest-packet 10 Hz grid, log |ratio|, wrapped phase) and rtbfm.window_features,
so the live detector sees the inputs it was trained on.
Reports leave-one-person-out accuracy, then fits on everything and writes the bundle.
"""
import argparse
import glob
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import rtbfm
from bfmtool.utils import append_mag_phase

FS = 10.0
MAX_GAP = 0.15        # same rejection rule as main_gui3.BFM_RT_MAX_GAP
SPLIT_GUARD_S = 2.0   # drop mixed-session windows within this of the activity switch
CLASSES = ["standing", "walking"]

DEFAULT_GLOBS = [
    "bfm_real_imag_csv/bfm_ri_data_*_20261006_*.csv",
    "bfm_real_imag_csv/bfm_ri_data_*_20261007_*.csv",
]


def parse_name(path):
    """(subject, kind) where kind is 'standing', 'walking' or 'mixed'; None if unlabelled."""
    name = os.path.basename(path)[len("bfm_ri_data_"):]
    subject = name.split("_")[0]
    if "Standing + Walking" in name or "S + W" in name:
        return subject, "mixed"
    for kind in CLASSES:
        if f"_{kind.capitalize()}_" in name:
            return subject, kind
    return None


def load_session(path):
    """Sorted, de-duplicated, complete packets on a uniform FS grid: (mag, phase, gap, t_grid)."""
    df = append_mag_phase(pd.read_csv(path))
    mag_cols = sorted((c for c in df.columns if c.endswith("_Ratio_Mag")), key=lambda c: int(c.split("_")[1]))
    phase_cols = [c.replace("_Mag", "_Phase") for c in mag_cols]
    df = df.dropna(subset=mag_cols + phase_cols + ["timestamp"]).sort_values("timestamp", kind="stable")
    t = df["timestamp"].to_numpy(float)
    keep = np.r_[True, np.diff(t) > 0]
    df, t = df[keep], t[keep] - t[keep][0]

    grid = np.arange(0.0, t[-1] + 1e-9, 1.0 / FS)
    j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    pick = np.where(np.abs(t[j - 1] - grid) <= np.abs(t[j] - grid), j - 1, j)
    gap = np.abs(t[pick] - grid) > (2.0 / FS)

    # identical to main_gui3._bfm_rt_window
    mag = np.log(np.clip(df[mag_cols].to_numpy(np.float32)[pick], 1e-6, None))
    phase = np.angle(np.exp(1j * df[phase_cols].to_numpy(np.float64)[pick])).astype(np.float32)
    return mag, phase, gap, grid


def session_windows(path, kind, window_s, hop_s, split_s):
    mag, phase, gap, grid = load_session(path)
    T, hop = int(round(window_s * FS)), max(1, int(round(hop_s * FS)))
    X, y = [], []
    for a in range(0, len(grid) - T + 1, hop):
        b = a + T
        if gap[a:b].mean() > MAX_GAP:
            continue
        if kind == "mixed":
            start, end = grid[a], grid[b - 1]
            if end > split_s - SPLIT_GUARD_S and start < split_s + SPLIT_GUARD_S:
                continue
            label = int(start >= split_s)
        else:
            label = CLASSES.index(kind)
        X.append(rtbfm.window_features(mag[a:b], phase[a:b], FS))
        y.append(label)
    return np.asarray(X), np.asarray(y)


def make_model():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, eval_metric="logloss", verbosity=0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--window", type=float, default=5.0, help="window length in seconds")
    p.add_argument("--hop", type=float, default=0.5, help="hop between training windows / live predictions")
    p.add_argument("--split", type=float, default=60.0, help="activity switch time in mixed sessions")
    p.add_argument("--glob", action="append", help="session CSV glob (repeatable)")
    p.add_argument("--out", default="bfm_rt_model_5s_xgb.joblib")
    a = p.parse_args()

    paths = sorted({f for g in (a.glob or DEFAULT_GLOBS) for f in glob.glob(g)})
    sessions = []
    for path in paths:
        parsed = parse_name(path)
        if parsed is None:
            print(f"  skip (no activity in name): {os.path.basename(path)}")
            continue
        X, y = session_windows(path, parsed[1], a.window, a.hop, a.split)
        if len(X):
            sessions.append(dict(path=path, subject=parsed[0], kind=parsed[1], X=X, y=y))
    subjects = sorted({s["subject"] for s in sessions})
    n_win = sum(len(s["y"]) for s in sessions)
    print(f"{len(sessions)} sessions, {n_win} windows of {a.window:g}s, subjects {subjects}")

    # leave-one-person-out: the honest estimate for a new user
    P, Y = [], []
    for subj in subjects:
        tr = [s for s in sessions if s["subject"] != subj]
        te = [s for s in sessions if s["subject"] == subj]
        m = make_model().fit(np.vstack([s["X"] for s in tr]), np.hstack([s["y"] for s in tr]))
        p_ = m.predict_proba(np.vstack([s["X"] for s in te]))[:, 1]
        y_ = np.hstack([s["y"] for s in te])
        print(f"  held-out {subj:8s}: standing correct {(p_[y_ == 0] < .5).mean():4.0%}  "
              f"walking correct {(p_[y_ == 1] >= .5).mean():4.0%}  AUC {roc_auc_score(y_, p_):.2f}")
        P.append(p_)
        Y.append(y_)
    P, Y = np.hstack(P), np.hstack(Y)
    bacc = ((P[Y == 0] < .5).mean() + (P[Y == 1] >= .5).mean()) / 2
    lopo = dict(auc=float(roc_auc_score(Y, P)), balanced_accuracy=float(bacc))
    print(f"  leave-one-person-out overall: AUC {lopo['auc']:.2f}  balanced accuracy {lopo['balanced_accuracy']:.0%}")

    dates = sorted({m.group(1) for s in sessions if (m := re.search(r"_(\d{8})_\d{6}\.csv$", s["path"]))})
    model = make_model().fit(np.vstack([s["X"] for s in sessions]), np.hstack([s["y"] for s in sessions]))
    bundle = dict(
        model=model, window_s=a.window, hop_s=a.hop, fs=FS, frequency=FS, classes=CLASSES,
        environments=["Pi beside router, sessions " + ", ".join(dates)],
        feature_names=rtbfm.feature_names(),
        sessions=[os.path.basename(s["path"]) for s in sessions], subjects=subjects,
        leave_one_person_out=lopo,
    )
    joblib.dump(bundle, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
