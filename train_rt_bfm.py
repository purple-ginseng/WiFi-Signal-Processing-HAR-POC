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

Spot-swap pilot (standing or lying vs sitting, no bundle written):
    python train_rt_bfm.py --pilot [--pilot-posture standing|lying] [--window 5] [--clf xgb|lr]
                           [--pca 8] [--pass-mark 0.8]

  * sessions: activity "Standing"/"Lying"/"Sitting", description containing the spot, e.g. "spotA R1"
  * usual = the pilot posture at A, sitting at B (training); the other pairs are swapped tests
  * reports, for feature sets A (rtbfm), B (static levels) and A+B:
      1. leave-one-session-out on usual sessions (sanity check)
      2. train on usual, predict swapped, scored per session (correct vs flipped)
      3. no-model check: session static-profile differences, spot vs posture
"""
import argparse
import glob
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import rtbfm
from bfmtool.utils import append_mag_phase

FS = 10.0
MAX_GAP = 0.15        # same rejection rule as main_gui3.BFM_RT_MAX_GAP
SPLIT_GUARD_S = 2.0   # drop mixed-session windows within this of the activity switch
CLASSES = ["standing", "walking"]
POSTURES = ["standing", "walking", "sitting", "lying"]

DEFAULT_GLOBS = [
    "bfm_real_imag_csv/bfm_ri_data_*_20261006_*.csv",
    "bfm_real_imag_csv/bfm_ri_data_*_20261007_*.csv",
]

# Spot-swap pilot: training layout; any other spot-posture pair is a swapped test session
PILOT_POSTURES = ["standing", "lying"]
PILOT_GLOB = "bfm_real_imag_csv/bfm_ri_data_*.csv"
FEATURE_SETS = ["A", "B", "A+B"]


def parse_name(path):
    """(subject, kind): kind is a POSTURES entry or 'mixed'; None if unlabelled."""
    name = os.path.basename(path)[len("bfm_ri_data_"):]
    subject = name.split("_")[0]
    if "Standing + Walking" in name or "S + W" in name:
        return subject, "mixed"
    for kind in POSTURES:
        if f"_{kind.capitalize()}_" in name:
            return subject, kind
    return None


def parse_spot(path):
    """Spot from the description, e.g. 'spotA R1' or 'spot B' -> 'A'/'B'; None if absent."""
    m = re.search(r"spot\s*([a-z0-9])(?![a-z0-9])", os.path.basename(path), re.IGNORECASE)
    return m.group(1).upper() if m else None


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


def static_window_features(mag, phase):
    """Static levels rtbfm discards: per-subcarrier mean log|ratio|, then circular-mean phase."""
    circ_mean = np.angle(np.exp(1j * phase.astype(np.float64)).mean(0))
    return np.r_[mag.mean(0), circ_mean].astype(np.float32)


def pilot_window_features(mag, phase, fs):
    """Feature set A (rtbfm.window_features) followed by feature set B (static levels)."""
    return np.r_[rtbfm.window_features(mag, phase, fs), static_window_features(mag, phase)]


def session_windows(path, kind, window_s, hop_s, split_s, classes=CLASSES,
                    feature_fn=rtbfm.window_features):
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
            label = classes.index(kind)
        X.append(feature_fn(mag[a:b], phase[a:b], FS))
        y.append(label)
    return np.asarray(X), np.asarray(y)


def make_model():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, eval_metric="logloss", verbosity=0)


def make_pilot_model(feature_set, n_static, clf="xgb", n_pca=8):
    """Columns are [rtbfm | static]; PCA on static levels is fitted on training windows only."""
    n_rt = len(rtbfm.feature_names())
    parts = []
    if "A" in feature_set:
        parts.append(("rt", StandardScaler(), list(range(n_rt))))
    if "B" in feature_set:
        parts.append(("static", make_pipeline(StandardScaler(), PCA(n_pca)),
                      list(range(n_rt, n_rt + n_static))))
    head = make_model() if clf == "xgb" else LogisticRegression(max_iter=5000)
    return make_pipeline(ColumnTransformer(parts), head)


def _fit_pilot(train, feature_set, n_static, a):
    """Fit on the training sessions; None when they don't cover both postures."""
    y = np.hstack([s["y"] for s in train])
    if len(set(y)) < 2:
        return None
    X = np.vstack([s["X"] for s in train])
    return make_pilot_model(feature_set, n_static, a.clf, a.pca).fit(X, y)


def _frac_correct(model, s):
    """Fraction of a session's windows predicted as its own posture."""
    if model is None:
        return np.nan
    return float(((model.predict_proba(s["X"])[:, 1] >= 0.5) == s["y"]).mean())


def _reading(fracs, pass_mark):
    """Session-level verdict against the pre-agreed pass mark (majority of windows per session)."""
    fracs = np.asarray(fracs, float)
    fracs = fracs[~np.isnan(fracs)]
    if not len(fracs):
        return "n/a"
    if (fracs > 0.5).mean() >= pass_mark:
        return "mostly correct"
    if (fracs < 0.5).mean() >= pass_mark:
        return "mostly flipped"
    return "in between/50-50"


def _print_pilot_table(title, sessions, fracs):
    print(f"\n{title}")
    print(f"  {'session':34s}" + "".join(f"{fs:>8s}" for fs in FEATURE_SETS))
    for i, s in enumerate(sessions):
        print(f"  {s['label']:34s}" + "".join(f"{fracs[fs][i]:8.2f}" for fs in FEATURE_SETS))
    correct = {fs: np.nansum(np.asarray(fracs[fs]) > 0.5) for fs in FEATURE_SETS}
    print(f"  {'sessions correct (majority)':34s}"
          + "".join(f"{f'{correct[fs]}/{len(sessions)}':>8s}" for fs in FEATURE_SETS))


def _pilot_no_model_check(sessions, n_rt, n_static):
    """Median RMS difference of session static profiles, same-subject pairs, by what differs."""
    n_sc = n_static // 2
    prof = []
    for s in sessions:
        static = s["X"][:, n_rt:n_rt + n_static].astype(np.float64)
        phase = np.angle(np.exp(1j * static[:, n_sc:]).mean(0))
        prof.append((static[:, :n_sc].mean(0), phase))
    groups = {"replicate (same spot+posture)": [], "posture only (same spot)": [],
              "spot only (same posture)": [], "both differ": []}
    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            a, b = sessions[i], sessions[j]
            if a["subject"] != b["subject"]:
                continue
            same_spot, same_kind = a["spot"] == b["spot"], a["kind"] == b["kind"]
            key = ("replicate (same spot+posture)" if same_spot and same_kind else
                   "posture only (same spot)" if same_spot else
                   "spot only (same posture)" if same_kind else "both differ")
            d_mag = np.sqrt(np.mean((prof[i][0] - prof[j][0]) ** 2))
            d_ph = np.sqrt(np.mean(np.angle(np.exp(1j * (prof[i][1] - prof[j][1]))) ** 2))
            groups[key].append((d_mag, d_ph))
    print("\n3. No-model check: RMS difference of session static profiles (same-subject pairs)")
    rep = np.median([d[0] for d in groups["replicate (same spot+posture)"]]) \
        if groups["replicate (same spot+posture)"] else np.nan
    for key, d in groups.items():
        if not d:
            print(f"  {key:32s} no pairs yet")
            continue
        mag = np.median([x[0] for x in d])
        print(f"  {key:32s} log-mag {mag:.3f} ({mag / rep:4.1f}x replicate)  "
              f"phase {np.median([x[1] for x in d]):.4f} rad  n={len(d)}")
    print("  Location shortcut risk: 'spot only' clearly above 'posture only'.")


def pilot_layout(posture):
    """(classes, usual spot per class): the pilot posture is class 0 at A, sitting class 1 at B."""
    return [posture, "sitting"], {posture: "A", "sitting": "B"}


def run_pilot(a):
    """Spot-swap pilot: train on usual sessions, predict swapped ones; never writes a bundle."""
    pilot_classes, usual_spot = pilot_layout(a.pilot_posture)
    sessions = []
    for path in sorted({f for g in (a.glob or [PILOT_GLOB]) for f in glob.glob(g)}):
        parsed, spot = parse_name(path), parse_spot(path)
        if parsed is None or parsed[1] not in pilot_classes or spot is None:
            continue
        subject, kind = parsed
        X, y = session_windows(path, kind, a.window, a.hop, a.split, pilot_classes, pilot_window_features)
        if not len(X):
            print(f"  skip (no complete windows): {os.path.basename(path)}")
            continue
        stamp = re.search(r"_(\d{8})_(\d{6})\.csv$", path)
        sessions.append(dict(
            path=path, subject=subject, kind=kind, spot=spot, X=X, y=y,
            usual=usual_spot[kind] == spot,
            label=f"{subject[:12]} {kind} @{spot} {stamp.group(2) if stamp else ''}",
        ))
    usual = [s for s in sessions if s["usual"]]
    swapped = [s for s in sessions if not s["usual"]]
    print(f"Spot-swap pilot: {len(usual)} usual + {len(swapped)} swapped sessions, "
          f"{a.window:g}s windows, hop {a.hop:g}s, clf {a.clf}, PCA {a.pca}, pass mark {a.pass_mark:.0%}")
    print("  usual layout: " + ", ".join(f"{k} @{v}" for k, v in usual_spot.items())
          + f"; subjects {sorted({s['subject'] for s in sessions})}")
    if not sessions:
        print(f"No pilot sessions found (need {'/'.join(c.capitalize() for c in pilot_classes)} "
              "with 'spotX' in the description).")
        return

    n_rt = len(rtbfm.feature_names())
    n_static = sessions[0]["X"].shape[1] - n_rt

    # 1) a feature set that fails here makes its swap result meaningless
    sanity = {fs: [_frac_correct(_fit_pilot([o for o in usual if o is not s], fs, n_static, a), s)
                   for s in usual] for fs in FEATURE_SETS}
    _print_pilot_table("1. Sanity check: leave-one-session-out on usual sessions "
                       "(fraction of windows correct)", usual, sanity)

    # 2) swap test, scored per session because 0.5 s hops overlap windows heavily
    if swapped:
        swap = {fs: [_frac_correct(model, s) for s in swapped]
                for fs in FEATURE_SETS for model in [_fit_pilot(usual, fs, n_static, a)]}
        _print_pilot_table("2. Swap test: train on usual, predict swapped "
                           "(fraction of windows correct; < 0.5 = flipped)", swapped, swap)
        verdicts = []
        for fs in FEATURE_SETS:
            v = _reading(swap[fs], a.pass_mark)
            if _reading(sanity[fs], a.pass_mark) != "mostly correct":
                v += " (sanity failed)"
            verdicts.append(f"{fs}: {v}")
        print("  reading -> " + "; ".join(verdicts))
    else:
        print("\n2. Swap test: no swapped sessions yet")

    _pilot_no_model_check(sessions, n_rt, n_static)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--window", type=float, default=5.0, help="window length in seconds")
    p.add_argument("--hop", type=float, default=0.5, help="hop between training windows / live predictions")
    p.add_argument("--split", type=float, default=60.0, help="activity switch time in mixed sessions")
    p.add_argument("--glob", action="append", help="session CSV glob (repeatable)")
    p.add_argument("--out", default="bfm_rt_model_5s_xgb.joblib")
    p.add_argument("--pilot", action="store_true", help="run the posture-vs-sitting spot-swap pilot")
    p.add_argument("--pilot-posture", choices=PILOT_POSTURES, default="standing",
                   help="pilot: posture paired with sitting (usual spot A)")
    p.add_argument("--clf", choices=["xgb", "lr"], default="xgb", help="pilot classifier")
    p.add_argument("--pca", type=int, default=8, help="pilot: PCA components for static levels")
    p.add_argument("--pass-mark", type=float, default=0.8, help="pilot: share of sessions for a verdict")
    a = p.parse_args()

    if a.pilot:
        run_pilot(a)
        return

    paths = sorted({f for g in (a.glob or DEFAULT_GLOBS) for f in glob.glob(g)})
    sessions = []
    for path in paths:
        parsed = parse_name(path)
        if parsed is None or parsed[1] not in CLASSES + ["mixed"]:
            print(f"  skip (no standing/walking in name): {os.path.basename(path)}")
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
