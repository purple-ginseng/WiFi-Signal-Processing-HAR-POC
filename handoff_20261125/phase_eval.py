import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from bfm_pipeline import make_model, CLASSES
import os; S = sys.argv[1]; JOBS = int(os.environ.get("JOBS", 10))
F, P = np.load(f"{S}/rate_feats.npz"), np.load(f"{S}/phase_feats.npz")
sess = F["sess_nearest"]; subj, date, act = F["subject"][sess], F["date"][sess], F["activity"][sess]
y = np.array([CLASSES.index(a) for a in act]); M = F["X_nearest"]
# day shift: standing windows, median of the main change feature per day, relative to all days
st = act == "Standing"
ds = pd.DataFrame({"mag absdiff_mean": M[:, 5], "old phdiff_mean": P["old10"][:, 0], "fixed phdiff_mean": P["fix10"][:, 0],
                   "reliable phdiff_mean": P["rel10"][:, 0], "date": date})[st]
med = ds.groupby("date").median(); print("Standing windows: per-day median / all-day median\n", (med / ds.drop(columns="date").median()).round(2).to_string(), "\n")
arms = {"magnitude 64": M, "+ old phase 10": np.c_[M, P["old10"]], "+ fixed phase 10": np.c_[M, P["fix10"]],
        "+ reliable phase 10": np.c_[M, P["rel10"]], "+ phase mirror 64": np.c_[M, P["mirror64"]], "phase mirror 64 alone": P["mirror64"]}
rows = []
for name, X in arms.items():
    for split, key in (("new person", subj), ("new day", date)):
        Pr = np.zeros((len(y), 5))
        for g in np.unique(key):
            te = key == g
            Pr[te] = make_model(JOBS).fit(X[~te], y[~te]).predict_proba(X[te])
        ok = Pr.argmax(1) == y
        votes = sum(Pr[sess == k].mean(0).argmax() == y[sess == k][0] for k in np.unique(sess))
        rows.append(dict(arm=name, split=split, acc=100 * ok.mean(), sessions=f"{votes}/390",
                         **{s: 100 * ok[subj == s].mean() for s in np.unique(subj)}))
        print(f"{name:22s} {split:10s} {rows[-1]['acc']:.1f}% {rows[-1]['sessions']}", flush=True)
pd.set_option("display.width", 260, "display.max_columns", 30)
R = pd.DataFrame(rows); print("\n", R.round(1).to_string(index=False)); R.to_csv(f"{S}/phase_eval.csv", index=False)
