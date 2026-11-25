"""Nearest vs linear (train+test), and nearest-trained models tested on 9.2 Hz-thinned sessions."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from bfm_pipeline import make_model, CLASSES
import os; d = np.load(sys.argv[1]); JOBS = int(os.environ.get("JOBS", 10))
subj, act, date, rate = d["subject"], d["activity"], d["date"], d["rate"]
def arm(how):
    s = d[f"sess_{how}"]; return d[f"X_{how}"], s, np.array([CLASSES.index(a) for a in act[s]])
Xn, sn, yn = arm("nearest"); Xl, sl, yl = arm("linear"); Xt, st, yt = arm("thin")
sess_acc = {}
rows = []
for split, key in (("new person", subj), ("new day", date)):
    P = {k: np.zeros((len(v), 5)) for k, v in (("nearest", yn), ("linear", yl), ("thin", yt))}
    for g in np.unique(key):
        ten, tel, tet = key[sn] == g, key[sl] == g, key[st] == g
        m = make_model(JOBS).fit(Xn[~ten], yn[~ten])
        P["nearest"][ten] = m.predict_proba(Xn[ten]); P["thin"][tet] = m.predict_proba(Xt[tet])
        P["linear"][tel] = make_model(JOBS).fit(Xl[~tel], yl[~tel]).predict_proba(Xl[tel])
        print(split, g, "done", flush=True)
    for how, (s, y) in (("nearest", (sn, yn)), ("linear", (sl, yl)), ("thin", (st, yt))):
        pr = P[how].argmax(1); ok = pr == y
        votes = sum(P[how][s == k].mean(0).argmax() == y[s == k][0] for k in np.unique(s))
        acc = pd.Series(ok).groupby(s).mean(); sess_acc[(split, how)] = acc
        rows.append(dict(split=split, arm=how, window_acc=100 * ok.mean(), sessions=f"{votes}/{len(np.unique(s))}",
                         **{x: 100 * ok[subj[s] == x].mean() for x in np.unique(subj)}))
pd.set_option("display.width", 250, "display.max_columns", 30)
R = pd.DataFrame(rows); print("\n", R.round(1).to_string(index=False))
R.to_csv(sys.argv[2], index=False)
# per-session view of the nearest arm: accuracy against rate / reuse, raw and within subject
for split in ("new person", "new day"):
    a = sess_acc[(split, "nearest")]; t = sess_acc[(split, "thin")]
    df = pd.DataFrame(dict(acc=a, thin=t, rate=rate[a.index], reuse=d["reuse_nearest"][a.index], subj=subj[a.index], day=date[a.index]))
    df["bucket"] = pd.cut(df.rate, [0, 9.4, 9.6, 10], labels=["<9.4", "9.4-9.6", ">9.6"])
    w = df.assign(acc_c=df.acc - df.groupby("subj").acc.transform("mean"), reuse_c=df.reuse - df.groupby("subj").reuse.transform("mean"))
    print(f"\n[{split}] session accuracy by rate bucket (nearest):")
    print(df.groupby("bucket", observed=True).agg(sessions=("acc", "size"), acc=("acc", "mean"), thin_acc=("thin", "mean")).mul([1, 100, 100]).round(1).to_string())
    print(f"corr(acc, reuse) raw {np.corrcoef(df.acc, df.reuse)[0,1]:+.3f}; within subject {np.corrcoef(w.acc_c, w.reuse_c)[0,1]:+.3f}")
    # thinning effect on sessions that were actually thinned (rate > 9.2)
    th = df[df.rate > 9.25]
    print(f"thinned sessions n={len(th)}: nearest {100*th.acc.mean():.1f}% -> thinned {100*th.thin.mean():.1f}%; "
          f"sessions worse {int((th.thin < th.acc - 1e-9).sum())}, better {int((th.thin > th.acc + 1e-9).sum())}")
