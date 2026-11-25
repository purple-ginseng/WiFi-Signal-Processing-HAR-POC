import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from bfm_pipeline import resample_nearest, FS, MAX_GAP, WIN_S, HOP_S
d = pd.read_csv(sys.argv[1], header=None, names="t w80 w2 subject activity spot date".split())
d["sess"] = np.r_[0, np.cumsum(np.diff(d.t.values) > 5)]
rows = []
for s, g in d.groupby("sess"):
    t = g.t.values
    idx, gap = resample_nearest(t, np.arange(len(t)))
    runs = np.diff(np.r_[0, gap.astype(int), 0]); starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]
    w, hop = int(WIN_S * FS), int(HOP_S * FS)
    wins = [gap[a:a + w].mean() for a in range(0, len(gap) - w + 1, hop)]
    dt = np.diff(t)
    rows.append(dict(subject=g.subject.iloc[0], activity=g.activity.iloc[0], spot=g.spot.iloc[0], date=g.date.iloc[0],
        reports=len(t), dur=t[-1] - t[0], rate=(len(t) - 1) / (t[-1] - t[0]), slots=len(gap), gaps=int(gap.sum()),
        gap_runs=len(starts), longest=int((ends - starts).max()) if len(starts) else 0,
        reused=int((np.diff(idx) == 0).sum()), maxdt=dt.max(), dt_gt02=int((dt > 0.2).sum()),
        windows=len(wins), win_any_gap=int(sum(x > 0 for x in wins)), win_skipped=int(sum(x > MAX_GAP for x in wins))))
r = pd.DataFrame(rows); r.to_csv(sys.argv[2], index=False)
print("sessions", len(r), "| slots", r.slots.sum(), "| gap slots", r.gaps.sum(), f"({r.gaps.sum()/r.slots.sum():.3%})",
      "| gap runs", r.gap_runs.sum(), "| sessions with any gap", (r.gaps > 0).sum())
print("windows", r.windows.sum(), "| with any gap", r.win_any_gap.sum(), "| skipped (>15%)", r.win_skipped.sum())
print("reused-report slots", r.reused.sum(), f"({r.reused.sum()/r.slots.sum():.2%})")
print("rate: min %.2f  p5 %.2f  median %.2f  max %.2f" % (r.rate.min(), r.rate.quantile(.05), r.rate.median(), r.rate.max()))
print("sessions below 9.5 Hz:", (r.rate < 9.5).sum(), " below 9.4:", (r.rate < 9.4).sum())
print("\nper day:"); print(r.groupby("date").agg(sessions=("rate","size"), rate_min=("rate","min"), rate_med=("rate","median"),
      gaps=("gaps","sum"), sess_w_gap=("gaps", lambda x: (x>0).sum()), reused=("reused","sum"), skipped=("win_skipped","sum")).round(2).to_string())
print("\nsessions with gaps:"); print(r[r.gaps > 0].sort_values("gaps", ascending=False)[["subject","activity","spot","date","rate","gaps","gap_runs","longest","maxdt","win_skipped"]].round(2).to_string())
print("\nslowest 8 sessions:"); print(r.nsmallest(8, "rate")[["subject","activity","spot","date","rate","gaps","reused","maxdt"]].round(3).to_string())
