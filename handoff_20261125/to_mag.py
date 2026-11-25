import sys, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
cols = pd.read_csv(src, nrows=0).columns
re_ = [c for c in cols if c.endswith("_Ratio_Real")]
im_ = [c.replace("_Real", "_Imag") for c in re_]
sc = np.array([int(c.split("_")[1]) for c in re_])
mags, ts, meta = [], [], []
for ch in pd.read_csv(src, usecols=["timestamp", *re_, *im_, "subject", "activity", "spot", "date"],
                      dtype={c: np.float32 for c in re_ + im_}, chunksize=20000):
    mags.append(np.hypot(ch[re_].to_numpy(np.float32), ch[im_].to_numpy(np.float32)))
    ts.append(ch["timestamp"].to_numpy(float)); meta.append(ch[["subject", "activity", "spot", "date"]].copy())
mag, t = np.concatenate(mags), np.concatenate(ts)
m = pd.concat(meta, ignore_index=True)
assert np.isfinite(mag).all(), "NaN in magnitude"
sess = np.r_[0, np.cumsum(np.diff(t) > 5)]
np.savez(out, mag=mag, t=t, sess=sess, sc=sc, subject=m.subject.to_numpy(str), activity=m.activity.to_numpy(str),
         spot=m.spot.to_numpy(str), date=m.date.to_numpy(str))
print(mag.shape, sess.max() + 1, "sessions")
