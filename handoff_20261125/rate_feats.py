"""Proposed 4 s features under three resampling arms: nearest, linear, thinned-to-9.2-Hz nearest."""
import os, sys, numpy as np
sys.path.insert(0, ".")
import rtbfm
from bfm_pipeline import FS, MAX_GAP, GAP_TOL
from joblib import Parallel, delayed
MAG_KEEP = np.array([not n.startswith("ph") for n in rtbfm.feature_names()])
W, HOP, TARGET = 40, 5, 9.2

def lvl_keep(mag, q=75):
    lvl = np.log(mag.mean(0)); return lvl <= np.percentile(lvl, q)

def extra(m):
    T = m.shape[0]; t = np.linspace(-1, 1, T)[:, None]
    c = m - m.mean(0); c = c - t * (c * t).sum(0) / (t * t).sum()
    sd = c.std(0); mv = sd > 1e-6
    if mv.sum() < 2: return np.zeros(31, np.float32)
    c, sd = c[:, mv], sd[mv]
    F = np.fft.rfft(c * np.hanning(T)[:, None], axis=0)
    P = (F.real ** 2 + F.imag ** 2).sum(1)[1:]; f = np.fft.rfftfreq(T, 1 / FS)[1:]
    p = P / (P.sum() + 1e-12); cs = np.cumsum(p)
    out = [(f * p).sum(), f[p.argmax()], p.max(), -(p * np.log(p + 1e-12)).sum() / np.log(len(p)),
           f[np.searchsorted(cs, 0.5)], f[min(np.searchsorted(cs, 0.9), len(f) - 1)]]
    out += [p[(f >= lo) & (f < lo + 0.25)].sum() for lo in np.arange(0, 2.5, 0.25)]
    e = (np.abs(np.diff(c, axis=0)) / sd).mean(1); e = np.convolve(e, np.ones(3) / 3, mode="valid")
    em = e.mean() + 1e-12; ec = e - e.mean()
    E = np.abs(np.fft.rfft(ec * np.hanning(len(ec)))) ** 2; fe = np.fft.rfftfreq(len(ec), 1 / FS)
    E, fe = E[1:], fe[1:]
    ac = np.correlate(ec, ec, "full")[len(ec) - 1:]; ac = ac / (ac[0] + 1e-12)
    lag = lambda s: ac[min(int(s * FS), len(ac) - 1)]
    k = np.arange(3, min(31, len(ac)))
    out += [e.std() / em, e.max() / (np.median(e) + 1e-12), np.percentile(e, 90) / (np.percentile(e, 10) + 1e-12),
            fe[E.argmax()], E.max() / (E.sum() + 1e-12), lag(0.5), lag(1.0), lag(1.5), lag(2.0),
            ac[k].max(), k[ac[k].argmax()] / FS]
    nb = T // 10; blk = c[:nb * 10].reshape(nb, 10, -1)
    out.append(np.median(blk.std(1).mean(0) / sd))
    h1, h2 = c[:T // 2].std(0), c[T // 2:].std(0)
    out.append(np.median(np.minimum(h1, h2) / (np.maximum(h1, h2) + 1e-12)))
    out.append((sd > 0.5 * sd.max()).mean())
    s2 = np.sort(sd ** 2)[::-1]; out.append(s2[:len(s2) // 10].sum() / s2.sum())
    return np.asarray(out, np.float32)

def proposed_features(mag):
    keep = lvl_keep(mag); psi = np.arctan(mag)[:, keep]
    base = rtbfm.window_features(psi, np.zeros_like(psi), FS)[MAG_KEEP]
    return np.concatenate([base, extra(psi)])

def grid_of(t):
    t = t - t[0]; g = np.arange(0, t[-1] + 1e-9, 1 / FS)
    j = np.clip(np.searchsorted(t, g), 1, len(t) - 1)
    pick = np.where(np.abs(t[j - 1] - g) <= np.abs(t[j] - g), j - 1, j)
    return t, g, pick, np.abs(t[pick] - g) > GAP_TOL

def resample(t, mag, how):
    if how == "thin":
        rate = (len(t) - 1) / (t[-1] - t[0])
        if rate > TARGET:
            keep = np.r_[True, np.diff(np.floor(np.arange(len(t)) * TARGET / rate)) > 0]; keep[-1] = True
            t, mag = t[keep], mag[keep]
    t, g, pick, gap = grid_of(t)
    if how == "linear":
        z = np.stack([np.interp(g, t, mag[:, s]) for s in range(mag.shape[1])], 1)
    else:
        z = mag[pick]
    return z, gap, (np.diff(pick) == 0).mean()

def session(t, mag):
    res = {}
    for how in ("nearest", "linear", "thin"):
        z, gap, reuse = resample(t, mag.astype(np.float64), how)
        starts = [a for a in range(0, len(z) - W + 1, HOP) if gap[a:a + W].mean() <= MAX_GAP]
        res[how] = (np.array([proposed_features(z[a:a + W]) for a in starts], np.float32), reuse)
    return res

if __name__ == "__main__":
    d = np.load(sys.argv[1]); mag, t, sess = d["mag"], d["t"], d["sess"]
    bounds = np.r_[0, np.where(np.diff(sess))[0] + 1, len(sess)]
    res = Parallel(n_jobs=int(os.environ.get("JOBS", 4)), verbose=2)(delayed(session)(t[a:b], mag[a:b]) for a, b in zip(bounds[:-1], bounds[1:]))
    out = {}
    for how in ("nearest", "linear", "thin"):
        out[f"X_{how}"] = np.concatenate([r[how][0] for r in res])
        out[f"sess_{how}"] = np.concatenate([[i] * len(r[how][0]) for i, r in enumerate(res)])
        out[f"reuse_{how}"] = np.array([r[how][1] for r in res])
    first = bounds[:-1]
    for k in ("subject", "activity", "spot", "date"): out[k] = d[k][first]
    out["rate"] = np.array([(b - a - 1) / (t[b - 1] - t[a]) for a, b in zip(bounds[:-1], bounds[1:])])
    np.savez(sys.argv[2], **out)
    for how in ("nearest", "linear", "thin"):
        print(how, out[f"X_{how}"].shape, "mean reuse %.2f%%" % (100 * out[f"reuse_{how}"].mean()),
              "nonfinite", int((~np.isfinite(out[f"X_{how}"])).sum()))
