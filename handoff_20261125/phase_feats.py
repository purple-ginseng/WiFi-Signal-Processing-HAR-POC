"""Phase feature variants on the same 4 s windows as rate_feats 'nearest'."""
import os, sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtbfm, rate_feats as R
from bfm_pipeline import FS, MAX_GAP
from joblib import Parallel, delayed
PH = np.array([n.startswith("ph") for n in rtbfm.feature_names()])
W, HOP = R.W, R.HOP

def ph10(phase):
    return rtbfm.window_features(np.zeros_like(phase), phase, FS)[PH]

def session(t, mag, kp, ks):
    _, g, pick, gap = R.grid_of(t)
    mag, kp, ks = mag[pick].astype(np.float64), kp[pick].astype(np.float64), ks[pick]
    old = np.pi * (1 / 512 + kp / 256)                    # current 9-bit decode
    true = np.pi * (1 / 64 + kp / 32)                     # correct 6-bit decode
    out = {k: [] for k in ("old10", "fix10", "rel10", "mirror64")}
    for a in range(0, len(mag) - W + 1, HOP):
        if gap[a:a + W].mean() > MAX_GAP: continue
        sl = slice(a, a + W)
        keep = R.lvl_keep(mag[sl])
        med = np.median(ks[sl], 0)
        rel = keep & (med >= 3) & (med <= 12)
        if rel.sum() < 4: rel = keep
        out["old10"].append(ph10(old[sl][:, keep])); out["fix10"].append(ph10(true[sl][:, keep]))
        out["rel10"].append(ph10(true[sl][:, rel]))
        u = np.unwrap(true[sl][:, rel], axis=0)
        out["mirror64"].append(np.concatenate([rtbfm.window_features(u, np.zeros_like(u), FS)[R.MAG_KEEP], R.extra(u)]))
    return {k: np.array(v, np.float32) for k, v in out.items()}

if __name__ == "__main__":
    S = sys.argv[1]
    m, c = np.load(f"{S}/mag5.npz"), np.load(f"{S}/codes5.npz")
    sess = m["sess"]; b = np.r_[0, np.where(np.diff(sess))[0] + 1, len(sess)]
    res = Parallel(n_jobs=int(os.environ.get("JOBS", 4)), verbose=1)(delayed(session)(m["t"][i:j], m["mag"][i:j], c["kphi"][i:j], c["kpsi"][i:j])
                                        for i, j in zip(b[:-1], b[1:]))
    out = {k: np.concatenate([r[k] for r in res]) for k in res[0]}
    ref = np.load(f"{S}/rate_feats.npz")["sess_nearest"]
    out["sess"] = np.concatenate([[i] * len(r["old10"]) for i, r in enumerate(res)])
    assert np.array_equal(out["sess"], ref), "windows do not line up with the magnitude features"
    np.savez(f"{S}/phase_feats.npz", **out)
    print({k: (v.shape, int((~np.isfinite(v)).sum())) for k, v in out.items()})
