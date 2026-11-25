"""Recover raw SU angle codes (phi 6-bit, psi 4-bit) from the 9/7-bit-decoded real/imag ratios."""
import sys, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
cols = pd.read_csv(src, nrows=0).columns
re_ = [c for c in cols if c.endswith("_Ratio_Real")]; im_ = [c.replace("_Real", "_Imag") for c in re_]
kp, ks = [], []
for ch in pd.read_csv(src, usecols=re_ + im_, chunksize=20000):
    r = ch[re_].to_numpy() + 1j * ch[im_].to_numpy()
    a = np.mod(np.angle(r), 2 * np.pi) / np.pi * 256 - 0.5
    b = np.arctan2(1, np.abs(r)) / np.pi * 256 - 0.5
    assert np.abs(a - np.round(a)).max() < 1e-3 and np.abs(b - np.round(b)).max() < 1e-3
    kp.append(np.round(a).astype(np.int8)); ks.append(np.round(b).astype(np.int8))
kp, ks = np.concatenate(kp), np.concatenate(ks)
print("phi codes", kp.min(), kp.max(), " psi codes", ks.min(), ks.max())
np.savez(out, kphi=kp, kpsi=ks)
