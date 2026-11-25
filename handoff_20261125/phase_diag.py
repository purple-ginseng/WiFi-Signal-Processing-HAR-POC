import sys, numpy as np, pandas as pd
S = sys.argv[1]  # work dir holding mag5.npz and codes5.npz
c = np.load(f"{S}/codes5.npz"); m = np.load(f"{S}/mag5.npz")
kp, ks, sess = c["kphi"].astype(np.int16), c["kpsi"], m["sess"]
act, date, t = m["activity"], m["date"], m["t"]
same = np.r_[False, sess[1:] == sess[:-1]]               # step stays inside a session
dk = np.diff(kp, axis=0)[same[1:]]                       # raw code step
true_step = (dk + 32) % 64 - 32                          # correct 6-bit wrap: steps in levels (5.6 deg)
old_step = np.abs(np.angle(np.exp(1j * np.pi * dk / 256)))  # what rtbfm sees with 9-bit decode (rad)
wrap = np.abs(dk) > 32                                   # crossing 0/360 deg, unwrappable only in 6-bit
d_step = date[1:][same[1:]]; a_step = act[1:][same[1:]]
print("== fake wrap jumps (true step crosses 0/360 deg) ==")
rows = []
for d in np.unique(date):
    k = d_step == d
    w = wrap[k]; o = old_step[k]
    rows.append(dict(date=d, wrap_share_of_steps=100 * w.mean(), subcarriers_ever_wrapping=int(w.any(0).sum()),
                     old_phdiff_from_wraps=100 * o[w].sum() / o.sum(), median_psi_code=np.median(ks[date == d])))
print(pd.DataFrame(rows).round(3).to_string(index=False))
# phase noise vs psi: near-static (Standing+Sitting) vs Walking, per psi code of the subcarrier at each step
psi_at = ks[1:][same[1:]]
ap = np.abs(true_step); aps = np.abs(np.diff(ks.astype(np.int16), axis=0)[same[1:]])
out = []
for q in range(16):
    sel = psi_at == q
    st = sel & np.isin(a_step, ["Standing", "Sitting"])[:, None]; wk = sel & (a_step == "Walking")[:, None]
    if st.sum() < 1000: continue
    out.append(dict(psi_code=q, psi_deg=round(180 * (1 / 64 + q / 32), 1), share=100 * sel.mean(),
                    phase_step_static=ap[st].mean(), phase_step_walk=ap[wk].mean(), phase_walk_over_static=ap[wk].mean() / ap[st].mean(),
                    psi_step_static=aps[st].mean(), psi_step_walk=aps[wk].mean(), psi_walk_over_static=aps[wk].mean() / aps[st].mean()))
print("\n== mean |step| per report, in code levels (phase 5.6 deg, psi 5.6 deg), by psi code ==")
print(pd.DataFrame(out).round(3).to_string(index=False))
