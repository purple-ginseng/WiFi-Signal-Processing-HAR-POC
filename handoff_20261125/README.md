# Handoff — 25 Nov 2026 session

The aim is to go through `proposed-real-time-model.md` (the 14 Nov study) one point at a time.
- Points 1 and 2 (capture rate, gaps, reused reports) are **done**.
- Point 3 (why phase hurts) is **in progress**. A decoder bug was found, but the evaluation has not been run yet.

Nothing in `rtbfm.py`, `main_gui*.py`, `bfmtool/` or the deployed bundles was changed.

## 0. Before you start on the new machine

- **Commit or copy the untracked files.** `bfm_pipeline.py`, `proposed-real-time-model.md`, `tests/test_bfm_pipeline.py` and this folder are untracked in git, so a clone will not include them. Every script here imports `bfm_pipeline`.
- **Data:** you only need `bfm_real_imag_csv/bfm_5_class_ri_data.csv` (3.9 GB). The `mp` version is not used.
- **Where to run:** run every script from the repo root. They do `sys.path.insert(0, ".")` to import `bfm_pipeline` and `rtbfm`.
- **Worker counts:** the `JOBS` environment variable sets the number of feature workers (default 4) and XGBoost threads (default 10). On 12 GB, `phase_feats.py` with 8 workers was killed, probably out of memory.
- **Claude's memory files** live in `~/.claude/projects/<repo-path-slug>/memory/` on the old machine. They are not in git. Copy them across if you want the same background on the new machine. Section 5 below lists the facts that matter here.

### Rebuild the intermediate files (W = any work dir)

```bash
mkdir -p W
python handoff_20261125/to_mag.py bfm_real_imag_csv/bfm_5_class_ri_data.csv W/mag5.npz        # |r| float32, 1.2 GB peak, ~1 min
python handoff_20261125/to_codes.py bfm_real_imag_csv/bfm_5_class_ri_data.csv W/codes5.npz    # raw phi/psi codes (int8)
JOBS=8 python handoff_20261125/rate_feats.py W/mag5.npz W/rate_feats.npz                     # proposed 64 feats, 3 resampling arms, ~4 min
```

What these files contain:
- `mag5.npz`: `mag` (453792×234), `t`, `sess` (0–389), `sc`, and per-row `subject`/`activity`/`spot`/`date`.
- How sessions are found: the combined CSV has no session column. A timestamp jump of more than 5 s starts a new session, which gives exactly 390 sessions (13 subjects × 30). The combined CSV holds 80 MHz rows only.
- For the gap count, also run the awk one-liner in section 1.

## 1. Gaps after resampling to 10 Hz — done

```bash
awk -F, 'NR>1{print $1","($6==""?0:1)","($240==""?0:1)","$(NF-3)","$(NF-2)","$(NF-1)","$NF}' \
    bfm_real_imag_csv/bfm_5_class_ri_data.csv > W/ri_meta.csv
python handoff_20261125/gaps.py W/ri_meta.csv W/gap_report.csv
```

This uses `bfm_pipeline.resample_nearest`: a 10 Hz grid, nearest report, and a slot counts as a gap if no report is within 0.2 s.

| | Count |
|---|---|
| Grid slots | 467,782 |
| Gap slots | **4**, in 3 sessions: S07 Walking 10 Nov (2), S07 Bending A 10 Nov (1), S09 Sitting B 13 Nov (1); each from a single ~0.5 s pause |
| 4 s windows | 90,660 |
| Windows with any gap | 24 |
| Windows skipped (>15% gaps) | **0** |
| Slots reusing the previous report | 16,206 (3.46%) |

- **Rates:** 9.18–9.89 Hz, median 9.75 Hz. 45 sessions are below 9.5 Hz and 24 are below 9.4 Hz, mostly 14 Nov (S13 ≈ 9.2 Hz, S12).
- **Why a low rate causes no gaps:** reports are still about 0.11 s apart, well inside the 0.2 s tolerance. Instead the grid occasionally reuses a report, about 8% of slots in the 9.2 Hz sessions.
- Per-session table: `results/gap_report.csv`.

## 2. Do reused reports cost accuracy? — done: no

```bash
python handoff_20261125/rate_eval.py W/rate_feats.npz W/rate_eval.csv     # 40 XGB fits, ~4.5 min on 12 cores
```

Setup: the proposed 64 features, 4 s windows, XGBoost exactly as `bfm_pipeline.make_model`, no smoothing.

| Test | New person | New day | Sessions correct (new person / new day) |
|---|---|---|---|
| Nearest (current), 3.5% reused | 91.5 | 90.4 | 383 / 383 |
| Linear interpolation, no reuse | 91.2 | 90.1 | 380 / 381 |
| Nearest-trained model, test sessions thinned to 9.2 Hz (8.4% reused) | 90.5 | 89.2 | 381 / 380 |

- **Removing reuse does not help** (linear is −0.3). Keep nearest.
- **A full drop to 9.2 Hz at test time costs about 1 point.** This is a worst case: every session at the slowest rate.
- **Slow sessions are not the weak ones.** Sessions below 9.4 Hz score 93.1% (new person), the highest group. Correlation between session accuracy and reuse is −0.07, both raw and within subject.
- **Small mismatch with the proposal:** this baseline (91.5 / 90.4) is 0.2–0.3 below the proposal's 91.8 / 90.6. The proposal loaded per-session files; this uses the combined CSV. The cause was not investigated.
- Per-subject table: `results/rate_eval.csv`.

## 3. Why does phase hurt? — in progress

### 3a. Decoder bug (confirmed)

`bfmtool/preprocessor.py::decompress_bfm_from_angles` hard-codes **phi 9 bits, psi 7 bits** (MU codebook). The Pi sends **SU feedback with codebook 1: phi 6 bits (codes 0–63), psi 4 bits (codes 0–15)**. `to_codes.py` recovers the integer codes exactly from the ri CSV; the rounding error is below 1e-13.

The decoding formulas:
- `phi_dec = pi(1/512 + k/256)` versus `phi_true = pi(1/64 + k/32)`, so **phi_dec = phi_true / 8 exactly**.
- `psi_dec = pi(1/512 + k/256)` versus `psi_true = pi(1/64 + k/32)`, so **psi_dec = psi_true / 8 exactly**.
- The ratio is `r = h1/h2 = e^{j·phi} cos(psi)/sin(psi)`, so `|r| = cot(psi_dec)` and `angle(r) = phi_dec`.

What this means:
- **Magnitude path: not affected.** The proposal's "psi = arctan|r|" is really `90° − psi_dec = 90° − psi_true/8`, a linear rescale of the true psi. Tree models don't care. The level mask ranks subcarriers the same way. (Small correction to the proposal's wording: `|r|` is the cotangent of the decoded psi, not the tangent of psi.)
- **Phase path: broken only at wraps.** When the true phi crosses 360°→0° (code 63→0), the decoded value jumps 44.6°. rtbfm's `angle(exp(j·diff))` cannot unwrap that, because 44.6° is under 180°. So a one-level true step (5.6°) becomes a 63-level jump. Away from wraps, old phase features are exactly the true features ÷ 8.

How often fake wrap jumps happen (`phase_diag.py`):

| Day | Wrap share of steps | Subcarriers ever wrapping | Share of old `phdiff` total from wraps |
|---|---|---|---|
| 4, 6, 7, 10, 13 Nov | ~0 | 0 (20 on 6 Nov) | ~0% |
| **11 Nov (S08)** | 0.91% | 45 | **43%** |
| **14 Nov (S11–S13)** | 0.11% | 56 | **6%** |

- On every other day, phi codes stay below about 42, so they never wrap. On 11 and 14 Nov they reach 63.
- On 11 Nov, almost half of the phase-change signal is artefact. That is a day-specific shift, and it fits the proposal's finding that phase hurts new-day accuracy (−3.5) more than new-person accuracy (−1.4).

### 3b. Phase noise depends on psi (`phase_diag.py`, correct 6-bit decode)

Mean step per report, in code levels (both phi and psi levels are 5.6°):

| psi code (true °) | Share of data | Phase step, standing/sitting | Phase step, walking | Walk/still | psi step, still | psi walk/still |
|---|---|---|---|---|---|---|
| 0 (2.8°) | 0.4% | 2.73 | 5.34 | 1.96 | 0.12 | 2.32 |
| 2 (14°) | 1.8% | 0.47 | 1.04 | 2.21 | 0.12 | 2.21 |
| 5 (31°) | 7.7% | 0.24 | 0.47 | 1.93 | 0.11 | 2.01 |
| 7–8 (42–48°) | 36% | 0.21 | 0.41 | ~1.98 | ~0.10 | ~2.08 |
| 10 (59°) | 7.5% | 0.24 | 0.45 | 1.89 | 0.11 | 1.90 |
| 13 (76°) | 1.9% | 0.44 | 0.98 | 2.23 | 0.11 | 2.16 |
| 15 (87°) | 0.2% | 1.90 | 4.11 | 2.17 | 0.20 | 1.94 |

- **Phase does carry motion information.** Walking moves phase about 2× more than standing at every psi, which is the same contrast as psi itself. In absolute terms phase moves about 2× more than psi.
- **Phase noise blows up toward psi = 0° or 90°.** There, one transmit antenna's weight is near zero, so its phase is ill-defined. Noise is 9–13× higher at the extremes than near 45°.
- **The level mask only covers one side.** `lvl25` drops large `|r|` (small psi). It keeps the psi-near-90° side, where phase is also noisy. The old phase features average over all kept subcarriers, so those noisy ones get mixed in.
- **Physical meaning:** phi is the relative phase between the router's two transmit antennas. Being relative, it cancels the common CFO and timing offsets, but any per-chain phase offset at the router is included (it may change with router restarts; not tested).

### 3c. Next step (built, not yet run)

```bash
JOBS=8 python handoff_20261125/phase_feats.py W     # needs W/mag5.npz, W/codes5.npz, W/rate_feats.npz
python handoff_20261125/phase_eval.py W              # 6 arms × 20 fits; also prints the per-day standing check
```

- `phase_feats.py` builds features on the same windows as `rate_feats` nearest, and asserts that they line up.
  - `old10`: rtbfm's 10 phase features, 9-bit decode, `lvl25`-kept subcarriers (reproduces the proposal's drop).
  - `fix10`: the same features with the correct 6-bit decode.
  - `rel10`: correct decode, kept subcarriers whose window-median psi code is 3–12.
  - `mirror64`: correct phase unwrapped in time on the reliable subcarriers, put through the full magnitude feature set (33 rtbfm + 31 extra).
- `phase_eval.py` arms: magnitude 64 alone, magnitude + `old10`, + `fix10`, + `rel10`, + `mirror64`, and `mirror64` alone.
- **Expectations, to check:**
  - `fix10` should differ from `old10` only through 11 and 14 Nov, so watch S08 and the new-day split.
  - If `mirror64` alone gets close to magnitude alone, phase carries the same information (largely redundant). The gain from combining then decides whether phase is worth keeping.
- **Ideas still open:** weight subcarriers by phase reliability (`sin 2·psi`) instead of a hard cut; use complex-change features on the true `r`; try 8 s windows.

### On "dropping phase makes it an RSSI method"

It doesn't. RSSI is one power number per packet. The magnitude features use 234 per-subcarrier **psi** angles. psi is itself a beamforming angle: the split of the steering vector between the router's two antennas, per subcarrier. It is spatial and frequency-resolved channel state, not received power. Dropping phi loses the relative-phase half of that state, which is what point 3 is trying to recover properly.

## 4. Things that must not be forgotten

- **Fixing `bfmtool/preprocessor.py` changes rtbfm's phase features**, so the deployed `bfm_rt_model_*.joblib` bundles would need retraining (see the CLAUDE.md landmine). Magnitude features only change scale, which doesn't matter for XGBoost but does for any linear or thresholded use. Fix it in a new code path, or re-decode from codes as `to_codes.py` does.
- **The decoder fix should read the codebook per frame.** Older Oct 2025 training pcaps mixed SU and about 5.7% MU frames. The official Nov set is all SU (codes never exceed 63/15).
- **Proposal step 4 ("no phase") rests on mis-decoded phase.** Re-test with 3c before treating it as settled.

## 5. Background facts (from memory files, not in the repo)

- **Official set:** S01–S13, 390 sessions, ch149 at 80 MHz, 4–14 Nov.
- **Setup changes between days:** the router, Pi and antennas are rebuilt every day, so the static fingerprint changes daily by design. On 10 Nov (S07) the antennas were straightened. 11 Nov (S08) was an ill-conditioned placement: high-`|r|` subcarriers very noisy, level spread 15–18 dB.
- **Masking:** the `lvl25` mask (drop the top quarter by static level) fixed S08 (68% → 91%). The proposal's version picks the mask per window.
- **Report the right splits:** leave-one-subject-out and leave-one-day-out are the main numbers. Pooled LOSO overstates accuracy.
- **Known weak spots:**
  - Standing vs sitting.
  - Walking vs bending.
  - S09 / 13 Nov.
  - S02 Spot B bends, likely shallow and pending a video check.
  - Sitting vs empty room, set aside by choice.
- **Day confound:** rtbfm features can tell days apart perfectly. Never re-record a single class on a new day, or the model learns the day.
- **Router restarts** appear to reset the static channel state.
- **Capture-rate rule:** 9.5–11 Hz was earlier measured as safe. Section 2 extends that down to about 9.2 Hz, at roughly 1 point.
