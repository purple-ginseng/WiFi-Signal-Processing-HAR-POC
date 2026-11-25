# Proposed real-time model

Study date: 14 Nov 2026. Data: 390 official sessions, subjects S01–S13, seven collection days (4–14 Nov), channel 149, five activities (Walking, Waving, Bending, Standing, Sitting).

Nothing in `rtbfm.py`, `main_gui3.py` or the deployed bundles has been changed. This file records the proposal so it can be implemented later.

## 1. Summary

| Model | Features | New person, 4 s | New day, 4 s | New person, 8 s | New day, 8 s |
|---|---|---|---|---|---|
| Current `rtbfm.py` | 43 | 88.6% | 84.5% | not run | not run |
| Proposed set | 64 | 91.8% | 90.6% | 94.0% | 93.0% |

- **New person** = leave-one-subject-out: train on 12 subjects, test the 13th.
- **New day** = leave-one-day-out: train on six days, test the seventh (person and day both unseen).
- All figures are window accuracy with no probability smoothing.

## 2. How to implement the proposed set

The proposed set changes the input and the features. The classifier is unchanged.

### Step 1 — Load and resample (unchanged)

Same as `bfm_pipeline.load_session`:

1. Read the real/imag CSV and keep 80 MHz rows only.
2. Build the complex ratio `r = real + j·imag` per subcarrier (234 subcarriers).
3. Resample to a 10 Hz grid by nearest report; flag a grid slot as a gap when no report is within 0.2 s.
4. Cut windows of 4 s (40 samples) with a 0.5 s hop (5 samples). Skip a window when more than 15% of its slots are gaps.

### Step 2 — Angle input

Replace `log|r|` with the angle

```
psi = arctan(|r|)
```

`|r|` is the tangent of the beamforming angle psi, which the router reports with uniform quantisation steps. `log|r|` stretches without limit when `|r|` is very large or very small, so the same small change looks huge on those subcarriers. `psi` is bounded between 0 and 90° and treats every subcarrier on the same scale.

### Step 3 — Per-window subcarrier mask

For each window, on its own data only:

1. `level = log(mean over time of |r|)` per subcarrier.
2. Keep the subcarriers whose level is at or below the 75th percentile of that window; drop the top quarter (about 59 of 234).

This removes the ill-conditioned subcarriers that appear after some router restarts. It needs no empty-room recording: choosing the mask from the window itself scored the same as choosing it from the day's empty room (89.4% against 89.3% on a new person).

### Step 4 — No phase features

Do not compute the 10 phase features (`phdiff_*`, `phrough_*`). Phase is not passed to the feature code at all.

### Step 5 — Features (64 in total)

1. **33 magnitude features** — the existing `rtbfm.window_features` maths applied to `psi` on the kept subcarriers, without the phase block:
   - `sd_*`, `absdiff_*`, `rough_*` (5 summaries each = 15)
   - `pfrac*` for the five bands (3 summaries each = 15)
   - `corr_mean`, `corr_absmean`, `corr_std` (3)
2. **31 new features** — described in section 6, reference code in the appendix.

### Step 6 — Model

XGBoost, same settings as `bfm_pipeline.make_model`: 250 trees, learning rate 0.08, depth 4, `reg_lambda` 1.0, `random_state` 0. Five classes.

### Integration notes

- Put the new feature code in a new module rather than editing `rtbfm.py`. The deployed `bfm_rt_model_*.joblib` bundles depend on `rtbfm.window_features` staying exactly as it is.
- A new model bundle must be trained on the new features, and the bundle should record the feature names so a mismatch is caught.
- `main_gui3.py` would import the new module for the live detector; the pytest suite needs a matching invariant test for the new bundle.
- Optional output smoothing (not included in the results below): an exponential average of the 4 s probabilities with alpha 0.3 gave 93.4% / 92.3%.

## 3. Results of the final model (no smoothing)

Sessions = recordings whose majority vote over windows is correct.

### 3.1 Per person (new-person split)

| Subject | 4 s window acc | 4 s sessions | 8 s window acc | 8 s sessions |
|---|---|---|---|---|
| S01 | 87.8% | 30/30 | 89.9% | 29/30 |
| S02 | 90.4% | 28/30 | 92.3% | 28/30 |
| S03 | 95.3% | 30/30 | 96.9% | 30/30 |
| S04 | 97.3% | 30/30 | 97.7% | 30/30 |
| S05 | 95.2% | 30/30 | 96.0% | 30/30 |
| S06 | 92.6% | 30/30 | 93.1% | 30/30 |
| S07 | 88.7% | 30/30 | 90.7% | 29/30 |
| S08 | 89.2% | 30/30 | 94.4% | 30/30 |
| S09 | 83.8% | 27/30 | 86.5% | 28/30 |
| S10 | 88.5% | 29/30 | 92.8% | 29/30 |
| S11 | 96.8% | 30/30 | 98.7% | 30/30 |
| S12 | 93.6% | 30/30 | 95.7% | 30/30 |
| S13 | 94.2% | 30/30 | 97.1% | 30/30 |
| **All** | **91.8%** | **384/390** | **94.0%** | **383/390** |

### 3.2 Per day (new-day split)

| Day | Subjects | 4 s window acc | 4 s sessions | 8 s window acc | 8 s sessions |
|---|---|---|---|---|---|
| 4 Nov | S01 | 87.8% | 30/30 | 89.9% | 29/30 |
| 6 Nov | S02, S03 | 93.5% | 58/60 | 95.7% | 59/60 |
| 7 Nov | S04, S05, S06 | 93.5% | 90/90 | 93.6% | 88/90 |
| 10 Nov | S07 | 88.7% | 30/30 | 90.7% | 29/30 |
| 11 Nov | S08 | 89.2% | 30/30 | 94.4% | 30/30 |
| 13 Nov | S09, S10 | 82.9% | 56/60 | 87.4% | 56/60 |
| 14 Nov | S11, S12, S13 | 93.0% | 90/90 | 95.6% | 90/90 |
| **All** | | **90.6%** | **384/390** | **93.0%** | **381/390** |

Per subject inside the new-day split (window accuracy, rounded):

| Window | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | S10 | S11 | S12 | S13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 s | 88 | 91 | 96 | 96 | 94 | 90 | 89 | 89 | 81 | 85 | 96 | 92 | 91 |
| 8 s | 90 | 93 | 98 | 97 | 94 | 90 | 91 | 94 | 84 | 91 | 98 | 94 | 94 |

### 3.3 Per activity (recall, %)

| Split | Window | Walking | Waving | Bending | Standing | Sitting |
|---|---|---|---|---|---|---|
| New person | 4 s | 93.9 | 94.4 | 93.1 | 86.4 | 91.3 |
| New day | 4 s | 90.6 | 94.0 | 92.4 | 85.5 | 90.6 |
| New person | 8 s | 96.2 | 96.0 | 95.6 | 89.5 | 92.6 |
| New day | 8 s | 92.9 | 95.7 | 95.7 | 88.8 | 91.9 |

### 3.4 Confusion matrix, 4 s, new day (row = true activity, % of its windows)

| True \ Predicted | Walking | Waving | Bending | Standing | Sitting |
|---|---|---|---|---|---|
| Walking | 90.6 | 1.5 | 6.2 | 0.1 | 1.6 |
| Waving | 2.8 | 94.0 | 0.4 | 2.3 | 0.5 |
| Bending | 7.2 | 0.5 | 92.4 | 0.0 | 0.0 |
| Standing | 0.3 | 3.3 | 0.0 | 85.5 | 10.9 |
| Sitting | 0.5 | 0.2 | 0.0 | 8.7 | 90.6 |

## 4. What did not help

All compared against the same feature set and window length.

| Idea | Result | Baseline it was compared with |
|---|---|---|
| Deeper or more regularised XGBoost (depth 3 with 500 trees; depth 6 with 300 trees; row and column subsampling) | 94.0–94.2% / 93.2% | 94.0% / 93.0% (8 s) |
| Ensemble of XGBoost + ExtraTrees + RandomForest | 94.2% / 93.5% | 94.0% / 93.0% (8 s) |
| Adding 4 s features to the 8 s window (multi-scale) | 94.2% / 93.2% | 94.0% / 93.0% (8 s) |
| Mask keeping the subcarriers nearest psi = 45° (keep 75%), magnitude only | 88.9% new person | 90.9% (4 s, level mask) |
| Same mask keeping 50% | 88.9% / 86.7% | 90.9% / 89.5% (4 s, level mask) |
| Subtracting the latest empty room's feature values | 87.2% / 86.8% | 90.9% / 89.5% (4 s) |
| Subtracting 30 s of the person's own standing | 80.7% / 78.5% | 90.9% / 89.5% (4 s) |
| Enrolment rows (5 × 30 s) added to training with weight 10 or 30 | 94.5–94.6% new person | 94.1% (8 s) |
| Centring on a 5 × 30 s enrolment recording | 94.4% / 93.5% at 8 s; 91.7% / 91.5% at 4 s | 94.1% / 93.1%; 91.8% / 90.6% |
| Centring on a 5 × 15 s enrolment recording | 93.7% / 93.5% at 8 s; 90.7% / 90.7% at 4 s | same as above |
| Keeping the phase features | 89.5% / 86.0% | 90.9% / 89.5% (4 s) |

Notes:

- The single-reference subtractions fail because one short recording is too noisy an estimate of a person's or a day's level.
- Enrolment centring barely moves the average but does lift the weakest subject (87% to 89% at 8 s, 84% to 86% at 4 s).
- Centring each person on **all** of their own data reached 95.7% / 94.2% at 8 s. That is an upper bound, not a usable method: it uses every recording of the test person with equal time per activity.

## 5. What the model is still weak at

1. **Standing against sitting.** Standing has the lowest recall (86% at 4 s). About 10–11% of standing windows are read as sitting and 8–9% of sitting windows as standing. Worst cases at 4 s: S09 standing 59%, S01 standing 72% and sitting 74%, S10 standing 75%, S07 standing 77%, S06 sitting 79%.
2. **Walking against bending.** 6% of walking windows are read as bending and 7% of bending as walking on a new day. Worst cases at 4 s: S02 bending 64% (the two Spot B bends that were suspected shallow), S10 walking 81%, S13 walking 85%, S03 and S06 walking 87%.
3. **S09 and the 13 Nov day.** S09 is the weakest subject in every configuration (83.8% / 81% at 4 s). Standing is read as sitting, and waving recall is 78%. The 13 Nov day as a whole scores 82.9% at 4 s.
4. **Small waves.** Standing is read as waving in 3% of windows; S01 and S07 each lose one standing session to waving at 8 s.
5. **Sessions that stay wrong at 4 s (6 of 390, new person):** S02 Bending Spot B Rounds 1 and 3 → Walking; S09 Standing Spot A Round 1 and Spot B Round 1 → Sitting; S09 Bending Spot B Round 2 → Walking; one S10 session.
6. **Not evaluated:** behaviour during a change of activity (every recording holds one constant activity), and sitting against an empty room (set aside earlier).

## 6. The new features

All 31 are computed on one window of `psi` after the mask (T samples × S kept subcarriers).

Common preparation:

- Remove each subcarrier's mean and straight-line trend over the window. Call the result `c`.
- `sd` = standard deviation of `c` per subcarrier. Subcarriers with `sd` below 1e-6 (never changed) are left out.
- If fewer than two subcarriers moved, all 31 features are zero.

### 6.1 Pooled spectrum shape (6 features)

Calculation: multiply `c` by a Hann window, take the FFT along time, square the amplitude, and add the power of all subcarriers into one spectrum. Drop the 0 Hz bin and divide by the total so the spectrum sums to 1 (`p` over frequencies `f`). Because power is added, the most active subcarriers dominate.

| Feature | Calculation | What it represents |
|---|---|---|
| `spec_centroid` | sum of `f · p` | Average frequency of the motion: slow sway is low, fast arm motion is high |
| `spec_peakf` | `f` at the largest `p` | The dominant rhythm of the movement |
| `spec_peakfrac` | largest `p` | How much of the motion sits at that one rhythm (periodic against irregular) |
| `spec_entropy` | `−sum(p · log p) / log(number of bins)` | 0 = one pure rhythm, 1 = power spread evenly like noise |
| `spec_p50f` | frequency below which half the power lies | Median motion frequency, less sensitive to one peak |
| `spec_p90f` | frequency below which 90% of the power lies | How far the motion extends into fast changes |

### 6.2 Fine frequency bands (10 features)

`fine0.00`, `fine0.25`, … `fine2.25`: share of `p` in each 0.25 Hz band from 0 to 2.5 Hz.

The existing `pfrac` bands are wide (0–0.5, 0.5–1, 1–2, 2–3, 3–5 Hz) and averaged per subcarrier. The fine bands separate rhythms the wide ones merge: in this data the main walking peak is near 0.31 Hz, bending near 0.62 Hz and waving near 1 Hz. With a 4 s window each band is exactly one FFT bin; with 8 s each band holds two.

### 6.3 Motion-envelope rhythm (11 features)

Calculation of the envelope: take the absolute sample-to-sample change of `c`, divide each subcarrier by its own `sd`, average over subcarriers, then smooth with a 3-sample moving average. The result `e(t)` is one curve that says how much the channel is changing at each instant, regardless of direction or of how strong each subcarrier is.

| Feature | Calculation | What it represents |
|---|---|---|
| `env_cv` | std of `e` / mean of `e` | Burstiness: steady motion is low, stop-and-go motion is high |
| `env_maxmed` | max of `e` / median of `e` | Size of the largest burst against the typical level |
| `env_p90p10` | 90th percentile / 10th percentile of `e` | Contrast between busy and quiet moments, robust to single spikes |
| `env_peakf` | frequency of the largest peak in the spectrum of `e` (mean removed, Hann window) | How often bursts repeat, e.g. one per bend or per arm swing |
| `env_peakfrac` | that peak's share of the envelope spectrum | How regular the bursts are |
| `env_ac05`, `env_ac10`, `env_ac15`, `env_ac20` | autocorrelation of `e` at lags 0.5, 1.0, 1.5, 2.0 s | Whether the activity pattern repeats after that delay |
| `env_acmax` | largest autocorrelation for lags from 0.3 to 3.0 s | Strength of the clearest repetition |
| `env_aclag` | the lag (s) where that maximum occurs | Period of the repetition |

The envelope captures rhythm even when subcarriers move in opposite directions, which would cancel in a plain average.

### 6.4 Steadiness (2 features)

| Feature | Calculation | What it represents |
|---|---|---|
| `st_ratio` | split the window into 1 s blocks; average the block standard deviations per subcarrier; divide by the full-window `sd`; take the median over subcarriers | Near 1 = variation is fast (within a second); near 0 = variation is slow drift between seconds |
| `half_ratio` | per subcarrier, smaller of (first-half sd, second-half sd) / larger; median over subcarriers | Near 1 = motion is the same throughout the window; low = it started or stopped part-way |

### 6.5 Spread over subcarriers (2 features)

| Feature | Calculation | What it represents |
|---|---|---|
| `act_frac` | share of subcarriers whose `sd` is above half of the largest `sd` | Whether the motion disturbs the whole band (whole-body motion) or a few subcarriers |
| `top_share` | variance of the most active 10% of subcarriers / total variance | How concentrated the motion energy is |

### Why they help

Most of these features are ratios or shares, so they do not depend on how large a person's movements are. The existing level features (`sd_*`, `absdiff_*`) shift from person to person; the new ones describe the timing and pattern of the motion instead. On their own they score 82.6% at 4 s and 90.1% at 8 s on a new person; added to the 33 magnitude features they give +0.9 points at both lengths.

## Appendix — reference code

The study scripts lived in a temporary folder, so the essential code is kept here. `mag` is `|r|` for one window (T × 234), `FS = 10`.

```python
import numpy as np
import rtbfm

FS = 10.0
MAG_KEEP = np.array([not n.startswith("ph") for n in rtbfm.feature_names()])


def lvl_keep(mag, q=75):
    """Keep-mask: drop the top quarter of subcarriers by static level."""
    lvl = np.log(mag.mean(0))
    return lvl <= np.percentile(lvl, q)


def extra(m):
    """31 extra features from one masked psi window (T, S)."""
    T = m.shape[0]
    t = np.linspace(-1, 1, T)[:, None]
    c = m - m.mean(0)
    c = c - t * (c * t).sum(0) / (t * t).sum()
    sd = c.std(0)
    mv = sd > 1e-6
    if mv.sum() < 2:
        return np.zeros(31, np.float32)
    c, sd = c[:, mv], sd[mv]

    # pooled power spectrum, 0 Hz bin dropped, normalised to sum 1
    F = np.fft.rfft(c * np.hanning(T)[:, None], axis=0)
    P = (F.real ** 2 + F.imag ** 2).sum(1)[1:]
    f = np.fft.rfftfreq(T, 1 / FS)[1:]
    p = P / (P.sum() + 1e-12)
    cs = np.cumsum(p)
    out = [(f * p).sum(), f[p.argmax()], p.max(),
           -(p * np.log(p + 1e-12)).sum() / np.log(len(p)),
           f[np.searchsorted(cs, 0.5)],
           f[min(np.searchsorted(cs, 0.9), len(f) - 1)]]
    out += [p[(f >= lo) & (f < lo + 0.25)].sum() for lo in np.arange(0, 2.5, 0.25)]

    # motion envelope: mean relative |first difference|, 3-sample smoothing
    e = (np.abs(np.diff(c, axis=0)) / sd).mean(1)
    e = np.convolve(e, np.ones(3) / 3, mode="valid")
    em = e.mean() + 1e-12
    ec = e - e.mean()
    E = np.abs(np.fft.rfft(ec * np.hanning(len(ec)))) ** 2
    fe = np.fft.rfftfreq(len(ec), 1 / FS)
    E, fe = E[1:], fe[1:]
    ac = np.correlate(ec, ec, "full")[len(ec) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lag = lambda s: ac[min(int(s * FS), len(ac) - 1)]
    k = np.arange(3, min(31, len(ac)))
    out += [e.std() / em, e.max() / (np.median(e) + 1e-12),
            np.percentile(e, 90) / (np.percentile(e, 10) + 1e-12),
            fe[E.argmax()], E.max() / (E.sum() + 1e-12),
            lag(0.5), lag(1.0), lag(1.5), lag(2.0),
            ac[k].max(), k[ac[k].argmax()] / FS]

    # steadiness: 1 s block std share, and first-half vs second-half
    nb = T // 10
    blk = c[:nb * 10].reshape(nb, 10, -1)
    out.append(np.median(blk.std(1).mean(0) / sd))
    h1, h2 = c[:T // 2].std(0), c[T // 2:].std(0)
    out.append(np.median(np.minimum(h1, h2) / (np.maximum(h1, h2) + 1e-12)))

    # spread of the motion over subcarriers
    out.append((sd > 0.5 * sd.max()).mean())
    s2 = np.sort(sd ** 2)[::-1]
    out.append(s2[:len(s2) // 10].sum() / s2.sum())
    return np.asarray(out, np.float32)


def proposed_features(mag, phase):
    """64 features for one window: 33 magnitude (rtbfm on psi) + 31 extra."""
    keep = lvl_keep(mag)
    psi = np.arctan(mag)[:, keep]
    base = rtbfm.window_features(psi, phase[:, keep], FS)[MAG_KEEP]
    return np.concatenate([base, extra(psi)])
```

Feature order of the 31 extras: `spec_centroid, spec_peakf, spec_peakfrac, spec_entropy, spec_p50f, spec_p90f, fine0.00 … fine2.25, env_cv, env_maxmed, env_p90p10, env_peakf, env_peakfrac, env_ac05, env_ac10, env_ac15, env_ac20, env_acmax, env_aclag, st_ratio, half_ratio, act_frac, top_share`.

`proposed_features` still calls `rtbfm.window_features` and discards the phase columns, which is how the study ran. A production version should compute only the magnitude block so phase is never needed.
