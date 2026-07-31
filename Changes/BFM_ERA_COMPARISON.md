# BFM data comparison across three collection eras

Branch: `fix/bfm-codebook-bit-width`
Measured: 2026-07-31
Status: **diagnostic record of the problem, measured with the UNFIXED decoder.**
These numbers are the "before" baseline. Re-run after the codebook fix to get "after".

---

## TL;DR

Three eras of BFM data were compared. Two independent defects were found, and they
entered at different times:

| Defect | Oct 2025 | Jul 04 2026 | Jul 31 2026 |
|---|---|---|---|
| Duplicate packets | clean | **48% duplicated** | clean (fixed) |
| Codebook decoded as (9,7) when data is (6,4) | correct — data really is (9,7) | **wrong** | **wrong** |

Net effect: standing-vs-walking separability collapsed from Cohen's d = **−1.606**
(Oct 2025, phase) to **−0.062** (Jul 04), partially recovering to **+0.101** (Jul 31)
once the duplicates were removed. The remaining gap is the codebook bug, which
`fix/bfm-codebook-bit-width` addresses but which has **not yet been validated against
freshly collected data**.

---

## 1. Datasets compared

| Key | File | Format |
|---|---|---|
| Oct25 standing | `bfm_processed_csv/bfm_data_standing_abel_alt_nofoil_20251010_142840.csv` | Ratio_Real/Imag |
| Oct25 walking | `bfm_processed_csv/bfm_data_walking_abel_alt_nofoil_20251010_143031.csv` | Ratio_Real/Imag |
| Jul04 standing | `bfm_mag_phase_csv/bfm_data_0407Test6ST_20260704_153830.csv` | `_Mag`/`_Phase` |
| Jul04 walking | `bfm_mag_phase_csv/bfm_data_0407Test6WT_20260704_154114.csv` | `_Mag`/`_Phase` |
| Jul31 standing | `bfm_mag_phase_csv/bfm_data_0731_Test4_Standing_20260731_150533.csv` | `_Mag`/`_Phase` |
| Jul31 walking | `bfm_mag_phase_csv/bfm_data_0731_Test4_Walking_20260731_150816.csv` | `_Mag`/`_Phase` |

All six carry the same 234 subcarriers, SCIDX −122..122. Oct25 files are real/imag and
were converted with `bfmtool.utils.append_mag_phase`; the Jul files were already
mag/phase (written by the pre-fix converter, hence the `_Mag`/`_Phase` names) and were
only renamed to `_Ratio_Mag`/`_Ratio_Phase` for comparison.

---

## 2. Capture integrity and codebook

| Key | rows raw | rows used | unique ts | dup rows | span (s) | rate (pkt/s) | φ11 idx range (distinct) | ψ21 idx range (distinct) | codebook |
|---|---|---|---|---|---|---|---|---|---|
| Oct25 standing | 951 | 935 | 935 | 0.0% | 90.3 | 10.4 | 19–343 (197) | 1–103 (101) | **(9,7)** |
| Oct25 walking | 938 | 928 | 928 | 0.0% | 90.3 | 10.3 | 20–331 (184) | 2–100 (88) | **(9,7)** |
| Jul04 standing | 1156 | 1156 | 599 | **48.2%** | 119.7 | 9.7 | 0–63 (64) | 0–15 (16) | **(6,4)** |
| Jul04 walking | 1148 | 1148 | 589 | **48.7%** | 120.0 | 9.6 | 0–63 (57) | 0–15 (16) | **(6,4)** |
| Jul31 standing | 1181 | 1181 | 1181 | 0.0% | 120.5 | 9.8 | 0–63 (25) | 2–13 (12) | **(6,4)** |
| Jul31 walking | 1180 | 1180 | 1180 | 0.0% | 120.5 | 9.8 | 0–63 (25) | 2–13 (12) | **(6,4)** |

"rows used" is after `filter_by_mode` on `(transmitter_address, receiver_address)`.
Only the Oct25 files still have MAC columns, so only they could be filtered — the
Jul files lost those columns to the pre-fix converter.

Packet rate is consistent (~10/s) across all three eras, so rate is **not** a
confounder. Session duration differs (90s vs 120s) — see instruction 6.

---

## 3. Value distributions

Magnitude = |h1/h2| = cot(ψ21). Phase = arg(h1/h2) = φ11 (radians).

| Key | mag mean | mag median | mag std | mag max | phase mean | phase min | phase max |
|---|---|---|---|---|---|---|---|
| Oct25 standing | 10.996 | 9.552 | 4.280 | 54.3 | 0.4418 | 0.2393 | 4.2154 |
| Oct25 walking | 10.838 | 9.552 | 3.831 | 32.6 | 0.4252 | 0.2516 | 4.0681 |
| Jul04 standing | 18.094 | 9.552 | 23.754 | 163.0 | 0.3977 | 0.0061 | 0.7793 |
| Jul04 walking | 18.089 | 9.552 | 23.748 | 163.0 | 0.3975 | 0.0061 | 0.7793 |
| Jul31 standing | 11.861 | 10.834 | 4.557 | 32.6 | 0.3721 | 0.0061 | 0.7793 |
| Jul31 walking | 11.869 | 10.834 | 4.537 | 32.6 | 0.3781 | 0.0061 | 0.7793 |

The Jul phase ceiling of exactly **0.7793 rad** is the signature of the bug: it is
`pi * (1/512 + 63/256)`, i.e. the largest angle a 6-bit index can produce when decoded
with the 9-bit formula. The true range should be ~`[0.049, 6.234]`.

---

## 4. Signal dynamics and separability

`std_t` = standard deviation over time per subcarrier, averaged across the 234
subcarriers. This is the quantity that actually discriminates activities.

| Key | mag std_t | phase std_t | distinct mag values | distinct phase values |
|---|---|---|---|---|
| Oct25 standing | 1.8120 | **0.3584** | 296 | 614 |
| Oct25 walking | 1.3376 | **0.2852** | 255 | 534 |
| Jul04 standing | 2.1887 | 0.0211 | 47 | 119 |
| Jul04 walking | 1.8304 | 0.0179 | 44 | 97 |
| Jul31 standing | 0.3750 | 0.0169 | 31 | 44 |
| Jul31 walking | 0.5295 | 0.0225 | 31 | 44 |

Separability of standing vs walking, Cohen's d on per-subcarrier temporal std
(positive = walking varies more, which is the physically expected direction):

| Era | d (magnitude) | d (phase) | walk/stand ratio, mag | walk/stand ratio, phase |
|---|---|---|---|---|
| Oct25 | −0.540 | **−1.606** | 0.738 | 0.796 |
| Jul04 | −0.058 | −0.062 | 0.836 | 0.849 |
| Jul31 | **+0.315** | **+0.101** | 1.412 | 1.328 |

### Reading these

- **Oct 2025** has by far the strongest separation (|d| = 1.606 on phase), but the
  **sign is negative** — standing varied *more* than walking. That is physically
  backwards and is an unresolved question, not a validated baseline. Do not treat
  Oct 2025 as "correct" without explaining the sign. Possible causes: label swap,
  subject/environment difference, or a genuine effect of the setup.
- **Jul 04** is effectively unusable: d ≈ −0.06 on both channels means standing and
  walking are statistically indistinguishable. The 48% duplicate rows suppress
  apparent temporal variation, and the codebook bug removes 8× of the dynamic range.
- **Jul 31** flips to the physically sensible positive sign and gets ~14× more
  separation than Jul 04, but absolute dynamics remain ~15× below Oct 2025
  (phase std_t 0.017–0.023 vs 0.285–0.358). Removing duplicates recovered real
  signal; the codebook bug is the remaining ceiling.

---

## 5. How to compare these correctly — read before re-running

These are the traps that produced wrong intermediate conclusions during this
investigation. Each one cost a wrong answer at least once.

**1. Do not use `convert_real_imag_to_mag_phase` from the GUI files for analysis.**
Use `bfmtool.utils.append_mag_phase`. Before commit `29439c8` the GUI copy emitted
`_Mag`/`_Phase` instead of `_Ratio_Mag`/`_Ratio_Phase` and dropped the four MAC
columns. The arithmetic was identical, so it produces plausible-looking but
unfilterable output.

**2. Normalise column names before comparing eras.**
Oct25 files are real/imag, Jul files are already polar with the old `_Mag`/`_Phase`
names. Rename Jul columns to `_Ratio_Mag`/`_Ratio_Phase`, or every selector matching
`endswith("Ratio_Mag")` silently returns an empty feature list — no exception, just
zero features.

**3. Apply `filter_by_mode` before computing anything.**
Monitor-mode captures pick up several beamformers. The Oct25 files contain a
16-packet and an 11-packet minority link mixed into the target link.
`bfm_data_0407Test4_20260704_141925.csv` contains **five** transmitters using **two
different codebooks**. Pooling them corrupts both statistics and codebook inference.
Jul-era mag/phase files cannot be filtered at all (MAC columns lost) — a known
limitation of that data, not something to work around.

**4. Deduplicate the Jul 04 files, or exclude them.**
They contain each packet ~twice (1156 rows / 599 unique timestamps). Duplicates
depress temporal variance and inflate row counts. Drop on `timestamp` +
subcarrier values. Jul 31 and Oct 25 need no deduplication.

**5. Compare `std` over time, not pooled `std`.**
Pooled std over all rows × subcarriers mixes across-subcarrier spread (large, static)
with over-time variation (small, the actual activity signal). Compute std along
axis 0 per subcarrier, then average. Pooled std says Jul 04 is the most variable
dataset; temporal std correctly says it is the least informative.

**6. Do not compare absolute magnitudes across eras while the codebook bug is live.**
Oct 25 decoded correctly; Jul data did not. The Jul magnitudes (mean 18.1, max 163)
are an artifact of `cot(ψ)` being evaluated at wrongly-tiny angles. Only compare
*within* an era, or *after* re-decoding everything with the fix.

**7. Round before thresholding recovered indices.**
Recovering `q = (phase/pi - 1/512) * 256` from stored floats gives e.g.
`63.000000000000014`, so a bare `q > 63` test misclassifies a (6,4) capture as (9,7).
This happened during this investigation. Use `np.round(q)` or a tolerance. The
shipped `infer_codebook()` is not affected — it reads integer indices from tshark.

**8. Expect phase wrapping above π.**
With a 6-bit φ the true angle spans nearly 2π, but real/imag storage makes
`arctan2` wrap to `(−π, π]`. Recovered indices therefore cap near 255, not 511.
This is inherent to the representation, not a defect — `e^{jφ}` is unchanged.

**9. Session durations differ (90s vs 120s).**
Packet rate is ~10/s in all eras so this mostly cancels, but if you compute anything
cumulative rather than per-packet, truncate to a common duration first.

**10. Do not read the two output directories as matched pairs.**
`bfm_real_imag_csv/` and `bfm_mag_phase_csv/` contain **disjoint** sessions written by
different code versions. The four Jul 04 mag/phase files carry a `pc_timestamp`
column that only the pre-restructure streaming path ever wrote.

---

## 6. Reproduction

Default `python3` on the dev Mac has no pandas. Use:
`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`

```python
import pandas as pd, numpy as np, re, sys, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from bfmtool.utils import get_bfm_columns, append_mag_phase, filter_by_mode

def load_real_imag(f):                      # Oct 2025 style
    df = pd.read_csv(f)
    df = filter_by_mode(df, mode_cols=['transmitter_address', 'receiver_address'])
    cd = get_bfm_columns(df)
    return append_mag_phase(df).drop(columns=cd['real_col'] + cd['imag_col'])

def load_mag_phase(f):                      # Jul 2026 style (pre-fix names)
    df = pd.read_csv(f)
    return df.rename(columns={c: c.replace("_Mag", "_Ratio_Mag")
                               .replace("_Phase", "_Ratio_Phase")
                              for c in df.columns if c.startswith("SCIDX")})

S = {
 "Oct25 standing": load_real_imag("bfm_processed_csv/bfm_data_standing_abel_alt_nofoil_20251010_142840.csv"),
 "Oct25 walking":  load_real_imag("bfm_processed_csv/bfm_data_walking_abel_alt_nofoil_20251010_143031.csv"),
 "Jul04 standing": load_mag_phase("bfm_mag_phase_csv/bfm_data_0407Test6ST_20260704_153830.csv"),
 "Jul04 walking":  load_mag_phase("bfm_mag_phase_csv/bfm_data_0407Test6WT_20260704_154114.csv"),
 "Jul31 standing": load_mag_phase("bfm_mag_phase_csv/bfm_data_0731_Test4_Standing_20260731_150533.csv"),
 "Jul31 walking":  load_mag_phase("bfm_mag_phase_csv/bfm_data_0731_Test4_Walking_20260731_150816.csv"),
}
sc = sorted(int(re.search(r'SCIDX_(-?\d+)_', c).group(1))
            for c in S["Oct25 standing"].columns if c.endswith("Ratio_Mag"))
MC = [f"SCIDX_{i}_Ratio_Mag" for i in sc]
PC = [f"SCIDX_{i}_Ratio_Phase" for i in sc]

# codebook (note the rounding — see instruction 7)
for k, df in S.items():
    m = df[MC].to_numpy(float); p = df[PC].to_numpy(float)
    qphi = np.round((p/np.pi - 1/512) * 256)
    qpsi = np.round((np.arctan(1/m)/np.pi - 1/512) * 256)
    cb = "(9,7)" if (np.nanmax(qphi) > 63 or np.nanmax(qpsi) > 15) else "(6,4)"
    print(f"{k:<16} phi<={np.nanmax(qphi):>4.0f} psi<={np.nanmax(qpsi):>4.0f}  {cb}")

# separability — temporal std, NOT pooled std (see instruction 5)
for era in ["Oct25", "Jul04", "Jul31"]:
    for name, cols in (("MAG", MC), ("PHASE", PC)):
        a = np.nanstd(S[f"{era} standing"][cols].to_numpy(float), axis=0)
        b = np.nanstd(S[f"{era} walking"][cols].to_numpy(float), axis=0)
        d = (np.nanmean(b) - np.nanmean(a)) / np.sqrt((np.nanvar(a) + np.nanvar(b))/2)
        print(f"{era} {name:<6} Cohen's d = {d:+.3f}")
```

---

## 7. Open items

1. **No raw angle CSVs exist for the Jul 31 sessions.** `bfm_raw_csv/` and
   `bfm_real_imag_csv/` contain no `0731` files, so Jul 31 cannot be re-decoded with
   the codebook fix — the mag/phase CSV is lossy. **Re-collect, or locate the pcaps,
   before trying to validate the fix on Jul 31 data.**
2. **Confirm the codebook from the frame, not by inference.** `infer_codebook()` reads
   the field width from the observed index range. The authoritative source is the
   VHT/HT MIMO Control field (Codebook Information / Nb). tshark is not installed on
   the dev Mac, so the parsing regex could not be written or verified.
3. **Explain the negative Cohen's d in Oct 2025.** Standing varying more than walking
   is backwards. Until this is understood, Oct 2025 is a high-contrast reference, not
   a correctness baseline.
4. **Post-fix validation target.** After re-extracting with the codebook fix, phase
   std_t should move from ~0.02 toward the ~0.29–0.36 range, and |Cohen's d| on phase
   should grow well beyond the +0.101 seen on Jul 31.
5. `main_gui2.py` still carries the old broken `convert_real_imag_to_mag_phase`
   (deliberately left out of `29439c8`).
