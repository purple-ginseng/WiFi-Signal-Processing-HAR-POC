# paperVisualization.ipynb - Issues & Critiques Log

## Status: POST-FIX (all code issues resolved)

---

## A. RUNTIME BUGS

| ID | Cell | Severity | Description | Status | Fix |
|----|------|----------|-------------|--------|-----|
| A1 | 28 | CRASH | `matrix_combined.shape` references undefined variable | FIXED | Changed to `matrix_combined_mag.shape` and `matrix_combined_phase.shape` |
| A2 | 62 | DATA CORRUPTION | `df = df[df.environment == 'foil']` overwrites global `df` | FIXED | Renamed to `df_foil`; updated cells 63-66 to use `df_foil` and `df_butter_filtered` |
| A3 | 72-73 | CRASH | References `df_processed` (undefined) | FIXED | Changed to `df_processed1` |
| A4 | 30 | MISLABEL | `ylabel='Magnitude'` but data is phase | FIXED | Changed to `ylabel='Phase (rad)'` |
| A5 | 28 | NO-OP | `df = df` self-assignment | FIXED | Removed; cell now uses existing `df` directly |

## B. HARDCODED PATHS

| ID | Cell | Path | Status | Fix |
|----|------|------|--------|-----|
| B1 | 23 | Windows absolute path to raw CSVs | FIXED | `"bfm_raw_csv/bfm_data_standing_abel_alt_nofoil_20251010_143551.csv"` |
| B2 | 23 | Windows absolute path to raw CSVs | FIXED | `"bfm_raw_csv/bfm_data_walking_abel_alt_nofoil_20251010_143736.csv"` |
| B3 | 24 | Same Windows paths | FIXED | Same relative paths as B1/B2 |
| B4 | 25 | Same Windows paths | FIXED | Same relative paths as B1/B2 |
| B5 | 28 | Windows absolute path to combined CSV | FIXED | Removed `FILE_PATH`; uses `df` from cell 1 |
| B6 | 1 | Windows backslash in relative path | FIXED (prior session) | `'Cleaned and combined/bfm_data.csv'` |

## C. ALGORITHMIC INCONSISTENCIES

| ID | Cell | Description | Status | Fix |
|----|------|-------------|--------|-----|
| C1 | 17 | `np.diff(x, n=3)` — 3rd order confirmed correct | FIXED | Docstring updated to match; `DIFF_ORDER = 3` constant added |
| C2 | 67 | `np.diff(x_clean, n=4)` — was 4th order | FIXED | Changed to `n=DIFF_ORDER_LOCAL` (= 3) |
| C3 | 17 vs 37 | MAD scaling inconsistent (raw vs scipy `scale='normal'`) | FIXED | Both now use `median_abs_deviation(scale='normal')` via `calculate_rolling_mad` |
| C4 | 17 vs 67 | Two different `create_r_features` implementations | FIXED | Cell 67 renamed to `create_r_features_loop`; both use identical window params |
| C5 | 17 | Window params were W_SPREAD=500, W_ACCEL=30 | FIXED | Now `W_SPREAD=100`, `W_LOCATION=100`, `W_FEATURE=53` |
| C6 | 67 | Window params were 500/500/30 | FIXED | Now `WINDOW_SIZE_SPREAD=100`, `WINDOW_SIZE_LOCATION=100`, `WINDOW_SIZE_FEATURE=53` |
| C7 | 37, 42 | `process_pipeline_steps` used old params | FIXED | Now references `W_SPREAD`, `W_LOCATION`, `IQR_F` from cell 17; cell 42 deduped |

## D. IEEE FORMATTING ISSUES

| ID | Cell(s) | Description | Status | Fix |
|----|---------|-------------|--------|-----|
| D1 | 26-27, 29, 34, 36, 38-39, 41, 43-44, 46 | `ax.set_title()` used | FIXED | Removed from all publication figure cells |
| D2 | multiple | rcParams re-set inconsistently | NOTED | Not changed (would alter workflow); documented for awareness |
| D3 | 30 | ylabel said Magnitude for phase data | FIXED | (same as A4) |
| D4 | 25, 27 | Phase axis labels inconsistent | FIXED | Cell 25: `Phase (rad)`, Cell 27: left as `Phase` (unitless ratio phase) |

## E. CODE QUALITY

| ID | Cell(s) | Description | Status | Fix |
|----|---------|-------------|--------|-----|
| E1 | multiple | Redundant imports across 10+ cells | NOTED | Not removed (notebook execution order independence); documented |
| E2 | 24 & 25 | `decompress_bfm_target_sc` duplicated | NOTED | Cell 25 now reuses definition from cell 24 (same kernel session) |
| E3 | 37 & 42 | `process_pipeline_steps` duplicated | FIXED | Cell 42 now reuses function from cell 37 |
| E4 | 20, 52, 77 | Empty cells | NOTED | Left as-is (notebook structure) |
| E5 | various | Debug/scratch cells | NOTED | Left as-is (exploratory notebook, not final paper code) |
| E6 | 5 | Session exclusion undocumented | FIXED | Added comment: "identified as corrupted/anomalous during data collection" |

## F. IEEE REVIEWER CRITIQUES (for paper preparation)

| ID | Severity | Critique | Addressable? |
|----|----------|----------|--------------|
| F1 | MAJOR | Diff order inconsistency | RESOLVED — standardized to 3rd order throughout |
| F2 | MAJOR | Single-session visualizations may not represent full dataset | OPEN — add multi-session or statistical summary figures |
| F3 | MAJOR | SNR threshold=5 not justified | OPEN — add histogram analysis, report % packets removed per environment |
| F4 | MAJOR | Only 2 activities — limited scope | OPEN — discuss in paper limitations section |
| F5 | MINOR | PCA explained variance never reported | OPEN — add `pca.explained_variance_ratio_` print/plot |
| F6 | MINOR | Figure titles instead of captions | RESOLVED — titles removed from all publication figures |
| F7 | MINOR | Session exclusion undocumented | RESOLVED — comment added in cell 5 |
| F8 | MINOR | Window params need sensitivity analysis | OPEN — consider ablation study |
| F9 | MINOR | No classification results in notebook | OPEN — separate notebook or section needed |
| F10 | MINOR | Two transmitter MACs unexplored | OPEN — potential future work |

---

## Summary of Changes Made

**Files modified:** `paperVisualization.ipynb`
**Files created:** `Changes/ISSUES_LOG.md`

### Fixes applied (15 total):
- 5 runtime bugs fixed (A1-A5)
- 6 hardcoded paths converted to relative (B1-B6)
- 7 algorithmic inconsistencies resolved (C1-C7)
- 4 IEEE formatting issues fixed (D1, D3, D4, E6)
- 1 code deduplication (E3)

### Parameters standardized:
- `DIFF_ORDER = 3` (3rd-order difference) — consistent across all cells
- `W_SPREAD = 100` (IQR winsorization and MAD lookback)
- `W_LOCATION = 100` (rolling median lookback)
- `W_FEATURE = 53` (~5.3s at 10 Hz WiFi sample rate)
- `MAD` calculation: `scipy.stats.median_abs_deviation(scale='normal')` — consistent everywhere

### New: Comprehensive Pipeline Figure (Cell 48)
- **10-panel figure** (5 rows × 2 columns): Magnitude path | Phase path
- Traces the **same subject data** (Abel, nofoil) through all 5 pipeline stages
- Row 0: Raw Compressed BFM (φ₁₁, ψ₂₁)
- Row 1: BFM Ratio (decompressed + channel ratio)
- Row 2: PCA Aggregation (234 subcarriers → 1 component)
- Row 3: Robust Z-Score (winsorization + standardization)
- Row 4: 3rd-Order LMAD (final discriminative feature, highlighted)
- **IEEE Sensors Journal compliant**: 7.16" width, Times New Roman, 8pt, 600 DPI
- **Colorblind-safe palette**: Tol Bright (#4477AA blue, #EE6677 rose)
- Saves to `plots/fig_pipeline_overview.pdf` and `.png`

### Items noted but not modified (preserving workflow):
- Redundant imports (E1) — standard for notebook cell independence
- Empty/debug cells (E4, E5) — exploratory notebook, not final paper code
- rcParams inconsistency (D2) — would alter visual output
