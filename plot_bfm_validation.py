"""
BFM dataset validation plots.
Compares Walking vs Standing to confirm the two classes are visually separable.

Usage:
    .venv_plot/Scripts/python plot_bfm_validation.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # no display needed — saves to PNG
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
WALKING_CSV  = "bfm_processed_csv/bfm_data_Walking_Test1_20260604_144555.csv"
STANDING_CSV = "bfm_processed_csv/bfm_data_Standing_Test1_20260604_144857.csv"
OUT_DIR      = Path("bfm_plots")
OUT_DIR.mkdir(exist_ok=True)

COLORS = {"Walking": "#2196F3", "Standing": "#FF5722"}

# Must match MAG_CLIP in main_gui2.py
MAG_CLIP = 32.585

# Subcarrier to use for single-carrier plots (middle of the band)
FOCUS_SCIDX = 0


# ── Helpers ──────────────────────────────────────────────────────────────────
def load(path, label):
    df = pd.read_csv(path)
    df["label"] = label
    # Apply magnitude clip to match the live pipeline's MAG_CLIP constant
    mc = [c for c in df.columns if c.endswith("_Mag")]
    df[mc] = df[mc].clip(upper=MAG_CLIP)
    # Normalise pc_timestamp to seconds-since-session-start
    df["t"] = df["pc_timestamp"] - df["pc_timestamp"].iloc[0]
    return df


def mag_cols(df):
    return [c for c in df.columns if c.endswith("_Mag")]


def phase_cols(df):
    return [c for c in df.columns if c.endswith("_Phase")]


def subcarrier_indices(df):
    import re
    return sorted(
        int(re.search(r"SCIDX_(-?\d+)_Mag", c).group(1))
        for c in mag_cols(df)
    )


def focus_col(df, scidx, kind="Mag"):
    col = f"SCIDX_{scidx}_{kind}"
    if col in df.columns:
        return col
    # fall back to the numerically closest available subcarrier
    avail = subcarrier_indices(df)
    closest = min(avail, key=lambda x: abs(x - scidx))
    return f"SCIDX_{closest}_{kind}"


def rolling_var(series, window_pts):
    return series.rolling(window=window_pts, center=True, min_periods=1).var()


def psd(series, fs):
    arr = series.dropna().values
    f, pxx = welch(arr, fs=fs, nperseg=min(256, len(arr) // 4))
    return f, pxx


# ── Load data ────────────────────────────────────────────────────────────────
print("Loading CSVs…")
w = load(WALKING_CSV,  "Walking")
s = load(STANDING_CSV, "Standing")

# Estimate sample rate (packets per second)
fs_w = len(w) / w["t"].iloc[-1] if w["t"].iloc[-1] > 0 else 10.0
fs_s = len(s) / s["t"].iloc[-1] if s["t"].iloc[-1] > 0 else 10.0
print(f"Walking:  {len(w)} rows, {w['t'].iloc[-1]:.1f}s, ~{fs_w:.1f} pkt/s")
print(f"Standing: {len(s)} rows, {s['t'].iloc[-1]:.1f}s, ~{fs_s:.1f} pkt/s")

mag_w  = w[mag_cols(w)].values    # (N_w, 234)
mag_s  = s[mag_cols(s)].values    # (N_s, 234)

sc_idx = subcarrier_indices(w)
focus  = focus_col(w, FOCUS_SCIDX, "Mag")
focusp = focus_col(w, FOCUS_SCIDX, "Phase")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Magnitude heatmap (subcarrier × time)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 1: Magnitude heatmaps…")

# Downsample in time for display (max 1000 time steps per heatmap)
MAX_T = 1000
step_w = max(1, len(w) // MAX_T)
step_s = max(1, len(s) // MAX_T)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("BFM Magnitude Heatmap (subcarrier × time)", fontsize=13)

for ax, data, t_vec, label, step in [
    (axes[0], mag_w,  w["t"].values,  "Walking",  step_w),
    (axes[1], mag_s,  s["t"].values,  "Standing", step_s),
]:
    Z = data[::step].T          # shape: (n_subcarriers, n_timesteps)
    im = ax.imshow(
        Z, aspect="auto", origin="lower",
        extent=[t_vec[0], t_vec[-1], sc_idx[0], sc_idx[-1]],
        vmin=np.percentile(Z, 2), vmax=np.percentile(Z, 98),
        cmap="viridis",
    )
    ax.set_title(label, color=COLORS[label], fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Subcarrier index")
    fig.colorbar(im, ax=ax, label="|H| magnitude")

plt.tight_layout()
out = OUT_DIR / "01_magnitude_heatmap.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Mean magnitude & rolling variance over time
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 2: Mean magnitude & variance over time…")

WIN_SEC = 1.0   # rolling window width in seconds

fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharey="row")
fig.suptitle("Mean Magnitude & Temporal Variance per Session", fontsize=13)

for col_idx, (df, label, fs) in enumerate([
    (w, "Walking",  fs_w),
    (s, "Standing", fs_s),
]):
    win_pts = max(3, int(WIN_SEC * fs))
    mean_mag = pd.Series(df[mag_cols(df)].mean(axis=1).values)
    var_mag  = rolling_var(mean_mag, win_pts)

    t = df["t"].values
    c = COLORS[label]

    axes[0, col_idx].plot(t, mean_mag, color=c, lw=0.6, alpha=0.8)
    axes[0, col_idx].set_title(f"{label} — mean |H|", color=c, fontweight="bold")
    axes[0, col_idx].set_xlabel("Time (s)")
    axes[0, col_idx].set_ylabel("Mean magnitude")

    axes[1, col_idx].plot(t, var_mag, color=c, lw=0.6, alpha=0.8)
    axes[1, col_idx].set_title(f"{label} — rolling variance ({WIN_SEC}s window)", color=c, fontweight="bold")
    axes[1, col_idx].set_xlabel("Time (s)")
    axes[1, col_idx].set_ylabel("Variance")

plt.tight_layout()
out = OUT_DIR / "02_mean_magnitude_variance.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Focus-subcarrier magnitude & phase time series (overlay)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"Plot 3: Focus subcarrier {FOCUS_SCIDX} time series…")

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
fig.suptitle(f"Subcarrier {FOCUS_SCIDX}: Magnitude & Phase over Time", fontsize=13)

for ax_row, kind, focus_fn in [
    (0, "Magnitude", lambda df: df[focus_col(df, FOCUS_SCIDX, 'Mag')]),
    (1, "Phase",     lambda df: df[focus_col(df, FOCUS_SCIDX, 'Phase')]),
]:
    ax = axes[ax_row]
    for df, label, fs in [(w, "Walking", fs_w), (s, "Standing", fs_s)]:
        sig = focus_fn(df).values
        t   = df["t"].values
        ax.plot(t, sig, color=COLORS[label], lw=0.7, alpha=0.8,
                label=f"{label} ({len(df)} pkt)")
    ax.set_title(kind)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(kind)
    ax.legend()

plt.tight_layout()
out = OUT_DIR / "03_focus_subcarrier_timeseries.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Power spectral density (Welch) of mean magnitude
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 4: Power spectral density…")

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("PSD of Mean Magnitude (Walking vs Standing)", fontsize=13)

for df, label, fs in [(w, "Walking", fs_w), (s, "Standing", fs_s)]:
    mean_mag = df[mag_cols(df)].mean(axis=1)
    f, pxx   = psd(mean_mag, fs)
    # Only plot up to 5 Hz (activity range)
    mask = f <= 5.0
    ax.semilogy(f[mask], pxx[mask], color=COLORS[label], lw=1.5, label=label)

ax.axvline(1.5, color="gray", ls="--", lw=0.8, label="~1.5 Hz (typical step)")
ax.axvline(0.3, color="gray", ls=":",  lw=0.8, label="~0.3 Hz (breathing)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power spectral density")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
out = OUT_DIR / "04_psd.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5 — Per-subcarrier mean magnitude (static snapshot, both classes)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 5: Per-subcarrier mean magnitude profile…")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("Per-subcarrier Mean ± Std Magnitude", fontsize=13)

for ax, df, label in [(axes[0], w, "Walking"), (axes[1], s, "Standing")]:
    mc   = mag_cols(df)
    mean = df[mc].mean(axis=0).values
    std  = df[mc].std(axis=0).values
    c    = COLORS[label]
    ax.fill_between(sc_idx, mean - std, mean + std, alpha=0.25, color=c)
    ax.plot(sc_idx, mean, color=c, lw=1.2, label=label)
    ax.set_title(label, color=c, fontweight="bold")
    ax.set_ylabel("|H| magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[1].set_xlabel("Subcarrier index")
plt.tight_layout()
out = OUT_DIR / "05_per_subcarrier_profile.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 6 — Summary stats box: variance distribution across all subcarriers
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 6: Variance distribution across subcarriers (boxplot)…")

var_w_per_sc = w[mag_cols(w)].var(axis=0).values
var_s_per_sc = s[mag_cols(s)].var(axis=0).values

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Temporal Variance per Subcarrier — Walking vs Standing", fontsize=13)

bp = ax.boxplot(
    [var_w_per_sc, var_s_per_sc],
    labels=["Walking", "Standing"],
    patch_artist=True,
    medianprops=dict(color="white", lw=2),
)
for patch, color in zip(bp["boxes"], [COLORS["Walking"], COLORS["Standing"]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_ylabel("Magnitude variance per subcarrier")
ax.grid(True, axis="y", alpha=0.3)

# Annotate medians
for i, var in enumerate([var_w_per_sc, var_s_per_sc], 1):
    ax.text(i, np.median(var), f"  median={np.median(var):.3f}", va="center", fontsize=9)

plt.tight_layout()
out = OUT_DIR / "06_variance_boxplot.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  -> {out}")


print(f"\nAll plots saved to {OUT_DIR.resolve()}")
