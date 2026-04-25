import glob
import os
import re
import time
from functools import lru_cache
from typing import Optional

# Configure matplotlib BEFORE importing pyplot for thread safety
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from scipy.signal import spectrogram, detrend
from scipy.ndimage import gaussian_filter, uniform_filter1d
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Disable matplotlib warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

DATA_DIR = "./bfm_processed_csv"
SUBCARRIER_PATTERN = re.compile(r"SCIDX_(-?\d+)_Ratio_(Real|Imag)", re.IGNORECASE)
CMAPS = ["turbo", "viridis", "jet", "plasma", "inferno", "magma", "cividis", "RdYlBu_r", "twilight", "hsv"]


def request_rerun() -> None:
    """Trigger a Streamlit rerun with backward-compatible fallbacks."""
    rerun_fn = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if rerun_fn is not None:
        rerun_fn()
        return
    raise RuntimeError("Streamlit does not expose a rerun mechanism in this version.")


def get_available_files(directory: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    return files


@st.cache_data(show_spinner=False)
def load_bfm_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure timestamp is numeric for temporal plots
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df


def collect_subcarrier_columns(columns: pd.Index) -> tuple[list[int], dict[int, dict[str, str]]]:
    mapping: dict[int, dict[str, str]] = {}
    for col in columns:
        match = SUBCARRIER_PATTERN.match(col)
        if not match:
            continue
        sc_idx = int(match.group(1))
        comp = match.group(2).lower()
        mapping.setdefault(sc_idx, {})[comp] = col
    filtered = {idx: comps for idx, comps in mapping.items() if "real" in comps and "imag" in comps}
    ordered = sorted(filtered.keys())
    return ordered, filtered


@lru_cache(maxsize=16)
def build_complex_matrix(cache_key: tuple[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Cache heavy conversion using file path and an optional downsample step."""
    path, downsample = cache_key
    df = load_bfm_dataframe(path)
    subcarrier_ids, mapping = collect_subcarrier_columns(df.columns)
    if not subcarrier_ids:
        raise ValueError("No SCIDX_* Ratio columns detected in the selected file.")

    real_cols = [mapping[idx]["real"] for idx in subcarrier_ids]
    imag_cols = [mapping[idx]["imag"] for idx in subcarrier_ids]

    df_real = df[real_cols].apply(pd.to_numeric, errors="coerce")
    df_imag = df[imag_cols].apply(pd.to_numeric, errors="coerce")

    # Fill NaN values with 0 to prevent downstream issues
    df_real = df_real.fillna(0.0)
    df_imag = df_imag.fillna(0.0)

    if downsample > 1:
        df_real = df_real.iloc[::downsample].reset_index(drop=True)
        df_imag = df_imag.iloc[::downsample].reset_index(drop=True)
        timestamps = (
            df["timestamp"].iloc[::downsample].reset_index(drop=True) if "timestamp" in df.columns else None
        )
    else:
        timestamps = df["timestamp"] if "timestamp" in df.columns else None

    complex_matrix = df_real.to_numpy(dtype=np.float32) + 1j * df_imag.to_numpy(dtype=np.float32)

    # Final check: replace any inf values that might have been introduced
    complex_matrix[~np.isfinite(complex_matrix)] = 0

    if timestamps is not None:
        timestamps = timestamps.to_numpy(dtype=np.float64)

    return complex_matrix, timestamps


def compute_summary_stats(complex_matrix: np.ndarray) -> dict[str, float]:
    magnitudes = np.abs(complex_matrix)
    mean_magnitude = float(np.nanmean(magnitudes))
    max_magnitude = float(np.nanmax(magnitudes))
    phase_samples = np.angle(complex_matrix)
    coherence = np.exp(1j * phase_samples)
    circular_strength = np.nanmean(np.abs(np.nanmean(coherence, axis=0)))
    circular_variance = float(1 - circular_strength)
    return {
        "mean_magnitude": mean_magnitude,
        "max_magnitude": max_magnitude,
        "circular_variance": circular_variance,
    }


def plot_temporal_traces(
    time_axis: np.ndarray,
    complex_samples: np.ndarray,
    column_indices: list[int],
    labels: list[int],
    highlight_time: Optional[float] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    magnitudes = np.abs(complex_samples)
    for pos, label in zip(column_indices, labels):
        ax.plot(time_axis, magnitudes[:, pos], label=f"SC {label}")
    ax.set_title("Temporal Magnitude Traces")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|H|")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize="small", loc="upper right")
    if highlight_time is not None:
        ax.axvline(highlight_time, color="#ff4d4f", linestyle="--", linewidth=1.2, alpha=0.8)
    return fig


def plot_phase_traces(
    time_axis: np.ndarray,
    complex_samples: np.ndarray,
    column_indices: list[int],
    labels: list[int],
    highlight_time: Optional[float] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    phases = np.unwrap(np.angle(complex_samples), axis=0)
    for pos, label in zip(column_indices, labels):
        ax.plot(time_axis, phases[:, pos], label=f"SC {label}")
    ax.set_title("Temporal Phase Traces")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize="small", loc="upper right")
    if highlight_time is not None:
        ax.axvline(highlight_time, color="#ff4d4f", linestyle="--", linewidth=1.2, alpha=0.8)
    return fig


def plot_pca_traces(
    time_axis: np.ndarray,
    complex_samples: np.ndarray,
    highlight_time: Optional[float] = None,
) -> plt.Figure:
    """Plot PCA n=1 for both magnitude and phase"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    # Magnitude PCA
    magnitudes = np.abs(complex_samples)
    if magnitudes.shape[0] > 1 and magnitudes.shape[1] > 1:
        pca_mag = PCA(n_components=1)
        mag_pca = pca_mag.fit_transform(magnitudes)
        ax1.plot(time_axis, mag_pca[:, 0], 'b-', linewidth=1.5)
        ax1.set_title(f"Magnitude PCA (n=1) - Explained Var: {pca_mag.explained_variance_ratio_[0]*100:.1f}%")
    else:
        ax1.plot(time_axis, np.mean(magnitudes, axis=1), 'b-', linewidth=1.5)
        ax1.set_title("Magnitude Mean (insufficient data for PCA)")
    ax1.set_ylabel("PC1 (Magnitude)")
    ax1.grid(True, alpha=0.3)
    if highlight_time is not None:
        ax1.axvline(highlight_time, color="#ff4d4f", linestyle="--", linewidth=1.2, alpha=0.8)

    # Phase PCA
    phases = np.unwrap(np.angle(complex_samples), axis=0)
    phases = np.nan_to_num(phases, nan=0.0, posinf=0.0, neginf=0.0)
    if phases.shape[0] > 1 and phases.shape[1] > 1:
        pca_phase = PCA(n_components=1)
        phase_pca = pca_phase.fit_transform(phases)
        ax2.plot(time_axis, phase_pca[:, 0], 'g-', linewidth=1.5)
        ax2.set_title(f"Phase PCA (n=1) - Explained Var: {pca_phase.explained_variance_ratio_[0]*100:.1f}%")
    else:
        ax2.plot(time_axis, np.mean(phases, axis=1), 'g-', linewidth=1.5)
        ax2.set_title("Phase Mean (insufficient data for PCA)")
    ax2.set_ylabel("PC1 (Phase)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)
    if highlight_time is not None:
        ax2.axvline(highlight_time, color="#ff4d4f", linestyle="--", linewidth=1.2, alpha=0.8)

    fig.suptitle("PCA Analysis (n=1)", y=0.98, fontweight='bold')
    fig.tight_layout()
    return fig


def compute_engineered_features(
    complex_samples: np.ndarray,
    winsorize_limits: tuple[float, float] = (0.05, 0.05),
) -> dict[str, np.ndarray]:
    """
    Compute 2D feature vectors using temporal feature engineering:
    1. Winsorize - clip extreme values to reduce outlier impact
    2. Standardize - zero mean, unit variance
    3. Accelerate - second derivative to capture motion dynamics

    Returns magnitude and phase PCA features after engineering.
    """
    # Get magnitude and phase from complex samples
    magnitudes = np.abs(complex_samples)
    phases = np.unwrap(np.angle(complex_samples), axis=0)
    phases = np.nan_to_num(phases, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply PCA (n=1) first to get 1D time series
    if magnitudes.shape[0] > 1 and magnitudes.shape[1] > 1:
        pca_mag = PCA(n_components=1)
        mag_pca = pca_mag.fit_transform(magnitudes).flatten()
        pca_phase = PCA(n_components=1)
        phase_pca = pca_phase.fit_transform(phases).flatten()
    else:
        mag_pca = np.mean(magnitudes, axis=1)
        phase_pca = np.mean(phases, axis=1)

    # 1. Winsorize - clip extreme values
    mag_winsorized = np.array(winsorize(mag_pca, limits=winsorize_limits))
    phase_winsorized = np.array(winsorize(phase_pca, limits=winsorize_limits))

    # 2. Standardize - zero mean, unit variance
    scaler = StandardScaler()
    if len(mag_winsorized) > 1:
        mag_standardized = scaler.fit_transform(mag_winsorized.reshape(-1, 1)).flatten()
        phase_standardized = scaler.fit_transform(phase_winsorized.reshape(-1, 1)).flatten()
    else:
        mag_standardized = mag_winsorized
        phase_standardized = phase_winsorized

    # 3. Accelerate - second derivative (acceleration)
    if len(mag_standardized) > 2:
        # First derivative (velocity)
        mag_velocity = np.gradient(mag_standardized)
        phase_velocity = np.gradient(phase_standardized)
        # Second derivative (acceleration)
        mag_acceleration = np.gradient(mag_velocity)
        phase_acceleration = np.gradient(phase_velocity)
    else:
        mag_acceleration = np.zeros_like(mag_standardized)
        phase_acceleration = np.zeros_like(phase_standardized)

    return {
        "mag_raw": mag_pca,
        "phase_raw": phase_pca,
        "mag_winsorized": mag_winsorized,
        "phase_winsorized": phase_winsorized,
        "mag_standardized": mag_standardized,
        "phase_standardized": phase_standardized,
        "mag_acceleration": mag_acceleration,
        "phase_acceleration": phase_acceleration,
    }


def plot_2d_feature_vector(
    time_axis: np.ndarray,
    features: dict[str, np.ndarray],
    highlight_time: Optional[float] = None,
    highlight_idx: Optional[int] = None,
) -> plt.Figure:
    """
    Plot 2D feature vector visualization:
    - Left: Time series of engineered features
    - Right: 2D scatter plot (magnitude vs phase features)
    """
    fig = plt.figure(figsize=(16, 10))

    # Create grid for subplots with more spacing
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.45, wspace=0.35,
                          left=0.08, right=0.95, top=0.90, bottom=0.08)

    # Left column: Time series at each processing stage
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    # Right column: 2D scatter plots
    ax4 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[2, 1])

    # Time series plots
    # 1. Raw PCA features
    ax1.plot(time_axis, features["mag_raw"], 'b-', linewidth=1, label='Magnitude PC1', alpha=0.8)
    ax1.plot(time_axis, features["phase_raw"], 'g-', linewidth=1, label='Phase PC1', alpha=0.8)
    ax1.set_ylabel("Raw PCA")
    ax1.set_title("Stage 1: Raw PCA Features", fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    if highlight_time is not None:
        ax1.axvline(highlight_time, color="#ff4d4f", linestyle="--", linewidth=1.5, alpha=0.8)

    # 2. Winsorized + Standardized
    ax2.plot(time_axis, features["mag_standardized"], 'b-', linewidth=1, label='Magnitude', alpha=0.8)
    ax2.plot(time_axis, features["phase_standardized"], 'g-', linewidth=1, label='Phase', alpha=0.8)
    ax2.set_ylabel("Standardized")
    ax2.set_title("Stage 2: Winsorized + Standardized", fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    if highlight_time is not None:
        ax2.axvline(highlight_time, color="#ff4d4f", linestyle="--", linewidth=1.5, alpha=0.8)

    # 3. Acceleration (2nd derivative)
    ax3.plot(time_axis, features["mag_acceleration"], 'b-', linewidth=1, label='Magnitude Accel', alpha=0.8)
    ax3.plot(time_axis, features["phase_acceleration"], 'orange', linewidth=1, label='Phase Accel', alpha=0.8)
    ax3.set_ylabel("Acceleration")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Stage 3: Acceleration (2nd Derivative)", fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    if highlight_time is not None:
        ax3.axvline(highlight_time, color="#ff4d4f", linestyle="--", linewidth=1.5, alpha=0.8)

    # 2D Scatter plots
    # Color by time for trajectory visualization
    colors = np.arange(len(time_axis))

    # 1. Raw features 2D
    scatter1 = ax4.scatter(features["mag_raw"], features["phase_raw"],
                           c=colors, cmap='viridis', s=10, alpha=0.6)
    if highlight_idx is not None and highlight_idx < len(features["mag_raw"]):
        ax4.scatter(features["mag_raw"][highlight_idx], features["phase_raw"][highlight_idx],
                   c='red', s=100, marker='x', linewidths=2, zorder=5)
    ax4.set_xlabel("Magnitude PC1")
    ax4.set_ylabel("Phase PC1")
    ax4.set_title("2D: Raw Features", fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 2. Standardized features 2D
    scatter2 = ax5.scatter(features["mag_standardized"], features["phase_standardized"],
                           c=colors, cmap='viridis', s=10, alpha=0.6)
    if highlight_idx is not None and highlight_idx < len(features["mag_standardized"]):
        ax5.scatter(features["mag_standardized"][highlight_idx], features["phase_standardized"][highlight_idx],
                   c='red', s=100, marker='x', linewidths=2, zorder=5)
    ax5.set_xlabel("Magnitude (std)")
    ax5.set_ylabel("Phase (std)")
    ax5.set_title("2D: Standardized", fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 3. Acceleration features 2D (final 2D feature vector)
    scatter3 = ax6.scatter(features["mag_acceleration"], features["phase_acceleration"],
                           c=colors, cmap='plasma', s=15, alpha=0.7)
    if highlight_idx is not None and highlight_idx < len(features["mag_acceleration"]):
        ax6.scatter(features["mag_acceleration"][highlight_idx], features["phase_acceleration"][highlight_idx],
                   c='red', s=100, marker='x', linewidths=2, zorder=5)
    ax6.set_xlabel("Magnitude Acceleration")
    ax6.set_ylabel("Phase Acceleration")
    ax6.set_title("2D Feature Vector (Final)", fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Add colorbar for time
    cbar = fig.colorbar(scatter3, ax=ax6, label='Time index')

    fig.suptitle("2D Feature Engineering: Winsorize → Standardize → Accelerate",
                 y=0.96, fontsize=13, fontweight='bold')
    return fig


def plot_constellation(
    complex_samples: np.ndarray,
    column_indices: list[int],
    labels: list[int],
    current_idx: Optional[int] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    for pos, label in zip(column_indices, labels):
        data = complex_samples[:, pos]
        ax.scatter(data.real, data.imag, s=10, alpha=0.4, label=f"SC {label}")
        if current_idx is not None:
            current_point = complex_samples[current_idx, pos]
            ax.scatter(
                current_point.real,
                current_point.imag,
                s=80,
                marker="x",
                color="#ff4d4f",
                linewidths=1.5,
            )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title("Complex Plane (Constellation)")
    ax.legend(fontsize="x-small", loc="upper right")
    return fig


def plot_heatmap(
    time_axis: np.ndarray,
    subcarrier_values: list[int],
    magnitude_window: np.ndarray,
    cmap: str = "turbo",
    highlight_time: Optional[float] = None,
    interpolation: str = "bilinear",
    smoothing: float = 0.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mask_zeros: bool = False,
) -> plt.Figure:
    """Enhanced heatmap with smoothing and fixed color scaling"""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Optionally mask near-zero values
    if mask_zeros:
        magnitude_display = np.where(magnitude_window < 0.1, np.nan, magnitude_window)
    else:
        magnitude_display = magnitude_window

    # Apply smoothing to reduce noise
    if smoothing > 0:
        magnitude_smooth = gaussian_filter(magnitude_display, sigma=smoothing)
    else:
        magnitude_smooth = magnitude_display

    if len(time_axis) == 1:
        span = 1.0
        extent = [
            time_axis[0] - span / 2,
            time_axis[0] + span / 2,
            subcarrier_values[0],
            subcarrier_values[-1],
        ]
    else:
        extent = [time_axis[0], time_axis[-1], subcarrier_values[0], subcarrier_values[-1]]

    # Calculate actual range in this window for info
    valid_data = magnitude_window[np.isfinite(magnitude_window) & (magnitude_window > 0)]
    if len(valid_data) > 0:
        actual_min, actual_max = valid_data.min(), valid_data.max()
    else:
        actual_min, actual_max = 0, 0

    # Use explicit vmin/vmax for consistent color scaling
    mesh = ax.imshow(
        magnitude_smooth.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
    )

    # Add window stats to title
    title = f"Magnitude Heatmap (Time × Subcarrier)\nWindow range: [{actual_min:.2f}, {actual_max:.2f}]"
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Subcarrier Index")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    if highlight_time is not None:
        ax.axvline(highlight_time, color="#ff4d4f", linestyle="--", linewidth=1.5, alpha=0.9)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("|H|", fontsize=10)
    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
    return fig


def compute_temporal_profiles(complex_matrix: np.ndarray) -> dict[str, np.ndarray]:
    magnitudes = np.abs(complex_matrix)
    mean_mag = np.nanmean(magnitudes, axis=1)
    peak_mag = np.nanmax(magnitudes, axis=1)
    energy = np.nanmean(np.square(magnitudes), axis=1)
    normalized = complex_matrix / (np.abs(complex_matrix) + 1e-9)
    phase_alignment = np.abs(np.nanmean(normalized, axis=1))
    return {
        "mean": mean_mag,
        "peak": peak_mag,
        "energy": energy,
        "alignment": phase_alignment,
    }


def plot_temporal_insights(
    time_axis: np.ndarray,
    profiles: dict[str, np.ndarray],
    highlight_time: Optional[float] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

    axes[0].plot(time_axis, profiles["mean"], label="Mean |H|", linewidth=1.4)
    axes[0].plot(time_axis, profiles["peak"], label="Peak |H|", linewidth=1.0, alpha=0.4)
    axes[0].set_ylabel("|H|")
    axes[0].legend(loc="upper right", fontsize="small")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, profiles["energy"], color="#3f8efc", linewidth=1.2)
    axes[1].set_ylabel("Energy")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_axis, profiles["alignment"], color="#5cdb5c", linewidth=1.2)
    axes[2].set_ylabel("Phase coherence")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    if highlight_time is not None:
        for ax in axes:
            ax.axvline(highlight_time, color="#ff4d4f", linestyle="--", linewidth=1.2, alpha=0.8)

    fig.suptitle("Temporal Insight Profiles", y=0.98)
    fig.tight_layout()
    return fig


def plot_polar_snapshot(
    phases: np.ndarray,
    magnitudes: np.ndarray,
    subcarrier_values: list[int],
    snapshot_idx: int,
) -> plt.Figure:
    phase_snapshot = phases[snapshot_idx]
    mag_snapshot = magnitudes[snapshot_idx]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="polar")
    sc_norm = Normalize(vmin=min(subcarrier_values), vmax=max(subcarrier_values))
    scatter = ax.scatter(
        phase_snapshot,
        mag_snapshot,
        c=subcarrier_values,
        cmap="twilight",
        s=25,
        alpha=0.8,
        norm=sc_norm,
    )
    ax.set_title(f"Polar View (Sample #{snapshot_idx})")
    ax.set_rlabel_position(-22.5)
    fig.colorbar(scatter, ax=ax, pad=0.1, label="Subcarrier Index")
    return fig


def plot_average_beam_pattern(complex_matrix: np.ndarray, subcarrier_values: list[int]) -> plt.Figure:
    avg_vector = np.nanmean(complex_matrix, axis=0)
    phases = np.angle(avg_vector)
    magnitudes = np.abs(avg_vector)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="polar")
    sc_norm = Normalize(vmin=min(subcarrier_values), vmax=max(subcarrier_values))
    scatter = ax.scatter(phases, magnitudes, c=subcarrier_values, cmap="plasma", s=35, norm=sc_norm)
    ax.set_title("Average Beam Pattern")
    fig.colorbar(scatter, ax=ax, pad=0.1, label="Subcarrier Index")
    return fig


def plot_spectrogram_analysis(
    time_axis: np.ndarray,
    complex_matrix: np.ndarray,
    fs: float = 100.0,
    cmap: str = "viridis",
) -> plt.Figure:
    """Time-frequency spectrogram of beamforming magnitude"""
    # Compute mean magnitude across subcarriers
    mean_mag = np.nanmean(np.abs(complex_matrix), axis=1)

    if len(mean_mag) < 16:
        # Not enough samples for meaningful spectrogram
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "Insufficient samples for spectrogram",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Spectrogram (Need more samples)")
        return fig

    # Detrend and compute spectrogram
    signal_detrended = detrend(mean_mag)
    nperseg = min(64, max(16, len(signal_detrended) // 4))
    f, t, Sxx = spectrogram(signal_detrended, fs=fs, nperseg=nperseg,
                            noverlap=nperseg // 2)

    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 3))
    pcm = ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap=cmap)
    ax.set_title("Time-Frequency Spectrogram (Mean Magnitude)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, fs / 2)
    cbar = fig.colorbar(pcm, ax=ax, label="Power (dB)")
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig


def plot_doppler_shift_analysis(
    complex_matrix: np.ndarray,
    fs: float = 100.0,
    cmap: str = "plasma",
) -> plt.Figure:
    """Doppler shift analysis across subcarriers"""
    magnitudes = np.abs(complex_matrix)

    # Clean up complex_matrix: replace any NaN or inf values
    complex_matrix_clean = np.copy(complex_matrix)
    complex_matrix_clean[~np.isfinite(complex_matrix_clean)] = 0

    phases = np.unwrap(np.angle(complex_matrix_clean), axis=0)

    # Remove any NaN or inf that might have been introduced by unwrap
    phases = np.nan_to_num(phases, nan=0.0, posinf=0.0, neginf=0.0)

    # Weighted coherent signal
    weights = magnitudes / (np.sum(magnitudes, axis=1, keepdims=True) + 1e-10)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    coherent_signal = np.sum(complex_matrix_clean * weights, axis=1)
    coherent_phase = np.unwrap(np.angle(coherent_signal))

    # Clean coherent_phase
    coherent_phase = np.nan_to_num(coherent_phase, nan=0.0, posinf=0.0, neginf=0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 1. Doppler spectrum
    if len(coherent_phase) >= 8 and np.any(coherent_phase != 0):
        try:
            coherent_detrended = detrend(coherent_phase)
            coherent_detrended = np.nan_to_num(coherent_detrended, nan=0.0, posinf=0.0, neginf=0.0)

            window = np.hanning(len(coherent_detrended))
            windowed = coherent_detrended * window
            fft_coherent = np.fft.fftshift(np.fft.fft(windowed))
            power_coherent = np.abs(fft_coherent) ** 2
            freqs = np.fft.fftshift(np.fft.fftfreq(len(coherent_phase), 1 / fs))

            ax1.plot(freqs, 10 * np.log10(power_coherent + 1e-10), 'b-', linewidth=2)
            ax1.set_title("Doppler Spectrum", fontsize=11, fontweight='bold')
            ax1.set_xlabel("Doppler Frequency (Hz)")
            ax1.set_ylabel("Power (dB)")
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        except Exception as e:
            ax1.text(0.5, 0.5, f"Error: {str(e)[:30]}", ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Doppler Spectrum (Error)")
    else:
        ax1.text(0.5, 0.5, "Need more samples or invalid data", ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("Doppler Spectrum")

    # 2. Phase velocity (derivative)
    phase_velocity = np.diff(phases, axis=0)
    phase_velocity = np.nan_to_num(phase_velocity, nan=0.0, posinf=0.0, neginf=0.0)

    mean_phase_vel = np.nanmean(phase_velocity, axis=1)
    mean_phase_vel = np.nan_to_num(mean_phase_vel, nan=0.0, posinf=0.0, neginf=0.0)

    time_axis = np.arange(len(mean_phase_vel)) / fs

    ax2.plot(time_axis, mean_phase_vel, 'orange', linewidth=2)
    ax2.fill_between(time_axis, mean_phase_vel, alpha=0.3, color='orange')
    ax2.set_title("Phase Velocity (Doppler)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Phase Change (rad/frame)")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


def plot_timeline_overview(
    time_axis: np.ndarray,
    profiles: dict[str, np.ndarray],
    current_idx: int,
    window_start: int,
    window_end: int,
) -> plt.Figure:
    """Mini overview showing full timeline with current position"""
    fig, ax = plt.subplots(figsize=(10, 1.5))

    # Plot mean magnitude as overview
    ax.plot(time_axis, profiles["mean"], 'b-', linewidth=1, alpha=0.7)
    ax.fill_between(time_axis, profiles["mean"], alpha=0.2)

    # Highlight current window
    ax.axvspan(time_axis[window_start], time_axis[window_end - 1],
               alpha=0.3, color='yellow', label='Current window')

    # Mark current position
    ax.axvline(time_axis[current_idx], color='red', linestyle='--',
               linewidth=2, alpha=0.9, label='Current frame')

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Mean |H|", fontsize=9)
    ax.set_title("Timeline Overview", fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    return fig


def compute_motion_metrics(complex_matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Compute motion-related metrics from beamforming data"""
    # Clean complex matrix
    complex_matrix_clean = np.copy(complex_matrix)
    complex_matrix_clean[~np.isfinite(complex_matrix_clean)] = 0

    magnitudes = np.abs(complex_matrix_clean)
    phases = np.unwrap(np.angle(complex_matrix_clean), axis=0)

    # Clean phases
    phases = np.nan_to_num(phases, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Temporal variance (motion indicator)
    temporal_variance = np.var(magnitudes, axis=0)
    temporal_variance = np.nan_to_num(temporal_variance, nan=0.0, posinf=0.0, neginf=0.0)
    avg_temporal_variance = float(np.nanmean(temporal_variance))

    # 2. Phase velocity variance (Doppler spread)
    if len(phases) > 1:
        phase_velocity = np.diff(phases, axis=0)
        phase_velocity = np.nan_to_num(phase_velocity, nan=0.0, posinf=0.0, neginf=0.0)
        doppler_variance = np.var(phase_velocity, axis=1)
        doppler_variance = np.nan_to_num(doppler_variance, nan=0.0, posinf=0.0, neginf=0.0)
        avg_doppler_variance = float(np.nanmean(doppler_variance))
    else:
        doppler_variance = np.array([0.0])
        avg_doppler_variance = 0.0

    # 3. Magnitude change rate
    if len(magnitudes) > 1:
        mag_diff = np.diff(magnitudes, axis=0)
        mag_diff = np.nan_to_num(mag_diff, nan=0.0, posinf=0.0, neginf=0.0)
        change_rate = np.nanmean(np.abs(mag_diff), axis=1)
        change_rate = np.nan_to_num(change_rate, nan=0.0, posinf=0.0, neginf=0.0)
        avg_change_rate = float(np.nanmean(change_rate))
    else:
        change_rate = np.array([0.0])
        avg_change_rate = 0.0

    return {
        "temporal_variance": temporal_variance,
        "avg_temporal_variance": avg_temporal_variance,
        "doppler_variance": doppler_variance,
        "avg_doppler_variance": avg_doppler_variance,
        "change_rate": change_rate,
        "avg_change_rate": avg_change_rate,
    }


def plot_motion_analysis(
    time_axis: np.ndarray,
    motion_metrics: dict[str, np.ndarray],
) -> plt.Figure:
    """Visualize motion detection metrics"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # Adjust time axis for diff-based metrics
    time_diff = time_axis[1:] if len(time_axis) > 1 else time_axis

    # 1. Change rate (motion activity)
    change_rate = motion_metrics["change_rate"]
    if len(change_rate) > 0:
        ax1.plot(time_diff[:len(change_rate)], change_rate, 'g-', linewidth=2)
        ax1.fill_between(time_diff[:len(change_rate)], change_rate, alpha=0.3, color='green')
        ax1.set_ylabel("Magnitude Change Rate", fontsize=10)
        ax1.set_title("Motion Activity Indicators", fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add threshold line
        threshold = np.percentile(change_rate, 75)
        ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.5,
                   label=f'75th percentile: {threshold:.3f}')
        ax1.legend(fontsize=8)

    # 2. Doppler variance (motion complexity)
    doppler_var = motion_metrics["doppler_variance"]
    if len(doppler_var) > 0:
        ax2.plot(time_diff[:len(doppler_var)], doppler_var, 'purple', linewidth=2)
        ax2.fill_between(time_diff[:len(doppler_var)], doppler_var, alpha=0.3, color='purple')
        ax2.set_ylabel("Doppler Variance", fontsize=10)
        ax2.set_xlabel("Time (s)", fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main() -> None:
    st.set_page_config("📡 Beamforming Explorer", layout="wide")
    st.title("📡 Enhanced Beamforming (BFM) Explorer")
    st.caption("Advanced visualization suite for temporal and spatial beamforming characteristics")

    # Feature highlights
    with st.expander("ℹ️ What's New - Enhanced Features", expanded=False):
        st.markdown("""
        ### 🎬 Smooth Playback Controls
        - **Presets**: Choose from Smooth (1 step), Normal (10 steps), or Fast (25 steps) playback modes
        - **Custom FPS**: Adjust playback speed from 5-60 FPS for ultra-smooth transitions
        - **Variable step size**: Fine-tune how many samples to skip per frame

        ### 🎨 Advanced Visualization
        - **Fixed color scaling**: Prevents flickering/blinking in heatmaps during playback
        - **Gaussian smoothing**: Adjustable sigma (0-2.0) to reduce noise in heatmaps
        - **Multiple colormaps**: 10 different color schemes including turbo, viridis, jet, plasma, and more
        - **Interpolation options**: Bilinear, nearest, bicubic, or gaussian interpolation

        ### 🌊 Frequency Domain Analysis
        - **Spectrogram**: Time-frequency analysis showing Doppler shifts over time
        - **Doppler spectrum**: FFT-based frequency analysis of phase changes
        - **Phase velocity**: Real-time Doppler shift visualization

        ### 🎯 Motion Detection
        - **Change rate tracking**: Magnitude variation indicating motion activity
        - **Doppler variance**: Phase velocity spread showing motion complexity
        - **Activity metrics**: Global statistics for motion characterization

        ### 🗺️ Navigation Enhancements
        - **Timeline overview**: Mini-map showing current position and window in full dataset
        - **Window highlighting**: Visual indication of current analysis window
        - **Frame position indicator**: Real-time position tracking with red marker
        """)

    files = get_available_files(DATA_DIR)
    if not files:
        st.error(f"No CSV files found under `{DATA_DIR}`.")
        st.stop()

    # Set default file for realtime activity testing
    default_file = "bfm_data_pg_standing_home_20251027_211523.csv"
    default_index = 0
    for i, f in enumerate(files):
        if default_file in f:
            default_index = i
            break

    selected_file = st.sidebar.selectbox("Select Beamforming Capture", files, index=default_index)

    state = st.session_state
    if "bfm_last_file" not in state:
        state.bfm_last_file = selected_file
    if selected_file != state.bfm_last_file:
        state.bfm_last_file = selected_file
        state.bfm_playback_idx = 0
        state.bfm_playing = False
        state.bfm_step = 25
        state.bfm_window_size = 250

    df = load_bfm_dataframe(selected_file)
    subcarrier_values, _ = collect_subcarrier_columns(df.columns)
    if not subcarrier_values:
        st.error("No SCIDX_* ratio columns found in the selected file.")
        st.stop()

    st.sidebar.markdown("### Pre-processing")
    downsample = st.sidebar.slider("Downsample factor", min_value=1, max_value=10, value=1, help="Keeps every Nth sample.")

    complex_matrix, timestamps = build_complex_matrix((selected_file, downsample))
    num_samples, num_subcarriers = complex_matrix.shape
    time_axis = np.arange(num_samples, dtype=float) if timestamps is None else (timestamps - timestamps[0]).astype(float)

    if num_samples == 0:
        st.warning("The selected file has no samples after processing.")
        st.stop()

    summary = compute_summary_stats(complex_matrix)
    motion_metrics = compute_motion_metrics(complex_matrix)

    info_cols = st.columns(4)
    info_cols[0].metric("Samples", f"{num_samples:,}")
    info_cols[1].metric("Subcarriers", f"{num_subcarriers}")
    info_cols[2].metric("Mean |H|", f"{summary['mean_magnitude']:.3f}")
    info_cols[3].metric("Max |H|", f"{summary['max_magnitude']:.3f}")

    # Sidebar stats
    st.sidebar.markdown("### 📊 Global Statistics")
    st.sidebar.metric("Circular variance", f"{summary['circular_variance']:.3f}")
    st.sidebar.metric("Motion activity", f"{motion_metrics['avg_change_rate']:.4f}",
                     help="Average magnitude change rate")
    st.sidebar.metric("Doppler spread", f"{motion_metrics['avg_doppler_variance']:.4f}",
                     help="Average phase velocity variance")

    state = st.session_state
    if "bfm_playback_idx" not in state:
        state.bfm_playback_idx = 0
    if "bfm_playing" not in state:
        state.bfm_playing = False
    if "bfm_step" not in state:
        state.bfm_step = 25

    state.bfm_playback_idx = int(np.clip(state.bfm_playback_idx, 0, num_samples - 1))
    max_step = max(1, min(200, max(1, num_samples - 1)))
    state.bfm_step = int(np.clip(state.bfm_step, 1, max_step))

    st.sidebar.markdown("### 🎬 Playback Presets")
    preset = st.sidebar.radio(
        "Speed preset",
        ["Custom", "🐌 Smooth (1 step)", "⚡ Normal (10 steps)", "🚀 Fast (25 steps)"],
        horizontal=False,
        label_visibility="collapsed"
    )

    # Apply preset values
    if preset == "🐌 Smooth (1 step)":
        state.bfm_step = 1
        playback_fps = 20
        st.sidebar.info("Smooth: 1 step, 20 FPS")
    elif preset == "⚡ Normal (10 steps)":
        state.bfm_step = 10
        playback_fps = 30
        st.sidebar.info("Normal: 10 steps, 30 FPS")
    elif preset == "🚀 Fast (25 steps)":
        state.bfm_step = 25
        playback_fps = 40
        st.sidebar.info("Fast: 25 steps, 40 FPS")
    else:  # Custom
        playback_fps = st.sidebar.slider("FPS (frames/sec)", 5, 60, 30, 5,
                                         help="Playback speed in frames per second")
        step_slider_val = st.sidebar.slider(
            "Playback step (samples)",
            min_value=1,
            max_value=max_step,
            value=state.bfm_step,
            key="bfm_step_slider",
        )
        state.bfm_step = int(step_slider_val)

    # Store FPS in state
    if "bfm_fps" not in state:
        state.bfm_fps = 30
    state.bfm_fps = playback_fps

    st.sidebar.markdown("### Timeline playback")
    ctrl_cols = st.sidebar.columns(3)
    if ctrl_cols[0].button("⏮️", use_container_width=True):
        state.bfm_playback_idx = 0
    play_label = "⏸️ Pause" if state.bfm_playing else "▶️ Play"
    if ctrl_cols[1].button(play_label, use_container_width=True):
        state.bfm_playing = not state.bfm_playing
    if ctrl_cols[2].button("⏭️ End", use_container_width=True):
        state.bfm_playback_idx = num_samples - 1
        state.bfm_playing = False

    slider_val = st.sidebar.slider(
        "Timeline index",
        min_value=0,
        max_value=num_samples - 1,
        value=state.bfm_playback_idx,
        key="bfm_frame_slider",
    )
    if slider_val != state.bfm_playback_idx:
        state.bfm_playback_idx = int(slider_val)

    highlight_idx = int(state.bfm_playback_idx)
    highlight_time = float(time_axis[highlight_idx])

    max_window = min(num_samples, 500)
    default_window = min(250, max_window)
    if "bfm_window_size" in state:
        state.bfm_window_size = int(np.clip(state.bfm_window_size, 1, max_window))
    else:
        state.bfm_window_size = default_window if default_window > 0 else 1
    window_size = st.sidebar.slider(
        "Window size (samples)",
        min_value=1,
        max_value=max_window,
        value=state.bfm_window_size,
        key="bfm_window_size",
    )
    window_size = int(window_size)
    window_start = max(0, highlight_idx - window_size + 1)
    window_end = highlight_idx + 1
    window_time = time_axis[window_start:window_end]
    complex_window = complex_matrix[window_start:window_end]
    magnitude_window = np.abs(complex_window)
    highlight_local_idx = len(window_time) - 1
    highlight_window_time = float(window_time[highlight_local_idx])

    temporal_profiles = compute_temporal_profiles(complex_matrix)
    window_profiles = {k: v[window_start:window_end] for k, v in temporal_profiles.items()}
    magnitudes = np.abs(complex_matrix)
    phases = np.angle(complex_matrix)

    # Progress bar
    progress_pct = (highlight_idx / max(1, num_samples - 1)) * 100
    st.progress(highlight_idx / max(1, num_samples - 1))

    st.caption(
        f"📍 **Timeline position:** sample {highlight_idx + 1}/{num_samples} ({progress_pct:.1f}%) • "
        f"⏱️ t = {highlight_time:.3f} s • "
        f"🪟 window [{window_start + 1} – {highlight_idx + 1}]"
    )

    frame_cols = st.columns(3)
    frame_cols[0].metric("Frame mean |H|", f"{temporal_profiles['mean'][highlight_idx]:.3f}")
    frame_cols[1].metric("Frame peak |H|", f"{temporal_profiles['peak'][highlight_idx]:.3f}")
    frame_cols[2].metric("Phase coherence", f"{temporal_profiles['alignment'][highlight_idx]:.3f}")

    st.sidebar.markdown("### 🎨 Visualization controls")

    # Colormap selection
    cmap_choice = st.sidebar.selectbox("Color Map", CMAPS, index=0,
                                       help="Color scheme for heatmaps and spectrograms")

    # Heatmap smoothing and interpolation
    with st.sidebar.expander("🔧 Heatmap Settings", expanded=False):
        heatmap_smoothing = st.slider("Smoothing (sigma)", 0.0, 2.0, 0.5, 0.1,
                                       help="Gaussian smoothing for heatmap (0=none)")
        heatmap_interpolation = st.selectbox("Interpolation",
                                              ["bilinear", "nearest", "bicubic", "gaussian"],
                                              index=0)
        use_fixed_colorscale = st.checkbox("Fixed color scale", value=True,
                                            help="Use global min/max to prevent color flickering")

        # Option to mask very low values
        mask_zeros = st.checkbox("Mask near-zero values", value=False,
                                 help="Set values below 0.1 to NaN (white) for clearer visualization")

    # Calculate global color scale if enabled
    if use_fixed_colorscale:
        if "bfm_global_vmin" not in state or state.bfm_last_file != selected_file:
            magnitudes_full = np.abs(complex_matrix)
            # Exclude zeros and very small values to get better color range for actual signals
            nonzero_mags = magnitudes_full[magnitudes_full > 0.1]
            if len(nonzero_mags) > 0:
                # Use 5th and 95th percentile of non-zero values for better dynamic range
                state.bfm_global_vmin = float(np.percentile(nonzero_mags, 5))
                state.bfm_global_vmax = float(np.percentile(nonzero_mags, 95))
            else:
                # Fallback if no significant data
                state.bfm_global_vmin = float(np.percentile(magnitudes_full, 2))
                state.bfm_global_vmax = float(np.percentile(magnitudes_full, 98))
        global_vmin = state.bfm_global_vmin
        global_vmax = state.bfm_global_vmax
        # Display current color scale
        st.sidebar.caption(f"Color range: [{global_vmin:.2f}, {global_vmax:.2f}]")
    else:
        global_vmin = None
        global_vmax = None
        st.sidebar.caption("Color range: Auto (per window)")

    # Subcarrier selection
    default_idx = np.linspace(0, len(subcarrier_values) - 1, num=4, dtype=int)
    default_selection = [subcarrier_values[i] for i in default_idx]
    subcarrier_choice = st.sidebar.multiselect(
        "Subcarriers to highlight",
        options=subcarrier_values,
        default=default_selection,
    )
    show_all_subcarriers = st.sidebar.checkbox(
        "Show all subcarriers",
        value=False,
        help="Display every subcarrier trace in the window (may be visually dense).",
        key="bfm_show_all_subcarriers",
    )
    if show_all_subcarriers:
        subcarrier_choice = list(subcarrier_values)
    if not subcarrier_choice:
        st.warning("Select at least one subcarrier to visualize temporal traces.")
        st.stop()

    subcarrier_lookup = {value: pos for pos, value in enumerate(subcarrier_values)}
    highlight_indices = [subcarrier_lookup[val] for val in subcarrier_choice]

    # Timeline overview minimap
    st.markdown("---")
    st.markdown("## 🗺️ Timeline Overview")
    fig = plot_timeline_overview(time_axis, temporal_profiles, highlight_idx, window_start, window_end)
    st.pyplot(fig)
    plt.close(fig)

    # Main visualizations
    st.markdown("---")
    st.markdown("## 📊 Temporal Analysis")

    # Magnitude traces
    col_mag, col_constellation = st.columns((2, 1))
    with col_mag:
        fig = plot_temporal_traces(
            window_time,
            complex_window,
            highlight_indices,
            subcarrier_choice,
            highlight_time=highlight_window_time,
        )
        st.pyplot(fig)
        plt.close(fig)
    with col_constellation:
        fig = plot_constellation(
            complex_window,
            highlight_indices,
            subcarrier_choice,
            current_idx=highlight_local_idx,
        )
        st.pyplot(fig)
        plt.close(fig)

    # Phase traces
    st.markdown("### Phase Time Series")
    fig = plot_phase_traces(
        window_time,
        complex_window,
        highlight_indices,
        subcarrier_choice,
        highlight_time=highlight_window_time,
    )
    st.pyplot(fig)
    plt.close(fig)

    # PCA Analysis
    st.markdown("### PCA Analysis (n=1)")
    fig = plot_pca_traces(
        window_time,
        complex_window,
        highlight_time=highlight_window_time,
    )
    st.pyplot(fig)
    plt.close(fig)

    # 2D Feature Vector with Temporal Engineering
    st.markdown("### 2D Feature Vector (Engineered)")
    engineered_features = compute_engineered_features(complex_window)
    fig = plot_2d_feature_vector(
        window_time,
        engineered_features,
        highlight_time=highlight_window_time,
        highlight_idx=highlight_local_idx,
    )
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.markdown("## 🔥 Magnitude Heatmap")
    fig = plot_heatmap(
        window_time,
        subcarrier_values,
        magnitude_window,
        cmap=cmap_choice,
        highlight_time=highlight_window_time,
        interpolation=heatmap_interpolation,
        smoothing=heatmap_smoothing,
        vmin=global_vmin,
        vmax=global_vmax,
        mask_zeros=mask_zeros,
    )
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.markdown("## 📈 Temporal Insights")
    fig = plot_temporal_insights(window_time, window_profiles, highlight_time=highlight_window_time)
    st.pyplot(fig)
    plt.close(fig)

    # Advanced frequency analysis
    st.markdown("---")
    st.markdown("## 🌊 Frequency Domain Analysis")

    # Calculate effective sampling rate
    if len(time_axis) > 1:
        effective_fs = 1.0 / np.mean(np.diff(time_axis)) if np.mean(np.diff(time_axis)) > 0 else 100.0
    else:
        effective_fs = 100.0

    col_spec, col_doppler = st.columns(2)
    with col_spec:
        fig = plot_spectrogram_analysis(window_time, complex_window, fs=effective_fs, cmap=cmap_choice)
        st.pyplot(fig)
        plt.close(fig)
    with col_doppler:
        fig = plot_doppler_shift_analysis(complex_window, fs=effective_fs, cmap=cmap_choice)
        st.pyplot(fig)
        plt.close(fig)

    # Motion analysis
    st.markdown("---")
    st.markdown("## 🎯 Motion Detection Analysis")
    fig = plot_motion_analysis(time_axis, motion_metrics)
    st.pyplot(fig)
    plt.close(fig)

    # Expandable advanced views
    st.markdown("---")
    st.markdown("## 🔬 Advanced Analysis (Click to Expand)")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        with st.expander("🔵 Polar Snapshot (Current Frame)", expanded=False):
            fig = plot_polar_snapshot(phases, magnitudes, subcarrier_values, highlight_idx)
            st.pyplot(fig)
            plt.close(fig)

    with col_exp2:
        with st.expander("📡 Average Beamforming Envelope", expanded=False):
            fig = plot_average_beam_pattern(complex_matrix, subcarrier_values)
            st.pyplot(fig)
            plt.close(fig)

    with st.expander("📋 Raw Metadata Preview", expanded=False):
        meta_cols = [c for c in df.columns if not c.startswith("SCIDX_")]
        if meta_cols:
            st.dataframe(df[meta_cols].head(20))
        else:
            st.info("No metadata columns found (only SCIDX_* data available)")

    if state.bfm_playing:
        if highlight_idx >= num_samples - 1:
            state.bfm_playing = False
        else:
            next_idx = min(highlight_idx + state.bfm_step, num_samples - 1)
            # Use FPS to determine sleep time
            sleep_time = 1.0 / state.bfm_fps
            time.sleep(sleep_time)
            state.bfm_playback_idx = next_idx
            request_rerun()


if __name__ == "__main__":
    main()
