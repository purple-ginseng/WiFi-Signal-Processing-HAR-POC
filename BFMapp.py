import glob
import os
import re
import time
from functools import lru_cache
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

DATA_DIR = "./bfm_processed_csv"
SUBCARRIER_PATTERN = re.compile(r"SCIDX_(-?\d+)_Ratio_(Real|Imag)", re.IGNORECASE)


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

    if downsample > 1:
        df_real = df_real.iloc[::downsample].reset_index(drop=True)
        df_imag = df_imag.iloc[::downsample].reset_index(drop=True)
        timestamps = (
            df["timestamp"].iloc[::downsample].reset_index(drop=True) if "timestamp" in df.columns else None
        )
    else:
        timestamps = df["timestamp"] if "timestamp" in df.columns else None

    complex_matrix = df_real.to_numpy(dtype=np.float32) + 1j * df_imag.to_numpy(dtype=np.float32)

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
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
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
    mesh = ax.imshow(
        magnitude_window.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="nearest"
    )
    ax.set_title("Magnitude Heatmap (Time × Subcarrier)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Subcarrier Index")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    if highlight_time is not None:
        ax.axvline(highlight_time, color="#ff4d4f", linestyle="--", linewidth=1.2, alpha=0.8)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("|H|")
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


def main() -> None:
    st.set_page_config("📡 Beamforming Explorer", layout="wide")
    st.title("📡 Beamforming (BFM) Explorer")
    st.caption("Inspect temporal and spatial beamforming characteristics from processed CSV logs.")

    files = get_available_files(DATA_DIR)
    if not files:
        st.error(f"No CSV files found under `{DATA_DIR}`.")
        st.stop()

    selected_file = st.sidebar.selectbox("Select Beamforming Capture", files)

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
    info_cols = st.columns(3)
    info_cols[0].metric("Samples", f"{num_samples:,}")
    info_cols[1].metric("Subcarriers", f"{num_subcarriers}")
    info_cols[2].metric("Mean |H|", f"{summary['mean_magnitude']:.3f}")
    st.sidebar.metric("Max |H|", f"{summary['max_magnitude']:.3f}")
    st.sidebar.metric("Circular variance", f"{summary['circular_variance']:.3f}")

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
    step_slider_val = st.sidebar.slider(
        "Playback step (samples)",
        min_value=1,
        max_value=max_step,
        value=state.bfm_step,
        key="bfm_step_slider",
    )
    state.bfm_step = int(step_slider_val)

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

    st.caption(
        f"Timeline position: sample {highlight_idx + 1}/{num_samples} • t = {highlight_time:.3f} s "
        f"(window {window_start + 1} – {highlight_idx + 1})"
    )

    frame_cols = st.columns(3)
    frame_cols[0].metric("Frame mean |H|", f"{temporal_profiles['mean'][highlight_idx]:.3f}")
    frame_cols[1].metric("Frame peak |H|", f"{temporal_profiles['peak'][highlight_idx]:.3f}")
    frame_cols[2].metric("Phase coherence", f"{temporal_profiles['alignment'][highlight_idx]:.3f}")

    st.sidebar.markdown("### Visualization controls")
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

    col_time, col_constellation = st.columns((2, 1))
    with col_time:
        st.pyplot(
            plot_temporal_traces(
                window_time,
                complex_window,
                highlight_indices,
                subcarrier_choice,
                highlight_time=highlight_window_time,
            )
        )
    with col_constellation:
        st.pyplot(
            plot_constellation(
                complex_window,
                highlight_indices,
                subcarrier_choice,
                current_idx=highlight_local_idx,
            )
        )

    st.pyplot(plot_heatmap(window_time, subcarrier_values, magnitude_window, highlight_time=highlight_window_time))
    st.pyplot(plot_temporal_insights(window_time, window_profiles, highlight_time=highlight_window_time))

    with st.expander("Polar snapshot", expanded=False):
        st.pyplot(plot_polar_snapshot(phases, magnitudes, subcarrier_values, highlight_idx))

    with st.expander("Average beamforming envelope", expanded=False):
        st.pyplot(plot_average_beam_pattern(complex_matrix, subcarrier_values))

    with st.expander("Raw metadata", expanded=False):
        meta_cols = [c for c in df.columns if not c.startswith("SCIDX_")]
        st.dataframe(df[meta_cols].head(20))

    if state.bfm_playing:
        if highlight_idx >= num_samples - 1:
            state.bfm_playing = False
        else:
            next_idx = min(highlight_idx + state.bfm_step, num_samples - 1)
            time.sleep(0.15)
            state.bfm_playback_idx = next_idx
            request_rerun()


if __name__ == "__main__":
    main()
