import streamlit as st
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, detrend
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
import time

DATA_DIR = "./data"
FS = 200.0  # Sampling rate per subcarrier
CMAPS = ["viridis", "jet", "turbo", "plasma", "inferno", "magma", "cividis", "RdYlBu_r"]

# --- Helper Functions ---
def compute_attention(seq, downsample=20):
    """Compute attention matrix for signal sequence"""
    if len(seq) < downsample:
        downsample = max(1, len(seq) // 10)
    seq_downsampled = seq[::downsample]
    x = torch.tensor(seq_downsampled, dtype=torch.float32).unsqueeze(-1)
    scores = torch.matmul(x, x.T) / np.sqrt(x.shape[-1])
    return F.softmax(scores, dim=-1).detach().numpy()

def draw_enhanced_waveforms(signal, complex_matrix, fs=FS, cmap="viridis", rssi_data=None):
    """Enhanced multi-panel waveform visualization with optional RSSI"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    time_axis = np.arange(len(signal)) / fs

    # 1. Amplitude Envelope
    ax1.plot(time_axis, signal, 'b-', linewidth=1.5, alpha=0.8)
    ax1.fill_between(time_axis, signal, alpha=0.3)
    ax1.set_title("CSI Amplitude Envelope (All Subcarriers)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Magnitude")
    ax1.grid(True, alpha=0.3)

    # 2. Per-Subcarrier Magnitude Traces (Top 5)
    magnitudes = np.abs(complex_matrix)
    avg_mag_per_subcarrier = np.mean(magnitudes, axis=0)
    top_n = min(5, magnitudes.shape[1])
    best_subcarriers = np.argsort(avg_mag_per_subcarrier)[-top_n:]

    time_axis_matrix = np.arange(magnitudes.shape[0]) / fs
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, top_n))
    for i, sc_idx in enumerate(best_subcarriers):
        ax2.plot(time_axis_matrix, magnitudes[:, sc_idx],
                label=f"SC {sc_idx}", linewidth=1.5, color=colors[i], alpha=0.8)
    ax2.set_title("Top 5 Subcarrier Magnitude Traces", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Magnitude")
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Moving Average with Std Dev
    window_size = max(5, len(signal) // 50)
    if window_size % 2 == 0:
        window_size += 1

    from scipy.ndimage import uniform_filter1d
    signal_smooth = uniform_filter1d(signal, size=window_size)

    ax3.plot(time_axis, signal, 'lightblue', linewidth=0.8, alpha=0.5, label='Raw')
    ax3.plot(time_axis, signal_smooth, 'darkblue', linewidth=2, label='Moving Avg')
    ax3.set_title("Signal with Moving Average", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Magnitude")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. RSSI over Time (if available) or Frequency Domain (FFT)
    if rssi_data is not None and len(rssi_data) > 0:
        # Plot RSSI over time
        rssi_time_axis = np.arange(len(rssi_data)) / fs
        ax4.plot(rssi_time_axis, rssi_data, 'r-', linewidth=2, alpha=0.8)
        ax4.fill_between(rssi_time_axis, rssi_data, alpha=0.3, color='red')
        ax4.set_title("RSSI over Time", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("RSSI (dBm)")
        ax4.grid(True, alpha=0.3)

        # Add mean line
        mean_rssi = np.mean(rssi_data)
        ax4.axhline(y=mean_rssi, color='darkred', linestyle='--',
                   label=f'Mean: {mean_rssi:.1f} dBm', linewidth=1.5)
        ax4.legend(loc='upper right')
    else:
        # Frequency Domain (FFT) - fallback when no RSSI
        signal_detrend = detrend(signal)
        window = np.hanning(len(signal_detrend))
        fft_result = np.fft.fft(signal_detrend * window)
        freqs = np.fft.fftfreq(len(signal), 1/fs)

        # Only plot positive frequencies
        pos_mask = freqs >= 0
        ax4.plot(freqs[pos_mask], np.abs(fft_result[pos_mask]), 'g-', linewidth=1.5)
        ax4.set_title("Frequency Spectrum (FFT)", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Magnitude")
        ax4.set_xlim(0, fs/2)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def draw_enhanced_heatmap(matrix, cmap, title="CSI Heatmap", aspect_ratio='auto', interpolation='bilinear',
                         vmin=None, vmax=None):
    """Enhanced heatmap with better visualization and consistent color scaling"""
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(14, 6))

    # Apply minimal smoothing to reduce noise while preserving structure
    # Lower sigma = less smoothing = more stable visualization
    matrix_smooth = gaussian_filter(matrix, sigma=0.5)

    # Create explicit normalization for consistent color mapping
    if vmin is not None and vmax is not None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    else:
        norm = None

    # Use consistent color scaling to prevent blinking
    cax = ax.imshow(matrix_smooth, aspect=aspect_ratio, cmap=cmap, origin='lower',
                    interpolation=interpolation, norm=norm)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time Frame", fontsize=11)
    ax.set_ylabel("Subcarrier Index", fontsize=11)

    # Add colorbar with label and fixed ticks
    cbar = fig.colorbar(cax, ax=ax, label="Magnitude")
    cbar.ax.tick_params(labelsize=9)

    # If using fixed scale, set explicit colorbar limits
    if vmin is not None and vmax is not None:
        cbar.set_ticks(np.linspace(vmin, vmax, 5))

    # Add grid for better readability
    ax.set_xticks(np.arange(0, matrix.shape[1], max(1, matrix.shape[1]//10)))
    ax.set_yticks(np.arange(0, matrix.shape[0], max(1, matrix.shape[0]//8)))
    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)

    plt.tight_layout()
    return fig

def draw_spectrogram(signal, fs=FS, cmap="viridis", nperseg=256):
    """Enhanced spectrogram visualization"""
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=min(nperseg, len(signal)//2),
                            noverlap=min(nperseg//2, len(signal)//4))
    Sxx_smooth = gaussian_filter(Sxx, sigma=1.2)

    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx_smooth + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 4))
    pcm = ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap=cmap)
    ax.set_title("Time-Frequency Spectrogram", fontsize=13, fontweight='bold')
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylim(0, fs / 2)
    ax.set_xlim(0, len(signal) / fs)
    cbar = fig.colorbar(pcm, ax=ax, label="Power (dB)")
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig

def draw_phase_analysis(complex_matrix, cmap, phase_vmin=-np.pi, phase_vmax=np.pi):
    """Detailed phase analysis visualization with consistent scaling"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))

    # Extract phase information
    phase_matrix = np.angle(complex_matrix)
    unwrapped_phase = np.unwrap(phase_matrix, axis=0)

    # 1. Phase heatmap with fixed scale [-π, π]
    im1 = ax1.imshow(phase_matrix.T, aspect='auto', cmap=cmap, origin='lower',
                     vmin=phase_vmin, vmax=phase_vmax)
    ax1.set_title("Instantaneous Phase Heatmap")
    ax1.set_xlabel("Time Frame")
    ax1.set_ylabel("Subcarrier Index")
    fig.colorbar(im1, ax=ax1, label="Phase (rad)")

    # 2. Unwrapped phase heatmap (use percentile-based scaling for stability)
    unwrap_vmin = np.percentile(unwrapped_phase, 2)
    unwrap_vmax = np.percentile(unwrapped_phase, 98)
    im2 = ax2.imshow(unwrapped_phase.T, aspect='auto', cmap=cmap, origin='lower',
                     vmin=unwrap_vmin, vmax=unwrap_vmax)
    ax2.set_title("Unwrapped Phase Heatmap")
    ax2.set_xlabel("Time Frame")
    ax2.set_ylabel("Subcarrier Index")
    fig.colorbar(im2, ax=ax2, label="Phase (rad)")

    # 3. Phase variance across subcarriers
    phase_var = np.var(unwrapped_phase, axis=1)
    ax3.plot(phase_var, 'purple', linewidth=2)
    ax3.set_title("Phase Variance (Motion Indicator)")
    ax3.set_xlabel("Time Frame")
    ax3.set_ylabel("Variance (rad²)")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)  # Fixed lower bound

    # 4. Phase velocity (time derivative)
    phase_velocity = np.diff(unwrapped_phase, axis=0)
    mean_phase_vel = np.mean(phase_velocity, axis=1)
    ax4.plot(mean_phase_vel, 'orange', linewidth=2)
    ax4.set_title("Mean Phase Velocity (Doppler)")
    ax4.set_xlabel("Time Frame")
    ax4.set_ylabel("Phase Change (rad/frame)")
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig

def draw_doppler_analysis(complex_matrix, cmap, fs=30.0):
    """Enhanced Doppler analysis with multiple insights"""
    magnitudes = np.abs(complex_matrix)
    weights = magnitudes / (np.sum(magnitudes, axis=1, keepdims=True) + 1e-10)
    coherent_signal = np.sum(complex_matrix * weights, axis=1)
    coherent_phase = np.unwrap(np.angle(coherent_signal))

    avg_mag_per_subcarrier = np.mean(magnitudes, axis=0)
    num_top_subcarriers = min(5, complex_matrix.shape[1])
    best_subcarriers = np.argsort(avg_mag_per_subcarrier)[-num_top_subcarriers:]
    selected_signal = np.mean(complex_matrix[:, best_subcarriers], axis=1)
    selected_phase = np.unwrap(np.angle(selected_signal))

    phase_matrix = np.unwrap(np.angle(complex_matrix), axis=0)
    phase_variance = np.var(phase_matrix, axis=1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))

    # 1. Coherent Doppler Spectrum
    coherent_detrended = detrend(coherent_phase)
    window = np.hanning(len(coherent_detrended))
    windowed = coherent_detrended * window
    fft_coherent = np.fft.fftshift(np.fft.fft(windowed))
    power_coherent = np.abs(fft_coherent) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(len(coherent_phase), 1/fs))

    ax1.plot(freqs, 10*np.log10(power_coherent + 1e-10), 'b-', linewidth=2)
    ax1.set_title("Coherent Doppler Spectrum", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Doppler Frequency (Hz)")
    ax1.set_ylabel("Power (dB)")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)

    # 2. Best Subcarriers Doppler Spectrum
    selected_detrended = detrend(selected_phase)
    windowed_selected = selected_detrended * window
    fft_selected = np.fft.fftshift(np.fft.fft(windowed_selected))
    power_selected = np.abs(fft_selected) ** 2

    ax2.plot(freqs, 10*np.log10(power_selected + 1e-10), 'g-', linewidth=2)
    ax2.set_title(f"Top {num_top_subcarriers} Subcarriers Doppler", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Doppler Frequency (Hz)")
    ax2.set_ylabel("Power (dB)")
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)

    # 3. Phase Variance (Motion Activity Indicator)
    time_axis = np.arange(len(phase_variance)) / fs
    ax3.plot(time_axis, phase_variance, 'purple', linewidth=2)
    ax3.set_title("Phase Variance (Motion Activity)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Phase Variance (rad²)")
    ax3.grid(True, alpha=0.3)

    # 4. Spectrogram of coherent signal
    nperseg = min(64, max(8, len(coherent_detrended)//4))
    f_spec, t_spec, Sxx = spectrogram(coherent_detrended, fs=fs, nperseg=nperseg)
    im = ax4.pcolormesh(t_spec, f_spec, 10*np.log10(Sxx + 1e-10), cmap=cmap)
    ax4.set_title("Coherent Signal Spectrogram", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax4, label="Power (dB)")

    plt.tight_layout()
    return fig

# --- Streamlit App ---
st.set_page_config("📶 CSI Stream Analyzer", layout="wide")
st.title("📶 Enhanced CSI Stream Analyzer")

# Sidebar controls
st.sidebar.header("⚙️ Visualization Controls")

# Quick Presets
st.sidebar.markdown("### 🎬 Quick Presets")
preset = st.sidebar.radio("", ["Custom", "🐌 Smooth", "⚡ Normal", "🚀 Fast"],
                          horizontal=False, label_visibility="collapsed")

# Apply preset values
if preset == "🐌 Smooth":
    frame_size = 60
    step_size = 1
    shifting_speed = 30
    st.sidebar.info("Smooth mode: Small steps (1) for fluid transitions")
elif preset == "⚡ Normal":
    frame_size = 60
    step_size = 5
    shifting_speed = 50
    st.sidebar.info("Balanced mode: Moderate steps (5) for general viewing")
elif preset == "🚀 Fast":
    frame_size = 40
    step_size = 15
    shifting_speed = 80
    st.sidebar.info("Fast scan: Large steps (15) for quick navigation")
else:  # Custom
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Window Settings")

    # Frame size control (30-120)
    frame_size = st.sidebar.slider("🎞️ Frame Size",
                                    min_value=30, max_value=120, value=60, step=5,
                                    help="Number of timestamp frames to display per view")

    # Step size control for smooth shifting
    step_size = st.sidebar.slider("👣 Step Size",
                                   min_value=1, max_value=30, value=3, step=1,
                                   help="Timestamps to shift per step (1=smoothest, 30=fastest)")

    st.sidebar.markdown("### ⚡ Animation Settings")

    # Shifting speed control (25-100)
    shifting_speed = st.sidebar.slider("🏃 Speed (FPS)",
                                        min_value=25, max_value=100, value=50, step=5,
                                        help="Frames per second when auto-advancing")

# Colormap selection
cmap = st.sidebar.selectbox("🎨 Color Map", CMAPS, index=0)

# Heatmap settings
st.sidebar.markdown("---")
st.sidebar.subheader("🔥 Heatmap Settings")
heatmap_interpolation = st.sidebar.selectbox("Interpolation",
                                              ["bilinear", "nearest", "bicubic", "gaussian"],
                                              index=0)

# File selection
files = sorted(glob.glob(os.path.join(DATA_DIR, "esp32_csi_*.csv")))
if not files:
    st.error("No CSI CSV files found in ./data/")
    st.stop()

selected_file = st.selectbox("📂 Select CSI File", files)

# Session state tracking
if "last_file" not in st.session_state:
    st.session_state.last_file = selected_file
if selected_file != st.session_state.last_file:
    st.session_state.row_index = 0
    st.session_state.last_file = selected_file

# Load and pivot CSI file
with st.spinner("Loading CSI data..."):
    df = pd.read_csv(selected_file)
    initial_rows = len(df)

    # Check which columns are present
    required_cols = ["timestamp", "subcarrier_index", "I", "Q", "magnitude"]
    optional_cols = ["phase_deg", "rssi", "mac_address"]

    # Verify required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.info(f"Available columns: {list(df.columns)}")
        st.stop()

    # Convert numeric columns
    df["I"] = pd.to_numeric(df["I"], errors="coerce")
    df["Q"] = pd.to_numeric(df["Q"], errors="coerce")
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["subcarrier_index"] = pd.to_numeric(df["subcarrier_index"], errors="coerce")

    # Convert optional columns if present
    if "rssi" in df.columns:
        df["rssi"] = pd.to_numeric(df["rssi"], errors="coerce")
    if "phase_deg" in df.columns:
        df["phase_deg"] = pd.to_numeric(df["phase_deg"], errors="coerce")

    # Remove rows with NaN values in required columns only
    df = df.dropna(subset=required_cols)
    st.sidebar.markdown(f"**Data rows:** {initial_rows:,} → {len(df):,}")

    # Ensure proper data types
    df["timestamp"] = df["timestamp"].astype(float)
    df["subcarrier_index"] = df["subcarrier_index"].astype(int)
    df["magnitude"] = df["magnitude"].astype(float)
    df["I"] = df["I"].astype(float)
    df["Q"] = df["Q"].astype(float)

    # Sample data if too large
    if len(df) > 100000:
        sample_rate = max(2, len(df) // 100000)
        df = df.iloc[::sample_rate]
        st.sidebar.warning(f"⚠️ Sampled (1/{sample_rate} rows)")

try:
    with st.spinner("Creating pivot tables..."):
        mag_pivot = df.pivot_table(index="timestamp", columns="subcarrier_index", values="magnitude", aggfunc='mean')
        I_pivot = df.pivot_table(index="timestamp", columns="subcarrier_index", values="I", aggfunc='mean')
        Q_pivot = df.pivot_table(index="timestamp", columns="subcarrier_index", values="Q", aggfunc='mean')
        complex_pivot = I_pivot + 1j * Q_pivot

except Exception as e:
    st.error(f"Pivot operation failed. Error: {str(e)[:100]}")
    st.stop()

mag_pivot = mag_pivot.sort_index()
complex_pivot = complex_pivot.sort_index()

# Fill NaN values with 0
mag_pivot = mag_pivot.fillna(0)
complex_pivot = complex_pivot.fillna(0)

if mag_pivot.empty or complex_pivot.empty:
    st.error("CSV has no usable CSI data.")
    st.stop()

num_subcarriers = mag_pivot.shape[1]
total_timestamps = len(mag_pivot)

st.sidebar.markdown(f"**Total Timestamps:** {total_timestamps:,}")
st.sidebar.markdown(f"**Subcarriers:** {num_subcarriers}")

# Calculate global color scale for consistent heatmaps (prevent blinking)
# Cache in session state to avoid recalculation
if "global_mag_vmin" not in st.session_state or st.session_state.last_file != selected_file:
    st.session_state.global_mag_vmin = np.percentile(mag_pivot.values, 2)
    st.session_state.global_mag_vmax = np.percentile(mag_pivot.values, 98)

global_mag_vmin = st.session_state.global_mag_vmin
global_mag_vmax = st.session_state.global_mag_vmax

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Heatmap Scaling")
use_global_scale = st.sidebar.checkbox("Fixed Color Scale", value=True,
                                       help="Use global min/max to prevent color blinking between frames")

# Advanced scaling options
if use_global_scale:
    st.sidebar.info(f"📊 Range: {global_mag_vmin:.2f} - {global_mag_vmax:.2f}")

    # Option to adjust percentiles
    with st.sidebar.expander("🔧 Advanced Scaling"):
        lower_percentile = st.slider("Lower Percentile", 0, 10, 2, 1,
                                     help="Adjust lower bound (default: 2nd percentile)")
        upper_percentile = st.slider("Upper Percentile", 90, 100, 98, 1,
                                     help="Adjust upper bound (default: 98th percentile)")

        if st.button("Recalculate Scale"):
            st.session_state.global_mag_vmin = np.percentile(mag_pivot.values, lower_percentile)
            st.session_state.global_mag_vmax = np.percentile(mag_pivot.values, upper_percentile)
            st.rerun()

        global_mag_vmin = st.session_state.global_mag_vmin
        global_mag_vmax = st.session_state.global_mag_vmax

# Adaptive frame size
if total_timestamps < frame_size:
    st.warning(f"⚠️ Only {total_timestamps} timestamps available (requested {frame_size})")
    frame_size = max(5, total_timestamps)

if "row_index" not in st.session_state:
    st.session_state.row_index = 0

start_idx = st.session_state.row_index
end_idx = min(start_idx + frame_size, total_timestamps)

if end_idx >= total_timestamps:
    st.info("🔄 Reached end of data — restarting from beginning.")
    st.session_state.row_index = 0
    time.sleep(0.5)
    st.rerun()

# Extract window data
mag_window = mag_pivot.iloc[start_idx:end_idx]
complex_window = complex_pivot.iloc[start_idx:end_idx]

# Flatten signal for analysis
signal = mag_window.values.flatten()

if np.isnan(signal).any():
    st.error("NaN values found in signal — check data integrity.")
    st.stop()

# Display metadata
col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
with col_meta1:
    st.metric("Frame Range", f"{start_idx + 1} - {end_idx}")
with col_meta2:
    st.metric("Samples in Window", f"{len(signal):,}")
with col_meta3:
    duration = (mag_window.index[-1] - mag_window.index[0])
    st.metric("Window Duration", f"{duration:.2f}s")
with col_meta4:
    overlap_percent = ((frame_size - step_size) / frame_size * 100) if frame_size > 0 else 0
    st.metric("Frame Overlap", f"{overlap_percent:.0f}%")

# Extract RSSI data for the current window
rssi_data_for_plot = None
if "rssi" in df.columns:
    window_df = df[(df["timestamp"] >= mag_window.index[0]) & (df["timestamp"] <= mag_window.index[-1])]
    if len(window_df) > 0:
        avg_rssi = window_df["rssi"].mean()
        st.sidebar.metric("Avg RSSI", f"{avg_rssi:.1f} dBm")

        # Prepare RSSI data for plotting (one value per timestamp)
        rssi_per_timestamp = window_df.groupby("timestamp")["rssi"].first()
        rssi_data_for_plot = rssi_per_timestamp.reindex(mag_window.index).ffill().values

# Display MAC address if available
if "mac_address" in df.columns:
    window_df = df[(df["timestamp"] >= mag_window.index[0]) & (df["timestamp"] <= mag_window.index[-1])]
    if len(window_df) > 0:
        mac_addrs = window_df["mac_address"].unique()
        if len(mac_addrs) == 1:
            st.sidebar.info(f"**MAC:** {mac_addrs[0]}")

# Progress indicator
progress = (st.session_state.row_index / max(1, total_timestamps - frame_size)) if total_timestamps > frame_size else 0
st.sidebar.markdown(f"**Progress:** {st.session_state.row_index}/{total_timestamps - frame_size}")
st.sidebar.progress(min(1.0, max(0.0, progress)))

# === VISUALIZATIONS ===
st.markdown("---")
st.markdown("## 📊 Enhanced Waveform Analysis")
st.pyplot(draw_enhanced_waveforms(signal, complex_window.values, fs=FS, cmap=cmap, rssi_data=rssi_data_for_plot))

st.markdown("---")
st.markdown("## 📈 Time-Frequency Spectrogram")
st.pyplot(draw_spectrogram(signal, fs=FS, cmap=cmap))

st.markdown("---")
st.markdown("## 🔥 CSI Magnitude Heatmap")
# Apply global color scaling if enabled to prevent blinking
mag_vmin = global_mag_vmin if use_global_scale else None
mag_vmax = global_mag_vmax if use_global_scale else None
st.pyplot(draw_enhanced_heatmap(mag_window.T.values, cmap=cmap,
                                 title="Subcarrier vs Time (Magnitude)",
                                 interpolation=heatmap_interpolation,
                                 vmin=mag_vmin, vmax=mag_vmax))

st.markdown("---")
st.markdown("## 🌊 Phase Analysis")
st.pyplot(draw_phase_analysis(complex_window.values, cmap=cmap))

st.markdown("---")
st.markdown("## 🔁 Doppler Analysis")
# Calculate effective sampling rate based on timestamps
actual_fs = len(mag_window) / max(0.001, (mag_window.index[-1] - mag_window.index[0]))
st.pyplot(draw_doppler_analysis(complex_window.values, cmap=cmap, fs=actual_fs))

# --- Navigation Controls ---
st.markdown("---")
auto_advance = st.sidebar.checkbox("🔄 Auto-advance", value=False)

if auto_advance:
    # Advance by step_size for smoother transitions
    st.session_state.row_index += step_size
    # Clamp to valid range
    st.session_state.row_index = min(st.session_state.row_index, total_timestamps - frame_size)
    time.sleep(1 / shifting_speed)
    st.rerun()
else:
    col_prev, col_reset, col_next = st.columns([1, 1, 1])
    with col_prev:
        if st.button("⬅️ Previous", use_container_width=True) and st.session_state.row_index > 0:
            st.session_state.row_index = max(0, st.session_state.row_index - step_size)
            st.rerun()
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.row_index = 0
            st.rerun()
    with col_next:
        if st.button("➡️ Next", use_container_width=True) and end_idx < total_timestamps:
            st.session_state.row_index = min(st.session_state.row_index + step_size,
                                              total_timestamps - frame_size)
            st.rerun()
