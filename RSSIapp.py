import streamlit as st
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, detrend, hilbert
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
import time

DATA_DIR = "./data"
FS = 200.0
CMAPS = ["viridis", "jet", "turbo", "plasma", "inferno", "magma", "cividis", "RdYlBu_r"]

# --- Helper Functions ---
def sorted_pkt_cols(df):
    """Get sorted packet column names"""
    pkt_cols = [c for c in df.columns if c.startswith("pkt")]
    return sorted(pkt_cols, key=lambda x: int(x[3:]))

def compute_attention(seq, downsample=10):
    """Compute attention matrix for signal sequence"""
    if len(seq) < downsample:
        downsample = max(1, len(seq) // 10)
    seq_downsampled = seq[::downsample]
    x = torch.tensor(seq_downsampled, dtype=torch.float32).unsqueeze(-1)
    scores = torch.matmul(x, x.T) / np.sqrt(x.shape[-1])
    return F.softmax(scores, dim=-1).detach().numpy()

def analyze_packet_header(packet_row):
    """Extract insights from PCAP/WiFi packet header bytes"""
    analysis = {}

    # Convert to numpy array
    pkt_data = packet_row.values if hasattr(packet_row, 'values') else np.array(packet_row)

    # Basic statistics
    analysis['mean'] = np.mean(pkt_data)
    analysis['std'] = np.std(pkt_data)
    analysis['min'] = np.min(pkt_data)
    analysis['max'] = np.max(pkt_data)
    analysis['range'] = analysis['max'] - analysis['min']

    # Entropy (information content)
    pkt_hist, _ = np.histogram(pkt_data, bins=256, range=(0, 256))
    pkt_prob = pkt_hist / np.sum(pkt_hist)
    analysis['entropy'] = -np.sum(pkt_prob[pkt_prob > 0] * np.log2(pkt_prob[pkt_prob > 0]))

    # Zero byte ratio (padding indicator)
    analysis['zero_ratio'] = np.sum(pkt_data == 0) / len(pkt_data)

    # Pattern detection (repeating values)
    unique_ratio = len(np.unique(pkt_data)) / len(pkt_data)
    analysis['uniqueness'] = unique_ratio

    # ASCII printable characters ratio (for frame type detection)
    printable_count = np.sum((pkt_data >= 32) & (pkt_data <= 126))
    analysis['ascii_ratio'] = printable_count / len(pkt_data)

    return analysis

def draw_packet_analysis(window_df, pkt_cols):
    """Visualize PCAP header patterns across window"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    # Extract packet data matrix
    pkt_matrix = window_df[pkt_cols].values

    # 1. Packet Heatmap (visualize raw bytes)
    im1 = ax1.imshow(pkt_matrix.T, aspect='auto', cmap='viridis', origin='lower',
                     interpolation='nearest', vmin=0, vmax=255)
    ax1.set_title("Packet Byte Heatmap (Raw PCAP Data)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time Frame")
    ax1.set_ylabel("Byte Position")
    ax1.set_ylim(0, len(pkt_cols))
    fig.colorbar(im1, ax=ax1, label="Byte Value")

    # 2. Byte Statistics Over Time
    means = np.mean(pkt_matrix, axis=1)
    stds = np.std(pkt_matrix, axis=1)

    ax2_twin = ax2.twinx()
    ax2.plot(means, 'b-', linewidth=2, label='Mean', alpha=0.8)
    ax2.fill_between(range(len(means)), means - stds, means + stds, alpha=0.3, color='blue')
    ax2_twin.plot(stds, 'r-', linewidth=2, label='Std Dev', alpha=0.8)

    ax2.set_title("Packet Statistics Over Time", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time Frame")
    ax2.set_ylabel("Mean Byte Value", color='b')
    ax2_twin.set_ylabel("Std Deviation", color='r')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    # 3. Entropy and Information Content
    entropies = []
    zero_ratios = []
    for i in range(len(window_df)):
        analysis = analyze_packet_header(pkt_matrix[i])
        entropies.append(analysis['entropy'])
        zero_ratios.append(analysis['zero_ratio'])

    ax3_twin = ax3.twinx()
    ax3.plot(entropies, 'g-', linewidth=2, label='Entropy')
    ax3_twin.plot(zero_ratios, 'orange', linewidth=2, label='Zero Ratio')

    ax3.set_title("Packet Entropy & Padding Analysis", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Time Frame")
    ax3.set_ylabel("Entropy (bits)", color='g')
    ax3_twin.set_ylabel("Zero Byte Ratio", color='orange')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')

    # 4. Byte Value Distribution
    all_bytes = pkt_matrix.flatten()
    ax4.hist(all_bytes, bins=64, range=(0, 256), color='purple', alpha=0.7, edgecolor='black')
    ax4.set_title("Byte Value Distribution (All Packets)", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Byte Value")
    ax4.set_ylabel("Frequency")
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=np.mean(all_bytes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_bytes):.1f}')
    ax4.legend()

    plt.tight_layout()
    return fig

def draw_enhanced_waveform(signal, fs=200):
    """Enhanced waveform with multiple views"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))

    t = np.linspace(0, len(signal)/fs, num=len(signal))

    # 1. Raw signal with envelope
    ax1.plot(t, signal, 'b-', linewidth=1, alpha=0.7, label='RSSI Signal')

    # Compute envelope using Hilbert transform
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    ax1.plot(t, envelope, 'r-', linewidth=2, alpha=0.8, label='Envelope')
    ax1.fill_between(t, signal, alpha=0.2)

    ax1.set_title("RSSI Waveform with Envelope", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Moving statistics
    window_size = max(10, len(signal) // 50)
    from scipy.ndimage import uniform_filter1d
    signal_smooth = uniform_filter1d(signal, size=window_size)

    ax2.plot(t, signal, 'lightblue', linewidth=0.8, alpha=0.5, label='Raw')
    ax2.plot(t, signal_smooth, 'darkblue', linewidth=2, label='Moving Avg')
    ax2.set_title("Signal with Moving Average", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def draw_spectrogram(signal, fs=200, cmap="viridis", ylim=(0, 100), vmin=None, vmax=None):
    """Enhanced spectrogram with fixed color scaling"""
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=min(128, len(signal)//2), noverlap=64)
    Sxx_smooth = gaussian_filter(Sxx, sigma=1)

    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx_smooth + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 4))
    pcm = ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("Time-Frequency Spectrogram", fontsize=13, fontweight='bold')
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylim(*ylim)
    ax.set_xlim(0, len(signal) / fs)
    cbar = fig.colorbar(pcm, ax=ax, label="Power (dB)")
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig

def draw_attention(attn_matrix, cmap='viridis'):
    """Attention matrix visualization"""
    fig, ax = plt.subplots(figsize=(5, 5))
    cax = ax.imshow(attn_matrix, cmap=cmap)
    ax.set_title("Attention Matrix", fontsize=12, fontweight='bold')
    fig.colorbar(cax, label='Attention Weight')
    plt.tight_layout()
    return fig

def draw_motion_analysis(signal, fs=200, cmap="viridis"):
    """Enhanced motion analysis for RSSI signals"""

    signal_detrended = detrend(signal)
    analytic_signal = hilbert(signal_detrended)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs

    # Motion activity indicator
    window_size = int(fs * 0.5)
    stride = window_size // 4
    activity_indicator = []
    time_windows = []

    for i in range(0, len(signal) - window_size, stride):
        window = signal[i:i + window_size]
        activity_indicator.append(np.var(window))
        time_windows.append((i + window_size//2) / fs)

    # Doppler-like analysis
    window = np.hanning(len(signal_detrended))
    windowed_signal = signal_detrended * window
    fft_result = np.fft.fftshift(np.fft.fft(windowed_signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/fs))
    power_spectrum = np.abs(fft_result) ** 2

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))

    # 1. Motion Activity
    ax1.plot(time_windows, activity_indicator, 'purple', linewidth=2)
    ax1.set_title("Motion Activity (Signal Variance)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Variance")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # 2. Instantaneous Frequency
    time_axis = np.arange(len(instantaneous_freq)) / fs
    ax2.plot(time_axis, instantaneous_freq, 'orange', linewidth=1)
    ax2.set_title("Instantaneous Frequency", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-50, 50)

    # 3. Power Spectrum
    ax3.plot(freqs, 10*np.log10(power_spectrum + 1e-10), 'blue', linewidth=2)
    ax3.set_title("Power Spectrum (Motion Detection)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Power (dB)")
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlim(-10, 10)

    # 4. Spectrogram with motion overlay
    nperseg = min(64, max(8, len(signal_detrended)//4))
    f_spec, t_spec, Sxx = spectrogram(signal_detrended, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    im = ax4.pcolormesh(t_spec, f_spec, 10*np.log10(Sxx + 1e-10), cmap=cmap)
    ax4.set_title("Spectrogram with Motion Events", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Frequency (Hz)")
    ax4.set_ylim(0, 20)

    # Overlay motion events
    if len(activity_indicator) > 0:
        activity_threshold = np.percentile(activity_indicator, 75)
        motion_times = [t for t, a in zip(time_windows, activity_indicator) if a > activity_threshold]
        for t in motion_times:
            ax4.axvline(x=t, color='red', alpha=0.3, linewidth=2)

    fig.colorbar(im, ax=ax4, label="Power (dB)")
    plt.tight_layout()
    return fig

# --- Streamlit App ---
st.set_page_config("📡 RSSI Stream Analyzer", layout="wide")
st.title("📡 Enhanced RSSI Stream Analyzer")

# Sidebar controls
st.sidebar.header("⚙️ Visualization Controls")

# Quick Presets
st.sidebar.markdown("### 🎬 Quick Presets")
preset = st.sidebar.radio("", ["Custom", "🐌 Smooth", "⚡ Normal", "🚀 Fast"],
                          horizontal=False, label_visibility="collapsed")

# Apply preset values
if preset == "🐌 Smooth":
    frame_size = 10
    step_size = 1
    shifting_speed = 30
    st.sidebar.info("Smooth mode: Small steps (1) for fluid transitions")
elif preset == "⚡ Normal":
    frame_size = 10
    step_size = 3
    shifting_speed = 50
    st.sidebar.info("Balanced mode: Moderate steps (3) for general viewing")
elif preset == "🚀 Fast":
    frame_size = 8
    step_size = 5
    shifting_speed = 80
    st.sidebar.info("Fast scan: Large steps (5) for quick navigation")
else:  # Custom
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Window Settings")

    # Frame size control
    frame_size = st.sidebar.slider("🎞️ Frame Size (rows)",
                                    min_value=5, max_value=50, value=10, step=1,
                                    help="Number of rows to display per view")

    # Step size control
    step_size = st.sidebar.slider("👣 Step Size",
                                   min_value=1, max_value=20, value=3, step=1,
                                   help="Rows to shift per step (1=smoothest, 20=fastest)")

    st.sidebar.markdown("### ⚡ Animation Settings")

    # Shifting speed control (25-100)
    shifting_speed = st.sidebar.slider("🏃 Speed (FPS)",
                                        min_value=25, max_value=100, value=50, step=5,
                                        help="Frames per second when auto-advancing")

# Colormap selection
st.sidebar.markdown("---")
cmap = st.sidebar.selectbox("🎨 Color Map", CMAPS, index=0)

# Spectrogram settings
st.sidebar.markdown("### 📊 Spectrogram Settings")
fmin, fmax = st.sidebar.slider("Frequency Range (Hz)", 0, int(FS//2), (0, 100))

use_fixed_scale = st.sidebar.checkbox("Fixed Color Scale", value=True,
                                      help="Prevent color blinking between frames")

# File selection
files = sorted(glob.glob(os.path.join(DATA_DIR, "wifisignal_data_*.csv")))
if not files:
    st.error("No wifisignal CSV files found in ./data/")
    st.stop()

selected_file = st.selectbox("📂 Select File", files)

# Session state tracking
if "last_file" not in st.session_state:
    st.session_state.last_file = selected_file
if selected_file != st.session_state.last_file:
    st.session_state.row_index = 0
    st.session_state.last_file = selected_file

# Load CSV
with st.spinner("Loading data..."):
    df = pd.read_csv(selected_file)
    pkt_cols = sorted_pkt_cols(df)
    df[pkt_cols] = df[pkt_cols].apply(pd.to_numeric, errors="coerce")
    samples_per_row = len(pkt_cols)
    total_rows = len(df)

    st.sidebar.markdown(f"**Total Rows:** {total_rows:,}")
    st.sidebar.markdown(f"**Samples/Row:** {samples_per_row}")

    # Calculate signal stats for fixed scaling
    all_signals = df[pkt_cols].values.flatten()
    global_signal_min = np.percentile(all_signals, 1)
    global_signal_max = np.percentile(all_signals, 99)

# Validate sufficient rows
if total_rows < frame_size:
    st.warning(f"⚠️ Only {total_rows} rows available (requested {frame_size})")
    frame_size = max(3, total_rows)

if "row_index" not in st.session_state:
    st.session_state.row_index = 0

start_idx = st.session_state.row_index
end_idx = min(start_idx + frame_size, total_rows)

# Restart when not enough rows
if end_idx >= total_rows:
    st.info("🔄 Reached end of data — restarting from beginning.")
    st.session_state.row_index = 0
    time.sleep(0.5)
    st.rerun()

# Get signal block
window_df = df.iloc[start_idx:end_idx]
signal_matrix = window_df[pkt_cols].astype(float).values
target_length = frame_size * samples_per_row
signal = signal_matrix.flatten()[:target_length]

# Extract label
if "label" in window_df.columns:
    try:
        label_mode = window_df["label"].mode()
        label = label_mode[0] if len(label_mode) > 0 else "N/A"
    except:
        label = "N/A"
else:
    label = "N/A"

if np.isnan(signal).any():
    st.error("NaN values found in signal — check CSV integrity.")
    st.stop()

# Display metadata
col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
with col_meta1:
    st.metric("Row Range", f"{start_idx + 1} - {end_idx}")
with col_meta2:
    st.metric("Samples", f"{len(signal):,}")
with col_meta3:
    st.metric("Label", label)
with col_meta4:
    overlap_percent = ((frame_size - step_size) / frame_size * 100) if frame_size > 0 else 0
    st.metric("Frame Overlap", f"{overlap_percent:.0f}%")

# Progress indicator
progress = (st.session_state.row_index / max(1, total_rows - frame_size)) if total_rows > frame_size else 0
st.sidebar.markdown(f"**Progress:** {st.session_state.row_index}/{total_rows - frame_size}")
st.sidebar.progress(min(1.0, max(0.0, progress)))

# === VISUALIZATIONS ===
st.markdown("---")
st.markdown("## 📊 Enhanced Waveform Analysis")
st.pyplot(draw_enhanced_waveform(signal, fs=FS))

st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📈 Time-Frequency Spectrogram")
    if use_fixed_scale:
        # Calculate global spectrogram scale
        f_temp, t_temp, Sxx_temp = spectrogram(signal, fs=FS, nperseg=min(128, len(signal)//2), noverlap=64)
        Sxx_db_temp = 10 * np.log10(gaussian_filter(Sxx_temp, sigma=1) + 1e-10)
        spec_vmin = np.percentile(Sxx_db_temp, 5)
        spec_vmax = np.percentile(Sxx_db_temp, 95)
        st.pyplot(draw_spectrogram(signal, fs=FS, cmap=cmap, ylim=(fmin, fmax),
                                   vmin=spec_vmin, vmax=spec_vmax))
    else:
        st.pyplot(draw_spectrogram(signal, fs=FS, cmap=cmap, ylim=(fmin, fmax)))

with col2:
    st.markdown("### 🧠 Attention Matrix")
    attn = compute_attention(signal, downsample=max(5, len(signal)//200))
    st.pyplot(draw_attention(attn, cmap=cmap))

st.markdown("---")
st.markdown("## 📦 PCAP Header Analysis")
st.pyplot(draw_packet_analysis(window_df, pkt_cols))

st.markdown("---")
st.markdown("## 🏃 Enhanced Motion Analysis")
st.pyplot(draw_motion_analysis(signal, fs=FS, cmap=cmap))

# --- Navigation Controls ---
st.markdown("---")
auto_advance = st.sidebar.checkbox("🔄 Auto-advance", value=False)

if auto_advance:
    st.session_state.row_index += step_size
    st.session_state.row_index = min(st.session_state.row_index, total_rows - frame_size)
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
        if st.button("➡️ Next", use_container_width=True) and end_idx < total_rows:
            st.session_state.row_index = min(st.session_state.row_index + step_size,
                                              total_rows - frame_size)
            st.rerun()
