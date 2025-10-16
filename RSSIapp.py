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
TARGET_LENGTH = int(FS * 10)  # 2000 samples for 10s
CMAPS = ["viridis", "jet", "turbo"]

# --- Functions ---
def sorted_pkt_cols(df):
    pkt_cols = [c for c in df.columns if c.startswith("pkt")]
    return sorted(pkt_cols, key=lambda x: int(x[3:]))

def compute_attention(seq, downsample=10):
    # Downsample to reduce computation (e.g., from 2000 to 200 samples)
    seq_downsampled = seq[::downsample]
    x = torch.tensor(seq_downsampled, dtype=torch.float32).unsqueeze(-1)
    scores = torch.matmul(x, x.T) / np.sqrt(x.shape[-1])
    return F.softmax(scores, dim=-1).detach().numpy()

def draw_waveform(signal, fs=200):
    fig, ax = plt.subplots(figsize=(8, 2))
    t = np.linspace(0, len(signal)/fs, num=len(signal))
    ax.plot(t, signal, linewidth=1)
    ax.set_title("Waveform (RSSI Signal)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    return fig

def draw_spectrogram(signal, fs=200, cmap="viridis", ylim=(0, 100)):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=128, noverlap=64)
    Sxx_smooth = gaussian_filter(Sxx, sigma=1)

    fig, ax = plt.subplots(figsize=(8, 3))
    pcm = ax.pcolormesh(t, f, Sxx_smooth, shading="gouraud", cmap=cmap)
    ax.set_title("Spectrogram")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(*ylim)
    ax.set_xlim(0, len(signal) / fs)
    fig.colorbar(pcm, ax=ax, label="Intensity")
    return fig

def draw_attention(attn_matrix):
    fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.imshow(attn_matrix, cmap='viridis')
    ax.set_title("Attention Matrix")
    fig.colorbar(cax)
    return fig

def draw_motion_analysis(signal, fs=200, cmap="viridis"):
    """Enhanced motion analysis for RSSI signals"""
    
    # Detrend signal
    signal_detrended = detrend(signal)
    
    # Compute analytic signal for instantaneous frequency analysis
    analytic_signal = hilbert(signal_detrended)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
    
    # Motion activity indicator (signal variance in sliding windows)
    window_size = int(fs * 0.5)  # 0.5 second windows
    stride = window_size // 4
    activity_indicator = []
    time_windows = []
    
    for i in range(0, len(signal) - window_size, stride):
        window = signal[i:i + window_size]
        activity_indicator.append(np.var(window))
        time_windows.append((i + window_size//2) / fs)
    
    # Doppler-like analysis (FFT of signal variations)
    window = np.hanning(len(signal_detrended))
    windowed_signal = signal_detrended * window
    fft_result = np.fft.fftshift(np.fft.fft(windowed_signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/fs))
    power_spectrum = np.abs(fft_result) ** 2
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
    
    # 1. Motion Activity Indicator
    ax1.plot(time_windows, activity_indicator, 'purple', linewidth=2)
    ax1.set_title("Motion Activity (Signal Variance)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Variance")
    ax1.grid(True, alpha=0.3)
    
    # 2. Instantaneous Frequency
    time_axis = np.arange(len(instantaneous_freq)) / fs
    ax2.plot(time_axis, instantaneous_freq, 'orange', linewidth=1)
    ax2.set_title("Instantaneous Frequency")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-50, 50)  # Focus on relevant frequency range
    
    # 3. Power Spectrum (Motion Detection)
    ax3.plot(freqs, 10*np.log10(power_spectrum + 1e-10), 'blue', linewidth=2)
    ax3.set_title("Power Spectrum (Motion Detection)")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Power (dB)")
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlim(-10, 10)  # Focus on motion-relevant frequencies
    
    # 4. Spectrogram with motion overlay
    f_spec, t_spec, Sxx = spectrogram(signal_detrended, fs=fs, nperseg=64, noverlap=32)
    im = ax4.pcolormesh(t_spec, f_spec, 10*np.log10(Sxx + 1e-10), cmap=cmap)
    ax4.set_title("Spectrogram with Motion Events")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Frequency (Hz)")
    ax4.set_ylim(0, 20)  # Focus on low frequencies for motion
    
    # Overlay motion events (high activity periods)
    activity_threshold = np.percentile(activity_indicator, 75)
    motion_times = [t for t, a in zip(time_windows, activity_indicator) if a > activity_threshold]
    for t in motion_times:
        ax4.axvline(x=t, color='red', alpha=0.3, linewidth=2)
    
    fig.colorbar(im, ax=ax4, label="Power (dB)")
    plt.tight_layout()
    return fig

# --- UI ---
st.set_page_config("📡 RSSI Stream", layout="wide")
st.title("📡 RSSI 10-Second Stream (Validated)")
st.markdown(f"**Sampling Frequency:** `{FS} Hz` | **Target Signal Length:** `{TARGET_LENGTH}` samples")

files = sorted(glob.glob(os.path.join(DATA_DIR, "wifisignal_data_*.csv")))
if not files:
    st.error("No CSV files found in ./data/")
    st.stop()

selected_file = st.selectbox("Select File", files)

# Reset if file changes
if "last_file" not in st.session_state:
    st.session_state.last_file = selected_file
if selected_file != st.session_state.last_file:
    st.session_state.row_index = 0
    st.session_state.last_file = selected_file

# Load CSV
df = pd.read_csv(selected_file)
pkt_cols = sorted_pkt_cols(df)
df[pkt_cols] = df[pkt_cols].apply(pd.to_numeric, errors="coerce")
samples_per_row = len(pkt_cols)
rows_needed = int(np.ceil(TARGET_LENGTH / samples_per_row))
total_rows = len(df)

# Validate sufficient rows available
if total_rows < rows_needed:
    st.error(f"Not enough rows in CSV. Need {rows_needed} rows, but only {total_rows} available.")
    st.stop()

cmap = st.sidebar.selectbox("🎨 Spectrogram Color Map", CMAPS)
fmin, fmax = st.sidebar.slider("Spectrogram Y-Axis (Hz)", 0, int(FS//2), (0, 100))

# Row index state
if "row_index" not in st.session_state:
    st.session_state.row_index = 0

start_idx = st.session_state.row_index
end_idx = start_idx + rows_needed

# Restart when not enough rows
if end_idx > total_rows:
    st.warning("Reached end of data — restarting.")
    st.session_state.row_index = 0
    time.sleep(0.5)  # Brief pause before restart
    st.rerun()

# Get signal block
window_df = df.iloc[start_idx:end_idx]
signal_matrix = window_df[pkt_cols].astype(float).values
signal = signal_matrix.flatten()[:TARGET_LENGTH]

# --- Validation ---
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

# Sidebar info
st.sidebar.markdown(f"**Selected File:** `{os.path.basename(selected_file)}`")
st.sidebar.markdown(f"**Label:** `{label}`")

# --- Plotting ---
st.markdown(f"**Rows:** {start_idx + 1} to {end_idx} | **Label:** `{label}`")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(draw_waveform(signal, fs=FS))
    st.pyplot(draw_spectrogram(signal, fs=FS, cmap=cmap, ylim=(fmin, fmax)))

with col2:
    attn = compute_attention(signal)
    st.pyplot(draw_attention(attn))

# --- Enhanced Motion Analysis ---
st.markdown("### 🏃 Enhanced Motion Analysis")
st.pyplot(draw_motion_analysis(signal, fs=FS, cmap=cmap))

# --- Auto-advance control
auto_advance = st.sidebar.checkbox("🔄 Auto-advance frames", value=False)
advance_fps = st.sidebar.slider("Auto-advance FPS", 1, 30, 10) if auto_advance else 10

if auto_advance:
    st.session_state.row_index += 1
    time.sleep(1 / advance_fps)
    st.rerun()
else:
    # Manual controls
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("⬅️ Previous") and st.session_state.row_index > 0:
            st.session_state.row_index -= 1
            st.rerun()
    with col_next:
        if st.button("➡️ Next") and end_idx < total_rows:
            st.session_state.row_index += 1
            st.rerun()