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
TARGET_LENGTH = int(FS * 30)  # 6000 samples over 30 seconds
CMAPS = ["viridis", "jet", "turbo"]

# --- Helper Functions ---
def compute_attention(seq, downsample=20):
    # Downsample to reduce computation (e.g., from 6000 to 300 samples)
    seq_downsampled = seq[::downsample]
    x = torch.tensor(seq_downsampled, dtype=torch.float32).unsqueeze(-1)
    scores = torch.matmul(x, x.T) / np.sqrt(x.shape[-1])
    return F.softmax(scores, dim=-1).detach().numpy()

def draw_spectrogram(signal, fs=FS, cmap="viridis"):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=128, noverlap=64)
    Sxx_smooth = gaussian_filter(Sxx, sigma=1)

    fig, ax = plt.subplots(figsize=(8, 3))
    pcm = ax.pcolormesh(t, f, Sxx_smooth, shading="gouraud", cmap=cmap)
    ax.set_title("Spectrogram (CSI Magnitude)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, fs / 2)  # Nyquist limit
    ax.set_xlim(0, len(signal) / fs)
    fig.colorbar(pcm, ax=ax, label="Intensity")
    return fig

def draw_attention(attn_matrix, cmap):
    fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.imshow(attn_matrix, cmap=cmap)
    ax.set_title("Attention Matrix")
    fig.colorbar(cax)
    return fig

def draw_heatmap(matrix, cmap, title="CSI Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 4))
    cax = ax.imshow(matrix, aspect='auto', cmap=cmap, origin='lower')
    ax.set_title(title)
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Subcarrier Index")
    fig.colorbar(cax, ax=ax, label="Magnitude")
    return fig

def draw_doppler_analysis(complex_matrix, cmap, fs=30.0):
    """Enhanced Doppler analysis with multiple insights"""
    
    # Method 1: Coherent averaging across subcarriers
    # Weight by magnitude to emphasize strong subcarriers
    magnitudes = np.abs(complex_matrix)
    weights = magnitudes / (np.sum(magnitudes, axis=1, keepdims=True) + 1e-10)
    coherent_signal = np.sum(complex_matrix * weights, axis=1)
    coherent_phase = np.unwrap(np.angle(coherent_signal))
    
    # Method 2: Select best subcarriers (highest average magnitude)
    avg_mag_per_subcarrier = np.mean(magnitudes, axis=0)
    num_top_subcarriers = min(5, complex_matrix.shape[1])  # Top N subcarriers (max 5)
    best_subcarriers = np.argsort(avg_mag_per_subcarrier)[-num_top_subcarriers:]
    selected_signal = np.mean(complex_matrix[:, best_subcarriers], axis=1)
    selected_phase = np.unwrap(np.angle(selected_signal))
    
    # Method 3: Differential phase analysis (subcarrier diversity)
    phase_matrix = np.unwrap(np.angle(complex_matrix), axis=0)
    phase_variance = np.var(phase_matrix, axis=1)  # Variance across subcarriers
    
    # Create subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
    
    # 1. Coherent Doppler Spectrum
    coherent_detrended = detrend(coherent_phase)
    window = np.hanning(len(coherent_detrended))
    windowed = coherent_detrended * window
    fft_coherent = np.fft.fftshift(np.fft.fft(windowed))
    power_coherent = np.abs(fft_coherent) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(len(coherent_phase), 1/fs))
    
    ax1.plot(freqs, 10*np.log10(power_coherent + 1e-10), 'b-', linewidth=2)
    ax1.set_title("Coherent Doppler Spectrum (Magnitude-Weighted)")
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
    ax2.set_title("Top 5 Subcarriers Doppler Spectrum")
    ax2.set_xlabel("Doppler Frequency (Hz)")
    ax2.set_ylabel("Power (dB)")
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # 3. Phase Variance (Motion Activity Indicator)
    time_axis = np.arange(len(phase_variance)) / fs
    ax3.plot(time_axis, phase_variance, 'purple', linewidth=2)
    ax3.set_title("Phase Variance Across Subcarriers (Motion Activity)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Phase Variance (rad²)")
    ax3.grid(True, alpha=0.3)
    
    # 4. Spectrogram of coherent signal
    nperseg = min(64, max(8, len(coherent_detrended)//4))  # Ensure nperseg >= 8
    f_spec, t_spec, Sxx = spectrogram(coherent_detrended, fs=fs, nperseg=nperseg)
    im = ax4.pcolormesh(t_spec, f_spec, 10*np.log10(Sxx + 1e-10), cmap=cmap)
    ax4.set_title("Coherent Signal Spectrogram")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax4, label="Power (dB)")
    
    plt.tight_layout()
    return fig

# --- Streamlit App ---
st.set_page_config("📶 CSI Stream Viewer", layout="wide")
st.title("📶 CSI 10-Second Stream (Multi-Subcarrier View)")
st.markdown(f"**Sampling Frequency:** `{FS} Hz` | **Target Samples:** `{TARGET_LENGTH}` magnitudes")

files = sorted(glob.glob(os.path.join(DATA_DIR, "esp32_csi_*.csv")))
if not files:
    st.error("No CSI CSV files found in ./data/")
    st.stop()

selected_file = st.selectbox("Select CSI File", files)

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

    df["I"] = pd.to_numeric(df["I"], errors="coerce")
    df["Q"] = pd.to_numeric(df["Q"], errors="coerce")
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    # Remove rows with NaN values to reduce memory usage
    df = df.dropna()
    st.sidebar.markdown(f"**Data rows:** {initial_rows:,} → {len(df):,} (after cleaning)")

    # Ensure proper data types
    df["timestamp"] = df["timestamp"].astype(float)
    df["subcarrier_index"] = df["subcarrier_index"].astype(int)
    df["magnitude"] = df["magnitude"].astype(float)
    df["I"] = df["I"].astype(float)
    df["Q"] = df["Q"].astype(float)

    # Sample data if too large (keep every Nth row to reduce memory usage)
    if len(df) > 100000:
        sample_rate = max(2, len(df) // 100000)
        df = df.iloc[::sample_rate]
        st.sidebar.warning(f"⚠️ Large dataset sampled (1/{sample_rate} rows kept)")

try:
    with st.spinner("Creating pivot tables..."):
        # Pivot magnitude, I, and Q separately to avoid complex number issues
        mag_pivot = df.pivot_table(index="timestamp", columns="subcarrier_index", values="magnitude", aggfunc='mean')
        I_pivot = df.pivot_table(index="timestamp", columns="subcarrier_index", values="I", aggfunc='mean')
        Q_pivot = df.pivot_table(index="timestamp", columns="subcarrier_index", values="Q", aggfunc='mean')

        # Reconstruct complex values after pivoting
        complex_pivot = I_pivot + 1j * Q_pivot

except Exception as e:
    st.error(f"Pivot operation failed. Error: {str(e)[:100]}")
    st.stop()

mag_pivot = mag_pivot.sort_index().dropna()
complex_pivot = complex_pivot.sort_index().dropna()

if mag_pivot.empty or complex_pivot.empty:
    st.error("CSV has no usable CSI data.")
    st.stop()

samples_per_row = mag_pivot.shape[1]
rows_needed = int(np.ceil(TARGET_LENGTH / samples_per_row))
total_rows = len(mag_pivot)

# Validate sufficient rows available
if total_rows < rows_needed:
    st.error(f"Not enough rows in CSV. Need {rows_needed} rows, but only {total_rows} available.")
    st.stop()

cmap = st.sidebar.selectbox("🎨 Spectrogram Color Map", CMAPS)

if "row_index" not in st.session_state:
    st.session_state.row_index = 0

start_idx = st.session_state.row_index
end_idx = start_idx + rows_needed

if end_idx > total_rows:
    st.warning("Reached end of data — restarting.")
    st.session_state.row_index = 0
    time.sleep(0.5)  # Brief pause before restart
    st.rerun()

mag_window = mag_pivot.iloc[start_idx:end_idx]
complex_window = complex_pivot.iloc[start_idx:end_idx]
signal = mag_window.values.flatten()[:TARGET_LENGTH]

if np.isnan(signal).any():
    st.error("NaN values found in signal — check data integrity.")
    st.stop()

st.markdown(f"**Rows:** {start_idx + 1} to {end_idx} | **Subcarriers:** `{samples_per_row}`")

# Progress indicator
progress = (st.session_state.row_index / max(1, total_rows - rows_needed)) if total_rows > rows_needed else 0
st.sidebar.markdown(f"**Progress:** {st.session_state.row_index}/{total_rows - rows_needed} ({progress*100:.1f}%)")
st.sidebar.progress(min(1.0, progress))

col1, col2 = st.columns(2)

with col1:
    st.pyplot(draw_spectrogram(signal, fs=FS, cmap=cmap))

with col2:
    attn = compute_attention(signal)
    st.pyplot(draw_attention(attn, cmap=cmap))

st.markdown("### 🔍 CSI Subcarrier Magnitude Heatmap")
st.pyplot(draw_heatmap(mag_window.T.values, cmap=cmap, title="Subcarrier vs Time (Magnitude)"))

st.markdown("### 🔁 Enhanced Doppler Analysis")
st.pyplot(draw_doppler_analysis(complex_window.values, cmap=cmap))

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