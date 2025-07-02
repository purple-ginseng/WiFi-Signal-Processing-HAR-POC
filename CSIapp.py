import streamlit as st
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
import time

DATA_DIR = "./data"
FS = 200.0  # Sampling rate per subcarrier
TARGET_LENGTH = int(FS * 10)  # 2000 samples over 10 seconds
CMAPS = ["viridis", "jet", "turbo"]

# --- Helper Functions ---
def compute_attention(seq):
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(-1)
    scores = torch.matmul(x, x.T) / np.sqrt(x.shape[0])
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

def draw_attention(attn_matrix):
    fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.imshow(attn_matrix, cmap='viridis')
    ax.set_title("Attention Matrix")
    fig.colorbar(cax)
    return fig

def draw_heatmap(matrix, title="CSI Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 4))
    cax = ax.imshow(matrix, aspect='auto', cmap='turbo', origin='lower')
    ax.set_title(title)
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Subcarrier Index")
    fig.colorbar(cax, ax=ax, label="Magnitude")
    return fig

def draw_doppler_phase_fft(complex_matrix):
    phase_signal = np.unwrap(np.angle(complex_matrix), axis=0)
    fft_matrix = np.fft.fftshift(np.fft.fft(phase_signal, axis=0), axes=0)
    power = np.abs(fft_matrix) ** 2

    fig, ax = plt.subplots(figsize=(10, 4))
    cax = ax.imshow(power.T, aspect='auto', cmap='plasma', origin='lower')
    ax.set_title("Doppler View (Phase FFT per Subcarrier)")
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Subcarrier Index")
    fig.colorbar(cax, ax=ax, label="Power")
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
df = pd.read_csv(selected_file)
df["I"] = pd.to_numeric(df["I"], errors="coerce")
df["Q"] = pd.to_numeric(df["Q"], errors="coerce")
df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

# Build complex CSI values
df["csi_complex"] = df["I"] + 1j * df["Q"]

mag_pivot = df.pivot(index="timestamp", columns="subcarrier_index", values="magnitude")
complex_pivot = df.pivot(index="timestamp", columns="subcarrier_index", values="csi_complex")

mag_pivot = mag_pivot.sort_index().dropna()
complex_pivot = complex_pivot.sort_index().dropna()

if mag_pivot.empty or complex_pivot.empty:
    st.error("CSV has no usable CSI data.")
    st.stop()

samples_per_row = mag_pivot.shape[1]
rows_needed = int(np.ceil(TARGET_LENGTH / samples_per_row))
total_rows = len(mag_pivot)

cmap = st.sidebar.selectbox("🎨 Spectrogram Color Map", CMAPS)

if "row_index" not in st.session_state:
    st.session_state.row_index = 0

start_idx = st.session_state.row_index
end_idx = start_idx + rows_needed

if end_idx >= total_rows:
    st.warning("Reached end of data — restarting.")
    st.session_state.row_index = 0
    st.rerun()

mag_window = mag_pivot.iloc[start_idx:end_idx]
complex_window = complex_pivot.iloc[start_idx:end_idx]
signal = mag_window.values.flatten()[:TARGET_LENGTH]

if np.isnan(signal).any():
    st.error("NaN values found in signal — check data integrity.")
    st.stop()

st.markdown(f"**Rows:** {start_idx + 1} to {end_idx} | **Subcarriers:** `{samples_per_row}`")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(draw_spectrogram(signal, fs=FS, cmap=cmap))

with col2:
    attn = compute_attention(signal)
    st.pyplot(draw_attention(attn))

st.markdown("### 🔍 CSI Subcarrier Magnitude Heatmap")
st.pyplot(draw_heatmap(mag_window.T.values, title="Subcarrier vs Time (Magnitude)"))

st.markdown("### 🔁 Doppler Phase FFT View (per Subcarrier)")
st.pyplot(draw_doppler_phase_fft(complex_window.values))

st.session_state.row_index += 1
time.sleep(1 / 30)
st.rerun()