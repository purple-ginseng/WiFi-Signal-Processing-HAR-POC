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
FS = 200.0
TARGET_LENGTH = int(FS * 10)  # 2000 samples for 10s
CMAPS = ["viridis", "jet", "turbo"]

# --- Functions ---
def sorted_pkt_cols(df):
    pkt_cols = [c for c in df.columns if c.startswith("pkt")]
    return sorted(pkt_cols, key=lambda x: int(x[3:]))

def compute_attention(seq):
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(-1)
    scores = torch.matmul(x, x.T) / np.sqrt(x.shape[0])
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

cmap = st.sidebar.selectbox("🎨 Spectrogram Color Map", CMAPS)
fmin, fmax = st.sidebar.slider("Spectrogram Y-Axis (Hz)", 0, int(FS//2), (0, 100))

# Row index state
if "row_index" not in st.session_state:
    st.session_state.row_index = 0

start_idx = st.session_state.row_index
end_idx = start_idx + rows_needed

# Restart when not enough rows
if end_idx >= total_rows:
    st.warning("Reached end of data — restarting.")
    st.session_state.row_index = 0
    st.rerun()

# Get signal block
window_df = df.iloc[start_idx:end_idx]
signal_matrix = window_df[pkt_cols].astype(float).values
signal = signal_matrix.flatten()[:TARGET_LENGTH]

# --- Validation ---
label = window_df["label"].mode()[0] if "label" in window_df else "N/A"
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

# --- Advance frame
st.session_state.row_index += 1
time.sleep(1 / 30)
st.rerun()