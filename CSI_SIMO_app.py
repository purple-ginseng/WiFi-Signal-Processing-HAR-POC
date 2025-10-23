#!/usr/bin/env python3
"""
CSI-SIMO Visualization App for HAR Data Analysis
=================================================
Single-Input Multiple-Output (SIMO) visualization for analyzing
captured CSI data from multiple receivers

Features:
- Multi-receiver comparison
- Time-series analysis
- Amplitude/Phase heatmaps
- Diversity combining visualization
- Interactive plots with Streamlit

Usage:
    streamlit run CSI_SIMO_app.py

Author: CSI HAR Research
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, filtfilt
from scipy.ndimage import gaussian_filter
import glob
import os

# =================== CONFIGURATION ===================
DATA_DIR = "./data"
CMAPS = ["viridis", "jet", "turbo", "plasma", "inferno", "magma", "cividis", "RdYlBu_r", "twilight"]
DEFAULT_FS = 50.0  # Default sampling rate (Hz)

# Page config
st.set_page_config(
    page_title="CSI-SIMO Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================== HELPER FUNCTIONS ===================
@st.cache_data
def load_csi_data(filepath):
    """Load CSI data from CSV file"""
    try:
        df = pd.read_csv(filepath)

        # Extract amplitude columns
        amp_cols = [col for col in df.columns if col.startswith('amplitude_')]
        amplitude = df[amp_cols].values

        # Extract phase columns
        phase_cols = [col for col in df.columns if col.startswith('phase_')]
        phase = df[phase_cols].values

        # Extract RSSI
        rssi = df['rssi'].values if 'rssi' in df.columns else None

        # Extract metadata
        metadata = {
            'activity': df['activity'].iloc[0] if 'activity' in df.columns else 'Unknown',
            'rx_id': df['rx_id'].iloc[0] if 'rx_id' in df.columns else 'Unknown',
            'num_packets': len(df),
            'num_subcarriers': amplitude.shape[1] if len(amplitude.shape) > 1 else 0
        }

        return amplitude, phase, rssi, metadata
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None, None, None


def compute_complex_csi(amplitude, phase):
    """Reconstruct complex CSI from amplitude and phase"""
    return amplitude * np.exp(1j * phase)


def find_latest_files():
    """Find latest CSI MISO files"""
    pattern = os.path.join(DATA_DIR, "csi_miso_*.csv")
    files = glob.glob(pattern)

    if not files:
        return None, None, None

    files.sort(key=os.path.getmtime, reverse=True)

    rx1_file = None
    rx2_file = None
    combined_file = None

    for f in files:
        basename = os.path.basename(f).lower()
        if 'rx1' in basename and not rx1_file:
            rx1_file = f
        elif 'rx2' in basename and not rx2_file:
            rx2_file = f
        elif 'combined' in basename and not combined_file:
            combined_file = f

        if rx1_file and rx2_file and combined_file:
            break

    return rx1_file, rx2_file, combined_file


# =================== PLOTTING FUNCTIONS ===================
def plot_amplitude_timeseries(amplitude, fs, rx_id="RX"):
    """Plot amplitude time-series for selected subcarriers"""
    fig, ax = plt.subplots(figsize=(12, 4))

    num_packets, num_subcarriers = amplitude.shape
    time_axis = np.arange(num_packets) / fs

    # Select top 5 subcarriers
    avg_amp = np.mean(amplitude, axis=0)
    top_indices = np.argsort(avg_amp)[-5:]

    colors = plt.cm.tab10(np.linspace(0, 1, 5))

    for i, sc_idx in enumerate(top_indices):
        ax.plot(time_axis, amplitude[:, sc_idx],
               label=f'SC {sc_idx}', linewidth=1.5,
               color=colors[i], alpha=0.8)

    ax.set_title(f'{rx_id}: Top 5 Subcarrier Amplitude', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_amplitude_heatmap(amplitude, fs, rx_id="RX", cmap='viridis'):
    """Plot amplitude heatmap"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Apply minimal smoothing
    amplitude_smooth = gaussian_filter(amplitude.T, sigma=0.5)

    im = ax.imshow(amplitude_smooth, aspect='auto', cmap=cmap,
                   origin='lower', interpolation='bilinear')

    ax.set_title(f'{rx_id}: CSI Amplitude Heatmap', fontsize=12, fontweight='bold')
    ax.set_xlabel('Packet Index')
    ax.set_ylabel('Subcarrier')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Amplitude', rotation=270, labelpad=20)

    plt.tight_layout()
    return fig


def plot_phase_heatmap(phase, fs, rx_id="RX", cmap='twilight'):
    """Plot phase heatmap"""
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(phase.T, aspect='auto', cmap=cmap,
                   origin='lower', interpolation='bilinear',
                   vmin=-np.pi, vmax=np.pi)

    ax.set_title(f'{rx_id}: CSI Phase Heatmap', fontsize=12, fontweight='bold')
    ax.set_xlabel('Packet Index')
    ax.set_ylabel('Subcarrier')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Phase (rad)', rotation=270, labelpad=20)

    plt.tight_layout()
    return fig


def plot_rssi_timeseries(rssi, fs, rx_id="RX"):
    """Plot RSSI time-series"""
    fig, ax = plt.subplots(figsize=(12, 4))

    time_axis = np.arange(len(rssi)) / fs

    ax.plot(time_axis, rssi, 'r-', linewidth=2, alpha=0.8, label='RSSI')
    ax.fill_between(time_axis, rssi, alpha=0.3, color='red')

    mean_rssi = np.mean(rssi)
    ax.axhline(y=mean_rssi, color='darkred', linestyle='--',
              label=f'Mean: {mean_rssi:.1f} dBm', linewidth=1.5)

    ax.set_title(f'{rx_id}: RSSI over Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RSSI (dBm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_spectrogram(signal, fs, cmap='viridis'):
    """Plot spectrogram"""
    fig, ax = plt.subplots(figsize=(12, 5))

    nperseg = min(256, len(signal) // 2)
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    Sxx_smooth = gaussian_filter(Sxx, sigma=1.0)
    Sxx_db = 10 * np.log10(Sxx_smooth + 1e-10)

    im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap=cmap)

    ax.set_title('Time-Frequency Spectrogram', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(0, fs / 2)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)', rotation=270, labelpad=20)

    plt.tight_layout()
    return fig


def plot_phase_variance(phase, fs, rx_id="RX"):
    """Plot phase variance (motion indicator)"""
    fig, ax = plt.subplots(figsize=(12, 4))

    unwrapped_phase = np.unwrap(phase, axis=0)
    phase_var = np.var(unwrapped_phase, axis=1)
    time_axis = np.arange(len(phase_var)) / fs

    ax.plot(time_axis, phase_var, 'purple', linewidth=2)
    ax.fill_between(time_axis, phase_var, alpha=0.3, color='purple')

    ax.set_title(f'{rx_id}: Phase Variance (Motion Indicator)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Variance')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_diversity_comparison(amp1, amp2, amp_combined, fs):
    """Compare diversity combining results"""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Average amplitude over subcarriers
    avg_amp1 = np.mean(amp1, axis=1)
    avg_amp2 = np.mean(amp2, axis=1)
    avg_combined = np.mean(amp_combined, axis=1)

    time_axis = np.arange(len(avg_amp1)) / fs

    ax.plot(time_axis, avg_amp1, label='RX1', linewidth=2, alpha=0.7, color='blue')
    ax.plot(time_axis, avg_amp2, label='RX2', linewidth=2, alpha=0.7, color='orange')
    ax.plot(time_axis, avg_combined, label='MRC Combined', linewidth=2.5, alpha=0.9, color='green')

    ax.set_title('Diversity Combining Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Average Amplitude')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# =================== MAIN APP ===================
def main():
    st.title("📊 CSI-SIMO Data Visualization & Analysis")
    st.markdown("**Single-Input Multiple-Output (SIMO) System** - Analyze captured CSI data from multiple receivers")

    # =================== SIDEBAR ===================
    with st.sidebar:
        st.header("⚙️ Settings")

        # Sampling rate
        fs = st.number_input("Sampling Rate (Hz)", min_value=1.0, max_value=200.0,
                            value=DEFAULT_FS, step=1.0)

        # Colormap
        cmap = st.selectbox("Colormap", CMAPS, index=0)

        st.markdown("---")
        st.header("📁 Data Loading")

        # Quick load button
        if st.button("🔍 Quick Load Latest Data", type="primary"):
            rx1_file, rx2_file, combined_file = find_latest_files()

            if rx1_file:
                st.session_state['rx1_file'] = rx1_file
                st.success(f"✅ RX1: {os.path.basename(rx1_file)}")

            if rx2_file:
                st.session_state['rx2_file'] = rx2_file
                st.success(f"✅ RX2: {os.path.basename(rx2_file)}")

            if combined_file:
                st.session_state['combined_file'] = combined_file
                st.success(f"✅ Combined: {os.path.basename(combined_file)}")

            if not rx1_file and not rx2_file and not combined_file:
                st.warning("⚠️ No CSI MISO files found in /data directory")

        st.markdown("---")

        # Manual file upload
        st.subheader("Manual Upload")

        rx1_upload = st.file_uploader("RX1 CSV File", type=['csv'], key='rx1_upload')
        if rx1_upload:
            st.session_state['rx1_file'] = rx1_upload

        rx2_upload = st.file_uploader("RX2 CSV File", type=['csv'], key='rx2_upload')
        if rx2_upload:
            st.session_state['rx2_file'] = rx2_upload

        combined_upload = st.file_uploader("Combined CSV File", type=['csv'], key='combined_upload')
        if combined_upload:
            st.session_state['combined_file'] = combined_upload

    # =================== MAIN CONTENT ===================
    tabs = st.tabs(["📈 RX1 Analysis", "📈 RX2 Analysis", "📈 Combined (MRC)", "🔀 Multi-RX Comparison"])

    # Load data
    rx1_data = None
    rx2_data = None
    combined_data = None

    if 'rx1_file' in st.session_state:
        amp, phase, rssi, meta = load_csi_data(st.session_state['rx1_file'])
        if amp is not None:
            rx1_data = {'amplitude': amp, 'phase': phase, 'rssi': rssi, 'metadata': meta}

    if 'rx2_file' in st.session_state:
        amp, phase, rssi, meta = load_csi_data(st.session_state['rx2_file'])
        if amp is not None:
            rx2_data = {'amplitude': amp, 'phase': phase, 'rssi': rssi, 'metadata': meta}

    if 'combined_file' in st.session_state:
        amp, phase, rssi, meta = load_csi_data(st.session_state['combined_file'])
        if amp is not None:
            combined_data = {'amplitude': amp, 'phase': phase, 'rssi': rssi, 'metadata': meta}

    # =================== TAB 1: RX1 Analysis ===================
    with tabs[0]:
        if rx1_data:
            st.subheader(f"RX1: {rx1_data['metadata']['activity']} ({rx1_data['metadata']['num_packets']} packets)")

            col1, col2 = st.columns(2)

            with col1:
                st.pyplot(plot_amplitude_timeseries(rx1_data['amplitude'], fs, 'RX1'))
                st.pyplot(plot_amplitude_heatmap(rx1_data['amplitude'], fs, 'RX1', cmap))

            with col2:
                if rx1_data['rssi'] is not None:
                    st.pyplot(plot_rssi_timeseries(rx1_data['rssi'], fs, 'RX1'))
                st.pyplot(plot_phase_heatmap(rx1_data['phase'], fs, 'RX1', 'twilight'))

            st.pyplot(plot_spectrogram(np.mean(rx1_data['amplitude'], axis=1), fs, cmap))
            st.pyplot(plot_phase_variance(rx1_data['phase'], fs, 'RX1'))

        else:
            st.info("📁 Load RX1 data from the sidebar to view analysis")

    # =================== TAB 2: RX2 Analysis ===================
    with tabs[1]:
        if rx2_data:
            st.subheader(f"RX2: {rx2_data['metadata']['activity']} ({rx2_data['metadata']['num_packets']} packets)")

            col1, col2 = st.columns(2)

            with col1:
                st.pyplot(plot_amplitude_timeseries(rx2_data['amplitude'], fs, 'RX2'))
                st.pyplot(plot_amplitude_heatmap(rx2_data['amplitude'], fs, 'RX2', cmap))

            with col2:
                if rx2_data['rssi'] is not None:
                    st.pyplot(plot_rssi_timeseries(rx2_data['rssi'], fs, 'RX2'))
                st.pyplot(plot_phase_heatmap(rx2_data['phase'], fs, 'RX2', 'twilight'))

            st.pyplot(plot_spectrogram(np.mean(rx2_data['amplitude'], axis=1), fs, cmap))
            st.pyplot(plot_phase_variance(rx2_data['phase'], fs, 'RX2'))

        else:
            st.info("📁 Load RX2 data from the sidebar to view analysis")

    # =================== TAB 3: Combined Analysis ===================
    with tabs[2]:
        if combined_data:
            st.subheader(f"Combined (MRC): {combined_data['metadata']['activity']} ({combined_data['metadata']['num_packets']} packets)")

            col1, col2 = st.columns(2)

            with col1:
                st.pyplot(plot_amplitude_timeseries(combined_data['amplitude'], fs, 'Combined'))
                st.pyplot(plot_amplitude_heatmap(combined_data['amplitude'], fs, 'Combined', cmap))

            with col2:
                if combined_data['rssi'] is not None:
                    st.pyplot(plot_rssi_timeseries(combined_data['rssi'], fs, 'Combined'))
                st.pyplot(plot_phase_heatmap(combined_data['phase'], fs, 'Combined', 'twilight'))

            st.pyplot(plot_spectrogram(np.mean(combined_data['amplitude'], axis=1), fs, cmap))
            st.pyplot(plot_phase_variance(combined_data['phase'], fs, 'Combined'))

        else:
            st.info("📁 Load Combined data from the sidebar to view analysis")

    # =================== TAB 4: Multi-RX Comparison ===================
    with tabs[3]:
        if rx1_data and rx2_data:
            st.subheader("Multi-Receiver Comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.pyplot(plot_amplitude_heatmap(rx1_data['amplitude'], fs, 'RX1', cmap))

            with col2:
                st.pyplot(plot_amplitude_heatmap(rx2_data['amplitude'], fs, 'RX2', cmap))

            # Combined heatmap
            if combined_data:
                st.pyplot(plot_amplitude_heatmap(combined_data['amplitude'], fs, 'MRC Combined', cmap))

                # Diversity comparison
                min_len = min(len(rx1_data['amplitude']), len(rx2_data['amplitude']), len(combined_data['amplitude']))
                st.pyplot(plot_diversity_comparison(
                    rx1_data['amplitude'][:min_len],
                    rx2_data['amplitude'][:min_len],
                    combined_data['amplitude'][:min_len],
                    fs
                ))
            else:
                st.info("📁 Load Combined data to see diversity combining results")

        else:
            st.info("📁 Load both RX1 and RX2 data from the sidebar to view comparison")


if __name__ == "__main__":
    main()
