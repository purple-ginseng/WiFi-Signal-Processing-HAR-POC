"""
CSI-MISO Data Capture GUI for Human Activity Recognition
==========================================================
Multiple-Input Single-Output (MISO) system:
- 1 Transmitter (TX): Broadcasts WiFi packets at 50 Hz
- 2+ Receivers (RX): Capture CSI and RSSI data simultaneously

Features:
- Dual/Multiple receiver synchronized capture
- Real-time CSI amplitude and phase visualization
- Activity labeling for HAR training
- CSV export with proper formatting
- Automatic diversity combining

Author: CSI HAR Research
Date: 2025
"""

import streamlit as st
import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import queue
import time
from collections import deque

# =================== CONFIGURATION ===================
BAUD_RATE = 921600
CSI_LENGTH = 128  # ESP32 CSI subcarriers (64 pairs of I/Q = 128 values)
BUFFER_SIZE = 1000  # Number of packets to buffer
SAMPLING_RATE = 50  # Expected packet rate (Hz)

# Activity labels for HAR
ACTIVITY_LABELS = [
    "Idle/Background",
    "Walking",
    "Running",
    "Sitting Down",
    "Standing Up",
    "Falling",
    "Arm Waving",
    "Jumping",
    "Bending",
    "Clapping",
    "Custom Activity"
]

# =================== HELPER FUNCTIONS ===================
def parse_csi_line(line):
    """
    Parse CSI line from ESP32 receiver
    Format: CSI:val1,val2,...,val128,RSSI:val
    Returns: (csi_array, rssi_value) or (None, None) if parse fails
    """
    try:
        if not line.startswith('CSI:'):
            return None, None

        # Split CSI and RSSI
        parts = line.split(',RSSI:')
        if len(parts) != 2:
            return None, None

        # Parse CSI values
        csi_str = parts[0].replace('CSI:', '')
        csi_values = [int(x) for x in csi_str.split(',')]

        # Parse RSSI
        rssi_value = -int(parts[1].strip())  # Make RSSI negative (dBm)

        # Validate CSI length
        if len(csi_values) != CSI_LENGTH:
            return None, None

        return np.array(csi_values, dtype=np.float32), rssi_value

    except Exception as e:
        return None, None


def csi_to_complex(csi_raw):
    """
    Convert raw CSI values to complex numbers (amplitude + phase)
    Input: array of 128 values (64 I/Q pairs)
    Output: 64 complex values
    """
    # Reshape to (64, 2) for I and Q components
    csi_iq = csi_raw.reshape(-1, 2)
    # Convert to complex: I + jQ
    csi_complex = csi_iq[:, 0] + 1j * csi_iq[:, 1]
    return csi_complex


def extract_amplitude_phase(csi_complex):
    """
    Extract amplitude and phase from complex CSI
    Returns: (amplitude_array, phase_array)
    """
    amplitude = np.abs(csi_complex)
    phase = np.angle(csi_complex)  # Phase in radians [-π, π]
    return amplitude, phase


def diversity_combining(csi_list):
    """
    Simple diversity combining: Maximum Ratio Combining (MRC)
    Input: list of complex CSI arrays from multiple receivers
    Output: combined complex CSI array
    """
    if len(csi_list) == 0:
        return None
    if len(csi_list) == 1:
        return csi_list[0]

    # MRC: weight by signal strength (amplitude)
    weights = [np.abs(csi) for csi in csi_list]
    total_weight = np.sum(weights, axis=0)

    # Avoid division by zero
    total_weight[total_weight == 0] = 1e-10

    # Weighted sum
    combined = np.zeros_like(csi_list[0])
    for csi, weight in zip(csi_list, weights):
        combined += csi * (weight / total_weight)

    return combined


# =================== SERIAL READER THREAD ===================
class SerialReader(threading.Thread):
    """Background thread to read serial data from ESP32 receiver"""

    def __init__(self, port, baud_rate, data_queue, rx_id):
        threading.Thread.__init__(self)
        self.port = port
        self.baud_rate = baud_rate
        self.data_queue = data_queue
        self.rx_id = rx_id  # Receiver ID (RX1, RX2, etc.)
        self.running = False
        self.serial_conn = None
        self.daemon = True

    def run(self):
        """Main thread loop"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            self.running = True

            while self.running:
                try:
                    if self.serial_conn.in_waiting:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()

                        # Parse CSI data
                        csi_raw, rssi = parse_csi_line(line)

                        if csi_raw is not None:
                            # Add to queue with timestamp and RX ID
                            self.data_queue.put({
                                'rx_id': self.rx_id,
                                'timestamp': time.time(),
                                'csi_raw': csi_raw,
                                'rssi': rssi
                            })

                except Exception as e:
                    continue

        except Exception as e:
            st.error(f"Error in serial reader {self.rx_id}: {e}")
        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()

    def stop(self):
        """Stop the thread"""
        self.running = False


# =================== STREAMLIT APP ===================
def main():
    st.set_page_config(page_title="CSI-MISO HAR Capture", layout="wide")

    st.title("🛜 CSI-MISO Data Capture for Human Activity Recognition")
    st.markdown("**Multiple-Input Single-Output (MISO) System** - 1 TX + Multiple RX")

    # Info box
    st.info("""
    **📡 Hardware Setup:**
    - **TX (Transmitter)**: `/dev/cu.usbmodem5A7C1147801` - Blue LED, broadcasts packets (can run on battery)
    - **RX1 (Receiver 1)**: `/dev/cu.usbmodem5A7C1161771` - Green LED, captures CSI
    - **RX2 (Receiver 2)**: `/dev/cu.usbmodem5A7C1154461` - Green LED, captures CSI

    Only RX devices need to be connected to the computer. TX can run standalone!
    """)

    # =================== SIDEBAR: CONFIGURATION ===================
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Get available serial ports
        ports = [p.device for p in serial.tools.list_ports.comports()]

        st.subheader("📡 Receiver Setup")

        # Port assignments:
        # TX: /dev/cu.usbmodem5A7C1147801 (Transmitter - not connected to computer during capture)
        # RX1: /dev/cu.usbmodem5A7C1161771
        # RX2: /dev/cu.usbmodem5A7C1154461

        # RX1 Configuration
        st.markdown("**Receiver 1 (RX1)**")
        rx1_port = st.selectbox("RX1 Port", ports, key='rx1_port',
                                index=ports.index('/dev/cu.usbmodem5A7C1161771') if '/dev/cu.usbmodem5A7C1161771' in ports else 0)

        # RX2 Configuration
        st.markdown("**Receiver 2 (RX2)**")
        rx2_enabled = st.checkbox("Enable RX2", value=True)
        if rx2_enabled:
            rx2_port = st.selectbox("RX2 Port", ports, key='rx2_port',
                                    index=ports.index('/dev/cu.usbmodem5A7C1154461') if '/dev/cu.usbmodem5A7C1154461' in ports else 0)
        else:
            rx2_port = None

        st.markdown("---")
        st.subheader("📊 Data Collection")

        # Activity selection
        activity = st.selectbox("Activity Label", ACTIVITY_LABELS)
        if activity == "Custom Activity":
            activity = st.text_input("Enter custom activity name", "My_Activity")

        # Recording settings
        duration = st.number_input("Recording Duration (seconds)", min_value=1, max_value=300, value=10)

        # Diversity combining
        use_diversity = st.checkbox("Use Diversity Combining (MRC)", value=True,
                                    help="Combine signals from multiple receivers for better SNR")

        st.markdown("---")
        st.subheader("💾 Export Settings")
        save_raw = st.checkbox("Save Raw CSI Values", value=True)
        save_amplitude = st.checkbox("Save Amplitude", value=True)
        save_phase = st.checkbox("Save Phase", value=True)
        save_rssi = st.checkbox("Save RSSI", value=True)

    # =================== MAIN AREA ===================
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Real-time CSI Visualization")
        chart_placeholder = st.empty()

    with col2:
        st.subheader("📊 Statistics")
        stats_placeholder = st.empty()

    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        start_btn = st.button("▶️ Start Capture", type="primary", use_container_width=True)

    with col_btn2:
        stop_btn = st.button("⏹️ Stop Capture", use_container_width=True)

    with col_btn3:
        export_btn = st.button("💾 Export Data", use_container_width=True)

    # =================== SESSION STATE ===================
    if 'capturing' not in st.session_state:
        st.session_state.capturing = False
    if 'data_buffer' not in st.session_state:
        st.session_state.data_buffer = {'rx1': [], 'rx2': []}
    if 'readers' not in st.session_state:
        st.session_state.readers = []
    if 'data_queue' not in st.session_state:
        st.session_state.data_queue = queue.Queue()

    # =================== START CAPTURE ===================
    if start_btn and not st.session_state.capturing:
        st.session_state.capturing = True
        st.session_state.data_buffer = {'rx1': [], 'rx2': []}
        st.session_state.start_time = time.time()

        # Start RX1 reader
        reader1 = SerialReader(rx1_port, BAUD_RATE, st.session_state.data_queue, 'rx1')
        reader1.start()
        st.session_state.readers.append(reader1)

        # Start RX2 reader if enabled
        if rx2_enabled and rx2_port:
            reader2 = SerialReader(rx2_port, BAUD_RATE, st.session_state.data_queue, 'rx2')
            reader2.start()
            st.session_state.readers.append(reader2)

        st.success(f"✅ Capture started! Recording '{activity}' for {duration} seconds...")

    # =================== STOP CAPTURE ===================
    if stop_btn and st.session_state.capturing:
        st.session_state.capturing = False

        # Stop all readers
        for reader in st.session_state.readers:
            reader.stop()
        st.session_state.readers = []

        st.info("⏹️ Capture stopped.")

    # =================== REAL-TIME UPDATE ===================
    if st.session_state.capturing:
        # Check if duration exceeded
        elapsed_time = time.time() - st.session_state.start_time

        if elapsed_time >= duration:
            st.session_state.capturing = False
            for reader in st.session_state.readers:
                reader.stop()
            st.session_state.readers = []
            st.success(f"✅ Recording complete! Captured {duration} seconds of data.")

        # Process queue
        while not st.session_state.data_queue.empty():
            data = st.session_state.data_queue.get()
            rx_id = data['rx_id']
            st.session_state.data_buffer[rx_id].append(data)

        # Visualization
        if len(st.session_state.data_buffer['rx1']) > 0 or len(st.session_state.data_buffer['rx2']) > 0:

            # Prepare data for plotting
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # RX1 Amplitude
            if len(st.session_state.data_buffer['rx1']) > 0:
                rx1_data = st.session_state.data_buffer['rx1'][-100:]  # Last 100 packets
                rx1_amplitudes = []
                for pkt in rx1_data:
                    csi_complex = csi_to_complex(pkt['csi_raw'])
                    amp, _ = extract_amplitude_phase(csi_complex)
                    rx1_amplitudes.append(amp)

                rx1_amplitudes = np.array(rx1_amplitudes)
                axes[0, 0].imshow(rx1_amplitudes.T, aspect='auto', cmap='viridis', interpolation='nearest')
                axes[0, 0].set_title('RX1: CSI Amplitude Heatmap')
                axes[0, 0].set_xlabel('Packet Index')
                axes[0, 0].set_ylabel('Subcarrier')

            # RX2 Amplitude
            if len(st.session_state.data_buffer['rx2']) > 0:
                rx2_data = st.session_state.data_buffer['rx2'][-100:]
                rx2_amplitudes = []
                for pkt in rx2_data:
                    csi_complex = csi_to_complex(pkt['csi_raw'])
                    amp, _ = extract_amplitude_phase(csi_complex)
                    rx2_amplitudes.append(amp)

                rx2_amplitudes = np.array(rx2_amplitudes)
                axes[0, 1].imshow(rx2_amplitudes.T, aspect='auto', cmap='viridis', interpolation='nearest')
                axes[0, 1].set_title('RX2: CSI Amplitude Heatmap')
                axes[0, 1].set_xlabel('Packet Index')
                axes[0, 1].set_ylabel('Subcarrier')

            # RSSI comparison
            if len(st.session_state.data_buffer['rx1']) > 0:
                rx1_rssi = [pkt['rssi'] for pkt in st.session_state.data_buffer['rx1'][-100:]]
                axes[1, 0].plot(rx1_rssi, label='RX1', linewidth=2)

            if len(st.session_state.data_buffer['rx2']) > 0:
                rx2_rssi = [pkt['rssi'] for pkt in st.session_state.data_buffer['rx2'][-100:]]
                axes[1, 0].plot(rx2_rssi, label='RX2', linewidth=2)

            axes[1, 0].set_title('RSSI Comparison')
            axes[1, 0].set_xlabel('Packet Index')
            axes[1, 0].set_ylabel('RSSI (dBm)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Diversity combined amplitude
            if use_diversity and len(st.session_state.data_buffer['rx1']) > 0 and len(st.session_state.data_buffer['rx2']) > 0:
                # Align timestamps and combine
                combined_amps = []
                min_len = min(len(st.session_state.data_buffer['rx1']), len(st.session_state.data_buffer['rx2']))

                for i in range(max(0, min_len - 100), min_len):
                    csi1_complex = csi_to_complex(st.session_state.data_buffer['rx1'][i]['csi_raw'])
                    csi2_complex = csi_to_complex(st.session_state.data_buffer['rx2'][i]['csi_raw'])

                    combined = diversity_combining([csi1_complex, csi2_complex])
                    amp, _ = extract_amplitude_phase(combined)
                    combined_amps.append(amp)

                if len(combined_amps) > 0:
                    combined_amps = np.array(combined_amps)
                    axes[1, 1].imshow(combined_amps.T, aspect='auto', cmap='plasma', interpolation='nearest')
                    axes[1, 1].set_title('Diversity Combined CSI Amplitude')
                    axes[1, 1].set_xlabel('Packet Index')
                    axes[1, 1].set_ylabel('Subcarrier')

            plt.tight_layout()
            chart_placeholder.pyplot(fig)
            plt.close()

            # Statistics
            with stats_placeholder.container():
                st.metric("Elapsed Time", f"{elapsed_time:.1f}s / {duration}s")
                st.metric("RX1 Packets", len(st.session_state.data_buffer['rx1']))
                st.metric("RX2 Packets", len(st.session_state.data_buffer['rx2']))

                if len(st.session_state.data_buffer['rx1']) > 0:
                    rx1_rate = len(st.session_state.data_buffer['rx1']) / max(elapsed_time, 0.1)
                    st.metric("RX1 Rate", f"{rx1_rate:.1f} Hz")

                if len(st.session_state.data_buffer['rx2']) > 0:
                    rx2_rate = len(st.session_state.data_buffer['rx2']) / max(elapsed_time, 0.1)
                    st.metric("RX2 Rate", f"{rx2_rate:.1f} Hz")

        # Auto-refresh
        time.sleep(0.1)
        st.rerun()

    # =================== EXPORT DATA ===================
    if export_btn:
        if len(st.session_state.data_buffer['rx1']) == 0 and len(st.session_state.data_buffer['rx2']) == 0:
            st.warning("⚠️ No data to export!")
        else:
            # Create export dataframe
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Export RX1 data
            if len(st.session_state.data_buffer['rx1']) > 0:
                export_data_rx1 = []

                for i, pkt in enumerate(st.session_state.data_buffer['rx1']):
                    csi_complex = csi_to_complex(pkt['csi_raw'])
                    amp, phase = extract_amplitude_phase(csi_complex)

                    row = {'packet_id': i, 'timestamp': pkt['timestamp'], 'rx_id': 'RX1', 'activity': activity}

                    if save_raw:
                        for j, val in enumerate(pkt['csi_raw']):
                            row[f'csi_raw_{j}'] = val

                    if save_amplitude:
                        for j, val in enumerate(amp):
                            row[f'amplitude_{j}'] = val

                    if save_phase:
                        for j, val in enumerate(phase):
                            row[f'phase_{j}'] = val

                    if save_rssi:
                        row['rssi'] = pkt['rssi']

                    export_data_rx1.append(row)

                df_rx1 = pd.DataFrame(export_data_rx1)
                filename_rx1 = f"./data/csi_miso_rx1_{activity}_{timestamp_str}.csv"
                df_rx1.to_csv(filename_rx1, index=False)
                st.success(f"✅ RX1 data exported: {filename_rx1}")

            # Export RX2 data
            if len(st.session_state.data_buffer['rx2']) > 0:
                export_data_rx2 = []

                for i, pkt in enumerate(st.session_state.data_buffer['rx2']):
                    csi_complex = csi_to_complex(pkt['csi_raw'])
                    amp, phase = extract_amplitude_phase(csi_complex)

                    row = {'packet_id': i, 'timestamp': pkt['timestamp'], 'rx_id': 'RX2', 'activity': activity}

                    if save_raw:
                        for j, val in enumerate(pkt['csi_raw']):
                            row[f'csi_raw_{j}'] = val

                    if save_amplitude:
                        for j, val in enumerate(amp):
                            row[f'amplitude_{j}'] = val

                    if save_phase:
                        for j, val in enumerate(phase):
                            row[f'phase_{j}'] = val

                    if save_rssi:
                        row['rssi'] = pkt['rssi']

                    export_data_rx2.append(row)

                df_rx2 = pd.DataFrame(export_data_rx2)
                filename_rx2 = f"./data/csi_miso_rx2_{activity}_{timestamp_str}.csv"
                df_rx2.to_csv(filename_rx2, index=False)
                st.success(f"✅ RX2 data exported: {filename_rx2}")

            # Export combined data if diversity enabled
            if use_diversity and len(st.session_state.data_buffer['rx1']) > 0 and len(st.session_state.data_buffer['rx2']) > 0:
                export_data_combined = []

                min_len = min(len(st.session_state.data_buffer['rx1']), len(st.session_state.data_buffer['rx2']))

                for i in range(min_len):
                    csi1_complex = csi_to_complex(st.session_state.data_buffer['rx1'][i]['csi_raw'])
                    csi2_complex = csi_to_complex(st.session_state.data_buffer['rx2'][i]['csi_raw'])

                    combined = diversity_combining([csi1_complex, csi2_complex])
                    amp, phase = extract_amplitude_phase(combined)

                    row = {
                        'packet_id': i,
                        'timestamp': (st.session_state.data_buffer['rx1'][i]['timestamp'] + st.session_state.data_buffer['rx2'][i]['timestamp']) / 2,
                        'rx_id': 'COMBINED',
                        'activity': activity
                    }

                    if save_amplitude:
                        for j, val in enumerate(amp):
                            row[f'amplitude_{j}'] = val

                    if save_phase:
                        for j, val in enumerate(phase):
                            row[f'phase_{j}'] = val

                    if save_rssi:
                        row['rssi'] = (st.session_state.data_buffer['rx1'][i]['rssi'] + st.session_state.data_buffer['rx2'][i]['rssi']) / 2

                    export_data_combined.append(row)

                df_combined = pd.DataFrame(export_data_combined)
                filename_combined = f"./data/csi_miso_combined_{activity}_{timestamp_str}.csv"
                df_combined.to_csv(filename_combined, index=False)
                st.success(f"✅ Combined data exported: {filename_combined}")

            st.balloons()


if __name__ == "__main__":
    main()
