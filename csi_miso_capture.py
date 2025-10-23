#!/usr/bin/env python3
"""
CSI-MISO Console Data Capture for HAR
======================================
Multiple-Input Single-Output (MISO) system:
- 1 Transmitter (TX): Broadcasts WiFi packets at 50 Hz
- 2 Receivers (RX): Capture CSI and RSSI data simultaneously

Usage:
    python csi_miso_capture.py --duration 10 --activity Walking

Author: CSI HAR Research
"""

import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import queue
import time
import argparse
import sys
from collections import defaultdict

# =================== CONFIGURATION ===================
BAUD_RATE = 921600
CSI_LENGTH = 128  # ESP32 CSI subcarriers (64 pairs of I/Q = 128 values)

# Default port assignments (adjust if needed)
DEFAULT_RX1_PORT = '/dev/cu.usbmodem5A7C1161771'
DEFAULT_RX2_PORT = '/dev/cu.usbmodem5A7C1154461'

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

    except Exception:
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
        self.rx_id = rx_id
        self.running = False
        self.serial_conn = None
        self.daemon = True
        self.packet_count = 0

    def run(self):
        """Main thread loop"""
        try:
            print(f"  [{self.rx_id}] Connecting to {self.port}...")
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            self.running = True
            print(f"  [{self.rx_id}] ✓ Connected")

            while self.running:
                try:
                    if self.serial_conn.in_waiting:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()

                        # Parse CSI data
                        csi_raw, rssi = parse_csi_line(line)

                        if csi_raw is not None:
                            self.packet_count += 1
                            # Add to queue with timestamp and RX ID
                            self.data_queue.put({
                                'rx_id': self.rx_id,
                                'timestamp': time.time(),
                                'csi_raw': csi_raw,
                                'rssi': rssi,
                                'packet_num': self.packet_count
                            })

                except Exception:
                    continue

        except Exception as e:
            print(f"  [{self.rx_id}] ✗ Error: {e}")
        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()

    def stop(self):
        """Stop the thread"""
        self.running = False


# =================== MAIN CAPTURE FUNCTION ===================
def capture_csi_data(rx1_port, rx2_port, duration, activity, output_dir='./data', use_diversity=True):
    """
    Capture CSI data from dual receivers

    Args:
        rx1_port: Serial port for receiver 1
        rx2_port: Serial port for receiver 2
        duration: Recording duration in seconds
        activity: Activity label
        output_dir: Output directory for CSV files
        use_diversity: Whether to use diversity combining
    """
    print("\n" + "=" * 60)
    print("CSI-MISO Data Capture for HAR")
    print("=" * 60)
    print(f"Activity: {activity}")
    print(f"Duration: {duration} seconds")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Create data queue
    data_queue = queue.Queue()

    # Create and start serial readers
    print("\n📡 Starting receivers...")
    reader1 = SerialReader(rx1_port, BAUD_RATE, data_queue, 'RX1')
    reader2 = SerialReader(rx2_port, BAUD_RATE, data_queue, 'RX2')

    reader1.start()
    reader2.start()

    time.sleep(2)  # Wait for connections

    # Data buffers
    data_buffer = {'RX1': [], 'RX2': []}

    print(f"\n▶️  Recording started! Perform '{activity}' activity...")
    print("=" * 60)

    start_time = time.time()
    last_update = start_time

    try:
        while True:
            elapsed = time.time() - start_time

            # Check if duration exceeded
            if elapsed >= duration:
                break

            # Process queue
            while not data_queue.empty():
                data = data_queue.get()
                rx_id = data['rx_id']
                data_buffer[rx_id].append(data)

            # Print status every second
            if time.time() - last_update >= 1.0:
                rx1_count = len(data_buffer['RX1'])
                rx2_count = len(data_buffer['RX2'])
                rx1_rate = rx1_count / max(elapsed, 0.1)
                rx2_rate = rx2_count / max(elapsed, 0.1)

                print(f"[{elapsed:6.1f}s] RX1: {rx1_count:4d} pkts ({rx1_rate:5.1f} Hz) | "
                      f"RX2: {rx2_count:4d} pkts ({rx2_rate:5.1f} Hz)")

                last_update = time.time()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n⏹️  Recording interrupted by user")

    finally:
        # Stop readers
        reader1.stop()
        reader2.stop()
        time.sleep(0.5)

    print("=" * 60)
    print(f"✓ Recording complete!")
    print(f"  RX1: {len(data_buffer['RX1'])} packets")
    print(f"  RX2: {len(data_buffer['RX2'])} packets")

    # =================== EXPORT DATA ===================
    print("\n💾 Exporting data...")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_files = []

    # Export RX1 data
    if len(data_buffer['RX1']) > 0:
        export_data_rx1 = []

        for pkt in data_buffer['RX1']:
            csi_complex = csi_to_complex(pkt['csi_raw'])
            amp, phase = extract_amplitude_phase(csi_complex)

            row = {
                'packet_id': pkt['packet_num'],
                'timestamp': pkt['timestamp'],
                'rx_id': 'RX1',
                'activity': activity,
                'rssi': pkt['rssi']
            }

            # Add amplitude
            for j, val in enumerate(amp):
                row[f'amplitude_{j}'] = val

            # Add phase
            for j, val in enumerate(phase):
                row[f'phase_{j}'] = val

            export_data_rx1.append(row)

        df_rx1 = pd.DataFrame(export_data_rx1)
        filename_rx1 = f"{output_dir}/csi_miso_rx1_{activity}_{timestamp_str}.csv"
        df_rx1.to_csv(filename_rx1, index=False)
        print(f"  ✓ RX1: {filename_rx1}")
        saved_files.append(filename_rx1)

    # Export RX2 data
    if len(data_buffer['RX2']) > 0:
        export_data_rx2 = []

        for pkt in data_buffer['RX2']:
            csi_complex = csi_to_complex(pkt['csi_raw'])
            amp, phase = extract_amplitude_phase(csi_complex)

            row = {
                'packet_id': pkt['packet_num'],
                'timestamp': pkt['timestamp'],
                'rx_id': 'RX2',
                'activity': activity,
                'rssi': pkt['rssi']
            }

            # Add amplitude
            for j, val in enumerate(amp):
                row[f'amplitude_{j}'] = val

            # Add phase
            for j, val in enumerate(phase):
                row[f'phase_{j}'] = val

            export_data_rx2.append(row)

        df_rx2 = pd.DataFrame(export_data_rx2)
        filename_rx2 = f"{output_dir}/csi_miso_rx2_{activity}_{timestamp_str}.csv"
        df_rx2.to_csv(filename_rx2, index=False)
        print(f"  ✓ RX2: {filename_rx2}")
        saved_files.append(filename_rx2)

    # Export combined data (diversity combining)
    if use_diversity and len(data_buffer['RX1']) > 0 and len(data_buffer['RX2']) > 0:
        print("\n  🔀 Applying diversity combining (MRC)...")
        export_data_combined = []

        min_len = min(len(data_buffer['RX1']), len(data_buffer['RX2']))

        for i in range(min_len):
            csi1_complex = csi_to_complex(data_buffer['RX1'][i]['csi_raw'])
            csi2_complex = csi_to_complex(data_buffer['RX2'][i]['csi_raw'])

            combined = diversity_combining([csi1_complex, csi2_complex])
            amp, phase = extract_amplitude_phase(combined)

            row = {
                'packet_id': i,
                'timestamp': (data_buffer['RX1'][i]['timestamp'] + data_buffer['RX2'][i]['timestamp']) / 2,
                'rx_id': 'COMBINED',
                'activity': activity,
                'rssi': (data_buffer['RX1'][i]['rssi'] + data_buffer['RX2'][i]['rssi']) / 2
            }

            # Add amplitude
            for j, val in enumerate(amp):
                row[f'amplitude_{j}'] = val

            # Add phase
            for j, val in enumerate(phase):
                row[f'phase_{j}'] = val

            export_data_combined.append(row)

        df_combined = pd.DataFrame(export_data_combined)
        filename_combined = f"{output_dir}/csi_miso_combined_{activity}_{timestamp_str}.csv"
        df_combined.to_csv(filename_combined, index=False)
        print(f"  ✓ Combined: {filename_combined}")
        saved_files.append(filename_combined)

    print("\n" + "=" * 60)
    print(f"✅ SUCCESS! Saved {len(saved_files)} files")
    print("=" * 60)

    return saved_files


# =================== MAIN ===================
def main():
    parser = argparse.ArgumentParser(description='CSI-MISO Data Capture for HAR')

    parser.add_argument('--rx1', type=str, default=DEFAULT_RX1_PORT,
                        help=f'RX1 serial port (default: {DEFAULT_RX1_PORT})')
    parser.add_argument('--rx2', type=str, default=DEFAULT_RX2_PORT,
                        help=f'RX2 serial port (default: {DEFAULT_RX2_PORT})')
    parser.add_argument('--duration', '-d', type=int, default=10,
                        help='Recording duration in seconds (default: 10)')
    parser.add_argument('--activity', '-a', type=str, required=True,
                        help='Activity label (e.g., Walking, Running, Sitting)')
    parser.add_argument('--output', '-o', type=str, default='./data',
                        help='Output directory (default: ./data)')
    parser.add_argument('--no-diversity', action='store_true',
                        help='Disable diversity combining')
    parser.add_argument('--list-ports', action='store_true',
                        help='List available serial ports and exit')

    args = parser.parse_args()

    # List ports if requested
    if args.list_ports:
        print("\n📡 Available Serial Ports:")
        print("=" * 60)
        ports = serial.tools.list_ports.comports()
        for i, port in enumerate(ports, 1):
            print(f"{i}. {port.device}")
            print(f"   Description: {port.description}")
            print(f"   Manufacturer: {port.manufacturer or 'N/A'}")
            print()
        sys.exit(0)

    # Run capture
    try:
        capture_csi_data(
            rx1_port=args.rx1,
            rx2_port=args.rx2,
            duration=args.duration,
            activity=args.activity,
            output_dir=args.output,
            use_diversity=not args.no_diversity
        )
    except KeyboardInterrupt:
        print("\n\n⏹️  Aborted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
