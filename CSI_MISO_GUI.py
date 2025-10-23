#!/usr/bin/env python3
"""
CSI-MISO GUI for Human Activity Recognition
============================================
Tkinter-based interface for capturing CSI data from multiple receivers

Features:
- Dual receiver CSI capture
- Real-time visualization
- Activity labeling
- Diversity combining (MRC)
- CSV export for HAR training

Author: PG CSI HAR Research
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import queue
import time
import os

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =================== CONFIGURATION ===================
BAUD_RATE = 921600
CSI_LENGTH = 128
BUFFER_SIZE = 100  # Number of packets to display

# Activity labels
ACTIVITY_LABELS = [
    "Idle",
    "Walking",
    "Running",
    "Sitting Down",
    "Standing Up",
    "Falling",
    "Arm Waving",
    "Jumping",
    "Bending",
    "Clapping"
]

# =================== HELPER FUNCTIONS ===================
def parse_csi_line(line):
    """Parse CSI line from ESP32"""
    try:
        if not line.startswith('CSI:'):
            return None, None

        parts = line.split(',RSSI:')
        if len(parts) != 2:
            return None, None

        csi_str = parts[0].replace('CSI:', '')
        csi_values = [int(x) for x in csi_str.split(',')]
        rssi_value = -int(parts[1].strip())

        if len(csi_values) != CSI_LENGTH:
            return None, None

        return np.array(csi_values, dtype=np.float32), rssi_value
    except Exception:
        return None, None


def csi_to_complex(csi_raw):
    """Convert raw CSI to complex numbers"""
    csi_iq = csi_raw.reshape(-1, 2)
    csi_complex = csi_iq[:, 0] + 1j * csi_iq[:, 1]
    return csi_complex


def extract_amplitude_phase(csi_complex):
    """Extract amplitude and phase from complex CSI"""
    amplitude = np.abs(csi_complex)
    phase = np.angle(csi_complex)
    return amplitude, phase


def diversity_combining(csi_list):
    """Maximum Ratio Combining (MRC)"""
    if len(csi_list) == 0:
        return None
    if len(csi_list) == 1:
        return csi_list[0]

    weights = [np.abs(csi) for csi in csi_list]
    total_weight = np.sum(weights, axis=0)
    total_weight[total_weight == 0] = 1e-10

    combined = np.zeros_like(csi_list[0])
    for csi, weight in zip(csi_list, weights):
        combined += csi * (weight / total_weight)

    return combined


# =================== SERIAL READER ===================
class SerialReader(threading.Thread):
    """Background thread to read CSI data"""

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
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=1)
            self.running = True

            while self.running:
                try:
                    if self.serial_conn.in_waiting:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                        csi_raw, rssi = parse_csi_line(line)

                        if csi_raw is not None:
                            self.packet_count += 1
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
            print(f"[{self.rx_id}] Error: {e}")
        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()

    def stop(self):
        self.running = False


# =================== MAIN GUI ===================
class CSI_MISO_GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSI-MISO Data Capture for HAR")

        # Make window responsive - get screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Set window to 80% of screen size, centered
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        x_position = int((screen_width - window_width) / 2)
        y_position = int((screen_height - window_height) / 2)

        self.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        # Allow window to be resizable
        self.minsize(1000, 600)  # Minimum size
        self.resizable(True, True)

        # State variables
        self.capturing = False
        self.data_queue = queue.Queue()
        self.data_buffer = {'RX1': [], 'RX2': []}
        self.readers = []
        self.start_time = None

        # Build UI
        self._build_ui()

        # Start update loop
        self.after(100, self._update_ui)

    def _build_ui(self):
        """Build the user interface"""

        # =================== TOP FRAME: Configuration ===================
        config_frame = ttk.LabelFrame(self, text="Configuration", padding=10)
        config_frame.pack(fill="x", padx=10, pady=5)

        # RX1 Port
        ttk.Label(config_frame, text="RX1 Port:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.rx1_port_var = tk.StringVar()
        self.rx1_combo = ttk.Combobox(config_frame, textvariable=self.rx1_port_var)
        self.rx1_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        # RX2 Port
        ttk.Label(config_frame, text="RX2 Port:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.rx2_port_var = tk.StringVar()
        self.rx2_combo = ttk.Combobox(config_frame, textvariable=self.rx2_port_var)
        self.rx2_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # Refresh ports button
        ttk.Button(config_frame, text="🔄 Refresh Ports", command=self._refresh_ports).grid(
            row=0, column=2, rowspan=2, padx=5, pady=5, sticky="ns")

        # Activity
        ttk.Label(config_frame, text="Activity:").grid(row=0, column=3, sticky="w", padx=5, pady=5)
        self.activity_var = tk.StringVar(value="Walking")
        activity_combo = ttk.Combobox(config_frame, textvariable=self.activity_var,
                                      values=ACTIVITY_LABELS)
        activity_combo.grid(row=0, column=4, sticky="ew", padx=5, pady=5)

        # Duration
        ttk.Label(config_frame, text="Duration (s):").grid(row=1, column=3, sticky="w", padx=5, pady=5)
        self.duration_var = tk.IntVar(value=10)
        duration_spin = ttk.Spinbox(config_frame, from_=1, to=300, textvariable=self.duration_var)
        duration_spin.grid(row=1, column=4, sticky="ew", padx=5, pady=5)

        # Diversity combining checkbox
        self.diversity_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text="Enable Diversity Combining (MRC)",
                       variable=self.diversity_var).grid(row=2, column=0, columnspan=5, sticky="w", padx=5, pady=5)

        # Make columns expandable
        config_frame.columnconfigure(1, weight=2)  # RX1 port column gets more space
        config_frame.columnconfigure(4, weight=1)  # Activity/Duration column expands

        # =================== MIDDLE FRAME: Visualization ===================
        viz_frame = ttk.LabelFrame(self, text="Real-time Visualization", padding=10)
        viz_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Create matplotlib figure - dynamic size based on window
        self.fig = Figure(figsize=(10, 5), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Make canvas resize with window
        viz_frame.rowconfigure(0, weight=1)
        viz_frame.columnconfigure(0, weight=1)

        # =================== BOTTOM FRAME: Status & Controls ===================
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill="both", padx=10, pady=5)

        # Make bottom frame split 50-50
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)

        # Left: Status log
        status_frame = ttk.LabelFrame(bottom_frame, text="Status Log", padding=10)
        status_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.status_text = scrolledtext.ScrolledText(status_frame, height=6, wrap=tk.WORD)
        self.status_text.pack(fill="both", expand=True)

        # Right: Statistics & Controls
        right_frame = ttk.Frame(bottom_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # Statistics
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding=10)
        stats_frame.pack(fill="both", expand=True, pady=(0, 5))

        self.elapsed_label = ttk.Label(stats_frame, text="Elapsed: 0.0s / 0s", font=("Arial", 11, "bold"))
        self.elapsed_label.pack(anchor="w", pady=2)

        self.rx1_label = ttk.Label(stats_frame, text="RX1: 0 packets (0.0 Hz)", font=("Arial", 10))
        self.rx1_label.pack(anchor="w", pady=2)

        self.rx2_label = ttk.Label(stats_frame, text="RX2: 0 packets (0.0 Hz)", font=("Arial", 10))
        self.rx2_label.pack(anchor="w", pady=2)

        # Control buttons
        btn_frame = ttk.LabelFrame(right_frame, text="Controls", padding=10)
        btn_frame.pack(fill="both", expand=False)

        # Make buttons stack in a grid for better responsiveness
        self.start_btn = ttk.Button(btn_frame, text="▶️ Start Capture", command=self._start_capture)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=2, pady=2)

        self.stop_btn = ttk.Button(btn_frame, text="⏹️ Stop Capture", command=self._stop_capture, state="disabled")
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        self.export_btn = ttk.Button(btn_frame, text="💾 Export Data", command=self._export_data)
        self.export_btn.grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        # Make button columns equal width
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        # Initialize ports
        self._refresh_ports()

    def _refresh_ports(self):
        """Refresh available serial ports"""
        ports = [p.device for p in serial.tools.list_ports.comports()]

        self.rx1_combo['values'] = ports
        self.rx2_combo['values'] = ports

        # Set defaults if available
        if '/dev/cu.usbmodem5A7C1161771' in ports:
            self.rx1_port_var.set('/dev/cu.usbmodem5A7C1161771')
        elif len(ports) > 0:
            self.rx1_port_var.set(ports[0])

        if '/dev/cu.usbmodem5A7C1154461' in ports:
            self.rx2_port_var.set('/dev/cu.usbmodem5A7C1154461')
        elif len(ports) > 1:
            self.rx2_port_var.set(ports[1])

        self._log(f"Found {len(ports)} serial ports")

    def _log(self, message):
        """Log message to status text"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)

    def _start_capture(self):
        """Start CSI capture"""
        rx1_port = self.rx1_port_var.get()
        rx2_port = self.rx2_port_var.get()

        if not rx1_port or not rx2_port:
            messagebox.showerror("Error", "Please select both RX1 and RX2 ports")
            return

        if rx1_port == rx2_port:
            messagebox.showerror("Error", "RX1 and RX2 must use different ports")
            return

        # Clear buffers
        self.data_buffer = {'RX1': [], 'RX2': []}
        self.data_queue = queue.Queue()

        # Start readers
        self._log("Starting capture...")
        reader1 = SerialReader(rx1_port, BAUD_RATE, self.data_queue, 'RX1')
        reader2 = SerialReader(rx2_port, BAUD_RATE, self.data_queue, 'RX2')

        reader1.start()
        reader2.start()

        self.readers = [reader1, reader2]
        self.capturing = True
        self.start_time = time.time()

        # Update button states
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        activity = self.activity_var.get()
        duration = self.duration_var.get()
        self._log(f"Recording '{activity}' for {duration} seconds...")

    def _stop_capture(self):
        """Stop CSI capture"""
        self.capturing = False

        for reader in self.readers:
            reader.stop()
        self.readers = []

        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

        self._log("Capture stopped")

    def _update_ui(self):
        """Update UI periodically"""
        # Process queue
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                rx_id = data['rx_id']
                self.data_buffer[rx_id].append(data)
            except queue.Empty:
                break

        # Update visualization and stats if capturing
        if self.capturing and self.start_time:
            elapsed = time.time() - self.start_time
            duration = self.duration_var.get()

            # Update stats
            rx1_count = len(self.data_buffer['RX1'])
            rx2_count = len(self.data_buffer['RX2'])
            rx1_rate = rx1_count / max(elapsed, 0.1)
            rx2_rate = rx2_count / max(elapsed, 0.1)

            self.elapsed_label.config(text=f"Elapsed: {elapsed:.1f}s / {duration}s")
            self.rx1_label.config(text=f"RX1: {rx1_count} packets ({rx1_rate:.1f} Hz)")
            self.rx2_label.config(text=f"RX2: {rx2_count} packets ({rx2_rate:.1f} Hz)")

            # Update plot
            self._update_plot()

            # Auto-stop when duration reached
            if elapsed >= duration:
                self._stop_capture()
                self._log(f"Recording complete! {rx1_count} + {rx2_count} packets captured")

        # Schedule next update
        self.after(100, self._update_ui)

    def _update_plot(self):
        """Update matplotlib visualization"""
        self.fig.clear()

        if len(self.data_buffer['RX1']) == 0 and len(self.data_buffer['RX2']) == 0:
            return

        # Create subplots with better spacing
        axes = self.fig.subplots(2, 2)
        self.fig.subplots_adjust(hspace=0.35, wspace=0.3, left=0.08, right=0.95, top=0.95, bottom=0.08)

        # RX1 Amplitude Heatmap
        if len(self.data_buffer['RX1']) > 0:
            rx1_data = self.data_buffer['RX1'][-BUFFER_SIZE:]
            rx1_amps = []
            for pkt in rx1_data:
                csi_complex = csi_to_complex(pkt['csi_raw'])
                amp, _ = extract_amplitude_phase(csi_complex)
                rx1_amps.append(amp)

            if len(rx1_amps) > 0:
                rx1_amps = np.array(rx1_amps).T
                im1 = axes[0, 0].imshow(rx1_amps, aspect='auto', cmap='viridis', interpolation='nearest')
                axes[0, 0].set_title('RX1: CSI Amplitude', fontsize=10, fontweight='bold')
                axes[0, 0].set_xlabel('Packet', fontsize=9)
                axes[0, 0].set_ylabel('Subcarrier', fontsize=9)
                axes[0, 0].tick_params(labelsize=8)
                self.fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # RX2 Amplitude Heatmap
        if len(self.data_buffer['RX2']) > 0:
            rx2_data = self.data_buffer['RX2'][-BUFFER_SIZE:]
            rx2_amps = []
            for pkt in rx2_data:
                csi_complex = csi_to_complex(pkt['csi_raw'])
                amp, _ = extract_amplitude_phase(csi_complex)
                rx2_amps.append(amp)

            if len(rx2_amps) > 0:
                rx2_amps = np.array(rx2_amps).T
                im2 = axes[0, 1].imshow(rx2_amps, aspect='auto', cmap='viridis', interpolation='nearest')
                axes[0, 1].set_title('RX2: CSI Amplitude', fontsize=10, fontweight='bold')
                axes[0, 1].set_xlabel('Packet', fontsize=9)
                axes[0, 1].set_ylabel('Subcarrier', fontsize=9)
                axes[0, 1].tick_params(labelsize=8)
                self.fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # RSSI Comparison
        if len(self.data_buffer['RX1']) > 0:
            rx1_rssi = [pkt['rssi'] for pkt in self.data_buffer['RX1'][-BUFFER_SIZE:]]
            axes[1, 0].plot(rx1_rssi, label='RX1', linewidth=2, color='blue')

        if len(self.data_buffer['RX2']) > 0:
            rx2_rssi = [pkt['rssi'] for pkt in self.data_buffer['RX2'][-BUFFER_SIZE:]]
            axes[1, 0].plot(rx2_rssi, label='RX2', linewidth=2, color='orange')

        axes[1, 0].set_title('RSSI Comparison', fontsize=10, fontweight='bold')
        axes[1, 0].set_xlabel('Packet', fontsize=9)
        axes[1, 0].set_ylabel('RSSI (dBm)', fontsize=9)
        axes[1, 0].tick_params(labelsize=8)
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        # Diversity Combined (if enabled)
        if self.diversity_var.get() and len(self.data_buffer['RX1']) > 0 and len(self.data_buffer['RX2']) > 0:
            min_len = min(len(self.data_buffer['RX1']), len(self.data_buffer['RX2']))
            combined_amps = []

            for i in range(max(0, min_len - BUFFER_SIZE), min_len):
                csi1 = csi_to_complex(self.data_buffer['RX1'][i]['csi_raw'])
                csi2 = csi_to_complex(self.data_buffer['RX2'][i]['csi_raw'])
                combined = diversity_combining([csi1, csi2])
                amp, _ = extract_amplitude_phase(combined)
                combined_amps.append(amp)

            if len(combined_amps) > 0:
                combined_amps = np.array(combined_amps).T
                im3 = axes[1, 1].imshow(combined_amps, aspect='auto', cmap='plasma', interpolation='nearest')
                axes[1, 1].set_title('Diversity Combined (MRC)', fontsize=10, fontweight='bold')
                axes[1, 1].set_xlabel('Packet', fontsize=9)
                axes[1, 1].set_ylabel('Subcarrier', fontsize=9)
                axes[1, 1].tick_params(labelsize=8)
                self.fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
        else:
            # Show message if diversity is disabled
            axes[1, 1].text(0.5, 0.5, 'Diversity Combining Disabled',
                          ha='center', va='center', fontsize=12,
                          transform=axes[1, 1].transAxes)
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])

        self.canvas.draw()

    def _export_data(self):
        """Export captured data to CSV"""
        if len(self.data_buffer['RX1']) == 0 and len(self.data_buffer['RX2']) == 0:
            messagebox.showwarning("No Data", "No data to export!")
            return

        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory", initialdir="./data")
        if not output_dir:
            return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        activity = self.activity_var.get()
        saved_files = []

        self._log("Exporting data...")

        # Export RX1
        if len(self.data_buffer['RX1']) > 0:
            data_list = []
            for pkt in self.data_buffer['RX1']:
                csi_complex = csi_to_complex(pkt['csi_raw'])
                amp, phase = extract_amplitude_phase(csi_complex)

                row = {
                    'packet_id': pkt['packet_num'],
                    'timestamp': pkt['timestamp'],
                    'rx_id': 'RX1',
                    'activity': activity,
                    'rssi': pkt['rssi']
                }

                for j, val in enumerate(amp):
                    row[f'amplitude_{j}'] = val
                for j, val in enumerate(phase):
                    row[f'phase_{j}'] = val

                data_list.append(row)

            df = pd.DataFrame(data_list)
            filename = f"{output_dir}/csi_miso_rx1_{activity}_{timestamp_str}.csv"
            df.to_csv(filename, index=False)
            saved_files.append(filename)
            self._log(f"Saved: {os.path.basename(filename)}")

        # Export RX2
        if len(self.data_buffer['RX2']) > 0:
            data_list = []
            for pkt in self.data_buffer['RX2']:
                csi_complex = csi_to_complex(pkt['csi_raw'])
                amp, phase = extract_amplitude_phase(csi_complex)

                row = {
                    'packet_id': pkt['packet_num'],
                    'timestamp': pkt['timestamp'],
                    'rx_id': 'RX2',
                    'activity': activity,
                    'rssi': pkt['rssi']
                }

                for j, val in enumerate(amp):
                    row[f'amplitude_{j}'] = val
                for j, val in enumerate(phase):
                    row[f'phase_{j}'] = val

                data_list.append(row)

            df = pd.DataFrame(data_list)
            filename = f"{output_dir}/csi_miso_rx2_{activity}_{timestamp_str}.csv"
            df.to_csv(filename, index=False)
            saved_files.append(filename)
            self._log(f"Saved: {os.path.basename(filename)}")

        # Export Combined
        if self.diversity_var.get() and len(self.data_buffer['RX1']) > 0 and len(self.data_buffer['RX2']) > 0:
            min_len = min(len(self.data_buffer['RX1']), len(self.data_buffer['RX2']))
            data_list = []

            for i in range(min_len):
                csi1 = csi_to_complex(self.data_buffer['RX1'][i]['csi_raw'])
                csi2 = csi_to_complex(self.data_buffer['RX2'][i]['csi_raw'])
                combined = diversity_combining([csi1, csi2])
                amp, phase = extract_amplitude_phase(combined)

                row = {
                    'packet_id': i,
                    'timestamp': (self.data_buffer['RX1'][i]['timestamp'] + self.data_buffer['RX2'][i]['timestamp']) / 2,
                    'rx_id': 'COMBINED',
                    'activity': activity,
                    'rssi': (self.data_buffer['RX1'][i]['rssi'] + self.data_buffer['RX2'][i]['rssi']) / 2
                }

                for j, val in enumerate(amp):
                    row[f'amplitude_{j}'] = val
                for j, val in enumerate(phase):
                    row[f'phase_{j}'] = val

                data_list.append(row)

            df = pd.DataFrame(data_list)
            filename = f"{output_dir}/csi_miso_combined_{activity}_{timestamp_str}.csv"
            df.to_csv(filename, index=False)
            saved_files.append(filename)
            self._log(f"Saved: {os.path.basename(filename)}")

        messagebox.showinfo("Export Complete", f"Saved {len(saved_files)} files to:\n{output_dir}")


# =================== MAIN ===================
if __name__ == "__main__":
    app = CSI_MISO_GUI()
    app.mainloop()
