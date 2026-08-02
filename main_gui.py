import os
import glob
import time
import threading
import subprocess
import platform
import socket
import csv
import datetime

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import numpy as np
import pandas as pd
import joblib
from scapy.all import rdpcap, Dot11, RadioTap

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import load_model

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from bfmtool.collector import BFMCollector
from bfmtool.extractor import BFMExtractor
from bfmtool.preprocessor import BFMPreprocessor
from bfmtool.utils import append_mag_phase, get_bfm_columns

from functools import partial


def convert_real_imag_to_mag_phase(df, mag_cols, phase_cols):
    """Convert SCIDX_i_Ratio_Real/Ratio_Imag columns to Ratio_Mag/Ratio_Phase.

    Thin wrapper over bfmtool.utils.append_mag_phase so this GUI and the
    training pipeline (combine_bfm_data.py) share a single definition. The
    "Ratio_" prefix is load-bearing: these are |h1/h2| and arg(h1/h2) between
    the two antennas, not absolute per-subcarrier amplitudes, and the feature
    selectors downstream match on endswith("Ratio_Mag").

    Every non-subcarrier column (timestamp, MAC addresses, label) is preserved
    — filter_by_mode() in the combine step needs transmitter_address and
    receiver_address.

    mag_cols/phase_cols are unused; kept for call-site compatibility.
    """
    col_dict = get_bfm_columns(df)

    if not col_dict["real_col"] or not col_dict["imag_col"]:
        return pd.DataFrame()

    return append_mag_phase(df).drop(
        columns=col_dict["real_col"] + col_dict["imag_col"]
    )


# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR      = './data'
PCAP_PATH     = './data/wifisignal.pcap'
PCA_COMPONENTS= 50
TEST_SIZE     = 0.3
EPOCHS        = 20
BATCH_SIZE_TF = 32
# Auto-detect tshark path based on platform
if platform.system() == "Windows":
    TSHARK_PATH = r'C:\Program Files\Wireshark\tshark.exe'
elif platform.system() == "Darwin":  # macOS
    # Check common macOS locations for tshark
    possible_paths = [
        '/Applications/Wireshark.app/Contents/MacOS/tshark',  # Wireshark.app bundle
        '/usr/local/bin/tshark',  # Homebrew installation
        '/opt/homebrew/bin/tshark'  # Apple Silicon Homebrew
    ]
    TSHARK_PATH = None
    for path in possible_paths:
        if os.path.exists(path):
            TSHARK_PATH = path
            break
    if TSHARK_PATH is None:
        TSHARK_PATH = '/usr/local/bin/tshark'  # Default fallback
else:  # Linux
    TSHARK_PATH = '/usr/bin/tshark'
# ───────────────────────────────────────────────────────────────────────────────

class MainApp(tk.Tk):
    def __init__(self):
        self.scaler = None
        super().__init__()
        self.title("Unified wifisignal Toolkit - WiFi Human Pose")
        self.geometry("1000x800")

        self.chunk_size = tk.IntVar(value=128)
        self.pred_thresh = tk.DoubleVar(value=0.5)
        self.pca = None
        self.label_encoder = None
        self.model = None
        self._stop_pred = threading.Event()

        self.pcap_transfer_thread = None
        self._stop_pcap_transfer = threading.Event()

        self.source_mode = tk.StringVar(value="RSSI-PCAP")
        self._stop_csi = threading.Event() 

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # BFM Settings
        self.bfm_is_setup = False
        self.bfm_collector = None
        self.bfm_extractor = None
        self.bfm_preprocessor = None
        self.bfm_collected = set()
        self.bfm_extracted = set()
        self.bfm_processed = set()

    def _on_close(self):
        self._stop_csi.set()
        self._stop_pcap_transfer_loop()

        # bfm close connection
        if self.bfm_collector is not None:
            self._toggle_bfm_setup()

        self.destroy()

    def _load_trained_model(self):
        try:
            self.model = load_model("best_model.keras")
            self.pca = joblib.load("pca_tf.pkl")
            self.label_encoder = joblib.load("label_encoder_tf.pkl")
            self.scaler = joblib.load("scaler.pkl")
            print("[INFO] Model, PCA, Scaler, and LabelEncoder loaded successfully.")
            return True
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load model or preprocessors:\n{e}")
            return False
        
    def _start_pcap_transfer(self):
        if self.pcap_transfer_thread and self.pcap_transfer_thread.is_alive():
            return
        self._stop_pcap_transfer.clear()
        self.pcap_transfer_thread = threading.Thread(target=self._transfer_loop, daemon=True)
        self.pcap_transfer_thread.start()

    def _transfer_loop(self):
        while not self._stop_pcap_transfer.is_set():
            try:
                if platform.system() == "Windows":
                    os.system('sshpass -p "123456" scp -O root@192.168.1.1:/tmp/csi.pcap ./data/wifisignal.pcap')
                else:
                    subprocess.run([
                        "sshpass", "-p", "123456", "scp", "-O",
                        "root@192.168.1.1:/tmp/csi.pcap", "./data/wifisignal.pcap"
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print("[Transfer Error]", e)
            time.sleep(1)

    def _stop_pcap_transfer_loop(self):
        self._stop_pcap_transfer.set()

    def _build_ui(self):
        global_frame = ttk.LabelFrame(self, text="Global Settings", padding=10)
        global_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(global_frame, text="Chunk Size:").grid(row=0, column=0, sticky="w", pady=2)
        self.chunk_scale = ttk.Scale(global_frame, from_=50, to=1000, variable=self.chunk_size, orient="horizontal")
        self.chunk_scale.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.chunk_value_label = ttk.Label(global_frame, textvariable=self.chunk_size, width=5)
        self.chunk_value_label.grid(row=0, column=2, padx=5, pady=2)

        global_frame.columnconfigure(1, weight=1)

        ttk.Label(global_frame, text="Prediction Threshold:").grid(row=1, column=0, sticky="w", pady=2)
        self.thresh_scale = ttk.Scale(global_frame, from_=0.1, to=1.0, variable=self.pred_thresh, orient="horizontal")
        self.thresh_scale.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        self.thresh_value_label = ttk.Label(global_frame, text=f"{self.pred_thresh.get():.2f}", width=5)
        self.thresh_value_label.grid(row=1, column=2, padx=5, pady=2)
        self.pred_thresh.trace_add("write", self._update_thresh_label)

        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=10, pady=10)

        col_frame = ttk.Frame(tabs)
        tabs.add(col_frame, text="Data Collection")
        self._build_collection_ui(col_frame)

        train_frame = ttk.Frame(tabs)
        tabs.add(train_frame, text="Model Training")
        self._build_training_ui(train_frame)

        pred_frame = ttk.Frame(tabs)
        tabs.add(pred_frame, text="Real‑time Prediction")
        self._build_prediction_ui(pred_frame)

    def _update_thresh_label(self, *args):
        val = self.pred_thresh.get()
        self.thresh_value_label.config(text=f"{val:.2f}")

    def _build_collection_ui(self, parent):
        src_row = ttk.Frame(parent)
        src_row.pack(fill="x", pady=(10,0))
        ttk.Label(src_row, text="Source:").pack(side="left")
        src_menu = ttk.OptionMenu(src_row, self.source_mode, self.source_mode.get(), "RSSI-PCAP", "CSI-UDP", "BFM-PCAP")
        src_menu.pack(side="left", padx=5)

        self.bfm_setup_btn = ttk.Button(parent, text="Setup BFM", command=self._toggle_bfm_setup)
        self.bfm_setup_btn.pack(pady=5)

        ttk.Label(parent, text="Label for this session:").pack(anchor="w", pady=(10,0))
        self.collect_label = ttk.Entry(parent)
        self.collect_label.pack(fill="x", padx=10)

        ttk.Label(parent, text="Duration (seconds):").pack(anchor="w", pady=(10,0))
        self.duration_entry = ttk.Entry(parent)
        self.duration_entry.insert(0, "120")
        self.duration_entry.pack(fill="x", padx=10)

        self.collect_btn = ttk.Button(parent, text="Start Collection", command=self._on_collect)
        self.collect_btn.pack(pady=10)

        self.collect_msg = ttk.Label(parent, text="", foreground="green")
        self.collect_msg.pack()

        # NEW — Progress bar and live time label
        self.progress = ttk.Progressbar(parent, length=400, mode="determinate")
        self.progress.pack(pady=5)
        self.timer_label = ttk.Label(parent, text="Time Remaining: 0s")
        self.timer_label.pack()

        # NEW — Real-time plot
        self.csi_fig = Figure(figsize=(6, 2.5))
        self.csi_ax = self.csi_fig.add_subplot(111)
        self.csi_ax.set_title("Real-time CSI Amplitudes")
        self.csi_line, = self.csi_ax.plot([], [], lw=1)
        self.csi_canvas = FigureCanvasTkAgg(self.csi_fig, master=parent)
        self.csi_canvas.get_tk_widget().pack(padx=10, pady=10, fill="x")

    def _on_collect(self):
        lbl = self.collect_label.get().strip()
        if not lbl:
            messagebox.showerror("Input Error", "Please enter a label first.")
            return

        try:
            duration = int(self.duration_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Duration must be an integer.")
            return

        if self.source_mode.get() == "RSSI-PCAP":
            self.collect_msg.config(text="Collecting RSSI via PCAP…", foreground="blue")
            self._start_pcap_transfer()
            target_fn = self._do_collection_wrapper
        elif self.source_mode.get() == "CSI-UDP":
            self.collect_msg.config(text="Collecting CSI via UDP…", foreground="blue")
            self._stop_csi.clear()
            target_fn = self._do_csi_collection
        elif self.source_mode.get() == "BFM-PCAP":
            self.collect_msg.config(text="Collecting BFM...", foreground="blue")
            target_fn = self._do_bfm_collection 
        
        self.collect_btn.config(state="disabled")
        threading.Thread(target=target_fn, args=(lbl, duration), daemon=True).start()

    def _do_bfm_collection(self, label, duration):
        # --- Component 1: Initialization ---
        start_ts = None
        try:

            # Must be set BEFORE run_tcpdump() starts the background pcap
            # collector thread, otherwise any file it downloads before this
            # line runs gets saved under the default 'bfm' filename instead
            # of the labeled one.
            self.bfm_collector.filename = partial(self.generate_bfm_filename, label)
            self.bfm_collector.run_tcpdump()
            start_ts = time.time()
            self.progress.config(maximum=duration) # Set the progress bar's max value

            # For real-time plotting
            bfm_x_data = []
            bfm_y_data = []
            MAX_POINTS = 200
            last_processed_files = set()

            while time.time() - start_ts < duration:
                elapsed = time.time() - start_ts
                self.progress['value'] = elapsed
                self.timer_label.config(text=f"Time Remaining: {max(duration - int(elapsed), 0)}s")

                # Check for new collected files and extract BFM data for plotting
                try:
                    collected_files = self.bfm_collector.get_collected_files()
                    new_files = collected_files - last_processed_files

                    if new_files:
                        for pcap_file in new_files:
                            # Quick extraction of BFM data for real-time plotting
                            bfm_data = self._extract_bfm_for_plot(pcap_file)
                            if bfm_data:
                                for _, value in bfm_data:
                                    bfm_x_data.append(elapsed)
                                    bfm_y_data.append(value)

                                # Limit data points for performance
                                if len(bfm_x_data) > MAX_POINTS:
                                    bfm_x_data = bfm_x_data[-MAX_POINTS:]
                                    bfm_y_data = bfm_y_data[-MAX_POINTS:]

                                # Update the plot
                                self.csi_line.set_xdata(bfm_x_data)
                                self.csi_line.set_ydata(bfm_y_data)
                                self.csi_ax.set_title("Real-time BFM Signal (phi11)")
                                self.csi_ax.relim()
                                self.csi_ax.autoscale_view()
                                self.csi_canvas.draw()

                        last_processed_files = collected_files.copy()
                except Exception as plot_error:
                    print(f"[BFM PLOT ERROR] {plot_error}")

                time.sleep(0.1)


        except Exception as e:
            # It's good practice to catch errors from your modules

            if self.bfm_collector is None:
                e = "Please click 'Setup BFM'"

            self.collect_msg.config(text=f"Error: {e}", foreground="red")
            print(f"[BFM ERROR] {e}")

        finally:
            # --- Component 3: Cleanup and GUI Reset ---
            # This block runs regardless of whether an error occurred or not.
            # Each stage is isolated in its own try/except so that a failure
            # partway through (e.g. one corrupt pcap chunk) can't silently
            # abort the rest of the pipeline and leave bfm_processed_csv
            # empty with no explanation.
            status_text, status_color = "[BFM] Collection finished, but nothing was saved.", "orange"

            if self.bfm_collector is not None:
                try:
                    self.bfm_collector.kill_tcpdump()
                except Exception as e:
                    print(f"[BFM ERROR] kill_tcpdump failed: {e}")

                to_be_extracted = self.bfm_collector.get_collected_files() - self.bfm_collected
                self.bfm_collected.update(self.bfm_collector.get_collected_files())

                if not to_be_extracted:
                    status_text, status_color = (
                        "[BFM] No pcap files were captured — check that traffic is flowing "
                        "to/from the router.",
                        "orange",
                    )
                else:
                    try:
                        self.bfm_extractor.extract(to_be_extracted)
                    except Exception as e:
                        print(f"[BFM ERROR] Extraction failed: {e}")

                    to_be_processed = self.bfm_extractor.get_extracted_files() - self.bfm_extracted
                    self.bfm_extracted.update(self.bfm_extractor.get_extracted_files())

                    if not to_be_processed:
                        status_text, status_color = (
                            "[BFM] Pcaps captured, but no valid BFM reports could be extracted "
                            "from them.",
                            "orange",
                        )
                    else:
                        try:
                            self.bfm_preprocessor.process(to_be_processed)
                        except Exception as e:
                            print(f"[BFM ERROR] Preprocessing failed: {e}")

                        try:
                            out_name, rows = self._merge_bfm_session_csv(
                                label, to_be_processed, start_ts, duration
                            )
                        except Exception as e:
                            print(f"[BFM ERROR] Merging session CSV failed: {e}")
                            out_name, rows = None, 0

                        if out_name:
                            status_text, status_color = (
                                f"[BFM] Saved {rows} rows → bfm_real_imag_csv/{out_name} "
                                f"and bfm_mag_phase_csv/{out_name}",
                                "green",
                            )
                        else:
                            status_text, status_color = (
                                "[BFM] BFM reports were extracted, but preprocessing/merging "
                                "into a CSV failed. Check the console log.",
                                "red",
                            )

            self.progress['value'] = 0
            self.timer_label.config(text="Time Remaining: 0s")
            self.collect_btn.config(state="normal") # Re-enable the button
            self.collect_msg.config(text=status_text, foreground=status_color)
            print(status_text)

    def _merge_bfm_session_csv(self, label, raw_csv_paths, start_ts=None, duration=None):
        """
        Merge this session's per-chunk processed CSVs (one per pcap rotation
        chunk, written by BFMPreprocessor into bfm_processed_csv/ under the
        same basename as their raw/pcap source) into a single
        bfm_data_{label}_{timestamp}.csv, matching the one-file-per-session
        convention the training notebooks expect. Two versions are written:
        the real/imag ratios into bfm_real_imag_csv/ and the magnitude/phase
        conversion into bfm_mag_phase_csv/ (same {label}_{timestamp} stem). The
        per-chunk fragments are removed afterward so bfm_processed_csv doesn't
        accumulate duplicates.

        tcpdump is started/stopped per session via SSH round-trips that don't
        line up exactly with start_ts/the Python-side timer, so raw packet
        timestamps can extend past either edge of the requested window. If
        start_ts and duration are given, rows are trimmed to
        [start_ts, start_ts + duration] using the 'timestamp' column
        (epoch seconds) so the saved CSV covers exactly what was requested.

        Returns (output_filename, row_count) or (None, 0) if nothing could be merged.
        """
        processed_dir = self.bfm_preprocessor.dir
        processed_paths = [
            processed_dir / os.path.basename(str(p)) for p in raw_csv_paths
        ]
        processed_paths = [p for p in processed_paths if p.is_file()]

        if not processed_paths:
            return None, 0

        frames = []
        for p in processed_paths:
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                print(f"[BFM ERROR] Could not read {p}: {e}")

        if not frames:
            return None, 0

        merged = pd.concat(frames, ignore_index=True)

        if start_ts is not None and duration is not None and "timestamp" in merged.columns:
            ts = pd.to_numeric(merged["timestamp"], errors="coerce")
            merged = merged[(ts >= start_ts) & (ts <= start_ts + duration)].reset_index(drop=True)
            if merged.empty:
                return None, 0

        merged["label"] = label

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"bfm_data_{label}_{ts}.csv"

        # 1) Real/imag ratios (before conversion) → bfm_real_imag_csv/
        real_imag_dir = "bfm_real_imag_csv"
        os.makedirs(real_imag_dir, exist_ok=True)
        merged.to_csv(os.path.join(real_imag_dir, out_name), index=False)

        # 2) Magnitude/phase (after conversion) → bfm_mag_phase_csv/
        mag_phase_dir = "bfm_mag_phase_csv"
        os.makedirs(mag_phase_dir, exist_ok=True)
        mag_phase = convert_real_imag_to_mag_phase(merged, [], [])
        if not mag_phase.empty:
            mag_phase["label"] = label
            mag_phase.to_csv(os.path.join(mag_phase_dir, out_name), index=False)

        # Remove the per-chunk fragments so bfm_processed_csv/ stays clean
        for p in processed_paths:
            try:
                p.unlink()
            except OSError:
                pass

        return out_name, len(merged)

    def _toggle_bfm_setup(self):
        """
        Handles the logic for the BFM setup/close toggle button.
        """
        # --- STATE 1: BFM is currently NOT set up ---
        if not self.bfm_is_setup:
            print("[BFM] Button clicked. Current state: DISCONNECTED. Attempting to set up...")
            try:
                # BFM Settings
                self.bfm_collector = BFMCollector(
                    host = '192.168.1.1',
                    username = 'root',
                    password = '123456',
                    local_pcap_dir = 'bfm_pcap',
                )
                self.bfm_extractor = BFMExtractor(
                    tshark_path = TSHARK_PATH,
                    csv_dir = 'bfm_raw_csv'
                )
                self.bfm_preprocessor = BFMPreprocessor(
                    dir = 'bfm_processed_csv'
                )

                self.bfm_collector.connect()
                self.bfm_collector.run_iperf3()


                # --- If your logic succeeds, update the state and UI ---
                self.bfm_is_setup = True
                self.bfm_setup_btn.config(text="Close BFM")
                self.collect_msg.config(text="BFM connection established.", foreground="blue")
                print("[BFM] ✅ SETUP succeeded.")

            except Exception as e:
                # --- If your logic fails, show an error ---
                messagebox.showerror(f"BFM Setup Failed\n{e}")
                print(f"[BFM ERROR] Custom setup failed: {e}")

        # --- STATE 2: BFM is currently SET UP ---
        else:
            print("[BFM] Button clicked. Current state: CONNECTED. Attempting to close...")
            try:
                self.bfm_collector.kill_iperf3()
                self.bfm_collector.close()

                self.bfm_collector = None
                self.bfm_extractor = None
                self.bfm_preprocessor = None

                # --- If your logic succeeds, update the state and UI ---
                self.bfm_is_setup = False
                self.bfm_setup_btn.config(text="Setup BFM")
                self.collect_msg.config(text="BFM connection closed.", foreground="black")
                print("[BFM] ✅ Placeholder logic for CLOSE succeeded.")

            except Exception as e:
                messagebox.showerror(f"BFM Close Error\n{e}")
                print(f"[BFM ERROR]: {e}")

    def _do_csi_collection(self, label, duration, ip="0.0.0.0", port=12345):
        import math
        import datetime
        import socket
        import time
        import os
        import csv
        import numpy as np

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"esp32_csi_{label}_{timestamp}.csv"
        path = os.path.join(DATA_DIR, fname)
        os.makedirs(DATA_DIR, exist_ok=True)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((ip, port))
        sock.settimeout(1.0)

        start_ts = time.time()
        rows = 0
        self.progress.config(maximum=duration)

        # For real-time plotting
        csi_x_data = []
        csi_y_data = []
        MAX_POINTS = 200

        try:
            with open(path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["timestamp", "subcarrier_index", "I", "Q", "magnitude", "phase"])

                while time.time() - start_ts < duration and not self._stop_csi.is_set():
                    elapsed = round(time.time() - start_ts, 4)
                    self.progress['value'] = elapsed
                    self.timer_label.config(text=f"Time Remaining: {max(duration - int(elapsed), 0)}s")

                    try:
                        data, _ = sock.recvfrom(4096)
                        line = data.decode(errors="ignore").strip()
                        if not line:
                            continue

                        iq_values = line.split(",")
                        iq_values = [v.strip() for v in iq_values if v.strip().lstrip('-').isdigit()]
                        iq_values = list(map(int, iq_values))

                        if len(iq_values) < 2:
                            continue

                        for idx in range(0, len(iq_values) - 1, 2):
                            subcarrier_index = idx // 2
                            I = iq_values[idx]
                            Q = iq_values[idx + 1]
                            magnitude = round(math.sqrt(I**2 + Q**2), 3)
                            phase = round(np.degrees(np.arctan2(Q, I)), 2)
                            writer.writerow([elapsed, subcarrier_index, I, Q, magnitude, phase])
                            rows += 1

                        # Real-time plot update
                        if len(iq_values) >= 2:
                            I0, Q0 = iq_values[0], iq_values[1]
                            mag0 = (I0**2 + Q0**2) ** 0.5
                            csi_x_data.append(elapsed)
                            csi_y_data.append(mag0)

                            if len(csi_x_data) > MAX_POINTS:
                                csi_x_data = csi_x_data[-MAX_POINTS:]
                                csi_y_data = csi_y_data[-MAX_POINTS:]

                            self.csi_line.set_xdata(csi_x_data)
                            self.csi_line.set_ydata(csi_y_data)
                            self.csi_ax.relim()
                            self.csi_ax.autoscale_view()
                            self.csi_canvas.draw()

                    except socket.timeout:
                        continue

        finally:
            sock.close()

        self.progress['value'] = 0
        self.timer_label.config(text="Time Remaining: 0s")
        self.collect_msg.config(text=f"Saved {rows} subcarrier rows → {fname}", foreground="green")
        self.collect_btn.config(state="normal")


    def _do_collection_wrapper(self, label, duration):
        self._do_collection(label, duration)
        self._stop_pcap_transfer_loop()

    def _do_collection(self, label, duration):
        start_ts = time.time()
        last_count = 0
        wifisignal_records = []

        while time.time() - start_ts < duration:
            try:
                all_pkts = rdpcap(PCAP_PATH)
                new_pkts = all_pkts[last_count:]
                last_count = len(all_pkts)

                for p in new_pkts:
                    if p.haslayer(Dot11):
                        data = self.extract_wifisignal_from_radiotap(p)
                        if data:
                            wifisignal_records.append(data)

            except Exception as e:
                print("[COLLECT ERROR]", e)

            time.sleep(1)

        if not os.path.isdir(DATA_DIR):
            os.makedirs(DATA_DIR)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"wifisignal_data_{label}_{ts}.csv"

        if wifisignal_records:
            num_features = len(wifisignal_records[0])
            df = pd.DataFrame(wifisignal_records, columns=[f"pkt{i}" for i in range(num_features)])
            df["label"] = label
            df.to_csv(os.path.join(DATA_DIR, fname), index=False)
            saved = len(wifisignal_records)
        else:
            saved = 0

        self.collect_msg.config(text=f"Collected {saved} packets over {duration}s → saved to {fname}", foreground="green")
        self.collect_btn.config(state="normal")

    def extract_wifisignal_from_radiotap(self, pkt):
        try:
            if pkt.haslayer():
                raw = bytes(pkt.getlayer())
                offset = 50
                length = 60

                if len(raw) >= offset + length:
                    wifisignal_bytes = raw
                    return list(wifisignal_bytes)
            return None
        except Exception as e:
            print("Failed to parse wifisignal:", e)
            return None
        
    def generate_bfm_filename(self, label):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        return f"bfm_data_{label}_{timestamp}.pcap"

    def _extract_bfm_for_plot(self, pcap_file):
        """
        Quickly extract BFM signal data from a pcap file for real-time plotting.
        Uses tshark to extract phi11 values from BFM reports.
        Returns a list of (timestamp, avg_phi11_value) tuples.
        """
        try:
            import subprocess
            import re

            if not hasattr(self, 'bfm_extractor') or self.bfm_extractor is None:
                return []

            # Use tshark to extract BFM data quickly (limit to first 20 packets for speed)
            display_filter = "wlan.fixed.category_code == 21"
            command = [
                self.bfm_extractor.tshark_path,
                '-r', str(pcap_file),
                '-Y', display_filter,
                '-V',
                '-c', '20'  # Only read first 20 packets
            ]

            bfm_values = []
            phi_regex = re.compile(r"φ11:\s*(\d+)")
            timestamp_regex = re.compile(r"Epoch (?:Arrival )?Time:\s*(\d+\.\d+)")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )

            current_packet_text = ""
            current_timestamp = None
            current_phi_values = []

            for line in process.stdout:
                if line.startswith('Frame '):
                    # Process previous packet
                    if current_timestamp and current_phi_values:
                        avg_phi = sum(current_phi_values) / len(current_phi_values)
                        bfm_values.append((current_timestamp, avg_phi))

                    # Reset for new packet
                    current_timestamp = None
                    current_phi_values = []
                    current_packet_text = line
                else:
                    current_packet_text += line

                    # Extract timestamp
                    if not current_timestamp:
                        ts_match = timestamp_regex.search(line)
                        if ts_match:
                            current_timestamp = float(ts_match.group(1))

                    # Extract phi11 values
                    phi_match = phi_regex.search(line)
                    if phi_match:
                        current_phi_values.append(int(phi_match.group(1)))

            # Process last packet
            if current_timestamp and current_phi_values:
                avg_phi = sum(current_phi_values) / len(current_phi_values)
                bfm_values.append((current_timestamp, avg_phi))

            process.wait()
            return bfm_values

        except Exception as e:
            print(f"[BFM EXTRACT ERROR] {e}")
            return []

    def _build_training_ui(self, parent):
        self.train_btn = ttk.Button(parent, text="Start Training", command=self._on_train)
        self.train_btn.pack(pady=10)

        self.fig = Figure(figsize=(5,4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.report_txt = scrolledtext.ScrolledText(parent, width=40, height=20)
        self.report_txt.pack(side="right", fill="y", padx=10, pady=10)

    def _on_train(self):
        self.train_btn.config(state="disabled")
        threading.Thread(target=self._do_training, daemon=True).start()

    def _do_training(self):
        files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        data, labels = [], []
        for f in files:
            df = pd.read_csv(f)
            wifisignalz = self.chunk_size.get()
            try:
                X = df[[f"pkt{i}" for i in range(wifisignalz)]].values
                y = df["label"].values
                data.append(X)
                labels.append(y)
            except Exception as e:
                continue

        if not data:
            messagebox.showerror("No Data", "No usable CSV files found in data folder.")
            self.train_btn.config(state="normal")
            return

        X = np.vstack(data)
        y = np.hstack(labels)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=TEST_SIZE, random_state=42)

        pca = PCA(n_components=PCA_COMPONENTS)
        X_train_p = pca.fit_transform(X_train)
        X_test_p  = pca.transform(X_test)

        num_classes = len(le.classes_)
        model = models.Sequential([
            layers.Input(shape=(PCA_COMPONENTS,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        cb = [callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
        model.fit(X_train_p, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE_TF, callbacks=cb, verbose=1)

        probs = model.predict(X_test_p)
        preds = np.argmax(probs, axis=1)

        cm = confusion_matrix(y_test, preds)
        self.ax.clear()
        self.ax.imshow(cm, interpolation='nearest', cmap='Blues')
        self.ax.set_title("Confusion Matrix")
        self.ax.set_xlabel("Predicted")
        self.ax.set_ylabel("True")
        self.canvas.draw()

        report = classification_report(y_test, preds, target_names=le.classes_)
        self.report_txt.delete("1.0", tk.END)
        self.report_txt.insert(tk.END, report)

        model.save("tf_model.keras")
        joblib.dump(pca, "pca_tf.pkl")
        joblib.dump(le, "label_encoder_tf.pkl")

        self.pca = pca
        self.label_encoder = le
        self.model = model

        messagebox.showinfo("Training Complete", "Model, PCA, and encoder saved.")
        self.train_btn.config(state="normal")

    def _build_prediction_ui(self, parent):
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=10)
        self.pred_btn = ttk.Button(controls, text="Start Prediction", command=self._start_pred)
        self.pred_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(controls, text="Stop Prediction", command=self._stop_pred.set)
        self.stop_btn.pack(side="left", padx=5)

        self.pred_label = ttk.Label(parent, text="Waiting...", font=("Helvetica", 36))
        self.pred_label.pack(pady=50)

    # def _start_pred(self):
    #     if not (self.pca and self.model and self.label_encoder):
    #         messagebox.showerror("Model Missing", "Please train a model first.")
    #         return
    #     self._stop_pred.clear()
    #     threading.Thread(target=self._prediction_loop, daemon=True).start()

    def _start_pred(self):
        if not self.model or not self.pca or not self.label_encoder or not self.scaler:
            if not self._load_trained_model():
                return
        self._stop_pred.clear()
        threading.Thread(target=self._prediction_loop, daemon=True).start()

    # def _prediction_loop(self):
    #     wifisignalz = self.chunk_size.get()
    #     while not self._stop_pred.is_set():
    #         try:
    #             packets = rdpcap(PCAP_PATH)
    #             wifisignal = np.array([len(p) for p in packets if p.haslayer(Dot11)])
    #             if len(wifisignal) >= wifisignalz:
    #                 window = wifisignal[-wifisignalz:].reshape(1, -1)
    #                 feat = self.pca.transform(window)
    #                 probs = self.model.predict(feat)[0]
    #                 idx = np.argmax(probs)
    #                 if probs[idx] >= self.pred_thresh.get():
    #                     label = self.label_encoder.inverse_transform([idx])[0]
    #                 else:
    #                     label = "Uncertain"

    #                 self.pred_label.config(text=f"{label} ({probs[idx]:.2f})")
    #         except Exception as e:
    #             print("Prediction error:", e)
    #         time.sleep(1)

    def _prediction_loop(self):
        wifisignalz = self.chunk_size.get()

        while not self._stop_pred.is_set():
            try:
                packets = rdpcap(PCAP_PATH)
                rssi = np.array([len(p) for p in packets if p.haslayer(Dot11)])

                if len(rssi) >= wifisignalz:
                    window = rssi[-wifisignalz:].reshape(1, -1)
                    window_scaled = self.scaler.transform(window)

                    if self.pca:
                        window_processed = self.pca.transform(window_scaled)
                    else:
                        window_processed = window_scaled

                    probs = self.model.predict(window_processed)[0]
                    idx = np.argmax(probs)

                    if probs[idx] >= self.pred_thresh.get():
                        label = self.label_encoder.inverse_transform([idx])[0]
                    else:
                        label = "Uncertain"

                    self.pred_label.config(text=f"{label} ({probs[idx]:.2f})")
            except Exception as e:
                print("Prediction error:", e)
            time.sleep(1)

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
