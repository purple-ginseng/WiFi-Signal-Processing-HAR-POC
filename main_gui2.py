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

# TensorFlow is heavy (5-10 s import on Apple Silicon) and only needed by
# the Training / Prediction tabs. Defer the imports until those code paths run
# so the Data Collection tab opens immediately.
tf = None
layers = models = callbacks = None
load_model = None


def _ensure_tf():
    global tf, layers, models, callbacks, load_model
    if tf is not None:
        return
    import tensorflow as _tf
    from tensorflow.keras import (
        layers as _layers,
        models as _models,
        callbacks as _callbacks,
    )
    from tensorflow.keras.models import load_model as _load_model

    tf, layers, models, callbacks, load_model = (
        _tf,
        _layers,
        _models,
        _callbacks,
        _load_model,
    )


import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from bfmtool.collector import BFMCollector
from bfmtool.extractor import BFMExtractor
from bfmtool.preprocessor import BFMPreprocessor

from functools import partial
from collections import deque
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = "./data"
PCAP_PATH = "./data/wifisignal.pcap"
PCA_COMPONENTS = 50
TEST_SIZE = 0.3
EPOCHS = 20
BATCH_SIZE_TF = 32


# OS-aware tshark detection (matches pages/live_activity_detection.py)
def _detect_tshark():
    if platform.system() == "Windows":
        return r"C:\Program Files\Wireshark\tshark.exe"
    if platform.system() == "Darwin":
        for p in (
            "/Applications/Wireshark.app/Contents/MacOS/tshark",
            "/opt/homebrew/bin/tshark",
            "/usr/local/bin/tshark",
        ):
            if os.path.exists(p):
                return p
        return "/usr/local/bin/tshark"
    return "/usr/bin/tshark"


TSHARK_PATH = _detect_tshark()
BUFFER_SIZE = 1000
WINDOW_SIZE = 5
RETRY_INTERVAL = 2.0
ENABLE_TRAFFIC_GENERATION = True


# ───────────────────────────────────────────────────────────────────────────────
def convert_real_imag_to_mag_phase(df, mag_cols, phase_cols):
    """Convert real/imag columns to magnitude/phase"""
    import re

    # Get all ratio columns
    ratio_cols = [
        col for col in df.columns if "Ratio_Real" in col or "Ratio_Imag" in col
    ]

    if not ratio_cols:
        return pd.DataFrame()

    # Extract subcarrier indices
    subcarrier_pattern = re.compile(r"SCIDX_(-?\d+)_Ratio_(Real|Imag)")
    subcarriers = set()

    for col in ratio_cols:
        match = subcarrier_pattern.match(col)
        if match:
            subcarriers.add(int(match.group(1)))

    subcarriers = sorted(subcarriers)

    # Convert to mag/phase
    mag_phase_data = []

    for idx, row in df.iterrows():
        row_features = {}

        for sc_idx in subcarriers:
            real_col = f"SCIDX_{sc_idx}_Ratio_Real"
            imag_col = f"SCIDX_{sc_idx}_Ratio_Imag"

            if real_col in df.columns and imag_col in df.columns:
                real_val = row[real_col]
                imag_val = row[imag_col]

                # Compute magnitude and phase
                mag = np.sqrt(real_val**2 + imag_val**2)
                phase = np.arctan2(imag_val, real_val)

                row_features[f"SCIDX_{sc_idx}_Mag"] = mag
                row_features[f"SCIDX_{sc_idx}_Phase"] = phase

        mag_phase_data.append(row_features)

    return pd.DataFrame(mag_phase_data)


# ───────────────────────────────────────────────────────────────────────────────
class LiveDataCollector:
    """
    Collects pcap data from router in real-time using BFMCollector.
    Uses dual-queue architecture: download thread continuously pulls pcaps,
    processing thread extracts/preprocesses in parallel for maximum throughput.
    """

    def __init__(self, host, user, password, mag_cols, phase_cols):
        self.host = host
        self.user = user
        self.password = password
        self.running = False
        self.connected = False

        # Dual-queue architecture for parallel download + processing
        self.download_queue = deque(maxlen=1000)  # Queue of pcap files to process
        self.packet_buffer = deque(maxlen=BUFFER_SIZE)  # Mag/phase features
        self.processed_buffer = deque(maxlen=BUFFER_SIZE)  # Ratio_Real/Imag for Doppler

        # Separate threads
        self.download_thread = None
        self.processing_thread = None

        self.mag_cols = mag_cols
        self.phase_cols = phase_cols
        self.connection_status = "Not connected"
        self.last_error = None

        # Statistics
        self.total_downloaded = 0
        self.total_processed = 0
        self.total_packets = 0

        # Stall-detection watchdog: if total bytes across /tmp/bfm_capture*
        # don't change for STALL_THRESHOLD seconds, the explicit tcpdump
        # command is re-issued on the router.
        self.STALL_THRESHOLD = 1.0
        self._last_total_bytes = -1
        self._last_growth_ts = 0.0
        self._tcpdump_cmd = (
            "killall tcpdump 2>/dev/null; "
            "rm -f /tmp/bfm_capture*; "
            "tcpdump -i mon0 -p -U -G 1 -W 10 -w /tmp/bfm_capture "
            "'wlan[24] == 21' > /dev/null 2>&1 &"  # wlan[24] == 21, (wlan[0] & 0xfc) == 0xd0
        )

        # Create directories for BFM pipeline
        import os

        os.makedirs("live_bfm_pcap", exist_ok=True)
        os.makedirs("live_bfm_raw_csv", exist_ok=True)
        os.makedirs("live_bfm_processed_csv", exist_ok=True)

        # BFM pipeline components
        self.bfm_collector = None
        self.bfm_extractor = BFMExtractor(
            tshark_path=TSHARK_PATH, csv_dir="live_bfm_raw_csv"
        )
        self.bfm_preprocessor = BFMPreprocessor(dir="live_bfm_processed_csv")

    def _launch_tcpdump_explicit(self, verify=True):
        """
        Issue the saved explicit tcpdump command on the router. Returns True
        if `ps w | grep '[t]cpdump'` shows the process within 1 s.
        """
        if self.bfm_collector is None or self.bfm_collector.client is None:
            return False
        print(f"[tcpdump] Issuing: {self._tcpdump_cmd}")
        try:
            self.bfm_collector.run_command(self._tcpdump_cmd)
        except Exception as e:
            print(f"[tcpdump] run_command failed: {e}")
            return False
        if not verify:
            return True
        time.sleep(1)
        try:
            ps_out, _ = self.bfm_collector.run_command("ps w | grep '[t]cpdump'")
        except Exception as e:
            print(f"[tcpdump] verify failed: {e}")
            return False
        if not ps_out:
            print("[tcpdump] ❌ not running after issue.")
            return False
        print(f"[tcpdump] ✅ running:\n  {ps_out}")
        # Reset stall watchdog so it doesn't re-fire immediately
        self._last_total_bytes = -1
        self._last_growth_ts = time.time()
        return True

    def _connect_with_retry(self):
        """Connect to router with retry logic every 2 seconds"""
        while self.running and not self.connected:
            try:
                print(f"[Connection] Attempting to connect to {self.host}...")
                self.connection_status = "Connecting..."

                # Create fresh BFMCollector. The explicit tcpdump command below
                # uses time-based rotation (-G 1 -W 10), so filecount=10 matches
                # the ring buffer size; filesize is unused on the time-based path.
                self.bfm_collector = BFMCollector(
                    host=self.host,
                    username=self.user,
                    password=self.password,
                    local_pcap_dir="live_bfm_pcap",
                    filename="live_bfm",
                    filesize=1,
                    filecount=10,
                )

                # Establish connection
                self.bfm_collector.connect()

                # Traffic generation — same logic as pages/live_activity_detection.py
                # which is confirmed working (BFM frames get captured when a WiFi
                # client is associated to the AP and exchanging normal traffic).
                if ENABLE_TRAFFIC_GENERATION:
                    print("[Connection] Starting iperf3 traffic generator...")
                    self.bfm_collector.run_iperf3()

                    # Initial ping at 1 ms / 1 kHz; MainApp._restart_router_ping
                    # immediately overrides this with the UI's current value
                    # right after start_collection() returns.
                    print("[Connection] Starting initial ping at 1 ms (1 kHz)...")
                    self.bfm_collector.run_command(
                        "ping -i 0.001 192.168.1.1 > /dev/null 2>&1 &"
                    )
                    print("[Connection] ✅ Traffic generation enabled ")
                else:
                    print("[Connection] ⚠️ Traffic generation DISABLED ")
                    print(
                        "[Connection] Relying on natural WiFi traffic from devices (phone, etc.)"
                    )

                # Wait 5 s after connection / traffic-gen so iperf3+ping settle
                # before tcpdump starts capturing.
                print("[Connection] Waiting 5 s before starting tcpdump...")
                time.sleep(5)

                # Launch tcpdump with the explicit command (and verify) — saved
                # on the instance so the stall watchdog can re-issue it.
                if not self._launch_tcpdump_explicit(verify=True):
                    raise RuntimeError("tcpdump did not start after 5-s settle")

                self.connected = True
                self.connection_status = "Connected ✓"
                self.last_error = None
                print("[Connection] ✅ Successfully connected and started tcpdump")
                break

            except Exception as e:
                self.connection_status = f"Connection failed: {str(e)[:50]}"
                self.last_error = str(e)
                print(f"[Connection] ❌ Failed: {e}")
                print(f"[Connection] Retrying in {RETRY_INTERVAL} seconds...")

                # Cleanup failed connection
                if self.bfm_collector:
                    try:
                        self.bfm_collector.close()
                    except:
                        pass
                    self.bfm_collector = None

                # Wait before retry
                time.sleep(RETRY_INTERVAL)

    def start_collection(self):
        """Start both download and processing threads"""
        if self.running:
            return

        self.running = True

        # Start download thread (pulls pcaps from router continuously)
        self.download_thread = threading.Thread(target=self._download_loop, daemon=True)
        self.download_thread.start()

        # Start processing thread (extracts/preprocesses from queue)
        self.processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self.processing_thread.start()

    def stop_collection(self):
        """Stop both threads and close connection"""
        self.running = False
        self.connected = False

        if self.bfm_collector:
            try:
                print("[Stop] Stopping tcpdump and closing connection...")
                self.bfm_collector.kill_tcpdump()

                # Only kill traffic generators if they were started
                if ENABLE_TRAFFIC_GENERATION:
                    print("[Stop] Stopping traffic generators...")
                    self.bfm_collector.kill_iperf3()
                    self.bfm_collector.run_command("killall ping")

                self.bfm_collector.close()
            except Exception as e:
                print(f"[Stop] Error during cleanup: {e}")
            finally:
                self.bfm_collector = None

        # Wait for both threads
        if self.download_thread:
            self.download_thread.join(timeout=3)
        if self.processing_thread:
            self.processing_thread.join(timeout=3)

        self.connection_status = "Stopped"

    def _download_loop(self):
        """
        Download thread: Continuously pulls pcap files from router.
        Adds new files to download_queue for processing thread.
        """
        # First, establish connection with retry
        self._connect_with_retry()

        if not self.connected:
            print("[Download] Failed to connect, exiting loop")
            return

        print("[Download] Starting download loop...")

        # Track processed files to avoid re-downloading
        processed_files = {}
        single_file_tracker = {}
        loop_count = 0  # Counter for periodic checks

        while self.running:
            try:
                loop_count += 1

                # Every 25 loops (~5 seconds), verify tcpdump is still running
                if loop_count % 25 == 0:
                    try:
                        output, _ = self.bfm_collector.run_command(
                            "pgrep -f 'tcpdump -i mon0'"
                        )
                        if output:
                            print(
                                f"[Download] tcpdump health check OK (PID: {output.strip()})"
                            )
                        else:
                            print(
                                "[Download] ⚠️ tcpdump NOT running! Attempting to restart..."
                            )
                            self.bfm_collector.run_tcpdump()
                            # self.bfm_collector.run_command("iw dev mon0 set channel 149")
                            # self.bfm_collector.run_command("tcpdump -i mon0 -U -p -w /tmp/bfm_capture -W 10 -C 1 'wlan[24] == 21' > /dev/null 2>&1 &")
                    except Exception as e:
                        print(f"[Download] tcpdump health check failed: {e}")

                # Check if we're still connected
                if not self.bfm_collector or not self.bfm_collector.client:
                    print("[Download] Connection lost, attempting to reconnect...")
                    self.connected = False
                    self._connect_with_retry()
                    if not self.connected:
                        time.sleep(RETRY_INTERVAL)
                        continue

                # Manually check for pcap files on router via SFTP
                try:
                    sftp = self.bfm_collector.client.open_sftp()
                    remote_files = sftp.listdir("/tmp/")
                    pcap_files = [
                        f for f in remote_files if f.startswith("bfm_capture")
                    ]

                    if not pcap_files:
                        # Debug: Show what files ARE in /tmp/
                        tmp_files_sample = [
                            f for f in remote_files[:20]
                        ]  # First 20 files
                        print(
                            f"[Download] No pcap files found on router (checked for 'bfm_capture*')"
                        )
                        print(f"[Download] Files in /tmp/ (sample): {tmp_files_sample}")
                        processed_files.clear()
                        single_file_tracker.clear()
                        time.sleep(0.3)
                        sftp.close()
                        continue

                    print(f"[Download] Found {len(pcap_files)} file(s): {pcap_files}")
                    file_stats = []
                    for filename in pcap_files:
                        remote_path = f"/tmp/{filename}"
                        try:
                            stat = sftp.stat(remote_path)
                        except FileNotFoundError:
                            continue
                        except Exception as e:
                            print(f"[Download] Stat error for {remote_path}: {e}")
                            continue
                        file_stats.append(
                            {
                                "path": remote_path,
                                "mtime": stat.st_mtime,
                                "size": stat.st_size,
                            }
                        )

                    if not file_stats:
                        print("[Download] No valid file stats collected")
                        time.sleep(0.3)
                        sftp.close()
                        continue

                    file_stats.sort(key=lambda x: x["mtime"])
                    print(
                        f"[Download] File stats: {[(f['path'].split('/')[-1], f['size'], 'bytes') for f in file_stats]}"
                    )

                    # ─── Stall watchdog ────────────────────────────────────
                    # If the total bytes across all bfm_capture* on the router
                    # have not increased for STALL_THRESHOLD seconds, re-issue
                    # the explicit tcpdump command. Catches the case where
                    # tcpdump silently dies, mon0 stops sniffing, or the AP's
                    # client stops triggering BFM frames.
                    total_bytes = sum(f["size"] for f in file_stats)
                    now = time.time()
                    if total_bytes != self._last_total_bytes:
                        self._last_total_bytes = total_bytes
                        self._last_growth_ts = now
                    elif now - self._last_growth_ts >= self.STALL_THRESHOLD:
                        idle = now - self._last_growth_ts
                        print(
                            f"[Watchdog] No pcap growth for {idle:.1f}s — re-issuing tcpdump."
                        )
                        self._launch_tcpdump_explicit(verify=True)
                        # Reset trackers so the freshly-cleared /tmp doesn't
                        # immediately count as another stall.
                        processed_files.clear()
                        single_file_tracker.clear()
                        self._last_growth_ts = now

                    candidates = []

                    if len(file_stats) >= 2:
                        print(
                            f"[Download] Multiple files detected, downloading all except newest"
                        )
                        candidates = file_stats[
                            :-1
                        ]  # Everything except the newest (likely active) file
                        single_file_tracker.clear()
                    else:
                        sole = file_stats[0]
                        tracker = single_file_tracker.get(sole["path"])
                        if (
                            tracker
                            and tracker["mtime"] == sole["mtime"]
                            and tracker["size"] == sole["size"]
                        ):
                            tracker["stable_checks"] += 1
                            print(
                                f"[Download] Single file {sole['path'].split('/')[-1]} stable check {tracker['stable_checks']}/2"
                            )
                        else:
                            single_file_tracker[sole["path"]] = {
                                "mtime": sole["mtime"],
                                "size": sole["size"],
                                "stable_checks": 1,
                            }
                            tracker = single_file_tracker[sole["path"]]
                            print(
                                f"[Download] Single file {sole['path'].split('/')[-1]} tracking started (size: {sole['size']} bytes)"
                            )

                        # Reduced from 3 to 2 for faster response
                        if tracker["stable_checks"] >= 2 and tracker["size"] > 0:
                            print(
                                f"[Download] Single file {sole['path'].split('/')[-1]} is stable, adding to candidates"
                            )
                            candidates.append(sole)
                        else:
                            print(
                                f"[Download] Single file {sole['path'].split('/')[-1]} not yet stable ({tracker['stable_checks']}/2 checks)"
                            )

                    print(f"[Download] Candidates for download: {len(candidates)}")
                    now_processed = []

                    for entry in candidates:
                        remote_path = entry["path"]
                        size_bytes = entry["size"]

                        if size_bytes < 100:
                            print(
                                f"[Download] Skipping {remote_path.split('/')[-1]} - too small ({size_bytes} bytes)"
                            )
                            continue

                        last_mtime = processed_files.get(remote_path)
                        if last_mtime == entry["mtime"]:
                            print(
                                f"[Download] Skipping {remote_path.split('/')[-1]} - already processed (mtime: {entry['mtime']})"
                            )
                            continue

                        print(
                            f"[Download] Processing candidate {remote_path.split('/')[-1]} ({size_bytes} bytes)"
                        )

                        local_filename = f"live_bfm{self.total_downloaded}.pcap"
                        local_path = Path("live_bfm_pcap") / local_filename

                        try:
                            sftp.get(remote_path, str(local_path))
                            processed_files[remote_path] = entry["mtime"]
                            now_processed.append(remote_path)
                            self.total_downloaded += 1

                            # DO NOT remove the remote file - let tcpdump manage rotation!
                            # Removing files that tcpdump is writing to causes it to keep
                            # writing to a deleted file descriptor, preventing new files.
                            # Instead, rely on -W flag to automatically rotate files.

                            single_file_tracker.pop(remote_path, None)

                            # Add to processing queue
                            self.download_queue.append(str(local_path))
                            print(
                                f"[Download] Queued {local_filename} ({size_bytes} bytes, queue: {len(self.download_queue)})"
                            )
                        except Exception as e:
                            print(f"[Download] Failed to download {remote_path}: {e}")

                    # Keep processed files in memory to avoid re-downloading
                    # tcpdump will rotate to new files automatically via -W flag

                    sftp.close()

                except Exception as e:
                    print(f"[Download] SFTP error: {e}")

                time.sleep(0.2)  # Check 5 times per second for new files

            except Exception as e:
                print(f"[Download] Error: {e}")
                import traceback

                traceback.print_exc()
                self.connected = False
                time.sleep(RETRY_INTERVAL)

    def _processing_loop(self):
        """
        Processing thread: Extracts and preprocesses pcap files from queue.
        Runs in parallel with download thread for maximum throughput.
        """
        print("[Processing] Starting processing loop...")

        while self.running:
            try:
                # Check if there are files to process
                if not self.download_queue:
                    time.sleep(0.1)  # Wait for new files
                    continue

                # Get next file from queue
                pcap_file = self.download_queue.popleft()
                pcap_path = Path(pcap_file)

                if not pcap_path.exists():
                    continue

                print(f"[Processing] {pcap_path.name}...")
                self.total_processed += 1

                # Step 1: Extract BFM data (phi/psi angles) - FAST: skip if empty
                csv_raw_path = Path("live_bfm_raw_csv") / (pcap_path.stem + ".csv")

                try:
                    self.bfm_extractor.pcap_to_csv(str(pcap_path), str(csv_raw_path))
                except Exception as e:
                    # Most failures are "No packets" - skip silently
                    if "No packets" not in str(e):
                        print(f"[Processing] Extraction error: {e}")
                    continue

                if not csv_raw_path.exists():
                    continue

                # Step 2: Preprocess to complex ratios
                csv_processed_path = Path("live_bfm_processed_csv") / csv_raw_path.name

                try:
                    self.bfm_preprocessor.process_file(
                        str(csv_raw_path), str(csv_processed_path)
                    )
                except Exception as e:
                    print(f"[Processing] Preprocessing error: {e}")
                    continue

                if not csv_processed_path.exists():
                    continue

                # Step 3: Convert to mag/phase and add to buffer
                df_processed = pd.read_csv(csv_processed_path)

                if df_processed.empty:
                    continue

                df_mag_phase = convert_real_imag_to_mag_phase(
                    df_processed, self.mag_cols, self.phase_cols
                )

                if df_mag_phase.empty:
                    continue

                # Add packets to buffers
                num_packets = len(df_mag_phase)
                for idx, row in df_mag_phase.iterrows():
                    self.packet_buffer.append(row.to_dict())

                # Also keep processed data for Doppler analysis
                for idx, row in df_processed.iterrows():
                    self.processed_buffer.append(row.to_dict())

                self.total_packets += num_packets
                print(
                    f"[Processing] ✓ +{num_packets} packets (buffer: {len(self.packet_buffer)}, total: {self.total_packets})"
                )

            except Exception as e:
                print(f"[Processing] Error: {e}")
                import traceback

                traceback.print_exc()
                time.sleep(0.1)

    def get_latest_packets(self, n=WINDOW_SIZE):
        """Get the latest N mag/phase packets from buffer"""
        if len(self.packet_buffer) < n:
            return None

        # Get last n packets
        packets = list(self.packet_buffer)[-n:]
        return pd.DataFrame(packets)

    def get_latest_processed(self, n=WINDOW_SIZE):
        """Get the latest N processed packets (Ratio_Real/Imag) from buffer"""
        if len(self.processed_buffer) < n:
            return None

        # Get last n packets
        packets = list(self.processed_buffer)[-n:]
        return pd.DataFrame(packets)

    def get_buffer_size(self):
        """Get current buffer size"""
        return len(self.packet_buffer)

    def get_connection_status(self):
        """Get connection status string"""
        return self.connection_status


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

        self.source_mode = tk.StringVar(value="BFM-PCAP")
        self._stop_csi = threading.Event()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # BFM Settings
        self.bfm_is_setup = False
        self.bfm_collector = None

    def _on_close(self):
        self._stop_csi.set()
        self._stop_pcap_transfer_loop()

        # bfm close connection
        if self.bfm_collector is not None:
            self._toggle_bfm_setup()

        self.destroy()

    def _load_trained_model(self):
        try:
            _ensure_tf()
            self.model = load_model("best_model.keras")
            self.pca = joblib.load("pca_tf.pkl")
            self.label_encoder = joblib.load("label_encoder_tf.pkl")
            self.scaler = joblib.load("scaler.pkl")
            print("[INFO] Model, PCA, Scaler, and LabelEncoder loaded successfully.")
            return True
        except Exception as e:
            messagebox.showerror(
                "Load Error", f"Could not load model or preprocessors:\n{e}"
            )
            return False

    def _start_pcap_transfer(self):
        if self.pcap_transfer_thread and self.pcap_transfer_thread.is_alive():
            return
        self._stop_pcap_transfer.clear()
        self.pcap_transfer_thread = threading.Thread(
            target=self._transfer_loop, daemon=True
        )
        self.pcap_transfer_thread.start()

    def _transfer_loop(self):
        while not self._stop_pcap_transfer.is_set():
            try:
                if platform.system() == "Windows":
                    os.system(
                        'sshpass -p "123456" scp -O root@192.168.1.1:/tmp/csi.pcap ./data/wifisignal.pcap'
                    )
                else:
                    subprocess.run(
                        [
                            "sshpass",
                            "-p",
                            "123456",
                            "scp",
                            "-O",
                            "root@192.168.1.1:/tmp/csi.pcap",
                            "./data/wifisignal.pcap",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            except Exception as e:
                print("[Transfer Error]", e)
            time.sleep(1)

    def _stop_pcap_transfer_loop(self):
        self._stop_pcap_transfer.set()

    def _build_ui(self):
        global_frame = ttk.LabelFrame(self, text="Global Settings", padding=10)
        global_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(global_frame, text="Chunk Size:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.chunk_scale = ttk.Scale(
            global_frame,
            from_=50,
            to=1000,
            variable=self.chunk_size,
            orient="horizontal",
        )
        self.chunk_scale.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.chunk_value_label = ttk.Label(
            global_frame, textvariable=self.chunk_size, width=5
        )
        self.chunk_value_label.grid(row=0, column=2, padx=5, pady=2)

        global_frame.columnconfigure(1, weight=1)

        ttk.Label(global_frame, text="Prediction Threshold:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.thresh_scale = ttk.Scale(
            global_frame,
            from_=0.1,
            to=1.0,
            variable=self.pred_thresh,
            orient="horizontal",
        )
        self.thresh_scale.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        self.thresh_value_label = ttk.Label(
            global_frame, text=f"{self.pred_thresh.get():.2f}", width=5
        )
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

    def _update_ping_rate_label(self, *args):
        try:
            ms = float(self.ping_interval_ms.get())
            hz = 1000.0 / ms if ms > 0 else 0.0
            self.ping_rate_label.config(text=f"(≈{hz:.0f} Hz)")
        except Exception:
            self.ping_rate_label.config(text="")

    def _restart_router_ping(self):
        """
        Kill any existing ping on the router and start a new one using the
        current `ping_interval_ms` value. Called at the start of each
        labeled-session collection so the user can retune between sessions.
        """
        if self.bfm_collector is None or not self.bfm_is_setup:
            return
        try:
            ms = float(self.ping_interval_ms.get())
        except Exception:
            ms = 1.0
        ms = max(0.1, min(ms, 60_000.0))  # sanity clamp
        seconds = ms / 1000.0
        # busybox ping accepts -i in seconds (decimal). 0.001 → 1 ms = 1 kHz.
        cmd = (
            "killall ping 2>/dev/null; "
            f"ping -i {seconds:g} 192.168.1.1 > /dev/null 2>&1 &"
        )
        try:
            self.bfm_collector.run_command(cmd)
            print(f"[Ping] Restarted at {ms:g} ms ({1000/ms:.0f} Hz)")
        except Exception as e:
            print(f"[Ping] Failed to restart: {e}")

    def _build_collection_ui(self, parent):
        src_row = ttk.Frame(parent)
        src_row.pack(fill="x", pady=(10, 0))
        ttk.Label(src_row, text="Source:").pack(side="left")
        src_menu = ttk.OptionMenu(
            src_row,
            self.source_mode,
            self.source_mode.get(),
            "RSSI-PCAP",
            "CSI-UDP",
            "BFM-PCAP",
        )
        src_menu.pack(side="left", padx=5)

        self.bfm_setup_btn = ttk.Button(
            parent, text="Setup BFM", command=self._toggle_bfm_setup
        )
        self.bfm_setup_btn.pack(pady=5)

        ttk.Label(parent, text="Label for this session:").pack(anchor="w", pady=(10, 0))
        self.collect_label = ttk.Entry(parent)
        self.collect_label.pack(fill="x", padx=10)

        ttk.Label(parent, text="Duration (seconds):").pack(anchor="w", pady=(10, 0))
        self.duration_entry = ttk.Entry(parent)
        self.duration_entry.insert(0, "120")
        self.duration_entry.pack(fill="x", padx=10)

        # Ping interval (ms) — refreshed at every Start Collection
        ping_row = ttk.Frame(parent)
        ping_row.pack(fill="x", pady=(10, 0), padx=10)
        ttk.Label(ping_row, text="Ping interval (ms):").pack(side="left")
        self.ping_interval_ms = tk.DoubleVar(value=1.0)
        ttk.Spinbox(
            ping_row,
            from_=1.0,
            to=1000.0,
            increment=1.0,
            textvariable=self.ping_interval_ms,
            width=8,
        ).pack(side="left", padx=6)
        self.ping_rate_label = ttk.Label(ping_row, text="(≈1000 Hz)")
        self.ping_rate_label.pack(side="left", padx=4)
        self.ping_interval_ms.trace_add("write", self._update_ping_rate_label)

        self.collect_btn = ttk.Button(
            parent, text="Start Collection", command=self._on_collect
        )
        self.collect_btn.pack(pady=10)

        self.collect_msg = ttk.Label(parent, text="", foreground="green")
        self.collect_msg.pack()

        # NEW — Progress bar and live time label
        self.progress = ttk.Progressbar(parent, length=400, mode="determinate")
        self.progress.pack(pady=5)
        self.timer_label = ttk.Label(parent, text="Time Remaining: 0s")
        self.timer_label.pack()

        # NEW — Live router-streaming status panel
        status_frame = ttk.LabelFrame(parent, text="Streaming Status", padding=8)
        status_frame.pack(fill="x", padx=10, pady=(8, 0))
        self.status_var = tk.StringVar(value="⚪ Not set up")
        self.dl_var = tk.StringVar(value="Downloaded: 0")
        self.proc_var = tk.StringVar(value="Processed: 0")
        self.buf_var = tk.StringVar(value="Buffer: 0")
        self.pkts_var = tk.StringVar(value="Packets: 0")
        ttk.Label(
            status_frame, textvariable=self.status_var, font=("Helvetica", 11, "bold")
        ).grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(status_frame, textvariable=self.dl_var).grid(row=0, column=1, padx=12)
        ttk.Label(status_frame, textvariable=self.proc_var).grid(
            row=0, column=2, padx=12
        )
        ttk.Label(status_frame, textvariable=self.buf_var).grid(
            row=0, column=3, padx=12
        )
        ttk.Label(status_frame, textvariable=self.pkts_var).grid(
            row=0, column=4, padx=12
        )
        for c in range(5):
            status_frame.columnconfigure(c, weight=1)

        # NEW — Doppler velocity analysis (replaces the mag/phase heatmaps).
        # Compact font sizes so the two stacked axes fit the GUI without
        # title/label overlap.
        self.csi_fig = Figure(figsize=(8, 3.8), tight_layout=True)
        self.vel_ax = self.csi_fig.add_subplot(211)
        self.absvel_ax = self.csi_fig.add_subplot(212, sharex=self.vel_ax)
        for _ax in (self.vel_ax, self.absvel_ax):
            _ax.tick_params(labelsize=7)
            _ax.grid(True, alpha=0.3)
        self.vel_ax.set_title(
            "Doppler Velocity Time-Series", fontsize=9, fontweight="bold", pad=2
        )
        self.vel_ax.set_ylabel("Phase Velocity (rad/packet)", fontsize=8)
        self.absvel_ax.set_title(
            "Motion Magnitude (used for activity detection)",
            fontsize=9,
            fontweight="bold",
            pad=2,
        )
        self.absvel_ax.set_ylabel("|Velocity| (rad/packet)", fontsize=8)
        self.absvel_ax.set_xlabel("Packet Number", fontsize=8)
        self.csi_canvas = FigureCanvasTkAgg(self.csi_fig, master=parent)
        self.csi_canvas.get_tk_widget().pack(padx=10, pady=10, fill="both", expand=True)
        self.WALK_THRESHOLD = 0.1  # rad/packet — same default as the Streamlit page

        # Current Doppler features panel (mean / max / std / energy)
        feat_frame = ttk.LabelFrame(parent, text="Current Doppler Features", padding=8)
        feat_frame.pack(fill="x", padx=10, pady=(0, 8))
        self.feat_mean_var = tk.StringVar(value="Mean Velocity: 0.0000")
        self.feat_max_var = tk.StringVar(value="Max Velocity: 0.0000")
        self.feat_std_var = tk.StringVar(value="Std Dev: 0.0000")
        self.feat_energy_var = tk.StringVar(value="Energy: 0.00e+00")
        ttk.Label(feat_frame, textvariable=self.feat_mean_var).grid(
            row=0, column=0, padx=12, sticky="w"
        )
        ttk.Label(feat_frame, textvariable=self.feat_max_var).grid(
            row=0, column=1, padx=12, sticky="w"
        )
        ttk.Label(feat_frame, textvariable=self.feat_std_var).grid(
            row=0, column=2, padx=12, sticky="w"
        )
        ttk.Label(feat_frame, textvariable=self.feat_energy_var).grid(
            row=0, column=3, padx=12, sticky="w"
        )
        for c in range(4):
            feat_frame.columnconfigure(c, weight=1)

        # Kick off the periodic UI tick (status + plot refresh on main thread)
        self.after(500, self._streaming_tick)

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
        """
        Run a single labeled streaming session. The download/processing threads
        owned by LiveDataCollector pull pcaps off the router via SFTP into
        live_bfm_pcap/, extract them via tshark into live_bfm_raw_csv/, and
        preprocess them into live_bfm_processed_csv/ in the background. The
        periodic UI tick (_streaming_tick) handles status + plot refresh, so
        this method only governs duration, the in-RAM CSV save, and the snapshot
        copy into the canonical bfm_pcap/raw_csv/processed_csv directories.
        """
        if self.bfm_collector is None:
            self.collect_msg.config(
                text="Please click 'Setup BFM' first.", foreground="red"
            )
            self.collect_btn.config(state="normal")
            return

        # Mark which files were already in the live dirs before this session
        snapshot_before = {
            "pcap": (
                set(os.listdir("live_bfm_pcap"))
                if os.path.isdir("live_bfm_pcap")
                else set()
            ),
            "raw": (
                set(os.listdir("live_bfm_raw_csv"))
                if os.path.isdir("live_bfm_raw_csv")
                else set()
            ),
            "proc": (
                set(os.listdir("live_bfm_processed_csv"))
                if os.path.isdir("live_bfm_processed_csv")
                else set()
            ),
        }

        try:
            # Spawn the SSH+tcpdump+SFTP+tshark+preprocess pipeline
            self.bfm_collector.start_collection()

            # Refresh the router-side ping with the current UI value, so the
            # user can retune between sessions without re-doing Setup.
            self._restart_router_ping()

            start_ts = time.time()
            self.progress.config(maximum=duration)

            while time.time() - start_ts < duration:
                elapsed = time.time() - start_ts
                self.progress["value"] = elapsed
                self.timer_label.config(
                    text=f"Time Remaining: {max(duration - int(elapsed), 0)}s"
                )
                # Plot + status refresh is driven by _streaming_tick on the main thread
                time.sleep(0.2)

        except Exception as e:
            self.collect_msg.config(text=f"Error: {e}", foreground="red")
            print(f"[BFM ERROR] {e}")

        finally:
            # 1) Save in-RAM mag/phase buffer to ./data/ for training
            if self.bfm_collector is not None:
                if len(self.bfm_collector.packet_buffer) > 0:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"bfm_data_{label}_{ts}.csv"
                    filepath = os.path.join(DATA_DIR, fname)
                    df_out = pd.DataFrame(list(self.bfm_collector.packet_buffer))
                    df_out["label"] = label
                    os.makedirs(DATA_DIR, exist_ok=True)
                    df_out.to_csv(filepath, index=False)
                    self.collect_msg.config(
                        text=f"Saved {len(df_out)} BFM packets → {fname}",
                        foreground="green",
                    )
                    print(f"[BFM] Saved {len(df_out)} records to {filepath}")
                else:
                    self.collect_msg.config(
                        text="[BFM] Collection finished, but no packets were captured.",
                        foreground="orange",
                    )

            # NOTE: streaming continues after a labeled session ends so the
            # user can record multiple sessions back-to-back without re-running
            # preflight. The collector is only torn down when "Close BFM" is
            # clicked.

            # 2) Snapshot the new files into the canonical bfm_* directories
            #    (matches the main_gui.py pipeline so live sessions can feed training)
            self._snapshot_session_to_bfm_dirs(label, snapshot_before)

            self.progress["value"] = 0
            self.timer_label.config(text="Time Remaining: 0s")
            self.collect_btn.config(state="normal")

    def _snapshot_session_to_bfm_dirs(self, label, snapshot_before):
        """
        Merge every file produced during this session into a single output per
        directory, named `bfm_data_{label}_{timestamp}.{ext}` to match the
        existing labeled-training corpus convention (one file per session).

        - PCAPs: concatenated with scapy (rdpcap + wrpcap) so the merged file
          is still a valid libpcap container.
        - CSVs: concatenated with pandas (single header row, all rows below).
        """
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_basename = f"bfm_data_{label}_{ts}"

        def _new_files(d, before_set):
            if not os.path.isdir(d):
                return []
            return sorted(
                os.path.join(d, f)
                for f in (set(os.listdir(d)) - before_set)
                if os.path.isfile(os.path.join(d, f))
            )

        pcap_files = _new_files("live_bfm_pcap", snapshot_before["pcap"])
        raw_files = _new_files("live_bfm_raw_csv", snapshot_before["raw"])
        proc_files = _new_files("live_bfm_processed_csv", snapshot_before["proc"])

        # 1) Merge pcaps with scapy
        if pcap_files:
            os.makedirs("bfm_pcap", exist_ok=True)
            dst_pcap = os.path.join("bfm_pcap", out_basename + ".pcap")
            try:
                from scapy.all import rdpcap, wrpcap

                merged = []
                for f in pcap_files:
                    try:
                        merged.extend(rdpcap(f))
                    except Exception as e:
                        print(f"[Snapshot] Skipping unreadable pcap {f}: {e}")
                if merged:
                    wrpcap(dst_pcap, merged)
                    print(
                        f"[Snapshot] Wrote {dst_pcap}  ({len(merged)} frames from {len(pcap_files)} chunks)"
                    )
            except Exception as e:
                print(f"[Snapshot] PCAP merge failed: {e}")

        # 2) Merge raw + processed CSVs with pandas (single header, all rows)
        for src_files, dst_dir in (
            (raw_files, "bfm_raw_csv"),
            (proc_files, "bfm_processed_csv"),
        ):
            if not src_files:
                continue
            os.makedirs(dst_dir, exist_ok=True)
            dst_csv = os.path.join(dst_dir, out_basename + ".csv")
            try:
                frames = []
                for f in src_files:
                    try:
                        df = pd.read_csv(f)
                        if not df.empty:
                            frames.append(df)
                    except Exception as e:
                        print(f"[Snapshot] Skipping bad CSV {f}: {e}")
                if frames:
                    merged_df = pd.concat(frames, ignore_index=True)
                    merged_df.to_csv(dst_csv, index=False)
                    print(
                        f"[Snapshot] Wrote {dst_csv}  ({len(merged_df)} rows from {len(src_files)} chunks)"
                    )
            except Exception as e:
                print(f"[Snapshot] CSV merge failed for {dst_dir}: {e}")

    # ─── Periodic UI tick (runs on Tk main thread) ─────────────────────────
    def _streaming_tick(self):
        """Refresh the status panel and the live heatmaps every 500 ms."""
        try:
            self._update_status_panel()
            if self.bfm_collector is not None and getattr(
                self.bfm_collector, "running", False
            ):
                self._update_live_plots()
        except Exception as e:
            print(f"[Streaming Tick] {e}")
        finally:
            self.after(500, self._streaming_tick)

    def _update_status_panel(self):
        c = self.bfm_collector
        if c is None:
            self.status_var.set("⚪ Not set up")
            self.dl_var.set("Downloaded: 0")
            self.proc_var.set("Processed: 0")
            self.buf_var.set("Buffer: 0")
            self.pkts_var.set("Packets: 0")
            return

        if c.connected:
            icon = "🟢"
        elif c.running:
            icon = "🟡"
        else:
            icon = "🔴"
        self.status_var.set(f"{icon} {c.connection_status}")

        self.dl_var.set(f"Downloaded: {c.total_downloaded}")
        self.proc_var.set(f"Processed: {c.total_processed}")
        self.buf_var.set(f"Buffer: {len(c.packet_buffer)}")
        self.pkts_var.set(f"Packets: {c.total_packets}")

    def _update_live_plots(self):
        """
        Pull the last N packets from the *processed* buffer (which has the
        complex Ratio_Real/Ratio_Imag columns), compute mean phase velocity
        across subcarriers, and refresh the two Doppler subplots plus the
        feature labels — same logic as Streamlit's Doppler tab.
        """
        c = self.bfm_collector
        if c is None or len(c.processed_buffer) < 3:
            return

        N = 100  # rolling window
        packets = list(c.processed_buffer)[-N:]
        if len(packets) < 3:
            return

        df = pd.DataFrame(packets)
        import re as _re

        sc_re = _re.compile(r"SCIDX_(-?\d+)_Ratio_(Real|Imag)$")

        real_map, imag_map = {}, {}
        for col in df.columns:
            m = sc_re.match(col)
            if not m:
                continue
            sc_idx = int(m.group(1))
            (real_map if m.group(2) == "Real" else imag_map)[sc_idx] = col

        common_sc = sorted(set(real_map) & set(imag_map))
        if not common_sc:
            return

        real_cols = [real_map[i] for i in common_sc]
        imag_cols = [imag_map[i] for i in common_sc]
        real_arr = df[real_cols].to_numpy(dtype=float)
        imag_arr = df[imag_cols].to_numpy(dtype=float)
        complex_matrix = real_arr + 1j * imag_arr  # (packets, subcarriers)
        complex_matrix = np.nan_to_num(complex_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Phase velocity = unwrap(phase).diff() across packet axis
        phases = np.angle(complex_matrix)
        phases = np.unwrap(phases, axis=0)
        phases = np.nan_to_num(phases, nan=0.0, posinf=0.0, neginf=0.0)
        phase_velocity = np.diff(phases, axis=0)
        phase_velocity = np.nan_to_num(phase_velocity, nan=0.0, posinf=0.0, neginf=0.0)
        mean_velocity = np.mean(phase_velocity, axis=1)
        abs_velocity = np.abs(mean_velocity)
        x = np.arange(len(mean_velocity))

        # Top — signed velocity
        self.vel_ax.clear()
        self.vel_ax.plot(
            x, mean_velocity, color="#e74c3c", linewidth=1.5, label="Mean Velocity"
        )
        self.vel_ax.fill_between(x, mean_velocity, alpha=0.3, color="#e74c3c")
        self.vel_ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        self.vel_ax.set_title(
            f"Doppler Velocity Time-Series — last {len(packets)} packets",
            fontsize=9,
            fontweight="bold",
            pad=2,
        )
        self.vel_ax.set_ylabel("Phase Velocity (rad/packet)", fontsize=8)
        self.vel_ax.tick_params(labelsize=7)
        self.vel_ax.grid(True, alpha=0.3)
        self.vel_ax.legend(loc="upper right", fontsize=7)

        # Bottom — motion magnitude with walking threshold
        self.absvel_ax.clear()
        self.absvel_ax.plot(
            x, abs_velocity, color="#3498db", linewidth=1.5, label="Motion Magnitude"
        )
        self.absvel_ax.fill_between(x, abs_velocity, alpha=0.3, color="#3498db")
        self.absvel_ax.axhline(
            self.WALK_THRESHOLD,
            color="orange",
            linestyle="--",
            alpha=0.6,
            label=f"Walking Threshold ({self.WALK_THRESHOLD:g})",
        )
        self.absvel_ax.set_title(
            "Motion Magnitude (used for activity detection)",
            fontsize=9,
            fontweight="bold",
            pad=2,
        )
        self.absvel_ax.set_ylabel("|Velocity| (rad/packet)", fontsize=8)
        self.absvel_ax.set_xlabel("Packet Number", fontsize=8)
        self.absvel_ax.tick_params(labelsize=7)
        self.absvel_ax.grid(True, alpha=0.3)
        self.absvel_ax.legend(loc="upper right", fontsize=7)

        self.csi_fig.tight_layout()
        self.csi_canvas.draw_idle()

        # Feature panel
        try:
            from scipy.signal import detrend

            mean_v = float(np.mean(abs_velocity))
            max_v = float(np.max(abs_velocity))
            std_v = float(np.std(mean_velocity))
            if len(mean_velocity) >= 3:
                detr = detrend(mean_velocity)
                detr = np.nan_to_num(detr, nan=0.0, posinf=0.0, neginf=0.0)
                energy = float(np.sum(np.abs(np.fft.fft(detr)) ** 2))
            else:
                energy = 0.0
            self.feat_mean_var.set(f"Mean Velocity: {mean_v:.4f}")
            self.feat_max_var.set(f"Max Velocity: {max_v:.4f}")
            self.feat_std_var.set(f"Std Dev: {std_v:.4f}")
            self.feat_energy_var.set(f"Energy: {energy:.2e}")
        except Exception as e:
            print(f"[Doppler features] {e}")

    def _toggle_bfm_setup(self):
        """
        Setup/close toggle. On setup, runs the preflight check in a background
        thread so the UI stays responsive. The collector is only created if
        every required check passes (or auto-fixes successfully).
        """
        # --- STATE 1: BFM is currently NOT set up ---
        if not self.bfm_is_setup:
            print("[BFM] Setup clicked — running preflight check...")
            self.bfm_setup_btn.config(state="disabled", text="Checking…")
            self.collect_msg.config(text="Running preflight check…", foreground="blue")
            threading.Thread(target=self._do_preflight_then_setup, daemon=True).start()
            return

        # --- STATE 2: BFM is currently SET UP ---
        print("[BFM] Close clicked — shutting down collector...")
        try:
            if self.bfm_collector is not None:
                self.bfm_collector.stop_collection()
                self.bfm_collector = None
            self.bfm_is_setup = False
            self.bfm_setup_btn.config(text="Setup BFM", state="normal")
            self.collect_msg.config(text="BFM connection closed.", foreground="black")
            print("[BFM] ✅ CLOSE succeeded.")
        except Exception as e:
            messagebox.showerror("BFM Close Error", str(e))
            print(f"[BFM ERROR]: {e}")

    def _do_preflight_then_setup(self):
        """Background-thread version: preflight checks, then create collector."""
        ok, lines = self._preflight_check(
            host="192.168.1.1", user="root", password="123456"
        )
        report = "\n".join(lines)

        # Marshal back onto the Tk main thread for UI updates
        if not ok:
            self.after(0, lambda: self._on_preflight_failed(report))
            return

        try:
            self.bfm_collector = LiveDataCollector(
                host="192.168.1.1",
                user="root",
                password="123456",
                mag_cols=[],
                phase_cols=[],
            )
            self.after(0, lambda: self._on_preflight_succeeded(report))
        except Exception as e:
            err = f"Collector init failed: {e}"
            self.after(0, lambda: self._on_preflight_failed(report + "\n\n" + err))

    def _on_preflight_succeeded(self, report):
        self.bfm_is_setup = True
        self.bfm_setup_btn.config(text="Close BFM", state="normal")
        print(f"[Preflight]\n{report}")
        # Kick the streaming threads NOW so the router actually connects.
        try:
            self.bfm_collector.start_collection()
            self.collect_msg.config(
                text="Preflight OK. Streaming started — click Start Collection to record a labeled session.",
                foreground="blue",
            )
        except Exception as e:
            self.collect_msg.config(
                text=f"Preflight OK but streaming failed to start: {e}",
                foreground="red",
            )
            print(f"[BFM ERROR] start_collection failed: {e}")
            messagebox.showinfo("Preflight passed", report)
            return

        # Give the download thread a few seconds to SSH in, start iperf3+ping,
        # and launch tcpdump, then verify with `ps | grep tcpdump` and surface
        # the result so the user sees the exact running command line.
        threading.Thread(
            target=self._verify_tcpdump_after_setup,
            args=(report,),
            daemon=True,
        ).start()

    def _verify_tcpdump_after_setup(self, preflight_report):
        """Wait briefly for streaming to come up, then `ps | grep tcpdump`."""
        import paramiko

        # Allow up to ~10 s for the download thread to start tcpdump
        ps_line = ""
        for _ in range(10):
            time.sleep(1)
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(
                    "192.168.1.1", username="root", password="123456", timeout=4
                )
                _, stdout, _ = client.exec_command("ps w | grep '[t]cpdump'")
                ps_line = stdout.read().decode().strip()
                client.close()
                if ps_line:
                    break
            except Exception as e:
                ps_line = f"(verify ssh failed: {e})"
        # Marshal back onto the Tk main thread for the messagebox
        self.after(0, lambda: self._show_setup_summary(preflight_report, ps_line))

    def _show_setup_summary(self, preflight_report, ps_line):
        if ps_line and "tcpdump" in ps_line:
            summary = (
                f"{preflight_report}\n\n"
                f"--- ps | grep tcpdump ---\n{ps_line}\n\n"
                "✅ tcpdump is running on the router. The streaming threads will\n"
                "now poll /tmp/bfm_capture* and pull rotated files via SFTP."
            )
            print(f"[Setup] tcpdump running:\n  {ps_line}")
            messagebox.showinfo("Setup complete", summary)
        else:
            summary = (
                f"{preflight_report}\n\n"
                f"--- ps | grep tcpdump ---\n{ps_line or '(no tcpdump process found)'}\n\n"
                "⚠️  tcpdump did not show up after 10 s. The download thread\n"
                "should keep retrying — watch the console for [Download] lines."
            )
            print(f"[Setup] WARNING: tcpdump not found yet:\n  {ps_line!r}")
            messagebox.showwarning("Setup partial", summary)

    def _on_preflight_failed(self, report):
        self.bfm_setup_btn.config(text="Setup BFM", state="normal")
        self.collect_msg.config(text="Preflight failed — see dialog.", foreground="red")
        print(f"[Preflight FAILED]\n{report}")
        messagebox.showerror("Preflight failed", report)

    def _preflight_check(self, host, user, password):
        """
        Run the same checks as the manual SOP:
          1. router pings
          2. SSH works
          3. tcpdump installed on router
          4. openssh-sftp-server installed on router
          5. mon0 monitor interface up (auto-creates if missing)
          6. live_bfm_* working directories exist locally

        Returns (all_required_passed, list_of_human_readable_lines).
        """
        import paramiko, subprocess as _sp

        lines = []
        ok = True

        # 1. Ping router
        try:
            if platform.system() == "Windows":
                ping_cmd = ["ping", "-w", "1", "-w", "2000", host]
            else:
                ping_cmd = ["ping", "-c", "1", "-W", "2000", host]
            r = _sp.run(ping_cmd, capture_output=True, timeout=4)
            if r.returncode == 0:
                lines.append(f"✓ Router {host} reachable")
            else:
                lines.append(f"✗ Router {host} unreachable — check the LAN cable")
                return False, lines
        except Exception as e:
            lines.append(f"✗ Ping failed: {e}")
            return False, lines

        # 2. SSH (paramiko, same client the streaming uses)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(host, username=user, password=password, timeout=6)
            lines.append(f"✓ SSH connected as {user}@{host}")
        except Exception as e:
            lines.append(f"✗ SSH failed: {e}")
            return False, lines

        def _ssh(cmd):
            _, out, err = client.exec_command(cmd)
            return out.read().decode().strip(), err.read().decode().strip()

        # 3. tcpdump
        out, _ = _ssh("which tcpdump")
        if out:
            lines.append(f"✓ tcpdump at {out}")
        else:
            lines.append(
                "✗ tcpdump NOT installed on router  — run: opkg install tcpdump-mini"
            )
            ok = False

        # 4. openssh-sftp-server (paramiko's sftp.get needs this)
        out, _ = _ssh("ls /usr/libexec/sftp-server 2>/dev/null")
        if out:
            lines.append("✓ openssh-sftp-server present")
        else:
            lines.append(
                "✗ openssh-sftp-server NOT installed — run: opkg install openssh-sftp-server"
            )
            ok = False

        # 5. mon0 — auto-create if missing
        out, _ = _ssh("ip link show mon0 2>&1")
        if "mon0:" in out and "can't find" not in out and "does not exist" not in out:
            lines.append("✓ mon0 monitor interface up")
        else:
            lines.append("… mon0 missing, attempting to create…")
            _ssh("iw phy phy0 interface add mon0 type monitor 2>&1")
            _ssh("ip link set mon0 up 2>&1")
            out, _ = _ssh("ip link show mon0 2>&1")
            if "mon0:" in out and "can't find" not in out:
                lines.append("✓ mon0 monitor interface created and brought up")
            else:
                lines.append("✗ Failed to create mon0 monitor interface")
                ok = False

        # 6. Local working directories
        for d in (
            "live_bfm_pcap",
            "live_bfm_raw_csv",
            "live_bfm_processed_csv",
            "bfm_pcap",
            "bfm_raw_csv",
            "bfm_processed_csv",
        ):
            os.makedirs(d, exist_ok=True)
        lines.append("✓ Local working directories ready")

        try:
            client.close()
        except Exception:
            pass
        return ok, lines

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
            with open(path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ["timestamp", "subcarrier_index", "I", "Q", "magnitude", "phase"]
                )

                while time.time() - start_ts < duration and not self._stop_csi.is_set():
                    elapsed = round(time.time() - start_ts, 4)
                    self.progress["value"] = elapsed
                    self.timer_label.config(
                        text=f"Time Remaining: {max(duration - int(elapsed), 0)}s"
                    )

                    try:
                        data, _ = sock.recvfrom(4096)
                        line = data.decode(errors="ignore").strip()
                        if not line:
                            continue

                        iq_values = line.split(",")
                        iq_values = [
                            v.strip()
                            for v in iq_values
                            if v.strip().lstrip("-").isdigit()
                        ]
                        iq_values = list(map(int, iq_values))

                        if len(iq_values) < 2:
                            continue

                        for idx in range(0, len(iq_values) - 1, 2):
                            subcarrier_index = idx // 2
                            I = iq_values[idx]
                            Q = iq_values[idx + 1]
                            magnitude = round(math.sqrt(I**2 + Q**2), 3)
                            phase = round(np.degrees(np.arctan2(Q, I)), 2)
                            writer.writerow(
                                [elapsed, subcarrier_index, I, Q, magnitude, phase]
                            )
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

        self.progress["value"] = 0
        self.timer_label.config(text="Time Remaining: 0s")
        self.collect_msg.config(
            text=f"Saved {rows} subcarrier rows → {fname}", foreground="green"
        )
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
            df = pd.DataFrame(
                wifisignal_records, columns=[f"pkt{i}" for i in range(num_features)]
            )
            df["label"] = label
            df.to_csv(os.path.join(DATA_DIR, fname), index=False)
            saved = len(wifisignal_records)
        else:
            saved = 0

        self.collect_msg.config(
            text=f"Collected {saved} packets over {duration}s → saved to {fname}",
            foreground="green",
        )
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

            if not hasattr(self, "bfm_extractor") or self.bfm_extractor is None:
                return []

            # Use tshark to extract BFM data quickly (limit to first 20 packets for speed)
            display_filter = "wlan.fixed.category_code == 21"
            command = [
                self.bfm_extractor.tshark_path,
                "-r",
                str(pcap_file),
                "-Y",
                display_filter,
                "-V",
                "-c",
                "20",  # Only read first 20 packets
            ]

            bfm_values = []
            phi_regex = re.compile(r"φ11:\s*(\d+)")
            timestamp_regex = re.compile(r"Epoch (?:Arrival )?Time:\s*(\d+\.\d+)")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )

            current_packet_text = ""
            current_timestamp = None
            current_phi_values = []

            for line in process.stdout:
                if line.startswith("Frame "):
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
        self.train_btn = ttk.Button(
            parent, text="Start Training", command=self._on_train
        )
        self.train_btn.pack(pady=10)

        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(
            side="left", fill="both", expand=True, padx=10, pady=10
        )

        self.report_txt = scrolledtext.ScrolledText(parent, width=40, height=20)
        self.report_txt.pack(side="right", fill="y", padx=10, pady=10)

    def _on_train(self):
        self.train_btn.config(state="disabled")
        threading.Thread(target=self._do_training, daemon=True).start()

    def _do_training(self):
        _ensure_tf()
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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=TEST_SIZE, random_state=42
        )

        pca = PCA(n_components=PCA_COMPONENTS)
        X_train_p = pca.fit_transform(X_train)
        X_test_p = pca.transform(X_test)

        num_classes = len(le.classes_)
        model = models.Sequential(
            [
                layers.Input(shape=(PCA_COMPONENTS,)),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        cb = [
            callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ]
        model.fit(
            X_train_p,
            y_train,
            validation_split=0.2,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE_TF,
            callbacks=cb,
            verbose=1,
        )

        probs = model.predict(X_test_p)
        preds = np.argmax(probs, axis=1)

        cm = confusion_matrix(y_test, preds)
        self.ax.clear()
        self.ax.imshow(cm, interpolation="nearest", cmap="Blues")
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
        self.pred_btn = ttk.Button(
            controls, text="Start Prediction", command=self._start_pred
        )
        self.pred_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(
            controls, text="Stop Prediction", command=self._stop_pred.set
        )
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
