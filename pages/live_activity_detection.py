"""
Live Real-time Activity Detection
Continuously pulls pcap data from router, extracts BFM signals, and performs real-time predictions
Uses BFMCollector for robust SSH connection management with retry logic
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import threading
import subprocess
from pathlib import Path
from collections import deque
from datetime import datetime

# Configure matplotlib BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Disable matplotlib warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Import from project modules — make the path setup robust regardless of how
# Streamlit / Python launches this file (cwd may not be the project root).
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bfmtool.collector import BFMCollector
from bfmtool.extractor import BFMExtractor
from bfmtool.preprocessor import BFMPreprocessor
from tensorflow.keras.models import load_model as keras_load_model
import joblib

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "best_bfm_model_nofoil.keras"
MODEL_PATH_OPEN = BASE_DIR / "best_bfm_model_open.keras"
PCA_PATH = BASE_DIR / "bfm_nofoil_pca.pkl"

# Router settings
ROUTER_HOST = "192.168.1.1"
ROUTER_USER = "root"
ROUTER_PASSWORD = "123456"
ROUTER_PCAP_PATH = "/tmp/bfm_capture*"  # BFMCollector creates files with this pattern

# Detection settings
WINDOW_SIZE = 5  # Number of packets for prediction (reduced for faster response)
UPDATE_INTERVAL = 1.0  # Seconds between updates
BUFFER_SIZE = 100  # Keep last N packets in memory
RETRY_INTERVAL = 2.0  # Seconds between connection retries

# Traffic generation settings
ENABLE_TRAFFIC_GENERATION = True  # Set to True for self ping, False for (mobile phone)
# If False: Relies on natural WiFi traffic from devices (phone, laptop, etc.)
# If True: Router generates its own traffic (100 Hz ping + iperf3) for testing

# Tshark path detection
import platform
import os

if platform.system() == "Windows":
    TSHARK_PATH = r'C:\Program Files\Wireshark\tshark.exe'
elif platform.system() == "Darwin":  # macOS
    possible_paths = [
        '/Applications/Wireshark.app/Contents/MacOS/tshark',
        '/usr/local/bin/tshark',
        '/opt/homebrew/bin/tshark'
    ]
    TSHARK_PATH = None
    for path in possible_paths:
        if os.path.exists(path):
            TSHARK_PATH = path
            break
    if TSHARK_PATH is None:
        TSHARK_PATH = '/usr/local/bin/tshark'
else:  # Linux
    TSHARK_PATH = '/usr/bin/tshark'

# Activity labels
ACTIVITY_LABELS = {0: 'Standing', 1: 'Walking'}

# Colors for activities
ACTIVITY_COLORS = {
    'Standing': '#3498db',  # Blue
    'Walking': '#e74c3c',   # Red
}

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_and_pca(model_type='nofoil'):
    """
    Load the trained model and PCA transformer

    Args:
        model_type: 'nofoil' or 'open' to select which model to use
    """
    try:
        # Select model path based on type
        if model_type == 'open':
            model_path = MODEL_PATH_OPEN
        else:
            model_path = MODEL_PATH

        if not model_path.exists():
            st.error(f"Model not found at {model_path}")
            return None, None, None, None

        if not PCA_PATH.exists():
            st.error(f"PCA not found at {PCA_PATH}")
            return None, None, None, None

        model = keras_load_model(model_path)

        # Load PCA dictionary
        pca_data = joblib.load(PCA_PATH)
        pca = pca_data['pca']
        feature_cols = pca_data['feature_cols']
        mag_cols = pca_data['mag_cols']
        phase_cols = pca_data['phase_cols']

        st.info(f"✓ Model: {model_path.name}")
        st.info(f"✓ Model input shape: {model.input_shape}")
        st.info(f"✓ PCA: {pca.n_features_in_} features → {pca.n_components_} components")

        return model, pca, feature_cols, (mag_cols, phase_cols)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None

# ═══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION FROM ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

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
        # don't change for STALL_THRESHOLD seconds, re-issue the explicit
        # tcpdump command on the router.
        self.STALL_THRESHOLD = 0.5
        self._last_total_bytes = -1
        self._last_growth_ts = 0.0
        self._tcpdump_cmd = (
            "killall tcpdump 2>/dev/null; "
            "rm -f /tmp/bfm_capture*; "
            "tcpdump -i mon0 -p -U -B 4096 -G 1 -W 10 -w /tmp/bfm_capture "
            "'wlan[24] == 21' > /dev/null 2>&1 &"
        )

        # Create directories for BFM pipeline
        import os
        os.makedirs('live_bfm_pcap', exist_ok=True)
        os.makedirs('live_bfm_raw_csv', exist_ok=True)
        os.makedirs('live_bfm_processed_csv', exist_ok=True)

        # BFM pipeline components
        self.bfm_collector = None
        self.bfm_extractor = BFMExtractor(
            tshark_path=TSHARK_PATH,
            csv_dir='live_bfm_raw_csv'
        )
        self.bfm_preprocessor = BFMPreprocessor(
            dir='live_bfm_processed_csv'
        )

    def _launch_tcpdump_explicit(self, verify=True):
        """
        Issue the saved explicit tcpdump command on the router. Returns True
        if `ps w | grep '[t]cpdump'` shows the process within 1 s.
        Uses time-based rotation (-G 1 -W 10) for sub-second freshness and -U
        for unbuffered packet flush.
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

                # Create fresh BFMCollector. Explicit tcpdump command below uses
                # time-based rotation (-G 1 -W 10) so filesize is unused; filecount
                # matches the ring buffer size.
                self.bfm_collector = BFMCollector(
                    host=self.host,
                    username=self.user,
                    password=self.password,
                    local_pcap_dir='live_bfm_pcap',
                    filename='live_bfm',
                    filesize=1,
                    filecount=10
                )

                # Establish connection
                self.bfm_collector.connect()

                # Optionally start traffic generation (disabled for monostatic phone experiments)
                if ENABLE_TRAFFIC_GENERATION:
                    print("[Connection] Starting iperf3 traffic generator...")
                    self.bfm_collector.run_iperf3()

                    print("[Connection] Starting 1 kHz ping for more traffic...")
                    self.bfm_collector.run_command("ping -i 0.001 192.168.1.1 > /dev/null 2>&1 &")
                    print("[Connection] ✅ Traffic generation enabled ")
                else:
                    print("[Connection] ⚠️ Traffic generation DISABLED ")
                    print("[Connection] Relying on natural WiFi traffic from devices (phone, etc.)")

                # Wait 5 s after traffic-gen so iperf3+ping settle before capture.
                print("[Connection] Waiting 5 s before starting tcpdump...")
                time.sleep(5)

                # Launch tcpdump with explicit -G 1 -W 10 -U command.
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
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
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

                # Every 10 loops (~1 second @ 0.1s sleep), verify tcpdump is still running
                if loop_count % 10 == 0:
                    try:
                        output, _ = self.bfm_collector.run_command("pgrep -f 'tcpdump -i mon0'")
                        if output:
                            print(f"[Download] tcpdump health check OK (PID: {output.strip()})")
                        else:
                            print("[Download] ⚠️ tcpdump NOT running! Attempting to restart...")
                            self._launch_tcpdump_explicit(verify=True)
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
                    remote_files = sftp.listdir('/tmp/')
                    pcap_files = [f for f in remote_files if f.startswith('bfm_capture')]

                    if not pcap_files:
                        # Debug: Show what files ARE in /tmp/
                        tmp_files_sample = [f for f in remote_files[:20]]  # First 20 files
                        print(f"[Download] No pcap files found on router (checked for 'bfm_capture*')")
                        print(f"[Download] Files in /tmp/ (sample): {tmp_files_sample}")
                        processed_files.clear()
                        single_file_tracker.clear()
                        time.sleep(0.1)
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
                        file_stats.append({
                            'path': remote_path,
                            'mtime': stat.st_mtime,
                            'size': stat.st_size
                        })

                    if not file_stats:
                        print("[Download] No valid file stats collected")
                        time.sleep(0.1)
                        sftp.close()
                        continue

                    file_stats.sort(key=lambda x: x['mtime'])
                    print(f"[Download] File stats: {[(f['path'].split('/')[-1], f['size'], 'bytes') for f in file_stats]}")

                    # ─── Stall watchdog ────────────────────────────────────
                    # If total bytes across all bfm_capture* on the router have
                    # not grown for STALL_THRESHOLD seconds, re-issue the
                    # explicit tcpdump command. Catches silent tcpdump death,
                    # mon0 dropout, or client disassociation.
                    total_bytes = sum(f['size'] for f in file_stats)
                    now = time.time()
                    if total_bytes != self._last_total_bytes:
                        self._last_total_bytes = total_bytes
                        self._last_growth_ts = now
                    elif now - self._last_growth_ts >= self.STALL_THRESHOLD:
                        idle = now - self._last_growth_ts
                        print(f"[Watchdog] No pcap growth for {idle:.1f}s — re-issuing tcpdump.")
                        self._launch_tcpdump_explicit(verify=True)
                        processed_files.clear()
                        single_file_tracker.clear()
                        self._last_growth_ts = now

                    candidates = []

                    if len(file_stats) >= 2:
                        print(f"[Download] Multiple files detected, downloading all except newest")
                        candidates = file_stats[:-1]  # Everything except the newest (likely active) file
                        single_file_tracker.clear()
                    else:
                        sole = file_stats[0]
                        tracker = single_file_tracker.get(sole['path'])
                        if tracker and tracker['mtime'] == sole['mtime'] and tracker['size'] == sole['size']:
                            tracker['stable_checks'] += 1
                            print(f"[Download] Single file {sole['path'].split('/')[-1]} stable check {tracker['stable_checks']}/2")
                        else:
                            single_file_tracker[sole['path']] = {
                                'mtime': sole['mtime'],
                                'size': sole['size'],
                                'stable_checks': 1
                            }
                            tracker = single_file_tracker[sole['path']]
                            print(f"[Download] Single file {sole['path'].split('/')[-1]} tracking started (size: {sole['size']} bytes)")

                        # 1 check: download immediately if non-empty (highest sensitivity)
                        if tracker['stable_checks'] >= 1 and tracker['size'] > 100:
                            print(f"[Download] Single file {sole['path'].split('/')[-1]} is stable, adding to candidates")
                            candidates.append(sole)
                        else:
                            print(f"[Download] Single file {sole['path'].split('/')[-1]} not yet stable ({tracker['stable_checks']}/1 checks, size {sole['size']}B)")

                    print(f"[Download] Candidates for download: {len(candidates)}")
                    now_processed = []

                    for entry in candidates:
                        remote_path = entry['path']
                        size_bytes = entry['size']

                        if size_bytes < 100:
                            print(f"[Download] Skipping {remote_path.split('/')[-1]} - too small ({size_bytes} bytes)")
                            continue

                        last_mtime = processed_files.get(remote_path)
                        if last_mtime == entry['mtime']:
                            print(f"[Download] Skipping {remote_path.split('/')[-1]} - already processed (mtime: {entry['mtime']})")
                            continue

                        print(f"[Download] Processing candidate {remote_path.split('/')[-1]} ({size_bytes} bytes)")

                        local_filename = f"live_bfm{self.total_downloaded}.pcap"
                        local_path = Path('live_bfm_pcap') / local_filename

                        try:
                            sftp.get(remote_path, str(local_path))
                            processed_files[remote_path] = entry['mtime']
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

                time.sleep(0.1)  # Check 10 times per second for new files

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
                csv_raw_path = Path('live_bfm_raw_csv') / (pcap_path.stem + '.csv')

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
                csv_processed_path = Path('live_bfm_processed_csv') / csv_raw_path.name

                try:
                    self.bfm_preprocessor.process_file(str(csv_raw_path), str(csv_processed_path))
                except Exception as e:
                    print(f"[Processing] Preprocessing error: {e}")
                    continue

                if not csv_processed_path.exists():
                    continue

                # Step 3: Convert to mag/phase and add to buffer
                df_processed = pd.read_csv(csv_processed_path)

                if df_processed.empty:
                    continue

                df_mag_phase = convert_real_imag_to_mag_phase(df_processed, self.mag_cols, self.phase_cols)

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
                print(f"[Processing] ✓ +{num_packets} packets (buffer: {len(self.packet_buffer)}, total: {self.total_packets})")

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

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def convert_real_imag_to_mag_phase(df, mag_cols, phase_cols):
    """Convert real/imag columns to magnitude/phase"""
    import re

    # Get all ratio columns
    ratio_cols = [col for col in df.columns if 'Ratio_Real' in col or 'Ratio_Imag' in col]

    if not ratio_cols:
        return pd.DataFrame()

    # Extract subcarrier indices
    subcarrier_pattern = re.compile(r'SCIDX_(-?\d+)_Ratio_(Real|Imag)')
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
            real_col = f'SCIDX_{sc_idx}_Ratio_Real'
            imag_col = f'SCIDX_{sc_idx}_Ratio_Imag'

            if real_col in df.columns and imag_col in df.columns:
                real_val = row[real_col]
                imag_val = row[imag_col]

                # Compute magnitude and phase
                mag = np.sqrt(real_val**2 + imag_val**2)
                phase = np.arctan2(imag_val, real_val)

                row_features[f'SCIDX_{sc_idx}_Mag'] = mag
                row_features[f'SCIDX_{sc_idx}_Phase'] = phase

        mag_phase_data.append(row_features)

    return pd.DataFrame(mag_phase_data)

def compute_doppler_features(df_processed):
    """
    Compute Doppler shift features from processed BFM data for motion detection.

    Args:
        df_processed: DataFrame with Ratio_Real and Ratio_Imag columns

    Returns:
        dict with Doppler features: mean_velocity, max_velocity, velocity_std, energy
    """
    from scipy.signal import detrend

    try:
        # Get all ratio columns
        ratio_cols = [col for col in df_processed.columns if 'Ratio_Real' in col or 'Ratio_Imag' in col]

        if not ratio_cols or len(df_processed) < 3:
            return {
                'doppler_mean_velocity': 0.0,
                'doppler_max_velocity': 0.0,
                'doppler_velocity_std': 0.0,
                'doppler_energy': 0.0
            }

        # Build complex matrix from real/imag ratios
        import re
        subcarrier_pattern = re.compile(r'SCIDX_(-?\d+)_Ratio_(Real|Imag)')
        subcarriers = set()

        for col in ratio_cols:
            match = subcarrier_pattern.match(col)
            if match:
                subcarriers.add(int(match.group(1)))

        subcarriers = sorted(subcarriers)

        # Build complex matrix (packets x subcarriers)
        complex_data = []
        for _, row in df_processed.iterrows():
            packet_complex = []
            for sc_idx in subcarriers:
                real_col = f'SCIDX_{sc_idx}_Ratio_Real'
                imag_col = f'SCIDX_{sc_idx}_Ratio_Imag'

                if real_col in df_processed.columns and imag_col in df_processed.columns:
                    real_val = row[real_col]
                    imag_val = row[imag_col]
                    complex_val = complex(real_val, imag_val)
                    packet_complex.append(complex_val)

            complex_data.append(packet_complex)

        complex_matrix = np.array(complex_data)  # Shape: (packets, subcarriers)

        # Clean up NaN/inf
        complex_matrix = np.nan_to_num(complex_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if complex_matrix.size == 0:
            return {
                'doppler_mean_velocity': 0.0,
                'doppler_max_velocity': 0.0,
                'doppler_velocity_std': 0.0,
                'doppler_energy': 0.0
            }

        # Compute phase for each subcarrier over time
        phases = np.angle(complex_matrix)
        phases = np.unwrap(phases, axis=0)  # Unwrap phase over time
        phases = np.nan_to_num(phases, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute phase velocity (derivative = Doppler shift)
        phase_velocity = np.diff(phases, axis=0)  # Shape: (packets-1, subcarriers)
        phase_velocity = np.nan_to_num(phase_velocity, nan=0.0, posinf=0.0, neginf=0.0)

        # Aggregate across subcarriers
        mean_phase_vel = np.mean(phase_velocity, axis=1)  # Mean across subcarriers
        mean_phase_vel = np.nan_to_num(mean_phase_vel, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute features
        mean_velocity = float(np.mean(np.abs(mean_phase_vel)))
        max_velocity = float(np.max(np.abs(mean_phase_vel)))
        velocity_std = float(np.std(mean_phase_vel))

        # Energy in Doppler spectrum
        if len(mean_phase_vel) >= 3:
            detrended = detrend(mean_phase_vel)
            detrended = np.nan_to_num(detrended, nan=0.0, posinf=0.0, neginf=0.0)
            fft_result = np.fft.fft(detrended)
            power = np.abs(fft_result) ** 2
            energy = float(np.sum(power))
        else:
            energy = 0.0

        return {
            'doppler_mean_velocity': mean_velocity,
            'doppler_max_velocity': max_velocity,
            'doppler_velocity_std': velocity_std,
            'doppler_energy': energy
        }

    except Exception as e:
        print(f"[Doppler] Error computing features: {e}")
        return {
            'doppler_mean_velocity': 0.0,
            'doppler_max_velocity': 0.0,
            'doppler_velocity_std': 0.0,
            'doppler_energy': 0.0
        }

def detect_activity_doppler(doppler_features):
    """
    Simple rule-based activity detection using Doppler features.

    Args:
        doppler_features: dict with Doppler features

    Returns:
        str: 'Standing' or 'Walking'
    """
    # Thresholds (tuned empirically)
    WALKING_VELOCITY_THRESHOLD = 0.1  # Mean velocity threshold
    WALKING_ENERGY_THRESHOLD = 0.01   # Energy threshold

    mean_vel = doppler_features['doppler_mean_velocity']
    energy = doppler_features['doppler_energy']

    # Walking typically has higher velocity and energy
    if mean_vel > WALKING_VELOCITY_THRESHOLD or energy > WALKING_ENERGY_THRESHOLD:
        return 'Walking'
    else:
        return 'Standing'

def make_prediction(model, pca, feature_cols, df_window, df_processed_window=None, use_doppler=False):
    """Make a prediction from a window of BFM data

    Args:
        model: Trained keras model expecting (batch, WINDOW_SIZE, 5)
        pca: PCA transformer
        feature_cols: List of feature column names
        df_window: DataFrame with mag/phase features
        df_processed_window: DataFrame with Ratio_Real/Imag (for Doppler)
        use_doppler: If True, use Doppler-based detection instead of model
    """
    if df_window is None or len(df_window) < WINDOW_SIZE:
        return None, None, None, None

    try:
        # Method 1: Doppler-based detection (faster, works with small windows)
        if use_doppler and df_processed_window is not None and len(df_processed_window) >= 3:
            doppler_features = compute_doppler_features(df_processed_window)
            predicted_activity = detect_activity_doppler(doppler_features)

            # Create pseudo-probabilities for visualization
            if predicted_activity == 'Walking':
                probs = np.array([0.2, 0.8])  # [Standing, Walking]
                confidence = 0.8
            else:
                probs = np.array([0.8, 0.2])
                confidence = 0.8

            return predicted_activity, confidence, probs, doppler_features

        # Method 2: Model-based detection (requires exact window size)
        # Reindex to ensure all features are present
        df_features = df_window.reindex(columns=feature_cols, fill_value=0.0)

        # Convert to numpy (WINDOW_SIZE, 468)
        feature_matrix = df_features.to_numpy(dtype=np.float32)

        # Clean NaN/inf
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply PCA to each row (WINDOW_SIZE, 468) -> (WINDOW_SIZE, 5)
        pca_features = pca.transform(feature_matrix)

        # Reshape for model (1, WINDOW_SIZE, 5)
        model_input = pca_features.reshape(1, WINDOW_SIZE, -1)

        # Predict
        probs = model.predict(model_input, verbose=0)[0]
        predicted_class = np.argmax(probs)
        confidence = float(probs[predicted_class])
        predicted_activity = ACTIVITY_LABELS[predicted_class]

        return predicted_activity, confidence, probs, None

    except Exception as e:
        print(f"[Prediction] Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_live_history(history):
    """Plot prediction history"""
    if not history:
        return None

    fig, ax = plt.subplots(figsize=(10, 3))

    timestamps = [h['timestamp'] for h in history]
    predictions = [h['prediction'] for h in history]
    confidences = [h['confidence'] for h in history]

    colors = [ACTIVITY_COLORS[pred] for pred in predictions]

    ax.bar(range(len(timestamps)), confidences, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_ylabel('Confidence', fontsize=10)
    ax.set_title('Live Prediction History', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACTIVITY_COLORS['Standing'], label='Standing'),
        Patch(facecolor=ACTIVITY_COLORS['Walking'], label='Walking'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig

def plot_probability_over_time(history):
    """Plot probability changes over time"""
    if not history or len(history) < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 3))

    timestamps = list(range(len(history)))
    standing_probs = [h['probs'][0] for h in history]
    walking_probs = [h['probs'][1] for h in history]

    ax.plot(timestamps, standing_probs, label='Standing', color=ACTIVITY_COLORS['Standing'], linewidth=2)
    ax.plot(timestamps, walking_probs, label='Walking', color=ACTIVITY_COLORS['Walking'], linewidth=2)
    ax.fill_between(timestamps, standing_probs, alpha=0.3, color=ACTIVITY_COLORS['Standing'])
    ax.fill_between(timestamps, walking_probs, alpha=0.3, color=ACTIVITY_COLORS['Walking'])

    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title('Activity Probability Over Time', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig

def plot_bfm_magnitude_timeseries(packet_buffer, num_packets=50, num_subcarriers=5):
    """
    Plot BFM magnitude signals over time for selected subcarriers

    Args:
        packet_buffer: deque of packet dictionaries with mag/phase features
        num_packets: Number of recent packets to show
        num_subcarriers: Number of subcarriers to plot (spaced evenly)
    """
    if len(packet_buffer) < 2:
        return None

    # Get recent packets
    recent_packets = list(packet_buffer)[-num_packets:]

    # Extract subcarrier indices from column names
    sample_packet = recent_packets[0]
    mag_cols = [col for col in sample_packet.keys() if 'Mag' in col]

    if not mag_cols:
        return None

    # Select evenly-spaced subcarriers to plot
    step = max(1, len(mag_cols) // num_subcarriers)
    selected_cols = mag_cols[::step][:num_subcarriers]

    # Extract time-series data
    time_steps = list(range(len(recent_packets)))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot each subcarrier
    for col in selected_cols:
        magnitudes = [packet[col] for packet in recent_packets]
        scidx = col.split('_')[1]  # Extract subcarrier index
        ax.plot(time_steps, magnitudes, label=f'SC {scidx}', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Packet Number', fontsize=10)
    ax.set_ylabel('Magnitude', fontsize=10)
    ax.set_title('BFM Magnitude Time-Series (Selected Subcarriers)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_bfm_phase_timeseries(packet_buffer, num_packets=50, num_subcarriers=5):
    """
    Plot BFM phase signals over time for selected subcarriers

    Args:
        packet_buffer: deque of packet dictionaries with mag/phase features
        num_packets: Number of recent packets to show
        num_subcarriers: Number of subcarriers to plot (spaced evenly)
    """
    if len(packet_buffer) < 2:
        return None

    # Get recent packets
    recent_packets = list(packet_buffer)[-num_packets:]

    # Extract subcarrier indices from column names
    sample_packet = recent_packets[0]
    phase_cols = [col for col in sample_packet.keys() if 'Phase' in col]

    if not phase_cols:
        return None

    # Select evenly-spaced subcarriers to plot
    step = max(1, len(phase_cols) // num_subcarriers)
    selected_cols = phase_cols[::step][:num_subcarriers]

    # Extract time-series data
    time_steps = list(range(len(recent_packets)))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot each subcarrier
    for col in selected_cols:
        phases = [packet[col] for packet in recent_packets]
        scidx = col.split('_')[1]  # Extract subcarrier index
        ax.plot(time_steps, phases, label=f'SC {scidx}', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Packet Number', fontsize=10)
    ax.set_ylabel('Phase (radians)', fontsize=10)
    ax.set_title('BFM Phase Time-Series (Selected Subcarriers)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig

def plot_doppler_velocity_timeseries(processed_buffer, num_packets=50):
    """
    Plot Doppler velocity (phase derivative) over time

    Args:
        processed_buffer: deque of packet dictionaries with Ratio_Real/Imag
        num_packets: Number of recent packets to show
    """
    if len(processed_buffer) < 3:
        return None

    # Get recent packets
    recent_packets = list(processed_buffer)[-num_packets:]

    # Extract ratio columns
    sample_packet = recent_packets[0]
    ratio_real_cols = [col for col in sample_packet.keys() if 'Ratio_Real' in col]
    ratio_imag_cols = [col for col in sample_packet.keys() if 'Ratio_Imag' in col]

    if not ratio_real_cols or not ratio_imag_cols:
        return None

    # Build complex matrix
    complex_matrix = []
    for packet in recent_packets:
        packet_complex = []
        for real_col, imag_col in zip(ratio_real_cols, ratio_imag_cols):
            complex_val = complex(packet[real_col], packet[imag_col])
            packet_complex.append(complex_val)
        complex_matrix.append(packet_complex)

    complex_matrix = np.array(complex_matrix)  # Shape: (packets, subcarriers)

    # Compute phase velocity (Doppler shift)
    phases = np.angle(complex_matrix)
    phases = np.unwrap(phases, axis=0)  # Unwrap phase over time
    phase_velocity = np.diff(phases, axis=0)  # Derivative

    # Average across subcarriers
    mean_velocity = np.mean(phase_velocity, axis=1)

    # Time steps
    time_steps = list(range(len(mean_velocity)))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # Plot 1: Mean velocity over time
    ax1.plot(time_steps, mean_velocity, color='#e74c3c', linewidth=2, label='Mean Velocity')
    ax1.fill_between(time_steps, mean_velocity, alpha=0.3, color='#e74c3c')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Packet Number', fontsize=10)
    ax1.set_ylabel('Phase Velocity (rad/packet)', fontsize=10)
    ax1.set_title('Doppler Velocity Time-Series', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)

    # Plot 2: Absolute velocity (motion magnitude)
    abs_velocity = np.abs(mean_velocity)
    ax2.plot(time_steps, abs_velocity, color='#3498db', linewidth=2, label='Motion Magnitude')
    ax2.fill_between(time_steps, abs_velocity, alpha=0.3, color='#3498db')
    ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Walking Threshold')
    ax2.set_xlabel('Packet Number', fontsize=10)
    ax2.set_ylabel('|Velocity| (rad/packet)', fontsize=10)
    ax2.set_title('Motion Magnitude (used for activity detection)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig

def plot_bfm_heatmap(packet_buffer, num_packets=50):
    """
    Plot BFM magnitude as a heatmap (subcarriers × time)

    Args:
        packet_buffer: deque of packet dictionaries with mag/phase features
        num_packets: Number of recent packets to show
    """
    if len(packet_buffer) < 2:
        return None

    # Get recent packets
    recent_packets = list(packet_buffer)[-num_packets:]

    # Extract magnitude columns
    sample_packet = recent_packets[0]
    mag_cols = sorted([col for col in sample_packet.keys() if 'Mag' in col])

    if not mag_cols:
        return None

    # Build matrix: rows = subcarriers, cols = time
    magnitude_matrix = []
    for col in mag_cols:
        time_series = [packet[col] for packet in recent_packets]
        magnitude_matrix.append(time_series)

    magnitude_matrix = np.array(magnitude_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(magnitude_matrix, aspect='auto', cmap='viridis', interpolation='nearest')

    # Set labels
    ax.set_xlabel('Packet Number', fontsize=10)
    ax.set_ylabel('Subcarrier Index', fontsize=10)
    ax.set_title('BFM Magnitude Heatmap (Subcarrier × Time)', fontsize=12, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude', fontsize=10)

    # Set tick labels (show every 10th subcarrier)
    y_ticks = range(0, len(mag_cols), max(1, len(mag_cols) // 10))
    y_labels = [mag_cols[i].split('_')[1] for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)

    plt.tight_layout()
    return fig

def plot_signal_constellation(packet_buffer, num_packets=10):
    """
    Plot constellation diagram (Real vs Imag) from recent packets
    Shows the complex signal distribution

    Args:
        packet_buffer: deque of packet dictionaries
        num_packets: Number of recent packets to include
    """
    if len(packet_buffer) < 2:
        return None

    # Get recent packets
    recent_packets = list(packet_buffer)[-num_packets:]

    # We need to reconstruct Real/Imag from Mag/Phase
    sample_packet = recent_packets[0]
    mag_cols = sorted([col for col in sample_packet.keys() if 'Mag' in col])
    phase_cols = sorted([col for col in sample_packet.keys() if 'Phase' in col])

    if not mag_cols or not phase_cols:
        return None

    # Collect all complex values
    real_vals = []
    imag_vals = []

    for packet in recent_packets:
        for mag_col, phase_col in zip(mag_cols, phase_cols):
            mag = packet[mag_col]
            phase = packet[phase_col]
            real = mag * np.cos(phase)
            imag = mag * np.sin(phase)
            real_vals.append(real)
            imag_vals.append(imag)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(real_vals, imag_vals, alpha=0.3, s=10, color='#3498db')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Real Component', fontsize=10)
    ax.set_ylabel('Imaginary Component', fontsize=10)
    ax.set_title(f'Signal Constellation Diagram (Last {num_packets} packets)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config("Live Activity Detection", layout="wide", page_icon="📡")
    st.title("📡 Live Real-time Activity Detection")
    st.caption("Continuously streaming BFM data from router for real-time activity recognition")

    # Sidebar configuration
    st.sidebar.markdown("### Configuration")

    # Traffic mode indicator
    if ENABLE_TRAFFIC_GENERATION:
        st.sidebar.info("🔄 **Traffic Mode:** \n\n Router generates its own traffic (ping + iperf3)")
    else:
        st.sidebar.warning("📱 **Traffic Mode:** \n\nRelying on natural device traffic (phone, etc.)\n\n⚠️ Make sure device is actively using WiFi!")

    st.sidebar.markdown("---")

    # Model selection
    st.sidebar.markdown("### Model Selection")
    model_type = st.sidebar.radio(
        "Select model:",
        options=['nofoil', 'open'],
        format_func=lambda x: {
            'nofoil': '🏠 No Foil (default)',
            'open': '🌐 Open Environment'
        }[x],
        help="Choose the model trained for your environment type"
    )

    # Load model
    with st.spinner(f"Loading {model_type} model and PCA..."):
        model, pca, feature_cols, (mag_cols, phase_cols) = load_model_and_pca(model_type)

    if model is None or pca is None:
        st.error("Failed to load model or PCA. Cannot proceed.")
        st.stop()

    st.success(f"✅ Model ({model_type}) and PCA loaded successfully")

    # Initialize session state
    # Check if collector needs to be recreated (e.g., after code update)
    if 'collector' not in st.session_state or not hasattr(st.session_state.collector, 'get_connection_status'):
        # Stop old collector if it exists
        if 'collector' in st.session_state:
            try:
                st.session_state.collector.stop_collection()
            except:
                pass

        # Create new collector with updated code
        st.session_state.collector = LiveDataCollector(
            ROUTER_HOST, ROUTER_USER, ROUTER_PASSWORD,
            mag_cols, phase_cols
        )
        st.session_state.is_collecting = False

    if 'live_history' not in st.session_state:
        st.session_state.live_history = []

    if 'is_collecting' not in st.session_state:
        st.session_state.is_collecting = False

    # Diagnostics section
    with st.expander("🔧 Router Diagnostics & Setup", expanded=False):
        st.markdown("### Check Router Connection")
        st.info("**Note:** BFMCollector uses paramiko for SSH, which automatically handles host key verification.")

        diag_col1, diag_col2 = st.columns(2)

        with diag_col1:
            if st.button("🔍 Test Connection via BFMCollector"):
                with st.spinner("Testing SSH connection..."):
                    try:
                        # Use BFMCollector for proper paramiko connection
                        test_collector = BFMCollector(
                            host=ROUTER_HOST,
                            username=ROUTER_USER,
                            password=ROUTER_PASSWORD,
                            local_pcap_dir='test_bfm_pcap'
                        )
                        test_collector.connect()

                        # Run a test command
                        output, error = test_collector.run_command('echo "Connection successful"')

                        test_collector.close()

                        if output:
                            st.success(f"✅ SSH connection successful!\n```\n{output}\n```")
                        else:
                            st.warning("⚠️ Connection established but no output received")

                    except Exception as e:
                        st.error(f"❌ Connection failed: {e}")

        with diag_col2:
            if st.button("📊 Check tcpdump Process"):
                with st.spinner("Checking router processes..."):
                    try:
                        # Use BFMCollector to check tcpdump
                        test_collector = BFMCollector(
                            host=ROUTER_HOST,
                            username=ROUTER_USER,
                            password=ROUTER_PASSWORD,
                            local_pcap_dir='test_bfm_pcap'
                        )
                        test_collector.connect()

                        # Check if tcpdump is running
                        output, error = test_collector.run_command("ps | grep '[t]cpdump'")

                        # Check for pcap files
                        files_output, _ = test_collector.run_command("ls -lh /tmp/bfm_capture* 2>/dev/null || echo 'No files found'")

                        test_collector.close()

                        if output:
                            st.success(f"✅ tcpdump is running:\n```\n{output}\n```")
                        else:
                            st.warning("⚠️ tcpdump is NOT running on router")

                        st.info(f"📁 Files on router:\n```\n{files_output}\n```")

                    except Exception as e:
                        st.error(f"❌ Check failed: {e}")

        st.markdown("### How It Works")
        st.markdown("""
        **BFMCollector automatically handles:**
        1. ✅ SSH connection with paramiko (auto-accepts host keys)
        2. ✅ Starts tcpdump on mon0 interface with BFM filter
        3. ✅ Downloads pcap files via SFTP
        4. ✅ Cleans up remote files after download

        **You just need to:**
        - Click "▶️ Start Live Detection" and it will connect automatically
        - Wait for connection status to show "🟢 Connected ✓"
        - System will retry every 2 seconds if connection fails
        """)

        st.markdown("### Manual Setup (Optional)")
        st.code("""
# If you want to manually start tcpdump on the router:
ssh root@192.168.1.1

# Start tcpdump (BFMCollector does this automatically)
tcpdump -i mon0 -w /tmp/bfm_capture -W 2 -C 1 'wlan[24] == 21' &

# Verify it's running
ps | grep tcpdump
        """, language="bash")

        st.info("**Note**: Make sure your router's WiFi interface is in monitor mode (mon0) and actively receiving WiFi traffic with BFM reports.")

    # Control panel
    st.markdown("---")
    st.markdown("## 🎮 Control Panel")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("▶️ Start Live Detection", use_container_width=True,
                    disabled=st.session_state.is_collecting):
            st.session_state.collector.start_collection()
            st.session_state.is_collecting = True
            st.rerun()

    with col2:
        if st.button("⏹️ Stop Detection", use_container_width=True,
                    disabled=not st.session_state.is_collecting):
            st.session_state.collector.stop_collection()
            st.session_state.is_collecting = False
            st.rerun()

    with col3:
        if st.button("🔄 Clear History", use_container_width=True):
            st.session_state.live_history = []
            st.rerun()

    # Connection status - dual-queue metrics
    status_cols = st.columns(6)

    # Show connection status from collector
    conn_status = st.session_state.collector.get_connection_status()
    if st.session_state.collector.connected:
        status_cols[0].metric("Status", "🟢 Connected")
    elif st.session_state.is_collecting:
        status_cols[0].metric("Status", "🟡 Connecting...")
    else:
        status_cols[0].metric("Status", "🔴 STOPPED")

    # Dual-queue stats
    collector = st.session_state.collector
    buffer_size = collector.get_buffer_size()
    queue_size = len(collector.download_queue) if hasattr(collector, 'download_queue') else 0

    status_cols[1].metric("Download Queue", f"{queue_size}")
    status_cols[2].metric("Packet Buffer", f"{buffer_size}")

    # Throughput stats
    total_downloaded = collector.total_downloaded if hasattr(collector, 'total_downloaded') else 0
    total_processed = collector.total_processed if hasattr(collector, 'total_processed') else 0
    total_packets = collector.total_packets if hasattr(collector, 'total_packets') else 0

    status_cols[3].metric("Downloaded", f"{total_downloaded}")
    status_cols[4].metric("Processed", f"{total_processed}")
    status_cols[5].metric("Total Packets", f"{total_packets}")

    # Show required vs current
    need_packets = WINDOW_SIZE - buffer_size if buffer_size < WINDOW_SIZE else 0
    if need_packets > 0:
        st.info(f"📊 Need {need_packets} more packets for prediction (window size: {WINDOW_SIZE})")
    else:
        st.success(f"✅ Ready for predictions! Using Doppler-based detection with {WINDOW_SIZE}-packet windows")

    # Show warnings if not collecting enough data
    if st.session_state.is_collecting and buffer_size == 0:
        st.warning("""
        ⚠️ **No packets in buffer!**

        Possible issues:
        1. tcpdump not running on router
        2. No WiFi traffic with BFM reports
        3. Wrong file path or permissions

        **Please check the Router Diagnostics above** 👆
        """)

    # Main detection loop
    if st.session_state.is_collecting:
        st.markdown("---")

        # Get latest windows (both mag/phase and processed for Doppler)
        df_window = st.session_state.collector.get_latest_packets(WINDOW_SIZE)
        df_processed_window = st.session_state.collector.get_latest_processed(WINDOW_SIZE)

        if df_window is not None:
            # Make prediction using Doppler detection (faster with small windows)
            use_doppler = True  # Use Doppler for quick response with 5-packet window
            predicted_activity, confidence, probs, doppler_features = make_prediction(
                model, pca, feature_cols, df_window, df_processed_window, use_doppler=use_doppler
            )

            if predicted_activity is not None:
                # Store in history
                current_time = datetime.now().strftime("%H:%M:%S")
                st.session_state.live_history.append({
                    'timestamp': current_time,
                    'prediction': predicted_activity,
                    'confidence': confidence,
                    'probs': probs
                })

                # Limit history size
                if len(st.session_state.live_history) > 50:
                    st.session_state.live_history = st.session_state.live_history[-50:]

                # Display current prediction
                st.markdown("## 🎯 Current Prediction")

                pred_cols = st.columns(4)

                # Predicted activity
                activity_color = ACTIVITY_COLORS[predicted_activity]
                pred_cols[0].markdown(f"### Predicted")
                pred_cols[0].markdown(f"<h1 style='color: {activity_color};'>{predicted_activity}</h1>",
                                    unsafe_allow_html=True)

                # Confidence
                pred_cols[1].markdown(f"### Confidence")
                pred_cols[1].markdown(f"<h1 style='color: {'#27ae60' if confidence > 0.7 else '#f39c12'};'>{confidence:.1%}</h1>",
                                    unsafe_allow_html=True)

                # Time
                pred_cols[2].markdown(f"### Time")
                pred_cols[2].markdown(f"<h1 style='color: #95a5a6;'>{current_time}</h1>",
                                    unsafe_allow_html=True)

                # Window size
                pred_cols[3].markdown(f"### Window")
                pred_cols[3].markdown(f"<h1 style='color: #3498db;'>{len(df_window)}</h1>",
                                    unsafe_allow_html=True)

                # Probability breakdown
                st.markdown("### 📊 Class Probabilities")
                prob_cols = st.columns(len(ACTIVITY_LABELS))
                for idx, (class_id, activity_name) in enumerate(ACTIVITY_LABELS.items()):
                    prob = probs[class_id]
                    prob_cols[idx].metric(
                        activity_name,
                        f"{prob:.1%}",
                        delta=None
                    )
        else:
            st.info(f"⏳ Collecting data... Need {WINDOW_SIZE} packets for prediction. Currently have {buffer_size} packets.")

        # Signal Visualizations (show if we have enough packets) - MOVED INSIDE collecting block
        if buffer_size >= 10:
            st.markdown("---")
            st.markdown("## 📡 Real-Time Signal Analysis")

            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 BFM Magnitude",
                "📈 BFM Phase",
                "🌊 Doppler Velocity",
                "🔥 Heatmap",
                "⭕ Constellation"
            ])

            with tab1:
                st.markdown("### BFM Magnitude Time-Series")
                st.caption("Shows magnitude changes over time for selected subcarriers. Magnitude indicates signal strength.")

                # Control: number of packets to show
                col1, col2 = st.columns([3, 1])
                with col2:
                    num_packets_mag = st.slider("Packets to display", 10, 100, 50, key="mag_packets")
                    num_subcarriers_mag = st.slider("Subcarriers", 3, 10, 5, key="mag_subcarriers")

                fig_mag = plot_bfm_magnitude_timeseries(
                    collector.packet_buffer,
                    num_packets=num_packets_mag,
                    num_subcarriers=num_subcarriers_mag
                )
                if fig_mag:
                    st.pyplot(fig_mag)
                    plt.close(fig_mag)  # Prevent memory leak
                else:
                    st.info("Collecting more packets for visualization...")

            with tab2:
                st.markdown("### BFM Phase Time-Series")
                st.caption("Shows phase changes over time for selected subcarriers. Phase changes indicate motion.")

                # Control: number of packets to show
                col1, col2 = st.columns([3, 1])
                with col2:
                    num_packets_phase = st.slider("Packets to display", 10, 100, 50, key="phase_packets")
                    num_subcarriers_phase = st.slider("Subcarriers", 3, 10, 5, key="phase_subcarriers")

                fig_phase = plot_bfm_phase_timeseries(
                    collector.packet_buffer,
                    num_packets=num_packets_phase,
                    num_subcarriers=num_subcarriers_phase
                )
                if fig_phase:
                    st.pyplot(fig_phase)
                    plt.close(fig_phase)  # Prevent memory leak
                else:
                    st.info("Collecting more packets for visualization...")

            with tab3:
                st.markdown("### Doppler Velocity Analysis")
                st.caption("Phase velocity (derivative) shows motion. Higher velocity = more movement (walking).")

                # Control: number of packets to show
                col1, col2 = st.columns([3, 1])
                with col2:
                    num_packets_doppler = st.slider("Packets to display", 10, 100, 50, key="doppler_packets")

                fig_doppler = plot_doppler_velocity_timeseries(
                    collector.processed_buffer,
                    num_packets=num_packets_doppler
                )
                if fig_doppler:
                    st.pyplot(fig_doppler)
                    plt.close(fig_doppler)  # Prevent memory leak

                    # Show current Doppler features if available
                    if df_processed_window is not None and len(df_processed_window) >= 3:
                        st.markdown("#### Current Doppler Features")
                        doppler_features = compute_doppler_features(df_processed_window)

                        feat_cols = st.columns(4)
                        feat_cols[0].metric("Mean Velocity", f"{doppler_features['doppler_mean_velocity']:.4f}")
                        feat_cols[1].metric("Max Velocity", f"{doppler_features['doppler_max_velocity']:.4f}")
                        feat_cols[2].metric("Std Dev", f"{doppler_features['doppler_velocity_std']:.4f}")
                        feat_cols[3].metric("Energy", f"{doppler_features['doppler_energy']:.2e}")
                else:
                    st.info("Need at least 3 packets for Doppler analysis...")

            with tab4:
                st.markdown("### BFM Magnitude Heatmap")
                st.caption("Visualizes all subcarriers over time. Brighter colors = stronger signal. Vertical patterns indicate motion.")

                # Control: number of packets to show
                col1, col2 = st.columns([3, 1])
                with col2:
                    num_packets_heatmap = st.slider("Packets to display", 20, 100, 50, key="heatmap_packets")

                fig_heatmap = plot_bfm_heatmap(
                    collector.packet_buffer,
                    num_packets=num_packets_heatmap
                )
                if fig_heatmap:
                    st.pyplot(fig_heatmap)
                    plt.close(fig_heatmap)  # Prevent memory leak
                else:
                    st.info("Collecting more packets for visualization...")

            with tab5:
                st.markdown("### Signal Constellation Diagram")
                st.caption("Complex plane representation (Real vs Imaginary). Shows signal distribution and quality.")

                # Control: number of packets to show
                col1, col2 = st.columns([3, 1])
                with col2:
                    num_packets_const = st.slider("Packets to display", 5, 50, 10, key="const_packets")

                fig_const = plot_signal_constellation(
                    collector.packet_buffer,
                    num_packets=num_packets_const
                )
                if fig_const:
                    st.pyplot(fig_const)
                    plt.close(fig_const)  # Prevent memory leak
                else:
                    st.info("Collecting more packets for visualization...")

        # Auto-refresh
        time.sleep(UPDATE_INTERVAL)
        st.rerun()

    # Visualizations (Prediction history - outside collecting block)
    if st.session_state.live_history:
        st.markdown("---")
        st.markdown("## 📈 Live Visualizations")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Prediction History")
            fig_history = plot_live_history(st.session_state.live_history)
            if fig_history:
                st.pyplot(fig_history)
                plt.close(fig_history)  # Prevent memory leak

        with col_right:
            st.markdown("### Probability Trends")
            fig_probs = plot_probability_over_time(st.session_state.live_history)
            if fig_probs:
                st.pyplot(fig_probs)
                plt.close(fig_probs)  # Prevent memory leak

        # Statistics
        st.markdown("---")
        st.markdown("### 📊 Session Statistics")

        from collections import Counter
        prediction_counts = Counter(h['prediction'] for h in st.session_state.live_history)

        stat_cols = st.columns(len(ACTIVITY_LABELS) + 1)
        stat_cols[0].metric("Total Predictions", len(st.session_state.live_history))

        for idx, (class_id, activity_name) in enumerate(ACTIVITY_LABELS.items()):
            count = prediction_counts.get(activity_name, 0)
            percentage = (count / len(st.session_state.live_history) * 100) if len(st.session_state.live_history) > 0 else 0
            stat_cols[idx + 1].metric(
                activity_name,
                f"{count}",
                delta=f"{percentage:.1f}%"
            )

if __name__ == "__main__":
    main()
