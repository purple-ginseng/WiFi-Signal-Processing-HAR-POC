import os
import time
import csv
import threading
from datetime import datetime
from scapy.all import rdpcap, Dot11

import tkinter as tk
from tkinter import ttk

# === Configurable constants ===
CSI_PCAP_PATH = './data/csi.pcap'
CHUNK_SIZE = 1000               # You can change to 3000 or 5000
SLEEP_INTERVAL = 0.2            # Capture frequency in seconds

def read_pcap(file_path):
    csi_data = []
    if os.path.exists(file_path):
        packets = rdpcap(file_path)
        for packet in packets:
            if packet.haslayer(Dot11):
                csi_data.append(len(packet))
    return csi_data


class CSICollectorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Wi-Fi CSI Posture Collector")

        self.is_collecting = False
        self.selected_label = tk.StringVar()
        self.status = tk.StringVar(value="Idle")
        self.timer_var = tk.StringVar(value="Time remaining: 60s")
        self.count_var = tk.StringVar(value="Captured datapoints: 0")

        # UI Components
        ttk.Label(root, text="Posture Label:").grid(row=0, column=0, padx=5, pady=5)
        self.label_combo = ttk.Combobox(root, textvariable=self.selected_label)
        self.label_combo['values'] = [
            'Standing', 'Sitting', 'Walking',
            'Sleeping - Left', 'Sleeping - Right', 'Sleeping - Up'
        ]
        self.label_combo.grid(row=0, column=1, padx=5, pady=5)
        self.label_combo.current(0)

        ttk.Button(root, text="Start", command=self.start_collection).grid(row=1, column=0, padx=10, pady=10)
        ttk.Button(root, text="Stop", command=self.stop_collection).grid(row=1, column=1, padx=10, pady=10)

        ttk.Label(root, textvariable=self.status).grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Label(root, textvariable=self.timer_var).grid(row=3, column=0, columnspan=2, pady=5)
        ttk.Label(root, textvariable=self.count_var).grid(row=4, column=0, columnspan=2, pady=5)

    def start_collection(self):
        if not self.is_collecting:
            self.is_collecting = True
            self.status.set("Collecting...")
            self.timer_var.set("Time remaining: 60s")
            self.count_var.set("Captured datapoints: 0")
            threading.Thread(target=self.collect_data, daemon=True).start()

    def stop_collection(self):
        self.is_collecting = False
        self.status.set("Stopped")

    def collect_data(self):
        label = self.selected_label.get().lower().replace(" ", "_")
        filename = f"data/csi_data_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        start_time = time.time()
        datapoint_counter = 0

        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "packet_length", "label"])

            while self.is_collecting:
                elapsed = time.time() - start_time
                remaining = max(0, int(60 - elapsed))
                self.timer_var.set(f"Time remaining: {remaining}s")

                # Read and chunk CSI data
                csi = read_pcap(CSI_PCAP_PATH)
                chunks = [csi[i:i + CHUNK_SIZE] for i in range(0, len(csi), CHUNK_SIZE)]

                for chunk in chunks:
                    if len(chunk) == CHUNK_SIZE:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for val in chunk:
                            writer.writerow([timestamp, val, label])
                        datapoint_counter += 1
                        self.count_var.set(f"Captured datapoints: {datapoint_counter}")

                if elapsed >= 60:
                    break

                time.sleep(SLEEP_INTERVAL)

        self.is_collecting = False
        self.status.set(f"Saved to {filename}")
        self.timer_var.set("Time remaining: 0s")


# Run the UI
if __name__ == "__main__":
    root = tk.Tk()
    app = CSICollectorUI(root)
    root.mainloop()
