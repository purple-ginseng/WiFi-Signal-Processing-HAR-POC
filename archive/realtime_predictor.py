import os
import time
import joblib
import threading
import traceback
import tkinter as tk
from tkinter import ttk
import numpy as np
from scapy.all import rdpcap, Dot11
import multiprocessing
from queue import Empty  # This works with multiprocessing.Queue too

from archive.viewer import VTKModelViewer

CSI_PCAP_PATH = './data/csi.pcap'
CHUNK_SIZE = 500

def launch_vtk(cmd_queue):
    try:
        viewer = VTKModelViewer()
        viewer.load_model("poses/sitting.obj")

        def poll_commands():
            while True:
                try:
                    cmd = cmd_queue.get_nowait()
                    print(f"[DEBUG] Received from queue: {cmd}")
                    if cmd.startswith("load_model:"):
                        label = cmd.split(":", 1)[1].lower().replace(" ", "_")
                        viewer.load_model(f"poses/{label}.obj")
                except Empty:
                    pass  # No message this cycle
                except Exception as e:
                    print(f"[poll_commands ERROR] {e}")
                    traceback.print_exc()
                time.sleep(0.5)

        threading.Thread(target=poll_commands, daemon=True).start()
        viewer.start()
    except Exception as e:
        print(f"[launch_vtk ERROR] {e}")
        traceback.print_exc()


def read_latest_csi(file_path):
    try:
        print(f"[DEBUG] Reading PCAP from: {file_path}")
        packets = rdpcap(file_path)
        csi = [len(p) for p in packets if p.haslayer(Dot11)]
        print(f"[DEBUG] Total packets: {len(packets)} | CSI frames: {len(csi)}")
        return np.array(csi[-CHUNK_SIZE:]) if len(csi) >= CHUNK_SIZE else None
    except Exception as e:
        print(f"[ERROR] Reading pcap: {e}")
        return None


class RealTimeCSIApp:
    def __init__(self, root, cmd_queue):
        self.root = root
        self.cmd_queue = cmd_queue
        self.root.title("Real-Time CSI Activity Recognition")
        self.predicted_action = tk.StringVar(value="Waiting for data...")
        self.running = True

        ttk.Label(root, text="Detected Activity:", font=("Helvetica", 36)).pack(pady=10)
        self.label = ttk.Label(root, textvariable=self.predicted_action, font=("Helvetica", 48), foreground="blue")
        self.label.pack(pady=5)

        # Load models
        try:
            self.pca = joblib.load('pca.pkl')
            self.clf = joblib.load('model.pkl')
            self.label_encoder = joblib.load('label_encoder.pkl')
            print("[INFO] Models loaded successfully.")
        except FileNotFoundError as e:
            print(f"[FATAL] Missing model file: {e}")
            self.running = False
            return

        self.last_label = None
        self.force_refresh = True

        # Load default model
        self.load_model("sitting")

        threading.Thread(target=self.predict_loop, daemon=True).start()

    def load_model(self, action_label):
        if self.cmd_queue:
            action_label = action_label.lower().replace(" ", "_")
            if self.force_refresh or action_label != self.last_label:
                print(f"[QUEUE] Sending load_model:{action_label}")
                self.cmd_queue.put(f"load_model:{action_label}")
                self.last_label = action_label

    def predict_loop(self):
        while self.running:
            data = read_latest_csi(CSI_PCAP_PATH)
            if data is not None:
                try:
                    feature = self.pca.transform([data])
                    prediction = self.clf.predict(feature)
                    label = self.label_encoder.inverse_transform(prediction)[0]

                    print(f"[INFO] Detected: {label}")
                    self.predicted_action.set(label)
                    self.load_model(label)
                except Exception as e:
                    print(f"[ERROR] Prediction failed: {e}")
                    traceback.print_exc()
            else:
                self.predicted_action.set("Waiting for data...")
            time.sleep(1)

    def stop(self):
        self.running = False


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    cmd_queue = multiprocessing.Queue()
    vtk_process = multiprocessing.Process(target=launch_vtk, args=(cmd_queue,))
    vtk_process.start()

    root = tk.Tk()
    app = RealTimeCSIApp(root, cmd_queue)

    def on_close():
        app.stop()
        vtk_process.terminate()
        vtk_process.join()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
