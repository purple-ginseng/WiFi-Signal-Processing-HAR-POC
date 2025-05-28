import os
import glob
import time
import threading
import subprocess
import platform

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import numpy as np
import pandas as pd
import joblib
from scapy.all import rdpcap, Dot11

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from scapy.all import RadioTap


import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR      = './data'
PCAP_PATH     = './data/wifisignal.pcap'
PCA_COMPONENTS= 50
TEST_SIZE     = 0.3
EPOCHS        = 20
BATCH_SIZE_TF = 32    # for TF training fit
# ───────────────────────────────────────────────────────────────────────────────

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Unified wifisignal Toolkit - WiFi Human Pose")
        self.geometry("1000x800")

        # Shared state
        self.chunk_size = tk.IntVar(value=300)
        self.pred_thresh = tk.DoubleVar(value=0.5)  # probability threshold
        self.pca          = None
        self.label_encoder= None
        self.model        = None
        self._stop_pred   = threading.Event()

        self.pcap_transfer_thread = None
        self._stop_pcap_transfer  = threading.Event()

        self._build_ui()

    def _start_pcap_transfer(self):
        """Start background transfer of wifisignal.pcap from router"""
        if self.pcap_transfer_thread and self.pcap_transfer_thread.is_alive():
            return  # Already running
        self._stop_pcap_transfer.clear()
        self.pcap_transfer_thread = threading.Thread(target=self._transfer_loop, daemon=True)
        self.pcap_transfer_thread.start()

    def _transfer_loop(self):
        while not self._stop_pcap_transfer.is_set():
            try:
                # Use native scp command (requires sshpass installed), or use a fallback
                if platform.system() == "Windows":
                    # Windows: fallback using plink.exe or rewrite into Python (below simpler way)
                    os.system('sshpass -p "123456" scp -O root@192.168.1.1:/tmp/csi.pcap ./data/wifisignal.pcap')
                else:
                    subprocess.run(
                        ["sshpass", "-p", "123456", "scp", "-O", "root@192.168.1.1:/tmp/csi.pcap", "./data/wifisignal.pcap"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
            except Exception as e:
                print("[Transfer Error]", e)
            time.sleep(1)  # wait before next fetch

    def _stop_pcap_transfer_loop(self):
        """Stop background pcap fetching"""
        self._stop_pcap_transfer.set()

    def _build_ui(self):
        # ── Global Controls ─────────────────────────────────────────────
        global_frame = ttk.LabelFrame(self, text="Global Settings", padding=10)
        global_frame.pack(fill="x", padx=10, pady=5)

        # Chunk Size slider + live display
        ttk.Label(global_frame, text="Chunk Size:")\
            .grid(row=0, column=0, sticky="w", pady=2)
        self.chunk_scale = ttk.Scale(
            global_frame,
            from_=50, to=1000,
            variable=self.chunk_size,
            orient="horizontal"
        )
        self.chunk_scale.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        # Label that shows the IntVar value automatically
        self.chunk_value_label = ttk.Label(
            global_frame,
            textvariable=self.chunk_size,
            width=5
        )
        self.chunk_value_label.grid(row=0, column=2, padx=5, pady=2)

        global_frame.columnconfigure(1, weight=1)

        # Prediction Threshold slider + live display
        ttk.Label(global_frame, text="Prediction Threshold:")\
            .grid(row=1, column=0, sticky="w", pady=2)
        self.thresh_scale = ttk.Scale(
            global_frame,
            from_=0.1, to=1.0,
            variable=self.pred_thresh,
            orient="horizontal"
        )
        self.thresh_scale.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        # Label we’ll update via a trace to show two decimals
        self.thresh_value_label = ttk.Label(global_frame, text=f"{self.pred_thresh.get():.2f}", width=5)
        self.thresh_value_label.grid(row=1, column=2, padx=5, pady=2)
        # whenever pred_thresh changes, update the text
        self.pred_thresh.trace_add("write", self._update_thresh_label)

        # ────────────────────────────────────────────────────────────────

        # ── Notebook Tabs ──────────────────────────────────────────────
        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=10, pady=10)

        # Data Collection Tab
        col_frame = ttk.Frame(tabs)
        tabs.add(col_frame, text="Data Collection")
        self._build_collection_ui(col_frame)

        # Training Tab
        train_frame = ttk.Frame(tabs)
        tabs.add(train_frame, text="Model Training")
        self._build_training_ui(train_frame)

        # Prediction Tab
        pred_frame = ttk.Frame(tabs)
        tabs.add(pred_frame, text="Real‑time Prediction")
        self._build_prediction_ui(pred_frame)

    def _update_thresh_label(self, *args):
        # format to two decimal places
        val = self.pred_thresh.get()
        self.thresh_value_label.config(text=f"{val:.2f}")
        
    # ── Data Collection ────────────────────────────────────────────────
    def _build_collection_ui(self, parent):
        ttk.Label(parent, text="Label for this session:").pack(anchor="w", pady=(10,0))
        self.collect_label = ttk.Entry(parent)
        self.collect_label.pack(fill="x", padx=10)

        # ── Duration setting ───────────────────────────────
        ttk.Label(parent, text="Duration (seconds):").pack(anchor="w", pady=(10,0))
        self.duration_entry = ttk.Entry(parent)
        self.duration_entry.insert(0, "120")
        self.duration_entry.pack(fill="x", padx=10)
        # ──────────────────────────────────────────────────

        self.collect_btn = ttk.Button(parent, text="Start Collection", command=self._on_collect)
        self.collect_btn.pack(pady=10)

        self.collect_msg = ttk.Label(parent, text="", foreground="green")
        self.collect_msg.pack()

    def extract_wifisignal_from_radiotap(self, pkt):
        try:
            if pkt.haslayer(RadioTap):
                raw = bytes(pkt.getlayer(RadioTap))

                # hex inspection
                offset = 50
                length = 60

                if len(raw) >= offset + length:
                    wifisignal_bytes = raw[offset:offset + length] 
                    return list(wifisignal_bytes)  
            return None
        except Exception as e:
            print("Failed to parse wifisignal:", e)
            return None


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

        self.collect_msg.config(text="Collecting...", foreground="blue")
        self.collect_btn.config(state="disabled")

        # Start fetching PCAP in background
        self._start_pcap_transfer()

        threading.Thread(
            target=self._do_collection_wrapper,
            args=(lbl, duration),
            daemon=True
        ).start()

    def _do_collection_wrapper(self, label, duration):
        """Wrapper to stop pcap transfer after collection finishes"""
        self._do_collection(label, duration)
        self._stop_pcap_transfer_loop()

    
    def _do_collection(self, label, duration):
        import datetime, os, time
        from scapy.all import rdpcap, Dot11

        start_ts = time.time()
        last_count = 0
        # Collect raw radiotap payloads for each packet
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
                            print("DATA", data)
                            wifisignal_records.append(data)

            except Exception as e:
                print("[COLLECT ERROR]", e)

            time.sleep(1)

        # Ensure output directory exists
        if not os.path.isdir(DATA_DIR):
            os.makedirs(DATA_DIR)

        # Build filename with timestamp
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"wifisignal_data_{label}_{ts}.csv"

        # Create DataFrame: one row per packet
        if wifisignal_records:
            num_features = len(wifisignal_records[0])
            df = pd.DataFrame(
                wifisignal_records,
                columns=[f"pkt{i}" for i in range(num_features)]
            )
            df["label"] = label
            df.to_csv(os.path.join(DATA_DIR, fname), index=False)
            saved = len(wifisignal_records)
        else:
            saved = 0

        # Update UI
        self.collect_msg.config(
            text=f"Collected {saved} packets over {duration}s → saved to {fname}",
            foreground="green"
        )
        self.collect_btn.config(state="normal")


    # ── Model Training ───────────────────────────────────────────────
    def _build_training_ui(self, parent):
        self.train_btn = ttk.Button(parent, text="Start Training", command=self._on_train)
        self.train_btn.pack(pady=10)

        # place for confusion matrix
        self.fig = Figure(figsize=(5,4))
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # classification report text
        self.report_txt = scrolledtext.ScrolledText(parent, width=40, height=20)
        self.report_txt.pack(side="right", fill="y", padx=10, pady=10)

    def _on_train(self):
        self.train_btn.config(state="disabled")
        threading.Thread(target=self._do_training, daemon=True).start()

    def _do_training(self):
        # load all CSVs
        files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        data, labels = [], []
        for f in files:
            df = pd.read_csv(f)
            wifisignalz = self.chunk_size.get()
            # flatten windows stored pkt0..pktN
            X = df[[f"pkt{i}" for i in range(wifisignalz)]].values
            y = df["label"].values
            data.append(X); labels.append(y)
        if not data:
            messagebox.showerror("No Data", "No CSV files found in data folder.")
            self.train_btn.config(state="normal")
            return

        X = np.vstack(data)
        y = np.hstack(labels)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=TEST_SIZE, random_state=42
        )

        # PCA
        # n_components = min(X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=PCA_COMPONENTS)
        X_train_p = pca.fit_transform(X_train)
        X_test_p  = pca.transform(X_test)

        # build model
        num_classes = len(le.classes_)
        model = models.Sequential([
            layers.Input(shape=(PCA_COMPONENTS,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        cb = [
            callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]

        model.fit(
            X_train_p, y_train,
            validation_split=0.2,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE_TF,
            callbacks=cb,
            verbose=1
        )

        # evaluate
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

        # save artifacts
        model.save("tf_model.keras")
        joblib.dump(pca, "pca_tf.pkl")
        joblib.dump(le, "label_encoder_tf.pkl")

        # update state
        self.pca           = pca
        self.label_encoder = le
        self.model         = model

        messagebox.showinfo("Training Complete", "Model, PCA, and encoder saved.")
        self.train_btn.config(state="normal")

    # ── Real‑time Prediction ────────────────────────────────────────
    def _build_prediction_ui(self, parent):
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=10)
        self.pred_btn = ttk.Button(controls, text="Start Prediction", command=self._start_pred)
        self.pred_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(controls, text="Stop Prediction", command=self._stop_pred.set)
        self.stop_btn.pack(side="left", padx=5)

        self.pred_label = ttk.Label(parent, text="Waiting...", font=("Helvetica", 36))
        self.pred_label.pack(pady=50)

    def _start_pred(self):
        if not (self.pca and self.model and self.label_encoder):
            messagebox.showerror("Model Missing", "Please train a model first.")
            return
        self._stop_pred.clear()
        threading.Thread(target=self._prediction_loop, daemon=True).start()

    def _prediction_loop(self):
        wifisignalz = self.chunk_size.get()
        while not self._stop_pred.is_set():
            try:
                packets = rdpcap(PCAP_PATH)
                wifisignal = np.array([len(p) for p in packets if p.haslayer(Dot11)])
                if len(wifisignal) >= wifisignalz:
                    window = wifisignal[-wifisignalz:].reshape(1, -1)
                    feat   = self.pca.transform(window)
                    probs  = self.model.predict(feat)[0]
                    idx    = np.argmax(probs)
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
