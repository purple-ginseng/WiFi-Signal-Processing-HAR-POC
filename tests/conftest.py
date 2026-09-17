"""Shared fixtures: router-free fakes, synthetic BFM data, Tk app builders."""
import os
import sys
import threading
from collections import deque
from pathlib import Path

# Silence TensorFlow's import-time log spam before main_gui* pulls it in.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest
import tkinter as tk
from tkinter import messagebox

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PCAP = FIXTURES / "bfm_sample.pcap"


# ─── Synthetic BFM data ────────────────────────────────────────────────────────
DEFAULT_SCIDX = [-122, -121, -3, -1, 1, 3, 121, 122]


def make_ratio_frame(n_rows=5, scidx=DEFAULT_SCIDX, t0=1_700_000_000.0, fs=10.0, seed=0):
    """Processed-style frame: timestamp, MACs, SCIDX_i_Ratio_Real/Imag."""
    rng = np.random.default_rng(seed)
    data = {
        "timestamp": t0 + np.arange(n_rows) / fs,
        "receiver_address": ["aa:bb:cc:dd:ee:01"] * n_rows,
        "transmitter_address": ["aa:bb:cc:dd:ee:02"] * n_rows,
    }
    for i in scidx:
        data[f"SCIDX_{i}_Ratio_Real"] = rng.normal(1.0, 0.1, n_rows)
        data[f"SCIDX_{i}_Ratio_Imag"] = rng.normal(0.0, 0.1, n_rows)
    return pd.DataFrame(data)


def make_packet_rows(n_rows=40, scidx=DEFAULT_SCIDX, t0=1_700_000_000.0, fs=10.0,
                     pc_timestamp=None, seed=0):
    """Rows shaped like LiveDataCollector.packet_buffer entries (mag/phase)."""
    import time as _time
    rng = np.random.default_rng(seed)
    if pc_timestamp is None:
        pc_timestamp = _time.time()
    rows = []
    for k in range(n_rows):
        row = {"timestamp": t0 + k / fs, "pc_timestamp": pc_timestamp}
        for i in scidx:
            row[f"SCIDX_{i}_Ratio_Mag"] = float(rng.uniform(0.5, 2.0))
            row[f"SCIDX_{i}_Ratio_Phase"] = float(rng.uniform(-6.0, 6.0))
        rows.append(row)
    return rows


# ─── Fakes ─────────────────────────────────────────────────────────────────────
class FakeCollector:
    """Stand-in for LiveDataCollector with the attributes the GUI reads."""

    def __init__(self, running=True, connected=True, status="Connected ✓"):
        self.running = running
        self.connected = connected
        self.connection_status = status
        self.packet_buffer = deque(maxlen=10000)
        self.processed_buffer = deque(maxlen=10000)
        self.total_downloaded = 0
        self.total_processed = 0
        self.total_packets = 0
        self.calls = []
        self.clock_offset = 0.0

    def start_collection(self):
        self.calls.append("start_collection")
        self.running = True

    def stop_collection(self):
        self.calls.append("stop_collection")
        self.running = False
        self.connected = False

    def get_clock_offset(self):
        return self.clock_offset

    def wait_for_capture_through(self, target_ts, timeout=25.0):
        self.calls.append(("wait_for_capture_through", target_ts))
        return True


class FakeModel:
    """predict_proba stub returning a fixed P(walking)."""

    def __init__(self, p_walk=0.9, n_features=None):
        self.p_walk = p_walk
        self.n_features_in_ = n_features
        self.seen = []

    def predict_proba(self, X):
        self.seen.append(np.asarray(X))
        n = len(X)
        return np.tile([1 - self.p_walk, self.p_walk], (n, 1))


# ─── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def dialogs(monkeypatch):
    """Record tkinter.messagebox calls instead of blocking on a dialog."""
    calls = []

    def _rec(kind):
        def fn(title, message, **kw):
            calls.append((kind, title, str(message)))
            return None
        return fn

    for kind in ("showerror", "showinfo", "showwarning"):
        monkeypatch.setattr(messagebox, kind, _rec(kind))
    return calls


@pytest.fixture(scope="session")
def display():
    """Skip GUI tests when no X display can be opened."""
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display available for Tk: {e}")
    root.destroy()
    return True


def _teardown(app):
    """Destroy the window, then cancel leftover timers so Tcl stays quiet."""
    try:
        app.destroy()
    except tk.TclError:
        pass
    try:
        for cb in app.tk.call("after", "info"):
            app.tk.call("after", "cancel", cb)
    except tk.TclError:
        pass


def _make_app(module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = module.MainApp()
    app.withdraw()
    app.update()
    return app


@pytest.fixture
def gui_app(display, dialogs, tmp_path, monkeypatch):
    """main_gui.MainApp inside a temp cwd, torn down after the test."""
    import main_gui
    app = _make_app(main_gui, tmp_path, monkeypatch)
    yield app
    _teardown(app)


@pytest.fixture
def gui3_app(display, dialogs, tmp_path, monkeypatch):
    """main_gui3.MainApp inside a temp cwd, torn down after the test."""
    import main_gui3
    app = _make_app(main_gui3, tmp_path, monkeypatch)
    yield app
    _teardown(app)


def run_until(pred, timeout=10.0, interval=0.05):
    """Poll pred() until truthy or timeout; returns the final value."""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return bool(pred())


def pump_until(app, pred, timeout=5.0, interval_ms=20):
    """Run app.mainloop() until pred() is truthy or timeout; returns pred()."""
    import time
    result = {"ok": False}
    deadline = time.time() + timeout

    def check():
        if pred():
            result["ok"] = True
            app.quit()
        elif time.time() > deadline:
            app.quit()
        else:
            app.after(interval_ms, check)

    # Worker threads call app.after(); that needs the main thread in mainloop.
    app.after(interval_ms, check)
    app.mainloop()
    return result["ok"]
