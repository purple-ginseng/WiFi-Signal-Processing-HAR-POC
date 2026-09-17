"""main_gui.MainApp on a real Tk window, router faked. Needs a display."""
import io
import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest

import main_gui
from conftest import make_ratio_frame, run_until

pytestmark = pytest.mark.gui


# ─── Fakes for the bfmtool trio main_gui wires together ────────────────────────
class FakeBFMCollector:
    fail_on = None                      # name of the method that should raise

    def __init__(self, **kw):
        self.kw = kw
        self.calls = []
        self.filename = None
        self.collected = set()
        self.offset = 0.0

    def _do(self, name):
        self.calls.append(name)
        if name == self.fail_on:
            raise ConnectionError(f"{name} failed")

    def connect(self): self._do("connect")
    def sync_clock(self): self._do("sync_clock")
    def run_iperf3(self): self._do("run_iperf3")
    def kill_iperf3(self): self._do("kill_iperf3")
    def close(self): self._do("close")
    def run_tcpdump(self): self._do("run_tcpdump")
    def stop_pcap_collection(self): self._do("stop_pcap_collection")
    def get_clock_offset(self): return self.offset
    def get_collected_files(self): return set(self.collected)


class FakeExtractor:
    def __init__(self, **kw):
        self.kw = kw
        self.extracted = set()
        self.tshark_path = "/nonexistent/tshark"

    def extract(self, files):
        for f in files:
            self.extracted.add(Path("raw") / (Path(f).stem + ".csv"))

    def get_extracted_files(self): return set(self.extracted)


class FakePreprocessor:
    def __init__(self, dir, frame_factory=None, **kw):
        self.dir = Path(dir)
        self.frame_factory = frame_factory or (lambda: make_ratio_frame(5))

    def process(self, paths):
        self.dir.mkdir(exist_ok=True)
        for p in paths:
            self.frame_factory().to_csv(self.dir / Path(p).name, index=False)


@pytest.fixture
def fake_bfmtool(monkeypatch):
    """Swap the bfmtool classes main_gui instantiates for fakes."""
    made = {}
    def mk_collector(**kw): made["collector"] = FakeBFMCollector(**kw); return made["collector"]
    def mk_extractor(**kw): made["extractor"] = FakeExtractor(**kw); return made["extractor"]
    def mk_pre(**kw): made["preprocessor"] = FakePreprocessor(**kw); return made["preprocessor"]
    monkeypatch.setattr(main_gui, "BFMCollector", mk_collector)
    monkeypatch.setattr(main_gui, "BFMExtractor", mk_extractor)
    monkeypatch.setattr(main_gui, "BFMPreprocessor", mk_pre)
    monkeypatch.setattr(FakeBFMCollector, "fail_on", None)
    return made


# ─── Construction / validation ─────────────────────────────────────────────────
def test_app_constructs(gui_app):
    app = gui_app
    assert app.source_mode.get() == "RSSI-PCAP"
    assert app.bfm_is_setup is False and app.bfm_collector is None
    assert app.bfm_setup_btn.cget("text") == "Setup BFM"
    assert app.bfm_collected == set() and app.bfm_extracted == set()


def _fill(app, subject="alice", activity="walking", description="", duration="120"):
    for entry, value in ((app.collect_subject, subject), (app.collect_activity, activity),
                         (app.collect_description, description), (app.duration_entry, duration)):
        entry.delete(0, "end")
        entry.insert(0, value)


@pytest.mark.parametrize("subject,activity", [("", "walking"), ("alice", ""), ("  ", "x")])
def test_collect_requires_subject_and_activity(gui_app, dialogs, subject, activity):
    _fill(gui_app, subject=subject, activity=activity)
    gui_app._on_collect()
    assert dialogs == [("showerror", "Input Error",
                        "Please enter both a subject and an activity.")]
    assert str(gui_app.collect_btn.cget("state")) == "normal"


def test_collect_requires_integer_duration(gui_app, dialogs):
    _fill(gui_app, duration="abc")
    gui_app._on_collect()
    assert dialogs == [("showerror", "Input Error", "Duration must be an integer.")]


def test_collect_dispatches_bfm_worker(gui_app, dialogs):
    app = gui_app
    got = []
    app._do_bfm_collection = lambda *a: got.append(a)
    app.source_mode.set("BFM-PCAP")
    _fill(app, subject=" alice ", activity="walking", description=" open ", duration="3")
    app._on_collect()
    assert run_until(lambda: got, timeout=3)
    assert got == [("alice", "walking", "open", 3)]
    assert app.collect_msg.cget("text") == "Collecting BFM..."
    assert str(app.collect_btn.cget("state")) == "disabled"


def test_threshold_label_tracks_slider(gui_app):
    gui_app.pred_thresh.set(0.25)
    assert gui_app.thresh_value_label.cget("text") == "0.25"


# ─── Setup / close state machine ───────────────────────────────────────────────
def test_setup_success(gui_app, dialogs, fake_bfmtool):
    app = gui_app
    app._toggle_bfm_setup()

    c = fake_bfmtool["collector"]
    assert c.calls == ["connect", "sync_clock", "run_iperf3"]
    assert c.kw["host"] == "192.168.1.1" and c.kw["local_pcap_dir"] == "bfm_pcap"
    assert fake_bfmtool["extractor"].kw["csv_dir"] == "bfm_raw_csv"
    assert fake_bfmtool["preprocessor"].dir == Path("bfm_processed_csv")
    assert app.bfm_is_setup is True
    assert app.bfm_collector is c
    assert app.bfm_setup_btn.cget("text") == "Close BFM"
    assert app.collect_msg.cget("text") == "BFM connection established."
    assert dialogs == []


@pytest.mark.parametrize("failing", ["connect", "sync_clock", "run_iperf3"])
def test_setup_failure_tears_down(gui_app, dialogs, fake_bfmtool, failing):
    app = gui_app
    FakeBFMCollector.fail_on = failing
    app._toggle_bfm_setup()

    c = fake_bfmtool["collector"]
    assert c.calls[-1] == "close"          # half-built collector is closed
    assert app.bfm_collector is None
    assert app.bfm_extractor is None and app.bfm_preprocessor is None
    assert app.bfm_is_setup is False
    assert app.bfm_setup_btn.cget("text") == "Setup BFM"
    assert dialogs == [("showerror", "BFM Setup Failed", f"{failing} failed")]


def test_close_success(gui_app, dialogs, fake_bfmtool):
    app = gui_app
    app._toggle_bfm_setup()
    c = fake_bfmtool["collector"]
    c.calls.clear()

    app._toggle_bfm_setup()
    assert c.calls == ["kill_iperf3", "close"]
    assert app.bfm_collector is None and app.bfm_is_setup is False
    assert app.bfm_setup_btn.cget("text") == "Setup BFM"
    assert app.collect_msg.cget("text") == "BFM connection closed."
    assert dialogs == []


def test_close_failure_still_resets(gui_app, dialogs, fake_bfmtool):
    app = gui_app
    app._toggle_bfm_setup()
    FakeBFMCollector.fail_on = "close"
    app._toggle_bfm_setup()
    assert app.bfm_collector is None and app.bfm_is_setup is False
    assert app.bfm_setup_btn.cget("text") == "Setup BFM"
    assert dialogs == [("showerror", "BFM Close Error", "close failed")]


def test_on_close_closes_active_bfm_session(gui_app, fake_bfmtool):
    app = gui_app
    app._toggle_bfm_setup()
    c = fake_bfmtool["collector"]
    c.calls.clear()
    app._on_close()
    assert c.calls == ["kill_iperf3", "close"]
    assert app._stop_pcap_transfer.is_set() and app._stop_csi.is_set()
    with pytest.raises(Exception):
        app.winfo_exists()


def test_on_close_without_setup_does_not_toggle(gui_app, fake_bfmtool):
    gui_app._on_close()
    assert "collector" not in fake_bfmtool


# ─── Labeled BFM session with the fake pipeline ────────────────────────────────
def test_bfm_collection_without_setup_reenables_button(gui_app):
    app = gui_app
    app.collect_btn.config(state="disabled")
    app._do_bfm_collection("alice", "walking", "", 1)
    assert str(app.collect_btn.cget("state")) == "normal"
    assert app.collect_msg.cget("text") == \
        "[BFM] Collection finished, but nothing was saved."


def test_bfm_collection_no_pcaps_reports_orange(gui_app, fake_bfmtool):
    app = gui_app
    app._toggle_bfm_setup()
    app._extract_bfm_for_plot = lambda p: []
    app._do_bfm_collection("alice", "walking", "", 1)
    assert "No pcap files were captured" in app.collect_msg.cget("text")
    assert str(app.collect_msg.cget("foreground")) == "orange"
    assert fake_bfmtool["collector"].calls[-1] == "stop_pcap_collection"


def test_bfm_collection_saves_session(gui_app, fake_bfmtool, tmp_path):
    app = gui_app
    app._toggle_bfm_setup()
    c = fake_bfmtool["collector"]
    app._extract_bfm_for_plot = lambda p: []

    start = time.time()
    # rows timestamped inside [start, start+1] so the window trim keeps them
    fake_bfmtool["preprocessor"].frame_factory = \
        lambda: make_ratio_frame(10, t0=start + 0.3, fs=50.0)
    c.collected = {Path("bfm_pcap") / "chunk0.pcap", Path("bfm_pcap") / "chunk1.pcap"}

    app._do_bfm_collection("alice", "walking", "open", 1)

    assert c.filename is not None and c.filename().startswith("bfm_data_alice_walking_open_")
    assert c.calls[3:] == ["run_tcpdump", "stop_pcap_collection"]
    assert app.bfm_collected == c.collected
    assert app.bfm_extracted == fake_bfmtool["extractor"].extracted

    ri = list((tmp_path / "bfm_real_imag_csv").glob("bfm_ri_data_alice_walking_open_*.csv"))
    mp = list((tmp_path / "bfm_mag_phase_csv").glob("bfm_mp_data_alice_walking_open_*.csv"))
    assert len(ri) == 1 and len(mp) == 1
    df_ri, df_mp = pd.read_csv(ri[0]), pd.read_csv(mp[0])
    assert len(df_ri) == 20                       # two chunks × 10 rows
    assert (df_mp["activity"] == "walking").all()
    for df in (df_ri, df_mp):
        assert (df["subject"] == "alice").all()
        cols = list(df.columns)
        assert cols.index("activity") == cols.index("subject") + 1
    assert list((tmp_path / "bfm_processed_csv").iterdir()) == []   # fragments removed

    assert app.collect_msg.cget("text").startswith("[BFM] Saved 20 rows")
    assert str(app.collect_msg.cget("foreground")) == "green"
    assert str(app.collect_btn.cget("state")) == "normal"
    assert app.progress["value"] == 0


def test_bfm_collection_second_session_only_uses_new_files(gui_app, fake_bfmtool, tmp_path):
    app = gui_app
    app._toggle_bfm_setup()
    c = fake_bfmtool["collector"]
    app._extract_bfm_for_plot = lambda p: []
    fake_bfmtool["preprocessor"].frame_factory = lambda: make_ratio_frame(4)

    c.collected = {Path("bfm_pcap") / "a.pcap"}
    app._do_bfm_collection("alice", "walking", "", 1)
    c.collected = {Path("bfm_pcap") / "a.pcap", Path("bfm_pcap") / "b.pcap"}
    app._do_bfm_collection("alice", "standing", "", 1)

    ri = sorted((tmp_path / "bfm_real_imag_csv").glob("*.csv"))
    assert len(ri) == 2
    frames = [pd.read_csv(p) for p in ri]
    assert all(len(df) == 4 for df in frames)   # b.pcap only, not a again
    assert sorted(df["activity"].iloc[0] for df in frames) == ["standing", "walking"]
    assert all((df["subject"] == "alice").all() for df in frames)


# ─── tshark parser for the live plot ───────────────────────────────────────────
TSHARK_V = """Frame 1: 300 bytes on wire
    Epoch Arrival Time: 1700000000.250000000
    SCIDX: -122, φ11: 10, ψ21: 3
    SCIDX: -121, φ11: 20, ψ21: 3
Frame 2: 300 bytes on wire
    Epoch Time: 1700000001.500000000
    SCIDX: -122, φ11: 5, ψ21: 3
Frame 3: 300 bytes on wire
    Epoch Time: 1700000002.000000000
"""


class _Proc:
    def __init__(self, text):
        self.stdout = io.StringIO(text)
        self.stderr = io.StringIO("")
    def wait(self): return 0


def test_extract_bfm_for_plot_parses_tshark_output(gui_app, monkeypatch, fake_bfmtool):
    app = gui_app
    app._toggle_bfm_setup()
    seen = {}
    def popen(cmd, **kw):
        seen["cmd"] = cmd
        return _Proc(TSHARK_V)
    monkeypatch.setattr(main_gui.subprocess, "Popen", popen)

    out = app._extract_bfm_for_plot(Path("bfm_pcap/x.pcap"))
    assert out == [(1700000000.25, 15.0), (1700000001.5, 5.0)]
    assert seen["cmd"][0] == "/nonexistent/tshark"
    assert "-Y" in seen["cmd"] and "wlan.fixed.category_code == 21" in seen["cmd"]


def test_extract_bfm_for_plot_without_extractor(gui_app):
    assert gui_app._extract_bfm_for_plot("x.pcap") == []


def test_extract_bfm_for_plot_swallows_errors(gui_app, monkeypatch, fake_bfmtool, capsys):
    app = gui_app
    app._toggle_bfm_setup()
    def popen(cmd, **kw): raise OSError("no tshark")
    monkeypatch.setattr(main_gui.subprocess, "Popen", popen)
    assert app._extract_bfm_for_plot("x.pcap") == []
    assert "[BFM EXTRACT ERROR] no tshark" in capsys.readouterr().out
