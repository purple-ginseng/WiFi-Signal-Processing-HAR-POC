"""main_gui3.MainApp on a real Tk window, router faked. Needs a display."""
import os
import shutil
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import main_gui3
from conftest import (SAMPLE_PCAP, FakeCollector, FakeModel, make_packet_rows,
                      make_ratio_frame, pump_until, run_until)

pytestmark = pytest.mark.gui


def _buttons(app):
    return app.bfm_setup_btn._widgets


# ─── Construction / widget wiring ──────────────────────────────────────────────
def test_app_constructs_with_bfm_defaults(gui3_app):
    app = gui3_app
    assert app.source_mode.get() == "BFM-PCAP"
    assert app.bfm_is_setup is False and app.bfm_collector is None
    assert app.status_var.get() == "⚪ Not set up"
    assert app.duration_entry.get() == "120"
    assert app.pred_label.cget("text") == "Waiting..."


def test_setup_button_mirrored_on_both_tabs(gui3_app):
    app = gui3_app
    assert len(_buttons(app)) == 2
    app.bfm_setup_btn.config(text="Close BFM", state="disabled")
    for b in _buttons(app):
        assert b.cget("text") == "Close BFM"
        assert str(b.cget("state")) == "disabled"


def test_widget_group_add_copies_current_state(gui3_app):
    import tkinter.ttk as ttk
    app = gui3_app
    app.bfm_setup_btn.config(text="Checking…", state="disabled")
    extra = app.bfm_setup_btn.add(ttk.Button(app, text="x"))
    assert extra.cget("text") == "Checking…"
    assert str(extra.cget("state")) == "disabled"


def test_threshold_label_tracks_slider(gui3_app):
    gui3_app.pred_thresh.set(0.73)
    assert gui3_app.thresh_value_label.cget("text") == "0.73"


def test_source_mode_toggles_plot_visibility(gui3_app):
    app = gui3_app
    csi = app.csi_canvas.get_tk_widget()
    doppler = app.bfm_doppler_canvas.get_tk_widget()

    assert csi.winfo_manager() == ""
    assert doppler.winfo_manager() == "pack"
    assert app.bfm_status_frame.winfo_manager() == "pack"

    app.source_mode.set("CSI-UDP")
    assert csi.winfo_manager() == "pack"
    assert doppler.winfo_manager() == ""
    assert app.bfm_status_frame.winfo_manager() == ""
    assert app.bfm_feat_frame.winfo_manager() == ""

    app.source_mode.set("BFM-PCAP")
    assert csi.winfo_manager() == ""
    assert doppler.winfo_manager() == "pack"


# ─── Collection input validation ───────────────────────────────────────────────
def _fill(app, subject="alice", activity="walking", description="", duration="120"):
    for entry, value in ((app.collect_subject, subject), (app.collect_activity, activity),
                         (app.collect_description, description), (app.duration_entry, duration)):
        entry.delete(0, "end")
        entry.insert(0, value)


@pytest.mark.parametrize("subject,activity", [("", "walking"), ("alice", ""), ("  ", "  ")])
def test_collect_requires_subject_and_activity(gui3_app, dialogs, subject, activity):
    app = gui3_app
    _fill(app, subject=subject, activity=activity)
    app._on_collect()
    assert dialogs == [("showerror", "Input Error",
                        "Please enter both a subject and an activity.")]
    assert str(app.collect_btn.cget("state")) == "normal"


def test_collect_requires_integer_duration(gui3_app, dialogs):
    _fill(gui3_app, duration="12.5")
    gui3_app._on_collect()
    assert dialogs == [("showerror", "Input Error", "Duration must be an integer.")]


def test_collect_dispatches_trimmed_args_to_bfm_worker(gui3_app, dialogs):
    app = gui3_app
    got = []
    app._do_bfm_collection = lambda *a: got.append(a)
    _fill(app, subject=" alice ", activity=" walking ", description=" d ", duration="7")
    app._on_collect()
    assert run_until(lambda: got, timeout=3)
    assert got == [("alice", "walking", "d", 7)]
    assert dialogs == []
    assert str(app.collect_btn.cget("state")) == "disabled"
    assert app.collect_msg.cget("text") == "Collecting BFM..."


def test_bfm_collection_without_setup_is_rejected(gui3_app):
    app = gui3_app
    app.collect_btn.config(state="disabled")
    app._do_bfm_collection("alice", "walking", "", 1)
    assert app.collect_msg.cget("text") == "Please click 'Setup BFM' first."
    assert str(app.collect_btn.cget("state")) == "normal"


# ─── Labeled session end-to-end with a fake collector ──────────────────────────
def _feed_processed(collector, delay, t_offset, n=12):
    """Push ratio rows into processed_buffer after the session has started."""
    def go():
        time.sleep(delay)
        base = time.time() + t_offset
        rows = make_ratio_frame(n, t0=base, fs=50.0).to_dict("records")
        collector.processed_buffer.extend(rows)
    threading.Thread(target=go, daemon=True).start()


def test_bfm_session_saves_trimmed_csvs(gui3_app, tmp_path):
    app = gui3_app
    fake = FakeCollector()
    app.bfm_collector = fake
    app.bfm_is_setup = True
    _feed_processed(fake, delay=0.2, t_offset=0.0)      # rows inside the window
    _feed_processed(fake, delay=0.25, t_offset=-100.0)  # rows well before start

    app._do_bfm_collection("alice", "walking", "open", 1)

    ri = list((tmp_path / "bfm_real_imag_csv").glob("bfm_ri_data_alice_walking_open_*.csv"))
    mp = list((tmp_path / "bfm_mag_phase_csv").glob("bfm_mp_data_alice_walking_open_*.csv"))
    assert len(ri) == 1 and len(mp) == 1

    df_ri = pd.read_csv(ri[0])
    assert len(df_ri) == 12                       # the -100 s rows were trimmed
    assert (df_ri["activity"] == "walking").all()
    assert (df_ri["subject"] == "alice").all()
    assert list(df_ri.columns).index("activity") == list(df_ri.columns).index("subject") + 1
    df_mp = pd.read_csv(mp[0])
    assert len(df_mp) == 12
    assert "SCIDX_-122_Ratio_Mag" in df_mp.columns
    assert (df_mp["subject"] == "alice").all()
    assert list(df_mp.columns).index("activity") == list(df_mp.columns).index("subject") + 1

    assert "start_collection" in fake.calls
    assert any(c[0] == "wait_for_capture_through" for c in fake.calls if isinstance(c, tuple))
    assert app.collect_msg.cget("text").startswith("Saved 12 BFM packets")
    assert str(app.collect_btn.cget("state")) == "normal"
    assert app.timer_label.cget("text") == "Time Remaining: 0s"
    assert fake.running   # streaming continues after a labeled session


def test_bfm_session_saves_untrimmed_when_clock_is_off(gui3_app, tmp_path, capsys):
    app = gui3_app
    fake = FakeCollector()
    fake.clock_offset = 5_000_000.0   # router thinks it's months ahead
    app.bfm_collector = fake
    _feed_processed(fake, delay=0.2, t_offset=0.0, n=5)

    app._do_bfm_collection("alice", "standing", "", 1)

    ri = list((tmp_path / "bfm_real_imag_csv").glob("*.csv"))
    assert len(ri) == 1
    df = pd.read_csv(ri[0])
    assert len(df) == 5
    assert df["subject"].tolist() == ["alice"] * 5
    assert df["activity"].tolist() == ["standing"] * 5
    assert "label" not in df.columns
    assert "check the router's clock" in capsys.readouterr().out


def test_bfm_session_with_no_packets_reports_orange(gui3_app, tmp_path):
    app = gui3_app
    app.bfm_collector = FakeCollector()
    app._do_bfm_collection("alice", "walking", "", 1)
    assert app.collect_msg.cget("text") == \
        "[BFM] Collection finished, but no packets were captured."
    assert str(app.collect_msg.cget("foreground")) == "orange"
    assert not (tmp_path / "bfm_real_imag_csv").exists()


def test_bfm_session_clears_stale_buffers_first(gui3_app):
    app = gui3_app
    fake = FakeCollector()
    fake.processed_buffer.extend(make_ratio_frame(3, t0=1.0).to_dict("records"))
    fake.packet_buffer.extend(make_packet_rows(3))
    app.bfm_collector = fake
    app._do_bfm_collection("alice", "walking", "", 1)
    assert len(fake.packet_buffer) == 0
    assert "no packets were captured" in app.collect_msg.cget("text")


# ─── Snapshot merge into bfm_* dirs ────────────────────────────────────────────
def test_snapshot_merges_new_session_files(gui3_app, tmp_path):
    app = gui3_app
    for d in ("live_bfm_pcap", "live_bfm_raw_csv", "live_bfm_processed_csv"):
        (tmp_path / d).mkdir()
    # a leftover from an earlier session must be ignored
    (tmp_path / "live_bfm_raw_csv" / "old.csv").write_text("timestamp,x\n1,2\n")
    shutil.copy(SAMPLE_PCAP, tmp_path / "live_bfm_pcap" / "old.pcap")
    before = {"pcap": {"old.pcap"}, "raw": {"old.csv"}, "proc": set()}

    make_ratio_frame(4).to_csv(tmp_path / "live_bfm_raw_csv" / "live_bfm0.csv", index=False)
    make_ratio_frame(6).to_csv(tmp_path / "live_bfm_raw_csv" / "live_bfm1.csv", index=False)
    shutil.copy(SAMPLE_PCAP, tmp_path / "live_bfm_pcap" / "live_bfm0.pcap")
    shutil.copy(SAMPLE_PCAP, tmp_path / "live_bfm_pcap" / "live_bfm1.pcap")

    app._snapshot_session_to_bfm_dirs("alice", "walking", "d", before)

    csvs = list((tmp_path / "bfm_raw_csv").glob("bfm_data_alice_walking_d_*.csv"))
    assert len(csvs) == 1 and len(pd.read_csv(csvs[0])) == 10

    pcaps = list((tmp_path / "bfm_pcap").glob("bfm_data_alice_walking_d_*.pcap"))
    assert len(pcaps) == 1
    from scapy.all import rdpcap
    assert len(rdpcap(str(pcaps[0]))) == 2 * len(rdpcap(str(SAMPLE_PCAP)))


def test_snapshot_with_nothing_new_writes_nothing(gui3_app, tmp_path):
    before = {"pcap": set(), "raw": set(), "proc": set()}
    gui3_app._snapshot_session_to_bfm_dirs("a", "b", "", before)
    assert not (tmp_path / "bfm_raw_csv").exists()
    assert not (tmp_path / "bfm_pcap").exists()


# ─── Status panel / live plots ─────────────────────────────────────────────────
def test_status_panel_reflects_collector_state(gui3_app):
    app = gui3_app
    fake = FakeCollector(running=True, connected=True, status="Connected ✓")
    fake.total_downloaded, fake.total_processed, fake.total_packets = 3, 2, 22
    fake.packet_buffer.extend(make_packet_rows(4))
    app.bfm_collector = fake

    app._update_status_panel()
    assert app.status_var.get() == "🟢 Connected ✓"
    assert app.dl_var.get() == "Downloaded: 3"
    assert app.proc_var.get() == "Processed: 2"
    assert app.buf_var.get() == "Buffer: 4"
    assert app.pkts_var.get() == "Packets: 22"

    fake.connected = False
    app._update_status_panel()
    assert app.status_var.get().startswith("🟡")
    fake.running = False
    app._update_status_panel()
    assert app.status_var.get().startswith("🔴")

    app.bfm_collector = None
    app._update_status_panel()
    assert app.status_var.get() == "⚪ Not set up"
    assert app.pkts_var.get() == "Packets: 0"


def test_live_plots_update_feature_panel(gui3_app):
    app = gui3_app
    fake = FakeCollector()
    app.bfm_collector = fake

    fake.processed_buffer.extend(make_ratio_frame(2).to_dict("records"))
    app._update_live_plots()
    assert app.feat_mean_var.get() == "Mean Velocity: 0.0000"   # <3 rows: untouched

    fake.processed_buffer.extend(make_ratio_frame(30, seed=5).to_dict("records"))
    app._update_live_plots()
    assert app.feat_mean_var.get() != "Mean Velocity: 0.0000"
    assert app.feat_max_var.get().startswith("Max Velocity: ")
    assert app.feat_std_var.get().startswith("Std Dev: ")
    assert app.feat_energy_var.get().startswith("Energy: ")
    assert app.bfm_absvel_ax.get_xlabel() == "Packet Number"


def test_streaming_tick_survives_bad_collector(gui3_app):
    class Broken:
        running = True
        connected = True
        connection_status = "x"
        total_downloaded = 0
        total_processed = 0
        total_packets = 0
        @property
        def packet_buffer(self): raise RuntimeError("boom")
    gui3_app.bfm_collector = Broken()
    gui3_app._streaming_tick()   # must not raise


# ─── Setup / close state machine ───────────────────────────────────────────────
def test_setup_click_runs_preflight_in_background(gui3_app):
    app = gui3_app
    started = threading.Event()
    app._do_preflight_then_setup = started.set
    app._toggle_bfm_setup()
    assert started.wait(3)
    for b in _buttons(app):
        assert b.cget("text") == "Checking…"
        assert str(b.cget("state")) == "disabled"
    assert app.collect_msg.cget("text") == "Running preflight check…"


def test_preflight_failure_resets_ui(gui3_app, dialogs):
    app = gui3_app
    app._preflight_check = lambda **kw: (False, ["✓ ping", "✗ SSH failed"])
    app._do_preflight_then_setup()
    app.update()
    assert app.bfm_is_setup is False and app.bfm_collector is None
    assert dialogs == [("showerror", "Preflight failed", "✓ ping\n✗ SSH failed")]
    for b in _buttons(app):
        assert b.cget("text") == "Setup BFM"
        assert str(b.cget("state")) == "normal"
    assert str(app.collect_msg.cget("foreground")) == "red"


def test_preflight_success_starts_streaming(gui3_app, dialogs, monkeypatch):
    app = gui3_app
    fake = FakeCollector(running=False, connected=False)
    monkeypatch.setattr(main_gui3, "LiveDataCollector", lambda **kw: fake)
    app._preflight_check = lambda **kw: (True, ["✓ all good"])
    app._verify_tcpdump_after_setup = lambda report: None   # would SSH out

    app._do_preflight_then_setup()
    app.update()

    assert app.bfm_is_setup is True and app.bfm_collector is fake
    assert fake.calls == ["start_collection"]
    for b in _buttons(app):
        assert b.cget("text") == "Close BFM"
        assert str(b.cget("state")) == "normal"
    assert app.collect_msg.cget("text").startswith("Preflight OK. Streaming started")
    assert dialogs == []


def test_preflight_success_but_stream_fails(gui3_app, dialogs, monkeypatch):
    app = gui3_app
    class Dead(FakeCollector):
        def start_collection(self): raise RuntimeError("no thread")
    monkeypatch.setattr(main_gui3, "LiveDataCollector", lambda **kw: Dead())
    app._preflight_check = lambda **kw: (True, ["ok"])
    app._do_preflight_then_setup()
    app.update()
    assert app.bfm_is_setup is True
    assert "streaming failed to start" in app.collect_msg.cget("text")
    assert dialogs == [("showinfo", "Preflight passed", "ok")]


def test_collector_init_failure_is_reported(gui3_app, dialogs, monkeypatch):
    app = gui3_app
    def boom(**kw): raise RuntimeError("tshark missing")
    monkeypatch.setattr(main_gui3, "LiveDataCollector", boom)
    app._preflight_check = lambda **kw: (True, ["ok"])
    app._do_preflight_then_setup()
    app.update()
    assert app.bfm_is_setup is False
    assert dialogs[0][1] == "Preflight failed"
    assert "Collector init failed: tshark missing" in dialogs[0][2]


def test_close_stops_collector_and_resets(gui3_app, dialogs):
    app = gui3_app
    fake = FakeCollector()
    app.bfm_collector, app.bfm_is_setup = fake, True
    app.bfm_setup_btn.config(text="Close BFM")

    app._toggle_bfm_setup()

    assert fake.calls == ["stop_collection"]
    assert app.bfm_collector is None and app.bfm_is_setup is False
    for b in _buttons(app):
        assert b.cget("text") == "Setup BFM"
    assert app.collect_msg.cget("text") == "BFM connection closed."
    assert dialogs == []


def test_close_failure_still_resets(gui3_app, dialogs):
    app = gui3_app
    class Stuck(FakeCollector):
        def stop_collection(self): raise RuntimeError("ssh hung")
    app.bfm_collector, app.bfm_is_setup = Stuck(), True
    app._toggle_bfm_setup()
    assert app.bfm_collector is None and app.bfm_is_setup is False
    assert dialogs == [("showerror", "BFM Close Error", "ssh hung")]
    assert _buttons(app)[0].cget("text") == "Setup BFM"


def test_preflight_failed_drops_partial_collector(gui3_app, dialogs):
    app = gui3_app
    fake = FakeCollector()
    app.bfm_collector = fake
    app._on_preflight_failed("report")
    assert fake.calls == ["stop_collection"]
    assert app.bfm_collector is None


def test_on_close_tears_down_setup_collector(gui3_app):
    app = gui3_app
    fake = FakeCollector()
    app.bfm_collector, app.bfm_is_setup = fake, True
    app._on_close()
    assert fake.calls == ["stop_collection"]
    with pytest.raises(Exception):
        app.winfo_exists()


def test_on_close_tears_down_partial_collector(gui3_app):
    app = gui3_app
    fake = FakeCollector()
    app.bfm_collector = fake      # bfm_is_setup stays False
    app._on_close()
    assert fake.calls == ["stop_collection"]
    assert app._stop_csi.is_set()
    assert app._stop_pcap_transfer.is_set()


# ─── Detector model loading / start ────────────────────────────────────────────
def _bundle_file(tmp_path, **overrides):
    bundle = dict(model=FakeModel(0.7), window_s=3.0, hop_s=0.25, frequency=10.0,
                  classes=["standing", "walking"], environments=["nofoil"])
    bundle.update(overrides)
    path = tmp_path / "fake_bundle.joblib"
    joblib.dump(bundle, path)
    return str(path)


def test_load_bfm_rt_model_reads_bundle(gui3_app, tmp_path, monkeypatch, dialogs):
    app = gui3_app
    path = _bundle_file(tmp_path)
    monkeypatch.setattr(main_gui3, "BFM_MODEL_PATHS", ["missing.joblib", path])

    assert app._load_bfm_rt_model() is True
    assert isinstance(app.bfm_rt_model, FakeModel)
    assert app.bfm_rt_window_s == 3.0 and app.bfm_rt_hop_s == 0.25
    assert app.bfm_rt_fs == 10.0          # 'frequency' alias honoured
    assert app.bfm_rt_classes == ["standing", "walking"]
    assert app.bfm_rt_cols is None
    text = app.pred_model_label.cget("text")
    assert "fake_bundle.joblib" in text and "3s window @ 10Hz" in text
    assert "nofoil" in text
    assert dialogs == []


def test_load_bfm_rt_model_prefers_fs_key(gui3_app, tmp_path, monkeypatch):
    path = _bundle_file(tmp_path, fs=20.0, frequency=10.0)
    monkeypatch.setattr(main_gui3, "BFM_MODEL_PATHS", [path])
    assert gui3_app._load_bfm_rt_model()
    assert gui3_app.bfm_rt_fs == 20.0


def test_load_bfm_rt_model_missing(gui3_app, monkeypatch, dialogs):
    monkeypatch.setattr(main_gui3, "BFM_MODEL_PATHS", ["nope.joblib"])
    assert gui3_app._load_bfm_rt_model() is False
    assert dialogs[0][:2] == ("showerror", "Model Missing")
    assert "nope.joblib" in dialogs[0][2]


def test_load_bfm_rt_model_corrupt(gui3_app, tmp_path, monkeypatch, dialogs):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a joblib file")
    monkeypatch.setattr(main_gui3, "BFM_MODEL_PATHS", [str(bad)])
    assert gui3_app._load_bfm_rt_model() is False
    assert dialogs[0][:2] == ("showerror", "Load Error")


def test_start_pred_requires_streaming(gui3_app, tmp_path, monkeypatch, dialogs):
    app = gui3_app
    monkeypatch.setattr(main_gui3, "BFM_MODEL_PATHS", [_bundle_file(tmp_path)])
    app._start_pred()
    assert dialogs[-1][:2] == ("showerror", "BFM Not Streaming")
    assert getattr(app, "pred_thread", None) is None


def test_start_pred_without_model_stops_early(gui3_app, monkeypatch, dialogs):
    monkeypatch.setattr(main_gui3, "BFM_MODEL_PATHS", [])
    gui3_app.bfm_collector = FakeCollector()
    gui3_app._start_pred()
    assert [d[1] for d in dialogs] == ["Model Missing"]


def test_start_pred_runs_once_and_stops(gui3_app, tmp_path, monkeypatch):
    app = gui3_app
    monkeypatch.setattr(main_gui3, "BFM_MODEL_PATHS", [_bundle_file(tmp_path, hop_s=0.02)])
    fake = FakeCollector()
    fake.packet_buffer.extend(make_packet_rows(80))
    app.bfm_collector = fake

    app._start_pred()
    first = app.pred_thread
    app._start_pred()                      # second click must not spawn another
    assert app.pred_thread is first

    assert pump_until(app, lambda: app.pred_label.cget("text").startswith("walking"))
    assert "P(walking)" in app.pred_prob_label.cget("text")

    app._stop_pred.set()
    assert pump_until(app, lambda: app.pred_label.cget("text") == "Waiting...")
    first.join(timeout=5)
    assert not first.is_alive()


def test_set_pred_text_marshals_to_main_thread(gui3_app):
    app = gui3_app
    app._set_pred_text("walking (0.91)", "raw 0.9")
    assert app.pred_label.cget("text") == "Waiting..."   # not yet applied
    app.update()
    assert app.pred_label.cget("text") == "walking (0.91)"
    assert app.pred_prob_label.cget("text") == "raw 0.9"
