"""main_gui3.LiveDataCollector offline: __init__ never touches the network."""
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main_gui3
from conftest import SAMPLE_PCAP, make_packet_frame, make_ratio_frame, run_until

pytestmark = pytest.mark.skipif(
    not os.path.exists(main_gui3.TSHARK_PATH),
    reason=f"tshark not found at {main_gui3.TSHARK_PATH}",
)


@pytest.fixture
def collector(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = main_gui3.LiveDataCollector("192.0.2.1", "root", "x", [], [])
    yield c
    c.running = False


def test_init_creates_live_dirs(collector, tmp_path):
    for d in ("live_bfm_pcap", "live_bfm_raw_csv", "live_bfm_processed_csv"):
        assert (tmp_path / d).is_dir()
    assert collector.connected is False
    assert collector.running is False
    assert collector.get_connection_status() == "Not connected"


def test_tcpdump_cmd_matches_healthcheck_pattern(collector):
    # _download_loop greps for this exact prefix; a drift makes the
    # health check re-issue tcpdump every second.
    assert "tcpdump -i mon0 -p -U -B 4096 -G 1" in collector._tcpdump_cmd
    assert "'wlan[24] == 21'" in collector._tcpdump_cmd


# ─── Buffer accessors ──────────────────────────────────────────────────────────
def test_latest_processed_timestamp_empty(collector):
    assert collector.latest_processed_timestamp() is None


def test_latest_processed_timestamp_is_max_and_skips_bad(collector):
    collector.processed_buffer.append(
        pd.DataFrame({"timestamp": [10.0, "12.5", None, "nan?", 11.0]})
    )
    assert collector.latest_processed_timestamp() == 12.5


def test_latest_processed_timestamp_only_looks_at_tail(collector):
    collector.processed_buffer.append(pd.DataFrame({"timestamp": [999.0] * 3}))
    collector.processed_buffer.append(pd.DataFrame({"timestamp": [1.0] * 64}))
    assert collector.latest_processed_timestamp() == 1.0


def test_get_latest_packets_requires_n(collector):
    collector.packet_buffer.append(make_packet_frame(3))
    assert collector.get_latest_packets(5) is None
    assert collector.get_latest_processed(1) is None


def test_get_latest_packets_returns_tail(collector):
    frame = make_packet_frame(10)
    collector.packet_buffer.append(frame.iloc[:7])
    collector.packet_buffer.append(frame.iloc[7:])   # tail spans two chunks
    df = collector.get_latest_packets(4)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert df["timestamp"].tolist() == frame["timestamp"].iloc[-4:].tolist()
    assert df.index.tolist() == [0, 1, 2, 3]


def test_get_latest_processed_returns_tail(collector):
    frame = make_ratio_frame(6)
    collector.processed_buffer.append(frame)
    df = collector.get_latest_processed(2)
    assert len(df) == 2
    assert df["timestamp"].iloc[-1] == frame["timestamp"].iloc[-1]
    assert collector.get_buffer_size() == 0   # packet_buffer untouched


# ─── Clock / capture tail ──────────────────────────────────────────────────────
def test_clock_offset_without_collector_is_zero(collector):
    assert collector.get_clock_offset() == 0.0


def test_clock_offset_delegates_and_swallows_errors(collector):
    class Good:
        def get_clock_offset(self): return 42.5
    class Bad:
        def get_clock_offset(self): raise RuntimeError("ssh down")
    collector.bfm_collector = Good()
    assert collector.get_clock_offset() == 42.5
    collector.bfm_collector = Bad()
    assert collector.get_clock_offset() == 0.0


def test_wait_for_capture_through_already_covered(collector):
    collector.processed_buffer.append(pd.DataFrame({"timestamp": [100.0]}))
    t = time.time()
    assert collector.wait_for_capture_through(99.0, timeout=5.0) is True
    assert time.time() - t < 1.0


def test_wait_for_capture_through_times_out(collector):
    collector.processed_buffer.append(pd.DataFrame({"timestamp": [100.0]}))
    assert collector.wait_for_capture_through(101.0, timeout=0.5) is False


def test_wait_for_capture_through_unblocks_when_tail_lands(collector):
    def feed():
        time.sleep(0.3)
        collector.processed_buffer.append(pd.DataFrame({"timestamp": [200.0]}))
    threading.Thread(target=feed, daemon=True).start()
    assert collector.wait_for_capture_through(200.0, timeout=5.0) is True


def test_launch_tcpdump_without_connection_is_false(collector):
    assert collector._launch_tcpdump_explicit() is False


def test_stop_collection_without_connection(collector):
    collector.running = True
    collector.stop_collection()
    assert collector.running is False
    assert collector.get_connection_status() == "Stopped"


def test_stop_collection_tears_down_router_session(collector):
    calls = []
    class FakeSSH:
        def kill_tcpdump(self): calls.append("kill_tcpdump")
        def kill_iperf3(self): calls.append("kill_iperf3")
        def close(self): calls.append("close")
    collector.bfm_collector = FakeSSH()
    collector.stop_collection()
    assert calls == ["kill_tcpdump", "kill_iperf3", "close"]
    assert collector.bfm_collector is None


# ─── Processing thread on a real pcap ──────────────────────────────────────────
def _run_processing(collector, timeout=30.0):
    collector.running = True
    t = threading.Thread(target=collector._processing_loop, daemon=True)
    t.start()
    run_until(lambda: not collector.download_queue and collector.total_packets > 0,
              timeout=timeout)
    # give the loop one more pass so counters settle before stopping
    time.sleep(0.3)
    collector.running = False
    t.join(timeout=5)


@pytest.mark.tshark
def test_processing_loop_fills_buffers_from_real_pcap(collector, tmp_path):
    local = tmp_path / "live_bfm_pcap" / "live_bfm0.pcap"
    shutil.copy(SAMPLE_PCAP, local)
    collector.download_queue.append(str(local))

    _run_processing(collector)

    assert collector.total_processed == 1
    assert collector.total_packets == 11
    assert len(collector.packet_buffer) == 11
    assert len(collector.processed_buffer) == 11

    row = collector.packet_buffer.tail(1).iloc[0].to_dict()
    assert "pc_timestamp" in row and abs(row["pc_timestamp"] - time.time()) < 60
    mags = [k for k in row if k.endswith("_Ratio_Mag")]
    phases = [k for k in row if k.endswith("_Ratio_Phase")]
    assert len(mags) == len(phases) > 0
    assert np.isfinite([row[k] for k in mags]).all()

    proc = collector.processed_buffer.tail(1).iloc[0].to_dict()
    assert any(k.endswith("_Ratio_Real") for k in proc)
    assert float(proc["timestamp"]) > 1e9

    assert (tmp_path / "live_bfm_raw_csv" / "live_bfm0.csv").is_file()
    assert (tmp_path / "live_bfm_processed_csv" / "live_bfm0.csv").is_file()
    assert collector.last_bfm_mhz == 80          # fixture is an 80 MHz capture


def _fake_pipeline(collector, monkeypatch, frame):
    """Skip tshark/decompression: every chunk 'processes' into frame."""
    monkeypatch.setattr(collector.bfm_extractor, "pcap_to_csv",
                        lambda src, dst: Path(dst).write_text("raw\n"))
    monkeypatch.setattr(collector.bfm_preprocessor, "process_file",
                        lambda src, dst: frame.to_csv(dst, index=False))


def _queue_chunk(collector, tmp_path, name="live_bfm0.pcap"):
    local = tmp_path / "live_bfm_pcap" / name
    local.write_bytes(b"pcap")
    collector.download_queue.append(str(local))


def test_processing_loop_flags_40mhz_chunk(collector, tmp_path, monkeypatch):
    _fake_pipeline(collector, monkeypatch, make_ratio_frame(3, scidx=range(-58, 50)))
    _queue_chunk(collector, tmp_path)

    _run_processing(collector)

    assert collector.last_bfm_mhz == 40
    log = (tmp_path / collector.DEBUG_LOG).read_text()
    assert log.count("[BANDWIDTH] BFM reports dropped to 40 MHz") == 1


def test_processing_loop_logs_drop_once_per_transition(collector, tmp_path, monkeypatch):
    _fake_pipeline(collector, monkeypatch, make_ratio_frame(3, scidx=range(-58, 50)))
    for k in range(3):
        _queue_chunk(collector, tmp_path, f"live_bfm{k}.pcap")

    _run_processing(collector)

    assert collector.total_packets == 9
    log = (tmp_path / collector.DEBUG_LOG).read_text()
    assert log.count("[BANDWIDTH]") == 1


def test_processing_loop_logs_one_report_blip_mid_chunk(collector, tmp_path, monkeypatch):
    chunk = pd.concat([make_ratio_frame(4, scidx=range(-122, 112)),
                       make_ratio_frame(1, scidx=range(-58, 50)),
                       make_ratio_frame(4, scidx=range(-122, 112))], ignore_index=True)
    _fake_pipeline(collector, monkeypatch, chunk)
    _queue_chunk(collector, tmp_path)

    _run_processing(collector)

    assert collector.last_bfm_mhz == 80            # newest report is full width again
    log = (tmp_path / collector.DEBUG_LOG).read_text()
    assert "[BANDWIDTH] BFM reports dropped to 40 MHz (1 of 9 in this chunk)" in log


def test_processing_loop_ignores_unrecognised_width(collector, tmp_path, monkeypatch):
    _fake_pipeline(collector, monkeypatch, make_ratio_frame(3))   # 8 subcarriers
    _queue_chunk(collector, tmp_path)

    _run_processing(collector)

    assert collector.total_packets == 3
    assert collector.last_bfm_mhz is None


# ─── Router rate-control reset ─────────────────────────────────────────────────
def test_reset_rate_control_without_connection_is_skipped(collector, tmp_path):
    assert collector.reset_rate_control() is None
    assert "rate reset skipped" in (tmp_path / collector.DEBUG_LOG).read_text()


def test_reset_rate_control_delegates_and_logs(collector, tmp_path):
    class Router:
        client = object()
        def reset_rate_control(self): return "tx bitrate:\t234.0 MBit/s VHT-MCS 5 80MHz"
    collector.bfm_collector = Router()
    assert collector.reset_rate_control().endswith("80MHz")
    assert "rate control reset: tx bitrate" in (tmp_path / collector.DEBUG_LOG).read_text()


def test_reset_rate_control_swallows_ssh_errors(collector, tmp_path):
    class Router:
        client = object()
        def reset_rate_control(self): raise OSError("channel closed")
    collector.bfm_collector = Router()
    assert collector.reset_rate_control() is None
    assert "rate reset failed: channel closed" in (tmp_path / collector.DEBUG_LOG).read_text()


@pytest.mark.tshark
def test_processing_loop_is_quiet_by_default(collector, tmp_path, capsys):
    # Per-chunk chatter floods the terminal over hours of streaming.
    assert main_gui3.BFM_VERBOSE is False
    local = tmp_path / "live_bfm_pcap" / "live_bfm0.pcap"
    shutil.copy(SAMPLE_PCAP, local)
    collector.download_queue.append(str(local))

    _run_processing(collector)

    assert collector.total_packets == 11
    out = capsys.readouterr().out
    for chatter in ("✓ +11 packets", "[Processing] live_bfm0.pcap",
                    "Creating DataFrame", "BFM data saved", "Original data shape",
                    "subcarriers."):
        assert chatter not in out


def test_vprint_follows_verbose_flag(collector, monkeypatch, capsys):
    collector._vprint("[Download] routine")
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(main_gui3, "BFM_VERBOSE", True)
    collector._vprint("[Download] routine")
    assert "[Download] routine" in capsys.readouterr().out


@pytest.mark.tshark
def test_processing_loop_skips_missing_and_empty_pcaps(collector, tmp_path):
    empty = tmp_path / "live_bfm_pcap" / "live_bfm1.pcap"
    empty.write_bytes(b"")
    collector.download_queue.append(str(tmp_path / "does_not_exist.pcap"))
    collector.download_queue.append(str(empty))

    collector.running = True
    t = threading.Thread(target=collector._processing_loop, daemon=True)
    t.start()
    run_until(lambda: not collector.download_queue, timeout=20)
    time.sleep(0.3)
    collector.running = False
    t.join(timeout=5)

    assert collector.total_packets == 0
    assert len(collector.packet_buffer) == 0
    assert collector.total_processed == 1   # only the file that existed


def test_processing_loop_drops_stale_backlog(collector, tmp_path):
    n = collector.MAX_PENDING_CHUNKS + 7
    for k in range(n):
        collector.download_queue.append(str(tmp_path / f"ghost{k}.pcap"))

    collector.running = True
    t = threading.Thread(target=collector._processing_loop, daemon=True)
    t.start()
    run_until(lambda: not collector.download_queue, timeout=10)
    collector.running = False
    t.join(timeout=5)

    log = (tmp_path / collector.DEBUG_LOG).read_text()
    assert "[QUEUE-DROP] skipped 7 stale chunk(s)" in log
