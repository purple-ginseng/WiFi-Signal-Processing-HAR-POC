"""bfmtool.session_merge: session CSVs, byte-level pcap merge, child process."""
import struct
import subprocess
import sys

import pandas as pd
import pytest
from scapy.all import rdpcap, wrpcap

from bfmtool import session_merge
import main_gui3
from conftest import REPO_ROOT, SAMPLE_PCAP, make_ratio_frame

HEADER = session_merge._PCAP_HEADER_LEN
RECORD = session_merge._RECORD_HEADER_LEN


def frames(path):
    return [(bytes(p), float(p.time)) for p in rdpcap(str(path))]


def first_record_len(data):
    order = session_merge._PCAP_BYTE_ORDER[data[:4]]
    return RECORD + struct.unpack_from(order + "I", data, HEADER + 8)[0]


def test_merge_matches_scapy_frame_for_frame(tmp_path):
    dst, scapy_dst = tmp_path / "bytes.pcap", tmp_path / "scapy.pcap"
    assert session_merge.merge_pcaps([SAMPLE_PCAP, SAMPLE_PCAP], dst) == 2

    pkts = list(rdpcap(str(SAMPLE_PCAP)))
    wrpcap(str(scapy_dst), pkts + pkts)
    assert frames(dst) == frames(scapy_dst)


def test_truncated_chunk_keeps_its_whole_records(tmp_path):
    data = SAMPLE_PCAP.read_bytes()
    keep = first_record_len(data)
    cut = tmp_path / "cut.pcap"
    cut.write_bytes(data[:HEADER + keep + RECORD + 1])   # one byte into record two

    dst = tmp_path / "out.pcap"
    assert session_merge.merge_pcaps([cut, SAMPLE_PCAP], dst) == 2
    # The merged file stays parseable past the cut: no misaligned records.
    assert frames(dst) == frames(SAMPLE_PCAP)[:1] + frames(SAMPLE_PCAP)


@pytest.mark.parametrize("content", [b"", b"short", b"not a pcap" * 10])
def test_unreadable_chunk_is_skipped(tmp_path, capsys, content):
    bad = tmp_path / "bad.pcap"
    bad.write_bytes(content)
    dst = tmp_path / "out.pcap"
    assert session_merge.merge_pcaps([bad, SAMPLE_PCAP], dst) == 1
    assert frames(dst) == frames(SAMPLE_PCAP)
    assert "Skipping unreadable pcap" in capsys.readouterr().out


def test_missing_chunk_is_skipped(tmp_path):
    dst = tmp_path / "out.pcap"
    assert session_merge.merge_pcaps([tmp_path / "gone.pcap", SAMPLE_PCAP], dst) == 1


def test_chunk_with_other_header_is_skipped(tmp_path, capsys):
    data = bytearray(SAMPLE_PCAP.read_bytes())
    data[20:24] = bytes(reversed(data[20:24]))    # different link type
    other = tmp_path / "other.pcap"
    other.write_bytes(data)

    dst = tmp_path / "out.pcap"
    assert session_merge.merge_pcaps([SAMPLE_PCAP, other], dst) == 1
    assert frames(dst) == frames(SAMPLE_PCAP)
    assert "header differs" in capsys.readouterr().out


def test_no_packets_writes_no_file(tmp_path):
    empty = tmp_path / "empty.pcap"
    empty.write_bytes(SAMPLE_PCAP.read_bytes()[:HEADER])
    dst = tmp_path / "out.pcap"
    assert session_merge.merge_pcaps([empty], dst) == 0
    assert not dst.exists()


def test_merge_csvs_skips_bad_files_and_unions_columns(tmp_path, capsys):
    wide = tmp_path / "wide.csv"
    narrow = tmp_path / "narrow.csv"
    empty = tmp_path / "empty.csv"
    bad = tmp_path / "bad.csv"
    make_ratio_frame(4).to_csv(wide, index=False)
    make_ratio_frame(3, scidx=[-1, 1]).to_csv(narrow, index=False)
    empty.write_text("timestamp,x\n")
    bad.write_text("")

    dst = tmp_path / "out.csv"
    assert session_merge.merge_csvs([wide, empty, bad, narrow], dst) == 7
    merged = pd.read_csv(dst)
    assert len(merged) == 7 and "SCIDX_-122_Ratio_Real" in merged
    assert merged["SCIDX_-122_Ratio_Real"].isna().sum() == 3
    assert "Skipping bad CSV" in capsys.readouterr().out


def test_failed_pcap_merge_still_writes_csv(tmp_path, capsys):
    raw = tmp_path / "raw.csv"
    make_ratio_frame(4).to_csv(raw, index=False)
    job = {"pcaps": [str(SAMPLE_PCAP)], "raws": [str(raw)],
           "dst_pcap": str(tmp_path / "no_such_dir" / "out.pcap"),
           "dst_csv": str(tmp_path / "out.csv")}
    session_merge.run_job(job)
    assert "PCAP merge failed" in capsys.readouterr().out
    assert len(pd.read_csv(tmp_path / "out.csv")) == 4


def session_job(n_rows=6):
    df = make_ratio_frame(n_rows)
    df["subject"], df["activity"] = "alice", "walking"
    return {"df": df, "ri_path": "ri/s.csv", "mp_path": "mp/s.csv"}


def test_session_csvs_match_the_gui_conversion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job = session_job()
    session_merge.run_in_child(job)

    # Exact file text: training data must not change by moving where it's written.
    assert (tmp_path / "ri/s.csv").read_text() == job["df"].to_csv(index=False)
    expected = main_gui3.convert_real_imag_to_mag_phase(job["df"], [], [])
    assert (tmp_path / "mp/s.csv").read_text() == expected.to_csv(index=False)


def test_session_without_subcarriers_writes_no_mag_phase(tmp_path):
    df = pd.DataFrame({"timestamp": [1.0, 2.0]})
    session_merge.write_session_csvs(df, str(tmp_path / "ri/s.csv"), str(tmp_path / "mp/s.csv"))
    assert (tmp_path / "ri/s.csv").is_file()
    assert not (tmp_path / "mp/s.csv").exists()


def test_child_exits_nonzero_when_session_write_fails(tmp_path, monkeypatch):
    """Otherwise the GUI would report Saved for a file that was never written."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ri").write_text("a file where the directory should be")
    calls = []
    real_run_job = session_merge.run_job

    def in_process(job):
        calls.append(job)
        real_run_job(job)

    monkeypatch.setattr(session_merge, "run_job", in_process)
    with pytest.raises(OSError):
        session_merge.run_in_child(session_job())
    assert len(calls) == 1


@pytest.mark.parametrize("error", [
    subprocess.CalledProcessError(1, "merge"),
    subprocess.TimeoutExpired("merge", 1),
    OSError("no interpreter"),
])
def test_child_failure_falls_back_in_process(tmp_path, monkeypatch, error):
    monkeypatch.chdir(tmp_path)

    def fail(*a, **k):
        raise error

    monkeypatch.setattr(session_merge.subprocess, "run", fail)
    session_merge.run_in_child(session_job())
    assert (tmp_path / "ri/s.csv").is_file() and (tmp_path / "mp/s.csv").is_file()


def test_child_runs_from_another_cwd_and_keeps_pythonpath(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    raw = tmp_path / "raw.csv"
    make_ratio_frame(4).to_csv(raw, index=False)
    session_merge.run_in_child({"pcaps": [str(SAMPLE_PCAP)], "raws": ["raw.csv"],
                                "dst_pcap": "out.pcap", "dst_csv": "out.csv"})
    assert frames(tmp_path / "out.pcap") == frames(SAMPLE_PCAP)
    assert len(pd.read_csv(tmp_path / "out.csv")) == 4


def test_child_does_not_load_gui_or_scapy():
    code = ("import sys; import bfmtool.session_merge; "
            "print(sorted({'tkinter', 'scapy', 'matplotlib'} & set(sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, cwd=REPO_ROOT, timeout=60)
    assert out.stdout.strip() == "[]"
