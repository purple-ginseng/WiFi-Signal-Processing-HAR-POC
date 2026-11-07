"""Write one session's output files, normally in a child process.

Doing this in main_gui3's process starves Tk of the GIL: scapy's per-packet
parsing alone froze the GUI for 2.3 s. Keep imports light so the child never
loads Tk or scapy.

    python -m bfmtool.session_merge    # reads a pickled job from stdin
"""
import os
import pickle
import struct
import subprocess
import sys
from pathlib import Path

import pandas as pd

from bfmtool.utils import append_mag_phase, get_bfm_columns

# A hung child must not leave Start Collection disabled.
CHILD_TIMEOUT_S = 120

_PCAP_HEADER_LEN = 24
_RECORD_HEADER_LEN = 16
# Magic -> byte order, for microsecond and nanosecond libpcap files.
_PCAP_BYTE_ORDER = {
    b"\xa1\xb2\xc3\xd4": ">",
    b"\xd4\xc3\xb2\xa1": "<",
    b"\xa1\xb2\x3c\x4d": ">",
    b"\x4d\x3c\xb2\xa1": "<",
}


def _complete_records_end(data):
    """Offset just past the last whole record; a chunk can end mid-record."""
    order = _PCAP_BYTE_ORDER[data[:4]]
    pos = _PCAP_HEADER_LEN
    while pos + _RECORD_HEADER_LEN <= len(data):
        caplen = struct.unpack_from(order + "I", data, pos + 8)[0]
        end = pos + _RECORD_HEADER_LEN + caplen
        if end > len(data):
            break
        pos = end
    return pos


def merge_pcaps(src_files, dst_path):
    """Concatenate pcap records byte for byte. Returns the chunks merged.

    Writes nothing when no chunk holds a whole packet. Chunks whose header
    differs are skipped, since their records can't share one file header.
    """
    header = None
    parts = []
    for path in src_files:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as e:
            print(f"[Snapshot] Skipping unreadable pcap {path}: {e}")
            continue
        if data[:4] not in _PCAP_BYTE_ORDER or len(data) < _PCAP_HEADER_LEN:
            print(f"[Snapshot] Skipping unreadable pcap {path}: not a pcap file")
            continue
        if header is not None and data[:_PCAP_HEADER_LEN] != header:
            print(f"[Snapshot] Skipping pcap {path}: header differs from the session's")
            continue
        end = _complete_records_end(data)
        if end > _PCAP_HEADER_LEN:
            header = data[:_PCAP_HEADER_LEN]
            parts.append(data[_PCAP_HEADER_LEN:end])

    if not parts:
        return 0
    with open(dst_path, "wb") as out:
        out.write(header)
        out.writelines(parts)
    return len(parts)


def merge_csvs(src_files, dst_path):
    """Concatenate CSVs under one header. Returns the rows written."""
    frames = []
    for path in src_files:
        try:
            df = pd.read_csv(path)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"[Snapshot] Skipping bad CSV {path}: {e}")
    if not frames:
        return 0
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(dst_path, index=False)
    return len(merged)


def write_session_csvs(df, ri_path, mp_path):
    """Write real/imag ratios and their mag/phase conversion.

    Errors propagate: this is the session's primary data, so a failed write
    must exit non-zero and trigger the in-process retry.
    """
    os.makedirs(os.path.dirname(ri_path), exist_ok=True)
    os.makedirs(os.path.dirname(mp_path), exist_ok=True)
    df.to_csv(ri_path, index=False)

    cols = get_bfm_columns(df)
    if cols["real_col"] and cols["imag_col"]:
        mag_phase = append_mag_phase(df).drop(columns=cols["real_col"] + cols["imag_col"])
        mag_phase.to_csv(mp_path, index=False)


def run_job(job):
    """Run one job. Merge failures are logged so one output can't cost the other."""
    if "df" in job:
        write_session_csvs(job["df"], job["ri_path"], job["mp_path"])
    if job.get("pcaps"):
        try:
            chunks = merge_pcaps(job["pcaps"], job["dst_pcap"])
            if chunks:
                print(f"[Snapshot] Wrote {job['dst_pcap']}  ({chunks} chunks)")
        except Exception as e:
            print(f"[Snapshot] PCAP merge failed: {e}")
    if job.get("raws"):
        try:
            rows = merge_csvs(job["raws"], job["dst_csv"])
            if rows:
                print(f"[Snapshot] Wrote {job['dst_csv']}  ({rows} rows)")
        except Exception as e:
            print(f"[Snapshot] CSV merge failed: {e}")


def run_in_child(job):
    """Run a job in a child process, or in-process if the child fails.

    Keeping the session's data matters more than a laggy finish. Relative job
    paths resolve against the caller's cwd, which the child inherits.
    """
    root = str(Path(__file__).resolve().parent.parent)
    search_path = (root, os.environ.get("PYTHONPATH"))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, search_path))}
    try:
        subprocess.run(
            [sys.executable, "-m", "bfmtool.session_merge"],
            input=pickle.dumps(job),
            check=True,
            timeout=CHILD_TIMEOUT_S,
            env=env,
        )
    except (OSError, subprocess.SubprocessError, pickle.PicklingError) as e:
        print(f"[Session save] Child process failed ({e}); running in-process")
        run_job(job)


if __name__ == "__main__":
    run_job(pickle.load(sys.stdin.buffer))
