"""main_gui3.FrameBuffer: the chunk-frame buffer behind packet_buffer/processed_buffer."""
import threading
from collections import deque

import numpy as np
import pandas as pd

import main_gui3
from conftest import make_packet_frame, make_ratio_frame


def _ts(n, start=0):
    return pd.DataFrame({"timestamp": np.arange(start, start + n, dtype=float)})


def test_empty_buffer():
    b = main_gui3.FrameBuffer(10)
    assert len(b) == 0
    assert b.tail().empty and b.tail(5).empty
    b.append(pd.DataFrame())   # empty chunks are ignored
    assert len(b) == 0


def test_cap_drops_oldest_rows_like_a_deque():
    b = main_gui3.FrameBuffer(10)
    ref = deque(maxlen=10)
    start = 0
    for n in (4, 3, 5, 1, 7, 12):
        chunk = _ts(n, start)
        b.append(chunk)
        ref.extend(chunk["timestamp"])
        start += n
        assert len(b) == len(ref)
        assert b.tail()["timestamp"].tolist() == list(ref)


def test_tail_spans_chunks_with_fresh_index():
    b = main_gui3.FrameBuffer(100)
    for k in range(4):
        b.append(_ts(5, 5 * k))
    out = b.tail(7)
    assert out["timestamp"].tolist() == list(np.arange(13, 20, dtype=float))
    assert out.index.tolist() == list(range(7))
    assert len(b.tail(1000)) == 20


def test_tail_matches_old_dict_rows_with_mixed_columns():
    # Chunks with different subcarrier sets: union columns, NaN-filled, as before.
    a = make_ratio_frame(3, scidx=[-2, 2])
    c = make_ratio_frame(2, scidx=[-2, 4], t0=2e9)
    b = main_gui3.FrameBuffer(100)
    b.append(a)
    b.append(c)
    old = pd.DataFrame(a.to_dict("records") + c.to_dict("records"))
    pd.testing.assert_frame_equal(b.tail(), old)


def test_session_csv_is_identical_to_old_dict_path(tmp_path):
    # _do_bfm_collection saves processed_buffer.tail(); must match the old CSV.
    chunks = [make_ratio_frame(11, t0=1.7e9 + k, seed=k) for k in range(6)]
    b = main_gui3.FrameBuffer(10000)
    for ch in chunks:
        b.append(ch)
    old = pd.DataFrame([r.to_dict() for ch in chunks for _, r in ch.iterrows()])
    b.tail().to_csv(tmp_path / "new.csv", index=False)
    old.to_csv(tmp_path / "old.csv", index=False)
    assert (tmp_path / "new.csv").read_bytes() == (tmp_path / "old.csv").read_bytes()


def test_clear_and_generation():
    b = main_gui3.FrameBuffer(10)
    g0 = b.generation
    b.append(_ts(3))
    assert b.generation == g0 + 1
    b.append(pd.DataFrame())   # nothing appended, nothing changed
    assert b.generation == g0 + 1
    b.clear()
    assert len(b) == 0 and b.tail().empty
    assert b.generation == g0 + 2


def test_identical_columns_share_one_index():
    b = main_gui3.FrameBuffer(100)
    b.append(make_packet_frame(3))
    b.append(make_packet_frame(3, seed=1))
    b.append(make_ratio_frame(3))
    first, second, third = b._frames
    assert second.columns is first.columns
    assert third.columns is not first.columns   # different layout keeps its own


def test_stores_consolidated_copy_not_callers_frame():
    # pandas 3 read_csv: one block per column. The buffer must not keep that layout.
    df = pd.DataFrame({f"c{i}": np.arange(3.0) for i in range(50)})
    for i in range(50):
        df[f"d{i}"] = np.arange(3.0)   # piecewise inserts -> many blocks
    assert df._mgr.nblocks > 1
    b = main_gui3.FrameBuffer(100)
    b.append(df)
    stored = b._frames[0]
    assert stored is not df and stored._mgr.nblocks == 1
    df.iloc[0, 0] = 99.0                # caller's later edits don't leak in
    assert stored.iloc[0, 0] == 0.0


def test_concurrent_append_and_read():
    b = main_gui3.FrameBuffer(500)
    stop = threading.Event()

    def writer():
        k = 0
        while not stop.is_set():
            b.append(_ts(7, k))
            k += 7

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        for _ in range(300):
            ts = b.tail(50)["timestamp"].to_numpy() if len(b) else np.array([])
            assert (np.diff(ts) == 1).all()   # always one contiguous, ordered run
            assert len(b) <= 500
    finally:
        stop.set()
        t.join(timeout=5)
