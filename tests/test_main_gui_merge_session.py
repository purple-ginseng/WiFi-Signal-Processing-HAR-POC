"""main_gui._merge_bfm_session_csv, called unbound on a stub self (no Tk)."""
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import main_gui
from conftest import make_ratio_frame

T0 = 1_700_000_000.0


def _stub(processed_dir):
    return SimpleNamespace(bfm_preprocessor=SimpleNamespace(dir=Path(processed_dir)))


def _write_chunks(processed_dir, frames):
    """Write frames as chunkN.csv; return the raw paths the merge expects."""
    processed_dir.mkdir(exist_ok=True)
    raw_paths = []
    for k, df in enumerate(frames):
        name = f"chunk{k}.csv"
        df.to_csv(processed_dir / name, index=False)
        # raw paths live elsewhere; only the basename is used for lookup
        raw_paths.append(Path("raw_dir") / name)
    return raw_paths


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _merge(workdir, frames, **kw):
    processed = workdir / "bfm_processed_csv"
    raw_paths = _write_chunks(processed, frames)
    return main_gui.MainApp._merge_bfm_session_csv(
        _stub(processed), "alice", "walking", "", raw_paths, **kw
    ), processed


def test_merges_chunks_and_writes_both_csvs(workdir):
    frames = [make_ratio_frame(4, t0=T0), make_ratio_frame(3, t0=T0 + 1.0)]
    (names, rows), processed = _merge(workdir, frames)

    ri_name, mp_name = names
    assert rows == 7
    assert ri_name.startswith("bfm_ri_data_alice_walking_")
    assert mp_name.startswith("bfm_mp_data_alice_walking_")

    ri = pd.read_csv(workdir / "bfm_real_imag_csv" / ri_name)
    mp = pd.read_csv(workdir / "bfm_mag_phase_csv" / mp_name)
    assert len(ri) == len(mp) == 7
    assert (ri["activity"] == "walking").all()
    assert (mp["activity"] == "walking").all()
    assert (ri["subject"] == "alice").all()
    assert (mp["subject"] == "alice").all()
    for df in (ri, mp):   # subject sits right before label
        cols = list(df.columns)
        assert cols.index("activity") == cols.index("subject") + 1
    assert "SCIDX_-122_Ratio_Real" in ri.columns
    assert "SCIDX_-122_Ratio_Mag" in mp.columns
    assert "SCIDX_-122_Ratio_Real" not in mp.columns


def test_chunk_fragments_removed_after_merge(workdir):
    (names, _), processed = _merge(workdir, [make_ratio_frame(2), make_ratio_frame(2)])
    assert names is not None
    assert list(processed.iterdir()) == []


def test_trims_to_requested_window(workdir):
    # 30 rows at 10 Hz span 3 s; ask for the middle 1 s
    frames = [make_ratio_frame(30, t0=T0)]
    (names, rows), _ = _merge(workdir, frames, start_ts=T0 + 1.0, duration=1.0)
    assert rows == 11   # inclusive on both edges: 1.0 .. 2.0
    ri = pd.read_csv(workdir / "bfm_real_imag_csv" / names[0])
    assert ri["timestamp"].min() >= T0 + 1.0
    assert ri["timestamp"].max() <= T0 + 2.0


def test_window_matching_nothing_saves_untrimmed(workdir, capsys):
    # Router clock months off host clock: never throw the session away
    frames = [make_ratio_frame(10, t0=T0)]
    (names, rows), _ = _merge(workdir, frames, start_ts=T0 + 1e6, duration=5.0)
    assert rows == 10
    assert "check the router's clock" in capsys.readouterr().out


def test_no_trim_without_window_args(workdir):
    (names, rows), _ = _merge(workdir, [make_ratio_frame(12, t0=T0)])
    assert rows == 12


def test_missing_processed_files_returns_none(workdir):
    processed = workdir / "bfm_processed_csv"
    processed.mkdir()
    out = main_gui.MainApp._merge_bfm_session_csv(
        _stub(processed), "a", "b", "", [Path("raw_dir/nothing.csv")]
    )
    assert out == (None, 0)
    assert not (workdir / "bfm_real_imag_csv").exists()


def test_empty_chunks_returns_none(workdir):
    empty = make_ratio_frame(0)
    (out, _) = _merge(workdir, [empty, empty])
    assert out == (None, 0)


def test_unreadable_chunk_is_skipped(workdir, capsys):
    processed = workdir / "bfm_processed_csv"
    raw_paths = _write_chunks(processed, [make_ratio_frame(3)])
    bad = processed / "bad.csv"
    bad.write_bytes(b"\x00\xff not a csv \x00")
    raw_paths.append(Path("raw_dir") / "bad.csv")

    (names, rows) = main_gui.MainApp._merge_bfm_session_csv(
        _stub(processed), "a", "b", "", raw_paths
    )
    assert rows == 3
    assert names is not None


def test_description_lands_in_filename(workdir):
    processed = workdir / "bfm_processed_csv"
    raw_paths = _write_chunks(processed, [make_ratio_frame(2)])
    (ri_name, mp_name), _ = main_gui.MainApp._merge_bfm_session_csv(
        _stub(processed), "alice", "walking", "no foil", raw_paths
    )
    assert "_alice_walking_no foil_" in ri_name
    assert "_alice_walking_no foil_" in mp_name


def test_subject_column_uses_gui_value_verbatim(workdir):
    processed = workdir / "bfm_processed_csv"
    raw_paths = _write_chunks(processed, [make_ratio_frame(3)])
    (ri_name, mp_name), _ = main_gui.MainApp._merge_bfm_session_csv(
        _stub(processed), "Bob Smith", "standing", "", raw_paths
    )
    for d, name in (("bfm_real_imag_csv", ri_name), ("bfm_mag_phase_csv", mp_name)):
        df = pd.read_csv(workdir / d / name)
        assert df["subject"].tolist() == ["Bob Smith"] * 3
        assert df["activity"].tolist() == ["standing"] * 3
        assert "label" not in df.columns
