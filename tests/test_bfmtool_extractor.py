"""bfmtool.extractor: per-packet SCIDX labels, so chunks mixing 80/40 MHz reports parse."""
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from bfmtool.extractor import BFMExtractor, parsed_bfm_list_to_df, records_to_bfm_df
from bfmtool.preprocessor import BFMPreprocessor
from bfmtool.utils import bfm_row_widths_mhz
from conftest import FIXTURES

MIXED_PCAP = FIXTURES / "bfm_80_then_40.pcap"   # frames: 80, 80, 40, 40 MHz
TSHARK = shutil.which("tshark") or "/usr/bin/tshark"


def _packet(scidx, phi=1, psi=2):
    return [{"SCIDX": k, "phi11": phi, "psi21": psi} for k in scidx]


def test_mixed_widths_line_up_by_scidx():
    df = parsed_bfm_list_to_df([_packet([-2, 2, 5], phi=7), _packet([-1, 2], phi=9)])
    assert list(df.columns) == [
        "SCIDX_-2_phi11", "SCIDX_-2_psi21", "SCIDX_-1_phi11", "SCIDX_-1_psi21",
        "SCIDX_2_phi11", "SCIDX_2_psi21", "SCIDX_5_phi11", "SCIDX_5_psi21"]
    assert df["SCIDX_2_phi11"].tolist() == [7, 9]
    assert df["SCIDX_-1_phi11"].isna().tolist() == [True, False]
    assert df["SCIDX_5_phi11"].isna().tolist() == [False, True]


def test_wider_packet_after_narrow_one_does_not_raise():
    # Positional rows used to raise "N columns passed, passed data had M columns".
    df = parsed_bfm_list_to_df([_packet([1]), _packet([1, 2, 3])])
    assert df.notna().sum(axis=1).tolist() == [2, 6]


def test_records_to_df_sorts_columns_by_tone():
    df = records_to_bfm_df([{"SCIDX_3_psi21": 1, "SCIDX_-4_phi11": 2, "SCIDX_3_phi11": 3}])
    assert list(df.columns) == ["SCIDX_-4_phi11", "SCIDX_3_phi11", "SCIDX_3_psi21"]


# ─── Real pcap through tshark ──────────────────────────────────────────────────
needs_tshark = pytest.mark.skipif(not os.path.exists(TSHARK), reason="tshark not found")


def _extract(pcap, tmp_path):
    out = tmp_path / (pcap.stem + ".csv")
    BFMExtractor(tshark_path=TSHARK, csv_dir=str(tmp_path), show_progress=False,
                 verbose=False).pcap_to_csv(str(pcap), str(out))
    return pd.read_csv(out)


def _tones(row):
    return sorted(int(c.split("_")[1]) for c, v in row.items() if c.endswith("_phi11") and pd.notna(v))


def _reordered(tmp_path, order):
    from scapy.all import rdpcap, wrpcap
    frames = rdpcap(str(MIXED_PCAP))
    path = tmp_path / "reordered.pcap"
    wrpcap(str(path), [frames[i] for i in order])
    return path


@pytest.mark.tshark
@needs_tshark
def test_80_then_40_chunk_keeps_true_tone_labels(tmp_path):
    df = _extract(MIXED_PCAP, tmp_path)
    assert len(df) == 4
    t80, t40 = _tones(df.iloc[0]), _tones(df.iloc[2])
    assert len(t80) == 234 and (t80[0], t80[-1]) == (-122, 122)
    assert len(t40) == 108 and (t40[0], t40[-1]) == (-58, 58)   # not -122..-12


@pytest.mark.tshark
@needs_tshark
def test_40_then_80_chunk_is_not_dropped(tmp_path):
    df = _extract(_reordered(tmp_path, [2, 3, 0, 1]), tmp_path)
    assert len(df) == 4
    assert [len(_tones(df.iloc[i])) for i in range(4)] == [108, 108, 234, 234]


@pytest.mark.tshark
@needs_tshark
def test_angles_do_not_depend_on_chunk_order(tmp_path):
    a = _extract(MIXED_PCAP, tmp_path)
    b = _extract(_reordered(tmp_path, [2, 3, 0, 1]), tmp_path)
    cols = [c for c in a.columns if c.startswith("SCIDX_")]
    pd.testing.assert_frame_equal(
        a[cols].reset_index(drop=True),
        b[cols].iloc[[2, 3, 0, 1]].reset_index(drop=True))


@pytest.mark.tshark
@needs_tshark
def test_mixed_chunk_preprocesses_to_both_widths(tmp_path):
    raw = tmp_path / "raw.csv"
    _extract(MIXED_PCAP, tmp_path).to_csv(raw, index=False)
    proc = tmp_path / "proc.csv"
    BFMPreprocessor(dir=str(tmp_path), verbose=False).process_file(str(raw), str(proc))
    df = pd.read_csv(proc)
    assert bfm_row_widths_mhz(df).tolist() == [80, 80, 40, 40]
    real = [c for c in df.columns if c.endswith("_Ratio_Real")]
    assert np.isfinite(df.loc[2, [c for c in real if pd.notna(df.loc[2, c])]].to_numpy(float)).all()
