"""bfmtool.utils channel-width helpers: 52/108/234 subcarriers -> 20/40/80 MHz."""
import numpy as np
import pandas as pd

from bfmtool.utils import append_mag_phase, bfm_row_widths_mhz, narrow_bfm_warning
from conftest import make_packet_frame, make_ratio_frame

SC_80 = list(range(-122, 112))   # 234 indices
SC_40 = list(range(-58, 50))     # 108 indices
SC_20 = list(range(-28, 24))     # 52 indices


def test_widths_from_subcarrier_count():
    for scidx, mhz in ((SC_80, 80), (SC_40, 40), (SC_20, 20)):
        assert (bfm_row_widths_mhz(make_ratio_frame(3, scidx=scidx)) == mhz).all()


def test_unrecognised_count_is_nan():
    # Synthetic test frames (8 subcarriers) must never look narrow.
    assert bfm_row_widths_mhz(make_ratio_frame(3)).isna().all()
    assert narrow_bfm_warning(make_ratio_frame(3)) is None


def test_mag_columns_used_when_no_real_columns():
    df = make_packet_frame(4, scidx=SC_40)
    assert (bfm_row_widths_mhz(df) == 40).all()


def test_no_bfm_columns_is_all_nan():
    df = pd.DataFrame({"timestamp": [1.0, 2.0]})
    assert bfm_row_widths_mhz(df).isna().all()
    assert narrow_bfm_warning(df) is None


def test_merged_session_counts_each_row_separately():
    # concat pads a 40 MHz chunk's missing subcarriers with NaN.
    merged = pd.concat(
        [make_ratio_frame(5, scidx=SC_80), make_ratio_frame(3, scidx=SC_40)],
        ignore_index=True,
    )
    assert bfm_row_widths_mhz(merged).tolist() == [80] * 5 + [40] * 3


def test_full_width_session_has_no_warning():
    assert narrow_bfm_warning(make_ratio_frame(6, scidx=SC_80)) is None


def test_warning_names_counts_and_widths():
    merged = pd.concat(
        [make_ratio_frame(5, scidx=SC_80), make_ratio_frame(3, scidx=SC_40),
         make_ratio_frame(2, scidx=SC_20)],
        ignore_index=True,
    )
    msg = narrow_bfm_warning(merged)
    assert msg.startswith("5 of 10 BFM reports (50.0%) are ")
    assert "20 MHz (52 subcarriers), 40 MHz (108 subcarriers)" in msg
    assert "instead of 80 MHz (234 subcarriers)" in msg
    assert "re-record" in msg


def test_isolated_blip_says_drop_not_rerecord():
    merged = pd.concat(
        [make_ratio_frame(150, scidx=SC_80), make_ratio_frame(1, scidx=SC_40)],
        ignore_index=True,
    )
    msg = narrow_bfm_warning(merged)
    assert msg.startswith("1 of 151 BFM reports (0.7%) are 40 MHz")
    assert "drop those rows; no need to re-record" in msg
    assert "re-record it" not in msg


def test_slide_above_blip_share_says_rerecord():
    merged = pd.concat(
        [make_ratio_frame(98, scidx=SC_80), make_ratio_frame(2, scidx=SC_40)],
        ignore_index=True,
    )
    assert "re-record it" in narrow_bfm_warning(merged)


def _wrapping_frame(n_rows, scidx, seed):
    # Steep phase ramp across subcarriers so unwrap has real 2π jumps to fix.
    df = make_ratio_frame(n_rows, scidx=scidx, seed=seed)
    for i in scidx:
        df[f"SCIDX_{i}_Ratio_Real"] = np.cos(0.9 * i)
        df[f"SCIDX_{i}_Ratio_Imag"] = np.sin(0.9 * i)
    return df


def test_mag_phase_survives_columns_only_narrow_rows_carry():
    # 20 MHz reports carry ±1/±11, which 80 MHz reports never do.
    sc80 = [k for k in SC_80 if k not in (-11, -1, 1, 11)]
    sc20 = [k for k in SC_20 if k != 0]
    wide, narrow = _wrapping_frame(6, sc80, 0), _wrapping_frame(1, sc20, 1)
    merged = pd.concat([wide.iloc[:3], narrow, wide.iloc[3:]], ignore_index=True)

    out = append_mag_phase(merged)
    ref = append_mag_phase(wide)
    cols80 = [f"SCIDX_{k}_Ratio_Phase" for k in sc80]
    cols20 = [f"SCIDX_{k}_Ratio_Phase" for k in sc20]
    got80 = out.drop(index=3)[cols80].to_numpy()
    assert not np.isnan(got80).any()
    np.testing.assert_allclose(got80, ref[cols80].to_numpy())
    assert not out.loc[3, cols20].isna().any()
    np.testing.assert_allclose(out.loc[3, cols20].to_numpy(float),
                               append_mag_phase(narrow)[cols20].to_numpy()[0])
