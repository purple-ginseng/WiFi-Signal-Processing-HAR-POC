"""Unit tests for bfm_pipeline helpers (no data files needed)."""
import numpy as np
import pytest

import bfm_pipeline as bp


def _s(subject, activity, desc, date, time):
    return bp.parse_name(f"bfm_ri_data_{subject}_{activity}_{desc}_{date}_{time}.csv")


def test_parse_name_official_validation_and_empty():
    s = _s("S08", "Waving", "Spot A Round 1", "20261111", "105000")
    assert (s["subject"], s["activity"], s["desc"], s["date"], s["time"]) == ("S08", "Waving", "Spot A Round 1", "20261111", "105000")
    assert s["official"] and not s["empty"]
    assert not _s("S08", "Standing", "Spot A Validate 2", "20261111", "104334")["official"]
    assert not _s("JW", "Waving", "Spot A Validate", "20261110", "101347")["official"]
    e = _s("Nobody", "Empty", "Empty Room 3 (antenna shift a bit)", "20261111", "102640")
    assert e["empty"] and e["desc"] == "Empty Room 3 (antenna shift a bit)"
    assert bp.parse_name("bfm_ri_data_garbage.csv") is None


def test_reference_room_is_last_empty_before_first_official_session():
    sessions = sorted([
        _s("Nobody", "Empty", "Empty Room 1", "20261111", "101436"),
        _s("Nobody", "Empty", "Empty Room 3 (again)", "20261111", "103336"),
        _s("S08", "Standing", "Spot A Validate", "20261111", "104000"),   # validation doesn't count
        _s("S08", "Walking", "Round 1", "20261111", "105000"),
        _s("Nobody", "Empty", "Empty Room 5", "20261111", "112818"),
        _s("Nobody", "Empty", "Empty Room", "20261112", "090000"),        # no official session that day
    ], key=lambda r: (r["date"], r["time"]))
    refs = bp.choose_reference_rooms(sessions)
    assert list(refs) == ["20261111"]
    assert refs["20261111"]["time"] == "103336"


def test_resample_nearest_grid_and_gap_flags():
    t = np.array([0.0, 0.1, 0.1, 0.2, 0.9, 1.0])          # duplicate stamp, then a 0.7 s hole
    z = np.arange(len(t))[:, None].astype(complex)
    zr, gap = bp.resample_nearest(t, z)
    assert len(zr) == 11                                      # 0.0 .. 1.0 at 10 Hz
    assert zr[0, 0] == 0 and zr[1, 0] == 1                    # duplicate dropped, first kept
    assert gap[5] and gap[6]                                  # 0.3 s from any report
    assert not gap[:4].any() and not gap[8:].any()            # 0.4 / 0.7 sit on the tolerance edge


def test_static_level_is_log_of_time_mean_magnitude():
    mag = np.array([[1.0, 4.0], [3.0, 4.0]])
    np.testing.assert_allclose(bp.static_level(mag), np.log([2.0, 4.0]))


def test_lvl25_drops_the_largest_level_quarter():
    rng = np.random.default_rng(0)
    gains = np.linspace(1, 50, 200)                            # distinct static levels
    mag = gains * (1 + 0.01 * rng.standard_normal((120, 200)))
    keep = bp.build_masks(mag)["lvl25"]
    assert keep.sum() == 150
    assert keep[:150].all() and not keep[150:].any()           # dropped set = biggest gains


def test_noise25_drops_the_noisiest_quarter_regardless_of_level():
    rng = np.random.default_rng(1)
    noise = np.r_[np.full(150, 0.01), np.full(50, 0.2)]
    mag = 5.0 * (1 + noise * rng.standard_normal((120, 200)))
    keep = bp.build_masks(mag)["noise25"]
    assert keep[:150].all() and not keep[150:].any()


def test_placement_report_spread_in_db():
    mag = np.tile(np.logspace(0, 1, 101), (50, 1))            # levels evenly spaced over 20 dB
    assert bp.placement_report(mag)["spread_db"] == pytest.approx(16.0)   # p10..p90 = 80% of it
