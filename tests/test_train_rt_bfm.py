"""train_rt_bfm.py: session labelling from filenames and window labels around the activity switch."""
import argparse

import numpy as np
import pytest

import rtbfm
import train_rt_bfm
from conftest import make_ratio_frame


@pytest.mark.parametrize("name,expected", [
    ("bfm_ri_data_Sidney_Standing_val-ses-1_20261007_102129.csv", ("Sidney", "standing")),
    ("bfm_ri_data_Jin Wei_Walking_val-ses-3_20261007_111040.csv", ("Jin Wei", "walking")),
    ("bfm_ri_data_Jin Wei_Standing + Walking_Test_20261006_095122.csv", ("Jin Wei", "mixed")),
    ("bfm_ri_data_Jareena_S + W_BT_20261006_112521.csv", ("Jareena", "mixed")),
    ("bfm_ri_data_Kenny_Sitting_x_20261007_120000.csv", ("Kenny", "sitting")),
    ("bfm_ri_data_Student 2_Lying_Spot A R1_20261014_102612.csv", ("Student 2", "lying")),
    ("bfm_ri_data_Nobody_None_Empty Room Test_20261013_093051.csv", None),
])
def test_parse_name(name, expected):
    assert train_rt_bfm.parse_name(f"bfm_real_imag_csv/{name}") == expected


def test_mixed_session_labels_and_guard(tmp_path):
    path = tmp_path / "bfm_ri_data_X_Standing + Walking_t_20261007_000000.csv"
    make_ratio_frame(n_rows=400, scidx=list(range(-8, 8))).to_csv(path, index=False)
    X, y = train_rt_bfm.session_windows(str(path), "mixed", window_s=2.0, hop_s=1.0, split_s=20.0)

    assert X.shape[1] == len(rtbfm.feature_names())
    assert np.isfinite(X).all()
    # windows overlapping 18-22 s are dropped; earlier ones standing, later walking
    assert list(y) == [0] * 17 + [1] * 17


@pytest.mark.parametrize("name,expected", [
    ("bfm_ri_data_P1_Standing_spotA R1_20261013_100000.csv", "A"),
    ("bfm_ri_data_P1_Sitting_spot B swap_20261013_100000.csv", "B"),
    ("bfm_ri_data_P1_Sitting_SpotB_20261013_100000.csv", "B"),
    ("bfm_ri_data_P1_Standing_Round 1_20261013_100000.csv", None),
    ("bfm_ri_data_P1_Standing_spotAusual_20261013_100000.csv", None),
])
def test_parse_spot(name, expected):
    assert train_rt_bfm.parse_spot(f"bfm_real_imag_csv/{name}") == expected


def test_pilot_window_features_layout():
    rng = np.random.default_rng(0)
    mag, phase = rng.normal(2.0, 0.1, (50, 16)), rng.uniform(-np.pi, np.pi, (50, 16))
    phase[:, 0] = np.r_[[3.1] * 25, [-3.1] * 25]  # straddles +-pi: circular mean, not arithmetic
    x = train_rt_bfm.pilot_window_features(mag, phase, 10.0)
    n_rt = len(rtbfm.feature_names())
    assert x.shape == (n_rt + 32,)
    np.testing.assert_allclose(x[n_rt:n_rt + 16], mag.mean(0), rtol=1e-5)
    assert abs(abs(x[n_rt + 16]) - np.pi) < 0.05


def _write_pilot_session(tmp_path, i, kind, spot):
    """Synthetic session whose static level depends on the spot only, never the posture."""
    df = make_ratio_frame(n_rows=300, scidx=list(range(-8, 8)), seed=i)
    if spot == "B":
        for k in range(-8, 0):
            df[f"SCIDX_{k}_Ratio_Real"] *= 1.5
    name = f"bfm_ri_data_P1_{kind}_spot{spot} R{i}_20261013_1000{i:02d}.csv"
    df.to_csv(tmp_path / name, index=False)


@pytest.mark.parametrize("posture", ["standing", "lying"])
def test_pilot_flags_a_location_shortcut(tmp_path, capsys, posture):
    other = posture.capitalize()
    layout = [(other, "A"), ("Sitting", "B")] * 3 + [(other, "B"), ("Sitting", "A")] * 2
    # the other pilot posture's sessions must be ignored
    _write_pilot_session(tmp_path, 99, "Lying" if posture == "standing" else "Standing", "A")
    for i, (kind, spot) in enumerate(layout):
        _write_pilot_session(tmp_path, i, kind, spot)
    a = argparse.Namespace(window=5.0, hop=2.5, split=60.0, glob=[str(tmp_path / "*.csv")],
                           clf="lr", pca=4, pass_mark=0.8, pilot_posture=posture)
    train_rt_bfm.run_pilot(a)
    out = capsys.readouterr().out

    assert "6 usual + 4 swapped sessions" in out
    assert f"usual layout: {posture} @A, sitting @B" in out
    # static levels encode the spot, so swapped sessions flip
    assert "B: mostly flipped;" in out
    assert "spot only (same posture)" in out and "n=" in out
