"""train_rt_bfm.py: session labelling from filenames and window labels around the activity switch."""
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
    ("bfm_ri_data_Kenny_Sitting_x_20261007_120000.csv", None),
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
