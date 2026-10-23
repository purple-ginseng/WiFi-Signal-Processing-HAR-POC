"""rtbfm.window_features on quantisation-flat subcarriers (constant over the whole window)."""
import numpy as np

import rtbfm

NAMES = rtbfm.feature_names()
RATIO = [i for i, n in enumerate(NAMES) if n.startswith(("rough", "phrough", "pfrac", "corr"))]


def _window(n_moving=20, n_flat=30, T=50, seed=0):
    rng = np.random.default_rng(seed)
    mag = np.empty((T, n_moving + n_flat), np.float32)
    mag[:, :n_moving] = rng.normal(2.3, 0.05, (T, n_moving))
    mag[:, n_moving:] = np.float32(2.3331)  # float32 mean removal leaves a residue here
    phase = np.empty_like(mag)
    phase[:, :n_moving] = rng.normal(0.4, 0.05, (T, n_moving))
    phase[:, n_moving:] = np.float32(0.41)
    return mag, phase


def test_flat_subcarriers_do_not_change_ratio_features():
    mag, phase = _window()
    with_flat = rtbfm.window_features(mag, phase, 10.0)
    moving_only = rtbfm.window_features(mag[:, :20], phase[:, :20], 10.0)
    np.testing.assert_allclose(with_flat[RATIO], moving_only[RATIO], rtol=1e-5, atol=1e-7)
    # size features still see the flat subcarriers as zero motion
    assert with_flat[NAMES.index("sd_med")] < moving_only[NAMES.index("sd_med")]


def test_ratio_features_stay_in_range():
    feats = rtbfm.window_features(*_window(), 10.0)
    corr = feats[[NAMES.index(n) for n in ("corr_mean", "corr_absmean", "corr_std")]]
    assert np.all(np.abs(corr) <= 1.0)
    pfrac = feats[[i for i, n in enumerate(NAMES) if n.startswith("pfrac") and n.endswith("_mean")]]
    assert np.all((pfrac >= 0) & (pfrac <= 1)) and pfrac.sum() <= 1.0 + 1e-6


def test_fully_flat_window_is_finite_zero_shape():
    mag, phase = _window(n_moving=0, n_flat=10)
    feats = rtbfm.window_features(mag, phase, 10.0)
    assert feats.shape == (len(NAMES),) and np.isfinite(feats).all()
    assert np.all(feats[RATIO] == 0)
