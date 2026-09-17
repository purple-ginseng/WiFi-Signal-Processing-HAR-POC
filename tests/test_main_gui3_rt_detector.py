"""main_gui3 real-time detector: MainApp methods bound onto a Tk-free stub."""
import os
import threading
import time

import joblib
import numpy as np
import pytest

import main_gui3
import rtbfm
from conftest import DEFAULT_SCIDX, FakeCollector, FakeModel, make_packet_rows

T0 = 1_700_000_000.0


class _Thresh:
    def __init__(self, v): self.v = v
    def get(self): return self.v


class Detector:
    """Only the attributes _bfm_rt_window / _prediction_loop read."""
    _resolve_bfm_rt_cols = main_gui3.MainApp._resolve_bfm_rt_cols
    _bfm_rt_window = main_gui3.MainApp._bfm_rt_window
    _prediction_loop = main_gui3.MainApp._prediction_loop

    def __init__(self, collector, fs=10.0, window_s=2.0, hop_s=0.02,
                 model=None, thresh=0.5):
        self.bfm_collector = collector
        self.bfm_rt_fs = fs
        self.bfm_rt_window_s = window_s
        self.bfm_rt_hop_s = hop_s
        self.bfm_rt_cols = None
        self.bfm_rt_classes = ["standing", "walking"]
        self.bfm_rt_model = model
        self.pred_thresh = _Thresh(thresh)
        self._stop_pred = threading.Event()
        self.texts = []

    def _set_pred_text(self, label, prob=""):
        self.texts.append((label, prob))


def _detector_with_rows(rows, **kw):
    c = FakeCollector()
    c.packet_buffer.extend(rows)
    return Detector(c, **kw)


# ─── _resolve_bfm_rt_cols ──────────────────────────────────────────────────────
def test_resolve_cols_sorted_numerically_and_cached():
    d = Detector(FakeCollector())
    row = make_packet_rows(1, scidx=[5, -3, 120, -120])[0]
    mag, phase = d._resolve_bfm_rt_cols(row)
    assert mag == [f"SCIDX_{i}_Ratio_Mag" for i in (-120, -3, 5, 120)]
    assert phase == [f"SCIDX_{i}_Ratio_Phase" for i in (-120, -3, 5, 120)]
    # cached: a different row does not change the answer
    assert d._resolve_bfm_rt_cols({}) == (mag, phase)


def test_resolve_cols_none_without_phase():
    d = Detector(FakeCollector())
    row = {"SCIDX_1_Ratio_Mag": 1.0, "SCIDX_2_Ratio_Mag": 1.0, "SCIDX_1_Ratio_Phase": 0.0}
    assert d._resolve_bfm_rt_cols(row) is None
    assert d._resolve_bfm_rt_cols({"timestamp": 1.0}) is None


# ─── _bfm_rt_window status strings ─────────────────────────────────────────────
def test_window_not_streaming():
    assert Detector(None)._bfm_rt_window() == "BFM not streaming"
    assert Detector(FakeCollector(running=False))._bfm_rt_window() == "BFM not streaming"


def test_window_waiting_when_too_few_rows():
    d = _detector_with_rows(make_packet_rows(1))
    assert d._bfm_rt_window() == "Waiting for BFM data…"


def test_window_stale_when_no_recent_packets():
    stale = time.time() - main_gui3.BFM_RT_STALE_S - 1
    d = _detector_with_rows(make_packet_rows(40, pc_timestamp=stale))
    assert d._bfm_rt_window() == "No live data"


def test_window_no_subcarrier_columns():
    rows = [{"timestamp": T0 + k / 10, "pc_timestamp": time.time()} for k in range(5)]
    assert _detector_with_rows(rows)._bfm_rt_window() == "No subcarrier columns"


def test_window_waiting_when_all_timestamps_duplicate():
    rows = make_packet_rows(10)
    for r in rows:
        r["timestamp"] = T0
    assert _detector_with_rows(rows)._bfm_rt_window() == "Waiting for BFM data…"


# ─── _bfm_rt_window gridding ───────────────────────────────────────────────────
def test_window_shape_and_transform_on_clean_10hz_stream():
    rows = make_packet_rows(60, fs=10.0)
    d = _detector_with_rows(rows, fs=10.0, window_s=2.0)
    mag, phase, gap = d._bfm_rt_window()

    S = len(DEFAULT_SCIDX)
    assert mag.shape == (20, S) and phase.shape == (20, S) and gap.shape == (20,)
    assert mag.dtype == np.float32 and phase.dtype == np.float32
    assert not gap.any()

    # last grid tick == newest packet, transformed as prep.py does
    mag_cols, phase_cols = d.bfm_rt_cols
    newest = rows[-1]
    np.testing.assert_allclose(
        mag[-1], np.log([newest[c] for c in mag_cols]), rtol=1e-5)
    expected_phase = np.angle(np.exp(1j * np.array([newest[c] for c in phase_cols])))
    np.testing.assert_allclose(phase[-1], expected_phase, rtol=1e-5)
    assert (phase > -np.pi - 1e-6).all() and (phase <= np.pi + 1e-6).all()


def test_window_clips_tiny_magnitudes_before_log():
    rows = make_packet_rows(30)
    for r in rows:
        r["SCIDX_-122_Ratio_Mag"] = 0.0
    mag, _, _ = _detector_with_rows(rows)._bfm_rt_window()
    assert np.isfinite(mag).all()
    assert np.allclose(mag[:, 0], np.log(1e-6))


def test_window_flags_gaps_on_sparse_stream():
    # 2 Hz packets on a 10 Hz grid: most ticks have nothing within 2/fs
    rows = make_packet_rows(20, fs=2.0)
    _, _, gap = _detector_with_rows(rows)._bfm_rt_window()
    assert gap.mean() > main_gui3.BFM_RT_MAX_GAP


def test_window_flags_dropout_inside_window():
    rows = make_packet_rows(40, fs=10.0)
    # remove 1 s (10 packets) in the middle of the last 2 s
    rows = rows[:25] + rows[35:]
    _, _, gap = _detector_with_rows(rows)._bfm_rt_window()
    assert 4 <= gap.sum() <= 8


def test_window_sorts_out_of_order_and_drops_duplicates():
    rows = make_packet_rows(40, fs=10.0)
    ordered = _detector_with_rows(rows)._bfm_rt_window()

    shuffled = list(rows)
    rng = np.random.default_rng(1)
    rng.shuffle(shuffled)
    shuffled += [dict(r) for r in rows[-5:]]   # duplicate timestamps
    scrambled = _detector_with_rows(shuffled)._bfm_rt_window()

    np.testing.assert_array_equal(ordered[0], scrambled[0])
    np.testing.assert_array_equal(ordered[1], scrambled[1])
    np.testing.assert_array_equal(ordered[2], scrambled[2])


def test_window_ignores_rows_with_nan_timestamp():
    rows = make_packet_rows(30)
    rows[-1]["timestamp"] = float("nan")
    mag, _, gap = _detector_with_rows(rows)._bfm_rt_window()
    assert mag.shape[0] == 20 and not gap.any()


# ─── Feature/model invariant (CLAUDE.md landmine) ──────────────────────────────
def test_window_feeds_rtbfm_features():
    d = _detector_with_rows(make_packet_rows(60))
    mag, phase, _ = d._bfm_rt_window()
    feats = rtbfm.window_features(mag, phase, d.bfm_rt_fs)
    assert feats.shape == (len(rtbfm.feature_names()),)
    assert np.isfinite(feats).all()


@pytest.mark.parametrize("path", main_gui3.BFM_MODEL_PATHS)
def test_deployed_bundle_matches_rtbfm_features(path):
    """rtbfm.window_features must match what trained the deployed bundle."""
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    bundle = joblib.load(path)
    names = rtbfm.feature_names()
    assert bundle["feature_names"] == names

    fs = float(bundle.get("fs", bundle.get("frequency", 10.0)))
    T = int(round(float(bundle["window_s"]) * fs))
    mag = np.log(np.random.default_rng(0).uniform(0.5, 2.0, (T, 234))).astype(np.float32)
    phase = np.random.default_rng(1).uniform(-np.pi, np.pi, (T, 234)).astype(np.float32)
    feats = rtbfm.window_features(mag, phase, fs)

    model = bundle["model"]
    assert getattr(model, "n_features_in_", len(names)) == len(feats)
    proba = model.predict_proba(feats[None])
    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)


# ─── _prediction_loop ──────────────────────────────────────────────────────────
def _run_loop(d, min_texts=3, timeout=5.0):
    t = threading.Thread(target=d._prediction_loop, daemon=True)
    t.start()
    deadline = time.time() + timeout
    while len(d.texts) < min_texts and time.time() < deadline:
        time.sleep(0.01)
    d._stop_pred.set()
    t.join(timeout=5)
    assert not t.is_alive()


def test_prediction_loop_reports_walking_above_threshold():
    model = FakeModel(p_walk=0.9)
    d = _detector_with_rows(make_packet_rows(60), model=model, thresh=0.5)
    _run_loop(d)

    labels = [lbl for lbl, _ in d.texts[:-1]]
    assert labels and all(l.startswith("walking (") for l in labels)
    assert d.texts[-1] == ("Waiting...", "")
    # first prediction is unsmoothed p, later ones EMA toward p
    assert d.texts[0][0] == "walking (0.90)"
    assert "threshold 0.50" in d.texts[0][1]
    assert model.seen[0].shape == (1, len(rtbfm.feature_names()))


def test_prediction_loop_reports_standing_below_threshold():
    d = _detector_with_rows(make_packet_rows(60), model=FakeModel(0.2), thresh=0.5)
    _run_loop(d)
    assert all(lbl.startswith("standing (") for lbl, _ in d.texts[:-1])


def test_prediction_loop_threshold_is_live():
    d = _detector_with_rows(make_packet_rows(60), model=FakeModel(0.6), thresh=0.7)
    _run_loop(d)
    assert all(lbl.startswith("standing (") for lbl, _ in d.texts[:-1])


def test_prediction_loop_ema_converges():
    # Fresh loop starts at raw p; a steady stream must not drift away from it
    d = _detector_with_rows(make_packet_rows(60), model=FakeModel(0.8))
    _run_loop(d, min_texts=6)
    smoothed = [float(lbl.split("(")[1].rstrip(")")) for lbl, _ in d.texts[:-1]]
    assert all(abs(s - 0.8) < 1e-6 for s in smoothed)


def test_prediction_loop_shows_status_when_no_window():
    d = Detector(FakeCollector(running=False), model=FakeModel())
    _run_loop(d)
    assert d.texts[0] == ("BFM not streaming", "")
    assert not d.bfm_rt_model.seen


def test_prediction_loop_holds_on_signal_gap():
    d = _detector_with_rows(make_packet_rows(20, fs=2.0), model=FakeModel())
    _run_loop(d)
    label, prob = d.texts[0]
    assert label == "Signal gap…"
    assert "of the window had no packets" in prob
    assert not d.bfm_rt_model.seen


def test_prediction_loop_survives_model_error(capsys):
    class Boom:
        def predict_proba(self, X): raise RuntimeError("boom")
    d = _detector_with_rows(make_packet_rows(60), model=Boom())
    _run_loop(d, min_texts=1, timeout=1.0)
    assert "Prediction error: boom" in capsys.readouterr().out
    assert d.texts[-1] == ("Waiting...", "")
