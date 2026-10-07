"""check_capture.py: index recovery through the real decoder, integrity flags, motion stats."""
import numpy as np
import pandas as pd
import pytest

import check_capture
from bfmtool.preprocessor import decompress_bfm_from_angles


def ratio_from_indices(k_phi, k_psi):
    """Complex h1/h2 exactly as bfmtool's preprocessor stores it."""
    v = decompress_bfm_from_angles(np.stack([k_phi, k_psi], axis=2).astype(float), Nc=1, Nr=2)
    return v[:, :, 0] / v[:, :, 1]


def write_session(path, k_phi, k_psi, t):
    z = ratio_from_indices(k_phi, k_psi)
    data = {"timestamp": t, "transmitter_address": ["aa:bb:cc:dd:ee:02"] * len(t)}
    for s in range(z.shape[1]):
        data[f"SCIDX_{s}_Ratio_Real"] = z[:, s].real
        data[f"SCIDX_{s}_Ratio_Imag"] = z[:, s].imag
    pd.DataFrame(data).to_csv(path, index=False)


def stand_then_walk(n=1200, n_sc=16, seed=0):
    """Standing: isolated quantisation flips. Walking: coherent random-walk drift across subcarriers."""
    rng = np.random.default_rng(seed)
    base = np.tile(rng.integers(20, 40, n_sc), (n, 1))
    k_phi = base + (rng.random((n, n_sc)) < 0.2)
    drift = np.cumsum(rng.choice([-1, 0, 1], size=n), axis=0)[:, None]
    k_phi[n // 2:] = base[n // 2:] + np.mod(drift[n // 2:] - drift[n // 2], 20)
    k_psi = np.full((n, n_sc), 8)
    t = 1_700_000_000.0 + np.arange(n) / 10.0
    return np.mod(k_phi, 64), k_psi, t


@pytest.mark.parametrize("phi_max,psi_max", [(63, 15), (511, 127)])
def test_angle_indices_round_trip_through_decoder(phi_max, psi_max):
    rng = np.random.default_rng(1)
    k_phi = rng.integers(0, phi_max + 1, (50, 8))
    k_psi = rng.integers(0, psi_max + 1, (50, 8))
    kp, ks, residual = check_capture.angle_indices(ratio_from_indices(k_phi, k_psi))
    assert residual < 1e-6
    np.testing.assert_array_equal(kp, k_phi)
    np.testing.assert_array_equal(ks, k_psi)


def test_integrity_flags_bad_rows():
    t = np.array([0.0, 0.1, 0.1, 0.3, 0.2, 0.4])
    z = np.ones((6, 3), complex)
    z[5, 1] = np.nan
    keep, info = check_capture.integrity(t, z)
    assert info["incomplete"] == 1
    assert info["dup_ts"] == 1
    assert info["nonmono"] == 1
    assert list(t[keep]) == [0.0, 0.1, 0.2, 0.3]


def test_walking_segment_scores_higher(tmp_path, capsys):
    k_phi, k_psi, t = stand_then_walk()
    path = tmp_path / "session.csv"
    write_session(path, k_phi, k_psi, t)

    check_capture.main([str(path), "--split", "60"])
    out = capsys.readouterr().out

    assert "issues: none" in out
    assert "100.0% SU" in out
    rows = {line.split()[0]: line.split() for line in out.splitlines()
            if line.strip().startswith(("standing", "walking"))}
    assert float(rows["walking"][2]) > 2 * float(rows["standing"][2])   # phi_jump
    assert float(rows["walking"][3]) > float(rows["standing"][3])       # coherence
    auc = float(out.split("AUC (walking > standing) = ")[1].split()[0])
    assert auc > 0.9


def test_no_split_reports_whole_session(tmp_path, capsys):
    k_phi, k_psi, t = stand_then_walk(n=200)
    path = tmp_path / "session.csv"
    write_session(path, k_phi, k_psi, t)
    check_capture.main([str(path), "--split", "none"])
    out = capsys.readouterr().out
    assert any(line.split()[:1] == ["all"] for line in out.splitlines())
    assert "separation" not in out
