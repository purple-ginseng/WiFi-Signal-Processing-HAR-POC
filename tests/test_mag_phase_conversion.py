"""convert_real_imag_to_mag_phase — shared by both GUIs and the training pipeline."""
import numpy as np
import pandas as pd
import pytest

import main_gui
import main_gui3
from conftest import make_ratio_frame

MODULES = [main_gui, main_gui3]
IDS = ["main_gui", "main_gui3"]


@pytest.mark.parametrize("mod", MODULES, ids=IDS)
class TestConvertRealImagToMagPhase:
    def test_values_match_definition(self, mod):
        df = make_ratio_frame(n_rows=6)
        out = mod.convert_real_imag_to_mag_phase(df, [], [])

        for i in (-122, 3, 122):
            r = df[f"SCIDX_{i}_Ratio_Real"].to_numpy()
            im = df[f"SCIDX_{i}_Ratio_Imag"].to_numpy()
            np.testing.assert_allclose(out[f"SCIDX_{i}_Ratio_Mag"], np.hypot(r, im))
            np.testing.assert_allclose(
                out[f"SCIDX_{i}_Ratio_Phase"], np.unwrap(np.arctan2(im, r))
            )

    def test_real_imag_columns_dropped(self, mod):
        out = mod.convert_real_imag_to_mag_phase(make_ratio_frame(), [], [])
        assert not [c for c in out.columns if c.endswith(("_Ratio_Real", "_Ratio_Imag"))]

    def test_metadata_columns_preserved(self, mod):
        # filter_by_mode() downstream needs the MAC columns intact
        df = make_ratio_frame()
        df["activity"] = "walking"
        out = mod.convert_real_imag_to_mag_phase(df, [], [])
        for col in ("timestamp", "receiver_address", "transmitter_address", "activity"):
            pd.testing.assert_series_equal(out[col], df[col])

    def test_ratio_prefix_is_kept(self, mod):
        # Feature selectors match on endswith("Ratio_Mag")
        out = mod.convert_real_imag_to_mag_phase(make_ratio_frame(), [], [])
        mags = [c for c in out.columns if c.endswith("Ratio_Mag")]
        phases = [c for c in out.columns if c.endswith("Ratio_Phase")]
        assert len(mags) == len(phases) == 8

    def test_subcarrier_order_is_numeric(self, mod):
        out = mod.convert_real_imag_to_mag_phase(make_ratio_frame(), [], [])
        idx = [int(c.split("_")[1]) for c in out.columns if c.endswith("Ratio_Mag")]
        assert idx == sorted(idx)

    def test_no_ratio_columns_returns_empty(self, mod):
        df = pd.DataFrame({"timestamp": [1.0, 2.0], "SCIDX_-1_phi11": [1, 2]})
        out = mod.convert_real_imag_to_mag_phase(df, [], [])
        assert out.empty

    def test_row_count_unchanged(self, mod):
        out = mod.convert_real_imag_to_mag_phase(make_ratio_frame(n_rows=17), [], [])
        assert len(out) == 17


def test_both_modules_agree():
    df = make_ratio_frame(n_rows=9, seed=3)
    pd.testing.assert_frame_equal(
        main_gui.convert_real_imag_to_mag_phase(df, [], []),
        main_gui3.convert_real_imag_to_mag_phase(df, [], []),
    )
