"""Filename helpers duplicated in both GUIs; every case runs against both."""
import re

import pytest

import main_gui
import main_gui3

MODULES = [main_gui, main_gui3]
IDS = ["main_gui", "main_gui3"]


@pytest.mark.parametrize("mod", MODULES, ids=IDS)
class TestSafeFilenamePart:
    def test_strips_whitespace(self, mod):
        assert mod._safe_filename_part("  alice  ") == "alice"

    @pytest.mark.parametrize("bad", list('<>:"/\\|?*'))
    def test_replaces_reserved_chars(self, mod, bad):
        assert mod._safe_filename_part(f"a{bad}b") == "a-b"

    def test_replaces_control_chars(self, mod):
        assert mod._safe_filename_part("a\x00b\x1fc\td") == "a-b-c-d"

    def test_keeps_safe_punctuation(self, mod):
        assert mod._safe_filename_part("Pi-V_4.test") == "Pi-V_4.test"


@pytest.mark.parametrize("mod", MODULES, ids=IDS)
class TestSessionName:
    def test_subject_activity_only(self, mod):
        assert mod._session_name("alice", "walking") == "alice_walking"

    def test_with_description(self, mod):
        assert mod._session_name("alice", "walking", "foil") == "alice_walking_foil"

    def test_blank_description_dropped(self, mod):
        assert mod._session_name("alice", "walking", "   ") == "alice_walking"

    def test_sanitizes_every_part(self, mod):
        assert mod._session_name("a/b", "c:d", "e?f") == "a-b_c-d_e-f"


@pytest.mark.parametrize("mod", MODULES, ids=IDS)
class TestBfmCsvFilename:
    @pytest.mark.parametrize("fmt", ["ri", "mp"])
    def test_format(self, mod, fmt):
        name = mod._bfm_csv_filename(fmt, "alice", "walking", "", "20260917_101010")
        assert name == f"bfm_{fmt}_data_alice_walking_20260917_101010.csv"

    def test_description_included(self, mod):
        name = mod._bfm_csv_filename("ri", "alice", "walking", "open", "20260917_101010")
        assert name == "bfm_ri_data_alice_walking_open_20260917_101010.csv"


def test_generate_bfm_filename_pattern():
    # main_gui-only: pcap name handed to BFMCollector.filename
    name = main_gui.MainApp.generate_bfm_filename(object(), "alice", "walking", "d")
    assert re.fullmatch(r"bfm_data_alice_walking_d_\d{8}_\d{6}\.pcap", name)


@pytest.mark.parametrize(
    "subject,activity,description",
    [
        ("alice", "walking", ""),
        ("Bob Smith", "standing", "no foil"),
        ("a/b", "c:d", "e*f"),
        ("  pad  ", "act", " \t"),
    ],
)
def test_both_modules_agree(subject, activity, description):
    ts = "20260917_101010"
    assert main_gui._session_name(subject, activity, description) == \
        main_gui3._session_name(subject, activity, description)
    for fmt in ("ri", "mp"):
        assert main_gui._bfm_csv_filename(fmt, subject, activity, description, ts) == \
            main_gui3._bfm_csv_filename(fmt, subject, activity, description, ts)
