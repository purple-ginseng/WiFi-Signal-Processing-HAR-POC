"""bfmtool.collector.BFMCollector.run_command with a fake SSH client (no router)."""
import io

import pytest

from bfmtool.collector import BFMCollector


class _FakeClient:
    def __init__(self):
        self.commands = []

    def exec_command(self, command):
        self.commands.append(command)
        return None, io.BytesIO(b"1234\n"), io.BytesIO(b"")


@pytest.mark.parametrize("verbose", [True, False])
def test_run_command_echo_follows_verbose(verbose, capsys):
    c = BFMCollector("192.0.2.1", "root", "x", verbose=verbose)
    c.client = _FakeClient()
    assert c.run_command("pgrep tcpdump") == ("1234", "")
    assert c.client.commands == ["pgrep tcpdump"]
    echoed = "Executing command: 'pgrep tcpdump'" in capsys.readouterr().out
    assert echoed is verbose


def test_verbose_defaults_on_for_existing_callers():
    assert BFMCollector("192.0.2.1", "root", "x").verbose is True


def test_not_connected_still_reports_when_quiet(capsys):
    c = BFMCollector("192.0.2.1", "root", "x", verbose=False)
    assert c.run_command("true") == (None, None)
    assert "Cannot run command" in capsys.readouterr().out
