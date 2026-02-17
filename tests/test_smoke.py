"""Smoke tests: imports work, CLI --help works."""

from click.testing import CliRunner

from mermaid_ascii.__main__ import main


def test_import():
    import mermaid_ascii
    assert mermaid_ascii is not None


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Mermaid flowchart" in result.output
