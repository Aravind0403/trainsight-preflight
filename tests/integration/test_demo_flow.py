from typer.testing import CliRunner
from trainsight import __version__
from trainsight.cli.main import app

runner = CliRunner()

def test_demo_command_runs_successfully():
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0
    assert "Preflight Quality Gate Passed" in result.output
    assert "CRITICAL CONTRACT VIOLATION" in result.output
    assert "COLLATOR LEAKAGE" in result.output

def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"TrainSight version: {__version__}" in result.output
