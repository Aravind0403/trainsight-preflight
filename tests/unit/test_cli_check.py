from typer.testing import CliRunner
from trainsight.cli.main import app

runner = CliRunner()

def test_cli_check_dpo_clean(clean_dpo_path):
    result = runner.invoke(app, [
        "check",
        str(clean_dpo_path),
        "--task", "dpo",
        "--tokenizer", "Qwen/Qwen2.5-0.5B-Instruct",
    ])
    assert result.exit_code == 0
    assert "DPO Preflight Summary" in result.output
    assert "Total Pairs" in result.output
    assert "Length Bias Ratio" in result.output
    assert "Preflight Quality Gate Passed" in result.output

def test_cli_check_dpo_defects_strict_fails(dpo_defects_path):
    result = runner.invoke(app, [
        "check",
        str(dpo_defects_path),
        "--task", "dpo",
        "--tokenizer", "Qwen/Qwen2.5-0.5B-Instruct",
        "--strict",
    ])
    assert result.exit_code == 1
    assert "DPO Preflight Summary" in result.output
    assert "Preflight Quality Gate Failed" in result.output
    assert "Identical Pairs" in result.output

def test_cli_check_dpo_defects_nostrict_passes(dpo_defects_path):
    result = runner.invoke(app, [
        "check",
        str(dpo_defects_path),
        "--task", "dpo",
        "--tokenizer", "Qwen/Qwen2.5-0.5B-Instruct",
        "--no-strict",
    ])
    assert result.exit_code == 0
    assert "Preflight Quality Gate Failed" in result.output

def test_cli_check_sft_clean(clean_sft_path):
    result = runner.invoke(app, [
        "check",
        str(clean_sft_path),
        "--task", "sft",
        "--tokenizer", "Qwen/Qwen2.5-0.5B-Instruct",
    ])
    assert result.exit_code == 0
    assert "SFT Preflight Summary" in result.output
    assert "Preflight Quality Gate Passed" in result.output

def test_cli_check_unsupported_task(clean_sft_path):
    result = runner.invoke(app, [
        "check",
        str(clean_sft_path),
        "--task", "unsupported_task",
    ])
    assert result.exit_code == 1
    assert "Unsupported task" in result.output
