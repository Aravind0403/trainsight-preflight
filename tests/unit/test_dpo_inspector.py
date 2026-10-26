import pytest
from trainsight.inspectors.dpo_inspector import DPOInspector

def test_clean_dpo_passes_preflight(clean_dpo_path):
    inspector = DPOInspector(tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct", max_total_len=512)
    report = inspector.inspect_file(clean_dpo_path)

    assert report.total_samples == 3
    assert report.valid_samples_count == 3
    assert report.empty_pair_count == 0
    assert report.identical_pairs_count == 0
    assert report.preference_reversal_count == 0
    assert report.prompt_mismatch_count == 0
    assert report.phase_quadrant == "Clean Execution"
    # Test backwards-compatible property aliases
    assert report.total_pairs == 3
    assert report.valid_pairs_count == 3
    assert report.invalid_pairs_count == 0

def test_dpo_defects_detected(dpo_defects_path):
    inspector = DPOInspector(tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct")
    report = inspector.inspect_file(dpo_defects_path)

    assert report.total_samples == 5
    assert report.empty_pair_count == 1
    assert report.identical_pairs_count == 1
    assert report.preference_reversal_count == 1
    assert report.prompt_mismatch_count == 1
    assert report.phase_quadrant == "Defects Detected"

def test_dpo_inspector_max_seq_len_threshold_kwarg(clean_dpo_path):
    # Tests backward-compatibility kwarg
    inspector = DPOInspector(
        tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_seq_len_threshold=512,
    )
    assert inspector.max_total_len == 512
    assert inspector.max_prompt_len == 256
    assert inspector.max_completion_len == 256
    report = inspector.inspect_file(clean_dpo_path)
    assert report.valid_samples_count == 3
