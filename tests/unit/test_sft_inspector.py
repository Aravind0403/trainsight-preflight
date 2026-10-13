import pytest
from trainsight.inspectors.sft_inspector import SFTInspector

def test_clean_sft_fixture_passes(clean_sft_path):
    inspector = SFTInspector(tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct", max_seq_len_threshold=256)
    report = inspector.inspect_file(clean_sft_path)
    assert report.total_samples == 25
    assert report.valid_samples_count == 25
    assert report.empty_completion_count == 0
    assert report.truncated_answers_count == 0
    assert report.invalid_rows_count == 0

def test_empty_answer_fails_preflight(empty_answer_path):
    inspector = SFTInspector(tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct", max_seq_len_threshold=256)
    report = inspector.inspect_file(empty_answer_path)
    assert report.empty_completion_count == 1

def test_truncated_answer_fails_preflight(truncated_answer_path):
    inspector = SFTInspector(tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct", max_seq_len_threshold=256)
    report = inspector.inspect_file(truncated_answer_path)
    assert report.truncated_answers_count == 1
