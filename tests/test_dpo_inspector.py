import pytest
from pathlib import Path
from trainsight.inspectors.dpo_inspector import DPOInspector


def test_dpo_inspector_identical_pair(tmp_path: Path):
    dpo_file = tmp_path / "test_dpo.jsonl"
    dpo_file.write_text(
        '{"prompt": "Q1", "chosen": "Same text", "rejected": "Same text"}\n'
        '{"prompt": "Q2", "chosen": "Good answer", "rejected": "Bad answer"}\n'
    )

    inspector = DPOInspector()
    report = inspector.inspect_file(dpo_file)

    assert report.total_samples == 2
    assert report.identical_pairs_count == 1
    assert len(report.warnings) >= 1
    assert "IDENTICAL" in report.warnings[0]


def test_dpo_inspector_near_identical(tmp_path: Path):
    dpo_file = tmp_path / "test_dpo.jsonl"
    dpo_file.write_text(
        '{"prompt": "Q1", "chosen": "The quick brown fox jumps over dog", "rejected": "The quick brown fox leaps over dog"}\n'
    )

    inspector = DPOInspector(jaccard_similarity_threshold=0.7)
    report = inspector.inspect_file(dpo_file)

    assert report.total_samples == 1
    assert report.near_identical_count == 1
