import pytest
from pathlib import Path
from transformers import AutoTokenizer

FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

@pytest.fixture
def clean_sft_path():
    return FIXTURES_DIR / "clean_sft.jsonl"

@pytest.fixture
def empty_answer_path():
    return FIXTURES_DIR / "faults" / "empty_answer.jsonl"

@pytest.fixture
def truncated_answer_path():
    return FIXTURES_DIR / "faults" / "truncated_answer.jsonl"

@pytest.fixture
def clean_dpo_path():
    return FIXTURES_DIR / "clean_dpo.jsonl"

@pytest.fixture
def dpo_defects_path():
    return FIXTURES_DIR / "faults" / "dpo_defects.jsonl"
