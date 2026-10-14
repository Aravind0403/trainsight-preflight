"""
Quickstart: Preflight file check + explicit in-memory batch reconciliation.
"""
from pathlib import Path
from transformers import AutoTokenizer
import torch
from trainsight import SFTInspector, SFTReconciler, SFTPreprocessor, LossContract

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 1. Run Preflight Inspection on Disk
fixture_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "clean_sft.jsonl"
inspector = SFTInspector(tokenizer_name=MODEL_NAME, max_seq_len_threshold=256)
report = inspector.inspect_file(str(fixture_path))
print(f"1. Preflight Inspection Passed: {report.valid_samples_count}/{report.total_samples} samples valid.")

# 2. Audit a Batch Directly Built from Preprocessed Samples
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
preprocessor = SFTPreprocessor(tokenizer_name=MODEL_NAME, max_seq_len=256)
contract = LossContract(max_seq_len=256, strict=True)
reconciler = SFTReconciler(tokenizer=tokenizer, contract=contract)

# Verified sample definition
sample = {"id": "row_001", "prompt": "What is the capital of France?", "completion": "The capital of France is Paris."}

# Obtain exact token boundaries from the formatted chat template
clean_ex = preprocessor.process_sample(sample)

# Construct model-ready batch directly from the preprocessed samples to guarantee row correspondence:
batch = {
    "input_ids": torch.tensor([clean_ex.input_ids]),
    "labels": torch.tensor([clean_ex.labels]),
    "attention_mask": torch.tensor([clean_ex.attention_mask]),
}

audit_report = reconciler.audit_batch(
    batch=batch,
    prompt_lengths=[clean_ex.prompt_tokens_len],
    row_ids=[sample["id"]],
    raw_samples=[sample],
)

print(f"2. Direct Batch Reconciliation Passed: {audit_report.passed} (Violations: {audit_report.violations_count})")
