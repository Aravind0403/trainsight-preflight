"""
Quickstart: Preflight file check + explicit in-memory batch reconciliation.
"""
from transformers import AutoTokenizer
import torch
from trainsight import SFTInspector, SFTReconciler, SFTPreprocessor, LossContract

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 1. Run Preflight Inspection on Disk
inspector = SFTInspector(tokenizer_name=MODEL_NAME, max_seq_len_threshold=256)
report = inspector.inspect_file("tests/fixtures/clean_sft.jsonl")
print(f"1. Preflight Inspection Passed: {report.valid_samples_count}/{report.total_samples} samples valid.")

# 2. Audit an In-Memory Dataloader Batch with Explicit Boundaries
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
preprocessor = SFTPreprocessor(tokenizer_name=MODEL_NAME, max_seq_len=256)

# Raw training samples
sample = {"id": "row_001", "prompt": "What is the capital of France?", "completion": "Paris.<|im_end|>"}

# Obtain exact token boundaries from the formatted chat template
clean_ex = preprocessor.process_sample(sample)
prompt_len = clean_ex.prompt_tokens_len  # Formatted prompt length including chat template delimiters

# Model-ready tensors (with answer-only loss masking)
batch = {
    "input_ids": torch.tensor([clean_ex.input_ids]),
    "labels": torch.tensor([clean_ex.labels]),
    "attention_mask": torch.tensor([clean_ex.attention_mask]),
}

reconciler = SFTReconciler(tokenizer=tokenizer, contract=LossContract(strict=True))
audit_report = reconciler.audit_batch(
    batch=batch,
    prompt_lengths=[prompt_len],
    row_ids=[sample["id"]],
    raw_samples=[sample],
)

print(f"2. In-Memory Batch Reconciliation Passed: {audit_report.passed} (Violations: {audit_report.violations_count})")
