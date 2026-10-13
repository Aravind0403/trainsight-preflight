"""
15-line Quickstart: Preflight file check + in-memory batch reconciliation.
"""
from transformers import AutoTokenizer
from trainsight import SFTInspector, SFTReconciler, LossContract

# 1. Run Preflight Inspection on Disk
inspector = SFTInspector(tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct", max_seq_len_threshold=256)
report = inspector.inspect_file("tests/fixtures/clean_sft.jsonl")
print(f"Preflight Passed: {report.valid_samples_count}/{report.total_samples} samples valid.")

# 2. Audit an In-Memory Dataloader Batch
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
reconciler = SFTReconciler(tokenizer=tokenizer, contract=LossContract(strict=True))
# reconciler.audit_batch(batch, prompt_lengths=[...])
print("TrainSight is ready for training pipeline integration.")
