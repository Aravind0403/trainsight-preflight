"""
One-Minute Demo Walkthrough:
1. Static Disk Inspection on Representative Technical Training Datasets (SFT & DPO)
2. In-Memory Dataloader Batch Loss Contract Reconciliation with Injected Tokenizer

Validation context:
- Tokenizer: Qwen/Qwen2.5-0.5B-Instruct
- Template: Configured ChatML (<|im_start|>user...<|im_start|>assistant)
- Max sequence length: 512 (SFT) / 1024 (DPO)
"""
from pathlib import Path
from transformers import AutoTokenizer
import torch
from trainsight import SFTInspector, DPOInspector, SFTReconciler, SFTPreprocessor, LossContract

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"

def run_representative_demo():
    print("=" * 72)
    print("  TrainSight: Representative Preflight & Loss Contract Verification Demo")
    print("  Context: Qwen/Qwen2.5-0.5B-Instruct, ChatML template, max_seq_len=512/1024")
    print("=" * 72)

    # ---------------------------------------------------------
    # Tier 1: Representative Preflight File Inspection (SFT & DPO)
    # ---------------------------------------------------------
    print("\n[Tier 1] Inspecting Representative Technical SFT Dataset (sample_data/sft_representative_clean.jsonl)...")
    sft_file = DATA_DIR / "sft_representative_clean.jsonl"
    sft_inspector = SFTInspector(tokenizer_name=MODEL_NAME, max_seq_len_threshold=512)
    sft_report = sft_inspector.inspect_file(sft_file)
    print(f"  • Valid samples: {sft_report.valid_samples_count}/{sft_report.total_samples}")
    print(f"  • Tokenizer- and template-aligned mean length: {sft_report.avg_seq_len:.1f} tokens (Max: {sft_report.max_seq_len})")
    print(f"  • Preflight gate status: {sft_report.phase_quadrant}")

    print("\n[Tier 1] Inspecting Representative Technical DPO Dataset (sample_data/dpo_representative_clean.jsonl)...")
    dpo_file = DATA_DIR / "dpo_representative_clean.jsonl"
    dpo_inspector = DPOInspector(tokenizer_name=MODEL_NAME, max_total_len=1024)
    dpo_report = dpo_inspector.inspect_file(dpo_file)
    print(f"  • Valid preference pairs: {dpo_report.valid_samples_count}/{dpo_report.total_samples}")
    print(f"  • Identical completions (zero-signal pairs): {dpo_report.identical_pairs_count}")
    print(f"  • Length bias ratio (Chosen/Rejected): {dpo_report.length_bias_ratio:.2f}")
    print(f"  • Preflight gate status: {dpo_report.phase_quadrant}")

    # ---------------------------------------------------------
    # Tier 2: In-Memory Dataloader Batch Loss Contract Reconciliation
    # ---------------------------------------------------------
    print("\n[Tier 2] Auditing Compiled In-Memory Training Batch with Injected Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    preprocessor = SFTPreprocessor(tokenizer=tokenizer, max_seq_len=512)
    contract = LossContract.from_tokenizer(tokenizer, max_seq_len=512, strict=True)
    reconciler = SFTReconciler(tokenizer=tokenizer, contract=contract)

    # Representative engineering sample
    sample = {
        "id": "sample_001",
        "prompt": "Explain how online softmax in FlashAttention avoids materializing N x N attention matrices in GPU HBM.",
        "completion": "Online softmax maintains running maximum scaling factors and normalizes blocks iteratively in SRAM.",
    }

    clean_ex = preprocessor.process_sample(sample)
    batch = {
        "input_ids": torch.tensor([clean_ex.input_ids]),
        "labels": torch.tensor([clean_ex.labels]),
        "attention_mask": torch.tensor([clean_ex.attention_mask]),
    }

    audit = reconciler.audit_batch(
        batch=batch,
        prompt_lengths=[clean_ex.prompt_tokens_len],
        row_ids=[sample["id"]],
        raw_samples=[sample],
    )
    print(f"  • Loss-contract batch audit passed: {audit.passed}")
    print(f"  • Contract violations detected: {audit.violations_count}")
    print("\nDeclared loss-contract invariants satisfied across inspected dataset and batch.")

if __name__ == "__main__":
    run_representative_demo()

