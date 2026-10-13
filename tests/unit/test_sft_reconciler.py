import pytest
import torch
from trainsight.core.contracts import LossContract
from trainsight.core.exceptions import ContractViolationError
from trainsight.reconcilers.sft_reconciler import SFTReconciler

def test_clean_batch_passes_reconciliation(tokenizer):
    p_text = tokenizer.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}], tokenize=False, add_generation_prompt=True)
    c_text = "Paris.<|im_end|>"
    p_ids = tokenizer.encode(p_text, add_special_tokens=False)
    c_ids = tokenizer.encode(c_text, add_special_tokens=False)
    full_ids = p_ids + c_ids

    # Mask prompt, supervise completion
    labels = [-100] * len(p_ids) + c_ids

    batch = {
        "input_ids": torch.tensor([full_ids]),
        "labels": torch.tensor([labels]),
        "attention_mask": torch.tensor([[1] * len(full_ids)]),
    }

    reconciler = SFTReconciler(tokenizer=tokenizer, contract=LossContract(strict=True))
    report = reconciler.audit_batch(batch=batch, prompt_lengths=[len(p_ids)], row_ids=["row_001"])
    assert report.passed is True
    assert report.violations_count == 0

def test_unmasked_prompt_fails_reconciliation(tokenizer):
    p_text = tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}], tokenize=False, add_generation_prompt=True)
    c_text = "Hello!<|im_end|>"
    p_ids = tokenizer.encode(p_text, add_special_tokens=False)
    c_ids = tokenizer.encode(c_text, add_special_tokens=False)
    full_ids = p_ids + c_ids

    # Corrupt prompt labels: set position 0 to token ID
    labels = [-100] * len(p_ids) + c_ids
    labels[0] = p_ids[0]

    batch = {
        "input_ids": torch.tensor([full_ids]),
        "labels": torch.tensor([labels]),
        "attention_mask": torch.tensor([[1] * len(full_ids)]),
    }

    reconciler = SFTReconciler(tokenizer=tokenizer, contract=LossContract(strict=True))
    with pytest.raises(ContractViolationError) as exc_info:
        reconciler.audit_batch(batch=batch, prompt_lengths=[len(p_ids)], row_ids=["row_001"])
    assert "PROMPT_LABELS_UNMASKED" in str(exc_info.value)

def test_unmasked_padding_fails_reconciliation(tokenizer):
    full_ids = [151644, 872, 198, 151645, 151643] # last is pad
    labels = [-100, -100, -100, 151645, 42]        # pad has label 42
    mask = [1, 1, 1, 1, 0]                         # last is pad (mask=0)

    batch = {
        "input_ids": torch.tensor([full_ids]),
        "labels": torch.tensor([labels]),
        "attention_mask": torch.tensor([mask]),
    }

    reconciler = SFTReconciler(tokenizer=tokenizer, contract=LossContract(strict=True))
    with pytest.raises(ContractViolationError) as exc_info:
        reconciler.audit_batch(batch=batch, prompt_lengths=[3], row_ids=["row_001"])
    assert "PADDING_LABELS_UNMASKED" in str(exc_info.value)
