import pytest
import torch
from trainsight.core.contracts import LossContract
from trainsight.core.exceptions import ContractViolationError
from trainsight.reconcilers.sft_reconciler import SFTReconciler

def test_token_zero_only_fails_causal_shift(tokenizer):
    # If the only supervised token is at index 0, it contributes 0 to causal shifted loss
    full_ids = [100, 200, 300]
    labels = [100, -100, -100]  # Only token 0 supervised!
    mask = [1, 1, 1]

    batch = {
        "input_ids": torch.tensor([full_ids]),
        "labels": torch.tensor([labels]),
        "attention_mask": torch.tensor([mask]),
    }

    reconciler = SFTReconciler(tokenizer=tokenizer, contract=LossContract(strict=True, answer_only_loss=False))
    with pytest.raises(ContractViolationError) as exc_info:
        reconciler.audit_batch(batch=batch, prompt_lengths=[0], row_ids=["row_001"])
    assert "NO_CAUSAL_SUPERVISION" in str(exc_info.value)
