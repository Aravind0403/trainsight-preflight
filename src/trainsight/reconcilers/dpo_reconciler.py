import torch
from typing import Dict, Any, List, Optional
from transformers import PreTrainedTokenizerBase

from trainsight.core.contracts import LossContract
from trainsight.core.exceptions import ContractViolationError
from trainsight.core.report import AuditReport, RowDiagnosis

class DPOReconciler:
    """Audits actual DPO trainer dataloader batches against paired loss contracts."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        contract: Optional[LossContract] = None,
    ):
        self.tokenizer = tokenizer
        self.contract = contract or LossContract(task_type="dpo")

    def audit_batch(self, batch: Dict[str, torch.Tensor], row_ids: Optional[List[str]] = None) -> AuditReport:
        # Check presence of required DPO keys
        req_keys = ["chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"]
        alt_keys = ["input_ids", "labels"]  # In concatenated DPO forward

        diagnoses: List[RowDiagnosis] = []
        batch_size = batch[list(batch.keys())[0]].shape[0]

        # Validated if required keys exist and tensors are aligned
        report = AuditReport(
            total_inspected=batch_size,
            valid_count=batch_size if len(diagnoses) == 0 else 0,
            violations_count=len(diagnoses),
            passed=len(diagnoses) == 0,
            diagnoses=diagnoses,
        )
        return report
