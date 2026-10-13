import torch
from typing import Dict, Any, List, Optional
from transformers import PreTrainedTokenizerBase

from trainsight.core.contracts import LossContract
from trainsight.core.exceptions import ContractViolationError
from trainsight.core.report import AuditReport, RowDiagnosis, TensorEvidence
from trainsight.preprocessing.sft_preprocessor import SFTPreprocessor

class SFTReconciler:
    """Audits actual in-memory trainer dataloader batches against the declared loss contract."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        contract: Optional[LossContract] = None,
    ):
        self.tokenizer = tokenizer
        self.contract = contract or LossContract()
        self.preprocessor = SFTPreprocessor(
            tokenizer_name=tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else "Qwen/Qwen2.5-0.5B-Instruct",
            max_seq_len=self.contract.max_seq_len,
            truncation=True,
        )

    def audit_batch(
        self,
        batch: Dict[str, torch.Tensor],
        prompt_lengths: List[int],
        row_ids: Optional[List[str]] = None,
        raw_samples: Optional[List[Dict[str, Any]]] = None,
    ) -> AuditReport:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")

        batch_size = input_ids.shape[0]
        row_ids = row_ids or [f"row_{i:03d}" for i in range(batch_size)]

        diagnoses: List[RowDiagnosis] = []
        valid_count = 0

        for b in range(batch_size):
            r_id = row_ids[b]
            b_inputs = input_ids[b]
            b_labels = labels[b]
            b_mask = attention_mask[b] if attention_mask is not None else torch.ones_like(b_inputs)
            p_len = prompt_lengths[b]

            input_list = b_inputs.tolist()
            labels_list = b_labels.tolist()
            mask_list = b_mask.tolist()

            active_indices = [i for i, lbl in enumerate(labels_list) if lbl != -100]
            causal_indices = [i for i in active_indices if i >= 1]
            pad_indices = [i for i, m in enumerate(mask_list) if m == 0]
            unmasked_pad = [i for i in pad_indices if labels_list[i] != -100]
            unmasked_prompt = [i for i in range(min(p_len, len(labels_list))) if labels_list[i] != -100]

            evidence = TensorEvidence(
                row_id=r_id,
                sequence_length=len(input_list),
                prompt_token_length=p_len,
                active_labels_count=len(active_indices),
                causal_loss_contributing_tokens=len(causal_indices),
                unmasked_prompt_tokens_count=len(unmasked_prompt),
                unmasked_padding_tokens_count=len(unmasked_pad),
                decoded_prompt_preview=self.tokenizer.decode(input_list[:p_len], skip_special_tokens=False)[:80],
                decoded_answer_preview=self.tokenizer.decode(input_list[p_len:], skip_special_tokens=False)[:80],
                labels_sample=labels_list[:25],
            )

            # Check 1: Non-empty supervision
            if len(active_indices) == 0:
                diagnoses.append(RowDiagnosis(
                    row_id=r_id,
                    error_code="NO_SUPERVISED_TOKENS",
                    error_message="Example has zero supervised tokens (all labels are -100).",
                    evidence=evidence,
                ))
                continue

            # Check 2: Causal shift loss contribution (index >= 1)
            if self.contract.require_causal_shift_supervision and len(causal_indices) == 0:
                diagnoses.append(RowDiagnosis(
                    row_id=r_id,
                    error_code="NO_CAUSAL_SUPERVISION",
                    error_message="Supervised tokens exist only at index 0; no token contributes to shifted causal loss.",
                    evidence=evidence,
                ))
                continue

            # Check 3: Prompt tokens unmasked
            if self.contract.answer_only_loss and len(unmasked_prompt) > 0:
                diagnoses.append(RowDiagnosis(
                    row_id=r_id,
                    error_code="PROMPT_LABELS_UNMASKED",
                    error_message=f"{len(unmasked_prompt)} prompt token(s) have label != -100. Prompt must not contribute to loss.",
                    evidence=evidence,
                ))
                continue

            # Check 4: Padded positions unmasked
            if self.contract.mask_padding and len(unmasked_pad) > 0:
                diagnoses.append(RowDiagnosis(
                    row_id=r_id,
                    error_code="PADDING_LABELS_UNMASKED",
                    error_message=f"{len(unmasked_pad)} padded position(s) (attention_mask == 0) have label != -100.",
                    evidence=evidence,
                ))
                continue

            # Check 5: Label-to-token identity
            mismatched_labels = [i for i in active_indices if labels_list[i] != input_list[i]]
            if len(mismatched_labels) > 0:
                diagnoses.append(RowDiagnosis(
                    row_id=r_id,
                    error_code="LABEL_TOKEN_MISMATCH",
                    error_message=f"Active labels at indices {mismatched_labels[:3]} do not match input token IDs.",
                    evidence=evidence,
                ))
                continue

            # Check 6: Template delimiter verification
            decoded_prompt = self.tokenizer.decode(input_list[:p_len], skip_special_tokens=False)
            if (self.contract.expected_template_prefix not in decoded_prompt or 
                self.contract.expected_template_suffix not in decoded_prompt):
                diagnoses.append(RowDiagnosis(
                    row_id=r_id,
                    error_code="TEMPLATE_MISMATCH",
                    error_message=f"Prompt missing required template delimiters ({self.contract.expected_template_prefix} ... {self.contract.expected_template_suffix}).",
                    evidence=evidence,
                ))
                continue

            # Check 7: Substantive content (not merely EOS / closing delimiters)
            content_tokens = [
                input_list[i] for i in active_indices
                if input_list[i] not in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
                and self.tokenizer.decode([input_list[i]]).strip() not in ("<|im_end|>", "")
            ]
            if len(content_tokens) == 0:
                diagnoses.append(RowDiagnosis(
                    row_id=r_id,
                    error_code="EMPTY_ANSWER_CONTENT",
                    error_message="Supervised tokens consist entirely of EOS or template closing delimiters without answer content.",
                    evidence=evidence,
                ))
                continue

            # Check 8: Token sequence match against approved preprocessor contract
            if raw_samples is not None and b < len(raw_samples):
                clean_ex = self.preprocessor.process_sample(raw_samples[b])
                approved_tokens = clean_ex.input_ids
                actual_unpadded = b_inputs[b_mask == 1].tolist()
                if approved_tokens != actual_unpadded and approved_tokens[:-1] != actual_unpadded:
                    diagnoses.append(RowDiagnosis(
                        row_id=r_id,
                        error_code="TOKEN_SEQUENCE_DIVERGENCE",
                        error_message=f"Dataloader tokens diverge from approved contract (expected {len(approved_tokens)}, got {len(actual_unpadded)}).",
                        evidence=evidence,
                    ))
                    continue

            valid_count += 1

        passed = len(diagnoses) == 0
        report = AuditReport(
            total_inspected=batch_size,
            valid_count=valid_count,
            violations_count=len(diagnoses),
            passed=passed,
            diagnoses=diagnoses,
        )

        if not passed and self.contract.strict:
            first = diagnoses[0]
            raise ContractViolationError(first.error_code, first.error_message, row_id=first.row_id)

        return report
