from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class TensorEvidence:
    row_id: str
    sequence_length: int
    prompt_token_length: int
    active_labels_count: int
    causal_loss_contributing_tokens: int
    unmasked_prompt_tokens_count: int
    unmasked_padding_tokens_count: int
    decoded_prompt_preview: str
    decoded_answer_preview: str
    labels_sample: List[Any]

@dataclass
class RowDiagnosis:
    row_id: str
    error_code: str
    error_message: str
    severity: str = "critical"  # "critical" or "warning"
    evidence: Optional[TensorEvidence] = None

@dataclass
class AuditReport:
    total_inspected: int
    valid_count: int
    violations_count: int
    passed: bool
    diagnoses: List[RowDiagnosis] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "total_inspected": self.total_inspected,
            "valid_count": self.valid_count,
            "violations_count": self.violations_count,
            "passed": self.passed,
            "diagnoses": [
                {
                    "row_id": d.row_id,
                    "error_code": d.error_code,
                    "error_message": d.error_message,
                    "severity": d.severity,
                }
                for d in self.diagnoses
            ],
        }
