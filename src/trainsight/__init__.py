"""
TrainSight: Preflight Checks for LLM Training Inputs and Loss Contracts.
"""

__version__ = "0.2.3"

from trainsight.core.contracts import LossContract
from trainsight.core.exceptions import (
    TrainSightError,
    ContractViolationError,
    PreflightQualityGateError,
)
from trainsight.core.report import AuditReport, RowDiagnosis, TensorEvidence
from trainsight.inspectors.sft_inspector import SFTInspector, SFTReport
from trainsight.inspectors.dpo_inspector import DPOInspector, DPOReport
from trainsight.preprocessing.sft_preprocessor import SFTPreprocessor, SFTExample
from trainsight.preprocessing.dpo_preprocessor import DPOPreprocessor, DPOExample
from trainsight.reconcilers.sft_reconciler import SFTReconciler
from trainsight.reconcilers.callbacks import TrainSightCallback

__all__ = [
    "LossContract",
    "TrainSightError",
    "ContractViolationError",
    "PreflightQualityGateError",
    "AuditReport",
    "RowDiagnosis",
    "TensorEvidence",
    "SFTInspector",
    "SFTReport",
    "DPOInspector",
    "DPOReport",
    "SFTPreprocessor",
    "SFTExample",
    "DPOPreprocessor",
    "DPOExample",
    "SFTReconciler",
    "TrainSightCallback",
]
