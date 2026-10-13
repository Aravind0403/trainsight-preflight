from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class LossContract:
    """Declared contract for loss calculation on training batches."""
    task_type: Literal["sft", "dpo"] = "sft"
    chat_template: str = "chatml"
    max_seq_len: int = 2048
    answer_only_loss: bool = True
    mask_padding: bool = True
    require_causal_shift_supervision: bool = True
    strict: bool = True
    expected_template_prefix: str = "<|im_start|>user"
    expected_template_suffix: str = "<|im_start|>assistant\n"
