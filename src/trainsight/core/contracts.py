from dataclasses import dataclass
from typing import Optional, Literal, Any

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
    expected_template_prefix: Optional[str] = "<|im_start|>user"
    expected_template_suffix: Optional[str] = "<|im_start|>assistant\n"
    check_template_delimiters: bool = True

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, **overrides) -> "LossContract":
        """
        Derives expected template delimiters from the supplied tokenizer where possible.
        Falls back to standard ChatML delimiters if template rendering is unavailable.
        """
        prefix = overrides.get("expected_template_prefix")
        suffix = overrides.get("expected_template_suffix")

        if prefix is None or suffix is None:
            marker = "__TRAINSIGHT_PROMPT_MARKER__"
            try:
                if hasattr(tokenizer, "apply_chat_template"):
                    rendered = tokenizer.apply_chat_template(
                        [{"role": "user", "content": marker}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if marker in rendered:
                        parts = rendered.split(marker)
                        prefix = prefix or parts[0].strip()
                        suffix = suffix or parts[1].strip()
            except Exception:
                pass

        return cls(
            expected_template_prefix=prefix if prefix is not None else "<|im_start|>user",
            expected_template_suffix=suffix if suffix is not None else "<|im_start|>assistant\n",
            **overrides,
        )
