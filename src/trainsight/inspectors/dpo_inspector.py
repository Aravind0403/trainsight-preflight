import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field


class DPOReport(BaseModel):
    total_samples: int
    valid_samples_count: int = 0
    invalid_rows_count: int = 0
    empty_pair_count: int = 0
    prompt_mismatch_count: int = 0
    duplicate_count: int = 0

    # Exact token metrics
    avg_prompt_len: float = 0.0
    avg_chosen_len: float = 0.0
    avg_rejected_len: float = 0.0
    length_bias_ratio: float = 1.0

    # Anomaly / Defect metrics
    identical_pairs_count: int = 0
    near_identical_count: int = 0
    preference_reversal_count: int = 0
    score_tie_count: int = 0

    # Budget tracking (1024 prompt / 1024 completion / 2048 total)
    over_budget_prompts_count: int = 0
    over_budget_completions_count: int = 0
    over_budget_total_count: int = 0

    phase_quadrant: str = "Clean Execution"
    warnings: List[str] = Field(default_factory=list)
    advisories: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

    @property
    def total_pairs(self) -> int:
        return self.total_samples

    @property
    def valid_pairs_count(self) -> int:
        return self.valid_samples_count

    @property
    def invalid_pairs_count(self) -> int:
        return self.invalid_rows_count

    @property
    def over_budget_completions(self) -> int:
        return self.over_budget_completions_count


class DPOInspector:
    """
    Inspector for Direct Preference Optimization (DPO) datasets.

    Replaces character heuristics with deterministic, model-aligned checks:
    - Parses only assistant responses (chosen[1] and rejected[1] in message format).
    - Verifies user message agreement with top-level prompt.
    - Reads score_chosen and score_rejected on a 1-10 scale (or legacy score fields).
    - Measures exact token lengths using model tokenizer & chat template.
    - Evaluates training budgets (1024 prompt / 1024 completion / 2048 total).
    - Distinguishes actionable defects (identical pairs, reversals) from advisories (score ties, truncations).
    - Fails closed when the tokenizer cannot be loaded.
    """

    def __init__(
        self,
        tokenizer_name: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct",
        max_prompt_len: int = 1024,
        max_completion_len: int = 1024,
        max_total_len: int = 2048,
        jaccard_similarity_threshold: float = 0.90,
        strict: bool = False,
        max_seq_len_threshold: Optional[int] = None,
    ):
        if max_seq_len_threshold is not None:
            max_total_len = max_seq_len_threshold
            if max_prompt_len > max_total_len:
                max_prompt_len = max_total_len // 2
            if max_completion_len > max_total_len:
                max_completion_len = max_total_len // 2

        self.tokenizer_name = tokenizer_name
        self.max_prompt_len = max_prompt_len
        self.max_completion_len = max_completion_len
        self.max_total_len = max_total_len
        self.jaccard_similarity_threshold = jaccard_similarity_threshold
        self.strict = strict

        self.tokenizer = None
        self.tokenizer_load_error = None

        if self.tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            except Exception as e:
                self.tokenizer = None
                self.tokenizer_load_error = str(e)
                if self.strict:
                    raise RuntimeError(
                        f"Unable to inspect: Tokenizer '{self.tokenizer_name}' could not be loaded ({e}). "
                        "Exact DPO inspection requires the tokenizer."
                    ) from e

    def _estimate_tokens_fallback(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text.strip()) // 4)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Word-level Jaccard similarity strictly on assistant responses."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0

    def _parse_dpo_sample(
        self, sample: Dict[str, Any]
    ) -> tuple[List[Dict[str, str]], str, str, str, bool, bool]:
        """
        Parses a DPO sample into:
            (prompt_messages, prompt_str, asst_chosen, asst_rejected, is_missing, is_prompt_mismatch)

        Supports:
        - Key-pair strings: {"prompt": "...", "chosen": "...", "rejected": "..."}
        - Message-list candidates: {"prompt": "...", "chosen": [{"role": "assistant", ...}], ...}
        - Full conversational histories: {"chosen": [u1, a1, u2, a_c], "rejected": [u1, a1, u2, a_r]}
        """
        if not isinstance(sample, dict):
            return [], "", "", "", True, False

        prompt_field = sample.get("prompt")
        raw_chosen = sample.get("chosen")
        raw_rejected = sample.get("rejected")

        is_chosen_list = isinstance(raw_chosen, list) and all(isinstance(m, dict) for m in raw_chosen)
        is_rejected_list = isinstance(raw_rejected, list) and all(isinstance(m, dict) for m in raw_rejected)

        prompt_messages: List[Dict[str, str]] = []
        prompt_str = ""
        asst_chosen = ""
        asst_rejected = ""
        prompt_mismatch = False

        if is_chosen_list and is_rejected_list:
            chosen_asst = [m for m in raw_chosen if m.get("role") == "assistant"]
            rejected_asst = [m for m in raw_rejected if m.get("role") == "assistant"]

            if chosen_asst:
                asst_chosen = str(chosen_asst[-1].get("content", "")).strip()
            if rejected_asst:
                asst_rejected = str(rejected_asst[-1].get("content", "")).strip()

            if prompt_field:
                if isinstance(prompt_field, str):
                    prompt_str = prompt_field.strip()
                    prompt_messages = [{"role": "user", "content": prompt_str}]
                elif isinstance(prompt_field, list) and all(isinstance(m, dict) for m in prompt_field):
                    prompt_messages = prompt_field
                    user_texts = [m.get("content", "") for m in prompt_field if m.get("role") == "user"]
                    prompt_str = "\n".join(user_texts).strip()

                # Verify agreement with user messages in candidate lists
                chosen_user_msgs = [m.get("content", "") for m in raw_chosen if m.get("role") == "user"]
                rejected_user_msgs = [m.get("content", "") for m in raw_rejected if m.get("role") == "user"]
                if chosen_user_msgs and prompt_str and chosen_user_msgs[0].strip() != prompt_str:
                    prompt_mismatch = True
                if rejected_user_msgs and prompt_str and rejected_user_msgs[0].strip() != prompt_str:
                    prompt_mismatch = True
            else:
                last_chosen_idx = len(raw_chosen) - 1
                while last_chosen_idx >= 0 and raw_chosen[last_chosen_idx].get("role") != "assistant":
                    last_chosen_idx -= 1

                last_rejected_idx = len(raw_rejected) - 1
                while last_rejected_idx >= 0 and raw_rejected[last_rejected_idx].get("role") != "assistant":
                    last_rejected_idx -= 1

                chosen_prefix = raw_chosen[:last_chosen_idx] if last_chosen_idx > 0 else []
                rejected_prefix = raw_rejected[:last_rejected_idx] if last_rejected_idx > 0 else []

                if chosen_prefix != rejected_prefix and chosen_prefix and rejected_prefix:
                    prompt_mismatch = True

                prompt_messages = chosen_prefix
                user_texts = [m.get("content", "") for m in prompt_messages if m.get("role") == "user"]
                prompt_str = "\n".join(user_texts).strip()
                if not prompt_str and prompt_messages:
                    prompt_str = " ".join([str(m.get("content", "")) for m in prompt_messages]).strip()
        else:
            if isinstance(prompt_field, str):
                prompt_str = prompt_field.strip()
                prompt_messages = [{"role": "user", "content": prompt_str}] if prompt_str else []
            elif isinstance(prompt_field, list) and all(isinstance(m, dict) for m in prompt_field):
                prompt_messages = prompt_field
                user_texts = [m.get("content", "") for m in prompt_field if m.get("role") == "user"]
                prompt_str = "\n".join(user_texts).strip()
            elif prompt_field:
                prompt_str = str(prompt_field).strip()
                prompt_messages = [{"role": "user", "content": prompt_str}] if prompt_str else []

            if isinstance(raw_chosen, str):
                asst_chosen = raw_chosen.strip()
            elif isinstance(raw_chosen, list):
                asst_msgs = [m for m in raw_chosen if isinstance(m, dict) and m.get("role") == "assistant"]
                asst_chosen = str(asst_msgs[-1].get("content", "")).strip() if asst_msgs else ""
            elif raw_chosen is not None:
                asst_chosen = str(raw_chosen).strip()

            if isinstance(raw_rejected, str):
                asst_rejected = raw_rejected.strip()
            elif isinstance(raw_rejected, list):
                asst_msgs = [m for m in raw_rejected if isinstance(m, dict) and m.get("role") == "assistant"]
                asst_rejected = str(asst_msgs[-1].get("content", "")).strip() if asst_msgs else ""
            elif raw_rejected is not None:
                asst_rejected = str(raw_rejected).strip()

        is_missing = not prompt_str or not asst_chosen or not asst_rejected
        return prompt_messages, prompt_str, asst_chosen, asst_rejected, is_missing, prompt_mismatch

    def inspect_file(self, file_path: Path) -> DPOReport:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        samples: List[Dict[str, Any]] = []
        invalid_rows_count = 0

        if file_path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(str(file_path))
                samples = table.to_pylist()
            except Exception as e:
                return DPOReport(
                    total_samples=0,
                    invalid_rows_count=1,
                    phase_quadrant="Invalid Dataset",
                    warnings=[f"Failed to read Parquet dataset '{file_path.name}': {e}"],
                    recommendations=["Ensure the file is a valid Parquet table and pyarrow is installed."],
                )
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        invalid_rows_count += 1
                        continue

        if not samples:
            return DPOReport(
                total_samples=0,
                invalid_rows_count=invalid_rows_count,
                phase_quadrant="Invalid Dataset" if invalid_rows_count > 0 else "Empty Dataset",
                warnings=["Dataset is empty or contains no valid records."],
                recommendations=["Check dataset path and jsonl file content."],
            )

        # Fail closed if tokenizer was requested but failed to load
        if self.tokenizer_name and self.tokenizer is None:
            return DPOReport(
                total_samples=len(samples),
                invalid_rows_count=invalid_rows_count,
                phase_quadrant="Unable to Inspect",
                warnings=[
                    f"Unable to inspect: Tokenizer '{self.tokenizer_name}' could not be loaded ({self.tokenizer_load_error}). "
                    "Exact DPO inspection requires the tokenizer."
                ],
                recommendations=[
                    "Ensure the model tokenizer files are cached locally or pass a valid local tokenizer path."
                ],
            )

        prompt_lens = []
        chosen_lens = []
        rejected_lens = []

        identical_count = 0
        near_identical_count = 0
        empty_pair_count = 0
        prompt_mismatch_count = 0
        seen_prompts = set()
        duplicate_count = 0
        preference_reversal_count = 0
        score_tie_count = 0

        over_budget_prompts_count = 0
        over_budget_completions_count = 0
        over_budget_total_count = 0

        valid_samples_count = 0

        for sample in samples:
            prompt_msgs, prompt_str, asst_chosen, asst_rejected, is_missing, is_prompt_mismatch = (
                self._parse_dpo_sample(sample)
            )

            if is_missing:
                empty_pair_count += 1
                continue

            if is_prompt_mismatch:
                prompt_mismatch_count += 1

            # Quality Score checks (reading score_chosen / score_rejected or fallbacks)
            c_score = sample.get("score_chosen")
            if c_score is None:
                c_score = sample.get("chosen_score") if sample.get("chosen_score") is not None else sample.get("chosen_rating")

            r_score = sample.get("score_rejected")
            if r_score is None:
                r_score = sample.get("rejected_score") if sample.get("rejected_score") is not None else sample.get("rejected_rating")

            if c_score is not None and r_score is not None:
                try:
                    c_val = float(c_score)
                    r_val = float(r_score)
                    if c_val < r_val:
                        preference_reversal_count += 1
                    elif c_val == r_val:
                        score_tie_count += 1
                except (ValueError, TypeError):
                    pass

            # Identical / Near-identical assistant response check
            if asst_chosen.strip().lower() == asst_rejected.strip().lower():
                identical_count += 1
            else:
                sim = self._jaccard_similarity(asst_chosen, asst_rejected)
                if sim >= self.jaccard_similarity_threshold:
                    near_identical_count += 1

            # Prompt deduplication check
            if prompt_str:
                prompt_key = prompt_str.lower()
                if prompt_key in seen_prompts:
                    duplicate_count += 1
                else:
                    seen_prompts.add(prompt_key)

            # Tokenization (exact BPE with full conversation template or fallback)
            if self.tokenizer is not None:
                try:
                    templated_prompt_msgs = prompt_msgs if prompt_msgs else [{"role": "user", "content": prompt_str}]
                    chat_prompt_str = self.tokenizer.apply_chat_template(
                        templated_prompt_msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    prompt_t = len(self.tokenizer.encode(chat_prompt_str, truncation=False))

                    # Full chosen conversation with assistant turn
                    full_chosen_msgs = list(templated_prompt_msgs) + [{"role": "assistant", "content": asst_chosen}]
                    chat_chosen_str = self.tokenizer.apply_chat_template(
                        full_chosen_msgs,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    full_chosen_t = len(self.tokenizer.encode(chat_chosen_str, truncation=False))
                    chosen_t = max(1, full_chosen_t - prompt_t)

                    # Full rejected conversation with assistant turn
                    full_rejected_msgs = list(templated_prompt_msgs) + [{"role": "assistant", "content": asst_rejected}]
                    chat_rejected_str = self.tokenizer.apply_chat_template(
                        full_rejected_msgs,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    full_rejected_t = len(self.tokenizer.encode(chat_rejected_str, truncation=False))
                    rejected_t = max(1, full_rejected_t - prompt_t)
                except Exception:
                    prompt_t = len(self.tokenizer.encode(prompt_str, truncation=False))
                    chosen_t = len(self.tokenizer.encode(asst_chosen, truncation=False))
                    rejected_t = len(self.tokenizer.encode(asst_rejected, truncation=False))
            else:
                prompt_t = self._estimate_tokens_fallback(prompt_str)
                chosen_t = self._estimate_tokens_fallback(asst_chosen)
                rejected_t = self._estimate_tokens_fallback(asst_rejected)

            prompt_lens.append(prompt_t)
            chosen_lens.append(chosen_t)
            rejected_lens.append(rejected_t)
            valid_samples_count += 1

            # Budget evaluation
            if prompt_t > self.max_prompt_len:
                over_budget_prompts_count += 1
            if chosen_t > self.max_completion_len or rejected_t > self.max_completion_len:
                over_budget_completions_count += 1
            if (prompt_t + chosen_t) > self.max_total_len or (prompt_t + rejected_t) > self.max_total_len:
                over_budget_total_count += 1

        avg_p = float(np.mean(prompt_lens)) if prompt_lens else 0.0
        avg_c = float(np.mean(chosen_lens)) if chosen_lens else 0.0
        avg_r = float(np.mean(rejected_lens)) if rejected_lens else 0.0
        length_bias_ratio = (avg_c / avg_r) if avg_r > 0 else 1.0

        warnings = []
        advisories = []
        recommendations = []

        # Actionable Defects (lead to non-clean quadrant)
        if identical_count > 0:
            warnings.append(
                f"⚠️ {identical_count} pairs have IDENTICAL chosen and rejected responses. "
                "DPO loss gradient will be zero; filter these pairs before training."
            )
            recommendations.append("Filter identical chosen/rejected pairs from the dataset.")

        if preference_reversal_count > 0:
            warnings.append(
                f"⚠️ {preference_reversal_count} preference pairs have REVERSED quality scores (score_chosen < score_rejected). "
                "DPO will optimize towards inferior responses!"
            )
            recommendations.append("Swap chosen/rejected responses or filter inverted quality rating rows.")

        if prompt_mismatch_count > 0:
            warnings.append(
                f"⚠️ {prompt_mismatch_count} pairs have user messages in chosen/rejected that do not match the prompt."
            )
            recommendations.append("Verify multi-turn message alignment with prompt.")

        if empty_pair_count > 0:
            warnings.append(f"⚠️ {empty_pair_count} samples have missing chosen, rejected, or prompt text.")
            recommendations.append("Drop corrupted rows with missing preference text.")

        if invalid_rows_count > 0:
            warnings.append(f"⚠️ {invalid_rows_count} malformed JSON lines were skipped.")
            recommendations.append("Validate JSON syntax.")

        if length_bias_ratio > 1.8 or length_bias_ratio < 0.55:
            warnings.append(
                f"⚠️ Severe length bias detected (Chosen/Rejected avg length ratio = {length_bias_ratio:.2f}). "
                "Model will exploit length over quality!"
            )
            recommendations.append("Apply length-normalized DPO loss (SimPO / DisPo) or rebalance dataset length distributions.")

        # Informational Advisories (confidence flags & training budget observations)
        if score_tie_count > 0:
            advisories.append(
                f"ℹ️ {score_tie_count} pairs have TIED quality scores (score_chosen == score_rejected). "
                "These are distinct responses with equal teacher confidence, not zero-gradient defects."
            )

        if over_budget_prompts_count > 0:
            advisories.append(
                f"ℹ️ {over_budget_prompts_count} prompts exceed the {self.max_prompt_len}-token budget and will be truncated during training."
            )

        if near_identical_count > 0:
            advisories.append(
                f"ℹ️ {near_identical_count} pairs are near-identical (Jaccard similarity > {self.jaccard_similarity_threshold * 100:.0f}%). "
                "DPO margin learning will be constrained."
            )

        if duplicate_count > 0:
            advisories.append(f"ℹ️ {duplicate_count} duplicate prompts detected.")

        # Phase Quadrant Assignment
        if (
            identical_count > 0
            or preference_reversal_count > 0
            or empty_pair_count > 0
            or prompt_mismatch_count > 0
            or invalid_rows_count > 0
        ):
            phase_quadrant = "Defects Detected"
        elif length_bias_ratio > 1.8 or length_bias_ratio < 0.55:
            phase_quadrant = "Quality Warnings"
        else:
            phase_quadrant = "Clean Execution"

        return DPOReport(
            total_samples=len(samples),
            valid_samples_count=valid_samples_count,
            invalid_rows_count=invalid_rows_count,
            empty_pair_count=empty_pair_count,
            prompt_mismatch_count=prompt_mismatch_count,
            duplicate_count=duplicate_count,
            avg_prompt_len=avg_p,
            avg_chosen_len=avg_c,
            avg_rejected_len=avg_r,
            length_bias_ratio=length_bias_ratio,
            identical_pairs_count=identical_count,
            near_identical_count=near_identical_count,
            preference_reversal_count=preference_reversal_count,
            score_tie_count=score_tie_count,
            over_budget_prompts_count=over_budget_prompts_count,
            over_budget_completions_count=over_budget_completions_count,
            over_budget_total_count=over_budget_total_count,
            phase_quadrant=phase_quadrant,
            warnings=warnings,
            advisories=advisories,
            recommendations=recommendations,
        )
