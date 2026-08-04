import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field


class SFTReport(BaseModel):
    total_samples: int
    avg_seq_len: float
    std_seq_len: float
    min_seq_len: int
    max_seq_len: int
    p95_seq_len: float
    p99_seq_len: float
    oom_risk_count: int
    empty_completion_count: int
    duplicate_count: int
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class SFTInspector:
    """Inspector for Supervised Fine-Tuning (SFT) datasets."""

    def __init__(self, max_seq_len_threshold: int = 2048, tokenizer_name: Optional[str] = None):
        self.max_seq_len_threshold = max_seq_len_threshold
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception:
                self.tokenizer = None

    def _estimate_token_count(self, text: str) -> int:
        """Fast heuristic token counter (~4 chars per token) or model-aligned BPE tokenizer."""
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text, truncation=False))
            except Exception:
                pass
        return max(1, len(text.strip()) // 4)

    def inspect_file(self, file_path: Path) -> SFTReport:
        """Reads JSONL dataset and computes token distribution, OOM risk, and anomalies."""
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        samples: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    samples.append(data)
                except json.JSONDecodeError:
                    continue

        if not samples:
            return SFTReport(
                total_samples=0,
                avg_seq_len=0.0,
                std_seq_len=0.0,
                min_seq_len=0,
                max_seq_len=0,
                p95_seq_len=0.0,
                p99_seq_len=0.0,
                oom_risk_count=0,
                empty_completion_count=0,
                duplicate_count=0,
                warnings=["Dataset is empty or contains invalid JSON lines."],
                recommendations=["Check dataset formatting. Ensure valid JSONL."],
            )

        seq_lengths = []
        empty_completions = 0
        seen_prompts = set()
        duplicates = 0
        oom_risk = 0

        for sample in samples:
            # Handle standard keys: 'prompt'/'completion', 'instruction'/'output', 'question'/'answer', or 'messages'
            prompt_text = sample.get("prompt") or sample.get("instruction") or sample.get("question") or ""
            completion_text = sample.get("completion") or sample.get("output") or sample.get("response") or sample.get("answer") or ""

            if isinstance(sample.get("messages"), list):
                full_text = " ".join([m.get("content", "") for m in sample["messages"]])
            else:
                full_text = f"{prompt_text} {completion_text}"

            if not completion_text and not sample.get("messages"):
                empty_completions += 1

            token_count = self._estimate_token_count(full_text)
            seq_lengths.append(token_count)

            if token_count > self.max_seq_len_threshold:
                oom_risk += 1

            prompt_key = prompt_text.strip().lower()
            if prompt_key:
                if prompt_key in seen_prompts:
                    duplicates += 1
                else:
                    seen_prompts.add(prompt_key)

        seq_lengths_arr = np.array(seq_lengths)
        avg_len = float(np.mean(seq_lengths_arr))
        std_len = float(np.std(seq_lengths_arr))
        min_len = int(np.min(seq_lengths_arr))
        max_len = int(np.max(seq_lengths_arr))
        p95_len = float(np.percentile(seq_lengths_arr, 95))
        p99_len = float(np.percentile(seq_lengths_arr, 99))

        warnings = []
        recommendations = []

        if oom_risk > 0:
            pct = (oom_risk / len(samples)) * 100
            warnings.append(f"⚠️ {oom_risk} samples ({pct:.1f}%) exceed max target length ({self.max_seq_len_threshold} tokens). High OOM risk on GPU!")
            recommendations.append(f"Truncate or filter samples exceeding {self.max_seq_len_threshold} tokens before launch.")

        if std_len > (avg_len * 0.75):
            warnings.append(f"⚠️ High sequence length variance (std={std_len:.1f} vs mean={avg_len:.1f}). Expect GPU idle bubbles during batching.")
            recommendations.append("Group dataset samples by length bucket (sequence length bucketing) during data loading.")

        if empty_completions > 0:
            warnings.append(f"⚠️ {empty_completions} samples have empty or missing completions.")
            recommendations.append("Filter out entries with empty completions to prevent model learning dummy tokens.")

        if duplicates > 0:
            pct = (duplicates / len(samples)) * 100
            warnings.append(f"⚠️ {duplicates} duplicate prompts detected ({pct:.1f}%).")
            recommendations.append("Deduplicate dataset prompts to avoid over-fitting and biased gradient updates.")

        return SFTReport(
            total_samples=len(samples),
            avg_seq_len=avg_len,
            std_seq_len=std_len,
            min_seq_len=min_len,
            max_seq_len=max_len,
            p95_seq_len=p95_len,
            p99_seq_len=p99_len,
            oom_risk_count=oom_risk,
            empty_completion_count=empty_completions,
            duplicate_count=duplicates,
            warnings=warnings,
            recommendations=recommendations,
        )
