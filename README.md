# TrainSight — checks for LLM training inputs and loss masks.

TrainSight inspects SFT and DPO datasets and provides utilities for auditing tokenized training batches. It helps identify empty answers, truncation, and supervision that conflicts with the declared training objective.

**Status**: Early-stage tooling. See the [compatibility table](#-tested-compatibility) for tested configurations. Automatic trainer integration is experimental.

[![CI](https://github.com/Aravind0403/transight/actions/workflows/ci.yml/badge.svg)](https://github.com/Aravind0403/transight/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Originally developed within [Aether Control](https://github.com/Aravind0403/aether-control-llm-infra).

---

## ⚠️ Important Installation Notice

> **Do not install from PyPI via `pip install trainsight`.**  
> The distribution name `trainsight` on PyPI is registered to an unrelated project (`modalgrasp/trainsight`). TrainSight must be installed from source in an isolated environment until an independent distribution name is published.

### Installation from Source

Use an isolated virtual environment to prevent package collisions:

```bash
git clone https://github.com/Aravind0403/transight.git
cd transight
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Verify your installation:
```bash
trainsight version
```

---

## 🎯 The Problem

A training pipeline can produce valid tensors that violate your intended objective—for example, supervising prompt tokens or removing the answer during truncation. TrainSight adds explicit checks for these conditions.

Common silent divergence points include:
- **Truncation Starvation**: When a prompt consumes the sequence length budget, the completion may be clipped partially or entirely. If all answer tokens are dropped, all labels become `-100`, leaving the example with zero supervised tokens.
- **Empty Answer Training**: If a dataset row has an empty completion, preprocessing may attach an end-of-sequence delimiter (e.g. `<|im_end|>`) and supervise only that delimiter.
- **Collator Mask Leakage**: Collators that fail to set `labels == -100` over prompt or padded positions result in loss calculation over prompt text or padding artifacts.
- **Template Drift**: Formatting mismatches (e.g. Alpaca tags in a pipeline expecting ChatML) alter token boundaries and delimiter tokens without triggering parsing errors.

Whether these faults cause training to crash, produce NaN gradients, or quietly alter the learned representations depends on the model architecture, loss reduction, and optimizer state. TrainSight verifies these contracts before and during training.

---

## ⚡ 5-Minute Interactive Demo

TrainSight includes an interactive demonstration that exercises both preflight file inspection and in-memory batch reconciliation:

```bash
trainsight demo
```

*(Note: The first run downloads tokenizer configuration files if not already cached locally.)*

The demo demonstrates:
1. **Clean Input Verification**: Audits a clean 3-row synthetic dataset and confirms zero false alarms from the actual inspection report.
2. **Preflight Truncation Detection**: Evaluates a prompt that consumes the token budget and extracts the resulting failure report.
3. **Synthetic Corrupted Batch Audit**: Constructs a synthetic tensor batch with unmasked prompt labels (`labels[:p_len] != -100`) and audits it against the loss contract.

---

## 🔬 Coverage Boundaries: Static vs. In-Memory

TrainSight maintains a clear boundary between static dataset files and in-memory tensors:

| Boundary | Method | Scope | What It Catches | What It Cannot Catch |
|---|---|---|---|---|
| **Tier 1: Static Disk Inspection** | `trainsight check <file>` | Scans entire dataset files on disk | Missing fields, empty completions, sequence truncation against token limits | Post-collation mutations, collator label leaks, runtime padding bugs |
| **Tier 2: Batch Reconciliation** | `SFTReconciler.audit_batch()` | Audits compiled in-memory batches | Prompt label leakage (`labels != -100`), padding supervision, causal shift ($i \ge 1$) | File-level statistics outside the sampled batch |

---

## 📋 Input Format & Supported Fields

TrainSight supports JSONL files with standard SFT and DPO conventions:

### SFT Record Examples
Key-pair format:
```json
{"prompt": "What is the capital of France?", "completion": "The capital of France is Paris."}
```
OpenAI message list format:
```json
{"messages": [{"role": "user", "content": "Solve: 15 * 8"}, {"role": "assistant", "content": "15 * 8 = 120."}]}
```

### DPO Record Example
```json
{"prompt": "Explain recursion.", "chosen": "Recursion occurs when a function calls itself...", "rejected": "It is when things repeat."}
```

---

## 💻 CLI Usage & Expected Outputs

### Inspecting an SFT Dataset
```bash
trainsight check data.jsonl --tokenizer Qwen/Qwen2.5-0.5B-Instruct --max-seq-len 2048
```

#### Expected Output on Clean Data:
```text
TrainSight Preflight Inspection
File: clean_sft.jsonl | Task: SFT | Max Length: 256

┌───────────────────────┬────────────┬────────┐
│ Metric                │      Value │ Status │
├───────────────────────┼────────────┼────────┤
│ Total Samples         │         25 │ OK     │
│ Valid Samples         │         25 │ OK     │
│ Empty Completions     │          0 │ OK     │
│ Truncated Answers     │          0 │ OK     │
│ Misaligned Loss Masks │          0 │ OK     │
│ Mean Sequence Length  │ 111.2 tok  │ Info   │
│ Max Sequence Length   │   246 tok  │ Info   │
└───────────────────────┴────────────┴────────┘

✓ Preflight Quality Gate Passed: Dataset meets all loss contract requirements.
```

#### Expected Output on Contract Violation:
```text
┌───────────────────────┬────────────┬────────┐
│ Metric                │      Value │ Status │
├───────────────────────┼────────────┼────────┤
│ Total Samples         │          1 │ OK     │
│ Valid Samples         │          0 │ FAIL   │
│ Empty Completions     │          0 │ OK     │
│ Truncated Answers     │          1 │ FAIL   │
│ Misaligned Loss Masks │          0 │ OK     │
└───────────────────────┴────────────┴────────┘

✗ Preflight Quality Gate Failed: Contract violations detected above.
```

### Exit Behavior
- **Default (`--strict`)**: Exits with returncode `1` if any empty completions, truncated answers, or invalid rows are detected. Exits with `0` if all examples satisfy the contract.
- **Non-Strict (`--no-strict`)**: Logs metrics and warnings but exits with code `0`.
- **Parsing Failures**: If the file does not exist, syntax is invalid, or the tokenizer cannot load, execution halts with code `1`.

---

## 🛠️ Python SDK Integration

### Explicit Batch Reconciliation (Recommended)
Extract and audit batches directly from your PyTorch or Hugging Face DataLoader before running a full training loop:

```python
from transformers import AutoTokenizer
from trainsight import SFTReconciler, LossContract

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
reconciler = SFTReconciler(
    tokenizer=tokenizer,
    contract=LossContract(
        answer_only_loss=True,
        mask_padding=True,
        require_causal_shift_supervision=True,
        strict=True,
    )
)

# Audit a batch from your training DataLoader
batch = next(iter(train_dataloader))
prompt_lengths = [len(tokenizer.encode(ex["prompt"])) for ex in dataset_batch]

report = reconciler.audit_batch(batch=batch, prompt_lengths=prompt_lengths)
print(f"Batch audit passed: {report.passed}")
```

### Experimental: Trainer Callback
TrainSight provides a `TrainSightCallback` for step-0 verification:

```python
from trainsight import TrainSightCallback

# Note: Pass trainer explicitly if your environment's callback dispatcher
# does not automatically forward dataloaders in kwargs.
callback = TrainSightCallback(trainer=trainer, fail_closed=True)
```
> **Caveats**: `TrainSightCallback` audits only the first sampled batch. It derives prompt lengths from dataset rows, which may not align if dataloaders apply random shuffling without persistent row IDs. For strict verification, use explicit batch reconciliation.

---

## 📊 Measured Benchmark: Local Fault-Injection Comparison

In a local fault-injection comparison, TrainSight and custom assertions detected five selected contract violations. The tested native processing path did not block them.

### Evaluation Qualifications
1. **Post-Collation Mutations**: The prompt unmasking and padding unmasking faults were introduced directly on collated batch tensors to test tensor audit capabilities.
2. **Assertions Coverage**: A focused set of independent custom assertions matched TrainSight's detection coverage on these specific batches. TrainSight's value is packaged preflight checks and shared contract enforcement, rather than unique tensor arithmetic.
3. **Scope**: These results evaluate specific injected faults in a single frozen configuration (`Qwen/Qwen2.5-0.5B-Instruct`, ChatML, CPU) and do not establish broad superiority over TRL.

| Case | Native TRL (0.24.0) | Custom Assertions | TrainSight (0.2.0) |
|---|---|---|---|
| **Clean Input** *(Control)* | Accepted | Accepted | **Accepted** *(No false alarms)* |
| **Empty Answer** *(Injected Fault)* | Accepted | Blocked | **Blocked** *(Caught Preflight)* |
| **Answer Removed by Truncation** *(Injected Fault)* | Accepted | Blocked | **Blocked** *(Caught Preflight)* |
| **Prompt Included in Loss** *(Post-Collation Mutation)* | Accepted | Blocked | **Blocked** *(Caught Reconciliation)* |
| **Template Mismatch** *(Injected Fault)* | Accepted | Blocked | **Blocked** *(Caught Reconciliation)* |
| **Padding Included in Loss** *(Post-Collation Mutation)* | Accepted | Blocked | **Blocked** *(Caught Reconciliation)* |

*Full evaluation runner, tensor evidence, and raw outputs are documented in [benchmarks/sft_comparison/report.md](benchmarks/sft_comparison/report.md).*

---

## 🔍 Tested Compatibility

TrainSight has been verified against the following environment:

| Dependency / Component | Verified Version | Notes |
|---|---|---|
| **Python** | `3.10`, `3.11`, `3.12` | Verified in CI matrix |
| **PyTorch** | `2.10.0` | CPU tensor validation |
| **Transformers** | `5.5.0` | FastTokenizer chat template processing |
| **TRL** | `0.24.0` | Tested with `SFTConfig(completion_only_loss=True)` |
| **Tokenizer & Template** | `Qwen/Qwen2.5-0.5B-Instruct` | ChatML template formatting |
| **Declared Objective** | Answer-only causal loss | Masked prompt (`labels=-100`), masked padding, active label at $i \ge 1$ |

> Permissive dependency ranges in `pyproject.toml` (e.g. `transformers>=4.40.0`) indicate compatibility bounds, not verified testing across every intervening release.

---

## 🤝 Feedback & Issue Reporting

If you encounter issues during early testing, please submit a report using our issue templates:
- [Report an Installation Problem](.github/ISSUE_TEMPLATE/installation_problem.md)
- [Report a False Alarm](.github/ISSUE_TEMPLATE/false_alarm.md) (valid input flagged incorrectly)
- [Report a Missed Violation](.github/ISSUE_TEMPLATE/missed_violation.md) (corrupted tensor unflagged)

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.
