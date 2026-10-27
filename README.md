# TrainSight — checks for LLM training inputs and loss masks.

TrainSight inspects SFT and DPO datasets and provides utilities for auditing tokenized training batches. It helps identify empty answers, truncation, and supervision that conflicts with the declared training objective.

**Status**: Early-stage tooling. See the [tested environments](#-tested-environments--ci-coverage) for tested configurations. Automatic trainer integration is experimental.

[![CI](https://github.com/Aravind0403/trainsight-preflight/actions/workflows/ci.yml/badge.svg)](https://github.com/Aravind0403/trainsight-preflight/actions)
[![PyPI](https://img.shields.io/pypi/v/trainsight-preflight.svg)](https://pypi.org/project/trainsight-preflight/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Originally developed within [Aether Control](https://github.com/Aravind0403/aether-control-llm-infra).

---

## 📦 Installation

Install TrainSight from PyPI:

```bash
pip install trainsight-preflight
```

> **Note**: The official distribution name on PyPI is **`trainsight-preflight`**. The CLI command and Python import namespace are both **`trainsight`**.  
> *(Do not install `trainsight`, which is an unrelated third-party package).*

To install with optional trainer support (TRL):
```bash
pip install "trainsight-preflight[trl]"
```

### Installation from Source
```bash
git clone https://github.com/Aravind0403/trainsight-preflight.git
cd trainsight-preflight
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[trl]"
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
Using the included clean SFT fixture (`tests/fixtures/clean_sft.jsonl`) with a 256-token limit:

```bash
trainsight check tests/fixtures/clean_sft.jsonl --tokenizer Qwen/Qwen2.5-0.5B-Instruct --max-seq-len 256
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

#### Expected Output on Contract Violation (Truncation Fault):
```text
TrainSight Preflight Inspection
File: truncated_answer.jsonl | Task: SFT | Max Length: 256

┌───────────────────────┬────────────┬────────┐
│ Metric                │      Value │ Status │
├───────────────────────┼────────────┼────────┤
│ Total Samples         │         25 │ OK     │
│ Valid Samples         │         24 │ FAIL   │
│ Empty Completions     │          0 │ OK     │
│ Truncated Answers     │          1 │ FAIL   │
│ Misaligned Loss Masks │          0 │ OK     │
│ Mean Sequence Length  │ 119.5 tok  │ Info   │
│ Max Sequence Length   │   256 tok  │ Info   │
└───────────────────────┴────────────┴────────┘

✗ Preflight Quality Gate Failed: Contract violations detected above.
```

### Inspecting a DPO Dataset
Using the included clean DPO fixture (`tests/fixtures/clean_dpo.jsonl`):

```bash
trainsight check tests/fixtures/clean_dpo.jsonl --task dpo --tokenizer Qwen/Qwen2.5-0.5B-Instruct
```

#### Expected Output on Clean Preference Data:
```text
TrainSight Preflight Inspection
File: clean_dpo.jsonl | Task: DPO | Max Length: 2048

                    DPO Preflight Summary                     
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric                          ┃           Value ┃ Status ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Total Pairs                     │               3 │ OK     │
│ Valid Pairs                     │               3 │ OK     │
│ Empty Pairs                     │               0 │ OK     │
│ Identical Pairs (Zero Gradient) │               0 │ OK     │
│ Preference Reversals            │               0 │ OK     │
│ Prompt Mismatches               │               0 │ OK     │
│ Near-Identical Pairs            │               0 │ Info   │
│ Duplicate Prompts               │               0 │ Info   │
│ Mean Prompt Length              │     37.0 tokens │ Info   │
│ Mean Chosen Length              │     19.7 tokens │ Info   │
│ Mean Rejected Length            │     11.7 tokens │ Info   │
│ Length Bias Ratio (C/R)         │            1.69 │ OK     │
│ Over-Budget Prompts (>1024)     │               0 │ OK     │
│ Over-Budget Completions (>1024) │               0 │ OK     │
│ Over-Budget Total (>2048)       │               0 │ OK     │
│ Phase Quadrant                  │ Clean Execution │ OK     │
└─────────────────────────────────┴─────────────────┴────────┘

✓ Preflight Quality Gate Passed: DPO dataset meets preference contract requirements.
```

### Exit Behavior
- **Default (`--strict`)**: Exits with returncode `1` if any empty completions, truncated answers, identical zero-gradient pairs, preference reversals, or invalid rows are detected. Exits with `0` if all examples satisfy the contract.
- **Non-Strict (`--no-strict`)**: Logs metrics, advisories, and warnings but exits with code `0`.
- **Parsing Failures**: If the file does not exist, syntax is invalid, or the tokenizer cannot load, execution halts with code `1`.

---

## 🛠️ Python SDK Integration

### Explicit Batch Reconciliation
To reconcile model input tensors against your loss contract, the audit requires exact row-level correspondence between raw samples and batch tensors.

The example below demonstrates this by constructing a model-ready batch directly from verified preprocessed samples (mirroring [`examples/01_quickstart_sft.py`](examples/01_quickstart_sft.py)). If you audit an active PyTorch dataloader instead, ensure your collator explicitly propagates row IDs and prompt boundaries alongside the batch, or evaluate batches using a sequential sampler without shuffling:

```python
from transformers import AutoTokenizer
import torch
from trainsight import SFTReconciler, SFTPreprocessor, LossContract

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
preprocessor = SFTPreprocessor(tokenizer_name=MODEL_NAME, max_seq_len=256)
contract = LossContract(max_seq_len=256, strict=True)
reconciler = SFTReconciler(tokenizer=tokenizer, contract=contract)

# 1. Define raw sample and derive token boundaries from formatted chat template
sample = {
    "id": "row_001",
    "prompt": "What is the capital of France?",
    "completion": "The capital of France is Paris.",
}
clean_ex = preprocessor.process_sample(sample)

# 2. Construct model-ready batch directly from preprocessed samples (guarantees 1:1 row alignment)
batch = {
    "input_ids": torch.tensor([clean_ex.input_ids]),
    "labels": torch.tensor([clean_ex.labels]),
    "attention_mask": torch.tensor([clean_ex.attention_mask]),
}

# 3. Audit batch tensors against declared loss contract
report = reconciler.audit_batch(
    batch=batch,
    prompt_lengths=[clean_ex.prompt_tokens_len],
    row_ids=[sample["id"]],
    raw_samples=[sample],
)
print(f"Batch audit passed: {report.passed}")
```

### Experimental: Trainer Callback
TrainSight provides an experimental `TrainSightCallback` for step-0 verification in Hugging Face / TRL trainers. Attach the callback directly to an initialized trainer:

```python
from trainsight import TrainSightCallback

# Attach callback to an initialized trainer instance
# (For a complete runnable SFTTrainer setup, see tests/integration/test_trl_integration.py)
callback = TrainSightCallback(trainer=trainer, fail_closed=True)
trainer.add_callback(callback)

trainer.train()
```

> **Known Limitations of `TrainSightCallback`**:
> 1. *Single Sampled Batch*: Audits only the first batch (step 0), not full dataset training batches.
> 2. *Shuffling Misalignment*: If the dataloader uses random shuffling, dataset row indexing will not match batch row ordering unless sequential sampling is configured.
> 3. *Circular Fallback Heuristic*: If the trainer's dataset is inaccessible and the callback falls back to deriving boundaries from the first unmasked label index, the check becomes circular: it derives prompt boundaries from the very labels being audited and cannot independently detect if prompt tokens were unmasked.

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

## 🔍 Tested Environments & CI Coverage

To maintain transparency between local frozen benchmark results and automated CI checks, their scopes are reported separately:

### Local Benchmark Environment
The exact pinned package versions used for the reproducible fault-injection evaluation ([benchmarks/sft_comparison/report.md](benchmarks/sft_comparison/report.md)):

| Component | Pinned Version / Setting | Scope |
|---|---|---|
| **Python** | `3.12.8` | Local runtime (macOS / Linux CPU) |
| **PyTorch** | `2.10.0` | Tensor manipulation and loss contract checking |
| **Transformers** | `5.5.0` | FastTokenizer ChatML template processing |
| **TRL** | `0.24.0` | Pinned comparison with `SFTConfig(completion_only_loss=True)` |
| **Tokenizer & Template** | `Qwen/Qwen2.5-0.5B-Instruct` | ChatML template formatting |
| **Declared Objective** | Answer-only causal loss | Masked prompt (`labels=-100`), masked padding, active label at $i \ge 1$ |

### Continuous Integration (CI) Coverage
Automated testing executed via GitHub Actions on every pull request and push to `main` ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

| Dimension | CI Scope | Details |
|---|---|---|
| **Python Versions** | `3.10`, `3.11`, `3.12` | Matrix build on `ubuntu-latest` |
| **Resolved Dependencies** | `pip install -e ".[dev,trl]"` | Resolves dependencies according to packaging bounds (`transformers>=4.40.0`, `trl>=0.8.0`, `torch>=2.0.0`). Does **not** pin historical `trl==0.24.0`. |
| **Executed Test Suite** | `pytest tests/ -v` | Unit tests (`test_sft_inspector`, `test_dpo_inspector`, `test_cli_check`, `test_sft_reconciler`, `test_causal_shift`) and integration tests (`test_demo_flow`, `test_trl_integration`). |
| **Trainer Integration Test** | `tests/integration/test_trl_integration.py` | Instantiates a minimal CPU `SFTTrainer` with sequential sampling and verifies that collated batch tensors reconcile against the loss contract under the resolved CI TRL version. |

> **Dependency Bounds Disclaimer**: Permissive dependency ranges in `pyproject.toml` indicate packaging bounds, not verified testing across every intervening release. The benchmark’s recorded dependency versions are listed above.

---

## 🤝 Feedback & Issue Reporting

If you encounter issues during early testing, please submit a report using our issue templates:
- [Report an Installation Problem](.github/ISSUE_TEMPLATE/installation_problem.md)
- [Report a False Alarm](.github/ISSUE_TEMPLATE/false_alarm.md) (valid input flagged incorrectly)
- [Report a Missed Violation](.github/ISSUE_TEMPLATE/missed_violation.md) (corrupted tensor unflagged)

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.
