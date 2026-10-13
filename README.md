# TrainSight

**Preflight checks for LLM training inputs and loss contracts.**

[![CI](https://github.com/Aravind0403/transight/actions/workflows/ci.yml/badge.svg)](https://github.com/Aravind0403/transight/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

TrainSight audits tokenization, sequence truncation, and loss masks before launching GPU training runs—and reconciles them against actual in-memory trainer dataloader batches.

Developed within [Aether Control](https://github.com/Aravind0403/aether-control-llm-infra).

---

## 🎯 The Problem

In modern LLM fine-tuning (SFT and DPO), pipelines frequently suffer from silent execution divergence:
- **Truncation Starvation**: A verbose prompt consumes the sequence length budget, silently eliminating 100% of the assistant answer tokens. Training continues with `labels == -100` everywhere, producing zero gradients or NaN loss.
- **Empty Answer Training**: Datasets with missing completions are collated with trailing EOS tokens, silently teaching models to terminate immediately.
- **Collator Mask Leakage**: Collators fail to set `labels == -100` on prompt or padding tokens, silently training the model on user prompts and padding artifacts.
- **Template Drift**: Discrepancies between chat template conventions (e.g. ChatML vs. Alpaca) alter token boundaries without throwing syntax errors.

Frameworks like TRL and PyTorch execute these corrupted batches without exception. TrainSight acts as a **fail-closed preflight linter and batch auditor**.

---

## ⚡ 5-Minute Quickstart

### 1. Installation

```bash
pip install trainsight
```

Or install from source:
```bash
git clone https://github.com/Aravind0403/transight.git
cd transight
pip install -e .
```

### 2. Run the Interactive Demo

Experience TrainSight detecting deliberate truncation and collator masking faults in under 15 seconds:

```bash
trainsight demo
```

### 3. Inspect a Dataset File on Disk

```bash
trainsight check train_data.jsonl --tokenizer Qwen/Qwen2.5-0.5B-Instruct --max-seq-len 2048
```

---

## 🔬 Two-Tiered Inspection Architecture

TrainSight enforces a strict boundary between static files and compiled tensors:

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│     TIER 1: DISK PREFLIGHT      │       │  TIER 2: TRAINER RECONCILIATION │
│    trainsight check data.jsonl  │  ───► │      TrainSightCallback()       │
├─────────────────────────────────┤       ├─────────────────────────────────┤
│ • Validates raw schema & syntax │       │ • Audits actual PyTorch batches │
│ • Pinpoints empty completions   │       │ • Checks labels == -100 on user │
│ • Catches answers eliminated    │       │ • Verifies pad attention masks  │
│   by sequence length budget     │       │ • Enforces causal shift (i >= 1)│
└─────────────────────────────────┘       └─────────────────────────────────┘
```

---

## 🛠️ Python SDK & Hugging Face / TRL Integration

### Zero-Friction Trainer Callback
Add `TrainSightCallback` to your `SFTTrainer` or `Trainer` to audit step 0 before any gradient steps execute:

```python
from trl import SFTTrainer
from trainsight import TrainSightCallback, LossContract

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    callbacks=[TrainSightCallback(contract=LossContract(strict=True))]
)
trainer.train()
```

If the dataloader produces unmasked prompt tokens or padding leakage, training halts immediately with a pinpoint diagnostic report.

---

## 📊 Measured Benchmark: Native TRL vs. TrainSight

Evaluated across six failure modes on CPU without full model weights ([view benchmark details](benchmarks/sft_comparison/report.md)):

| Failure Mode | Native TRL | Custom Assertions | TrainSight |
|---|---|---|---|
| **Clean Input** | Accepted | Accepted | **Accepted** |
| **Empty Answer** | Accepted *(Silent Fault)* | Blocked | **Blocked** *(Caught Preflight)* |
| **Answer Removed by Truncation** | Accepted *(Silent Fault)* | Blocked | **Blocked** *(Caught Preflight)* |
| **Prompt Included in Loss** | Accepted *(Silent Leak)* | Blocked | **Blocked** *(Caught Reconciliation)* |
| **Template Mismatch** | Accepted *(Silent Drift)* | Blocked | **Blocked** *(Caught Reconciliation)* |
| **Padding Included in Loss** | Accepted *(Silent Leak)* | Blocked | **Blocked** *(Caught Reconciliation)* |

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.
