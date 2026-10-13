# TrainSight 5-Minute Onboarding Guide

Get started with TrainSight in 3 easy steps.

---

## Step 1: Install TrainSight

In a clean Python 3.10+ virtual environment:

```bash
pip install -e .
```

Verify your installation:
```bash
trainsight version
```

---

## Step 2: Run the Demo

Run the built-in interactive demo:
```bash
trainsight demo
```

You will see:
1. **Clean Baseline**: TrainSight verifies a clean 3-row dataset without false alarms.
2. **Preflight Truncation Failure**: TrainSight flags a sequence where a verbose prompt eliminated 100% of the assistant answer tokens.
3. **Trainer Collator Failure**: TrainSight audits a simulated PyTorch batch and catches prompt token loss leakage.

---

## Step 3: Run on Your Own Dataset

Inspect an SFT JSONL file:
```bash
trainsight check path/to/dataset.jsonl --tokenizer Qwen/Qwen2.5-0.5B-Instruct --max-seq-len 2048
```

Inspect a DPO JSONL file:
```bash
trainsight check path/to/dpo_pairs.jsonl --task dpo --tokenizer Qwen/Qwen2.5-0.5B-Instruct
```
