# TrainSight 5-Minute Onboarding Guide

Get started with TrainSight in an isolated virtual environment.

---

## Step 1: Install from Source

> **Note**: Do not run `pip install trainsight` (that name belongs to an unrelated PyPI package). Install from source in a dedicated virtual environment:

```bash
git clone https://github.com/Aravind0403/transight.git
cd transight
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Verify the installation:
```bash
trainsight version
```

---

## Step 2: Run the Demo

Run the interactive demonstration:
```bash
trainsight demo
```

*(Note: The first run downloads tokenizer files if not already cached locally.)*

You will observe:
1. **Clean Input Verification**: Preflight validation on 3 clean examples passing without false alarms.
2. **Truncation Fault Detection**: Preflight quality gate catching an answer eliminated by token limits.
3. **Synthetic Corrupted Batch Audit**: Batch reconciler identifying unmasked prompt labels.

---

## Step 3: Run on Your Own Dataset

Inspect an SFT JSONL dataset on disk:
```bash
trainsight check path/to/dataset.jsonl --tokenizer Qwen/Qwen2.5-0.5B-Instruct --max-seq-len 2048
```

Inspect a DPO JSONL dataset:
```bash
trainsight check path/to/dpo_pairs.jsonl --task dpo --tokenizer Qwen/Qwen2.5-0.5B-Instruct
```
