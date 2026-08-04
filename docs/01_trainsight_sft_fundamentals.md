# Learning Dossier 01: trainsight SFT Fundamentals & Data Inspection Logic

> **Folder Path**: `docs/dossier/01_trainsight_sft_fundamentals.md`  
> **Session Topic**: `trainsight` CLI Architecture, SFT Inspection Logic, and GPU Pre-flight Checks  
> **Date**: July 21, 2026  

---

## 1. What is the aim of `trainsight`?

`trainsight` is a production-grade pre-training and post-training data validation CLI and Kubernetes InitContainer.

### The Core Problem It Solves
In AI infrastructure, over **80% of fine-tuning and post-training job failures** (such as loss flatlining, gradient divergence, GPU OOM crashes, and idle GPU bubble waste) stem from **data corruptions**, not bugs in training scripts or CUDA code. 

Currently, engineers launch expensive training runs ($500–$2,000+ per run) and only discover data issues 4 hours into execution. 

### `trainsight` Objective
`trainsight` acts as a **5-minute CPU pre-flight inspector**. It runs locally on your laptop or inside a Kubernetes InitContainer before GPU allocation, catching data corruptions, sequence anomalies, and VRAM explosion risks *before* wasting GPU compute.

---

## 2. How do we know the data is JSON/JSONL only?

In `trainsight/inspectors/sft_inspector.py`, we implement strict line-by-line parsing:

```python
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
```

### Format Validation Mechanics
1. **JSONL Line-by-Line Parsing**: Each row must be a self-contained, valid JSON object.
2. **Exception Interception**: `json.JSONDecodeError` catches syntax errors (missing quotes, unescaped characters, truncated strings).
3. **Validation Report**: If 0 valid samples are loaded or syntax errors exceed thresholds, `trainsight` flags a critical warning:  
   `"Dataset is empty or contains invalid JSON lines."`

---

## 3. What are we trying to achieve from this?

| Objective | Engineering Impact |
| :--- | :--- |
| **1. Compute Waste Prevention** | Detects data corruptions in **5 seconds on CPU** vs **4 hours on an A100 GPU cluster**. |
| **2. GPU Memory (OOM) Protection** | Flags sequence lengths exceeding target VRAM limits before CUDA allocation. |
| **3. Batching Efficiency** | Identifies high sequence length variance that causes GPU padding idle bubbles. |
| **4. Gradient Quality Guard** | Eliminates empty completions and duplicate prompts that cause over-fitting and gradient corruption. |
| **5. Production CI/CD Gate** | Offers a `--strict` flag (exit code 1) for Kubernetes InitContainers to block invalid jobs from starting. |

---

## 4. What parameters does the inspector check on, and what is the logic behind them?

Here is the exact breakdown of parameters checked in `sft_inspector.py`:

```
                 📊 Dataset Metrics Summary                  
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Metric                   ┃ Value       ┃ Status           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Total Samples            │ 5           │ ✅ OK            │
│ Avg Sequence Length      │ 25.8 tokens │ ℹ️ Info          │
│ Std Dev Length           │ 21.0 tokens │ ⚠️ High Variance │
│ P95 Sequence Length      │ 58.0 tokens │ ℹ️ Info          │
│ Max Sequence Length      │ 66 tokens   │ ✅ OK            │
│ OOM Risk Samples (>2048) │ 0           │ ✅ None          │
│ Duplicate Prompts        │ 1           │ ⚠️ Duplicates    │
│ Empty Completions        │ 1           │ ❌ Corrupt       │
└──────────────────────────┴─────────────┴──────────────────┘
```

### Parameter 1: Estimated Token Count (`seq_len`)
* **Logic**: `len(text) // 4` (fast character heuristic).
* **Engineering Reason**: Full PyTorch/HuggingFace tokenizers (e.g., tiktoken or transformers) introduce heavy dependencies and slower execution. For fast pre-flight profiling, $\sim 4$ characters per token provides an accurate CPU heuristic for BPE-tokenized text.

### Parameter 2: Sequence Length Variance (`avg_seq_len` & `std_seq_len`)
* **Logic**: `np.mean(seq_lengths)` and `np.std(seq_lengths)`.  
  *Threshold Flag*: `std_seq_len > (avg_seq_len * 0.75)`
* **Engineering Reason**: When standard deviation is high relative to the mean, grouping short and long sequences into the same batch forces PyTorch to pad short sequences to match the longest sequence. This wastes GPU memory bandwidth and creates idle CUDA core bubbles.

### Parameter 3: Max Length & OOM Risk (`max_seq_len` & `oom_risk_count`)
* **Logic**: `token_count > max_seq_len_threshold` (e.g. 2048).
* **Engineering Reason**: Attention matrix memory scales as $\mathcal{O}(S^2)$ or $\mathcal{O}(S)$ with FlashAttention, while KV-cache scales linearly with sequence length $S$. When a sequence explodes past the context window, VRAM requirements spike instantly, triggering PyTorch `CUDA out of memory` crashes.

### Parameter 4: Duplicate Prompt Detection (`duplicate_count`)
* **Logic**: Normalized key hash set matching `prompt.strip().lower()`.
* **Engineering Reason**: Duplicate prompts force the optimizer to apply identical loss gradient steps repeatedly. This causes model over-fitting, memorization, and degraded evaluation performance.

### Parameter 5: Empty Completion Detection (`empty_completion_count`)
* **Logic**: Check for missing/empty `completion`, `output`, or `response` strings.
* **Engineering Reason**: An empty completion forces the cross-entropy loss calculation to penalize all predictions except the immediate `<eos>` (End-Of-Sequence) token, teaching the model to output blank responses.

---

## 5. Just with `token_count` and `prompt_key`, can we fully understand SFT?

**No, but it gives us the essential 80/20 first-order hardware and dataset health check.**

* **What `token_count` & `prompt_key` solve**:
  - Hardware execution safety (Will it OOM? Will batch padding waste GPU memory?).
  - Surface-level dataset health (Are completions missing? Are there duplicate prompts?).

* **What additional checks are needed for complete SFT inspection (Upcoming Extensions)**:
  1. **Prompt-to-Completion Ratio**: Flagging instances where prompts are 4000 tokens but completions are 2 tokens.
  2. **Format/Template Conformance**: Validating special tokens (e.g. ChatML `<|im_start|>user`, `<|im_end|>`).
  3. **Label Loss Masking Check**: Verifying that loss is computed *only* on assistant completion tokens, not user prompt tokens.
  4. **Vocabulary / Token Diversity**: Measuring unique token ratios (Zipf's Law) to detect repetitive garbage data.

---

## 6. What is SFT, and why is it important to inspect it?

### What is SFT (Supervised Fine-Tuning)?
Supervised Fine-Tuning is **Stage 1 of the post-training alignment pipeline**:

$$\text{Pre-trained Base LLM} \xrightarrow[\text{(Prompt, Response) Pairs}]{\text{SFT (Cross-Entropy Loss)}} \text{SFT Model} \xrightarrow[\text{Preference Pairs}]{\text{DPO / RLHF}} \text{Aligned Assistant}$$

- A **Base Model** (e.g. Llama 3 Base) only knows how to predict the next raw internet token (e.g., completing a web page).
- **SFT** fine-tunes the base model on curated `(Instruction, Response)` examples using standard cross-entropy loss to teach it how to act as a helpful conversational assistant.

### Why SFT Dataset Inspection is Critical
SFT forms the foundation for all downstream Alignment (RLHF / DPO / GRPO). If the SFT dataset contains formatting errors, bad code syntax, cut-off completions, or duplicate prompts:
1. The model adopts these negative generation habits permanently.
2. Downstream RLHF reward models cannot score responses accurately because policy baselines are broken.
3. **"Garbage In, Garbage Out"**: Inspecting SFT data ensures the foundation model is clean before executing expensive Stage 2 (RLHF) sweeps.
