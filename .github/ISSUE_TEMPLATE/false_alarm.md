---
name: False Alarm
about: Report a valid training input or batch flagged incorrectly by TrainSight
title: "[FALSE ALARM] "
labels: false-alarm
---

**What was flagged?**:
[Describe the check that failed, e.g. TEMPLATE_MISMATCH or TRUNCATION]

**Tokenizer & Chat Template**:
- Tokenizer: [e.g. meta-llama/Meta-Llama-3-8B-Instruct]
- Template: [e.g. ChatML, Llama-3]

**Minimal Reproducible Example**:
```json
{"prompt": "...", "completion": "..."}
```

**Why is this valid?**:
[Explain why this input should pass the contract]
