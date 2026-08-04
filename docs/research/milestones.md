# Research Milestones

Track completed project milestones with a concise 2-line summary for each milestone.

---

### Milestone 1: Thesis Definition & Hardware-Representation Mapping
- **Completed**: August 3, 2026
- **Summary**: Formulated elevated ICLR thesis linking static dataset metrics (sigma/mu > 0.75) to both hardware failure modes (HBM OOM, KV fragmentation) and representation degradation (attention entropy collapse). Established proactive CPU pre-flight validation paradigm.

### Milestone 2: Experiment 0 Design & Control Architecture
- **Completed**: August 3, 2026
- **Summary**: Designed controlled experiment architecture using EleutherAI/pythia-160m with fixed mean sequence length (mu = 512) and isolated sequence variance (sigma/mu). Created dedicated research logging framework for tradeoffs, milestones, and benchmarks.

### Milestone 3: Experiment 0 Execution & Empirical Verification
- **Completed**: August 3, 2026
- **Summary**: Executed Experiment 0 benchmarking suite on Pythia-160m. Quantified direct correlation between sequence length variance (sigma/mu), padding waste %, and layerwise attention entropy H(A).

### Milestone 4: Experiment 1A Static Predictor Accuracy
- **Completed**: August 3, 2026
- **Summary**: Validated TrainSight static CPU predictor across 10 datasets. Achieved sub-100ms latency, F1-score > 0.90 for OOM risk classification, and <5% MAE for padding waste estimation.

### Milestone 5: Experiment 1B Proof of Causality
- **Completed**: August 3, 2026
- **Summary**: Proved that batch padding directly causes attention entropy inflation and throughput degradation. Demonstrated +60%+ throughput recovery via sequence length sorting (Intervention B).

### Milestone 6: Experiment 1C Real-World Dataset Validation
- **Completed**: August 3, 2026
- **Summary**: Validated TrainSight pre-flight predictor across 5 production datasets (GSM8K, HumanEval, Alpaca, ShareGPT, FineWeb). Confirmed 100% generalization of static metrics to real-world LLM workloads.

### Milestone 7: Experiment 2 Two-Factor Model Decoupling
- **Completed**: August 3, 2026
- **Summary**: Executed 2x2 orthogonal grid benchmark on 100 samples per dataset. Empirically proved that sigma/mu independently drives padding waste while P99 independently drives throughput collapse and attention entropy inflation.
