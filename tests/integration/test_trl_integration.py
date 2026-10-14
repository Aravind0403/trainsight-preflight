import pytest
from pathlib import Path
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from torch.utils.data import DataLoader, SequentialSampler

from trainsight.reconcilers.sft_reconciler import SFTReconciler
from trainsight.core.contracts import LossContract
from trainsight.preprocessing.sft_preprocessor import SFTPreprocessor

def test_trl_sft_trainer_reconciliation(tmp_path):
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 1-layer tiny CPU model for integration testing
    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = 1
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    model = AutoModelForCausalLM.from_config(config)

    sample = {"id": "row_001", "prompt": "What is Paris?", "completion": "Capital of France."}
    preprocessor = SFTPreprocessor(tokenizer_name=model_name, max_seq_len=256)
    clean_ex = preprocessor.process_sample(sample)

    formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": sample["prompt"]}], tokenize=False, add_generation_prompt=True)
    ds = Dataset.from_list([{"prompt": formatted_prompt, "completion": sample["completion"] + "<|im_end|>"}])

    sft_config = SFTConfig(
        output_dir=str(tmp_path),
        max_length=256,
        per_device_train_batch_size=1,
        completion_only_loss=True,
        bf16=False,
        fp16=False,
        use_cpu=True,
        report_to="none",
        logging_strategy="no",
        save_strategy="no",
    )
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=ds, processing_class=tokenizer)
    dl = DataLoader(trainer.train_dataset, batch_size=1, sampler=SequentialSampler(trainer.train_dataset), collate_fn=trainer.data_collator)
    batch = next(iter(dl))

    reconciler = SFTReconciler(tokenizer=tokenizer, contract=LossContract(max_seq_len=256, strict=True))
    report = reconciler.audit_batch(
        batch=batch,
        prompt_lengths=[clean_ex.prompt_tokens_len],
        row_ids=[sample["id"]],
        raw_samples=[sample],
    )
    assert report.passed is True
    assert report.violations_count == 0
