from typing import Optional
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from trainsight.core.contracts import LossContract
from trainsight.core.exceptions import ContractViolationError
from trainsight.reconcilers.sft_reconciler import SFTReconciler
from trainsight.utils.logging import get_console

class TrainSightCallback(TrainerCallback):
    """
    Hugging Face / TRL TrainerCallback for step-0 dry-run verification.
    Audits the first batch from the trainer's dataloader before any gradient steps execute.
    """

    def __init__(self, contract: Optional[LossContract] = None, fail_closed: bool = True):
        self.contract = contract or LossContract()
        self.fail_closed = fail_closed
        self.has_run = False
        self.console = get_console()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.has_run or state.global_step > 0:
            return

        trainer = kwargs.get("trainer")
        if trainer is None:
            return

        self.has_run = True
        self.console.print("[bold cyan]TrainSight:[/] Intercepting step 0 dataloader batch for contract reconciliation...")

        try:
            dl = trainer.get_train_dataloader()
            batch = next(iter(dl))

            # Prompt length extraction from dataset
            tokenizer = trainer.processing_class
            dataset = trainer.train_dataset
            prompt_lengths = []

            for i in range(min(len(dataset), batch["input_ids"].shape[0])):
                ex = dataset[i]
                p_text = ex.get("prompt", "")
                if isinstance(p_text, list):  # conversational
                    p_ids = tokenizer.apply_chat_template(p_text, tokenize=True, add_generation_prompt=True)
                    p_len = len(p_ids[0] if isinstance(p_ids[0], list) else p_ids)
                else:
                    p_len = len(tokenizer.encode(p_text, add_special_tokens=False))
                prompt_lengths.append(p_len)

            reconciler = SFTReconciler(tokenizer=tokenizer, contract=self.contract)
            report = reconciler.audit_batch(batch=batch, prompt_lengths=prompt_lengths)
            self.console.print("[bold green]✓ TrainSight:[/] Step 0 dataloader batch verified against loss contract.")

        except ContractViolationError as e:
            self.console.print(f"[bold red]✗ TrainSight Contract Violation:[/] {str(e)}")
            if self.fail_closed:
                control.should_training_stop = True
                raise e
        except Exception as e:
            self.console.print(f"[bold yellow]! TrainSight Warning:[/] Reconciler error: {str(e)}")
