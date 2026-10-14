from typing import Optional, Any
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from trainsight.core.contracts import LossContract
from trainsight.core.exceptions import ContractViolationError
from trainsight.reconcilers.sft_reconciler import SFTReconciler
from trainsight.utils.logging import get_console

class TrainSightCallback(TrainerCallback):
    """
    [EXPERIMENTAL] TrainerCallback for step-0 dry-run batch verification.
    
    Known limitations:
    - Inspects only a single sampled batch at step 0, not full-dataset tensor coverage.
    - If dataloader uses random shuffling, dataset index mapping may not correspond 1:1
      unless sequential sampling or row IDs are preserved in collation.
    - In standard Transformers callback dispatchers, the `trainer` reference is not
      automatically supplied in kwargs; pass `trainer` directly if automatic dataloader
      extraction fails.
    """

    def __init__(
        self,
        trainer: Optional[Any] = None,
        contract: Optional[LossContract] = None,
        fail_closed: bool = True,
    ):
        self.trainer = trainer
        self.contract = contract or LossContract()
        self.fail_closed = fail_closed
        self.has_run = False
        self.console = get_console()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.has_run or state.global_step > 0:
            return

        trainer = self.trainer or kwargs.get("trainer")
        train_dataloader = kwargs.get("train_dataloader")
        processing_class = kwargs.get("processing_class") or kwargs.get("tokenizer")

        dl = None
        tokenizer = None

        if trainer is not None:
            try:
                dl = trainer.get_train_dataloader()
                tokenizer = trainer.processing_class
            except Exception as e:
                self.console.print(f"[bold yellow]! TrainSight Warning:[/] Failed to get dataloader from trainer: {e}")

        if dl is None and train_dataloader is not None:
            dl = train_dataloader
            tokenizer = processing_class

        if dl is None or tokenizer is None:
            self.console.print(
                "[bold yellow]! TrainSight Warning (Experimental Callback):[/] "
                "Could not access dataloader or tokenizer from callback kwargs. "
                "Pass trainer explicitly: TrainSightCallback(trainer=trainer), "
                "or audit batches directly with SFTReconciler.audit_batch()."
            )
            return

        self.has_run = True
        self.console.print("[bold cyan]TrainSight (Experimental):[/] Auditing step 0 dataloader batch against loss contract...")

        try:
            batch = next(iter(dl))

            # Prompt length extraction from dataset if accessible
            prompt_lengths = []
            dataset = getattr(trainer, "train_dataset", None) if trainer else None

            if dataset is not None:
                for i in range(min(len(dataset), batch["input_ids"].shape[0])):
                    ex = dataset[i]
                    p_text = ex.get("prompt", "")
                    if isinstance(p_text, list):  # conversational messages
                        p_ids = tokenizer.apply_chat_template(p_text, tokenize=True, add_generation_prompt=True)
                        p_len = len(p_ids[0] if isinstance(p_ids[0], list) else p_ids)
                    else:
                        p_len = len(tokenizer.encode(p_text, add_special_tokens=False))
                    prompt_lengths.append(p_len)
            else:
                # Fallback: estimate prompt length from first unmasked label index
                labels = batch["labels"]
                for b in range(labels.shape[0]):
                    active = (labels[b] != -100).nonzero(as_tuple=True)[0]
                    p_len = active[0].item() if len(active) > 0 else labels.shape[1]
                    prompt_lengths.append(p_len)

            reconciler = SFTReconciler(tokenizer=tokenizer, contract=self.contract)
            report = reconciler.audit_batch(batch=batch, prompt_lengths=prompt_lengths)

            if report.passed:
                self.console.print("[bold green]✓ TrainSight:[/] Step 0 dataloader batch verified against loss contract.")
            else:
                diag = report.diagnoses[0]
                self.console.print(f"[bold red]✗ TrainSight Contract Violation:[/] [{diag.error_code}] (Row: {diag.row_id}): {diag.error_message}")
                if self.fail_closed:
                    control.should_training_stop = True
                    raise ContractViolationError(diag.error_code, diag.error_message, row_id=diag.row_id)

        except ContractViolationError:
            if self.fail_closed:
                control.should_training_stop = True
                raise
        except Exception as e:
            self.console.print(f"[bold red]✗ TrainSight Audit Failed:[/] {type(e).__name__}: {str(e)}")
            if self.fail_closed:
                control.should_training_stop = True
                raise ContractViolationError("AUDIT_EXECUTION_FAILURE", str(e))
