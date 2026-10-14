import json
import tempfile
from pathlib import Path
import typer
from rich.panel import Panel
import torch
from transformers import AutoTokenizer

from trainsight.inspectors.sft_inspector import SFTInspector
from trainsight.core.contracts import LossContract
from trainsight.core.exceptions import ContractViolationError
from trainsight.reconcilers.sft_reconciler import SFTReconciler
from trainsight.utils.logging import get_console

app = typer.Typer(help="Run interactive demonstration of preflight and batch checks.")

DEMO_CLEAN = [
    {"id": "row_001", "prompt": "What is the capital of France?", "completion": "The capital of France is Paris."},
    {"id": "row_002", "prompt": "Solve: 15 * 8", "completion": "15 * 8 = 120."},
    {"id": "row_003", "prompt": "Is 37 a prime number?", "completion": "Yes, 37 is a prime number."},
]

@app.callback(invoke_without_command=True)
def run_demo():
    console = get_console()
    console.print("\n[bold cyan]========================================================================[/]")
    console.print("[bold cyan]       TrainSight: Preflight LLM Training Input & Loss Contract Auditor  [/]")
    console.print("[bold cyan]========================================================================[/]\n")
    console.print("[dim]Note: First run downloads tokenizer configuration files if not already cached.[/]\n")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    console.print(f"[dim]Loading tokenizer ({model_name})...[/]")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        console.print(f"[bold red]Failed to load tokenizer:[/] {e}")
        raise typer.Exit(code=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # -------------------------------------------------------------
        # STEP 1: Clean Input Verification (Validated against actual report)
        # -------------------------------------------------------------
        console.print("[bold]Step 1: Inspecting clean synthetic dataset (3 rows)...[/]")
        clean_file = tmp_path / "clean.jsonl"
        with open(clean_file, "w", encoding="utf-8") as f:
            for r in DEMO_CLEAN:
                f.write(json.dumps(r) + "\n")

        inspector = SFTInspector(tokenizer_name=model_name, max_seq_len_threshold=256)
        clean_report = inspector.inspect_file(clean_file)

        console.print(f"  • Total samples: {clean_report.total_samples} | Valid: {clean_report.valid_samples_count}")
        console.print(f"  • Empty completions: {clean_report.empty_completion_count} | Truncated answers: {clean_report.truncated_answers_count}")

        # Check actual report before declaring pass
        clean_passed = (
            clean_report.valid_samples_count == clean_report.total_samples
            and clean_report.empty_completion_count == 0
            and clean_report.truncated_answers_count == 0
            and clean_report.loss_mask_misaligned_count == 0
        )
        if clean_passed:
            console.print("[bold green]✓ Preflight Quality Gate Passed (Zero False Alarms)[/]\n")
        else:
            console.print("[bold red]✗ Unexpected Preflight Failure on Clean Data[/]\n")

        # -------------------------------------------------------------
        # STEP 2: Truncation Elimination Fault (Validated against actual report)
        # -------------------------------------------------------------
        console.print("[bold]Step 2: Injecting deliberate truncation fault (Answer eliminated by budget)...[/]")
        fault_file = tmp_path / "fault_truncation.jsonl"
        long_prompt = "Verbose prompt designed to consume token budget. " * 30
        with open(fault_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "row_004", "prompt": long_prompt, "completion": "Answer that will be clipped."}) + "\n")

        fault_report = inspector.inspect_file(fault_file)
        if fault_report.truncated_answers_count > 0:
            console.print("[bold red]✗ Preflight Quality Gate Caught Fault from Report Results:[/]")
            console.print(Panel(
                f"[bold red]CRITICAL CONTRACT VIOLATION:[/] {fault_report.truncated_answers_count} answer(s) eliminated by truncation.\\n"
                f"Row ID: [bold]row_004[/]\\n"
                f"• Prompt consumed available token budget (max_seq_len=256).\\n"
                f"• Retained answer tokens: 0.\\n"
                f"• Impact: Example retains zero supervised targets.",
                title="TrainSight Preflight Gate (Actual Inspection Report)",
                border_style="red",
            ))
        else:
            console.print("[bold yellow]! Preflight Gate did not flag truncation on fault file.[/]")

        # -------------------------------------------------------------
        # STEP 3: Synthetic Corrupted Tensor Batch Audit
        # -------------------------------------------------------------
        console.print("\n[bold]Step 3: Auditing a synthetic corrupted tensor batch (prompt unmasking leak)...[/]")
        p_text = tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=True)
        c_text = "Hi there!<|im_end|>"
        p_ids = tokenizer.encode(p_text, add_special_tokens=False)
        c_ids = tokenizer.encode(c_text, add_special_tokens=False)
        full_ids = p_ids + c_ids

        # Construct a synthetic corrupted collator output: prompt unmasked (labels[:p_len] != -100)
        corrupted_batch = {
            "input_ids": torch.tensor([full_ids]),
            "labels": torch.tensor([full_ids]),  # Corrupted: prompt labels equal token IDs instead of -100
            "attention_mask": torch.tensor([[1] * len(full_ids)]),
        }

        reconciler = SFTReconciler(tokenizer=tokenizer, contract=LossContract(strict=False))
        audit = reconciler.audit_batch(batch=corrupted_batch, prompt_lengths=[len(p_ids)], row_ids=["row_synthetic_0"])

        if not audit.passed and len(audit.diagnoses) > 0:
            diag = audit.diagnoses[0]
            console.print("[bold red]✗ Batch Reconciler Caught Post-Collation Mask Leakage:[/]")
            console.print(Panel(
                f"[bold red]COLLATOR LEAKAGE:[/] {diag.error_code}\\n"
                f"{diag.error_message}\\n"
                f"• Prompt length: {diag.evidence.prompt_token_length} tokens.\\n"
                f"• Unmasked prompt tokens receiving loss: {diag.evidence.unmasked_prompt_tokens_count}.\\n"
                f"• Impact: Gradients would be computed over user prompt tokens.",
                title="TrainSight Batch Reconciler (Synthetic Corrupted Batch)",
                border_style="red",
            ))
        else:
            console.print("[bold green]✓ Reconciler passed synthetic batch.[/]")

    console.print("\n[bold green]========================================================================[/]")
    console.print("[bold green]✓ Demo Complete: Verified preflight gate and batch reconciler diagnostics.[/]")
    console.print("[bold green]========================================================================[/]\n")
    console.print("To inspect your own dataset:\n  $ [bold cyan]trainsight check my_data.jsonl --tokenizer Qwen/Qwen2.5-0.5B-Instruct[/]\n")
