from pathlib import Path
from typing import Optional
import typer
from rich.table import Table

from trainsight.inspectors.sft_inspector import SFTInspector
from trainsight.inspectors.dpo_inspector import DPOInspector
from trainsight.utils.logging import get_console

app = typer.Typer(help="Inspect raw training datasets on disk before GPU allocation.")

@app.callback(invoke_without_command=True)
def check(
    file_path: Path = typer.Argument(..., help="Path to JSONL dataset file"),
    task: str = typer.Option("sft", "--task", "-t", help="Task type: sft or dpo"),
    tokenizer: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--tokenizer", "-m", help="Tokenizer name or path"),
    max_seq_len: int = typer.Option(2048, "--max-seq-len", "-l", help="Max sequence length"),
    strict: bool = typer.Option(True, "--strict/--no-strict", help="Fail with non-zero exit code on contract violations"),
):
    console = get_console()
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File not found: {file_path}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold cyan]TrainSight Preflight Inspection[/]")
    console.print(f"File: [bold]{file_path.name}[/] | Task: [bold]{task.upper()}[/] | Max Length: {max_seq_len}\n")

    if task.lower() == "sft":
        inspector = SFTInspector(
            tokenizer_name=tokenizer,
            max_seq_len_threshold=max_seq_len,
            strict=strict,
        )
        report = inspector.inspect_file(file_path)

        table = Table(title="SFT Preflight Summary", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="left")

        table.add_row("Total Samples", str(report.total_samples), "[green]OK[/]")
        table.add_row("Valid Samples", str(report.valid_samples_count), "[green]OK[/]")
        
        status_empty = "[green]OK[/]" if report.empty_completion_count == 0 else "[bold red]FAIL[/]"
        table.add_row("Empty Completions", str(report.empty_completion_count), status_empty)

        status_trunc = "[green]OK[/]" if report.truncated_answers_count == 0 else "[bold red]FAIL[/]"
        table.add_row("Truncated Answers", str(report.truncated_answers_count), status_trunc)

        status_mask = "[green]OK[/]" if report.loss_mask_misaligned_count == 0 else "[bold red]FAIL[/]"
        table.add_row("Misaligned Loss Masks", str(report.loss_mask_misaligned_count), status_mask)

        table.add_row("Mean Sequence Length", f"{report.avg_seq_len:.1f} tokens", "[dim]Info[/]")
        table.add_row("Max Sequence Length", f"{report.max_seq_len} tokens", "[dim]Info[/]")

        console.print(table)

        passed = (
            report.invalid_rows_count == 0
            and report.empty_completion_count == 0
            and report.truncated_answers_count == 0
            and report.loss_mask_misaligned_count == 0
        )

        if passed:
            console.print("\n[bold green]✓ Preflight Quality Gate Passed:[/] Dataset meets all loss contract requirements.\n")
        else:
            console.print("\n[bold red]✗ Preflight Quality Gate Failed:[/] Contract violations detected above.\n")
            if strict:
                raise typer.Exit(code=1)

    elif task.lower() == "dpo":
        inspector = DPOInspector(
            tokenizer_name=tokenizer,
            max_prompt_len=max_seq_len // 2,
            max_completion_len=max_seq_len // 2,
            max_total_len=max_seq_len,
            strict=strict,
        )
        report = inspector.inspect_file(file_path)

        table = Table(title="DPO Preflight Summary", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="left")

        table.add_row("Total Pairs", str(report.total_samples), "[green]OK[/]")
        table.add_row("Valid Pairs", str(report.valid_samples_count), "[green]OK[/]")

        status_empty = "[green]OK[/]" if report.empty_pair_count == 0 else "[bold red]FAIL[/]"
        table.add_row("Empty Pairs", str(report.empty_pair_count), status_empty)

        status_identical = "[green]OK[/]" if report.identical_pairs_count == 0 else "[bold red]FAIL[/]"
        table.add_row("Identical Pairs (Zero Gradient)", str(report.identical_pairs_count), status_identical)

        status_reversal = "[green]OK[/]" if report.preference_reversal_count == 0 else "[bold red]FAIL[/]"
        table.add_row("Preference Reversals", str(report.preference_reversal_count), status_reversal)

        status_mismatch = "[green]OK[/]" if report.prompt_mismatch_count == 0 else "[bold red]FAIL[/]"
        table.add_row("Prompt Mismatches", str(report.prompt_mismatch_count), status_mismatch)

        status_near = "[dim]Info[/]" if report.near_identical_count == 0 else "[yellow]WARN[/]"
        table.add_row("Near-Identical Pairs", str(report.near_identical_count), status_near)

        table.add_row("Duplicate Prompts", str(report.duplicate_count), "[dim]Info[/]")

        # Exact token lengths
        table.add_row("Mean Prompt Length", f"{report.avg_prompt_len:.1f} tokens", "[dim]Info[/]")
        table.add_row("Mean Chosen Length", f"{report.avg_chosen_len:.1f} tokens", "[dim]Info[/]")
        table.add_row("Mean Rejected Length", f"{report.avg_rejected_len:.1f} tokens", "[dim]Info[/]")

        # Length bias ratio: chosen / rejected
        is_severe_bias = report.length_bias_ratio > 1.8 or report.length_bias_ratio < 0.55
        status_bias = "[bold red]FAIL[/]" if is_severe_bias else "[green]OK[/]"
        table.add_row("Length Bias Ratio (C/R)", f"{report.length_bias_ratio:.2f}", status_bias)

        # Training budgets
        status_budget_p = "[green]OK[/]" if report.over_budget_prompts_count == 0 else "[yellow]WARN[/]"
        table.add_row(f"Over-Budget Prompts (>{inspector.max_prompt_len})", str(report.over_budget_prompts_count), status_budget_p)

        status_budget_c = "[green]OK[/]" if report.over_budget_completions_count == 0 else "[yellow]WARN[/]"
        table.add_row(f"Over-Budget Completions (>{inspector.max_completion_len})", str(report.over_budget_completions_count), status_budget_c)

        status_budget_t = "[green]OK[/]" if report.over_budget_total_count == 0 else "[yellow]WARN[/]"
        table.add_row(f"Over-Budget Total (>{inspector.max_total_len})", str(report.over_budget_total_count), status_budget_t)

        status_quadrant = (
            "[green]OK[/]" if report.phase_quadrant == "Clean Execution"
            else ("[bold red]FAIL[/]" if report.phase_quadrant == "Defects Detected" else "[yellow]WARN[/]")
        )
        table.add_row("Phase Quadrant", report.phase_quadrant, status_quadrant)

        console.print(table)

        if report.warnings:
            console.print("\n[bold red]Actionable Defects & Warnings:[/]")
            for w in report.warnings:
                console.print(f"  {w}")

        if report.advisories:
            console.print("\n[dim]Advisories:[/]")
            for a in report.advisories:
                console.print(f"  {a}")

        if report.recommendations:
            console.print("\n[bold cyan]Recommendations:[/]")
            for r in report.recommendations:
                console.print(f"  - {r}")

        passed = (
            report.invalid_rows_count == 0
            and report.empty_pair_count == 0
            and report.identical_pairs_count == 0
            and report.preference_reversal_count == 0
            and report.prompt_mismatch_count == 0
            and not is_severe_bias
            and report.phase_quadrant not in ("Invalid Dataset", "Unable to Inspect")
        )

        if passed:
            console.print("\n[bold green]✓ Preflight Quality Gate Passed:[/] DPO dataset meets preference contract requirements.\n")
        else:
            console.print("\n[bold red]✗ Preflight Quality Gate Failed:[/] Contract violations or defects detected above.\n")
            if strict:
                raise typer.Exit(code=1)

    else:
        console.print(f"[bold red]Error:[/] Unsupported task '{task}'. Supported tasks are 'sft' and 'dpo'.")
        raise typer.Exit(code=1)
