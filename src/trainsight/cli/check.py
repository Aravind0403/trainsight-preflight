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
            max_seq_len_threshold=max_seq_len,
        )
        report = inspector.inspect_file(file_path)
        console.print(f"Total pairs: {report.total_pairs} | Valid pairs: {report.valid_pairs_count}")
        if report.invalid_pairs_count > 0 and strict:
            raise typer.Exit(code=1)
