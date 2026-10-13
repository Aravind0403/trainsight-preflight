from typing import List, Dict, Any, Optional
from rich.text import Text
from rich.table import Table
from transformers import PreTrainedTokenizerBase
from trainsight.utils.logging import get_console

def render_token_mask_diff(
    input_ids: List[int],
    labels: List[int],
    attention_mask: List[int],
    prompt_len: int,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = 50,
) -> Table:
    """Renders a Rich table displaying tokens with colored loss-mask status."""
    table = Table(title="Token Loss Mask Inspection", show_lines=True)
    table.add_column("Idx", justify="right", style="cyan")
    table.add_column("Token", justify="left")
    table.add_column("Input ID", justify="right")
    table.add_column("Label", justify="right")
    table.add_column("Mask", justify="center")
    table.add_column("Status", justify="left")

    for i in range(min(len(input_ids), max_tokens)):
        tok_id = input_ids[i]
        lbl = labels[i]
        m = attention_mask[i]
        tok_str = repr(tokenizer.decode([tok_id]))

        if m == 0:
            if lbl == -100:
                status = "[dim blue]PAD (Masked)[/]"
            else:
                status = "[bold red on white]PAD LEAK (Supervised!)[/]"
        elif i < prompt_len:
            if lbl == -100:
                status = "[dim]Prompt (Masked)[/]"
            else:
                status = "[bold red]PROMPT LEAK (Supervised!)[/]"
        else:
            if lbl == -100:
                status = "[yellow]Target (Ignored)[/]"
            elif lbl == tok_id:
                status = "[bold green]Target (Supervised)[/]"
            else:
                status = "[bold red]Target (ID Mismatch)[/]"

        table.add_row(str(i), tok_str, str(tok_id), str(lbl), str(m), status)

    return table
