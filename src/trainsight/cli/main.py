import typer
from trainsight import __version__
from trainsight.cli.check import check
from trainsight.cli.demo import run_demo
from trainsight.utils.logging import get_console

app = typer.Typer(
    name="trainsight",
    help="Preflight checks for LLM training inputs and loss contracts.",
    add_completion=False,
)

app.command(name="check")(check)
app.command(name="demo")(run_demo)

@app.command(name="version")
def version():
    """Prints TrainSight version."""
    get_console().print(f"TrainSight version: [bold cyan]{__version__}[/]")

if __name__ == "__main__":
    app()
