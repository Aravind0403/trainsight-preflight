from rich.console import Console
from rich.logging import RichHandler
import logging

_console = None

def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console(highlight=False)
    return _console

def setup_logger(name: str = "trainsight") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(console=get_console(), rich_tracebacks=True, show_time=False)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
