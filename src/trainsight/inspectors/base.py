from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class BaseInspector(ABC):
    """Abstract base class for dataset inspectors."""
    @abstractmethod
    def inspect_file(self, file_path: Path) -> Any:
        pass
