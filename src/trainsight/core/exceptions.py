class TrainSightError(Exception):
    """Base exception for all TrainSight errors."""
    pass

class ContractViolationError(TrainSightError):
    """Raised when actual training tensors violate the intended loss contract."""
    def __init__(self, check_name: str, message: str, row_id: str = "unknown"):
        self.check_name = check_name
        self.row_id = row_id
        super().__init__(f"[{check_name}] (Row: {row_id}): {message}")

class PreflightQualityGateError(TrainSightError):
    """Raised when raw dataset inspection fails a pre-flight quality gate."""
    pass
