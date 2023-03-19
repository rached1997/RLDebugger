from dataclasses import dataclass


@dataclass
class QTargetConfig:
    """
    period: represents the number of calls required to run the checks
    """
    period: int = 0
