from dataclasses import dataclass
from typing import Dict


@dataclass
class CheckInitializationConfig:
    """
    disabled: disable the check or not
    """

    disabled: bool = False


@dataclass
class CheckMonotonicityConfig:
    """
    disabled: disable the check or not
    """

    disabled: bool = False


@dataclass
class CheckQuickChangeConfig:
    """
    disabled: disable the check or not
    strong_decrease_thresh: the min accepted value if the acceleration to measure how fast the value is changing
    """

    disabled: bool = False
    strong_decrease_thresh: float = 0.22


@dataclass
class ExplorationPramConfig:
    """
    period: represents the number of calls required to run the checks
    starting_value: The starting value of the exploration parameter
    ending_value:  The ending value of the exploration parameter
    """

    period: int = 1000
    starting_value: int = 1
    ending_value: int = 0
    check_initialization: CheckInitializationConfig = CheckInitializationConfig()
    check_monotonicity: CheckMonotonicityConfig = CheckMonotonicityConfig()
    check_quick_change: CheckQuickChangeConfig = CheckQuickChangeConfig()
