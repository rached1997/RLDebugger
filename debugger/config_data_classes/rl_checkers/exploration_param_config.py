from dataclasses import dataclass
from typing import Dict

@dataclass
class CheckInitializationConfig:
    disabled: bool = False

@dataclass
class CheckMonotonicityConfig:
    disabled: bool = False

@dataclass
class CheckQuickChangeConfig:
    disabled: bool = False
    strong_decrease_thresh: int = 5
    acceleration_points_ratio: float = 0.5

@dataclass
class ExplorationPramConfig:
    period: int = 1000
    starting_value: int = 1
    ending_value: int = 0
    check_initialization: CheckInitializationConfig = CheckInitializationConfig()
    check_monotonicity: CheckMonotonicityConfig = CheckMonotonicityConfig()
    check_quick_change: CheckQuickChangeConfig = CheckQuickChangeConfig()
