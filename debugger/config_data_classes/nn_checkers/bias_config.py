from dataclasses import dataclass


@dataclass
class NumericIns:
    disabled: bool = False


@dataclass
class Div:
    disabled: bool = False
    window_size: int = 5
    mav_max_thresh: float = 100000000
    inc_rate_max_thresh: int = 2


@dataclass
class BiasConfig:
    period: int = 10
    start: int = 10
    skip_run_threshold: int = 2
    targets_perp_min_thresh = 0.5
    numeric_ins: NumericIns = NumericIns()
    div: Div = Div()
