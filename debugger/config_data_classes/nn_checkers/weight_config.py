from dataclasses import dataclass


@dataclass
class NumericIns:
    disabled: bool = False


@dataclass
class Neg:
    disabled: bool = False
    ratio_max_thresh: float = 0.95


@dataclass
class Dead:
    disabled: bool = False
    val_min_thresh: float = 0.00001
    ratio_max_thresh: float = 0.95


@dataclass
class Div:
    disabled: bool = False
    window_size: int = 5
    mav_max_thresh: int = 100000000
    inc_rate_max_thresh: int = 2


@dataclass
class InitialWeight:
    disabled: bool = False
    f_test_alpha: float = 0.1


@dataclass
class WeightConfig:
    start: int = 100
    period: int = 10
    skip_run_threshold: int = 10
    numeric_ins: NumericIns = NumericIns()
    neg: Neg = Neg()
    dead: Dead = Dead()
    div: Div = Div()
    initial_weight: InitialWeight = InitialWeight()
