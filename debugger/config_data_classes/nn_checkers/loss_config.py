from dataclasses import dataclass


@dataclass
class NumericIns:
    disabled: bool = False


@dataclass
class NonDec:
    disabled: bool = False
    window_size: int = 5
    decr_percentage: float = 0.05


@dataclass
class Div:
    disabled: bool = False
    incr_abs_rate_max_thresh: int = 2
    window_size: int = 5


@dataclass
class Fluct:
    disabled: bool = False
    window_size: int = 50
    smoothness_ratio_min_thresh: float = 0.5


@dataclass
class InitLoss:
    size_growth_rate: int = 2
    size_growth_iters: int = 5
    dev_ratio: float = 1.0


@dataclass
class LossConfig:
    period: int = 10
    numeric_ins: NumericIns = NumericIns()
    non_dec: NonDec = NonDec()
    div: Div = Div()
    fluct: Fluct = Fluct()
    init_loss: InitLoss = InitLoss()
