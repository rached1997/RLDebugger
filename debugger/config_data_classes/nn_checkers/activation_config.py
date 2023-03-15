from dataclasses import dataclass


@dataclass
class Dead:
    disabled: bool = False
    act_min_thresh: float = 0.00001
    act_maj_percentile: int = 95
    neurons_ratio_max_thresh: float = 0.5


@dataclass
class Saturation:
    disabled: bool = False
    ro_histo_bins_count: int = 50
    ro_histo_min: float = 0.0
    ro_histo_max: float = 1.0
    ro_max_thresh: float = 0.85
    neurons_ratio_max_thresh: float = 0.5


@dataclass
class Distribution:
    disabled: bool = False
    std_acts_min_thresh: float = 0.5
    std_acts_max_thresh: float = 2.0
    f_test_alpha: float = 0.025


@dataclass
class Range:
    disabled: bool = False


@dataclass
class Output:
    patience: int = 5


@dataclass
class NumericalInstability:
    disabled: bool = False


@dataclass
class ActivationConfig:
    period: int = 10
    start: int = 10
    buff_scale: int = 10
    patience: int = 5
    Dead: Dead = Dead()
    Saturation: Saturation = Saturation()
    Distribution: Distribution = Distribution()
    Range: Range = Range()
    Output: Output = Output()
    Numerical_Instability: NumericalInstability = NumericalInstability()
