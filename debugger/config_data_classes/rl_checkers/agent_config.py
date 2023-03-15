from dataclasses import dataclass


@dataclass
class TargetUpdateConfig:
    disabled: bool = False


@dataclass
class SimilarityConfig:
    disabled: bool = False


@dataclass
class WrongModelOutConfig:
    disabled: bool = False


@dataclass
class KLdivConfig:
    disabled: bool = False
    div_threshold: float = 0.1


@dataclass
class AgentConfig:
    period: int = 100
    target_update: TargetUpdateConfig = TargetUpdateConfig()
    similarity: SimilarityConfig = SimilarityConfig()
    wrong_model_out: WrongModelOutConfig = WrongModelOutConfig()
    kl_div: KLdivConfig = KLdivConfig()
