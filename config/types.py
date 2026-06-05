from dataclasses import dataclass, field
from typing import Literal
ThresholdType = Literal["mahalanobis", "posterior"]
@dataclass
class clusterConfig:
    n_comp: int
    cov_type: str = "full"
    threshold: float = 0.95
    threshold_type: ThresholdType = "mahalanobis"
    isPca: bool = False
@dataclass
class mappingConfig:
    keywords: list[str]
    latRng: list[int]
    lngRng: list[int]
    ROI: dict = field(default_factory=dict)
    cm_num: int = 1


@dataclass
class pipelineConfig:
    map: mappingConfig
    cluster: clusterConfig