from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy._typing._array_like import NDArray

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
    latRng: list[float]
    lngRng: list[float]
    name: str
    source: str
    ROI: dict = field(default_factory=dict)
    cm_num: int = 1


@dataclass
class pipelineConfig:
    map: mappingConfig
    cluster: clusterConfig


@dataclass
class statisticsConfig:
    means: float
    covariances: float


# class results:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)


@dataclass
class results:
    prob_map: NDArray[np.float64]
    cluster_map: NDArray[np.float64]
    pred: NDArray[np.float64]
    probs: NDArray[np.float64]
    means: NDArray[np.float64]
    covariances: NDArray[np.float64]
    pca_obj: Any = None


@dataclass
class input_obj:
    data: NDArray[np.float64]
    map_shape: NDArray[np.float64]
    indices: NDArray[np.float64]
    files: list[str]


# class input_obj:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
