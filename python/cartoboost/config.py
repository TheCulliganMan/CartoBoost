from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ChoiceStrEnum(str, Enum):
    """String-valued enums for finite configuration choices."""

    def __str__(self) -> str:
        return self.value


class SplitPolicy(ChoiceStrEnum):
    """Schema-aware split search policy for stable estimators."""

    AUTO = "auto"
    AXIS_ONLY = "axis_only"
    STRUCTURED = "structured"


@dataclass(frozen=True)
class BoosterConfig:
    """Shared, typed configuration for the stable native booster family.

    Estimators may still expose task-specific options, but this object is the
    portable configuration payload used by artifact metadata and forecast
    wrappers.  ``split_policy`` controls candidate generation without asking
    callers to enumerate internal splitter names.
    """

    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 4
    min_samples_leaf: int = 20
    min_gain: float = 1e-8
    split_policy: SplitPolicy = SplitPolicy.AUTO
    n_threads: int | None = None

    def __post_init__(self) -> None:
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")
        if self.min_gain < 0:
            raise ValueError("min_gain must be non-negative")
        if self.n_threads is not None and self.n_threads <= 0:
            raise ValueError("n_threads must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["split_policy"] = self.split_policy.value
        return payload


class Backend(ChoiceStrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    WEBGPU = "webgpu"
    MLX = "mlx"


class FallbackMode(ChoiceStrEnum):
    GLOBAL_MEAN_VECTOR = "global_mean_vector"
    RAISE = "raise"


class ExplanationAlgorithm(ChoiceStrEnum):
    AUTO = "auto"


class ExplanationDecomposition(ChoiceStrEnum):
    FEATURES = "features"
    WEIGHTS = "weights"


class Objective(ChoiceStrEnum):
    AUTO = "auto"
    LAMBDARANK = "lambdarank"
    RMSE_WAPE = "rmse_wape"
    EXPECTED_UTILITY = "expected_utility"
    MAX_SCORE = "max_score"


class Kernel(ChoiceStrEnum):
    LINEAR = "linear"
    NONE = "none"
    EXPONENTIAL = "exponential"


class Method(ChoiceStrEnum):
    CROSTON = "croston"


class Drift(ChoiceStrEnum):
    ORDINARY = "ordinary"


class LeafPredictor(ChoiceStrEnum):
    CONSTANT = "constant"
    LINEAR = "linear"


class FuzzyKernel(ChoiceStrEnum):
    LINEAR = "linear"
    TRIANGULAR = "triangular"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    BISQUARE = "bisquare"
    EPANECHNIKOV = "epanechnikov"
    TRICUBE = "tricube"


class Growth(ChoiceStrEnum):
    LINEAR = "linear"
    FLAT = "flat"
    LOGISTIC = "logistic"


class ComponentMode(ChoiceStrEnum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class SeasonalityMode(ChoiceStrEnum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class RegulatorStandardization(ChoiceStrEnum):
    AUTO = "auto"
    NONE = "none"


class TrendUncertaintyPolicy(ChoiceStrEnum):
    LAPLACE = "laplace"
    NORMAL = "normal"


class FitLoss(ChoiceStrEnum):
    SQUARED = "squared"
    HUBER = "huber"


class ValidationObjective(ChoiceStrEnum):
    MEAN_SQUARED_ERROR = "mean_squared_error"


class GraphBackbone(ChoiceStrEnum):
    DCRNN = "dcrnn"
    GRAPH_WAVENET = "graph_wavenet"
    TEMPORAL_GRAPH_ATTENTION = "temporal_graph_attention"
    DELAY_AWARE_GRAPH_TRANSFORMER = "delay_aware_graph_transformer"


class OverlayKernel(ChoiceStrEnum):
    NONE = "none"
    LINEAR = "linear"
