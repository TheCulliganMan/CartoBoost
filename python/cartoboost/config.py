from __future__ import annotations

from enum import Enum


class ChoiceStrEnum(str, Enum):
    """String-valued enums for finite configuration choices."""

    def __str__(self) -> str:
        return self.value


class Backend(ChoiceStrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
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
