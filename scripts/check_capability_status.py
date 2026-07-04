from __future__ import annotations

import sys

from cartoboost.capabilities import capability_table, validate_capability_table

REQUIRED_CLASSES = {
    "CartoBoostRegressor",
    "CartoBoostClassifier",
    "CartoBoostRanker",
    "AutoGeoModel",
    "EntityEmbedding",
    "PairEmbedding",
    "SpatioTemporalAdaptiveEmbedding",
    "HistoricalAnalogRetriever",
    "MultiViewSpatialAttention",
    "DirectionalPairForecaster",
    "TemporalSSMForecaster",
    "SelectiveStateSpaceBlock",
    "InvertedTemporalTransformer",
    "PropagationDelayGraphForecaster",
    "ConditionalFlowDistributionHead",
    "ChoiceSetTransformer",
    "GeoTemporalDiffusionScenarioModel",
    "GraphNeuralOperator",
}


def main() -> int:
    rows = capability_table()
    errors = validate_capability_table()
    present = {str(row["class_name"]) for row in rows}
    for class_name in sorted(REQUIRED_CLASSES - present):
        errors.append(f"{class_name} has no capability status row")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
