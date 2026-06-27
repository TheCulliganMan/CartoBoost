# CartoBoost Neural Model Guides

Use neural models when stable identifiers carry repeated residual signal.
These guides are for learned ID structure, not for graph relationships or
forecasting panels.

Use [CartoBoost Forecasting Model Guides](../forecasting-models/index.md) for
`NeuralPanelForecaster` and `LaneNeuralPanelForecaster`, and [CartoBoost Graph
Model Guides](../graph-models/index.md) when the relationship network itself is
the thing being modeled.

## Choose A Guide

| Guide | Best when |
| --- | --- |
| [CartoBoost Neural Embedding Regressor](cartoboost-neural-embedding.md) | The supervised ID embedding model is the artifact you train, score, save, and serve. |
| [CartoBoost NeuralEmbeddingFeatures](cartoboost-neural-embedding-features.md) | Learned ID vectors should become columns for another estimator or ablation. |
| [CartoBoost NeuralEmbeddingRegressor](cartoboost-neural-embedding-regressor.md) | One wrapper should learn ID embeddings and fit the downstream CartoBoost regressor. |

See [Neural Features](../../neural-features.md) for the artifact contract,
fallback behavior, and failure modes.
