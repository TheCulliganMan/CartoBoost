# Neural Embedding Models

Pickup zones and route pairs may retain useful
signal after measured trip features have been accounted for. Neural embeddings
represent those IDs as learned vectors, with three ways to use the result.

## Choose A Guide

| Your workflow | Guide |
| --- | --- |
| Train, save, and serve the supervised ID model directly | [Neural Embedding Regressor](cartoboost-neural-embedding.md) |
| Reuse learned vectors as columns for another estimator | [NeuralEmbeddingFeatures](cartoboost-neural-embedding-features.md) |
| Fit ID embeddings and a downstream CartoBoost regressor together | [NeuralEmbeddingRegressor](cartoboost-neural-embedding-regressor.md) |

## Validation

Fit embeddings using training rows only. Compare against the same measured
features without embeddings, then separate repeated-ID results from unseen-ID
results. A gain on familiar pickup zones does not establish a gain on zones
absent from training.

See [Neural Features](../../neural-features.md) for input shapes, unseen-ID
behavior, saving, and loading.

## When Identity Is Only Part Of The Problem

Use [graph models](../graph-models/index.md) when explicit relationships between
zones matter. Use [neural panel forecasting](../forecasting-models/neural-panel.md)
for future values across aligned histories. Use [deep models](../deep-models.md)
for ordered pairs, competing candidates, response curves, or constrained choices.
