# Neural Model Guides

These guides cover neural ID embedding models and neural feature-generation
workflows. Use this section when stable taxi identifiers carry repeated
residual signal: pickup zones, dropoff zones, pickup-dropoff pairs, zone-hour
buckets, H3/S2 cells, or route clusters.

Use [Forecasting Model Guides](../forecasting-models/index.md) for
`NeuralPanelForecaster`, `LaneNeuralPanelForecaster`, N-BEATS, and N-HiTS
forecasting models. Use [Graph Features](../../graph-features.md) when the
relationship network itself is the model structure.

## Pick A Guide

| Model guide | Best first use | Notes |
| --- | --- | --- |
| [Standalone Embedding Regressor](standalone-embedding.md) | Train, evaluate, save, and serve a supervised ID embedding model directly. | Best when the embedding model is the artifact under test. |
| [Embedding Feature Workflows](embedding-features.md) | Turn learned ID vectors into dense columns for another model. | Best for ablation studies and neural-augmented boosting. |

## Scientific Fit

Neural embeddings are useful for studying repeated-ID effects such as:

- zone-specific fare or duration residuals after controlling for distance and
  hour;
- recurring pickup-dropoff pair behavior not captured by scalar route features;
- high-cardinality spatial cells where one-hot features would be too wide;
- repeated market behavior under random, tail, or out-of-time splits;
- whether support-aware shrinkage changes rare-zone stability.

They are weaker evidence for cold-start generalization. If a validation split
holds out zones or routes unseen during training, the model must use fallback
vectors or fallback IDs. Report embedding results with the split protocol and
do not treat repeated-ID gains as proof that the model understands unseen
zones.

## Read Next

[Neural Embedding Models And Features](../../neural-features.md) remains the
full contract page for artifacts, fallback behavior, benchmark reporting, and
failure modes.
