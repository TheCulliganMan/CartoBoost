import {NeuralModelExample} from '@site/src/components/ModelingLabClient';

# CartoBoost NeuralEmbeddingFeatures

Use `NeuralEmbeddingFeatures` when learned ID vectors should become columns for
another estimator. This page is for feature generation and ablation studies,
not for serving a neural model directly.

## When To Use

- Stable high-cardinality IDs recur across train and prediction rows.
- You want to compare base features against base features plus ID embeddings.
- Several downstream models should use the same train-side embedding columns.
- You can report repeated-ID and cold-ID splits separately.

## Basic Fit

```python
from cartoboost.neural import NeuralEmbeddingFeatures

features = NeuralEmbeddingFeatures(dim=8, random_state=7)
features.fit(train_ids, train_target)

train_embedding_cols = features.transform(train_ids)
valid_embedding_cols = features.transform(valid_ids)
```

Append the generated columns to the dense features used by
`CartoBoostRegressor`, LightGBM, XGBoost, or another baseline. Fit the
embedding table only on training rows when evaluating a holdout.

## Interactive Example

<NeuralModelExample title="Neural embedding feature browser model" pipeline="embedding" />

## Use When

| Need | Better surface |
| --- | --- |
| Serve the supervised ID model directly. | `NeuralEmbeddingStandaloneRegressor` |
| Generate reusable embedding columns. | `NeuralEmbeddingFeatures` |
| Fit embeddings and a CartoBoost regressor together. | `NeuralEmbeddingRegressor` |

## Validation

If validation rows contain IDs observed during training, describe the result as
repeated-ID generalization. If validation rows contain unseen IDs, report the
fallback behavior explicitly.
