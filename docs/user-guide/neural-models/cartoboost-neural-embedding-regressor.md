import {NeuralModelExample} from '@site/src/components/ModelingLabClient';

# CartoBoost NeuralEmbeddingRegressor

Use `NeuralEmbeddingRegressor` when you want one wrapper that learns ID
embeddings, appends them to the dense row features, and fits a CartoBoost
regressor.

## When To Use

- Stable IDs carry repeated residual signal.
- You want a single object for the embedding fit and final regressor fit.
- You need a direct comparison against the same model without embedding
  columns.
- You can validate repeated-ID and cold-ID behavior separately.

## Interactive Example

<NeuralModelExample title="Neural embedding regressor browser model" pipeline="embedding" />

## Public Contract

```python
from cartoboost.neural import NeuralEmbeddingRegressor

model = NeuralEmbeddingRegressor(
    dim=8,
    n_estimators=200,
    random_state=7,
)
model.fit(X_train, y_train, ids=train_ids)
pred = model.predict(X_valid, ids=valid_ids)
```

## Inputs

| Input | Meaning |
| --- | --- |
| `X` | Dense row features for the final regressor. |
| `y` | Numeric target. |
| `ids` | Stable high-cardinality IDs used to learn embedding vectors. |

## Use When

| Need | Better surface |
| --- | --- |
| Serve the supervised ID model directly. | `NeuralEmbeddingStandaloneRegressor` |
| Generate reusable embedding columns. | `NeuralEmbeddingFeatures` |
| Fit embeddings and a CartoBoost regressor together. | `NeuralEmbeddingRegressor` |

## Validation

Compare against a non-neural `CartoBoostRegressor` on the same dense features.
If the gain disappears under cold-ID validation, describe the model as
capturing repeated-ID signal rather than unseen-entity structure.
