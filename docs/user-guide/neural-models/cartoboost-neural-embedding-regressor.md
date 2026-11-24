---
sidebar_label: Embedding + tree regressor
---

import {NeuralModelExample} from '@site/src/components/ModelingLabClient';

# CartoBoost NeuralEmbeddingRegressor

Train ID embeddings and a boosted regressor through one object with
`NeuralEmbeddingRegressor`. Learned vectors are appended to the dense features,
so repeated pickup-zone or route identity can contribute alongside trip
distance, hour, and other measurements.

## When To Use

- Stable IDs carry repeated residual signal.
- You want a single object for the embedding fit and final regressor fit.
- You need a direct comparison against the same model without embedding
  columns.
- You can validate repeated-ID and cold-ID behavior separately.

## Interactive Example

Run the small example to inspect inputs and model behavior. It is a browser
demonstration, not a benchmark on held-out taxi trips.

<NeuralModelExample title="Neural embedding regressor browser model" pipeline="embedding" />

## Python Example

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

## Limitations

- The wrapper adds training cost and tuning choices beyond the base regressor.
- Unseen IDs use documented fallback behavior rather than a learned identity effect.
- Any comparison must give the non-neural baseline the same dense features and split.
