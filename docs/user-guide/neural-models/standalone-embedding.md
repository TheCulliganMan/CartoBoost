# Standalone Embedding Regressor

Use `NeuralEmbeddingStandaloneRegressor` when the learned ID embedding model is
the artifact you want to train, score, save, and serve directly. This works
best when train and prediction populations share stable IDs such as pickup
zones, dropoff zones, pickup-dropoff pairs, zone-hour buckets, or trip
clusters.

## Basic Fit

```python
import numpy as np
from cartoboost.neural import NeuralEmbeddingStandaloneRegressor

pickup_zone = np.array([132, 161, 132, 236, 161, 236], dtype=np.uint64)
dense = np.array(
    [
        [1.0, 6.0],
        [2.5, 8.0],
        [1.2, 6.0],
        [3.1, 17.0],
        [2.7, 8.0],
        [3.3, 17.0],
    ],
    dtype=float,
)
log_fare = np.array([2.7, 3.1, 2.8, 3.4, 3.2, 3.5])

model = NeuralEmbeddingStandaloneRegressor(dim=4, n_estimators=20, random_state=7)
model.fit(pickup_zone, log_fare, dense=dense)

pred = model.predict(pickup_zone, dense=dense)
mae = model.score(pickup_zone, log_fare, dense=dense)
model.save("taxi-neural-standalone.json")
```

## Direct Contract

- `fit(ids, y, dense=None)` trains one supervised embedding model.
- `predict(ids, dense=None)` returns one prediction per row.
- `score(ids, y, dense=None)` reports mean absolute error.
- `save(path)` and `load(path)` persist the complete standalone artifact.

`ids` is a one-dimensional unsigned integer array. When `dense` is provided, it
must have the same row count as `ids`.

## Validation

Report random, temporal, and cold-ID splits separately when they support
different claims. Under cold-zone or cold-route holdouts, report fallback
behavior explicitly because unseen IDs cannot recover learned ID-specific
effects.
