---
sidebar_label: Standalone embedding regressor
---

import {NeuralModelExample} from '@site/src/components/ModelingLabClient';

# CartoBoost Neural Embedding Regressor

Train and serve the supervised ID model itself with
`NeuralEmbeddingStandaloneRegressor`. It combines stable identifiers, such as
pickup zones or route pairs, with optional dense measurements. Use it when you
want the embedding model's prediction directly rather than embedding columns
for another estimator.

## Use When

- The embedding model itself is the artifact under test.
- Stable ids recur in both training and prediction data.
- You want a direct model, not embedding features for another estimator.

## Python Example

```python
from cartoboost.neural import NeuralEmbeddingStandaloneRegressor
import numpy as np

entity_id = np.array([132, 161, 132, 236, 161, 236], dtype=np.uint64)
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
target = np.array([2.7, 3.1, 2.8, 3.4, 3.2, 3.5])

model = NeuralEmbeddingStandaloneRegressor(dim=4, n_estimators=20, random_state=7)
model.fit(entity_id, target, dense=dense)

pred = model.predict(entity_id, dense=dense)
r2 = model.score(entity_id, target, dense=dense)
model.save("neural-standalone.json")
```

## Interactive Example

Run the small example to inspect inputs and model behavior. It is a browser
demonstration, not a benchmark on held-out taxi trips.

<NeuralModelExample title="Neural embedding browser model" pipeline="embedding" />

## Inputs

- `fit(ids, y, dense=None)` trains one supervised embedding model.
- `predict(ids, dense=None)` returns one prediction per row.
- `score(ids, y, dense=None)` reports R-squared.
- `save(path)` and `load(path)` persist the complete standalone artifact.

`ids` is a one-dimensional unsigned integer array. When `dense` is provided, it
must have the same row count as `ids`.

## Validation

Report random, temporal, and cold-ID splits separately when they support
different claims. Under cold-ID or cold-route holdouts, report fallback
behavior explicitly because unseen IDs cannot recover learned ID-specific
effects.

## Limitations

- Learned ID effects do not generalize naturally to unseen identifiers.
- Embeddings can memorize frequent entities under random splits.
- Inspect temporal and cold-ID results before treating embeddings as structural signal.
