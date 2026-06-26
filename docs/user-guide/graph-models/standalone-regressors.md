# Standalone Graph Regressors

Use standalone graph regressors when the graph representation and row scorer
should be trained, evaluated, saved, and served as one model artifact.

Available regressors:

- `Node2VecStandaloneRegressor`
- `GraphSageStandaloneRegressor`
- `HeteroGraphSageStandaloneRegressor`
- `HinSageStandaloneRegressor`

Each supports `fit`, `predict`, `score`, `save`, and `load`.

## Directed Pair Regression

Use this pattern for origin-destination outcomes such as log duration or log
fare when each row has a source zone and target zone.

```python
import numpy as np
from cartoboost.graph import Node2VecStandaloneRegressor

edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
pickup = np.array([0, 1, 2, 3], dtype=np.uint64)
dropoff = np.array([1, 2, 3, 0], dtype=np.uint64)
distance_hour = np.array([[4.2, 8], [2.0, 9], [7.1, 17], [3.5, 22]], dtype=float)
log_duration = np.array([2.1, 1.6, 2.8, 1.9])

model = Node2VecStandaloneRegressor(dim=8, epochs=2, n_estimators=20, seed=11)
model.fit(
    node_count=4,
    edges=edges,
    row_nodes=pickup,
    row_targets=dropoff,
    dense=distance_hour,
    y=log_duration,
)

pred = model.predict(pickup, row_targets=dropoff, dense=distance_hour)
model.save("taxi-node2vec-regressor.json")
```

Use `GraphSageStandaloneRegressor` when zone attributes should shape the
learned representation.

```python
from cartoboost.graph import GraphSageStandaloneRegressor

zone_features = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.6, 0.3],
        [0.2, 0.7],
    ],
    dtype=np.float32,
)

model = GraphSageStandaloneRegressor(input_dim=2, hidden_dims=(4,), epochs=2)
model.fit(
    node_features=zone_features,
    edges=edges,
    row_nodes=pickup,
    row_targets=dropoff,
    y=log_duration,
)
```

## Validation

Compare against a graph-free route or zone baseline under the same split. Use
cold-source, cold-target, cold-route, or temporal splits when the claim is
about generalizing beyond repeated observed edges.
