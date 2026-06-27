import {NeuralModelExample} from '@site/src/components/ModelingLabClient';

# CartoBoost GraphSAGE Models

Use GraphSAGE when node attributes should shape the graph representation. The
graph is homogeneous: one node type and one edge type, with numeric features on
each node.

## When To Use

- Nodes have useful attributes at prediction time.
- Neighbor aggregation should smooth or transfer signal across connected nodes.
- You want a graph regressor or link predictor over a homogeneous graph.
- A graph-free tabular baseline is part of the comparison.

## Interactive Example

<NeuralModelExample title="GraphSAGE browser model" pipeline="graphsage" />

## Python Regressor

```python
import numpy as np
from cartoboost.graph import GraphSageStandaloneRegressor

edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
source = np.array([0, 1, 2, 3], dtype=np.uint64)
target = np.array([1, 2, 3, 0], dtype=np.uint64)
dense = np.array([[4.2, 8], [2.0, 9], [7.1, 17], [3.5, 22]], dtype=float)
y = np.array([2.1, 1.6, 2.8, 1.9])
node_features = np.array(
    [[1.0, 0.0], [0.0, 1.0], [0.6, 0.3], [0.2, 0.7]],
    dtype=np.float32,
)

model = GraphSageStandaloneRegressor(input_dim=2, hidden_dims=(8,), epochs=2)
model.fit(
    node_features=node_features,
    edges=edges,
    row_nodes=source,
    row_targets=target,
    dense=dense,
    y=y,
)

pred = model.predict(
    node_features=node_features,
    row_nodes=source,
    row_targets=target,
    dense=dense,
)
```

## Python Link Predictor

```python
from cartoboost.graph import GraphSageLinkPredictor

predictor = GraphSageLinkPredictor(input_dim=2, hidden_dims=(8,), epochs=2)
predictor.fit(node_features=node_features, edges=edges)
scores = predictor.predict_scores(
    node_features=node_features,
    pairs=[(0, 1), (0, 3), (3, 2)],
)
```

## Validation

GraphSAGE can overstate quality when node attributes are computed with
validation labels or future rows. Keep node features train-side for deployment
claims, and report whether cold nodes appear in the holdout.
