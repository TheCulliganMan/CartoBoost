import {NeuralModelExample} from '@site/src/components/ModelingLabClient';

# CartoBoost Node2Vec Graph Models

Use Node2Vec when graph topology is the signal: which nodes connect, how
directed flow moves through the graph, and which source-target pairs appear in
similar neighborhoods. Node attributes are optional; the model learns from the
edge structure itself.

## When To Use

- Flow or co-occurrence patterns matter more than node-level attributes.
- Rows are attached to a source node, or to a directed source-target pair.
- You need a graph regressor or link predictor that can be saved and loaded.
- The validation split can distinguish train-side topology from held-out
  labels or held-out edges.

## Interactive Example

<NeuralModelExample title="Node2Vec graph browser model" pipeline="node2vec" />

## Public Contract

### Regressor

```python
import numpy as np
from cartoboost.graph import Node2VecStandaloneRegressor

edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
source = np.array([0, 1, 2, 3], dtype=np.uint64)
target = np.array([1, 2, 3, 0], dtype=np.uint64)
dense = np.array([[4.2, 8], [2.0, 9], [7.1, 17], [3.5, 22]], dtype=float)
y = np.array([2.1, 1.6, 2.8, 1.9])

model = Node2VecStandaloneRegressor(dim=8, epochs=2, n_estimators=40, seed=11)
model.fit(
    node_count=4,
    edges=edges,
    row_nodes=source,
    row_targets=target,
    dense=dense,
    y=y,
)

pred = model.predict(source, row_targets=target, dense=dense)
model.save("node2vec-regressor.json")
```

### Link Predictor

```python
from cartoboost.graph import Node2VecLinkPredictor

predictor = Node2VecLinkPredictor(dim=8, walk_length=8, walks_per_node=4, epochs=2)
predictor.fit(node_count=4, edges=edges)

candidate_pairs = [(0, 1), (0, 3), (3, 2)]
scores = predictor.predict_scores(candidate_pairs)
report = predictor.report(candidate_pairs, labels=[1, 1, 0], query_ids=[0, 0, 3], k=2)
```

## Use When

| Need | Better first choice |
| --- | --- |
| Graph topology is the main signal. | `Node2VecStandaloneRegressor` or `Node2VecLinkPredictor` |
| Node attributes should drive representation learning. | GraphSAGE |
| Relation ids matter. | HeteroGraphSAGE |
| Node types and relation triples matter. | HinSAGE |

## Validation

If the browser or Python workflow builds embeddings from validation edges, call
that transductive scoring. For deployment-style evidence, build the graph from
train-side edges, then score held-out labels or candidate pairs separately.
