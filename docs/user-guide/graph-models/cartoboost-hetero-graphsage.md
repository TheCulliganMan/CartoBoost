import {NeuralModelExample} from '@site/src/components/ModelingLabClient';

# CartoBoost HeteroGraphSAGE Models

Use HeteroGraphSAGE when edge relation IDs matter but the graph can still be
represented with one node-feature matrix. It is the middle ground between
homogeneous GraphSAGE and schema-heavy HinSAGE.

## When To Use

- Relations such as `view`, `purchase`, `transfer`, or `belongs_to` should not
  be collapsed into one edge type.
- Node attributes still live in one numeric feature table.
- The model needs relation-aware aggregation without a strict node-type schema.

## Interactive Example

<NeuralModelExample title="HeteroGraphSAGE browser model" pipeline="hetero_graphsage" />

## Public Contract

### Regressor

```python
import numpy as np
from cartoboost.graph import HeteroGraphSageStandaloneRegressor

typed_edges = [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 1), (0, 2, 0)]
source = np.array([0, 1, 2, 3], dtype=np.uint64)
target = np.array([1, 2, 3, 0], dtype=np.uint64)
dense = np.array([[4.2, 8], [2.0, 9], [7.1, 17], [3.5, 22]], dtype=float)
y = np.array([2.1, 1.6, 2.8, 1.9])
node_features = np.array(
    [[1.0, 0.0], [0.0, 1.0], [0.6, 0.3], [0.2, 0.7]],
    dtype=np.float32,
)

model = HeteroGraphSageStandaloneRegressor(
    input_dim=2,
    relation_count=2,
    hidden_dims=(8,),
    epochs=2,
)
model.fit(
    node_features=node_features,
    edges=typed_edges,
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

### Link Predictor

```python
from cartoboost.graph import HeteroGraphSageLinkPredictor

predictor = HeteroGraphSageLinkPredictor(input_dim=2, relation_count=2, hidden_dims=(8,))
predictor.fit(node_features=node_features, edges=typed_edges)
scores = predictor.predict_scores(
    node_features=node_features,
    pairs=[(0, 1), (0, 3), (3, 2)],
)
```

## Use When

| Need | Better first choice |
| --- | --- |
| Relation ids matter with one node-feature matrix. | `HeteroGraphSageStandaloneRegressor` or `HeteroGraphSageLinkPredictor` |
| Only topology is available. | Node2Vec |
| Node attributes matter but relation ids do not. | GraphSAGE |
| Node types and relation triples must be validated. | HinSAGE |

## Compute Backend

`HeteroGraphSageConfig` and `HeteroGraphSageFeatureEncoder.from_config(...)`
accept `backend="auto"`, `"cpu"`, or an installed accelerated backend such as
`"metal"`. On Apple-platform builds with native Metal support, Metal routes
dense self/relation forward layers through the shared native backend kernel.
Relation aggregation and training backpropagation remain CPU work.

## Validation

Hold relation IDs fixed across model comparisons. If a relation type appears
only in validation, report it as a cold-relation case instead of mixing it into
the ordinary holdout score.
