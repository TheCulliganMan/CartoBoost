import {NeuralModelExample} from '@site/src/components/ModelingLabClient';

# CartoBoost HinSAGE Models

Use HinSAGE when node types and relation triples are part of the modeling
contract. It is the strictest graph surface in the user guide: edges are typed,
nodes are typed, and allowed source-type/relation/target-type triples are
validated.

## When To Use

- Node types are meaningful, not just labels.
- Edge relation triples must be validated.
- Direction and source-target type constraints affect the scientific claim.
- You need typed graph regression or typed link scoring.

## Interactive Example

<NeuralModelExample title="HinSAGE browser model" pipeline="hinsage" />

## Public Contract

### Regressor

```python
import numpy as np
from cartoboost.graph import HinSageStandaloneRegressor

node_types = np.array([0, 1, 1, 0], dtype=np.uint64)
edge_type_triples = [(0, 0, 1), (1, 1, 0)]
typed_edges = [(0, 1, 0), (1, 2, 1), (2, 3, 1), (3, 0, 0), (0, 2, 0)]
source = np.array([0, 1, 2, 3], dtype=np.uint64)
target = np.array([1, 2, 3, 0], dtype=np.uint64)
dense = np.array([[4.2, 8], [2.0, 9], [7.1, 17], [3.5, 22]], dtype=float)
y = np.array([2.1, 1.6, 2.8, 1.9])
node_features = np.array(
    [[1.0, 0.0], [0.0, 1.0], [0.6, 0.3], [0.2, 0.7]],
    dtype=np.float32,
)

model = HinSageStandaloneRegressor(
    input_dim=2,
    node_type_count=2,
    edge_type_triples=edge_type_triples,
    hidden_dims=(8,),
    epochs=2,
)
model.fit(
    node_features=node_features,
    node_types=node_types,
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
from cartoboost.graph import HinSageLinkPredictor

predictor = HinSageLinkPredictor(
    input_dim=2,
    node_type_count=2,
    edge_type_triples=edge_type_triples,
    hidden_dims=(8,),
)
predictor.fit(node_features=node_features, node_types=node_types, edges=typed_edges)
scores = predictor.predict_scores(
    node_features=node_features,
    pairs=[(0, 1), (0, 3), (3, 2)],
)
```

## Use When

| Need | Better first choice |
| --- | --- |
| Node types and relation triples must be enforced. | `HinSageStandaloneRegressor` or `HinSageLinkPredictor` |
| Relation ids matter but node types do not. | HeteroGraphSAGE |
| Homogeneous node attributes are enough. | GraphSAGE |
| Only topology is available. | Node2Vec |

## Compute Backend

`HinSageConfig` and `HinSageFeatureEncoder.from_config(...)` accept
`backend="auto"`, `"cpu"`, or an installed accelerated backend such as
`"metal"`. On Apple-platform builds with native Metal support, Metal routes
the dense typed GraphSAGE forward layers through the shared native backend
kernel. Schema validation, typed neighbor sampling, and training
backpropagation remain CPU work.

## Validation

Schema failures should fail clearly. Do not coerce unknown node types or
relation triples into a default type for a benchmark. Report cold-node,
cold-type, and cold-relation cases separately when they occur.
