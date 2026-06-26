# Graph Link Predictors

Use standalone link predictors when the question is about plausible movement or
ranking rather than a continuous target. Examples include ranking likely
dropoff zones from a pickup zone or scoring whether a route appears in a future
time block.

Available predictors:

- `Node2VecLinkPredictor`
- `GraphSageLinkPredictor`
- `HeteroGraphSageLinkPredictor`
- `HinSageLinkPredictor`

## Basic Fit

```python
from cartoboost.graph import Node2VecLinkPredictor

edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]

predictor = Node2VecLinkPredictor(dim=8, walk_length=8, walks_per_node=4, epochs=2)
predictor.fit(node_count=4, edges=edges)

candidate_pairs = [(0, 1), (0, 3)]
scores = predictor.predict_scores(candidate_pairs)
report = predictor.report(candidate_pairs, labels=[1, 0], query_ids=[0, 0], k=1)
```

`report` can include AUC, average precision, and per-query ranking metrics.

## Validation

Use temporal edge holdouts for future-route claims and grouped query metrics
for dropoff-candidate ranking. Keep source and target roles explicit unless the
study deliberately assumes an undirected graph.
