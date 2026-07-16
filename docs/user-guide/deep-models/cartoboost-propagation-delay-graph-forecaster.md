import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost PropagationDelayGraphForecaster

Use `PropagationDelayGraphForecaster` when directed graph edges carry known or
estimated propagation delays. The model consumes node-time panels, directed
edges, edge distances, known future covariates, and delay priors.

## Python Example

```python
from cartoboost.deep import GraphTemporalFrame, PropagationDelayGraphForecaster

frame = GraphTemporalFrame(
    node_ids=["PULocationID:161", "PULocationID:236", "PULocationID:132"],
    timestamps=[0, 1, 2, 3, 4, 5],
    target=[[42, 35, 18], [44, 36, 19], [51, 40, 24], [58, 46, 31], [55, 45, 34], [49, 43, 30]],
    indptr=[0, 2, 3, 3],
    indices=[1, 2, 2],
    data=[0.7, 0.3, 1.0],
    edge_distances=[1.2, 8.4, 2.1],
    horizon=2,
    frequency="hourly",
)

model = PropagationDelayGraphForecaster(horizon=2, edge_delay_prior=[1, 2, 1])
model.fit(frame)
forecast = model.predict(2)
```

`DelayAwareGraphTransformer` and `DynamicAdjacencyTransformer` are aliases.
`SpatioTemporalGraphForecaster(backbone="delay_aware_graph_transformer")`
routes to the same implementation.

## Browser WASM Example

<DeepModelWasmExample model="PropagationDelayGraphForecaster" />

## Use When

Use this model when directed edges have meaningful propagation delays known
before the forecast cutoff. Prefer a simpler graph model when delay priors are unavailable.

## Validation

Compare against non-graph temporal and static-adjacency graph baselines. Report
edge-delay sensitivity and keep all graph inputs cutoff-safe.

## Limitations

- Incorrect delays can underperform a static adjacency model.
- Future-derived graph weights or delays leak holdout information.
- Accelerator availability varies by build and must be reported.
