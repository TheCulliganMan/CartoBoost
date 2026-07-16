import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost SpatioTemporalGraphForecaster

Use `SpatioTemporalGraphForecaster` when node-time targets live on directed
weighted edges and the graph is part of the forecast hypothesis. It is the
generic deep-model facade for graph sequence forecasting.

## Python Example

```python
import numpy as np
from cartoboost.deep import SpatioTemporalGraphForecaster
from cartoboost.forecasting import GraphTemporalFrame

target = np.array(
    [
        [12.0, 8.0, 5.0],
        [14.0, 9.0, 6.0],
        [18.0, 12.0, 7.0],
        [21.0, 16.0, 9.0],
        [19.0, 17.0, 11.0],
        [16.0, 15.0, 12.0],
    ],
    dtype=float,
)

frame = GraphTemporalFrame(
    node_ids=["node_a", "node_b", "node_c"],
    timestamps=list(range(target.shape[0])),
    target=target,
    indptr=[0, 2, 3, 3],
    indices=[1, 2, 2],
    data=[0.7, 0.3, 1.0],
    horizon=2,
    frequency="hourly",
)

model = SpatioTemporalGraphForecaster(
    backbone="dcrnn",
    diffusion_steps=2,
    hidden_size=8,
)
model.fit(frame)
forecast = model.predict(2)
```

## Browser WASM Example

<DeepModelWasmExample model="SpatioTemporalGraphForecaster" />

## When To Use

- The target is a regular node-time matrix.
- Directed weighted edges are known before the forecast cutoff.
- Neighboring nodes plausibly influence each other over time.
- You need graph-aware forecasts rather than graph embeddings for rows.

## Use When

| Need | Better first choice |
| --- | --- |
| Node-time forecasting on directed weighted edges. | `SpatioTemporalGraphForecaster` |
| Forecasting with the explicit DCRNN API. | `DCRNNForecaster` |
| Row-level graph regression or link scoring. | Graph model guides |
| Panel forecasting without adjacency. | `NeuralPanelForecaster` or `CartoBoostLagForecaster` |

## Validation

Use rolling-origin validation and compare against seasonal naive,
`CartoBoostLagForecaster`, and a panel neural model when available. Report
errors by horizon and node, and keep the graph restricted to information known
at the cutoff.

## Limitations

- Graph construction and edge weights can dominate model behavior.
- Missing nodes or changing topology need explicit handling.
- Graph models add compute cost and should beat graph-free panel baselines on external origins.
