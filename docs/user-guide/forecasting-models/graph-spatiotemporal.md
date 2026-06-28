import {ForecastModelExample} from '@site/src/components/ModelingLabClient';

# Graph Spatiotemporal Forecasting

Use `DCRNNForecaster` when each forecast series is a node in a known directed
graph and neighboring nodes can lead, lag, or diffuse signal into one another.
This is for sensor networks, route flows, road segments, zone flows, equipment
networks, or other panels where the graph is part of the modeling claim.

Do not use a graph forecaster only because node ids exist. The edges should
represent known movement, influence, adjacency, or dependency available at the
forecast cutoff.

## Interactive Example

<ForecastModelExample title="Graph-style panel forecast sanity check" model="neural_panel" sample="spatial" />

The embedded browser example uses the same panel forecasting surface and a
multi-location demand panel. Use it as a quick shape check for panel behavior.
Evaluate graph-specific quality in Python with your own adjacency and the
same rolling-origin split used by the baselines.

## Public Contract

```python
import numpy as np
from cartoboost.forecasting import DCRNNForecaster, GraphTemporalFrame

target = np.array(
    [
        [12.0, 8.0, 5.0, 4.0],
        [14.0, 9.0, 6.0, 5.0],
        [18.0, 12.0, 7.0, 5.5],
        [21.0, 16.0, 9.0, 6.0],
        [19.0, 17.0, 11.0, 7.0],
        [16.0, 15.0, 12.0, 8.0],
        [14.0, 12.0, 10.0, 8.5],
        [13.0, 10.0, 8.0, 7.5],
    ],
    dtype=float,
)

frame = GraphTemporalFrame(
    node_ids=["sensor_a", "sensor_b", "sensor_c", "sensor_d"],
    timestamps=list(range(target.shape[0])),
    target=target,
    indptr=[0, 2, 3, 4, 4],
    indices=[1, 2, 2, 3],
    data=[0.7, 0.3, 1.0, 1.0],
    horizon=2,
    frequency="hourly",
)

model = DCRNNForecaster(
    diffusion_steps=2,
    hidden_size=8,
    epochs=160,
    learning_rate=0.03,
    backend="auto",
)
model.fit(frame)

forecast = model.predict(2)
metrics = model.backtest(frame=frame, train_size=6)
model.save("graph-forecast.json")
```

`forecast` is a numeric array with shape `[horizon, node]`. `backtest` returns
horizon-level MAE, RMSE, and WAPE for the supplied cutoff.

`backend="auto"` is the default. On Apple-platform wheels built with native
Metal support, `backend="metal"` routes the DCRNN decoder head through the
shared Metal affine kernel. Diffusion state updates, graph validation, and
training remain deterministic Rust code. If the requested accelerator is
unavailable, construction fails with the available backend list.

## Inputs

| Input | Meaning |
| --- | --- |
| `node_ids` | Stable ids for the graph nodes. |
| `timestamps` | Regular time steps for the panel. |
| `target` | Matrix shaped `[time, node]`. |
| `indptr`, `indices`, `data` | Directed CSR adjacency for the graph. |
| `horizon` | Maintained forecast horizon for the frame. |
| `frequency` | Frequency label such as `"hourly"` or `"daily"`. |
| `covariates` | Optional node-time features shaped `[time, node, feature]`. |

Prediction before `fit`, invalid CSR arrays, non-finite targets, missing graph
edges, or incompatible shapes should be treated as data errors.

## When To Use

- The graph topology is stable and known before the forecast cutoff.
- Neighboring nodes plausibly move before or after the target node.
- You can compare against panel-only baselines on the same rolling-origin split.
- You need horizon-by-node diagnostics, not only one aggregate score.

## Use When

| Need | Better first choice |
| --- | --- |
| Transparent last-value or seasonal baseline. | `NaiveForecaster` or `SeasonalNaiveForecaster` |
| Shared lag and calendar features across many panels. | `CartoBoostLagForecaster` |
| Direct neural panel forecasts without explicit adjacency. | `NeuralPanelForecaster` |
| Directed graph diffusion across panel nodes. | `DCRNNForecaster` |

## Validation

Use rolling-origin validation. Keep the graph fixed to information available at
the cutoff and compare against seasonal naive, `CartoBoostLagForecaster`, and
`NeuralPanelForecaster` when the panel is large enough.

Report:

| Metric | Why it matters |
| --- | --- |
| MAE, RMSE, WAPE by horizon | Shows whether graph signal helps near and far horizons. |
| Error by node | Finds nodes where graph diffusion helps or hurts. |
| Error by graph distance | Checks whether upstream/downstream structure explains residuals. |
| Baseline table | Prevents a graph model from replacing a simpler panel model without evidence. |

## Limitations

- `DCRNNForecaster` is not a replacement for validating the graph itself.
- If the graph changes over time, keep only edges known at the cutoff.
- Do not fill missing adjacency with an empty graph or silently fall back to a
  panel-only model.
- `STAEformerForecaster` is a reserved transformer-style interface marker; use
  `DCRNNForecaster` for graph forecasting quality claims.
