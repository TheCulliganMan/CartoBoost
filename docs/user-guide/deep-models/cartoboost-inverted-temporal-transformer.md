---
sidebar_label: Inverted temporal transformer
---

import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# Inverted Temporal Transformer

Learn relationships across synchronized taxi-zone histories with
`InvertedTemporalTransformer`. Each entity is an attention token, letting the
model compare zones without requiring a supplied adjacency graph. Cross-entity
ablations help show whether those interactions contribute beyond each zone's
own history.

## Python Example

```python
from cartoboost.deep import EntityPanelFrame, InvertedTemporalTransformer

frame = EntityPanelFrame(
    entity_ids=["PULocationID:161", "PULocationID:236", "PULocationID:132"],
    timestamps=[0, 1, 2, 3, 4, 5],
    y=[
        [42, 35, 18],
        [44, 36, 19],
        [51, 40, 24],
        [58, 46, 31],
        [55, 45, 34],
        [49, 43, 30],
    ],
    freq="hourly",
)

model = InvertedTemporalTransformer(lookback=4, horizon=2)
model.fit(frame)
forecast = model.predict(2)
quantiles = model.predict_quantiles(2)  # Keys: 0.1, 0.5, 0.9
```

`InvertedEntityTransformer` is an alias. The same implementation is reachable
through `TemporalEntityTransformer(architecture="inverted_transformer")`.
That facade also exposes `backtest(splitter)` with a `RollingOriginSplitter`:
each fold refits on its training rows and returns held-out predictions, actual
values, RMSE, and MAE. Panels must have ordered, contiguous time rows.

Quantiles use a Gaussian residual assumption with one scale per entity,
estimated from fitted one-step delta errors. The median equals the point
forecast; these intervals do not claim horizon-specific calibration. Assess
coverage on held-out rolling-origin folds. Artifacts saved without residual
scales still support point forecasts and require refitting before quantile
prediction.

## Interactive Example

Run the small example to inspect inputs and model behavior. It is a browser
demonstration, not a benchmark on held-out taxi trips.

<DeepModelWasmExample model="InvertedTemporalTransformer" />

## Use When

| Your task | Start with |
| --- | --- |
| Cross-zone attention without supplied adjacency. | `InvertedTemporalTransformer` |
| Directed edges are known and meaningful. | [SpatioTemporalGraphForecaster](cartoboost-spatiotemporal-graph-forecaster.md) |
| Shared supervised lag and calendar features. | [CartoBoost Lag](../forecasting-models/cartoboost-lag.md) |

## Validation

Report horizon-wise error and an ablation that removes cross-entity attention.
Use this only when synchronized entities are the modeling unit.

## Limitations

- Requires aligned histories and enough repeated time windows.
- Cross-entity attention can overfit stable identity effects.
- Report seed sensitivity and cold-entity behavior.
