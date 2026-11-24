---
sidebar_label: Geotemporal diffusion
---

import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# Geotemporal Diffusion Scenarios

Explore graph-wide shocks around an existing point forecast with
`GeoTemporalDiffusionScenarioModel`. A scenario can vary demand across connected
taxi zones while retaining the supplied forecast as its center.

This is an experimental scenario generator. Its synthetic evidence supports
mechanism checks, not calibrated probabilities for real demand shocks.

## Python Example

```python
from cartoboost.deep import GeoTemporalDiffusionScenarioModel

model = GeoTemporalDiffusionScenarioModel(
    scenario_count=64,
    diffusion_steps=2,
    shock_scale=0.6,
)
scenarios = model.generate(
    point_forecast=[[42, 35, 18], [44, 36, 19]],
    edges=[{"source": 0, "target": 1, "weight": 0.7}],
)
```

`FlowScenarioGenerator` and `ConditionalResidualDiffusion` are aliases.

## Use When

| Your task | Start with |
| --- | --- |
| Sensitivity to graph-wide residual shocks. | `GeoTemporalDiffusionScenarioModel` |
| The primary directed-graph point forecast. | [SpatioTemporalGraphForecaster](cartoboost-spatiotemporal-graph-forecaster.md) |
| Intervals with explicit held-out calibration. | [Probabilistic and conformal models](../forecasting-models/probabilistic-conformal.md) |

## Interactive Example

Run the small example to inspect inputs and model behavior. It is a browser
demonstration, not a benchmark on held-out taxi trips.

<DeepModelWasmExample model="GeoTemporalDiffusionScenarioModel" />

## Validation

Report scenario mean, variance, spatial correlation, and comparison to the
point forecast. Keep capability metadata visible because this is experimental.

## Limitations

- Evidence is synthetic and does not establish real-world scenario calibration.
- Scenario quality depends on the supplied point forecast and graph.
- Treat outputs as sensitivity analysis, not guaranteed probability statements.
