import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost GeoTemporalDiffusionScenarioModel

Use `GeoTemporalDiffusionScenarioModel` for experimental graph-wide residual
scenario generation around an existing point forecast. It is not a replacement
for the point forecaster and is excluded from stable model selection.

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

Use this experimental model for stress scenarios around an existing point
forecast on a known graph. Do not use it as the primary point forecaster.

## Browser WASM Example

<DeepModelWasmExample model="GeoTemporalDiffusionScenarioModel" />

## Validation

Report scenario mean, variance, spatial correlation, and comparison to the
point forecast. Keep capability metadata visible because this is experimental.

## Limitations

- Evidence is synthetic and does not establish real-world scenario calibration.
- Scenario quality depends on the supplied point forecast and graph.
- Treat outputs as sensitivity analysis, not guaranteed probability statements.
