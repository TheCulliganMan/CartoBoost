import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost GeoTemporalDiffusionScenarioModel

Use `GeoTemporalDiffusionScenarioModel` for experimental graph-wide residual
scenario generation around an existing point forecast. It is not a replacement
for the point forecaster and is excluded from stable model selection.

## Public Contract

```python
from cartoboost.preview.deep import GeoTemporalDiffusionScenarioModel

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

## Browser WASM Example

<DeepModelWasmExample model="GeoTemporalDiffusionScenarioModel" />

## Validation

Report scenario mean, variance, spatial correlation, and comparison to the
point forecast. Keep capability metadata visible because this is experimental.
