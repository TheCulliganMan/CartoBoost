---
sidebar_label: Graph neural operator
---

import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# Graph Neural Operator

Map a spatial field to forecast, residual, and uncertainty fields with
`GraphNeuralOperator`. Inputs combine values across locations, coordinates,
weighted edges, and optional external fields—for example, pickup demand across
a connected set of taxi zones.

This is an experimental model. Use the examples to understand its input and
output shapes, and compare against pointwise and spatial baselines before
interpreting field-transfer performance.

## Python Example

```python
from cartoboost.deep import GraphNeuralOperator

operator = GraphNeuralOperator(smoothing=0.25, coordinate_scale=0.1)
prediction = operator.predict(
    field_values=[[42, 35, 18], [44, 36, 19], [51, 40, 24]],
    coordinates=[[0.0, 0.0], [0.5, 0.4], [1.0, 0.1]],
    edges=[{"source": 0, "target": 1, "weight": 0.7}],
    exogenous_fields=[[0.1, 0.2, 0.0], [0.1, 0.3, 0.1], [0.2, 0.2, 0.1]],
)
benchmark = GraphNeuralOperator.synthetic_benchmark()
```

`FourierGeoOperator` and `SpatioTemporalOperator` are aliases.

## Use When

| Your task | Start with |
| --- | --- |
| Experimental mapping of one spatial field to another. | `GraphNeuralOperator` |
| Coordinate-based interpolation with uncertainty. | [Gaussian-process geostatistics](../geostatistics-models.md) |
| A time forecast with explicit adjacency. | [SpatioTemporalGraphForecaster](cartoboost-spatiotemporal-graph-forecaster.md) |

## Interactive Example

Run the small example to inspect inputs and model behavior. It is a browser
demonstration, not a benchmark on held-out taxi trips.

<DeepModelWasmExample model="GraphNeuralOperator" />

## Validation

Report the maintained synthetic benchmark and compare against a pointwise MLP
proxy before making any field-transfer claim.

## Limitations

- Current evidence is synthetic and mechanism-oriented.
- Coordinate scaling and graph construction strongly affect learned fields.
- Real-data use needs comparison with pointwise and spatial baselines.
