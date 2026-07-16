import {DeepModelWasmExample} from '@site/src/components/ModelingLabClient';

# CartoBoost GraphNeuralOperator

Use `GraphNeuralOperator` for advanced experimental field-to-field mapping on
regional or gridded panels. It consumes field values, coordinates, graph edges,
and optional exogenous fields, then returns future, residual, and uncertainty
fields.

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

Use this experimental operator for field-to-field prediction when coordinates
and graph structure define the output domain. Start with kriging or a pointwise
model for ordinary interpolation tasks.

## Browser WASM Example

<DeepModelWasmExample model="GraphNeuralOperator" />

## Validation

Report the maintained synthetic benchmark and compare against a pointwise MLP
proxy before making any field-transfer claim.

## Limitations

- Current evidence is synthetic and mechanism-oriented.
- Coordinate scaling and graph construction strongly affect learned fields.
- Real-data use needs comparison with pointwise and spatial baselines.
