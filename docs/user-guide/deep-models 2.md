# Generic Deep Models

`cartoboost.deep` contains native-backed models for repeated entities, ordered
pairs, graph sequences, response curves, event probabilities, residual
correction, and constrained candidate selection. The public vocabulary is
generic: entity, node, source, target, candidate, response, utility, risk, and
constraint.

Use these models when the unit being predicted has structure that ordinary
row-level boosting does not describe directly:

| Need | Model |
| --- | --- |
| Repeated ordered source-target rows | `DirectionalPairForecaster` |
| Candidate values with monotone response | `ResponseCurveModel` |
| Calibrated binary event probability | `EventOutcomeModel` |
| Correct a known baseline numeric estimate | `ServiceTimeResidualModel` |
| Node-time forecasting on directed weighted edges | `SpatioTemporalGraphForecaster` |
| Select one candidate per decision group | `ConstrainedDecisionOptimizer` |

All fitting and scoring behavior is implemented in Rust under
`cartoboost-neural` and exposed through PyO3 and wasm. Python classes handle
frames, pandas conversion, file IO, and sklearn-style ergonomics.

## Compute Backends

Deep model constructors accept `backend="auto"` by default. Valid backend names
are `auto`, `cpu`, `cuda`, `rocm`, `metal`, and `webgpu`. `auto` selects an
available accelerator when the native build advertises one and otherwise uses
CPU. Explicit accelerator requests hard-fail if that backend is not present in
the build, so benchmark runs do not silently fall back to weaker hardware.

Use `cartoboost.deep.available_deep_backends()` to inspect Python wheel support.
Wasm builds expose `availableDeepBackends()` and include `webgpu` in the backend
contract for browser runtimes.

Use `cartoboost.deep.backend_dispatch_report("metal", len=1048576)` to verify
that the local native extension can dispatch a Metal command buffer. The report
includes the selected backend, operation name, checksum, elapsed milliseconds,
and whether the operation ran on an accelerator. Treat this as backend runtime
evidence, not as a claim that every model kernel has moved to that backend.
For response, event, service residual, and DCRNN prediction, Metal-selected
artifacts use the shared backend affine scorer for dense prediction heads.

## Response Curves

`ResponseCurveModel` fits a generic relationship between context features, a
scalar candidate value, and an observed response. When `monotone` is set, the
native head enforces the candidate effect direction.

```python
from cartoboost.deep import ResponseCurveFrame, ResponseCurveModel

frame = ResponseCurveFrame.from_pandas(
    df,
    feature_cols=["region_feature", "time_feature", "entity_feature"],
    candidate_value_col="candidate_value",
    response_col="response",
    group_col="decision_id",
)

model = ResponseCurveModel(
    response_type="binary",
    monotone="decreasing",
    calibration="isotonic",
    backend="auto",
)
model.fit(frame)
curve = model.predict_curve(frame)
best = model.best_candidate(frame)
```

Saved artifacts include the model class, version, schema hash, feature weights,
candidate slope, and calibration metadata. Reloaded artifacts replay the same
predictions.

## Ordered Pairs

`DirectionalPairForecaster` preserves order. A row with `source_id="A"` and
`target_id="B"` is represented differently from `source_id="B"` and
`target_id="A"`.

```python
from cartoboost.deep import DirectionalPairForecaster, DirectionalPairFrame

frame = DirectionalPairFrame.from_pandas(
    df,
    timestamp_col="timestamp",
    source_col="source_id",
    target_col="target_id",
    target_value_col="observed_value",
    numeric_covariates=["distance", "baseline_estimate"],
)

model = DirectionalPairForecaster(
    lookback=28,
    horizon=7,
    backbone="residual_mlp",
)
model.fit(frame)
pred = model.predict(frame)
```

Use this surface when direction is part of the scientific unit. Do not
pre-collapse pairs unless the process is genuinely unordered.

## Event Probability

`EventOutcomeModel` provides a generic calibrated binary event model.

```python
from cartoboost.deep import EventOutcomeModel

model = EventOutcomeModel(calibration="temperature")
model.fit(features_train, event_train)
probability = model.predict_proba(features_holdout)
report = model.calibration_report(features_holdout, event_holdout)
```

The calibration report currently exposes Brier score from the Python wrapper;
the native artifact stores the positive rate and temperature metadata used to
replay calibrated probabilities.

## Baseline Residual Correction

`ServiceTimeResidualModel` predicts a residual around a required baseline
numeric estimate. The prediction is always `baseline_value + residual_mean`.

```python
from cartoboost.deep import ServiceTimeResidualModel

rows = [
    {
        "baseline_value": 12.0,
        "actual_value": 13.5,
        "features": [0.2, 1.0, 4.0],
    }
]

model = ServiceTimeResidualModel()
model.fit(rows)
prediction = model.predict(rows, return_interval=True)
```

This model hard-fails on malformed rows instead of replacing missing baselines
or features with defaults.

## Graph Sequences

`SpatioTemporalGraphForecaster` is a generic facade over the directed weighted
graph forecasting core. It accepts the same native `GraphTemporalFrame` contract
used by the forecasting graph guide and supports the `dcrnn` backbone today.
The facade reserves `graph_wavenet` and `temporal_graph_attention` names for
compatible future native backbones.

## Candidate Selection

`ConstrainedDecisionOptimizer` selects one row per `decision_id` from scored
candidates. It applies hard constraints first, then scores feasible candidates
by the configured objective.

```python
from cartoboost.deep import ConstrainedDecisionOptimizer

optimizer = ConstrainedDecisionOptimizer(
    objective="risk_adjusted_utility",
    constraints={
        "min_response_probability": 0.7,
        "max_risk_score": 0.15,
    },
    fallback="raise",
)
choice = optimizer.select(candidate_rows)
```

The native selector returns the chosen candidate, score, and reason code for
each decision group.
