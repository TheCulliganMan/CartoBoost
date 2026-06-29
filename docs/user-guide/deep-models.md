# CartoBoost Deep Model Guides

Use `cartoboost.deep` when the modeling unit is more structured than one
ordinary row: ordered pairs, candidate response curves, event probabilities,
baseline residual correction, graph sequences, or constrained candidate
selection.

These guides are for Python developers and data scientists choosing a model
surface. Start with the page that matches the unit being modeled, then validate
against a simpler baseline under the same split.

## Choose A Guide

| Need | Guide |
| --- | --- |
| Repeated ordered source-target rows | [CartoBoost DirectionalPairForecaster](deep-models/cartoboost-directional-pair-forecaster.md) |
| Candidate values with monotone response | [CartoBoost ResponseCurveModel](deep-models/cartoboost-response-curve-model.md) |
| Calibrated binary event probability | [CartoBoost EventOutcomeModel](deep-models/cartoboost-event-outcome-model.md) |
| Correct a known baseline numeric estimate | [CartoBoost ServiceTimeResidualModel](deep-models/cartoboost-service-time-residual-model.md) |
| Node-time forecasting on directed weighted edges | [CartoBoost SpatioTemporalGraphForecaster](deep-models/cartoboost-spatiotemporal-graph-forecaster.md) |
| Select one candidate per decision group | [CartoBoost ConstrainedDecisionOptimizer](deep-models/cartoboost-constrained-decision-optimizer.md) |

## Backend Choice

Deep model constructors default to `backend="cpu"`. `backend="auto"` is accepted
as a CPU-resolving alias for ordinary workflows. Request a specific accelerator
only when the environment has been provisioned for it and the run needs that
hardware contract.

Use `cartoboost.deep.available_deep_backends()` to inspect the installed wheel.
If a requested accelerator is unavailable, treat that as an environment error
rather than silently changing the benchmark or production contract.
On Apple-platform builds with the native Metal feature, `backend="metal"` is
available for the shared dense, affine, and graph-score kernels used by the
deep response, event, service-residual, graph, and neural forecasting surfaces.
That includes macOS, iOS, tvOS, and visionOS builds where the native backend is
compiled in. On Linux or WSL builds with ROCm support compiled in and a usable
HIP device present, `backend="rocm"` is advertised for the same verified shared
kernels. On Windows or Linux builds with the CUDA driver and NVRTC available,
`backend="cuda"` is advertised for the same verified shared kernels.

## Validation Defaults

| Model family | Minimum comparison |
| --- | --- |
| Ordered pair forecasting | Pair baseline or row-level `CartoBoostRegressor` on the same pair covariates. |
| Response curves | Simple candidate rule and grouped holdout response metrics. |
| Event outcomes | Dummy probability, calibration metrics, and threshold-free classification metrics. |
| Residual correction | Required baseline alone versus corrected prediction. |
| Graph sequences | Seasonal naive, `CartoBoostLagForecaster`, and graph-free panel model. |
| Decision optimization | Baseline rule, selected utility, constraint violations, and fallback rate. |
