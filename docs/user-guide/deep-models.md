# CartoBoost Deep Model Guides

> CartoBoost 0.3 does not ship the former NumPy representation or selective-SSM
> modules. They have no import or compatibility namespace; any future return
> requires a native binding and real-data evidence.

Use `cartoboost.preview.deep` when the modeling unit is more structured than one
ordinary row: ordered pairs, candidate response curves, event probabilities,
baseline residual correction, graph sequences, or constrained candidate
selection.

`RegimeMoEForecaster` combines six named regime experts with router entropy,
expert usage, expert predictions, combined predictions, and single-expert
comparison metrics. `ConditionalFlowDistributionHead` fits a
native residual distribution head over model hidden state plus optional horizon,
entity/pair, and graph context features; it emits deterministic joint scenario
samples, log likelihoods, marginal quantiles, tail-risk summaries, and
calibration metrics when actual residuals are supplied. Artifacts record the
model class,
architecture, artifact version, schema hash, ID maps, hash-bucket settings,
embedding dimension, seed, feature roles, training cutoff, training metrics,
save/load parity status, and backend metadata.
`GeoTemporalDiffusionScenarioModel` is an experimental scenario generator that
diffuses deterministic residual shocks over a graph around an existing point
forecast panel. It reports scenario mean, variance, spatial correlation, and
comparison to the point forecast, and its metadata explicitly excludes it from
stable model selection and primary benchmark evidence.
`GraphNeuralOperator` is an advanced experimental field-to-field layer for
spatial panels. It consumes field values, coordinates, optional graph edges,
and optional exogenous fields, then returns future, residual, and uncertainty
fields. `FourierGeoOperator` and `SpatioTemporalOperator` are aliases for the
same native-backed surface in this first cut.
`ChoiceSetTransformer` scores candidates jointly within each decision set. It
emits utility, softmax choice probability, optional nested probability,
counterfactual best candidates, and calibration metrics when chosen labels are
available. `UtilityNet`, `NestedChoiceHead`, and
`CounterfactualCandidateScorer` are aliases for that native-backed surface.
Foundation adapters such as `ChronosAdapter`, `TimesFMAdapter`,
`MoiraiAdapter`, `TimeGPTAdapter`, and `TabPFNAdapter` are optional wrappers for
external models. They can cache outputs with external version and model-hash
metadata, but they are never core dependencies and are only eligible for
automatic model-selection orchestration; choose a registered model explicitly.

These guides are for Python developers and data scientists choosing a model
surface. Start with the page that matches the unit being modeled, then validate
against a simpler baseline under the same split.

## Evidence Status

| Model surface | Architecture | Evidence label |
| --- | --- | --- |
| `DirectionalPairForecaster(architecture="pair_embedding_mlp")` | `pair_embedding_mlp` | synthetic claim evidence |
| `InvertedTemporalTransformer` | `inverted_transformer` | synthetic claim evidence |
| `PropagationDelayGraphForecaster` | `delay_aware_graph_transformer` | synthetic claim evidence |
| `ConditionalFlowDistributionHead` | `conditional_residual_sampler` | synthetic claim evidence |
| `ChoiceSetTransformer` | `choice_set_utility_softmax` | synthetic claim evidence |
| `ResponseCurveModel`, `EventOutcomeModel`, `ServiceTimeResidualModel`, `ConstrainedDecisionOptimizer` | native utility/residual heads | API contract only |
| `GeoTemporalDiffusionScenarioModel` | `conditional_residual_diffusion` | experimental only |
| `GraphNeuralOperator` | `graph_neural_operator` | experimental only |

The generated capability matrix is maintained at
`docs/reference/capability-matrix.md`, with the machine-readable artifact at
`docs/assets/capabilities/model_capabilities.json`. Docs CI should run
`PYTHONPATH=python python scripts/check_capability_status.py` so exported model
classes cannot ship without architecture, backend, parameter, native-core,
evidence, and maturity status.

## Choose A Guide

| Need | Guide |
| --- | --- |
| Mixed geo-temporal regimes with named experts | [CartoBoost RegimeMoEForecaster](deep-models/cartoboost-regime-moe-forecaster.md) |
| Wide synchronized panels with entity-token attention | [CartoBoost InvertedTemporalTransformer](deep-models/cartoboost-inverted-temporal-transformer.md) |
| Directed graph propagation with known lag priors | [CartoBoost PropagationDelayGraphForecaster](deep-models/cartoboost-propagation-delay-graph-forecaster.md) |
| Joint multi-horizon uncertainty from hidden-state context | [CartoBoost ConditionalFlowDistributionHead](deep-models/cartoboost-conditional-flow-distribution-head.md) |
| Experimental graph-wide residual scenario generation | [CartoBoost GeoTemporalDiffusionScenarioModel](deep-models/cartoboost-geotemporal-diffusion-scenario-model.md) |
| Advanced experimental spatial field-to-field mapping | [CartoBoost GraphNeuralOperator](deep-models/cartoboost-graph-neural-operator.md) |
| Candidate competition and counterfactual best selection | [CartoBoost ChoiceSetTransformer](deep-models/cartoboost-choice-set-transformer.md) |
| Optional foundation model features and baselines | `cartoboost.FoundationForecastFeatures` |
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

`backend="webgpu"` is available in native builds that include the WebGPU
feature and expose a compatible adapter. It runs the verified vector-add,
affine-head, dense-layer, and pair-scoring kernels; an explicit request fails
when the feature or adapter is unavailable.

The browser bundle exposes the same adapter through the asynchronous
`webgpuDispatchReport` export. It resolves only after a real WebGPU compute
pass and readback complete, so browser applications can verify availability
without blocking the JavaScript event loop.

`InvertedTemporalTransformer` models synchronized wide panels with entities as
tokens. It reports horizon-wise metrics, cross-entity ablation, and metadata
showing that it avoids quadratic time-token attention. The same path is exposed
through `TemporalEntityTransformer(architecture="inverted_transformer")`.

`PropagationDelayGraphForecaster` models directed graph diffusion where an
upstream node can affect a downstream node after an explicit lag. It is also
available through
`SpatioTemporalGraphForecaster(backbone="delay_aware_graph_transformer")`.
Artifacts include edge-delay sensitivity, save/load parity metadata, and a
backend contract reserving CUDA, ROCm, and MLX accelerator targets. The current
verified implementation selects CPU and raises clearly if an accelerator is
requested before native kernels are available.

`ConditionalFlowDistributionHead` reports
`architecture="conditional_residual_sampler"` because the current native math is
a conditional location/scale residual sampler, not an invertible normalizing
flow. Fit it on the hidden state emitted by a deep
forecaster and the matching residual vector; pass optional horizon embeddings,
entity or pair embeddings, and graph context when those features are part of
the upstream model state. Prediction returns samples, marginal quantiles, joint
scenario paths, log likelihood, tail-risk metrics, and calibration diagnostics
such as CRPS proxy, pinball loss, interval coverage, interval width, joint-path
calibration, and tail-event calibration when actuals are provided. Save/load
round trips preserve the fitted native JSON artifact.

`GeoTemporalDiffusionScenarioModel` generates plausible future residual
scenario panels from a point forecast and directed weighted graph edges. It is
for stress and scenario analysis, not the primary point forecast. The current
surface is native-backed, deterministic, and experimental: metadata sets
`capability_tier="experimental"`, `auto_geo_enabled="false"`, and
`primary_benchmark_evidence="false"`.

`GraphNeuralOperator` maps spatial fields to future fields with graph smoothing,
coordinate Fourier signals, temporal deltas, and optional exogenous fields. Use
it for gridded or regional field experiments such as residual field evolution
or event-intensity-to-response maps. It is marked
`capability_tier="advanced_experimental"` until real-data benchmark evidence is
available.

`ChoiceSetTransformer` reports `architecture="choice_set_utility_softmax"`.
It models candidate competition within decision groups through a native utility
softmax, not candidate-candidate attention. Candidate value, candidate features,
context features, optional entity or pair embeddings, and existing
utility/probability fields feed the utility head. The report includes
choice probabilities, nested probabilities when `nest_id` is present,
counterfactual best candidates by decision, and Brier/ECE calibration when
binary `chosen` labels are supplied.

Foundation model adapters are optional comparators and feature generators. Use
the adapter-specific extras such as `cartoboost[chronos]`,
`cartoboost[timesfm]`, `cartoboost[moirai]`, `cartoboost[timegpt]`, or
`cartoboost[tabpfn]`, or provide an explicit backend. Missing dependencies
raise a clear skip reason. Cached outputs include external version metadata,
model hash, input hash, output shape, and whether the adapter was explicitly
enabled for orchestration.

Use `cartoboost.preview.deep.available_deep_backends()` to inspect the installed wheel.
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

## Input Validation

Deep model frames require finite numeric targets, features, covariates,
baseline predictions, candidate values, coordinates, and edge weights. Missing
or infinite numeric values hard-fail at frame construction or model fitting
instead of being replaced with defaults. Impute numeric values upstream when
missingness is meaningful; keep missing identifiers as explicit string tokens
when the model should learn an unknown or fallback identity.

## Validation Defaults

| Model family | Minimum comparison |
| --- | --- |
| Ordered pair forecasting | Pair baseline or row-level `CartoBoostRegressor` on the same pair covariates. |
| Response curves | Simple candidate rule and grouped holdout response metrics. |
| Event outcomes | Dummy probability, calibration metrics, and threshold-free classification metrics. |
| Residual correction | Required baseline alone versus corrected prediction. |
| Graph sequences | Seasonal naive, `CartoBoostLagForecaster`, and graph-free panel model. |
| Decision optimization | Baseline rule, selected utility, constraint violations, and fallback rate. |
