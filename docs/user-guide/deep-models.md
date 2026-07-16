# Deep Models

Use `cartoboost.deep` when one row is not enough to describe the prediction
problem. These models handle ordered source-target pairs, candidate response
curves, event probabilities, residual correction, graph sequences, scenario
generation, and constrained decisions.

Start with the table below. Open the dedicated guide for a runnable Python and
browser example, required inputs, validation design, and limitations. Most of
these models are specialized or experimental, so compare them against a simpler
boosting, forecasting, graph, or statistical baseline on the same holdout.

## Maturity And Evidence

| Model surface | Architecture | Evidence label |
| --- | --- | --- |
| `DirectionalPairForecaster(architecture="pair_embedding_mlp")` | `pair_embedding_mlp` | synthetic claim evidence |
| `InvertedTemporalTransformer` | `inverted_transformer` | synthetic claim evidence |
| `PropagationDelayGraphForecaster` | `delay_aware_graph_transformer` | synthetic claim evidence |
| `ConditionalFlowDistributionHead` | `conditional_residual_sampler` | synthetic claim evidence |
| `ChoiceSetTransformer` | `choice_set_utility_softmax` | synthetic claim evidence |
| `ResponseCurveModel`, `EventOutcomeModel`, `ServiceTimeResidualModel`, `ConstrainedDecisionOptimizer` | native utility/residual heads | API behavior only |
| `GeoTemporalDiffusionScenarioModel` | `conditional_residual_diffusion` | experimental only |
| `GraphNeuralOperator` | `graph_neural_operator` | experimental only |

See the [Model Capabilities](../reference/capability-matrix.md) table for
backend, parameter, evidence, and maturity details.

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
feature and expose a compatible adapter. It implements the complete shared
operation contract: dense and affine work, pair scoring and distance, sparse
CSR forward/backward kernels, row softmax, AdamW, layer normalization, scalar
graphs, and tanh-MLP training. An explicit request fails when the feature or
adapter is unavailable.

The browser bundle exposes the same adapter through the asynchronous
`webgpuCapabilities` and operation-specific asynchronous exports. Capability
probing resolves only after a real WebGPU compute pass and readback complete,
so browser applications can verify availability without blocking the
JavaScript event loop. The browser exports cover all operations in the native
contract and return updated optimizer/training state where mutation cannot be
represented directly across the JavaScript boundary.

Browser N-BEATS and N-HiTS can use `runNeuralForecastWebgpu`. The asynchronous
route performs window training with the WebGPU tanh-MLP kernel and dispatches
every recursive hidden layer through WebGPU dense inference; its response uses
the same forecast and backend-metadata shape as `runForecast`.

`runGraphDiffusionWebgpu` accepts the standard browser graph-temporal frame,
normalizes its CSR edge weights, and keeps every configured diffusion and
horizon step on browser WebGPU. It returns the normal graph forecast response,
including optional graph-aware metrics and explicit accelerated-operation
metadata.

Browser nearest-neighbor Gaussian-process prediction accepts a backend in its
geostatistics options. `runGeostatisticsWebgpu` computes the full transformed
query-by-observation distance matrix on WebGPU, then performs only the small
per-neighborhood covariance solves on CPU. Metadata distinguishes the GPU
distance operation from the retained CPU solve.

`InvertedTemporalTransformer` models synchronized wide panels with entities as
tokens. It reports horizon-wise metrics, cross-entity ablation, and metadata
showing that it avoids quadratic time-token attention. The same path is exposed
through `TemporalEntityTransformer(architecture="inverted_transformer")`.

`PropagationDelayGraphForecaster` models directed graph diffusion where an
upstream node can affect a downstream node after an explicit lag. It is also
available through
`SpatioTemporalGraphForecaster(backbone="delay_aware_graph_transformer")`.
Artifacts include edge-delay sensitivity, save/load parity metadata, and a
shared backend contract supporting CPU, CUDA, ROCm/HIP, Metal, DirectML, and
WebGPU. Explicit unavailable devices raise clearly; `auto` resolves to a
compatible available backend during model construction and stores that concrete
selection with the fitted artifact.

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
the adapter-specific packages such as `chronos-forecasting`, `timesfm`,
`uni2ts`, `nixtla`, or `tabpfn`, or provide an explicit backend. Missing dependencies
raise a clear skip reason. Cached outputs include external version metadata,
model hash, input hash, output shape, and whether the adapter was explicitly
enabled for orchestration.

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
On Windows builds with the `directml` feature and a DirectX 12-capable adapter,
`backend="directml"` provides the CUDA-parity tensor surface for dense and
affine scoring, pair scoring, sparse diffusion and softmax forward/backward,
AdamW, layer normalization, and scalar-graph inference.

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
