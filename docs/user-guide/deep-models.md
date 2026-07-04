# CartoBoost Deep Model Guides

Use `cartoboost.deep` when the modeling unit is more structured than one
ordinary row: ordered pairs, candidate response curves, event probabilities,
baseline residual correction, graph sequences, or constrained candidate
selection.

Use `cartoboost.representation` when multiple deep surfaces need the same ID,
source-target, node-time, or context-dependent embedding contract. The stable
first cut includes `EntityEmbedding`, `PairEmbedding`, and
`SpatioTemporalAdaptiveEmbedding`. `RegimeRouter` adds deterministic
entity/context routing for mixture-of-experts surfaces and records router
entropy plus expert usage. `HistoricalAnalogRetriever` adds exact normalized
KNN memory for similar historical contexts, with explainable analog IDs,
distances, persisted memory, and optional cutoff filtering to avoid future
leakage. `SelfSupervisedPretrainer` creates reusable entity embeddings from
cutoff-safe feature summaries and records masked entity, masked pair,
graph-denoising, temporal-order, spatial-neighbor, and future-patch proxy
tasks. It emits reusable entity, pair, node, and temporal encoder outputs.
`MultiViewSpatialAttention` fuses several spatial views, records learned view
weights, emits ablation reports, and can transform when one fitted view is
missing. `RegimeMoEForecaster` combines six named regime experts with router
entropy, expert usage, expert predictions, combined predictions, and
single-expert comparison metrics. `ConditionalFlowDistributionHead` fits a
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
default AutoGeoModel selection and primary benchmark evidence.
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
`RetrievalAugmentedForecaster` and `RetrievalAugmentedPairModel` use
`HistoricalAnalogRetriever` memory to retrieve similar past contexts, attend to
their targets with deterministic inverse-distance weights, and return
explainable analog IDs, distances, and retrieved targets. Cutoff filtering keeps
future rows out of the memory query.
Foundation adapters such as `ChronosAdapter`, `TimesFMAdapter`,
`MoiraiAdapter`, `TimeGPTAdapter`, and `TabPFNAdapter` are optional wrappers for
external models. They can cache outputs with external version and model-hash
metadata, but they are never core dependencies and are only eligible for
AutoGeoModel-style orchestration when explicitly enabled.

These guides are for Python developers and data scientists choosing a model
surface. Start with the page that matches the unit being modeled, then validate
against a simpler baseline under the same split.

## Choose A Guide

| Need | Guide |
| --- | --- |
| Shared entity, pair, or adaptive context embeddings | `cartoboost.representation` |
| Fuse multiple graph or spatial views | `cartoboost.representation.MultiViewSpatialAttention` |
| Mixed geo-temporal regimes with named experts | [CartoBoost RegimeMoEForecaster](deep-models/cartoboost-regime-moe-forecaster.md) |
| CPU deterministic selective state-space forecasting | [CartoBoost TemporalSSMForecaster](deep-models/cartoboost-temporal-ssm-forecaster.md) |
| Wide synchronized panels with entity-token attention | [CartoBoost InvertedTemporalTransformer](deep-models/cartoboost-inverted-temporal-transformer.md) |
| Directed graph propagation with known lag priors | [CartoBoost PropagationDelayGraphForecaster](deep-models/cartoboost-propagation-delay-graph-forecaster.md) |
| Joint multi-horizon uncertainty from hidden-state context | [CartoBoost ConditionalFlowDistributionHead](deep-models/cartoboost-conditional-flow-distribution-head.md) |
| Experimental graph-wide residual scenario generation | [CartoBoost GeoTemporalDiffusionScenarioModel](deep-models/cartoboost-geotemporal-diffusion-scenario-model.md) |
| Advanced experimental spatial field-to-field mapping | [CartoBoost GraphNeuralOperator](deep-models/cartoboost-graph-neural-operator.md) |
| Candidate competition and counterfactual best selection | [CartoBoost ChoiceSetTransformer](deep-models/cartoboost-choice-set-transformer.md) |
| Retrieval-augmented rare-pattern forecasting | `cartoboost.representation.RetrievalAugmentedForecaster` |
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

Representation primitives are CPU-deterministic in the first cut. Their
artifact metadata already reserves `cuda`, `rocm`, and `mlx` as supported
accelerator targets, but `selected` remains `cpu` until native kernels for
those backends are implemented and validated. Do not report accelerated
representation results unless the artifact backend says that accelerator was
selected.

`TemporalSSMForecaster` and `SelectiveStateSpaceBlock` expose the selective
state-space backbone under `architecture="selective_ssm"`.
This is a deterministic CPU recurrence with one public architecture name;
accelerator kernels should attach to the same surface later.
Runtime scaling reports cover lookbacks 64, 128, 256, 512, and 1024, and
artifacts record save/load parity.

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

`ConditionalFlowDistributionHead` models joint residual uncertainty instead of
only independent quantile bands. Fit it on the hidden state emitted by a deep
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

`ChoiceSetTransformer` models candidate competition within decision groups
instead of scoring each candidate independently. Candidate value, candidate
features, context features, optional entity or pair embeddings, and existing
utility/probability fields feed a native utility head. The report includes
choice probabilities, nested probabilities when `nest_id` is present,
counterfactual best candidates by decision, and Brier/ECE calibration when
binary `chosen` labels are supplied.

`RetrievalAugmentedForecaster` fits exact normalized KNN memory over entity,
pair, time, recent-history, weather/event, graph-neighborhood, and residual
regime context keys. Prediction retrieves cutoff-safe analogs, computes
inverse-distance attention over retrieved targets, and can return the analog
IDs, distances, attention weights, and retrieved targets used for each
prediction. `RetrievalAugmentedPairModel` stores directional `source->target`
analog IDs for pair-specific retrieval.

Foundation model adapters are optional comparators and feature generators. Use
the adapter-specific extras such as `cartoboost[chronos]`,
`cartoboost[timesfm]`, `cartoboost[moirai]`, `cartoboost[timegpt]`, or
`cartoboost[tabpfn]`, or provide an explicit backend. Missing dependencies
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
