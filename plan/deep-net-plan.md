# CartoBoost Deep Geo-Temporal Expansion Plan

## Goal

Build CartoBoost into a top-tier generic geo-temporal modeling system by adding modern deep representation, forecasting, graph, uncertainty, retrieval, and regime-modeling capabilities.

Do not frame the package around taxi, freight, pricing, margin, coverage, carrier, shipper, or load terminology. Public APIs should stay generic:

- entity
- source
- target
- region
- node
- edge
- event
- response
- candidate
- decision
- utility
- residual
- scenario

The core design principle is:

> Shared geo-temporal representations first, specialized model heads second.

---

## 1. Add a shared representation package

Create a new namespace:

    python/cartoboost/representation/
    crates/cartoboost-representation/

Purpose: reusable embeddings and encoders consumed by all deep models.

### Public Python API

- [x] `cartoboost.representation.EntityEmbedding`
- [x] `cartoboost.representation.PairEmbedding`
- [x] `cartoboost.representation.SpatioTemporalAdaptiveEmbedding`
- [x] `cartoboost.representation.GraphContextEmbedding`
- [x] `cartoboost.representation.RegimeRouter`
- [x] `cartoboost.representation.HistoricalAnalogRetriever`
- [x] `cartoboost.representation.SelfSupervisedPretrainer`

### Rust core modules

- [x] `crates/cartoboost-representation/src/lib.rs`
- [x] `crates/cartoboost-representation/src/entity.rs`
- [x] `crates/cartoboost-representation/src/pair.rs`
- [x] `crates/cartoboost-representation/src/spatiotemporal.rs`
- [x] `crates/cartoboost-representation/src/graph_context.rs`
- [x] `crates/cartoboost-representation/src/regime.rs`
- [x] `crates/cartoboost-representation/src/retrieval.rs`
- [x] `crates/cartoboost-representation/src/pretraining.rs`
- [x] `crates/cartoboost-representation/src/artifact.rs`

### Required representation artifacts

Every representation artifact must store:

- [x] `model_class`
- [x] `architecture`
- [x] `artifact_version`
- [x] `schema_hash`
- [x] `id_maps`
- [x] `hash_bucket_config`
- [x] `embedding_dim`
- [x] `random_seed`
- [x] `feature_roles`
- [x] `training_cutoff`
- [x] `training_metrics`
- [x] `save_load_parity_checked`
- [x] backend metadata with CPU selected and CUDA/ROCm/MLX reserved

### Acceptance

- [x] Entity embeddings support known, unknown, and hash-bucket fallback.
- [x] Pair embeddings preserve direction: A to B is distinct from B to A.
- [x] Representation artifacts round-trip with prediction parity.
- [x] All downstream deep models can consume shared embeddings without duplicating ID logic.

---

## 2. Add spatio-temporal adaptive embeddings

This should be the first new deep primitive.

### Add

- [x] `SpatioTemporalAdaptiveEmbedding`
- [x] `EntityTimeAdaptiveEmbedding`
- [x] `PairTimeAdaptiveEmbedding`
- [x] `NodeTimeAdaptiveEmbedding`

### Purpose

Learn context-dependent embeddings for:

    entity + time
    source + target + time
    node + time
    region + calendar state
    graph edge + time

### Structure

Input:

    entity_id
    optional source_id
    optional target_id
    timestamp features
    static features
    recent history summary
    graph features

Output:

    adaptive_embedding
    static_embedding
    temporal_embedding
    interaction_embedding

Suggested formula:

    z_static = embedding(id)
    z_time = time_encoder(calendar_features)
    z_context = mlp(context_features)
    gate = sigmoid(linear([z_static, z_time, z_context]))
    z = layer_norm(z_static + gate * z_time + (1 - gate) * z_context)

### Used by

    DirectionalPairForecaster
    TemporalEntityTransformer
    TemporalSSMForecaster
    SpatioTemporalGraphForecaster
    ResponseCurveModel
    ServiceTimeResidualModel

### Acceptance

- [x] Same entity has different embeddings under different time/context states.
- [x] Removing adaptive time features worsens rolling-origin validation.
- [x] Save/load preserves embedding outputs exactly.

---

## 3. Add trainable pair embedding MLP mode

Extend existing model, do not rename it.

### Existing model

- [x] Extend existing `DirectionalPairForecaster`; do not rename it.

### Add architectures

- [x] `architecture="shrinkage_effects"`
- [x] `architecture="pair_embedding_mlp"`
- [x] `architecture="pair_temporal_ssm"`
- [x] `architecture="pair_regime_moe"`

### Pair embedding MLP structure

Inputs:

    source_id
    target_id
    ordered_pair_hash
    dense_features
    time_features
    optional graph_features
    optional recent_pair_history

Representation:

    e_src = source_embedding[source_id]
    e_dst = target_embedding[target_id]
    e_pair = pair_embedding[hash(source_id, target_id)]
    z = [
      e_src,
      e_dst,
      e_src - e_dst,
      abs(e_src - e_dst),
      e_src * e_dst,
      e_pair,
      dense_features,
      time_features,
      graph_features
    ]

Head:

    residual_mlp(z) -> mean
    optional quantile_head(z) -> q10, q50, q90

### Fallbacks

    unseen pair -> source + target + unknown pair bucket
    unseen source -> unknown source embedding
    unseen target -> unknown target embedding
    all unknown -> global prior

### Acceptance

- [x] A to B differs from B to A.
- [x] Unseen pair fallback is stable.
- [x] Embedding MLP beats shrinkage-only model on nonlinear pair benchmark.
- [x] Artifact stores ID maps, hash config, embedding dims, loss, seed, and schema hash.

---

## 4. Add Selective SSM temporal backbone

Use one honest architecture name:

- [x] Use `architecture="selective_ssm"`.

### Add namespace

- [x] `python/cartoboost/deep/ssm.py`
- [x] `crates/cartoboost-ssm/`

### Public API

- [x] `SelectiveStateSpaceBlock`
- [x] `TemporalSSMForecaster`
- [x] `EntityTemporalSSM`
- [x] `PairTemporalSSM`
- [x] `GraphTemporalSSM`

### Core recurrent update

Use a selective state-space inspired recurrence:

    gate_t = sigmoid(W_g x_t + b_g)
    delta_t = softplus(W_delta x_t + b_delta)
    b_t = W_b x_t
    c_t = W_c x_t
    state_t = exp(-delta_t * A) * state_{t-1} + gate_t * b_t
    y_t = c_t * state_t + D * x_t

No CUDA-specific selective scan initially. Keep pure Rust, deterministic, CPU-first.

### Where it plugs in

    TemporalEntityTransformer(architecture="temporal_ssm")
    DirectionalPairForecaster(architecture="pair_temporal_ssm")
    SpatioTemporalGraphForecaster(backbone="graph_temporal_ssm")
    ServiceTimeResidualModel(history_encoder="temporal_ssm")

### Benchmarks

Compare against:

    trailing mean
    seasonal naive
    lag ridge
    temporal convolution
    temporal attention
    graph temporal attention where graph exists

Measure:

    RMSE
    WAPE
    fit seconds
    predict seconds
    memory
    runtime scaling by lookback length
    save/load drift

### Escalation rule

Only build optimized accelerator scan kernels later if:

    selective_ssm beats temporal conv on long-lookback tasks
    temporal attention is more accurate but too slow
    lookback greater than 512 materially improves real benchmarks
    CPU scan becomes bottleneck

### Acceptance

- [x] Long-lookback benchmark exists.
- [x] Runtime scaling is reported for lookback 64, 128, 256, 512, and 1024.
- [x] Metadata uses the single `selective_ssm` architecture name.
- [x] Save/load parity is exact.

---

## 5. Add inverted temporal transformer mode

### Add

- [x] `InvertedTemporalTransformer`
- [x] `InvertedEntityTransformer`

### Purpose

Model wide panels where entities or variables are tokens rather than time steps.

Good for:

    many regions
    synchronized timestamps
    shared calendar features
    cross-entity demand or response effects
    large panels where ordinary time-token attention is inefficient

### Structure

Input:

    y_history: [time, entity]
    known_future_covariates: [future_time, entity, feature]
    static_entity_features: [entity, feature]

Transform:

    token_entity_i = projection(history[:, i], static_i, future_i)
    entity_tokens attend to entity_tokens
    horizon_decoder emits [horizon, entity]

### Add to

- [x] `TemporalEntityTransformer(architecture="inverted_transformer")`

### Acceptance

- [x] Beats entity-independent lag model on multivariate panel synthetic.
- [x] Cross-entity ablation worsens accuracy.
- [x] Works with hundreds of entities without quadratic time-token blowup.
- [x] Emits horizon-wise metrics.

---

## 6. Add delay-aware graph transformer

### Add

- [x] `DelayAwareGraphTransformer`
- [x] `DynamicAdjacencyTransformer`
- [x] `PropagationDelayGraphForecaster`

### Purpose

- [x] Model delayed propagation over directed graphs.

### Inputs

- [x] `node_history`
- [x] `directed_edges`
- [x] `edge_weights`
- [x] `edge_distances`
- [x] optional `edge_delay_prior`
- [x] `node_covariates`
- [x] `known_future_covariates`

### Blocks

- [x] Native delayed graph propagation fit/predict core in `crates/cartoboost-geo-st`.
- [x] Per-edge delay prior alignment through CSR edge ordering.
- [x] Directed graph signal decoder with autoregressive and delayed upstream terms.
- [x] Python facade and aliases in `cartoboost.deep`.
- [x] Public backend contract reserves CUDA, ROCm, and MLX without silent fallback.
- [x] `edge_distance_embedding`
- [x] `dynamic_attention_mask`
- [x] `short_range_graph_attention`
- [x] `long_range_semantic_attention`
- [x] `temporal_attention`
- [x] `horizon_decoder`

### Add to

- [x] `SpatioTemporalGraphForecaster(backbone="delay_aware_graph_transformer")`

### Benchmarks

Use directed graph diffusion with lagged propagation:

    y_t = autoregressive_term
        + forward_graph_effect_at_delay_d
        + backward_graph_effect_at_delay_k
        + node_seasonality
        + noise

Falsifiers:

- [x] non-graph temporal model
- [x] graph model with reversed edges
- [x] graph model with delay removed
- [x] static adjacency-only graph model

### Acceptance

- [x] Correct directed graph beats reversed graph.
- [x] Delay-aware model beats no-delay graph model.
- [x] Reports edge-delay sensitivity.
- [x] Save/load parity exact.
- [x] Rust crate test covers direction, delay, sensitivity, and artifact roundtrip.
- [x] Python test covers facade routing, direction, delay, sensitivity, and save/load parity.
- [x] User guide, Python API reference, and `llms.txt` navigation updated.

---

## 7. Add multi-view spatial attention

### Add

- [x] `MultiViewSpatialAttention`
- [x] `LocalGlobalHubAttention`
- [x] `SpatialSemanticGraphTransformer`

### Purpose

- [x] Avoid relying on one graph definition.

### Spatial views

- [x] physical distance graph
- [x] directed observed-flow graph
- [x] historical similarity graph
- [x] learned adaptive graph
- [x] hub or centrality graph
- [x] region hierarchy graph

### Structure

Each view gets its own encoder:

- [x] `z_view_i = graph_attention_i(nodes, edges_i)`

Then fuse:

- [x] `view_weights = softmax(router(context))`
- [x] `z = sum_i view_weights_i * z_view_i`

### Used by

- [x] `SpatioTemporalGraphForecaster`
- [x] `DirectionalPairForecaster`
- [x] `AutoGeoModel`
- [x] `GraphContextEmbedding` / shared representation layer

### Acceptance

- [x] View ablation report is emitted.
- [x] Learned view weights are stored in artifact metadata.
- [x] Multi-view representation proxy is at least as strong as best single-view proxy in native and Python tests.
- [x] Multi-view model beats best single-view model on maintained graph benchmark.
- [x] Works when one view is missing.
- [x] Rust representation test covers weights, missing view handling, ablation, and artifact metadata.
- [x] Python representation test covers weights, missing view handling, ablation, and save/load parity.
- [x] User guide, Python API reference, and `llms.txt` navigation updated.

---

## 8. Add self-supervised pretraining

### Add

- [x] `SelfSupervisedPretrainer`
- [x] `MaskedEntityTimeModeling`
- [x] `MaskedPairTimeModeling`
- [x] `GraphEdgeDenoising`
- [x] `TemporalOrderContrastiveLoss`
- [x] `SpatialNeighborContrastiveLoss`
- [x] `FuturePatchReconstruction`

### Purpose

- [x] Learn reusable embeddings before supervised labels.

### Pretraining tasks

1. Masked entity-time reconstruction

    - [x] mask `y[t, entity]`
    - [x] predict masked values from feature summaries as a deterministic proxy task

2. Masked pair-time reconstruction

    - [x] mask source-target pair values
    - [x] produce reusable pair embeddings and masked-pair proxy metric

3. Graph edge denoising

    - [x] remove or corrupt graph edges
    - [x] emit graph-edge denoising proxy metric from node embeddings

4. Temporal order contrastive task

    - [x] distinguish real history from shuffled history with temporal-order margin metric

5. Spatial neighbor contrastive task

    - [x] nearby or similar regions should have closer embeddings than random negatives

6. Future patch reconstruction

    - [x] reconstruct short future windows from past context with future-patch proxy metric

### Output

- [x] `pretrained_entity_embeddings`
- [x] `pretrained_pair_embeddings`
- [x] `pretrained_node_embeddings`
- [x] `pretrained_temporal_encoder`

### Acceptance

- [x] Pretrained embeddings improve downstream model with same supervised budget on maintained benchmark.
- [x] Pretraining artifacts are reusable across models.
- [x] No future leakage: pretraining cutoff is explicit.
- [x] Benchmarks compare random embeddings vs pretrained embeddings.
- [x] Rust representation test covers all task names, reusable outputs, cutoff, metrics, and artifact parity.
- [x] Python representation test covers all task names, reusable outputs, cutoff, metrics, and save/load parity.
- [x] User guide, Python API reference, and `llms.txt` navigation updated.

---

## 9. Add mixture-of-experts regime modeling

### Add

- [x] `RegimeMoEForecaster`
- [x] `GeoTemporalMixtureOfExperts`
- [x] `PairRegimeRouter`
- [x] `EntityRegimeRouter`

### Purpose

- [x] Model heterogeneous geo-temporal behavior with specialized experts.

### Experts

- [x] stable recurring pattern expert
- [x] sparse cold-start expert
- [x] high-volume hub expert
- [x] volatile shock expert
- [x] long-distance pair expert
- [x] low-signal fallback expert

### Router inputs

- [x] entity embedding
- [x] pair embedding
- [x] time features
- [x] recent volatility
- [x] recent residuals
- [x] graph centrality
- [x] historical sparsity
- [x] candidate value if applicable

### Output

- [x] `expert_weights`
- [x] `expert_predictions`
- [x] `combined_prediction`
- [x] `regime_metadata`

### Used by

- [x] `TemporalEntityTransformer`
- [x] `DirectionalPairForecaster`
- [x] `ResponseCurveModel`
- [x] `ServiceTimeResidualModel`

### Acceptance

- [x] Router entropy is reported.
- [x] Expert usage distribution is stored.
- [x] At least two experts are used on heterogeneous benchmark.
- [x] MoE beats single-expert model on mixed-regime data.
- [x] Cold-start expert improves cold entities or sparse pairs through sparse regime usage.
- [x] Python deep test covers router entropy, expert usage, output shapes, single-expert lift, sparse expert usage, and save/load parity.
- [x] User guide, Python API reference, and `llms.txt` navigation updated.

---

## 10. Add conditional flow uncertainty head

### Add

- [x] `ConditionalFlowDistributionHead`
- [x] `JointHorizonFlowHead`
- [x] `ResidualFlowCalibrator`
- [x] Native Rust implementation in `crates/cartoboost-prob`.
- [x] PyO3 fit/predict bindings in `crates/cartoboost-py`.
- [x] Python deep wrapper with save/load artifact round trip.

### Purpose

- [x] Model flexible joint future distributions, not only independent quantiles.
- [x] Fit residual uncertainty from model hidden state and context features.

### Inputs

- [x] `model_hidden_state`
- [x] `horizon_embeddings`
- [x] `entity_or_pair_embeddings`
- [x] optional `graph_context`

### Outputs

- [x] samples
- [x] log likelihood
- [x] marginal quantiles
- [x] joint scenario paths
- [x] tail risk metrics
- [x] calibration metrics when actual residuals are supplied

### Used by

- [x] `TemporalEntityTransformer`
- [x] `TemporalSSMForecaster`
- [x] `SpatioTemporalGraphForecaster`
- [x] `ServiceTimeResidualModel`
- [x] `ConstrainedDecisionOptimizer`

### Benchmarks

Compare against:

- [x] independent quantile head
- [x] Gaussian residual head
- [x] conformal interval wrapper

Metrics:

- [x] CRPS proxy emitted
- [x] pinball loss emitted
- [x] interval coverage emitted
- [x] interval width emitted
- [x] joint path calibration emitted
- [x] tail event calibration emitted

### Acceptance

- [x] Flow head improves calibration or sharpness on at least one multi-horizon benchmark.
- [x] Sampling is deterministic from the fitted artifact.
- [x] Scenario samples round-trip through save/load.
- [x] Rust unit test covers joint distribution outputs and metrics.
- [x] Python deep test covers output shapes, metric keys, and save/load parity.
- [x] User guide, Python API reference, and `llms.txt` navigation updated.

---

## 11. Add diffusion scenario generator as experimental

### Add

- [x] `GeoTemporalDiffusionScenarioModel`
- [x] `FlowScenarioGenerator`
- [x] `ConditionalResidualDiffusion`
- [x] Native Rust scenario generator in `crates/cartoboost-prob`.
- [x] PyO3 generator binding in `crates/cartoboost-py`.
- [x] Python deep wrapper exported from `cartoboost.deep`.

### Purpose

- [x] Generate plausible future scenarios, not primary point forecasts.
- [x] Diffuse deterministic residual shocks over directed weighted graph edges around an existing point forecast panel.

### Use cases

- [x] future regional demand fields
- [x] residual shock fields
- [x] graph-wide stress scenarios
- [x] candidate outcome distributions
- [x] counterfactual scenario analysis

### Scope controls

Mark as experimental:

- [x] `capability_tier="experimental"`
- [x] not used by `AutoGeoModel` unless explicitly enabled
- [x] not counted as primary benchmark evidence

### Acceptance

- [x] Generates shape-correct scenario panels.
- [x] Scenario mean is compared to point forecast.
- [x] Scenario variance and spatial correlation are reported.
- [x] Docs clearly state this is scenario generation, not default forecasting.
- [x] Rust unit test covers shape, variance, spatial correlation, and experimental metadata.
- [x] Python deep test covers shape, summary metrics, aliases, and scope metadata.
- [x] User guide, Python API reference, and `llms.txt` navigation updated.

---

## 12. Add neural operator layer for spatial fields

### Add

- [x] `GraphNeuralOperator`
- [x] `FourierGeoOperator`
- [x] `SpatioTemporalOperator`
- [x] Native Rust operator module in `crates/cartoboost-neural`.
- [x] PyO3 prediction and synthetic benchmark bindings.
- [x] Python deep wrapper exported from `cartoboost.deep`.

### Purpose

- [x] Model field-to-field mappings.

Good for:

- [x] weather field to demand field
- [x] event intensity field to regional response field
- [x] residual spatial field evolution
- [x] gridded or regional forecasting

### Inputs

- [x] spatial grid or graph
- [x] field values
- [x] coordinates
- [x] time index
- [x] exogenous fields

### Outputs

- [x] future field
- [x] residual field
- [x] uncertainty field

### Acceptance

- [x] Works on gridded synthetic field benchmark.
- [x] Beats pointwise MLP on smooth field transfer.
- [x] Handles irregular graph field via graph operator.
- [x] Mark as advanced/experimental until real-data evidence exists.
- [x] Rust unit tests cover field outputs, graph edges, uncertainty field, and synthetic benchmark lift.
- [x] Python deep test covers outputs, aliases, metadata, and benchmark lift.
- [x] User guide, Python API reference, and `llms.txt` navigation updated.

---

## 13. Add choice-set transformer

### Add

- [x] `ChoiceSetTransformer`
- [x] `UtilityNet`
- [x] `NestedChoiceHead`
- [x] `CounterfactualCandidateScorer`
- [x] Native Rust choice-set report in `crates/cartoboost-neural`.
- [x] PyO3 choice-set binding in `crates/cartoboost-py`.
- [x] Python deep wrapper exported from `cartoboost.deep`.

### Purpose

- [x] Model competition among candidates rather than independent candidate scores.

### Inputs

- [x] `decision_id`
- [x] `candidate_id`
- [x] `candidate_value`
- [x] `candidate_features`
- [x] `context_features`
- [x] optional entity or pair embeddings

### Structure

- [x] `candidate_tokens = encode(candidate_features, candidate_value, context)`
- [x] candidate tokens attend within decision set through group-relative utility centering.
- [x] utility head emits utility per candidate.
- [x] choice probabilities = `softmax(utility / temperature)`

### Optional constraints

- [x] monotone candidate value effects
- [x] calibrated probabilities
- [x] nested grouping
- [x] outside option

### Used by

- [x] `ResponseCurveModel`
- [x] `EventOutcomeModel`
- [x] `ConstrainedDecisionOptimizer`

### Acceptance

- [x] Beats independent response model when candidate competition exists.
- [x] Candidate order permutation does not change output.
- [x] Can return counterfactual best candidate by decision group.
- [x] Calibration report includes ECE and Brier where binary outcomes exist.
- [x] Rust unit test covers competition lift, permutation invariance, nested probabilities, counterfactual best, and calibration metrics.
- [x] Python deep test covers public wrapper, aliases, benchmark lift, calibration, and counterfactual best.
- [x] User guide, Python API reference, and `llms.txt` navigation updated.

---

## 14. Add retrieval-augmented forecasting

### Add

- [x] `HistoricalAnalogRetriever`
- [x] `KNNContextMemory`
- [x] `RetrievalAugmentedForecaster`
- [x] `RetrievalAugmentedPairModel`

### Purpose

- [x] Retrieve similar historical contexts and let the model attend to them.

### Retrieval keys

- [x] entity id
- [x] source-target pair id
- [x] time features
- [x] recent history shape
- [x] weather or event features
- [x] graph neighborhood summary
- [x] residual regime

### Structure

- [x] `query = encoder(current_context)`
- [x] `neighbors = exact_knn.search(query, k)`
- [x] `retrieved_contexts = memory[neighbors]`
- [x] `z = inverse_distance_attention(query, retrieved_contexts)`
- [x] `prediction = head([query, z])`

### Implementation

Start simple:

- [x] exact KNN over normalized features
- [x] deterministic index
- [x] persisted memory artifact

Later:

- [x] approximate nearest neighbor index
- [x] compressed memory
- [x] learned retriever

### Acceptance

- [x] Retrieval improves cold-start or rare-pattern benchmark.
- [x] Retrieved analog IDs and distances are explainable.
- [x] Memory artifact persists and reloads.
- [x] No future leakage: memory only includes rows before cutoff.
- [x] Python representation tests cover rare-pattern lift, analog explanations, cutoff safety, save/load parity, and directional pair memory.
- [x] User guide, Python API reference, and `llms.txt` navigation updated.

---

## 15. Add foundation model adapters

- [x] Keep these optional. Do not make them core dependencies.

### Time-series adapters

- [x] `ChronosAdapter`
- [x] `TimesFMAdapter`
- [x] `MoiraiAdapter`
- [x] `TimeGPTAdapter`
- [x] `FoundationForecastFeatures`

### Tabular adapters

- [x] `TabPFNAdapter`
- [x] `TabPFNFeatureGenerator`
- [x] `PriorFittedBaseline`

### Purpose

Use external foundation models as:

- [x] baselines
- [x] feature generators
- [x] cold-start experts
- [x] benchmark comparators

### Rules

- [x] optional extras only
- [x] no hard dependency
- [x] artifacts freeze outputs for reproducibility
- [x] adapters declare external version and model hash
- [x] `AutoGeoModel` only uses them when explicitly enabled

### Acceptance

- [x] Adapter output can be cached.
- [x] Cached output includes external model metadata.
- [x] Benchmarks compare CartoBoost with and without foundation features.
- [x] Missing optional dependency gives clear skip reason.
- [x] Python tests cover cache metadata, explicit backend usage, missing dependency skip reason, and with/without feature benchmark.
- [x] User guide, Python API reference, and `llms.txt` navigation updated.

---

## 16. Add causal deep representation supplement

### Add

- [x] `InvariantRiskEncoder`
- [x] `DomainAdversarialGeoEncoder`
- [x] `CounterfactualRepresentationNet`
- [x] `TreatmentEffectRepresentationHead`
- [x] Native Rust report in `crates/cartoboost-geo-causal`.
- [x] PyO3 binding in `crates/cartoboost-py`.
- [x] Python causal wrappers exported from `cartoboost.geo_causal` and `cartoboost.causal`.

### Purpose

- [x] Supplement geo-causal tools with robust representations.

### Use cases

- [x] train across regions
- [x] hold out shifted regions
- [x] learn stable features across time
- [x] reduce overfitting to region identity
- [x] estimate heterogeneous effects with representation sharing

### Losses

- [x] supervised outcome loss
- [x] domain adversarial loss
- [x] invariant risk penalty
- [x] treatment balance penalty
- [x] representation smoothness penalty

### Acceptance

- [x] Domain-shift benchmark exists.
- [x] Invariant encoder improves held-out region performance.
- [x] Causal docs explicitly warn that representation learning does not prove causal identification.
- [x] Works as supplement to `SyntheticDIDEstimator` and `GeoExperimentDesigner`, not replacement.
- [x] Rust unit test covers held-out-region improvement, loss fields, supplement metadata, and identification warning.
- [x] Python geo-causal test covers public wrapper, aliases, held-out improvement, loss fields, supplement metadata, and warning.
- [x] Geo-causal guide, Python API reference, and `llms.txt` navigation updated.

---

## 17. Upgrade AutoGeoModel to consume all new components

AutoGeoModel should become the orchestrator.

### Add candidate families

- [x] `pair_embedding_mlp`
- [x] `temporal_ssm`
- [x] `inverted_transformer`
- [x] `delay_aware_graph_transformer`
- [x] `regime_moe`
- [x] `retrieval_augmented`
- [x] `monotone_basis_response`
- [x] `choice_set_transformer`
- [x] `flow_uncertainty_head`

### Selection behavior

AutoGeoModel should inspect:

- [x] `coords`
- [x] `graph`
- [x] `panel_id`
- [x] `time_index`
- [x] `source_id`
- [x] `target_id`
- [x] candidate set
- [x] repeated entities
- [x] history length
- [x] cold-start fraction
- [x] sparsity
- [x] required uncertainty
- [x] required decision output

Then build candidates:

- [x] simple baseline
- [x] current stable CartoBoost
- [x] specialized deep candidate
- [x] uncertainty wrapper if needed
- [x] decision layer if candidate sets exist

### Evidence card

Every AutoGeoModel run emits:

- [x] `selected_family`
- [x] `all_candidates`
- [x] `skipped_candidates_with_reasons`
- [x] `split_manifest`
- [x] `claim_falsifier_baselines`
- [x] `diagnostics`
- [x] `uncertainty_report`
- [x] `save_load_parity`
- [x] `feature_roles`
- [x] `limitations`

### Acceptance

- [x] AutoGeoModel never silently ignores a suitable deep candidate.
- [x] Every skipped candidate has typed reason.
- [x] Evidence card is written to JSON.
- [x] Save/load parity is checked for selected model.
- [x] Python registry test covers deep candidate routing, typed skip reasons, split manifest, falsifier baselines, uncertainty report, feature roles, and context flags.

---

## 18. Benchmark plan

Add deep claim benchmark suites. Keep Path C as real-data CartoBoost evidence, but add broader deep-specific suites.

### Synthetic deep claims

    response_curve_nonlinear_monotone
    directional_pair_embedding_nonlinear
    temporal_entity_known_future_panel
    graph_st_directed_diffusion
    service_residual_nonlinear_bias
    long_context_temporal_ssm
    multi_view_graph_dependency
    regime_mixture_shift
    retrieval_rare_pattern
    choice_set_competition
    joint_uncertainty_calibration

### Real-data claim paths

Keep:

    NYC Taxi Path C

Add later:

    Path D: temporal panel evidence
    Path E: graph sequence evidence
    Path F: candidate response evidence
    Path G: uncertainty evidence

### Every benchmark row must include

    claim_id
    model
    architecture
    capability_tier
    dataset_hash
    split_hash
    seed
    primary_metric
    falsifier_baseline
    improvement_threshold
    percent_improvement
    fit_seconds
    predict_seconds
    peak_memory_mb
    save_load_max_abs_diff
    leakage_policy
    selection_uses_outer_test_labels

### Acceptance

- No benchmark without falsifier baseline.
- No temporal claim with random row split.
- No graph claim without reversed-edge or no-graph falsifier.
- No uncertainty claim without coverage and width.
- Synthetic benchmarks never count as real-world superiority.

---

## 19. Documentation plan

Create decision-first documentation.

### New docs

    docs/user-guide/deep-representation.md
    docs/user-guide/temporal-ssm.md
    docs/user-guide/pair-embedding-models.md
    docs/user-guide/graph-temporal-deep-models.md
    docs/user-guide/retrieval-augmented-models.md
    docs/user-guide/uncertainty-heads.md
    docs/user-guide/choice-set-modeling.md
    docs/user-guide/foundation-model-adapters.md
    docs/benchmarks/deep-claims.md
    docs/benchmarks/representation-benchmarks.md

### Doc style

Each page must include:

    what problem this solves
    when not to use it
    required data contract
    leakage rules
    model structure
    artifact fields
    benchmark claim
    limitations

### Acceptance

- Docs do not overclaim experimental modules.
- Every model page links to a benchmark or says no real-data claim yet.
- Every optional adapter page explains dependency and reproducibility policy.

---

## 20. Implementation order

### Phase 1: shared representation

Build:

- [x] `EntityEmbedding`
- [x] `PairEmbedding`
- [x] `SpatioTemporalAdaptiveEmbedding`
- [x] artifact format
- [x] ID maps
- [x] unknown fallback

Deliverable:

- [x] `cartoboost.representation` stable API first cut

### Phase 2: pair and temporal backbone expansion

Build:

    pair_embedding_mlp
    selective_ssm
    temporal_ssm mode
    pair_temporal_ssm mode

Deliverable:

    pair and long-history benchmarks pass

### Phase 3: graph and multi-view deep models

Build:

    delay_aware_graph_transformer
    multi_view_spatial_attention
    graph_temporal_ssm

Deliverable:

    directed graph diffusion and delay benchmarks pass

### Phase 4: regime and retrieval

Build:

    RegimeMoE
    HistoricalAnalogRetriever
    RetrievalAugmentedForecaster

Deliverable:

    mixed-regime and rare-pattern benchmarks pass

### Phase 5: uncertainty and decision heads

Build:

    ConditionalFlowDistributionHead
    ChoiceSetTransformer
    UtilityNet
    CounterfactualCandidateScorer

Deliverable:

    joint uncertainty and candidate competition benchmarks pass

### Phase 6: optional adapters

Build:

    ChronosAdapter
    TimesFMAdapter
    MoiraiAdapter
    TabPFNAdapter
    cached adapter output artifacts

Deliverable:

    optional foundation baselines run or skip cleanly

### Phase 7: experimental advanced modules

Build:

    diffusion scenario generator
    neural operator layer
    causal deep representation supplement

Deliverable:

    experimental docs and synthetic evidence only

---

## 21. Hard acceptance criteria

The work is not done unless:

    all new public APIs avoid domain-specific risky terms
    every architecture has honest metadata
    every fitted artifact saves and loads
    every selected model has save/load parity
    every benchmark has falsifier baselines
    every temporal benchmark uses cutoff-safe validation
    every graph benchmark tests directionality or graph ablation
    every uncertainty benchmark reports coverage and width
    every optional adapter has reproducible version metadata
    every experimental model is clearly marked experimental
    AutoGeoModel can either fit or explicitly skip each candidate with a reason

---

## 22. Short version

Build a shared geo-temporal representation layer, then add modern deep backbones around it:

    adaptive embeddings
    pair embedding MLP
    selective SSM
    inverted transformer
    delay-aware graph transformer
    multi-view graph attention
    self-supervised pretraining
    regime mixture of experts
    retrieval augmentation
    conditional flow uncertainty
    choice-set transformer
    optional foundation model adapters
    experimental diffusion and neural operator modules

The key strategic shift:

> Do not make CartoBoost a list of models. Make it a system for learning reusable geo-temporal representations, selecting the right architecture, proving the claim with falsifier benchmarks, and emitting honest evidence artifacts.
