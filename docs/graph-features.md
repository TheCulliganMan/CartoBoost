# Graph Models And Features

Use graph models when connections between locations or entities carry signal
that ordinary row features miss. For taxi data, a directed pickup-to-dropoff
network can represent route frequency, travel patterns, or relationships
between zones. Import these tools from `cartoboost.graph`.

Start with the [graph model guides](user-guide/graph-models/index.md) to choose
a model. Return here for input shapes, directionality, generated features, and
saved-model behavior.

Graph support has two entry points:

- **Standalone graph models** fit, predict, score, save, and load directly.
- **Graph feature generators** produce dense columns for another estimator.

Keep direction explicit. Source-target facts are usually not interchangeable
with the reverse edge.

```mermaid
flowchart LR
    A["Typed graph + node features"] --> B["CartoBoost graph model"]
    B --> C["fit / predict / score"]
    C --> D["Graph artifact"]

    A --> E["Graph feature transformer"]
    E --> F["Dense graph columns"]
    F --> G["Downstream tabular model"]
```

## Choosing A Graph Family

| Family | Scientific use | Contract |
| --- | --- | --- |
| Node2Vec | Flow topology is the main signal and node attributes are not required. Useful for directed pickup-dropoff networks, repeated OD markets, or route neighborhoods. | Transductive random-walk embeddings over nodes present at fit time. |
| GraphSAGE | Zone attributes matter, such as airport flag, borough, recent pickup volume, or socioeconomic/context variables. | Homogeneous graph with one node and edge type plus node features. |
| HeteroGraphSAGE | Relation IDs matter, but you do not need a strict node-type schema. | Typed edges with relation-aware aggregation. |
| HinSAGE | Node types and relation triples are part of the causal or measurement design. | Typed nodes, typed relation triples, validation of source and target node types, and relation-aware sampling. |

The distinction matters for interpretation. Node2Vec asks whether observed flow
contexts alone explain residual variation. GraphSAGE-style models ask whether
node attributes and neighbor aggregation explain the variation. HinSAGE asks
whether typed scientific relations are valid modeling structure.

## CartoBoost Graph Models

CartoBoost graph regressors train the graph representation and row scorer as
one artifact. Use them when the graph should be evaluated as a model, not
merely as preprocessing.

Available regressors:

- `Node2VecStandaloneRegressor`
- `GraphSageStandaloneRegressor`
- `HeteroGraphSageStandaloneRegressor`
- `HinSageStandaloneRegressor`

Each supports `fit`, `predict`, `score`, `save`, and `load`.

### Directed Pair Regression

Use this pattern for origin-destination outcomes such as log duration or log
fare when each row has a source zone and target zone.

```python
import numpy as np
from cartoboost.graph import Node2VecStandaloneRegressor

edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
pickup = np.array([0, 1, 2, 3], dtype=np.uint64)
dropoff = np.array([1, 2, 3, 0], dtype=np.uint64)
distance_hour = np.array([[4.2, 8], [2.0, 9], [7.1, 17], [3.5, 22]], dtype=float)
log_duration = np.array([2.1, 1.6, 2.8, 1.9])

model = Node2VecStandaloneRegressor(dim=8, epochs=2, n_estimators=20, seed=11)
model.fit(
    node_count=4,
    edges=edges,
    row_nodes=pickup,
    row_targets=dropoff,
    dense=distance_hour,
    y=log_duration,
)

pred = model.predict(pickup, row_targets=dropoff, dense=distance_hour)
model.save("graph-node2vec-regressor.json")
```

Use `GraphSageStandaloneRegressor` instead when zone attributes should shape
the learned representation.

```python
from cartoboost.graph import GraphSageStandaloneRegressor

zone_features = np.array(
    [
        [1.0, 0.0],  # airport-like zone
        [0.0, 1.0],  # central-business zone
        [0.6, 0.3],
        [0.2, 0.7],
    ],
    dtype=np.float32,
)

model = GraphSageStandaloneRegressor(input_dim=2, hidden_dims=(4,), epochs=2)
model.fit(
    node_features=zone_features,
    edges=edges,
    row_nodes=pickup,
    row_targets=dropoff,
    y=log_duration,
)
```

## CartoBoost Link Prediction

Use CartoBoost link predictors when the question is about plausible movement
or ranking rather than a continuous target. Examples include ranking likely
dropoff zones from a pickup zone or scoring whether a route appears in a future
time block.

Available predictors:

- `Node2VecLinkPredictor`
- `GraphSageLinkPredictor`
- `HeteroGraphSageLinkPredictor`
- `HinSageLinkPredictor`

```python
from cartoboost.graph import Node2VecLinkPredictor

predictor = Node2VecLinkPredictor(dim=8, walk_length=8, walks_per_node=4, epochs=2)
predictor.fit(node_count=4, edges=edges)

candidate_pairs = [(0, 1), (0, 3)]
scores = predictor.predict_scores(candidate_pairs)
report = predictor.report(candidate_pairs, labels=[1, 0], query_ids=[0, 0], k=1)
```

`report` can include AUC, average precision, and per-query ranking metrics.

## Direction Is A Scientific Contract

Movement is usually asymmetric. Access rules, routing constraints, congestion,
and direction can all make `source -> target` meaningfully different from
`target -> source`. Do not collapse directional facts into one undirected edge
unless the study explicitly assumes symmetry.

Represent direction and role explicitly:

```yaml
graph:
  directed: true
  node_types:
    - source_zone
    - target_zone
    - source_target_pair
    - trip
    - time_bucket
  edge_types:
    - [source_zone, trips_to, target_zone]
    - [target_zone, reverse_trips_to, source_zone]
    - [trip, picked_up_in, source_zone]
    - [trip, dropped_off_in, target_zone]
    - [trip, observed_on, source_target_pair]
    - [source_target_pair, observed_in, time_bucket]
  directionality:
    materialize_reverse_edges: true
    preserve_source_target_roles: true
    create_od_pair_nodes: true
    compute_asymmetry_features: true
```

`materialize_reverse_edges` lets callers add reverse typed relations when those
relations should be learnable. `preserve_source_target_roles` records that
source and target columns are not interchangeable. When `create_od_pair_nodes`
is enabled through `GraphFeatureTransformer`, callers must pass `node_ids`; the
transformer appends stable tuple IDs such as
`("od_pair", pickup_zone, dropoff_zone)` for materialized pair nodes.

Directional feature extraction is opt-in through
`directionality.compute_asymmetry_features`. Common generated feature families
include:

- `source_target_embedding`
- `target_source_embedding`
- `forward_reverse_similarity_delta`
- `source_outbound_strength`
- `target_inbound_strength`
- `flow_imbalance_ratio`
- `directed_temporal_drift`
- `source_target_affinity`
- `target_source_affinity`

Generic package names are source-target oriented. Domain-specific labels such
as pickup/dropoff, origin/destination, or route-market names should be added in
the feature-engineering layer above CartoBoost.

## Node2Vec Details

`node2vec` follows the Grover and Leskovec design: second-order biased random
walks generate graph contexts, then a skip-gram negative-sampling objective
learns one dense vector per node. The return parameter `p` controls immediate
backtracking; the in-out parameter `q` controls whether walks stay local or
explore outward. CartoBoost keeps transitions directed, applies optional
non-negative edge weights, and trains deterministically for fixed settings.

Operational implications:

- Node2Vec is transductive; it learns vectors for nodes present during fit.
- `directed=True` means walks follow outgoing edges only.
- edge weights can represent flow volume, recency-weighted volume, acceptance
  rate, price pressure, or other source-target strength.
- OD problems should preserve source and target roles through distinct node IDs
  or OD-pair nodes.

```yaml
graph_embeddings:
  encoder:
    family: node2vec
    dim: 32
    walk_length: 16
    walks_per_node: 8
    window_size: 5
    epochs: 3
    p: 1.0
    q: 0.5
    seed: 7
    normalize: true
  directionality:
    preserve_source_target_roles: true
    compute_asymmetry_features: true
```

Reference: Grover and Leskovec, "node2vec: Scalable Feature Learning for
Networks" (KDD 2016).

## HinSAGE Details

Use HinSAGE when relation validity is part of the model specification. Edges are
integer triples `(source_node_id, target_node_id, relation_id)`, and
`node_types` assigns one type to each node.

```yaml
graph_embeddings:
  encoder:
    family: hinsage
    input_dim: 8
    node_type_count: 5
    edge_type_triples:
      - [0, 0, 1]  # source_zone trips_to target_zone
      - [1, 1, 0]  # target_zone reverse_trips_to source_zone
      - [3, 2, 0]  # trip picked_up_in source_zone
      - [3, 3, 1]  # trip dropped_off_in target_zone
      - [3, 4, 2]  # trip observed_on source_target_pair
    neighbor_samples: [25, 25, 10, 10, 20]
    hidden_dims: [16]
    epochs: 20
```

CartoBoost validates that:

- `node_type_count` is positive;
- relation IDs are zero-based and ordered;
- `edge_type_triples` are present and match the configured relation count;
- every edge relation exists;
- each edge source and target node type matches its relation triple;
- `neighbor_samples`, when supplied, has one cap per relation.

## Optional Feature Generation

Use `GraphFeatureTransformer` when graph structure is a feature source for a
separate model. This is useful for scientific ablations: fit a structured model,
then add graph-derived columns and measure whether directed flow structure
changes the same validation split.

```python
from cartoboost.graph import GraphFeatureTransformer

transformer = GraphFeatureTransformer.from_config(config)
bundle = transformer.fit_transform(
    node_features,
    edges=typed_edges,
    node_types=node_types,
    edge_weights=edge_weights,
    edge_timestamps=edge_timestamps,
)

X_graph = bundle.embeddings
feature_names = bundle.feature_names
metadata = bundle.training_config_metadata()
```

Use `HinSageFeatureEncoder` directly when you only need graph embeddings or
link-prediction features:

```python
from cartoboost.graph import HinSageConfig, HinSageFeatureEncoder

encoder = HinSageFeatureEncoder.from_config(
    HinSageConfig(
        input_dim=8,
        node_type_count=3,
        edge_type_triples=[(0, 0, 1), (1, 1, 0)],
        neighbor_samples=[25, 25],
    )
)

bundle = encoder.fit(node_features, edges=typed_edges, node_types=node_types)
link_bundle = encoder.link_embeddings(bundle.embeddings, pairs=[(0, 1), (1, 0)])
```

A `GraphFeatureBundle` provides dense graph columns, stable feature names, node
identifiers when provided, optional sparse graph sets, and provenance describing
encoder family, directedness, relation mapping, and generated feature names.
Persist that provenance with the downstream model metadata whenever graph
columns are generated outside the final model fit.

## Graph Regularization

Use graph regularization when the row graph itself is part of the modeling
contract. A `CsrGraph` stores sparse non-negative relations, `GraphLaplacian`
scores roughness across connected observations, and `GraphSmoother` can smooth
residual or leaf vectors against that graph.

The underlying graph surface covers row graphs, symbolic relation groups, and
split constraints:

| Primitive | Purpose |
| --- | --- |
| `CsrGraph` | Sparse supplied graph with deterministic validation and non-negative edge weights. |
| `GraphLaplacian` | Scores the graph roughness term `f' L f` for connected predictions. |
| `GraphSmoother` | Smooths residuals or update vectors over the supplied graph. |
| `GraphRegularizedBooster` | Booster wrapper for ordinary loss plus a graph smoothness penalty. |
| `GraphSplitRegularization` | Penalizes split candidates that create rough row-level updates across graph edges. |
| `GraphLeafSmoothing` | Aggregates a row graph to a leaf graph and smooths constant leaf values. |
| `SymbolicRelationSet` | Stores origin adjacency, destination adjacency, corridor similarity, reverse-edge similarity, neighbor-zone similarity, segment similarity, entity similarity, hierarchy parent/child, smooth-group, and no-smooth-group relations. |
| `RuleCompiler` | Compiles supplied relation groups, reverse edges, hierarchies, and violations into deterministic numeric features and penalties. |
| `MonotoneConstraintSet` and `InteractionConstraintSet` | Enforce split-search constraints for domain assumptions that must hold during training. |

`GraphSplitRegularization` adjusts candidate split gain by penalizing rough
row-level updates across supplied graph edges. `GraphLeafSmoothing` applies the
same graph contract after each fitted constant-leaf tree: training rows are
assigned to leaves, the row graph is aggregated to a leaf graph, and constant
leaf updates are smoothed before prediction updates are added.

Both options require a graph whose node count matches the training row count.
Leaf smoothing is intentionally limited to hard-routed constant leaves so the
leaf graph and update vector have a single unambiguous interpretation.

Graph penalties are explicit model features, not hidden fallbacks. If the graph
is missing, malformed, has the wrong node count, or contains invalid weights,
the model rejects it instead of silently training an unregularized model.

## Network Embedding Primitives

Node2Vec can also be used as a lower-level deterministic feature pipeline. The
underlying primitives are:

| Primitive | Purpose |
| --- | --- |
| `AliasSampler` | Seeded weighted sampling for graph transitions and negative examples. |
| `RandomWalkGenerator` | Directed and weighted second-order walks with fixed-seed deterministic output. |
| `Node2VecTrainer` | Skip-gram negative-sampling trainer over the generated walk contexts. |
| `EdgeEmbeddingModel` | Combines source and target node vectors into edge-level representations. |
| `EmbeddingFeatureTransformer` | Emits dense origin, destination, reverse-similarity, dot-product, neighborhood-density, and local-context columns for booster matrices. |

Use these when you need network context as explicit columns for a downstream
model or an ablation, rather than a standalone graph regressor artifact. Fixed
seeds are expected to reproduce the same embedding features.

## Directed Metapaths

Use typed directed metapaths when a relationship only makes sense in one
direction:

```yaml
meta_paths:
  - [pickup_zone, trips_to, dropoff_zone, reverse_trips_to, pickup_zone]
  - [trip, observed_on, source_target_pair, observed_in, time_bucket]
  - [source_zone, source_hour_volume, time_bucket]
```

`DirectedMetaPath` validates node/relation/node paths against `GraphSchema`.
`MetaPathWalkGenerator` can consume the relation path directly for random-walk
contexts, making direction part of the walk contract instead of a naming
convention.

## Validation And Reporting

Graph results should be reported with the scientific split they answer:

- random or tail splits test repeated-node interpolation;
- out-of-time splits test temporal transfer;
- cold-zone or cold-route splits test whether graph structure generalizes to
  held-out zones or OD pairs;
- link reports should name candidate construction, negative sampling, query
  groups, and ranking metric `k`.

Keep feature-generation and standalone-model claims separate. A standalone graph
model claim is about the graph model artifact. A feature-generation claim is
about a downstream model that consumed graph columns under a fixed
feature-generation contract.
