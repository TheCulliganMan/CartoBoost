# Graph Model Guides

These guides cover graph models for taxi relationships: directed pickup to
dropoff movement, repeated route markets, zone neighborhoods, typed relations,
and link likelihood. Use this section when the relationship network is the
modeling surface, not just a helper column.

Use [Boosting Model Guides](../boosting-models/index.md) for row-level boosted
trees. Use [Neural Model Guides](../neural-models/index.md) for learned ID
embeddings without an explicit graph.

## Pick A Guide

| Model guide | Best first use | Notes |
| --- | --- | --- |
| [Standalone Graph Regressors](standalone-regressors.md) | Predict fare, duration, demand residuals, or other continuous targets from graph context. | Covers Node2Vec, GraphSAGE, HeteroGraphSAGE, and HinSAGE regressors. |
| [Graph Link Predictors](link-predictors.md) | Score or rank plausible source-target movements. | Useful for pickup-to-dropoff likelihood and route-candidate ranking. |
| [Graph Feature Workflows](feature-workflows.md) | Emit graph-derived dense columns for another estimator. | Best for ablation studies and graph-augmented boosting. |

## Choosing A Graph Family

| Family | Scientific use | Contract |
| --- | --- | --- |
| Node2Vec | Flow topology is the main signal and node attributes are not required. | Transductive random-walk embeddings over nodes present at fit time. |
| GraphSAGE | Zone attributes matter, such as airport flag, borough, or recent pickup volume. | Homogeneous graph with one node and edge type plus node features. |
| HeteroGraphSAGE | Relation IDs matter, but strict node-type schema validation is not required. | Typed edges with relation-aware aggregation. |
| HinSAGE | Node types and relation triples are part of the scientific design. | Typed nodes, typed relation triples, and relation-aware sampling. |

Direction is part of the contract for most taxi graphs. Do not collapse
`PULocationID -> DOLocationID` and `DOLocationID -> PULocationID` unless the
study explicitly assumes symmetry.

## Read Next

[Graph Models And Features](../../graph-features.md) remains the full contract
page for directionality, encoder details, artifacts, feature bundles, graph
regularization, and failure modes.
