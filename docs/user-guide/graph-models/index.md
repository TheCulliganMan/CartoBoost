# Graph Models

Use graph models when relationship structure is the thing you want to test.
That usually means directed origin-to-destination flow, repeated route
markets, neighborhood structure, or link likelihood.

Use [CartoBoost Boosting Model Guides](../boosting-models/index.md) for
row-level trees and [CartoBoost Neural Model Guides](../neural-models/index.md)
when repeated IDs should be embedded without an explicit graph.

## Choose A Guide

| Guide | Best when |
| --- | --- |
| [CartoBoost Node2Vec Graph Models](cartoboost-node2vec.md) | Topology and flow patterns matter more than node attributes. |
| [CartoBoost GraphSAGE Models](cartoboost-graphsage.md) | Node attributes should shape the learned representation. |
| [CartoBoost HeteroGraphSAGE Models](cartoboost-hetero-graphsage.md) | Relation IDs matter, but a single node-feature table is enough. |
| [CartoBoost HinSAGE Models](cartoboost-hinsage.md) | Node types and valid source-relation-target combinations must be enforced. |

Keep direction explicit. A source-to-target fact is usually not the same as
the reverse fact.

See [Graph Features](../../graph-features.md) for input shapes, directionality,
generated features, saving and loading, and common failure modes.
