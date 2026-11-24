# Graph Models

Graph models use directed pickup-to-dropoff flows, neighborhood structure,
and different relation types to learn representations for regression or link
prediction.

## Choose A Guide

| What the graph tells you | Start with |
| --- | --- |
| Connectivity and flow patterns carry the signal | [Node2Vec](cartoboost-node2vec.md) |
| Node measurements should be combined with neighborhood information | [GraphSAGE](cartoboost-graphsage.md) |
| Different edge relations need different treatment, with one node-feature table | [HeteroGraphSAGE](cartoboost-hetero-graphsage.md) |
| Node types and allowed source/relation/target triples must be enforced | [HinSAGE](cartoboost-hinsage.md) |

Each guide covers regression and link prediction, a Python example, a browser
demonstration, and validation choices.

## Keep Direction And Time Explicit

A pickup-to-dropoff edge is different from its reverse. Build the graph from
information available before the validation cutoff, and keep held-out labels
or edges out of representation training. Report repeated-node performance
separately from cold-node and cold-edge results.

A graph is useful only if it adds information. Compare with a
[boosted tree](../boosting-models/index.md) on measured trip features and an
[ID embedding model](../neural-models/index.md) without adjacency.

For future values on a node-time matrix, use
[graph forecasting](../forecasting-models/graph-spatiotemporal.md). For graph
construction, input shapes, and persistence, see [Graph Features](../../graph-features.md).
