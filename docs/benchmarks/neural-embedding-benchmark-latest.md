# Neural Embedding Benchmark

Repeated identifiers help the hybrid embedding model on this synthetic fixture;
held-out origin IDs can hurt it. Random, tail, and temporal splits show larger
MAE gains than the geographic split, so the validation design changes the
interpretation.

The run uses 2,000 generated rows and 128 ID cells. It is a diagnostic for ID
reuse and cold-start behavior, not a real NYC taxi quality comparison. See the
[neural model guides](../user-guide/neural-models/index.md) to choose a workflow.

## Reproduce

```sh
uv run python scripts/run_neural_embedding_benchmark.py \
  --output target/validation/neural_benchmark.json
```

## Data And Setup

| Field | Value |
| --- | ---: |
| Seed | 42 |
| Rows | 2,000 |
| Dense features | 8 |
| ID cells | 128 |
| Embedding dimension | 16 |
| Train fraction | 0.8 |

## Scenario Breakdown

| Scenario | ID key | Base MAE | Hybrid MAE | MAE improvement | Read |
| --- | --- | ---: | ---: | ---: | --- |
| Random | origin | 0.6741 | 0.4812 | +0.1929 | Strong repeated-ID gain. |
| Tail | origin | 0.6873 | 0.4551 | +0.2322 | Strong rare/tail gain in fixture. |
| Temporal blocked | origin | 0.6861 | 0.4570 | +0.2291 | Reused IDs over time help. |
| Geographic blocked | origin | 0.6724 | 0.6403 | +0.0320 | Smaller spatial gain. |
| Cold destination | destination | 0.6952 | 0.6887 | +0.0065 | Near tie. |
| Cold origin | origin | 0.7810 | 0.7967 | -0.0157 | Warning case: cold IDs can hurt. |

## Interpretation

This is exactly why neural rows must be split-specific. A random repeated-ID
gain does not imply cold-start generalization. Use neural residual embeddings
only when production IDs recur and the deployment split also improves.

