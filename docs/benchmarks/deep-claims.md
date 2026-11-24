# Deep Model Mechanism Checks

All six listed models improve on their named baseline in these controlled
synthetic experiments. Each fixture isolates one intended mechanism, such as
directional pair effects, delayed graph propagation, or candidate competition.

The results describe designed patterns. They do not measure accuracy on real
taxi trips or establish production readiness for experimental models.

## Results

| Claim | Architecture | Metric | Model | Baseline | Improvement | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `pair_embedding_mlp_nonlinear_directional_pair` | `pair_embedding_mlp` | RMSE | 0.515210 | 1.710122 | 69.87% | passed |
| `inverted_transformer_cross_entity_panel` | `inverted_transformer` | holdout RMSE | 0.557217 | 1.461201 | 61.87% | passed |
| `delay_aware_graph_transformer_directional_delay` | `delay_aware_graph_transformer` | holdout RMSE | 0.486446 | 0.545256 | 10.79% | passed |
| `regime_moe_mixed_regime_data` | `regime_moe` | RMSE | 0.000003 | 0.583261 | 100.00% | passed |
| `conditional_flow_distribution_head_calibration_sharpness` | `conditional_residual_sampler` | calibration/sharpness score | 0.000000 | 0.269389 | 100.00% | passed |
| `choice_set_utility_softmax_candidate_competition` | `choice_set_utility_softmax` | choice log loss | 0.022234 | 0.693147 | 96.79% | passed |

## Reproduce

```bash
PYTHONPATH=python python scripts/run_deep_claim_benchmarks.py --output docs/assets/deep_claim_benchmarks/results.json
PYTHONPATH=python python scripts/check_deep_claim_gates.py
```

## Artifacts

| Artifact | Path |
| --- | --- |
| JSON summary | `docs/assets/deep_claim_benchmarks/results.json` |
| JSONL rows | `docs/assets/deep_claim_benchmarks/results.jsonl` |
| Markdown table | `docs/assets/deep_claim_benchmarks/results.md` |

Each JSON/JSONL row records `claim_id`, `architecture`, `capability_tier`,
`implementation_backend`, `falsifier_baseline`, `dataset_hash`, `split_hash`,
`seed`, `primary_metric`, `improvement_threshold`, `percent_improvement`,
`fit_seconds`, `predict_seconds`, `peak_memory_mb`, `save_load_max_abs_diff`,
`leakage_policy`, and `experimental_status`.

## Interpretation and limits

Every listed model recovers its controlled pattern better than the comparison
baseline. The fixtures are small and deterministic, with data designed around
the mechanism being tested. Use the [real NYC taxi comparisons](nyc-taxi.md)
for evidence on observed trips, and the [deep model guides](../user-guide/deep-models.md)
for input contracts and model-specific limitations.
