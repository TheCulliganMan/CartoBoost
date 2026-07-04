# Deep Claim Benchmarks

These results are deterministic synthetic claim checks for exported deep model
surfaces. They are not real-data evidence. The purpose is to prevent API guide
claims from outrunning the current implementation.

Command:

```bash
PYTHONPATH=python python scripts/run_deep_claim_benchmarks.py --output docs/assets/deep_claim_benchmarks/results.json
PYTHONPATH=python python scripts/check_deep_claim_gates.py
```

Artifacts:

| Artifact | Path |
| --- | --- |
| JSON summary | `docs/assets/deep_claim_benchmarks/results.json` |
| JSONL rows | `docs/assets/deep_claim_benchmarks/results.jsonl` |
| Markdown table | `docs/assets/deep_claim_benchmarks/results.md` |

| Claim | Architecture | Metric | Model | Baseline | Improvement | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `pair_embedding_mlp_nonlinear_directional_pair` | `pair_embedding_mlp` | RMSE | 0.526011 | 22.372103 | 97.65% | passed |
| `selective_ssm_lite_long_memory_panel` | `selective_ssm_lite` | rolling-origin RMSE | 0.503747 | 0.871346 | 42.19% | passed |
| `inverted_transformer_cross_entity_panel` | `inverted_transformer` | holdout RMSE | 0.507959 | 1.461201 | 65.24% | passed |
| `delay_aware_graph_transformer_directional_delay` | `delay_aware_graph_transformer` | holdout RMSE | 0.486446 | 0.545256 | 10.79% | passed |
| `regime_moe_mixed_regime_data` | `regime_moe` | RMSE | 0.000042 | 0.583261 | 99.99% | passed |
| `retrieval_augmented_forecaster_rare_pattern` | `retrieval_augmented_forecaster` | RMSE | 0.522673 | 7.444953 | 92.98% | passed |
| `conditional_flow_distribution_head_calibration_sharpness` | `conditional_residual_sampler` | calibration/sharpness gate | 0.000000 | 0.269389 | 100.00% | passed |
| `choice_set_utility_softmax_candidate_competition` | `choice_set_utility_softmax` | choice log loss | 0.022234 | 0.693147 | 96.79% | passed |

Each JSON/JSONL row records `claim_id`, `architecture`, `capability_tier`,
`implementation_backend`, `falsifier_baseline`, `dataset_hash`, `split_hash`,
`seed`, `primary_metric`, `improvement_threshold`, `percent_improvement`,
`fit_seconds`, `predict_seconds`, `peak_memory_mb`, `save_load_max_abs_diff`,
`leakage_policy`, and `experimental_status`.

Interpretation: the passing synthetic gates support the listed synthetic claim
evidence labels in the deep model guide and capability matrix. They do not
promote experimental surfaces to real-data evidence.

Limitations: these fixtures are small, deterministic, and designed to isolate
the advertised mechanism. Real deployment claims still require a dataset-specific
split, serious baselines, timing, and recorded artifacts under the benchmark
standard used elsewhere in CartoBoost.
