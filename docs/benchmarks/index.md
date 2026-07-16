# Benchmarks

These reports show where CartoBoost has been measured, what the comparisons
mean, and where the evidence is still limited. Start with the result table for
the task closest to your own, then inspect its data, split, features, command,
runtime, and artifact before transferring the conclusion.

Each report answers the same reader questions:

- What data was used?
- What split was used?
- Which models saw which features?
- What command produced the artifact?
- What did the metric table say?
- Which plots should I inspect?
- What claim is allowed, and what claim is not allowed?

CartoBoost is evaluated on taxi-shaped structure: pickup and dropoff zones,
route distance, periodic hour/day effects, repeated IDs, pickup/dropoff
topology, and lane-demand history. The benchmark docs separate real NYC TLC
evidence from synthetic mechanism checks and report mixed results when external
baselines are lower on a task.

The maintained evidence covers NYC TLC zone and lane tasks, graph forecasting,
air-quality interpolation, housing sanity checks, synthetic spatial fields,
synthetic graph diffusion, and synthetic geo-causal panels. Real datasets
support task-specific quality comparisons; synthetic datasets demonstrate
mechanisms and failure modes only.

## Report Map

| Report | Evidence type | What to inspect first |
| --- | --- | --- |
| [Benchmark Methodology](methodology.md) | Shared evaluation protocol. | Data identity, split rules, metric roster, timing, and reproducibility. |
| [NYC Taxi Benchmarks](nyc-taxi.md) | Real TLC fare, duration, and pickup-demand regression. | Current-code CartoBoost versus external baselines, RMSE/MAE/R2/WAPE tables, timing breakdown. |
| [NYC Taxi Path C Claims](nyc-taxi-path-c.md) | Real TLC tests of geo-temporal behavior. | Directional, temporal, known-future, spatial-transfer, and residual-correction results. |
| [Forecasting Tool Benchmark](forecasting.md) | Real taxi lane demand, synthetic taxi-shaped forecasting, M4 sample, M5 full-roster sample, and M5/M6 full-run protocols. | RMSE/WAPE tables, M5/M6 model rosters, run commands, horizon plot, forecast-line plot. |
| [Model Benchmark Suite](model-suite.md) | Public tabular regression and graph diagnostics. | CartoBoost versus external baselines, validation-search selections, full RMSE/MAE/R2/WAPE tables. |
| [Deep Claim Benchmarks](deep-claims.md) | Synthetic mechanism checks for deep models. | Seven result rows, exact command, and JSON artifact. |
| [Taxi Zone Acceptance](taxi-zone.md) | Deterministic taxi-lane feature acceptance. | Lane heatmap, hour profile, route midpoint geometry. |
| [Neural Embedding Benchmark](neural-embedding-benchmark-latest.md) | Synthetic repeated-ID/cold-ID diagnostic. | Scenario table showing random/tail gains and cold-origin failure. |

## Current Maintained Artifacts

| Artifact | Path |
| --- | --- |
| NYC regression JSON | `docs/assets/nyc_taxi_benchmarks/results.json` |
| NYC regression JSONL metrics | `docs/assets/nyc_taxi_benchmarks/results.jsonl` |
| NYC regression report | `docs/assets/nyc_taxi_benchmarks/results.md` |
| NYC Path C claims JSON | `docs/assets/nyc_taxi_benchmarks/path_c_claims.json` |
| NYC Path C claims JSONL | `docs/assets/nyc_taxi_benchmarks/path_c_claims.jsonl` |
| NYC Path C claims report | `docs/assets/nyc_taxi_benchmarks/path_c_claims.md` |
| NYC repeated regression JSON | `docs/assets/nyc_taxi_benchmarks/repeated_results.json` |
| NYC repeated regression report | `docs/assets/nyc_taxi_benchmarks/repeated_results.md` |
| NYC forecasting JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_library_benchmark_real.json` |
| Forecasting suite JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_committed_suite.json` |
| Forecasting full-roster suite JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_committed_suite_full_roster.json` |
| M4 forecasting JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m4_committed.json` |
| M5 forecasting JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m5_committed.json` |
| M6 forecasting JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m6_committed.json` |
| Synthetic forecasting suite JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_library_suite_synthetic.json` |
| M4 sample suite JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_m4_suite_sample.json` |
| M5 full-roster sample JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json` |
| M5 full forecasting JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` |
| M6 full forecasting JSON | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` |
| Model diagnostic suite JSON | `docs/assets/model_benchmarks_public/results.json` |
| Model diagnostic suite JSONL metrics | `docs/assets/model_benchmarks_public/results.jsonl` |
| Model diagnostic suite aggregate JSON | `docs/assets/model_benchmarks_public/results_aggregate.json` |
| Model diagnostic suite report | `docs/assets/model_benchmarks_public/results.md` |
| Lane acceptance JSON | `docs/assets/lane_level_tests/acceptance_metrics.json` |
| Deep claim benchmark JSON | `docs/assets/deep_claim_benchmarks/results.json` |
| Deep claim benchmark JSONL | `docs/assets/deep_claim_benchmarks/results.jsonl` |
| Deep claim benchmark report | `docs/assets/deep_claim_benchmarks/results.md` |

## How To Read The Results

A result is usable as benchmark evidence when it names the dataset, command,
split, feature policy, models, metrics, and artifact path. It is only a public
quality claim when the comparison uses complete required baselines, same rows,
comparable feature access, no test-set peeking, equal tuning budget, and
uncertainty or repeatability evidence.

Random splits show interpolation. Spatial, grouped, cold-ID, or out-of-time
splits are the evidence for deployment risk. Synthetic fixtures are useful for
debugging and feature acceptance, not for real-world superiority claims.
