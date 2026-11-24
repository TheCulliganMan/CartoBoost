# Benchmarks

Compare CartoBoost with baseline models on documented datasets and splits. The reports below separate real NYC taxi comparisons from
controlled synthetic checks, with commands and artifacts for inspection.

Start with the task you care about:

- **Fare and duration:** the [NYC taxi comparison](nyc-taxi.md) includes random,
  spatial, and later-time holdouts. Results differ by target and split.
- **Pickup demand:** the [forecasting comparison](forecasting.md) includes a
  real 24-lane NYC panel. Seasonal naive is the strongest reported RMSE baseline.
- **Model mechanisms:** [taxi zone checks](taxi-zone.md),
  [neural embedding checks](neural-embedding-benchmark-latest.md), and
  [deep model checks](deep-claims.md) isolate specific behavior on synthetic data.

Use [Reading benchmark results](methodology.md) to interpret target scales,
validation splits, timing, and the limits of a comparison.

## Report Map

| Report | Evidence type | What to inspect first |
| --- | --- | --- |
| [Reading Benchmark Results](methodology.md) | Guide to interpreting the reports. | Target scale, holdout type, metric meaning, and timing. |
| [NYC Taxi Benchmarks](nyc-taxi.md) | Real TLC fare, duration, and pickup-demand regression. | Current-code CartoBoost versus external baselines, RMSE/MAE/R2/WAPE tables, timing breakdown. |
| [NYC Taxi Path C Claims](nyc-taxi-path-c.md) | Real TLC tests of geo-temporal behavior. | Directional, temporal, known-future, spatial-transfer, and residual-correction results. |
| [Forecasting Results](forecasting.md) | Real taxi demand, synthetic demand, M4 samples, M5 demand, and M6 return proxies. | Ranked model tables, metric-specific winners, run commands, and diagnostic plots. |
| [Model Benchmark Suite](model-suite.md) | Public tabular regression and graph diagnostics. | CartoBoost versus external baselines, validation-search selections, full RMSE/MAE/R2/WAPE tables. |
| [Deep Model Mechanism Checks](deep-claims.md) | Synthetic mechanism checks for deep models. | Six result rows, exact command, and JSON artifact. |
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

Compare the task and holdout with your deployment. A model that handles familiar
pickup zones may behave differently on a cold-zone split; a strong one-step
forecast may weaken over a longer horizon. Inspect the per-split rows before
using an aggregate score to choose a model.

The reports retain weaker results as well as wins. Synthetic fixtures show
controlled behavior, while real-data tables support conclusions only for their
stated samples, features, settings, and baselines.
