# Forecasting Tool Benchmark

For the deterministic forecasting architecture, see
[Forecasting Overhaul](../forecasting_overhaul.md). The benchmark command
surface accepts fixed no-hyperopt runs:

```sh
python scripts/forecasting_library_benchmark.py --suite committed --no-hyperopt
python scripts/forecasting_m4.py --committed --no-hyperopt
python scripts/forecasting_m5.py --committed --official-wrmsse --no-hyperopt
python scripts/forecasting_m6.py --committed --official-style --no-hyperopt
```

The commands above do not imply a benchmark win by themselves. Public claims
must cite the artifact, split, seed, model roster, and metric table. For M5 and
M6 claims, cite `official_metrics` in addition to the shared point metrics.

## Artifact Metric Contract

Every benchmark artifact keeps shared point metrics under `metrics`: RMSE, MAE,
MASE, WAPE, SMAPE, and bias. M5 and M6 artifacts also include
`official_metrics`:

| Source | Primary field | Contents | Claim boundary |
| --- | --- | --- | --- |
| `m5` | `official_metrics.m5` | Level-aware WRMSSE by total, state, store, item, and item-store levels, plus model rankings and per-series breakdowns. | Uses one-step RMSSE scaling and recent unit-sales volume weights because sell prices are not present in the shared benchmark frame. |
| `m6` | `official_metrics.m6` | Five-bucket rank-probability scores, per-asset probability rows, deterministic long/short decision rows, and model rankings. | Still an audit proxy, not an official M6 submission file. |

## Forecasting Overhaul Committed Run

Run date: June 19, 2026. These runs used the deterministic CartoBoost-only
roster, fixed seed 42, no hyperopt, and the committed benchmark sample settings.
Artifacts are committed under `docs/assets/nyc_taxi_benchmarks/`.

| Suite | Command | Artifact | Result |
| --- | --- | --- | --- |
| Real NYC taxi lane demand | `uv run --group dev python scripts/forecasting_library_benchmark.py --source nyc-taxi --year 2024 --months 1 --taxi-type yellow --lanes 24 --horizon 7 --no-download --model-roster cartoboost --output docs/assets/nyc_taxi_benchmarks/forecasting_library_benchmark_real.json --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_plots` | `forecasting_library_benchmark_real.json` | `cartoboost_lag` and `cartoboost_auto_forecast` tied: RMSE 67.846536, MAE 48.685209, WAPE 0.156442. |
| Synthetic committed suite | `uv run --group dev python scripts/forecasting_library_benchmark.py --suite committed --no-hyperopt --model-roster cartoboost --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_committed_suite.json` | `forecasting_overhaul_committed_suite.json` | `cartoboost_lag` ranked first by mean RMSE ratio, 1.061999 vs `cartoboost_auto_forecast` 1.077314. |
| Synthetic committed suite, full external roster | `uv run --group dev python scripts/forecasting_library_benchmark.py --suite committed --no-hyperopt --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_committed_suite_full_roster.json` | `forecasting_overhaul_committed_suite_full_roster.json` | `lightgbm_lag` ranked first by mean RMSE ratio at 1.069525. |
| M4 committed sample | `uv run --group dev python scripts/forecasting_m4.py --committed --no-hyperopt --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m4_committed.json` | `forecasting_overhaul_m4_committed.json` | `cartoboost_lag` ranked first with mean RMSE ratio 1.000000 and 6/6 wins-or-ties; `cartoboost_auto_forecast` mean ratio 1.146289 with 3/6 wins-or-ties. |
| M5 committed sample | `uv run --group dev python scripts/forecasting_m5.py --committed --official-wrmsse --no-hyperopt --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m5_committed.json` | `forecasting_overhaul_m5_committed.json` | `cartoboost_lag` won point metrics and WRMSSE: RMSE 2.455906, MAE 1.149696, WAPE 0.918936, WRMSSE 0.564643. `cartoboost_auto_forecast` WRMSSE was 0.706687. |
| M6 committed sample | `uv run --group dev python scripts/forecasting_m6.py --committed --official-style --no-hyperopt --output docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m6_committed.json` | `forecasting_overhaul_m6_committed.json` | `cartoboost_auto_forecast` won point metrics: RMSE 0.013865, MAE 0.008398, WAPE 1.143823. `cartoboost_lag` won RPS: 0.257276 vs auto 0.267355. |

Interpretation: the CartoBoost-only benchmark set does not support a blanket
auto-route win. `cartoboost_lag` wins or ties real taxi, synthetic, M4, and M5,
including M5 WRMSSE. `cartoboost_auto_forecast` wins M6 point RMSE/MAE/WAPE,
while `cartoboost_lag` wins the deterministic M6 RPS proxy.

## Competition Results Snapshot

| Competition | Artifact | Scope | Result |
| --- | --- | --- | --- |
| M5 committed CartoBoost sample | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m5_committed.json` | Committed M5-style sample, CartoBoost-only roster with auto and lag | `cartoboost_lag` won: RMSE 2.455906, MAE 1.149696, WAPE 0.918936, WRMSSE 0.564643. |
| M5 Forecasting Accuracy comparison sample | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json` | 100 item-store sample, 90 recent days, 28-day holdout, full 14-model roster without the auto alias | `statsforecast_autoets` won RMSE at 2.525734; CartoBoost lag RMSE was 2.678097. |
| M5 Forecasting Accuracy full-corpus check | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` | 30,490 item-store full-corpus check, 90 recent days, 28-day holdout, lag-only fast roster | CartoBoost lag RMSE 2.634879, MAE 1.332997, WAPE 0.923884. |
| M6 committed CartoBoost sample | `docs/assets/nyc_taxi_benchmarks/forecasting_overhaul_m6_committed.json` | Committed M6-style point proxy sample, CartoBoost-only roster with auto and lag | `cartoboost_auto_forecast` won point RMSE at 0.013865; `cartoboost_lag` won RPS at 0.257276. |
| M6 financial assets | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` | 100-symbol daily-return proxy, 28-day holdout, full 14-model roster without the auto alias | `statsforecast_autoarima` won RMSE at 0.013402; CartoBoost lag RMSE was 0.014348. |

## Bottom Line

The forecasting benchmark has maintained views for real taxi demand, synthetic
taxi-shaped diagnostics, committed M4/M5/M6 samples, and larger M5/M6
competition-style proxy runs:

- Real NYC taxi lane demand: CartoBoost-only plots include
  `cartoboost_auto_forecast`, which ties `cartoboost_lag`.
- Synthetic taxi-shaped committed suite: `cartoboost_lag` is the best
  CartoBoost method in the CartoBoost-only artifact, with mean RMSE
  ratio 1.061999 versus `cartoboost_auto_forecast` at 1.077314. The full
  external-roster artifact ranks `lightgbm_lag` first.
- M4 96-series-per-group committed sample: `cartoboost_lag` wins by mean RMSE
  ratio, 1.000000 versus `cartoboost_auto_forecast` at 1.146289. This is still a
  sample, not a full M4 corpus claim.
- M5 committed sample: `cartoboost_auto_forecast` is present in the committed
  artifact, but `cartoboost_lag` wins both point RMSE and WRMSSE.
- M6 committed sample: `cartoboost_auto_forecast` is present in the committed
  artifact and wins point RMSE/MAE/WAPE; `cartoboost_lag` wins the RPS
  proxy.
- M5 comparison sample: Kaggle M5 Accuracy files are now a first-class source
  with a full 14-model roster sample over 100 item-store daily unit-sales
  series, the official 28-day holdout shape, and the same model-family table
  style as the M4 sample.
- M5 full-corpus check: The June 19, 2026 run also completed over all 30,490
  bottom-level series using the public M5 mirror, a 90-day recent-history
  window, and a CartoBoost-only fast roster. That artifact is coverage and
  throughput evidence, not a model bakeoff.
- M6 full-run protocol: M6 assets are now a first-class source for a daily
  return point-forecast proxy over the public M6 asset panel. The June 19, 2026
  run completed over 100 symbols and 38,219 daily-return rows.

Forecasting claims should stay split-specific. The taxi run, committed
CartoBoost-only samples, and M5/M6 full-roster artifacts are separate pieces of
evidence. M5 and M6 are real competition panels, but the M5 model comparison
below is a bounded 100-series sample, not a full external-library bakeoff across
all 30,490 bottom-level series.

## Real NYC Taxi Lane Demand

| Field | Value |
| --- | --- |
| Source | NYC TLC yellow taxi trips |
| Period | January 2024 |
| Raw rows | 2,964,624 |
| Clean rows | 2,827,685 |
| Aggregated rows | 744 |
| Series | 24 pickup/dropoff lanes |
| Days | 31 |
| Horizon | 7 days |
| Static covariates | pickup zone, dropoff zone, distance, airport-lane flag, pickup borough code |

Reproduce:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source nyc-taxi \
  --year 2024 \
  --months 1 \
  --taxi-type yellow \
  --lanes 24 \
  --horizon 7 \
  --no-download \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_library_benchmark_real.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_plots
```

### Real Taxi Result

| Model | Library | RMSE | MAE | WAPE | Read |
| --- | --- | ---: | ---: | ---: | --- |
| `cartoboost_lag` | CartoBoost | 67.847 | 48.685 | 0.1564 | Tied best on the CartoBoost-only taxi run. |
| `cartoboost_auto_forecast` | CartoBoost | 67.847 | 48.685 | 0.1564 | Tied `cartoboost_lag`; present in the taxi plots. |

The taxi plots include `cartoboost_auto_forecast`. This table is CartoBoost-only;
use the full-roster artifact for external-library comparisons.

### Real Taxi Plots

![Forecasting metric comparison](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_tool_metric_comparison.png)

![Horizon RMSE by tool](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_horizon_rmse_by_tool.png)

![Forecast lines for top lanes](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_forecast_lines.png)

![Actual vs predicted demand](../assets/nyc_taxi_benchmarks/forecasting_plots/nyc-taxi_actual_vs_predicted.png)

## Synthetic Taxi-Shaped Suite

The synthetic suite uses four taxi-shaped forecasting problems: weekly lane
demand, airport calendar events, route-mix shifts, and borough monthly pulses.
It uses 12 series, 180 days, horizon 14, and 3 rolling-origin folds.

| Model | Mean RMSE ratio to problem best | Wins/ties | Top-3 finishes |
| --- | ---: | ---: | ---: |
| `cartoboost_lag` | 1.062 | 3 | 4 |
| `cartoboost_auto_forecast` | 1.077 | 1 | 4 |

The table above is the fixed CartoBoost-only committed artifact. The full
external-roster committed artifact ranks `lightgbm_lag` first at 1.069525. This
suite is a stress test and wiring check. It is not real TLC evidence.

## M4 Sample Suite

The maintained M4 artifact scores the first 96 series from each M4 group. It is
included to check non-taxi behavior and library interoperability.

| Model | Mean RMSE ratio to group best | Wins/ties | Top-3 finishes |
| --- | ---: | ---: | ---: |
| `cartoboost_lag` | 1.000 | 6 | 6 |
| `cartoboost_auto_forecast` | 1.146 | 3 | 6 |

### M4 Model RMSE

M4 groups have different scales, so the ratio table above is the better
cross-group comparison. The table below reports the arithmetic mean of each
model's group RMSE from the same committed CartoBoost-only artifact.

| Model | Mean RMSE | Mean RMSE ratio to group best |
| --- | ---: | ---: |
| `cartoboost_lag` | 385.707610 | 1.000000 |
| `cartoboost_auto_forecast` | 486.094303 | 1.146289 |

This is a committed CartoBoost-only M4 sample, not a full M4 claim. Use
`--m4-series-limit 0` and include the external roster before making full-corpus
or cross-library M4 statements.

## M5 Full Competition Run

The M5 loader consumes Kaggle M5 Forecasting Accuracy files directly when they
exist locally. When downloads are allowed and those files are absent, it uses
Nixtla's public M5 mirror and records the mirror URL in the artifact. It
hard-fails under `--no-download` when the required files are missing.

| Field | Value |
| --- | --- |
| Source | Kaggle M5 Forecasting Accuracy files |
| Required files | `calendar.csv` and `sales_train_evaluation.csv` or `sales_train_validation.csv`; public mirror download is used when local files are absent and downloads are allowed |
| Series | Full bottom-level item-store corpus when `--m5-series-limit 0` is used |
| Split | Last 28 days from the supplied training/evaluation file |
| Horizon | 28 daily steps |
| Scoring | Shared point metrics plus `official_metrics.m5` |
| Maintained comparison artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json` |
| Maintained full-corpus artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` |

Reproduce the maintained full-roster comparison sample:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m5 \
  --model-roster full \
  --m5-data-dir data/forecasting_benchmarks/m5 \
  --m5-series-limit 100 \
  --m5-history-days 90 \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_plots
```

Reproduce the maintained full-corpus CartoBoost fast run:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m5 \
  --model-roster cartoboost \
  --m5-data-dir data/forecasting_benchmarks/m5 \
  --m5-series-limit 0 \
  --m5-history-days 90 \
  --cartoboost-n-estimators 1 \
  --cartoboost-max-depth 3 \
  --cartoboost-min-samples-leaf 20 \
  --no-candidate-selection \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m5_plots
```

Use a positive `--m5-series-limit` only for smoke tests. Results from limited
runs must be labeled as samples, not as M5 full-corpus evidence.

### M5 Full-Roster Sample Result

Command run on June 19, 2026:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m5 \
  --model-roster full \
  --m5-data-dir data/forecasting_benchmarks/m5 \
  --m5-series-limit 100 \
  --m5-history-days 90 \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_plots
```

| Field | Value |
| --- | --- |
| Artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_sample.json` |
| Plots | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full_roster_plots/` |
| Source files | `data/forecasting_benchmarks/m5/datasets/sales_train_evaluation.csv`, `calendar.csv` |
| Mirror URL | `https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip` |
| Series | 100 item-store series from 30,490 available |
| Rows | 9,000 daily unit-sales rows |
| Days materialized | 90 recent days from 1,941 available days |
| Horizon | 28 daily steps |
| Roster | `full`: CartoBoost, functime, StatsForecast, Prophet, XGBoost lag, LightGBM lag |
| Candidate selection | Enabled, shared one-origin calibration |
| Total runtime | 93.734 seconds |
| Winner | `statsforecast_autoets` |
| Best RMSE | 2.525734 |
| CartoBoost RMSE | 2.678097 |
| CartoBoost MAE | 1.187821 |
| CartoBoost WAPE | 0.958196 |

### M5 Model RMSE

The table below reports every model present in the M5 full-roster sample
artifact, ranked by RMSE. That artifact predates the `cartoboost_auto_forecast`
alias; the committed CartoBoost-only table above is the M5 artifact that
includes auto.

The committed CartoBoost-only M5 overhaul artifact contains the auto route. That
run is separate from the full external-roster sample below. Use
`official_metrics.m5` for WRMSSE.

| Committed CartoBoost model | Artifact | RMSE | MAE | WAPE | Read |
| --- | --- | ---: | ---: | ---: | --- |
| `cartoboost_lag` | `forecasting_overhaul_m5_committed.json` | 2.455906 | 1.149696 | 0.918936 | Best point metrics and best WRMSSE, 0.564643. |
| `cartoboost_auto_forecast` | `forecasting_overhaul_m5_committed.json` | 2.530322 | 1.210252 | 0.967338 | WRMSSE 0.706687. |

| Model | RMSE | MAE | WAPE |
| --- | ---: | ---: | ---: |
| `statsforecast_autoets` | 2.525734 | 1.141999 | 0.921232 |
| `statsforecast_dynamic_optimized_theta` | 2.556517 | 1.163750 | 0.938779 |
| `statsforecast_autotbats` | 2.602055 | 1.156588 | 0.933001 |
| `functime_ridge` | 2.606775 | 1.207878 | 0.974376 |
| `statsforecast_autotheta` | 2.607077 | 1.196042 | 0.964828 |
| `statsforecast_autoarima` | 2.655754 | 1.194312 | 0.963433 |
| `cartoboost_lag` | 2.678097 | 1.187821 | 0.958196 |
| `functime_snaive` | 2.678097 | 1.187821 | 0.958196 |
| `functime_lightgbm` | 2.678097 | 1.187821 | 0.958196 |
| `statsforecast_seasonal_naive` | 2.678097 | 1.187821 | 0.958196 |
| `statsforecast_autoces` | 2.678097 | 1.187821 | 0.958196 |
| `prophet_additive` | 2.678097 | 1.187821 | 0.958196 |
| `xgboost_lag` | 2.793477 | 1.500446 | 1.210386 |
| `lightgbm_lag` | 2.825295 | 1.253991 | 1.011575 |

This sample does not support a claim that CartoBoost beats the M5 model
families; `cartoboost_lag` ranks seventh by RMSE. The all-series M5 artifact is
a lag-only CartoBoost full-corpus check:

| Field | Value |
| --- | --- |
| Artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m5_full.json` |
| Scope | 30,490 item-store series, 2,744,100 rows, 90 recent days, 28-day holdout |
| Roster | Legacy lag-only CartoBoost fast run |
| Candidate selection | Disabled |
| CartoBoost settings | 1 estimator, learning rate 0.06, max depth 3, min samples leaf 20 |
| Total runtime | 21.941 seconds |
| CartoBoost RMSE | 2.634879 |
| CartoBoost MAE | 1.332997 |
| CartoBoost WAPE | 0.923884 |

## M6 Full Competition Proxy Run

The M6 loader consumes the public M6 methods `assets_m6.csv` file with
`symbol`, `date`, and `price` columns. If the file is missing, the runner fetches
it from the M6 methods repository unless `--no-download` is set.

| Field | Value |
| --- | --- |
| Source | M6 methods asset price panel |
| Required file | `assets_m6.csv` |
| Series | Every symbol in the assets file when `--m6-series-limit 0` is used |
| Transform | Calendar-daily adjusted-price panel with forward-filled non-trading days and daily returns as the target |
| Split | Last `--m6-horizon` calendar days from the daily return panel |
| Default horizon | 28 daily steps |
| Scoring | Shared point metrics plus `official_metrics.m6` |
| Maintained artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` |

Reproduce:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m6 \
  --m6-assets-path data/forecasting_benchmarks/m6/assets_m6.csv \
  --m6-series-limit 0 \
  --m6-horizon 28 \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m6_plots
```

### M6 Result

Command run on June 19, 2026:

```sh
uv run --group bench python scripts/forecasting_library_benchmark.py \
  --source m6 \
  --m6-assets-path data/forecasting_benchmarks/m6/assets_m6.csv \
  --m6-series-limit 0 \
  --m6-horizon 28 \
  --output docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json \
  --plot-dir docs/assets/nyc_taxi_benchmarks/forecasting_m6_plots
```

| Field | Value |
| --- | --- |
| Artifact | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_full.json` |
| Plots | `docs/assets/nyc_taxi_benchmarks/forecasting_m6_plots/` |
| Source URL | `https://raw.githubusercontent.com/Mcompetitions/M6-methods/main/assets_m6.csv` |
| Series | 100 symbols |
| Rows | 38,219 daily-return rows |
| Days | 383 calendar days |
| Horizon | 28 calendar days |
| Roster | `full`: CartoBoost, functime, StatsForecast, Prophet, XGBoost lag, LightGBM lag |
| Candidate selection | Enabled, shared one-origin calibration |
| Total runtime | 173.591 seconds |
| Winner | `statsforecast_autoarima` |
| Best RMSE | 0.013402 |
| CartoBoost RMSE | 0.014348 |
| CartoBoost MAE | 0.009357 |
| CartoBoost WAPE | 1.271868 |
| CartoBoost RMSE ratio vs best | 1.070585 |

| Rank | Model | Read |
| ---: | --- | --- |
| 1 | `statsforecast_autoarima` | Best RMSE on this daily-return proxy. |
| 2 | `statsforecast_autoets` | Second by RMSE. |
| 3 | `functime_ridge` | Best non-StatsForecast model by RMSE. |
| 4 | `statsforecast_autotbats` | Strong but slower than simpler baselines. |
| 14 | `cartoboost_lag` | 7.1% higher RMSE than the best forecasting-library model. |

### M6 Model RMSE

The table below reports every model present in the M6 full-run proxy artifact,
ranked by RMSE. That artifact predates the `cartoboost_auto_forecast` alias; the
committed CartoBoost-only table above is the M6 artifact that includes auto.

The committed CartoBoost-only M6 overhaul artifact contains the auto route on
the point-forecast proxy harness. Use `official_metrics.m6` for RPS:

| Committed CartoBoost model | Artifact | RMSE | MAE | WAPE | Read |
| --- | --- | ---: | ---: | ---: | --- |
| `cartoboost_auto_forecast` | `forecasting_overhaul_m6_committed.json` | 0.013865 | 0.008398 | 1.143823 | Best point metrics; RPS 0.267355. |
| `cartoboost_lag` | `forecasting_overhaul_m6_committed.json` | 0.014440 | 0.009290 | 1.265338 | Best RPS, 0.257276. |

| Model | RMSE | MAE | WAPE |
| --- | ---: | ---: | ---: |
| `statsforecast_autoarima` | 0.013402 | 0.007400 | 1.005844 |
| `statsforecast_autoets` | 0.013408 | 0.007456 | 1.013524 |
| `functime_ridge` | 0.013474 | 0.007670 | 1.042553 |
| `statsforecast_autotbats` | 0.013477 | 0.007663 | 1.041580 |
| `statsforecast_autoces` | 0.013522 | 0.007617 | 1.035289 |
| `statsforecast_dynamic_optimized_theta` | 0.013669 | 0.008204 | 1.115187 |
| `functime_snaive` | 0.013674 | 0.007500 | 1.019396 |
| `functime_lightgbm` | 0.013674 | 0.007500 | 1.019396 |
| `statsforecast_seasonal_naive` | 0.013674 | 0.007500 | 1.019396 |
| `prophet_additive` | 0.013674 | 0.007500 | 1.019396 |
| `lightgbm_lag` | 0.013674 | 0.007500 | 1.019396 |
| `statsforecast_autotheta` | 0.013683 | 0.008228 | 1.118462 |
| `xgboost_lag` | 0.014246 | 0.008896 | 1.209160 |
| `cartoboost_lag` | 0.014348 | 0.009357 | 1.271868 |

This is intentionally named a proxy run. The official M6 competition combined
rank-probability forecasts and investment decisions, with RPS and investment
return as official scoring dimensions. The artifact includes deterministic
rank-probability scoring and decision rows, but those rows are audit evidence
rather than an official M6 submission file.

## Limitations

- Interval coverage is only benchmark evidence when actually computed.
- Real taxi panel is short: 31 days and a 7-day holdout.
- Synthetic suite is diagnostic.
- M4 artifact is a 96-series-per-group sample.
- M5 full-roster evidence is a 100-series sample with a 90-day materialized
  history window; the full 30,490-series artifact is a lag-only CartoBoost fast
  run.
- M6 artifacts include deterministic RPS and decision rows on top of the daily
  point-forecast proxy, but they are not official leaderboard submission
  payloads.
- External model availability depends on optional benchmark dependencies.
