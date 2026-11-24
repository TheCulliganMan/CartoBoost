# CartoBoost input, batch, and model audit — 2026-11-24

This follow-up covers the Python input boundary, model inference and training,
forecasting compatibility, serialization, and the shared Rust implementations.
The timing baseline is a frozen copy of the local Python package and native
extension after the preceding pass, based on Git revision
`d223db7414664b57e74da4a74f1627a5950074b2`. The changes are local.
[Preceding workspace audit](workspace-audit-2026-11-24.md)

## Correctness

- Native NumPy inputs can be C-contiguous, Fortran-contiguous, or strided. The
  native adapter reads logical row order and borrows memory in the supported
  contiguous layout. Previously a Fortran buffer passed the contiguous check
  and could silently change the observations.
- Dense numeric DataFrames bypass object conversion. Mixed categorical tables
  retain original categories, sparse sets retain high-cardinality signals, and
  nullable tables preserve missing values through validation. Label and entity
  ID adapters preserve integers above `2**53` even in mixed signed/unsigned
  tables that would otherwise promote to floating point.
- Tabular tests compare singleton and batch outputs and input layouts, including
  class labels, probabilities, margins, rankings, additive predictions, and
  SHAP contributions. Grouped and graph outputs retain the entire context they
  require. A graph forecast or choice set cannot be split into independent rows.
- Prophet compatibility preserves subsecond timestamps and forecast row order,
  logistic cap/floor and trend units, holiday activation, regressor prior scales,
  and explicit standardization. The sample path retains the requested 1,000
  uncertainty draws. Repeated access to history uses fitted diagnostics cached
  by model identity.
- Forecast snapshots retain mixed DataFrame IDs, times, and covariates. Recursive
  lag forecasts reuse exact EWM state and validate the input history once per
  call. Nearest-neighbor GP evaluates the covariance triangle consumed by the
  solver. Independent covariance and lag-recurrence oracles exercise both paths.
- RegimeMoE learns the reference mean and median at fit time and uses explicitly
  supplied recent residuals. The earlier fit used target-only residuals and
  predictions depended on the other rows in a batch. The browser report calls
  the same Rust model and calculates real diagnostics. Older artifacts and
  pickles still load with a warning and reproduce their previous behavior.
  Refit an old model to use the corrected semantics.
- Continuous response scoring now uses the continuous score field. Constrained
  decisions match candidate IDs within their decision group. The temporal models
  run real rolling-origin refits and return quantile forecasts from fitted
  residual scales. The inverted model’s native least-squares fit agrees with
  NumPy on full-rank, deficient, and underdetermined fixtures. Older inverted
  artifacts still predict and need refitting to estimate residual quantiles.
- Spatial lag, Durbin, and two-stage models reuse bounded, exact solver work on
  repeated calls with the same network. A change in weights invalidates reuse.
  Tests compare predictions against the original computation and independently
  solve `(I - rho W) prediction = structural_mean` on changed networks.

## Measurements

Runs used Python 3.12.13, NumPy 1.26.4, and a 10-core Apple M5 on macOS ARM64.
Measurements use generated inputs with fixed settings and no training or
prediction simplifications between variants. Timings are steady medians unless
marked cold. RSS is the process peak including setup and imports. These inputs
measure runtime and numerical behavior. Forecast quality on held-out real data
requires separate evidence.

The initial input and Prophet pass completed 600 paired workloads across 20
main operations plus auxiliary, sparse/categorical, and Prophet operations.
The main matrix spans NumPy, float32, Fortran, strided, Pandas, Polars, DuckDB,
and lists, from one to one million rows. The before/after comparison checks
output summaries and numerical parity. The 35 core Prophet runs preserved all
fitted artifacts byte for byte and forecast, component, and sample values to
`1e-12`. The 45 facade cases also compared output columns and timestamps.

| Operation | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Pandas regression prediction, 1 row | 12.11 µs | 2.51 µs | 4.82× |
| Pandas numeric fit input parsing, 1M × 8 | 254.465 ms | 1.602 ms | 158.8× |
| Pandas class probabilities, 1M rows | 240.762 ms | 87.220 ms | 2.76× |
| Pandas ranking prediction, 1M rows | 114.486 ms | 88.941 ms | 1.29× |
| Pandas regression fit, 1M rows, 4 trees, 4 threads | 363.9 ms | 127.5 ms | 2.85× |
| Point forecast metrics, 1M rows | 2843.539 ms | 6.795 ms | 418× |
| Event outcome prediction, Pandas, 65,536 rows | 114.77 ms | 4.681 ms | 24.5× |
| Prophet fit, history 16,384 | 105.65 ms | 80.58 ms | 1.31× |
| Prophet future forecast, history 16,384 and horizon 128 | 16.45 ms | 5.34 ms | 3.08× |
| Prophet predictive samples, same history/horizon, 1,000 draws | 424.78 ms | 6.31 ms | 67.27× |
| Prophet history and future, same history/horizon, warm | 1124.38 ms | 38.21 ms | 29.4× |

Three fresh-process trials gave a median speedup of **4.50× cold** and
**19.82× warm** for history 4,096 plus horizon 128, and **4.78× cold** and
**25.03× warm** for history 16,384 plus horizon 128. Million-row numeric
regression prediction was close to unchanged. Some operation/format combinations
are limited by model compute, memory layout, or timing variation. The ranking
scheduler was checked separately at 32 rows and 128 trees because parallelizing
small requests had made them slower.

The mix-of-experts fixture covered fit, predict, and components with NumPy and
Pandas, one to 65,536 query rows. Singleton NumPy predict was **20.85×** faster,
65,536-row NumPy predict **2.62×** faster, and 4,096-row NumPy fit **9.36×** faster.
The maximum singleton/batch difference fell from 1.766 to zero. Fitted-model
reported RMSE agreed with independently scored predictions to `5e-17`.

The maintained `scripts/benchmark_input_batch_paths.py --help` describes the
operation, dimensions, layouts, and baseline options for repeatable comparisons.
The original detailed logs and fixture files are indexed below.

### Neural, graph, and spatial comparisons

All 168 neural and graph cases matched the frozen result byte for byte. The
12 cases per model cover 1, 32, 1,024, and 65,536 query rows or graph nodes,
with NumPy, Fortran, and Pandas inputs. The table shows the median ratio
across cases and the full observed range. Node2Vec encoding operates on
its fitted graph and has no prediction input to reformat.

| Model | Median speedup | Range |
| --- | ---: | ---: |
| Embedding features | 4.24× | 0.63–39.95× |
| Embedding regressor | 1.61× | 0.98–2.96× |
| Node2Vec regressor | 1.68× | 1.10–2.92× |
| GraphSAGE regressor | 1.48× | 0.98–3.14× |
| Heterogeneous GraphSAGE regressor | 1.41× | 0.99–2.95× |
| HinSAGE regressor | 1.45× | 1.24–3.14× |
| Node2Vec link predictor | 6.33× | 1.23–40.86× |
| GraphSAGE link predictor | 2.33× | 0.80–31.40× |
| Heterogeneous GraphSAGE link predictor | 1.76× | 1.04–29.70× |
| HinSAGE link predictor | 2.36× | 1.04–28.97× |
| Node2Vec encoder | 10.90× | 1.71–16.19× |
| GraphSAGE encoder | 1.92× | 0.83–5.01× |
| Heterogeneous GraphSAGE encoder | 1.89× | 1.10–4.97× |
| HinSAGE encoder | 1.87× | 1.17–5.18× |

Spatial comparisons covered four fit cases and 48 predict/score cases. Point
predictions matched exactly and scores agreed within `1e-12`. Repeated prediction
on a fixed 512-node NumPy network gave the following medians. The first call still
prepares the same system and only saw small changes for the lag-based models.

| Model | Warm before ms | Warm after ms | Warm speedup | Cold before → after ms |
| --- | ---: | ---: | ---: | ---: |
| SpatialLagRegressor | 11.201 | 0.153 | 73.35× | 10.965 → 10.301 |
| SpatialErrorRegressor | 0.055 | 0.012 | 4.49× | 0.059 → 0.016 |
| SpatialDurbinRegressor | 10.438 | 0.178 | 58.61× | 10.503 → 10.278 |
| SpatialTwoStageLeastSquares | 11.064 | 0.141 | 78.70× | 10.884 → 10.475 |

### Forecasting, deep, and causal comparisons

The forecast family runner completed 52 paired cases across 40 Python model
configurations, including two backtests and additional long-history or large
spatial queries. It compared values, series, timestamps, and metadata, with
`1e-10` numeric tolerance and the fit wall-clock metadata excluded. Some paper
graph models used an explicit two-epoch profile on both sides. The LSTTN family
used lookback 48, recent window 12, periodicity 6, two epochs, and horizon 3.
The default four-week LSTTN training profile was outside this timing run.

| Forecast operation | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Lag, 8 × 4,096 history, 168 steps | 4.495 ms | 2.554 ms | 1.76× |
| Lag with EWM, same dimensions | 9.221 ms | 2.708 ms | 3.40× |
| NNGP mean and variance, 10,000 train / 1,024 query, 32 neighbors | 5.019 ms | 3.998 ms | 1.26× |
| Residual NNGP, same dimensions | 4.661 ms | 3.977 ms | 1.17× |
| Lag with EWM rolling backtest, 3 folds | 40.276 ms | 39.492 ms | 1.02× |

Some small predictions and fits varied or were slower. In repeated 65,536-row
ranker fits, NumPy was close to unchanged and Pandas speedups varied from 0.56×
to 0.94×. Ranker fit improved on the separately sized one-million-row case.

The final deep comparison covered 81 operations across 11 implementation
families, including continuous/binary response and all four directional
architectures. At fixed settings, shape, group, record order, strings, and
numerical comparisons passed (`rtol=atol=1e-9`). The median speedup across
cases was 3.00×. All 12 causal comparisons had identical output hashes.

| Operation and fixture | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Conditional flow point prediction, 65,536 rows | 1233.333 ms | 16.655 ms | 74.05× |
| Service residual prediction, 65,536 rows | 137.679 ms | 18.650 ms | 7.38× |
| Binary response prediction, 65,536 rows | 150.644 ms | 28.795 ms | 5.23× |
| Choice-set scoring, 65,536 candidates | 163.785 ms | 95.119 ms | 1.72× |
| Graph neural operator, 1,024 nodes and 32 history steps | 23.447 ms | 0.601 ms | 39.04× |
| Inverted temporal forecast, 32 entities and horizon 128 | 0.824 ms | 0.076 ms | 10.90× |
| Causal panel construction, 32,768 Pandas rows and 32 unused columns | 77.722 ms | 32.853 ms | 2.37× |
| Causal representation report, 65,536 Pandas rows | 158.604 ms | 104.339 ms | 1.52× |

The stable temporal fixture uses correlated linear histories. A separate
random/sinusoidal stress fixture reproduced large long-horizon extrapolations
in the original and migrated inverted model. The native migration preserves
these extrapolation semantics. Synthetic DID fitting, experiment design, and
placebo execution were close to unchanged on the tested fixtures.

## Model and operation coverage

| Model area | Comparisons and independent checks |
| --- | --- |
| Boosted regression, classification, ranking | Dense fitting, input parsing, labels, probabilities, margins, ranking metrics, additive prediction, SHAP, score, sample weights, sparse sets, categorical columns, and artifact schema verification. |
| Classical and automatic forecasting | Local, seasonal, exponential smoothing, autoregression, state-space, intermittent demand, Auto model selection, lag and direct, ensembles, decomposition, and target transforms. Forecast tests cover multi-series identity, recursive horizons, selection metrics, intervals, and rolling cutoffs. |
| Graph and neural forecasting | N-BEATS, N-HiTS, NeuralPanel and LaneNeuralPanel, DCRNN, STAEformer, GraphWaveNet, the five paper graph profiles, and market forecasts. Full graph/context and horizon dimensions are checked. Array-returning bindings preserve the existing methods. |
| Embeddings and graph learning | Neural embedding features, four graph encoders, five standalone regressors, and four link predictors. Parity checks include whole-graph inference, ID precision, artifact roundtrips, layouts, and zero-width features where supported. |
| Deep models | Event outcome, response curves, service residuals, directional pair architectures, choice sets, decisions, conditional flows, scenarios, graph operators, temporal entity models, inverted temporal models, and RegimeMoE. Each model uses its native predictions. Full-call seeded draws are compared for stochastic operations. |
| Spatial, causal, and probabilistic | NNGP and residual NNGP, spatial lag/error/Durbin/two-stage, synthetic DID, experiment design, placebos, causal representations, quantile forecasts, and conformal calibration. Network models are tested with changing weights and independent dense-solver answers. |

Naive, Seasonal Naive, Theta, Optimized Theta, and CartoBoost Lag use requested
interval levels to generate per-series Gaussian residual intervals. Naive and
Seasonal Naive widen by the square root of elapsed recurrence steps. Theta and
Lag use the fitted one-step residual scale as an approximation at other
horizons. Levels and scales persist in artifacts. See the model guides for the
assumptions and history requirements.

## Final verification and cleanup

Across both source phases of this follow-up, 983 main paired cases completed,
plus 20 final Pandas spot checks after the categorical precision changes.
The final run used four threads and checked five operations at 1, 32, 65,536,
and 1,000,000 rows. Every output comparison passed. For one million rows,
fit input parsing took 154.900 → 1.127 ms (137.45×) and class probabilities took
195.880 → 42.854 ms (4.57×). Single-row Pandas regression took 17.196 → 2.343 µs.
One-row fit input parsing was 27.278 → 32.110 µs on that run.

- Full Python suite passed 1,480 tests with 79.28% coverage. The final release
  extension passed 133 focused tests after the constant-decimal R² regression
  was added. All 21 spatial Rust tests passed on that source.
- The full Rust workspace passed 924 tests. CPU compilation on Rust 1.85, all
  benchmark compilation, and the fuzz-target check passed.
- `UV_NO_SYNC=1 uv run --group dev pre-commit run --all-files` passed after
  addressing the Rust lint findings. Full and v1 validation scripts, public API
  and artifact compatibility checks, docs examples, stored benchmark evidence
  checks, and local scale/performance gates passed.
- TypeScript, the production site and WASM build, and all 18 JavaScript tests
  passed. The build emitted MapLibre dependency-expression warnings. Native
  array tests on NumPy 2 passed 78 cases, with 92 optional-package cases skipped
  in that environment. Those packages were exercised in the main environment.
- `cargo clean` removed 12.0 GiB from the workspace. Cleaning the fuzz workspace
  removed another 154.4 MiB. Both target directories are absent. The installed
  extension and shipped WASM bundle retained their hashes and passed smoke
  checks after cleanup.

## Remaining constraints

The measured hardware was the Apple M5 CPU. Accelerator availability, real-data
held-out quality, and latency on other platforms require separate runs. The
current ResponseCurve calibration selector and EventOutcome selector values
still do not select a different calibration algorithm. The built-in event
artifact uses temperature 1. ResponseCurve candidate selection uses max score.

## Run artifacts

The raw records below remain in `/tmp`, independent of Cargo build outputs.

- `cartoboost-batch-paired-{main,million,functions,threads16,threads128,wide,forecast,thresholds,fit-million}.json`
- `cartoboost-batch-paired-{auxiliary,neural-graph}.json` and `cartoboost-batch-special-final-{baseline,candidate}.json`
- `cartoboost-prophet-paired.json`, `cartoboost-prophet-cold-paired.json`, and `cartoboost-regime-paired.json`
- `cartoboost-neural-families-paired.json`, `cartoboost-spatial-families-paired.json`, and `cartoboost-forecast-families/paired.json`
- Fixture and runner sources by operation are kept beside the records. The frozen package location is recorded in `cartoboost-batch-baseline-path.txt`.
- `cartoboost-deep-families-final-paired.json`, `cartoboost-causal-paired.json`, and their individual variant files.
- `cartoboost-audit-final-pandas-paired.json` and `cartoboost-audit-final-validation.json` hold final timing and validation records. The successful lint rerun and post-rebuild checks are in `cartoboost-audit-edge-recheck.log`.
