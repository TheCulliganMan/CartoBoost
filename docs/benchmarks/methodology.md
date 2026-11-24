# Reading Benchmark Results

A benchmark answers a specific question about a dataset and a holdout. Start
with the result, then check whether that question matches yours. Predicting a
later week of taxi demand is different from predicting trips in a pickup zone
the model has never seen.

## What a report contains {#required-fields}

The [benchmark reports](index.md) pair metric tables with the command, data
source, sample size, split, model settings, timings, and artifact paths. JSON
artifacts provide the detailed rows behind the prose and tables.

| Detail | What it tells you |
| --- | --- |
| Data source and sample | Which trips, dates, lanes, or generated patterns were measured. |
| Target and transformation | Whether an error is in counts, seconds, dollars, or a transformed scale. |
| Holdout definition | Whether the model faces familiar trips, later timestamps, or unseen places. |
| Feature access | Which information each model received and when it became available. |
| Model settings and selection | How much fitting and tuning each comparison used. |
| Timing | The cost of loading, fitting, and prediction in the recorded environment. |
| Artifact | The exact metrics, settings, and metadata associated with that run. |

## Metrics By Task

| Task | Read together | Watch for |
| --- | --- | --- |
| Fare or duration regression | RMSE, MAE, R2, and target scale | Large errors hidden by an average or a log transform. |
| Taxi demand forecasting | RMSE, MAE, WAPE, and error by horizon | A strong aggregate score hiding weak longer-horizon forecasts. |
| Quantile or interval prediction | Pinball loss, coverage, and interval width | Wide intervals that achieve coverage without being useful. |
| Classification | Logloss, ROC-AUC or PR-AUC, Brier score, and calibration | Good ranking with poorly calibrated probabilities. |
| Grouped ranking | NDCG, MAP, and MRR by query group | Scores computed across unrelated candidate groups. |
| Spatial prediction | Holdout error and residual spatial diagnostics | Nearby training observations making a random split look too easy. |

Fit and prediction timings describe the recorded hardware and workload. They
are most useful alongside data size, estimator settings, and whether imports,
data loading, or conversion time are included.

## Interpreting A Comparison

**Random holdouts** measure performance on held-out rows from the sampled
population. **Spatial and cold-ID holdouts** test unfamiliar places or
identifiers. **Out-of-time and rolling-origin holdouts** test later observations
using earlier history. The split often changes the model ranking.

Read the model roster as carefully as the winning row. A result against a
seasonal-naive model answers a different question from a result against tuned
LightGBM or XGBoost. Ties and mixed intervals are meaningful results; small
metric differences do not establish a general winner.

Synthetic datasets isolate a known mechanism, such as midnight wraparound or
route membership. Real NYC TLC comparisons measure accuracy on observed trips
or aggregated demand. Success on a designed pattern does not establish
real-world superiority.

## Reproducing Results

Run the report's command from a source checkout with the listed dependencies
and data available. The command identifies the sample, seed, split, roster,
and settings associated with the table. Changing any of those produces a
different comparison.

The generated JSON is the detailed run record. Inspect model failures, timings,
and per-split results alongside aggregate scores. Missing required real inputs
or dependencies stop the maintained benchmark runners with an error.
