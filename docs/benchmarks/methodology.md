# Benchmark Methodology

CartoBoost benchmark claims must be tied to real commands, fixed comparable
settings, and recorded artifacts. A benchmark page should let a reader see what
was run, which data were used, which split was evaluated, which models were
compared, and which metric supports the interpretation.

## Required Fields

Each benchmark report should name:

- command and working directory;
- data source and whether the data are real, synthetic, or generated acceptance
  data;
- sample size and task definitions;
- train/test split or CV fold construction;
- model roster and comparable estimator settings;
- comparability audit covering shared rows or horizons, metric roster,
  feature-access policy, skipped requested models, and whether any model
  selection used holdout labels;
- metric table with timing fields;
- artifact paths for JSON, JSONL, markdown, and plots;
- limitations that affect interpretation.

The maintained benchmark families use leakage-safe splits, serious external
comparators, task-specific diagnostics, fit and prediction timing, and recorded
artifacts. The current family roster is:

| Family | Required evidence |
| --- | --- |
| NYC TLC zone/lane demand | TLC source hashes, spatial/grouped splits, LightGBM/XGBoost/CatBoost/sklearn baselines, residual Moran's I, block error, regional calibration and interval width. |
| METR-LA / PEMS graph forecasting | Fixed sensor graph hashes, rolling-origin graph splits, PyTorch Geometric/DCRNN baselines, horizon error, graph-distance residual decay. |
| EPA air-quality interpolation | EPA source hashes, buffered monitor holdouts, PySAL/PyKrige/GSTools baselines, variogram and spatial residual diagnostics. |
| California housing sanity | Fixed public sample, strong tabular baselines, spatial block error as a sanity diagnostic only. |
| Synthetic spatial fields | Generator manifest hash, block splits, kriging/spatial-regression baselines, variogram and residual autocorrelation diagnostics. |
| Synthetic graph diffusion | Generator manifest hash, rolling graph splits, graph neural baselines, horizon and graph-distance diagnostics. |
| Synthetic geo-causal lift panels | Generator manifest hash, rolling panel splits, placebo summaries, and known-effect error metrics. |

## Metrics By Task

Classification reports should include logloss, ROC-AUC or PR-AUC, Brier score,
ECE, fit time, prediction time, and save/load probability drift. Ranking
reports should include NDCG, MAP, MRR, fit time, prediction time, and save/load score drift.
Categorical reports should state the number of categories, chosen
encoding strategy, unknown-category rate, and whether the saved model
round-tripped predictions within tolerance. Unsupported export checks should
assert loud `NotImplementedError` failures for categorical regressor export and
classifier/ranker portable-weight or ONNX export.

## Interpreting A Comparison

Do not use stale artifacts after changing benchmark-affecting code. If feature
generation, fitting, prediction, metric computation, or split construction
changes, rerun the affected benchmark before updating public claims.

Do not frame benchmark pages around process labels such as cleanup or
provenance. Lead with the current-code result, then show command, data, split,
model roster, metrics, timing, artifact paths, and limits.

## Reproducing Results

Run the exact command printed in the report from a source checkout with the
documented optional dependencies installed. Keep the dataset identity, sample
size, split boundaries, feature access, model roster, estimator settings, and
seed unchanged when comparing against the published table.

Generated outputs belong under `target/` unless the run is intentionally being
recorded as maintained evidence. A reproduced result should include the JSON
artifact, environment versions, wall-clock timing, and any skipped model with
its explicit failure reason.

Synthetic runs are useful for verifying execution and controlled mechanisms.
They do not substitute for real-data comparisons when the conclusion concerns
forecast accuracy, spatial transfer, or deployment performance.
