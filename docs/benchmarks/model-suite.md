# Model Benchmark Suite

## Bottom Line

The standard model suite is a bounded regression benchmark for checking
ordinary tabular behavior, public graph-regression plumbing, validation-search
discipline, timing, and artifact reporting before making broader claims.

The current maintained public run uses seed 42, a 5,000-row deterministic sample
for California housing, and an equal-budget inner-validation search with three
candidates per tunable model family before final holdout scoring. It requests
LightGBM, XGBoost, CatBoost, scikit-learn HistGradientBoosting, RandomForest,
ExtraTrees, Ridge, mean, and graph-specific diagnostic rows where applicable.
The artifact also repeats the same protocol with seeds 42, 43, and 44 for
comparison intervals.

This run does not support a CartoBoost winner claim. The best completed
external baseline has lower RMSE than the single current-code `cartoboost` row
on diabetes, California housing, karate random, and karate group holdout.
CartoBoost is close to XGBoost on the California housing sample, but
HistGradientBoosting is clearly lower RMSE in this maintained run.

The refreshed artifact records a comparability audit in both JSON and Markdown.
Every tunable requested model uses three inner-validation candidates, no model
selects on outer test labels, and all requested external baselines complete in
the local Python 3.13 benchmark environment: LightGBM, XGBoost, CatBoost,
HistGradientBoosting, RandomForest, ExtraTrees, Ridge, and mean.

## Reproduce

```sh
PYTHONPATH=python uv run --group dev --group bench python \
  scripts/run_model_benchmark_suite.py \
  --output-dir docs/assets/model_benchmarks_public \
  --datasets diabetes,california_housing,karate \
  --n-rows 5000 \
  --models mean,cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting,random_forest,extra_trees,ridge,node2vec_regressor,graphsage_regressor \
  --n-estimators 24 \
  --graph-dim 4 \
  --graph-epochs 2 \
  --selection-mode validation_search \
  --validation-trials 3 \
  --repeat-seeds 42,43,44 \
  --no-plots
```

Artifacts:

- `docs/assets/model_benchmarks_public/results.json`
- `docs/assets/model_benchmarks_public/results.jsonl`
- `docs/assets/model_benchmarks_public/results_aggregate.json`
- `docs/assets/model_benchmarks_public/results.md`

`results.json` and `results.md` include the runtime resource snapshot,
comparability audit, and output artifact sizes for this run: `results.json`
329,988 bytes, `results.jsonl` 137,248 bytes, and `results.md` 18,616 bytes.

## Baseline Environment

| Key | Package | Import | Version | Required class available |
| --- | --- | --- | --- | ---: |
| scikit-learn | `scikit-learn` | `sklearn` | `1.9.0` |  |
| XGBoost | `xgboost` | `xgboost` | `3.3.0` | true |
| LightGBM | `lightgbm` | `lightgbm` | `4.6.0` | true |
| CatBoost | `catboost` | `catboost` | `1.2.10` | true |

## Comparability Audit

| Check | Result |
| --- | --- |
| Same outer splits for requested models | true |
| Primary and selection metric | RMSE |
| Selection uses outer test labels | false |
| Equal tunable trial budget | true |
| Tunable trial count | 3 |
| Completed external baselines | CatBoost, ExtraTrees, HistGradientBoosting, LightGBM, mean, RandomForest, Ridge, XGBoost |
| Skipped requested external baselines | none |
| Completed CartoBoost-family rows | `cartoboost`, `graphsage_regressor`, `node2vec_regressor` |
| CartoBoost/external comparison rows | 4 |

## Selection and Leakage Policy

- Every tunable model family chooses from the same three-candidate grid on
  deterministic inner validation rows drawn only from the outer training split.
- The public CartoBoost comparison uses one validation-selected `cartoboost`
  row retrained on the full outer training split; graph, neural, and
  link-prediction rows are diagnostics.
- Neural and graph feature gates use deterministic inner train/validation rows
  inside the training split only.
- The best external baseline is selected only for reporting after every model
  has already been scored on the same held-out split.

## Dataset Sources

| Workload | Source | Rows | Features | SHA-256 fingerprint |
| --- | --- | ---: | ---: | --- |
| Diabetes | `sklearn.datasets.load_diabetes` bundled public regression dataset. | 442 | 10 | `d0e115e7bf84c3d7f4c1b43e7e1cb0bf35cd01ad1e0fd239320748b66f1f3888` |
| California housing | `sklearn.datasets.fetch_california_housing` deterministic 5,000-row seed-42 sample from the 20,640-row public California housing dataset. | 5,000 | 8 | `d0f75cd29b2fa35166c72d168c78cd2f206ab5b1c2d6a29e38437c55d3fa77ad` |
| Karate | Embedded Zachary karate club edge list and post-split labels from the benchmark harness constants. | 78 | 5 | `069058a0030b0e4859fbfb8254bc70c9f73eceb83c0fad5e2f1eba22352a6824` |

## Comparison Summary

For each regression split, this table compares the single primary `cartoboost`
row with the lowest-RMSE external baseline that finished under the same split
and global benchmark settings.

| Workload / split | CartoBoost RMSE | CartoBoost WAPE | Best external baseline | External RMSE | External WAPE | RMSE delta | R2 delta | Result |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Diabetes / random | 54.3467 | 0.2875 | Ridge | 51.5180 | 0.2657 | +2.8287 | -0.0470 | External lower RMSE |
| California housing / random | 0.6301 | 0.2264 | HistGradientBoosting | 0.5958 | 0.2138 | +0.0342 | -0.0307 | External lower RMSE |
| Karate / random | 0.2665 | 0.2210 | XGBoost | 0.0488 | 0.0409 | +0.2177 | -1.1713 | External lower RMSE |
| Karate / group holdout | 0.2661 | 0.1695 | XGBoost | 0.2584 | 0.1157 | +0.0077 | -0.0393 | External lower RMSE |

## Repeated Comparison

The repeated comparison uses seeds 42, 43, and 44 with the same model roster,
validation-search budget, split policy, and dataset definitions. Negative RMSE
and WAPE deltas favor CartoBoost; positive R2 deltas favor CartoBoost.

| Workload / split | Best external baseline counts | RMSE delta mean | RMSE delta 95% CI | WAPE delta mean | R2 delta mean | R2 delta 95% CI | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| California housing / random | HistGradientBoosting: 3 | 0.030574 | 0.026971 to 0.034176 | 0.012214 | -0.029786 | -0.030655 to -0.028918 | External lower RMSE |
| Diabetes / random | HistGradientBoosting: 1, Ridge: 2 | 2.249828 | 0.676595 to 3.823060 | 0.016519 | -0.039117 | -0.065315 to -0.012918 | External lower RMSE |
| Karate / group holdout | ExtraTrees: 1, XGBoost: 2 | 0.112862 | -0.066401 to 0.292125 | 0.091720 | nan | nan to nan | Mixed interval |
| Karate / random | ExtraTrees: 1, XGBoost: 2 | 0.100129 | -0.017642 to 0.217899 | 0.097043 | -0.526215 | -1.172317 to 0.119886 | Mixed interval |

## Validation Search Selections

The table records the selected inner-validation candidate for the primary
CartoBoost row and the best external baseline on each split. Full candidate
tables are in `docs/assets/model_benchmarks_public/results.md`.

| Workload / split | Model | Selected trial | Validation RMSE | Inner train rows | Inner validation rows | Selected config |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Diabetes / random | CartoBoost | 2 | 59.4526 | 283 | 70 | `{"learning_rate": 0.1, "max_depth": 4, "min_samples_leaf": 10, "n_estimators": 18}` |
| Diabetes / random | Ridge | 1 | 56.2885 | 283 | 70 | `{"ridge_alpha": 0.1}` |
| California housing / random | CartoBoost | 1 | 0.6444 | 3,200 | 800 | `{"learning_rate": 0.08, "max_depth": 4, "min_samples_leaf": 20, "n_estimators": 24}` |
| California housing / random | HistGradientBoosting | 1 | 0.6286 | 3,200 | 800 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |
| Karate / random | CartoBoost | 1 | 0.2520 | 50 | 12 | `{"learning_rate": 0.08, "max_depth": 4, "min_samples_leaf": 20, "n_estimators": 24}` |
| Karate / random | XGBoost | 2 | 0.2817 | 50 | 12 | `{"learning_rate": 0.1, "max_depth": 4, "n_estimators": 18}` |
| Karate / group holdout | CartoBoost | 2 | 0.3446 | 42 | 10 | `{"learning_rate": 0.1, "max_depth": 4, "min_samples_leaf": 10, "n_estimators": 18}` |
| Karate / group holdout | XGBoost | 1 | 0.0765 | 42 | 10 | `{"learning_rate": 0.08, "max_depth": 4, "n_estimators": 24}` |

## Interpretation

Use this page to diagnose benchmark plumbing, leakage-safe selection, and
model-family behavior. The maintained run now uses equal-budget validation
search for every tunable family, but it is still a bounded single-seed
benchmark. External baselines are stronger on every maintained split, so any
reusable CartoBoost model improvement should be justified by larger real-data
evidence and then rerun through this fixed protocol before public claims
change.
