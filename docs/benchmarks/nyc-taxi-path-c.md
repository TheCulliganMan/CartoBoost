# NYC Taxi Geo-Temporal Tests

These experiments use 2024 NYC TLC yellow taxi data to test five concrete
geo-temporal modeling questions. Results apply to the documented sample and
splits; they do not establish universal superiority or evaluate deep models.

## Data And Outputs

Path C uses the maintained 2024 yellow taxi benchmark frame from
`scripts/run_nyc_taxi_quality_benchmarks.py` and writes claim artifacts under
`docs/assets/nyc_taxi_benchmarks/`.

| Field | Value |
| --- | --- |
| Source | NYC TLC trip records |
| Taxi type | Yellow |
| Period | 2024 |
| Tasks | Duration, fare, pickup demand |
| Claim runner | `scripts/run_nyc_taxi_path_c_claims.py` |
| Result checker | `scripts/check_nyc_taxi_path_c_gates.py` |
| JSON artifact | `docs/assets/nyc_taxi_benchmarks/path_c_claims.json` |
| JSONL rows | `docs/assets/nyc_taxi_benchmarks/path_c_claims.jsonl` |
| Markdown artifact | `docs/assets/nyc_taxi_benchmarks/path_c_claims.md` |

## Reproduce

```sh
PYTHONPATH=python uv run --group dev --group bench python \
  scripts/run_nyc_taxi_path_c_claims.py \
  --no-download \
  --sample-size 100000 \
  --months 1,2,3,4,5,6,7,8,9,10,11,12 \
  --n-estimators 12 \
  --output-dir docs/assets/nyc_taxi_benchmarks

uv run --group dev python scripts/check_nyc_taxi_path_c_gates.py
```

The runner stops when real TLC inputs or required taxi-zone geometry are
missing. Synthetic smoke output is for development only and is not included in
these real-data results.

## Questions And Success Criteria

| Question | Prediction unit | Evaluation split | Baselines | Success criterion |
| --- | --- | --- | --- | --- |
| Directional structure | Pickup zone to dropoff zone by time bucket | Pickup-zone spatial holdout | Unordered pair, source+target additive | Ordered model has at least 2% lower duration and fare RMSE. |
| Temporal structure | Pickup zone x real hourly timestamp | Rolling-origin zone-time by actual pickup timestamp | Trailing mean, seasonal naive, pooled Ridge | Temporal model has at least 2% lower pickup-demand RMSE than seasonal naive. Trailing mean and pooled Ridge remain reported falsifier rows. |
| Known-future sensitivity | Zone-time demand | Rolling-origin zone-time by actual pickup timestamp | Future-known covariates ablated | Full model has positive ablation delta and at least 1% lower RMSE. |
| Spatial transfer | Completed trips from held-out pickup zones | Pickup-zone spatial holdout | Target-encoded zone-only, mean | Trip-level route/geometry features improve duration and fare RMSE by at least 1% over trip-level falsifiers. |
| Residual correction | Completed trip with baseline estimate | Pickup-zone spatial holdout | Raw baseline, global residual mean, linear residual model | Nonlinear correction has at least 1% lower duration and fare RMSE. |

Each result row records the question id, task, split kind, train/test index hashes,
dataset hash, model, architecture, capability tier, comparison baseline, primary
metric, nonzero materiality threshold, percent improvement, prediction-unit
metadata, rolling-origin cutoff timestamp where applicable, RMSE, MAE, WAPE, R2,
fit and predict time, peak memory, save/load parity, feature access policy,
train-only target encoding, and the guarantee that selection does not use outer
test labels.
