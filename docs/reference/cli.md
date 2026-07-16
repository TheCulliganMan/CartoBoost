# Command-Line Reference

The Rust CLI trains, predicts, inspects, and evaluates models from dense numeric
CSV files. It is available from a source checkout, not from the PyPI wheel.
Prefix every command with `cargo run -p cartoboost-cli --`.

The CLI is intentionally narrower than the Python API. For sparse sets, graph
features, neural residual embeddings, rolling-origin forecasting, or
leakage-aware split generation, use the Python API and benchmark scripts.

## `train`

```sh
cargo run -p cartoboost-cli -- train --data <csv> [--config <toml>] [--model-out <path>] [--output json|csv]
```

Trains a dense numeric CSV model. If `--model-out` is omitted, the CLI writes
`cartoboost-model.json`.

JSON output:

```json
{"ok": true, "command": "train", "rows": 100, "features": 8, "model_path": "model.json", "trees": 100}
```

## `predict`

```sh
cargo run -p cartoboost-cli -- predict --model <path> --input <csv> [--predictions-out <path>] [--output json|csv]
```

Loads a model and predicts dense numeric rows. Prediction CSV output uses:

```csv
row,prediction
0,1.25
1,2.5
```

## `eval`

```sh
cargo run -p cartoboost-cli -- eval --model <path> --data <csv> [--output json|csv]
```

Computes mean absolute error against the target column stored in the model, or
the last data column when the model has no target name.

Use `eval` only on a named holdout file. It does not create random,
out-of-time, spatial, or grouped splits for you.

## `inspect`

```sh
cargo run -p cartoboost-cli -- inspect [--model <path>] [--config <toml>] [--data <csv>] [--output json|csv]
```

Summarizes model, config, and data inputs without training.

## Accepted Options

| Command | Options |
| --- | --- |
| `train` | `--data`, `--config`, `--model-out`, `--output`, `--help` |
| `predict` | `--model`, `--input`, `--predictions-out`, `--output`, `--help` |
| `eval` | `--model`, `--data`, `--output`, `--help` |
| `inspect` | `--model`, `--config`, `--data`, `--output`, `--help` |

Unknown options fail fast.

## Reproducible Evaluation Flow

For a CLI-backed comparison, create train and validation CSVs once, then reuse
them for every model:

```sh
cargo run -p cartoboost-cli -- train \
  --data taxi_train.csv \
  --config configs/regression.toml \
  --model-out target/evidence/cartoboost-model.json \
  --output json

cargo run -p cartoboost-cli -- predict \
  --model target/evidence/cartoboost-model.json \
  --input taxi_validation_features.csv \
  --predictions-out target/evidence/cartoboost-predictions.csv \
  --output csv

cargo run -p cartoboost-cli -- eval \
  --model target/evidence/cartoboost-model.json \
  --data taxi_validation_with_target.csv \
  --output json
```

Record the split definition, target transformation, row counts, feature
columns, config file, and output paths with the reported metrics. If LightGBM,
XGBoost, or another baseline uses a different feature file or split, the result
is not a fair model-choice comparison.

## Forecasting Script

Forecasting V1 is exposed through `scripts/forecast.py`:

```sh
python scripts/forecast.py fit \
  --input examples/forecasting/forecast_cli_input.csv \
  --timestamp-col timestamp \
  --target-col demand \
  --series-id-col series_id \
  --freq D \
  --model theta \
  --horizon 7 \
  --season-length 7 \
  --artifact-dir target/forecasting/theta \
  --output target/forecasting/theta_forecast.csv
```

Commands:

| Command | Purpose |
| --- | --- |
| `fit` | Reads CSV history, writes `model.json`, `resolved_config.json`, and optional forecast CSV. |
| `predict` | Reads a saved forecast artifact directory and writes a forecast CSV. |
| `backtest` | Runs a deterministic final-window backtest and writes JSON metrics. |
| `compare` | Scores one or more forecasting models on the same holdout. |

Forecasting options:

| Option | Notes |
| --- | --- |
| `--input` | CSV history. Required for `fit`, `backtest`, and `compare`. |
| `--timestamp-col` | Timestamp column such as `timestamp` or `pickup_hour`. |
| `--target-col` | Target column such as `demand`, `fare`, or `duration`. |
| `--series-id-col` | Optional panel id such as `series_id` or `lane_id`. |
| `--freq` | Frequency: `D`, `H`, `W`, or `M`. |
| `--model` | `naive`, `seasonal_naive`, `theta`, `optimized_theta`, `ets`, `auto_arima`, `cartoboost_lag`, or `all` for `compare`. |
| `--horizon` | Positive forecast horizon. |
| `--season-length` | Seasonal cycle for seasonal naive and theta-style models. |
| `--output` | Forecast CSV or JSON metrics path. |
| `--artifact-dir` | Directory for model/config/metrics artifacts. |
| `--config` | JSON or simple TOML-style config file; CLI flags override file values. |

Forecast CSVs include `series_id`, `timestamp`, `model`, `horizon`,
`forecast`, `lower_80`, and `upper_80`. Invalid configs, missing columns,
unknown model names, and missing artifact directories exit nonzero with a
message on stderr.

For scientific forecasting comparisons, prefer `compare` or `backtest` over
manually fitting separate models. Those commands keep the forecast rows aligned
by `series_id`, `timestamp`, and `horizon`, which is required for honest
forecast metrics.
