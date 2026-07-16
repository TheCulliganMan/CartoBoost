# Getting Started

This guide takes you from a modeling question to a validated CartoBoost model.
It is written for Python developers and data scientists working with tabular,
spatial, or time-indexed data. You will install the package, fit a model, choose
an appropriate validation split, compare a baseline, and save the result.

If you already know which API you need, jump to [Choose A Model](user-guide/model-types.md)
or the [Python API Reference](reference/python-api.md).

## 1. Define The Prediction Task

CartoBoost is meant for regression and forecasting tasks where temporal or
spatial structure is part of the hypothesis. Typical questions include:

- Can time, location, and distance explain the target under a leakage-aware split?
- Does source-to-target direction change the estimate compared with treating the two ends as unordered IDs?
- Can demand be forecast from lagged demand, calendar features, and context?
- Do spatial splitters or sparse zone memberships recover signal that an
  axis-only model misses?

Define the target and evaluation split before selecting model features. For
temporal-spatial data, random splits can overstate quality when near-duplicate
times, zones, or route patterns appear in both train and validation data.

## 2. Match Features To The Data Structure

A first regression table might include:

- dense numeric columns: distance, coordinates, time of day, weekday, passenger count;
- periodic columns: hour-of-day with period `24`, day-of-week with period `7`;
- sparse-set columns: zones, route cells, or memberships derived from source
  and target identifiers;
- graph context: directed source-to-target flows when source and target roles should remain distinct.

Declare the split policy and feature schema for the scientific structure you
want to test:

- use `SplitPolicy.AXIS_ONLY` as the dense baseline;
- use `SplitPolicy.STRUCTURED` with periodic, spatial-pair, and sparse-set
  schema entries when those structures are part of the question.

## 3. Install

```sh
uv add cartoboost
```

Verify the install:

```sh
python -c "import cartoboost; print(cartoboost.__version__)"
python examples/quickstart.py
```

For pandas-backed forecasting examples, install the explicit optional extra:

```sh
uv add "cartoboost[pandas]"
```

Optional packages are installed only when needed. For example, use
`cartoboost[polars]` for Polars inputs, `cartoboost[optuna]` for Optuna tuning,
or `cartoboost[onnx]` for the supported ONNX export subset.

## 4. Fit A Regression Model

Run the maintained [NumPy quickstart](../examples/quickstart.py) for a
complete fit, leakage-aware split, baseline comparison, and artifact
round-trip. It is the canonical first example used by the README and browser
documentation.

Start with `SplitPolicy.AXIS_ONLY` if the study only needs dense numeric
features. Use `SplitPolicy.STRUCTURED` only when the schema declares the
periodic, spatial, or sparse structure and the same validation split is used.

## 5. Use Sparse Memberships

Use sparse-set features when a trip belongs to multiple route or zone-derived
sets and a wide one-hot matrix would be awkward or unstable.

```python
schema = {
    "dense": [
        {"name": "trip_distance", "kind": "numeric"},
        {"name": "pickup_hour", "kind": "periodic", "period": 24},
        {"name": "pickup_x", "kind": "numeric"},
        {"name": "pickup_y", "kind": "numeric"},
    ],
    "sparse_sets": [
        {"name": "zone_ids", "kind": "sparse_set"},
    ],
}

model = CartoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    min_samples_leaf=30,
    split_policy="structured",
)

model.fit(
    X_train_dense,
    y_train,
    sparse_sets={"zone_ids": zone_ids_train},
    feature_schema=schema,
)
```

See [Spatial Modeling](spatial_modeling.md) and
[Sparse Features](sparse_features.md) for zone membership examples and blocked
evaluation patterns.

## 6. Forecast Demand

Use `ForecastFrame` when the target is future demand, fare, duration, or another
time-indexed quantity. Panel data should identify the series, such as
pickup/dropoff lane or pickup zone.

```python
from cartoboost.forecasting import ForecastFrame, ThetaForecaster

import pandas as pd

daily_lanes = pd.DataFrame(
    {
        "lane_id": ["route_a"] * 10 + ["route_b"] * 10,
        "date": list(pd.date_range("2026-01-01", periods=10, freq="D")) * 2,
        "pickup_trips": [
            20, 21, 24, 23, 25, 28, 29, 27, 30, 31,
            35, 36, 38, 37, 40, 42, 43, 41, 45, 46,
        ],
    }
)

frame = ForecastFrame.from_pandas(
    daily_lanes,
    timestamp_col="date",
    target_col="pickup_trips",
    series_id_col="lane_id",
    freq="D",
)

model = ThetaForecaster(season_length=7, prediction_interval_levels=[0.8, 0.95])
model.fit(frame)
forecast = model.predict(horizon=3).to_pandas()
```

Forecast tables are deterministic: `series_id`, `timestamp`, `horizon`,
`model`, `mean`, and interval columns such as `lower_80` and `upper_80`.
Use [Forecasting](forecasting.md) for rolling-origin backtesting, CartoBoost lag
features, artifact persistence, CLI workflows, and model selection.

## 7. Validate Against Strong Baselines

For temporal-spatial problems, hold out the latest rows before trusting model
quality:

```python
import cartoboost
from cartoboost.geo import PanelIndex, TimeIndex
from cartoboost.validation import native_out_of_time_split

time = TimeIndex(pickup_times, frequency="h")
panel = PanelIndex(["nyc_taxi"] * len(pickup_times), time=time)
manifest = native_out_of_time_split(
    panel,
    min_train_size=int(len(pickup_times) * 0.8),
    horizon=len(pickup_times) - int(len(pickup_times) * 0.8),
    step=len(pickup_times) - int(len(pickup_times) * 0.8),
    dataset_fingerprint="sha256:...",
    coordinate_crs_note="not_applicable",
    model_version=cartoboost.__version__,
    dependency_versions={"cartoboost": cartoboost.__version__},
)
_, train_idx, validation_idx = manifest.folds()[0]

model.fit(X_all[train_idx], y_all[train_idx])
predictions = model.predict(X_all[validation_idx])
```

Report the split design, target transform, feature set, RMSE, MAE, R2, training
time, prediction time, and model settings. Compare against serious baselines on
the same train/validation rows, such as LightGBM or XGBoost for tabular
regression and appropriate local or external forecasting models for demand
forecasting.

For benchmark claims, document out-of-time, spatial-blocked, grouped, and
leakage-aware validation details in the benchmark writeup.

## 8. Add Graph Or Neural Structure When Justified

Graph, neural, and causal surfaces solve more specialized problems. Add them
only when the data and validation design can test the extra structure; they are
intentionally excluded from the core quickstart.

Use learned embeddings when high-cardinality IDs carry stable signal that is not
captured by dense features alone.

```python
from cartoboost import NeuralEmbeddingRegressor

neural_model = NeuralEmbeddingRegressor(
    dim=16,
    base_model_kwargs={"n_estimators": 80, "split_policy": "axis_only"},
    final_model_kwargs={
        "n_estimators": 120,
        "split_policy": "structured",
    },
)

neural_model.fit(X_train, y_train, ids=pickup_zone_ids_train)
predictions = neural_model.predict(X_validation, ids=pickup_zone_ids_validation)
```

Use CartoBoost graph features or CartoBoost graph models when the observed
units are connected entities, such as directed pickup/dropoff lanes,
borough-zone hierarchies, or repeated OD-pair flow patterns. See
[CartoBoost Graph Models And Features](graph-features.md) and
[CartoBoost Neural Embedding Models And Features](neural-features.md).

## 9. Save Reproducible Artifacts

```python
model.save("cartoboost-regressor.json")
loaded = CartoBoostRegressor.load("cartoboost-regressor.json")

model.save_weights("cartoboost-regressor.weights.json")
weights_loaded = CartoBoostRegressor.load_weights("cartoboost-regressor.weights.json")
```

Use `save` for CartoBoost JSON model artifacts and `save_weights` for portable
prediction artifacts. ONNX export is available only for dense axis-tree
constant-leaf models when the optional `onnx` dependency is installed.

## 10. Source Checkout Checks

For a source checkout, run the full local validation suite with:

```sh
just validate
```

For a faster Python-focused loop:

```sh
uv run --group dev pre-commit run --all-files
uv run --group dev pytest
```
