---
sidebar_label: Your first model
description: Fit a taxi-duration model, compare a structured schema with an axis baseline, and save the result.
---

# Your first model

Fit a taxi-duration regressor, hold out the latest observations, and compare
ordinary axis splits with a schema that understands the daily cycle. Then save
and reload the fitted model.

This walkthrough uses a small, generated taxi-shaped dataset so you can run it
immediately. Its scores demonstrate the workflow; they are not evidence of
performance on NYC taxi trips.

## Install

```sh
uv add cartoboost
```

NumPy is the only required Python runtime dependency. See
[Installation](installation.md) for supported platforms, optional integrations,
and source builds.

## Create the data and holdout

The rows below represent hourly trip observations. The features are trip
distance and pickup hour; the target is duration in minutes. A native temporal
split keeps the final 48 observations out of training.

```python
from datetime import datetime, timedelta

import cartoboost
import numpy as np
from cartoboost import CartoBoostRegressor
from cartoboost.geo import PanelIndex, TimeIndex
from cartoboost.schema import FeatureSchema, NumericSpec, PeriodicSpec
from cartoboost.validation import native_out_of_time_split

rng = np.random.default_rng(7)
rows = 240
pickup_hour = np.arange(rows, dtype=float) % 24
trip_distance = rng.uniform(1, 8, rows)
X = np.column_stack([trip_distance, pickup_hour])
y = (
    8 + 1.8 * trip_distance
    + 2 * np.sin(2 * np.pi * pickup_hour / 24)
    + rng.normal(0, 0.15, rows)
)

pickup_times = [
    (datetime(2026, 1, 1) + timedelta(hours=i)).isoformat()
    for i in range(rows)
]
panel = PanelIndex(
    ["nyc_taxi"] * rows,
    time=TimeIndex(pickup_times, frequency="h"),
)
split = native_out_of_time_split(
    panel,
    min_train_size=192,
    horizon=48,
    step=48,
    dataset_fingerprint="sha256:generated-taxi-quickstart",
    coordinate_crs_note="not_applicable",
    model_version=cartoboost.__version__,
    dependency_versions={"cartoboost": cartoboost.__version__},
)
_, train_idx, test_idx = split.folds()[0]
```

For real data, record a fingerprint of the actual input and choose the holdout
from the deployment question. Use [spatial validation](user-guide/spatial-cv-best-practices.md)
when the goal is prediction in withheld pickup zones or route corridors.

## Fit two comparable models

The schema identifies pickup hour as periodic with a 24-hour cycle. Both models
use the same observations, features, and training settings. Only the split
policy changes.

```python
schema = FeatureSchema.from_specs([
    NumericSpec("trip_distance"),
    PeriodicSpec("pickup_hour", 24),
])
settings = dict(
    n_estimators=40,
    learning_rate=0.06,
    max_depth=4,
    min_samples_leaf=8,
    random_state=7,
)

baseline = CartoBoostRegressor(split_policy="axis_only", **settings)
model = CartoBoostRegressor(split_policy="structured", **settings)

for candidate in (baseline, model):
    candidate.fit(X[train_idx], y[train_idx], feature_schema=schema)

predictions = model.predict(X[test_idx])
print("Prediction shape:", predictions.shape)  # (48,)
print("Axis holdout RMSE:", -baseline.score(X[test_idx], y[test_idx]))
print("Structured holdout RMSE:", -model.score(X[test_idx], y[test_idx]))
```

`axis_only` is a useful dense baseline. `structured` enables the structures
declared by the feature schema. On your own data, keep the same holdout and
compare against LightGBM or XGBoost as well. Report RMSE, MAE, R², training time,
and prediction time with the model settings. `score` returns negative RMSE,
so the example negates it to display error in minutes. This generated-data
comparison is not a model-selection result.

## Save what you fitted

A model artifact preserves the information needed for prediction, including its
feature schema. Keep inference columns in the same order as training columns.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    artifact = Path(directory) / "taxi-duration.json"
    model.save(artifact)
    restored = CartoBoostRegressor.load(artifact)
    np.testing.assert_allclose(
        predictions,
        restored.predict(X[test_idx]),
    )
```

Choose a persistent path in your application. Use `save_weights` and
`load_weights` for portable prediction artifacts. The
[artifact guide](model_artifact.md) explains compatibility and the supported
ONNX export subset.

## Add the structure your task needs

You now have a fit, a holdout, a baseline, and an artifact. The next step depends
on the information available at prediction time.

| Your data contains… | Add… | Read |
| --- | --- | --- |
| Pickup coordinates | A spatial pair in the feature schema. Use projected units for distance-based validation. | [Spatial modeling](spatial_modeling.md) |
| `PULocationID` or `DOLocationID` | A categorical feature with a mapping fitted on training data. | [Categorical features](user-guide/categorical-features.md) |
| Several route cells or zone memberships per trip | Sparse-set inputs, with matching names at fit and predict time. | [Sparse features](sparse_features.md) |
| Repeated pickup or dropoff IDs | Learned embeddings, evaluated on temporal and unseen-ID holdouts where relevant. | [Neural models](user-guide/neural-models/index.md) |
| Observed connections between zones | Graph inputs with an explicit direction and observation cutoff. | [Graph models](user-guide/graph-models/index.md) |
| A future demand target | A `ForecastFrame`, a horizon, and rolling-origin backtests. | [Forecasting](forecasting.md) |

For example, sparse memberships accompany the dense matrix rather than becoming
a wide one-hot table:

```python
# Supply one membership collection per trip in each split.
model.fit(
    X_train_dense,
    y_train,
    sparse_sets={"zone_ids": zone_ids_train},
    feature_schema={
        "dense": [
            {"name": "trip_distance", "kind": "numeric"},
            {"name": "pickup_hour", "kind": "periodic", "period": 24},
        ],
        "sparse_sets": [{"name": "zone_ids", "kind": "sparse_set"}],
    },
)
predictions = model.predict(
    X_test_dense,
    sparse_sets={"zone_ids": zone_ids_test},
)
```

This last fragment uses your own dense matrices and memberships. The earlier
three Python blocks form the complete runnable first model.

## Where to go next

- [Choose a model](user-guide/model-types.md) for classification, ranking,
  demand forecasting, spatial estimation, and deep models.
- [Inspect a browser model](../modeling-lab) to explore a fitted example
  interactively without installing Python.
- [Read the benchmark reports](benchmarks/index.md) for real data, comparable
  baselines, and task-specific results.
- [Use the Python API](reference/python-api.md) for method contracts and
  [parameters](user-guide/parameters.md) for estimator controls.

From a source checkout, `uv run python examples/quickstart.py` runs the maintained
NumPy example with a spatial pair, baseline comparison, and artifact round-trip.
