# Python Estimator

`CartoBoostRegressor` is the public Python estimator for row-level regression.
It follows sklearn conventions for parameter inspection, cloning, pipelines,
and grid search over the supported API surface.

## Model The Scientific Structure First

Start by deciding what structure the rows carry. A row may represent one
observation, one time-bucket aggregate, one route aggregate, or a residual from
another model. `CartoBoostRegressor` is most useful when the response is
expected to vary with structured place/time effects:

- Dense numeric measurements such as trip distance, projected pickup/dropoff
  coordinates, fare history, duration history, hour, and day features.
- Sparse memberships such as route IDs, H3/S2 cells, service areas, or
  overlapping corridor definitions.
- Boundaries that are not purely axis-aligned, including local hotspots and
  radial neighborhoods.
- Fuzzy regions where nearby coordinates or times should not jump abruptly at a
  learned split.
- Objectives where the mean is not the scientific estimand, such as robust
  residual modeling or conditional quantiles.

Use a baseline before adding specialized structure. A typical study starts with
an axis-only or `auto` CartoBoost fit, compares against LightGBM or XGBoost on
the same split and feature set, then adds spatial, periodic, sparse-set,
fuzzy, or robust controls only when they answer a stated modeling question.

## A Spatial-Temporal Regression Template

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=5,
    min_samples_leaf=20,
    split_policy="structured",
    fuzzy=True,
    fuzzy_bandwidth=0.05,
    fuzzy_kernel="gaussian",
)

model.fit(
    X_train_dense,
    y_train,
    sparse_sets={"zone_memberships": zone_memberships_train},
)
predictions = model.predict(
    X_test_dense,
    sparse_sets={"zone_memberships": zone_memberships_test},
)
```

Use dense columns for coordinates, projected x/y values, distances, and
periodic time features. Use `sparse_sets=` for zones, grid cells, service
areas, or route memberships when a row can belong to zero, one, or many
locations.

For robust residuals, set `loss="mae"` or `loss="huber"`. For conditional
intervals or service-level targets, use `loss="quantile"` with
`quantile_alpha`.

## Basic Estimator Usage

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    split_policy="axis_only",
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

`predict` returns a NumPy array.

## Table Inputs

`CartoBoostRegressor` accepts NumPy arrays plus dataframe-style inputs with
column names. Install optional table integrations as needed:

```sh
uv add duckdb
uv add polars
```

DuckDB relations can be passed directly; CartoBoost materializes them at the
fit/predict boundary and preserves relation column names:

```python
import duckdb
from cartoboost import CartoBoostRegressor

trips = duckdb.sql("select trip_distance, hour, log_fare from taxi_training")

model = CartoBoostRegressor(split_policy="structured")
model.fit(
    trips.select("trip_distance, hour"),
    trips.select("log_fare"),
)
prediction = model.predict(trips.select("trip_distance, hour"))
```

## Sparse-Set Features

Sparse-set columns are passed separately from dense features:

```python
zone_memberships = [[132, 138], [161], [236], []]

model = CartoBoostRegressor(
    n_estimators=2,
    learning_rate=0.5,
    max_depth=1,
    min_samples_leaf=1,
    split_policy="structured",
)
model.fit(X_dense, y, sparse_sets={"zone_memberships": zone_memberships})
predictions = model.predict(X_dense, sparse_sets={"zone_memberships": zone_memberships})
```

Sparse IDs must be non-negative integers. Duplicate IDs inside a row are
normalized before training. Models that contain sparse-list splits require
`sparse_sets=` at prediction time.

## Feature Schema

Feature schemas make dense periodic features and sparse-set columns explicit:

```python
schema = {
    "dense": [
        {"name": "distance_m", "kind": "numeric"},
        {"name": "hour_of_day", "kind": "periodic", "period": 24},
    ],
    "sparse_sets": [
        {"name": "zone_memberships", "kind": "sparse_set"},
    ],
}

model.fit(
    X_dense,
    y,
    sparse_sets={"zone_memberships": zone_memberships},
    feature_schema=schema,
)
```

Use schemas when the fitted artifact needs to communicate the scientific role
of each column, not only its position in an array. See
[Feature Schema](../feature_schema.md) for validation rules and saved schema
representation.

## Sample Weights

`sample_weight` must have the same length as `y` and contain finite
non-negative values.

```python
model.fit(X_train, y_train, sample_weight=weights)
```

Weights affect split scoring and leaf values during training.

## Missing Numeric Values

CartoBoost's supervised estimators require finite numeric features, targets,
coordinates, sparse-set weights, and sample weights. They do not silently
impute numeric `NaN`, `None`, `pd.NA`, `inf`, or `-inf` values. Clean or impute
numeric columns before calling `fit`, `predict`, or explanation helpers.

Categorical columns are different: pandas categorical, string, or object
columns can carry missing category sentinels, which are encoded as a stable
missing-category token during fit and reused from the saved artifact.

## Optuna Tuning

Optuna tuning works through the same estimator contract. Install the optional
dependencies with `uv add optuna scikit-learn`, then optimize an
objective that constructs a fresh `CartoBoostRegressor` for each trial.

```python
import optuna
from sklearn.model_selection import cross_val_score

from cartoboost import CartoBoostRegressor


def objective(trial):
    model = CartoBoostRegressor(
        n_estimators=trial.suggest_int("n_estimators", 50, 300),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 1, 6),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 32),
    )
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=3,
        scoring="neg_mean_squared_error",
    )
    return float(-scores.mean())


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
best_model = CartoBoostRegressor(**study.best_params).fit(X_train, y_train)
```

Keep each trial self-contained: create the estimator inside the objective and
fit it only on that trial's data split. Tune the typed split policy, fuzzy
routing, losses, or linear leaves only after the validation design is fixed.

## Additive Values And SHAP

`predict_additive_values(X)` returns per-row additive components whose sums
match `predict(X)`. Use these artifacts to inspect which fitted components
move predictions before turning the model into a scientific claim.

SHAP support is exposed through:

```python
explainer = model.make_shap_explainer(X_background)
explanation = model.explain_shap(X_test, background=X_background)
```

For an exact, low-latency attribution to the fitted ensemble components, use
`decomposition="weights"`. It returns initial-prediction and per-tree values
without SHAP permutation sampling. This is a component audit rather than
original-column attribution; see [SHAP Support](../shap.md) for the choice and
`TreeExplainer` compatibility details.

Run `uv add shap` or add the optional SHAP dependency in
your local environment. Sparse-list models can be explained through the helper
by supplying matching foreground and background sparse sets.

## Save And Load

```python
model.save("model.cartoboost.json")
loaded = CartoBoostRegressor.load("model.cartoboost.json")

model.save_weights("model.weights.json")
weights_loaded = CartoBoostRegressor.load_weights("model.weights.json")
```

JSON artifacts preserve training metadata when available, including the split policy,
leaf predictor, fuzzy settings, loss, schema, and sparse-set requirements.
Save artifacts when later interpretation, audit, or rerun comparisons depend on
the exact modeling contract.

## Common Errors

| Error | Cause |
| --- | --- |
| `ImportError` | The package import failed or the installed package is incomplete. |
| `NotImplementedError` | The installed package does not support a requested feature. |
| `ValueError` | Invalid parameters, mismatched row counts, or incompatible sparse/schema inputs. |
| `RuntimeError` | Prediction or save was called before fit. |
