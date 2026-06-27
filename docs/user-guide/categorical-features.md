# CartoBoost Categorical Features

CartoBoost accepts categorical columns in the CartoBoost regressor,
classifier, and ranker Python wrappers. The fitted artifact records the
mapping used during training so prediction and save/load use the same
encoding.

## Supported Inputs

Categorical detection covers:

- pandas `category`, `string`, and non-numeric object columns;
- columns explicitly marked with `FeatureKind.CATEGORICAL`;
- ordinal columns explicitly marked with `FeatureKind.ORDINAL`.

Missing category sentinels such as `None`, `NaN`, `pd.NA`, and `NaT` are
normalized to one stable missing-category token before encoding.

```python
from cartoboost import CartoBoostRegressor, FeatureKind

schema = {
    "dense": [
        {"name": "location_id", "kind": FeatureKind.CATEGORICAL},
        {"name": "service_tier", "kind": FeatureKind.ORDINAL},
        {"name": "trip_distance", "kind": FeatureKind.NUMERIC},
    ]
}

model = CartoBoostRegressor(splitters=["axis"])
model.fit(X_train, fare_train, feature_schema=schema)
```

## Encoding Rules

Low-cardinality nominal columns use stable numeric indicator columns. Two-value
columns use one-hot indicators; three- or four-value columns add deterministic
subset partition indicators so a tree can search simple category partitions
without a separate categorical tree builder. Larger low-cardinality columns use
one-hot indicators, and high-cardinality nominal columns use smoothed
target-statistic encoding with an explicit unknown-category value. Ordinal
columns use a deterministic ordered category mapping.

The target statistic is learned only from the training rows passed to `fit`,
and the fitted training matrix uses leave-one-out smoothed values so a row does
not encode its own target directly. For classifiers it uses encoded class ids;
for rankers it uses relevance labels. Unknown prediction-time categories map to
all-zero indicator columns, `-1` for ordinal columns, or the train-side global
mean for target-stat columns.

## Artifact Behavior

```python
model.save("fare-categorical.json")
restored = CartoBoostRegressor.load("fare-categorical.json")
restored.predict(X_validation)
```

When categorical columns are used, CartoBoost writes a wrapper artifact that
contains the native model payload plus the categorical encoder mapping. This is
required for bit-stable or tolerance-stable prediction after loading.

## Baseline Comparisons

When comparing against LightGBM, XGBoost, sklearn, or a one-hot pipeline, keep
the comparison leakage-safe:

- fit every encoder on the same training rows;
- evaluate every model on the same validation rows;
- record whether unknown validation categories were present;
- report the model roster, sample size, split definition, metrics, and timing.
