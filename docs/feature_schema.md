# Feature Schema

Declare the role of each feature, then carry that meaning into the saved
model. A schema distinguishes trip distance from a zone identifier, marks hour
as periodic, and names the sparse memberships required at prediction time.

Use a schema whenever input semantics affect the fit:

- **Periodic:** hour, weekday, or seasonal phase wraps around.
- **Categorical or ordinal:** pickup/dropoff zones, vendor codes, or ordered
  service tiers are labels rather than continuous measurements.
- **Sparse:** a trip belongs to several zones, route cells, or geographic parents.
- **Reusable:** a saved model needs the same feature roles on later data.

Dense models can fit without a schema. An explicit schema makes the input
contract inspectable and keeps feature definitions consistent across comparisons.
See [Sparse features](sparse_features.md) for list-valued inputs.

## Compact Python Format

Declare dense features and sparse-set columns separately:

```python
schema = {
    "dense": [
        {"name": "pickup_x", "kind": "spatial", "other": "pickup_y"},
        {"name": "pickup_y", "kind": "numeric"},
        {"name": "location_id", "kind": "categorical"},
        {"name": "service_tier", "kind": "ordinal"},
        {"name": "distance_m", "kind": "numeric"},
        {"name": "hour_of_day", "kind": "periodic", "period": 24},
    ],
    "sparse_sets": [
        {"name": "zone_ids", "kind": "sparse_set"},
        {"name": "parent_zone", "kind": "zone_sparse_set"},
        {"name": "child_zone", "kind": "zone_sparse_set"},
    ],
}
```

Pass the schema during training:

```python
model.fit(
    X_dense,
    y,
    sparse_sets={"zone_ids": zone_ids},
    feature_schema=schema,
)
```

## Modeling Effects

Periodic declarations let the structured split policy treat values near the cycle boundary as
neighbors. For cyclic time, `hour_of_day=23` and `hour_of_day=0` can be
adjacent late-night behavior rather than opposite ends of a numeric line.

Sparse-set declarations tell the model that a row can belong to multiple
scientific groups at once, such as pickup zone, dropoff zone, parent borough,
grid cell, or route corridor. The model checks membership in the sparse row
instead of treating those IDs as continuous quantities.

Categorical declarations keep dense labels out of numeric split semantics.
Low-cardinality categorical columns use stable one-hot or subset partition
indicators, ordinal columns use stable ordered codes with unknowns below seen
categories, and high-cardinality categorical columns use smoothed target-stat
encodings. The saved model artifact records the encoder mapping so prediction
applies the same contract.

Numeric dense columns remain eligible for axis candidates. Declare a coordinate
pair with `SpatialPairSpec("pickup_x", "pickup_y")` (or the equivalent
`{"kind": "spatial", "other": "pickup_y"}` entry) to mark both named
coordinates for the native diagonal and radial candidates. Unpaired numeric
columns are not searched as arbitrary spatial pairs.

## Saved Schema Format

CartoBoost stores supported schema dictionaries in a compact artifact payload:

```json
{
  "names": [
    "pickup_x",
    "pickup_y",
    "location_id",
    "distance_m",
    "hour_of_day",
    "zone_ids"
  ],
  "kinds": [
    "Numeric",
    "Spatial",
    "Categorical",
    "Numeric",
    {"Periodic": {"period": 24}},
    "SparseSet"
  ]
}
```

## Validation Rules

- Schema length must equal dense feature count plus sparse-set column count.
- `kind` must be numeric, categorical, ordinal, periodic, or sparse-set.
- Periodic entries require a positive period.
- Categorical and ordinal entries must correspond to dense columns, not
  sparse-set columns.
- Sparse-set entries correspond to sparse columns supplied through
  `sparse_sets=`.
- Geographic sparse identifiers can be declared with `zip_sparse_set`,
  `zip3_sparse_set`, `zone_sparse_set`, `region_sparse_set`, `h3_sparse_set`, or
  equivalent aliases (`ZipSparseSet`, `ZoneSparseSet`, `RegionSparseSet`,
  `H3SparseSet`, `GeoSparseSet`, `GeoAbstractSparseSet`). These aliases resolve
  to the underlying sparse-set feature type.

## Training Behavior

When a schema is present:

- periodic candidates use declared periods and do not rely on observed values
  covering a full cycle;
- categorical encoders preserve category mappings across prediction and
  save/load;
- sparse candidates prefer schema-declared sparse-set columns;
- numeric dense columns remain eligible for numeric and spatial split
  candidates.

A schema changes how the model interprets the matrix, even when the raw values
stay the same. Save it with the model and use the same definitions when
comparing feature choices or scoring later trips.
