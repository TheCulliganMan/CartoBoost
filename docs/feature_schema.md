# Feature Schema

Feature schemas tell CartoBoost what each input column means. The same numeric
array can contain scalar variables, cyclic time, and list-valued geographic or
membership features. Without a schema, the model can still fit, but saved
artifacts and future prediction calls have less evidence about which feature
semantics were intended.

Use a schema when:

- hour, weekday, or season should wrap around rather than behave like a line;
- pickup/dropoff zone IDs, vendor codes, service tiers, or other dense labels
  should be treated as categorical or ordinal values;
- pickup/dropoff zones, H3 cells, S2 cells, ZIPs, or route memberships are
  supplied as sparse lists;
- you need saved artifacts to validate feature roles at prediction time;
- you are comparing models and need the feature-generation contract to stay
  fixed across runs.

## Compact Python Format

Declare dense features and sparse-set columns separately:

```python
schema = {
    "dense": [
        {"name": "pickup_x", "kind": "numeric"},
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

Periodic declarations let splitters treat values near the cycle boundary as
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

Numeric dense columns remain eligible for numeric and spatial split candidates.
The current schema does not express named latitude/longitude pairs or richer
spatial roles; diagonal and Gaussian splitters still work from dense numeric
feature pairs according to the current candidate search.

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
    "Numeric",
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

- periodic splitters use declared periods and do not rely on observed values
  covering a full cycle;
- categorical encoders preserve category mappings across prediction and
  save/load;
- sparse splitters prefer schema-declared sparse-set columns;
- numeric dense columns remain eligible for numeric and spatial split
  candidates.

Keep schema changes out of benchmark reruns unless the feature contract itself
is the tested change. A changed schema can change the scientific comparison even
when the raw matrix values are unchanged.
