# Evaluating Geographic Models

Evaluate geographic models with a split that withholds place, time, groups, or
a documented combination of those axes. Treat a random row split as a
diagnostic because nearby taxi trips can share pickup/dropoff
zone effects, road geometry, traffic shocks, weather, and calendar pressure.

## Required Metadata

Every geographic benchmark report should record:

- dataset fingerprint
- split id
- split manifest hash
- coordinate CRS note
- model version
- dependency versions
- random seed

The split manifest hash is the stable identity for the exact train/test rows.
If benchmark code changes the split implementation or the row order, rerun the
benchmark and update the manifest hash in the report.

## Python Example

```python
from cartoboost.geo import CoordinateMatrix, spatial_block_cv_manifest

coords = CoordinateMatrix(
    x=pickup_zone_centroid_x,
    y=pickup_zone_centroid_y,
    crs="EPSG:2263",
)

manifest = spatial_block_cv_manifest(
    coords,
    n_folds=5,
    dataset_fingerprint="sha256:9c694e388df6104ec6187f95e25c5198acff19667c8cf5ac025d0ee1a2d09900",
    coordinate_crs_note="NYC TLC taxi zones projected to EPSG:2263 before splitting",
    model_version="0.2.32",
    dependency_versions={"cartoboost": "0.2.32"},
    random_seed=42,
    split_id="pickup_zone_block_cv_v1",
)

print(manifest.hash())
for fold_id, train_idx, test_idx in manifest.folds():
    print(fold_id, len(train_idx), len(test_idx))
```

## Split Types

Use `spatial_block_cv_manifest` when the claim is about generalizing to held-out
places. It sorts projected coordinates into deterministic spatial blocks.

Use `buffered_spatial_cv_manifest` when nearby train rows would leak the held-out
place. It removes training rows within the configured buffer distance of each
test block.

Use `group_spatial_cv_manifest` when a full pickup zone, route family, customer,
or other spatial group must be absent from training.

Use `rolling_origin_panel_split_manifest` for panel demand forecasting where
future rows must be held out for every zone or lane.

Use `spatial_temporal_blocked_split_manifest` when a benchmark claim needs both
held-out places and held-out future time.

## Wasm And Modeling Lab Example

The modeling lab can display the same manifest hash produced by Python by
calling the wasm helper with the split manifest JSON:

```ts
const manifestHash = geoSplitManifestHash(JSON.stringify(splitManifest))
```

For browser examples, keep the manifest visible next to the model metrics:

```ts
const report = {
  model: "cartoboost_regressor",
  datasetFingerprint,
  splitId: splitManifest.split_id,
  splitManifestHash: geoSplitManifestHash(JSON.stringify(splitManifest)),
  coordinateCrsNote: splitManifest.coordinate_crs_note,
  rmse,
  mae,
}
```

This is enough for the page to distinguish a random diagnostic score from a
leakage-safe geographic claim.

The executable contract for the Python split-manifest example is checked by
`scripts/check_docs_examples.py` in CI.
