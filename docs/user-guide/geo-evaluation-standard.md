# Evaluating Geographic Models

Predicting another trip in a familiar
pickup zone, a trip in an unseen zone, and next week's demand are different
tests. Choose place, time, group, or combined holdouts accordingly; use a random
split as a diagnostic of the easier repeated-context task.

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
import cartoboost
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
    model_version=cartoboost.__version__,
    dependency_versions={"cartoboost": cartoboost.__version__},
    random_seed=42,
    split_id="pickup_zone_block_cv_v1",
)

print(manifest.hash())
for fold_id, train_idx, test_idx in manifest.folds():
    print(fold_id, len(train_idx), len(test_idx))
```

## Split Types

| What deployment must handle | Split helper | What is withheld |
| --- | --- | --- |
| Unseen places | `spatial_block_cv_manifest` | Deterministic blocks of projected coordinates |
| Unseen places without nearby training observations | `buffered_spatial_cv_manifest` | A test block plus a buffer removed from training |
| Entire unseen zones or route families | `group_spatial_cv_manifest` | Complete spatial groups |
| Future demand in each panel series | `rolling_origin_panel_split_manifest` | The next horizon at each origin |
| Both unseen places and future periods | `spatial_temporal_blocked_split_manifest` | Spatial and temporal blocks together |

Buffer distances must use the same units as the projected coordinates. Inspect
fold sizes after buffering: a wide exclusion zone can leave too little history
for a useful comparison.

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

Keep the manifest with the scored predictions so a comparison can be traced
back to its exact rows.

{/* Executable example coverage: scripts/check_docs_examples.py */}
