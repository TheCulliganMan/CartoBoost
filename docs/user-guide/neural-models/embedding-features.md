# Embedding Feature Workflows

Use neural embedding features when learned ID vectors should become dense
columns for another model. This path is appropriate for ablation studies:
compare a structured model to the same model plus embedding columns on the same
split, then report whether the ID representation explains additional residual
variation.

## Feature-Generation Contract

For `dim=4` and prefix `neural.pickup_zone`, generated columns are:

- `neural.pickup_zone_00`
- `neural.pickup_zone_01`
- `neural.pickup_zone_02`
- `neural.pickup_zone_03`

The downstream model input is:

```text
[original_dense_features..., neural feature block]
```

## Neural-Augmented Boosting

`NeuralEmbeddingRegressor` learns ID vectors and appends them to a final
tabular model. By default, it uses residual training:

1. Fit a baseline model with structured inputs.
2. Compute residuals: `residual = y - baseline.predict(X_structured)`.
3. Fit an embedding table on `(ids, residual)`.
4. Transform IDs into an embedding matrix.
5. Concatenate structured features and embeddings.
6. Fit the final downstream model on the augmented matrix and original target.

```python
import numpy as np
from cartoboost import NeuralEmbeddingRegressor

rng = np.random.default_rng(0)
ids = rng.integers(1, 200, size=1000, dtype=np.uint64)
X = rng.normal(size=(1000, 8))
y = 1.2 * X[:, 0] - 0.8 * X[:, 1] + (ids % 7) * 0.1

model = NeuralEmbeddingRegressor(
    dim=16,
    final_model_kwargs={"n_estimators": 80, "learning_rate": 0.08, "max_depth": 4},
)
model.fit(X, y, ids=ids)
pred = model.predict(X, ids=ids)
```

Residual mode focuses embeddings on what the structured model missed. It does
not directly add neural residuals to the final output; it exposes learned
representations to the final model and lets that model decide when the signal
is useful.

## Controls

- `oof_folds > 1` trains final-model embeddings with out-of-fold residuals.
- `support_prior_strength` shrinks rare IDs toward prior vectors.
- `fallback_ids` supports hierarchical fallback chains such as zone to service
  zone to borough to global representative.
- 2D `ids` supports multi-key embeddings such as pickup zone, dropoff zone,
  pickup-dropoff pair, zone-hour bucket, or trip cluster.
- `neighbor_ids` supports graph-aware fallback by averaging known adjacent-zone
  or typed-neighbor embeddings for unseen spatial IDs.

See [Neural Embedding Models And Features](../../neural-features.md) for
artifact details and benchmark reporting.
