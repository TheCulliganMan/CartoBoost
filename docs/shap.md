# SHAP Support

SHAP explanations help audit fitted CartoBoost models after training. For
taxi-trip studies, use them to ask which modeled covariates contributed to a
prediction: distance, hour, pickup/dropoff memberships, graph-derived columns,
neural embedding columns, or fitted tree weights.

SHAP is an explanation layer, not a new model and not proof of causality. It is
most useful after the validation design is fixed, because explanations inherit
the same feature-generation choices, split protocol, and data limitations as
the model being explained.

Install the optional dependency before using SHAP:

```sh
uv add shap
```

For a source checkout:

```sh
uv sync --group dev
```

## When To Use It

Use SHAP when you need to:

- compare the contribution of route distance, hour, zone memberships, and
  spatial features for individual taxi predictions;
- inspect whether sparse pickup/dropoff IDs dominate a model unexpectedly;
- audit graph or neural feature-generation blocks after they have been appended
  as dense columns;
- verify additive prediction decomposition for debugging and reporting.

Feature names matter. If graph or neural embeddings are appended as generated
columns, SHAP explains those generated columns, not the standalone graph or
neural training process that produced them.

## Basic Usage

```python
import shap
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

explainer = shap.Explainer(model, X_train)
explanation = explainer(X_test)
```

This is SHAP's general model-callable route. It is appropriate for dense
column-level explanations, but its selected algorithm depends on SHAP and the
feature count. For a model with many columns, pass a deliberately sized,
representative background sample and choose a SHAP algorithm that fits the
latency and fidelity you need.

`explanation` is a `shap.Explanation`, so it works with SHAP plotting helpers:

```python
shap.plots.beeswarm(explanation)
shap.plots.waterfall(explanation[0])
```

CartoBoost also provides convenience helpers:

```python
explanation = model.explain_shap(X_test, background=X_train)
explainer = model.make_shap_explainer(X_train)
```

The module-level helpers are available from
`cartoboost.explain`; estimator methods are the preferred application
interface because they preserve CartoBoost's fitted feature metadata.

## Additive Weight Decomposition

By default, SHAP decomposes predictions over input features. CartoBoost can also
decompose fitted additive prediction weights: the initial prediction and one
component per fitted tree.

```python
explanation = model.explain_shap(
    X_test,
    background=X_train,
    decomposition="weights",
)
```

The explanation feature names are `init_prediction`, `tree_0`, `tree_1`, and so
on. This is an exact closed-form SHAP decomposition: CartoBoost centers each
component on its background mean and uses the background prediction as the base
value. It runs directly from native additive values and never selects SHAP's
`PermutationExplainer`, even for models with hundreds of trees. The raw
additive matrix is also available directly:

```python
additive = model.predict_additive_values(X_test)
prediction = additive.sum(axis=1)
```

The direct explainer also provides the conventional SHAP values array when a
pipeline stores arrays rather than `shap.Explanation` objects:

```python
explainer = model.make_shap_explainer(
    X_train,
    decomposition="weights",
)
component_shap_values = explainer.shap_values(X_test)
```

For each row and component, this is `component_value -
mean(background_component)`. `explainer.expected_value` is the sum of the
background component means. Consequently, `expected_value +
component_shap_values.sum(axis=1)` exactly reconstructs `model.predict(X)` up
to floating-point precision.

## Choosing A Decomposition

| Need | Use | Runtime behavior | Attribution meaning |
| --- | --- | --- | --- |
| Which input columns moved this prediction? | Default `decomposition="features"` | Native TreeExplainer for dense axis trees; SHAP's general adapter for structured routing. | Input columns, including any generated dense columns. |
| Store a fast, exact audit trail across a large ensemble | `decomposition="weights"` | One native additive-values call; no permutation sampling or repeated prediction calls. | Initial prediction plus each fitted tree contribution. |
| Analyze sparse-set IDs | Default decomposition with both `sparse_sets` and `background_sparse_sets` | SHAP runs on CartoBoost's binary sparse-ID representation. | Dense columns and active sparse IDs such as `pickup_zone=132`. |

Weight decomposition is not a substitute for feature attribution: it answers
which fitted ensemble components contributed, rather than which original
columns caused routing through those components. It is the recommended path
when an existing pipeline previously stored a `TreeExplainer`-style per-tree
audit and needs predictable latency.

## TreeExplainer Support

For fitted dense models whose selected trees use ordinary axis-aligned splits
and constant leaves, `make_shap_explainer` and `explain_shap` automatically use
SHAP's native `TreeExplainer`. CartoBoost exports the trained Rust tree
ensemble, including the initial prediction, learning-rate-scaled leaves,
thresholds, missing-value direction, and node weights, directly into SHAP's
tree representation. The resulting object exposes the normal TreeExplainer
methods such as `shap_values` and returns a standard `shap.Explanation`.

```python
explainer = model.make_shap_explainer(X_train)
explanation = explainer(X_test)
feature_values = explainer.shap_values(X_test)
```

The model object itself is still not passed to `shap.TreeExplainer(model)`:
SHAP only recognizes its built-in library classes at that entry point. Use the
CartoBoost helper, which selects the native tree backend automatically.

CartoBoost's structured splitters, sparse-set routing, fuzzy routing, and
linear leaves have semantics beyond an axis-aligned constant-leaf ensemble.
They remain supported by the default feature SHAP path, but are not converted
lossily into a TreeExplainer model. Their helper call uses the general SHAP
adapter with the supplied background data; choose its algorithm and background
size explicitly when runtime matters. Use `decomposition="weights"` for the
always-exact, low-latency tree-component audit across every supported CartoBoost
model.

## Sparse Sets

Models trained with `sparse_sets=` can be explained through the CartoBoost
helper. Sparse IDs are exposed to SHAP as binary features named `column=id`.
This makes pickup/dropoff memberships auditable while preserving the model's
sparse-list training contract.

```python
explanation = model.explain_shap(
    X_test,
    background=X_train,
    sparse_sets={"taxi_zones": taxi_zones_test},
    background_sparse_sets={"taxi_zones": taxi_zones_train},
)
```

For reusable explainers, pass the background sparse sets when creating the
explainer:

```python
explainer = model.make_shap_explainer(
    X_train,
    sparse_sets={"taxi_zones": taxi_zones_train},
)
```

## Additivity Check

For regression, SHAP values should add back to the model prediction:

```python
prediction = model.predict(X_test)
reconstructed = explanation.base_values + explanation.values.sum(axis=1)
```

This additivity property is the main sanity check for dense and sparse-set
explanations.

## Current Limits

- CartoBoost estimators are callable after fitting, so `shap.Explainer(model,
  background)` works directly for dense prediction workflows.
- `decomposition="weights"` is CartoBoost's fast exact explanation path. It
  attributes predictions to the initial value and fitted tree components, not
  to original input columns. Use the default feature decomposition when you
  need column-level attributions.
- Dense axis-tree models use SHAP's native `TreeExplainer` through
  `make_shap_explainer` and `explain_shap`. Calling `shap.TreeExplainer` on a
  CartoBoost estimator directly is not supported; use the CartoBoost helper.
- Dense Python, NumPy, and pandas inputs are supported through existing
  estimator input handling.
- Sparse-set models are supported through CartoBoost helpers because they need
  the sparse-ID encoding described above.
- SHAP explains generated graph or neural columns only after those columns are
  part of the model input; standalone graph and neural artifacts have their own
  modeling contracts.
