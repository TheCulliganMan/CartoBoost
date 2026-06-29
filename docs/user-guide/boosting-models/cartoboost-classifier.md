import {RegressionModelExample} from '@site/src/components/ModelingLabClient';

# CartoBoost Classifier

Use `CartoBoostClassifier` for binary or multiclass labels when the decision
boundary may depend on time, location, route membership, or sparse signals.
It fits binary logistic loss for two classes and multiclass logistic loss for
three or more classes.

## Public Contract

```python
from cartoboost import CartoBoostClassifier

clf = CartoBoostClassifier(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=4,
    min_samples_leaf=20,
    splitters=["axis", "diagonal_2d", "gaussian_2d", "periodic:24"],
    class_weight="balanced",
)
clf.fit(X_train, binary_label)
prob_positive = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]
```

## Browser WASM Example

The browser bundle currently exposes the shared boosted-tree runner through
`runRegressionModel`. Use this example to inspect the same splitter, loss, and
visualization path in WASM; use the Python classifier API above for class-label
training and probability calibration.

<RegressionModelExample title="Boosted tree browser classifier analog" mode="auto" loss="log_l2" />

## Use When

| Need | Better first choice |
| --- | --- |
| Binary or multiclass labels. | `CartoBoostClassifier` |
| Numeric target values. | `CartoBoostRegressor` |
| Ordered candidates within query groups. | `CartoBoostRanker` |
| Calibrated uncertainty intervals. | Probabilistic and conformal models |

## Validation

Report CartoBoost classifier quality with logloss plus threshold-free metrics such as
ROC-AUC or PR-AUC when the positive class is rare. Compare against dummy and
standard tabular baselines on the same split.

For workflow and method details, see [Python API Reference](../../reference/python-api.md).
