import {RegressionModelExample} from '@site/src/components/ModelingLabClient';

# CartoBoost Classifier

Predict a class and its probability with `CartoBoostClassifier`. A taxi-trip
label might indicate whether a journey exceeds a duration threshold; pickup
zone, hour, and route features can help define the decision boundary.

The estimator uses binary logistic loss for two classes and multiclass logistic
loss for three or more. Check probability calibration on held-out trips before
using a probability as a decision threshold.

## Python Example

```python
from cartoboost import CartoBoostClassifier

clf = CartoBoostClassifier(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=4,
    min_samples_leaf=20,
    split_policy="structured",
    class_weight="balanced",
)
clf.fit(X_train, binary_label)
prob_positive = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]
```

## Interactive Example

Explore shared tree-splitting behavior on a numeric target below. This browser
example runs regression; use the Python classifier above to train labels and
evaluate class probabilities.

<RegressionModelExample title="Explore boosted-tree splits" mode="auto" loss="log_l2" />

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

## Limitations

- Probability calibration must be checked on held-out rows.
- Rare classes need enough examples in every training and evaluation split.
- Spatial or temporal structure is used only when declared in the feature schema.
