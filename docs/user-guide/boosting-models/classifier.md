# CartoBoost Classifier

Use `CartoBoostClassifier` when each row has a discrete taxi-domain label and
the decision boundary may depend on pickup/dropoff coordinates, hour, route
memberships, or sparse zone signals. Examples include airport-trip flags,
high-delay buckets, service-risk classes, and route status labels.

The classifier fits binary logistic loss for two classes and multiclass
logistic loss for three or more classes. It exposes sklearn-style `fit`,
`predict`, `predict_proba`, `decision_function`, `class_weight`, `save`, and
`load` behavior.

## Basic Fit

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
clf.fit(X_train, airport_trip_flag)
prob_airport = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]
```

## Validation

Report classifier quality with logloss plus threshold-free metrics such as
ROC-AUC or PR-AUC when the positive class is rare. Compare against dummy and
standard tabular baselines on the same train/test split before interpreting a
CartoBoost gain.

For a workflow-oriented introduction, see
[Classification Quickstart](../classification-quickstart.md). For full method
contracts, see the [Python API Reference](../../reference/python-api.md).
