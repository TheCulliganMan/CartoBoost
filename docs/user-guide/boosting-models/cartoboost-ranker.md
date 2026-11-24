import {RegressionModelExample} from '@site/src/components/ModelingLabClient';

# CartoBoost Ranker

Order candidates within a decision using `CartoBoostRanker`. For example, rank
candidate pickup zones for the same taxi dispatch query. Scores are comparable
within that query; they do not represent an absolute fare or probability.

Choose pairwise logistic or LambdaRank objectives, keep query groups together,
and evaluate the ordering with grouped ranking metrics.

## Python Example

```python
from cartoboost import CartoBoostRanker

ranker = CartoBoostRanker(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=4,
    split_policy="structured",
    objective="lambdarank",
)
ranker.fit(X_train, relevance_train, groups=query_sizes_train)
scores = ranker.predict(X_test)
metrics = ranker.score_groups(X_test, relevance_test, groups=query_sizes_test)
```

## Interactive Example

Explore shared tree-splitting behavior on a numeric target below. This browser
example runs regression; use the Python ranker above for query groups and
ranking metrics.

<RegressionModelExample title="Explore axis-aligned tree splits" mode="axis" loss="l2" />

Rows for each query must be contiguous. Pass `groups` as group sizes or
contiguous query ids, or set `group_col` when the query id is a column in `X`.

## Use When

| Need | Better first choice |
| --- | --- |
| Rank candidates within each query. | `CartoBoostRanker` |
| Predict an absolute score or amount. | `CartoBoostRegressor` |
| Predict a class label or probability. | `CartoBoostClassifier` |
| Forecast future time points. | Forecasting models |

## Validation

Use ranking metrics that match the decision: NDCG for graded relevance, MAP
for retrieval-style relevance, and MRR when the first useful candidate matters
most. Compare against simple popularity or dense tabular baselines under the
same group split.

For workflow and method details, see [Python API Reference](../../reference/python-api.md).

## Limitations

- Ranking labels are meaningful only within each query group.
- Group boundaries must remain intact during splitting and prediction.
- Offline ranking metrics do not by themselves establish downstream decision value.
