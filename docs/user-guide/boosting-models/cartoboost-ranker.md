import {RegressionModelExample} from '@site/src/components/ModelingLabClient';

# CartoBoost Ranker

Use `CartoBoostRanker` when rows are only comparable within a query group.
Typical uses are candidate ordering, route ranking, or planning contexts where
only within-group comparisons matter.

The CartoBoost ranker uses pairwise logistic or LambdaRank objectives and
reports grouped metrics from predictions.

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

## Browser WASM Example

The browser bundle currently exposes the shared boosted-tree runner through
`runRegressionModel`. Use this example to inspect WASM split behavior and model
visualization; use the Python ranker API above when query groups and ranking
metrics are required.

<RegressionModelExample title="Boosted tree browser ranking analog" mode="axis" loss="l2" />

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
