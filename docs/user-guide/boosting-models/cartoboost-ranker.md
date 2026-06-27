# CartoBoost Ranker

Use `CartoBoostRanker` when rows are only comparable within a query group.
Typical uses are candidate ordering, route ranking, or planning contexts where
only within-group comparisons matter.

The CartoBoost ranker uses pairwise logistic or LambdaRank objectives and
reports grouped metrics from predictions.

## Basic Fit

```python
from cartoboost import CartoBoostRanker

ranker = CartoBoostRanker(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=4,
    splitters=["axis", "diagonal_2d", "gaussian_2d"],
    objective="lambdarank",
)
ranker.fit(X_train, relevance_train, groups=query_sizes_train)
scores = ranker.predict(X_test)
metrics = ranker.score_groups(X_test, relevance_test, groups=query_sizes_test)
```

Rows for each query must be contiguous. Pass `groups` as group sizes or
contiguous query ids, or set `group_col` when the query id is a column in `X`.

## Validation

Use ranking metrics that match the decision: NDCG for graded relevance, MAP
for retrieval-style relevance, and MRR when the first useful candidate matters
most. Compare against simple popularity or dense tabular baselines under the
same group split.

For workflow and method details, see [Python API Reference](../../reference/python-api.md).
