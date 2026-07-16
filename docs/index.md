# CartoBoost Documentation

[![PyPI](https://img.shields.io/pypi/v/cartoboost.svg)](https://pypi.org/project/cartoboost/)
[![Python](https://img.shields.io/pypi/pyversions/cartoboost.svg)](https://pypi.org/project/cartoboost/)
[![CI](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/ci.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/ci.yml)
[![Docs](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/pages.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/pages.yml)
[![Release](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/publish-pypi.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/TheCulliganMan/CartoBoost/blob/main/LICENSE)

CartoBoost is a Python toolkit for structured regression, classification,
ranking, and forecasting when time, location, route structure, or repeated IDs
are part of the signal. These docs are organized around practical modeling
workflows: preparing data, choosing an estimator, validating it without
leakage, interpreting results, and deploying reproducible artifacts.

## What It Is Good For

- Row-level tabular modeling for fare, duration, risk, or demand targets.
- Forecasting demand and count series with rolling-origin validation.
- Graph modeling when direction or relationship structure matters.
- Learned ID embeddings when stable identifiers carry repeated residual signal.
- Leakage-aware evaluation against strong baselines on the same split.

## Choose Your Path

- **First model:** [Getting Started](getting-started.md) covers installation,
  fitting, validation, baseline comparison, and persistence.
- **Model selection:** [Choose A Model](user-guide/model-types.md) maps data and
  prediction tasks to the appropriate estimator family.
- **Tabular and spatial ML:** [Boosting Model Guides](user-guide/boosting-models/index.md)
  cover regression, classification, ranking, categorical data, and structured splits.
- **Time series:** [Forecasting](forecasting.md) covers validated frames,
  rolling-origin backtesting, metrics, artifacts, and CLI workflows.
- **Production integration:** [Python API](reference/python-api.md),
  [CLI Reference](reference/cli.md), and [Model Artifacts](model_artifact.md).
- **Evidence:** [Benchmarks](benchmarks/index.md) reports commands, datasets,
  splits, metrics, and limitations.

## Model Families

- [CartoBoost Boosting Model Guides](user-guide/boosting-models/index.md): row-level tree models.
- [CartoBoost Forecasting Model Guides](user-guide/forecasting-models/index.md): one guide per forecast family.
- [Geo-Causal Experiment Models](user-guide/geo-causal-models.md): synthetic DID, GeoLift-style design, and spillover diagnostics.
- [CartoBoost Graph Model Guides](user-guide/graph-models/index.md): directed movement, link prediction, and graph features.
- [CartoBoost Neural Model Guides](user-guide/neural-models/index.md): standalone ID embeddings and embedding features.
- [Benchmark Overview](benchmarks/index.md): current benchmark evidence and limits.

Stable estimators are available from the package root. The
`cartoboost.models` registry provides machine-readable metadata for tools that
need to enumerate supported model surfaces.

## Reference

- [Forecasting](forecasting.md): frame contracts, backtesting, artifacts, and shared forecast rules.
- [Feature Catalog](feature_catalog.md): full capability map.
- [Sparse Features](sparse_features.md): sparse sets, H3/S2 point cells, and decoded route-cell encoders.
- [Python API](reference/python-api.md): public classes and methods.
- [CLI Reference](reference/cli.md): command behavior.

## Install

```sh
uv add cartoboost
```

Optional dependencies are installed directly:

```sh
uv add shap optuna polars onnx
```
