# CartoBoost Documentation

[![PyPI](https://img.shields.io/pypi/v/cartoboost.svg)](https://pypi.org/project/cartoboost/)
[![Python](https://img.shields.io/pypi/pyversions/cartoboost.svg)](https://pypi.org/project/cartoboost/)
[![CI](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/ci.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/ci.yml)
[![Docs](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/pages.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/pages.yml)
[![Release](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/release-version.yml/badge.svg)](https://github.com/TheCulliganMan/CartoBoost/actions/workflows/release-version.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/TheCulliganMan/CartoBoost/blob/main/LICENSE)

CartoBoost is a Python toolkit for structured regression, classification,
ranking, and forecasting when time, location, route structure, or repeated IDs
are part of the signal. The docs focus on how to use the model families, what
each one is good for, and which baseline to compare against.

## What It Is Good For

- Row-level tabular modeling for fare, duration, risk, or demand targets.
- Forecasting demand and count series with rolling-origin validation.
- Graph modeling when direction or relationship structure matters.
- Learned ID embeddings when stable identifiers carry repeated residual signal.
- Leakage-aware evaluation against strong baselines on the same split.

## Start Here

- [Getting Started](getting-started.md): first run, install paths, and local workflow.
- [Choose A Model](user-guide/model-types.md): pick the right entry point.
- [CartoBoost Boosting Model Guides](user-guide/boosting-models/index.md): row-level tree models.
- [CartoBoost Forecasting Model Guides](user-guide/forecasting-models/index.md): one guide per forecast family.
- [CartoBoost Graph Model Guides](user-guide/graph-models/index.md): directed movement, link prediction, and graph features.
- [CartoBoost Neural Model Guides](user-guide/neural-models/index.md): standalone ID embeddings and embedding features.
- [Benchmark Overview](benchmarks/index.md): current benchmark evidence and limits.

## Reference

- [Forecasting](forecasting.md): frame contracts, backtesting, artifacts, and shared forecast rules.
- [Feature Catalog](feature_catalog.md): full capability map.
- [Python API](reference/python-api.md): public classes and methods.
- [CLI Reference](reference/cli.md): command behavior.

## Install

```sh
uv add cartoboost
```

Optional extras:

```sh
uv add "cartoboost[explain,optuna,polars,onnx]"
```
