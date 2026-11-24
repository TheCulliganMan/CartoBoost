---
title: Documentation
sidebar_label: Documentation
hide_title: true
description: CartoBoost Python documentation, model guides, examples, API reference, and benchmarks.
---

import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';

<span className="docs-kicker">Documentation</span>

# CartoBoost

<p className="docs-lead">Python APIs for structured boosting, forecasting, graph models, and spatial analysis.</p>

Use these guides to prepare inputs, fit a model, choose a validation split, and
inspect predictions. Examples use NYC taxi fares, trip durations, and pickup demand.

<div className="docs-route-grid">
  <Link className="docs-route-card" to="/docs/getting-started">
    <span className="docs-route-label">Tutorial</span>
    <h2>Quickstart</h2>
    <p>Go from NumPy arrays to a fitted regressor, a temporal holdout, and a saved artifact.</p>
  </Link>
  <Link className="docs-route-card" to="/docs/user-guide/model-types">
    <span className="docs-route-label">Models</span>
    <h2>Model selection</h2>
    <p>Match trip-level predictions, demand series, and connected zones to a model.</p>
  </Link>
  <Link className="docs-route-card" to="/modeling-lab">
    <span className="docs-route-label">Interactive</span>
    <h2>Browser examples</h2>
    <p>Fit a sample and inspect predictions in the Modeling Lab. No Python setup required.</p>
  </Link>
  <Link className="docs-route-card" to="/docs/benchmarks">
    <span className="docs-route-label">Evaluation</span>
    <h2>Benchmarks</h2>
    <p>See the datasets, splits, baselines, timings, and limits behind each result.</p>
  </Link>
</div>

## Install CartoBoost

```sh
uv add cartoboost
```

The base package requires NumPy. Install dataframe, plotting, or geospatial
integrations only when you need them. [Check installation details](installation.md),
then follow the [complete first-fit example](getting-started.md).

## Model families

| You want to predict… | Start with… | What the guide helps you decide |
| --- | --- | --- |
| A trip's duration or fare | [Boosting models](user-guide/boosting-models/index.md) | Which numeric, periodic, spatial, or sparse features belong in the model. |
| Future pickup demand | [Forecasting](forecasting.md) | How to construct series, choose a horizon, and backtest at realistic cutoffs. |
| Outcomes across connected zones | [Graph models](user-guide/graph-models/index.md) | Whether observed relationships add signal beyond a tabular baseline. |
| Effects associated with repeated IDs | [Neural embeddings](user-guide/neural-models/index.md) | How learned pickup or dropoff representations behave on held-out IDs. |
| Pair, sequence, or distribution outcomes | [Deep models](user-guide/deep-models.md) | Which input contract fits the task and what the browser examples demonstrate. |
| Values between observed locations | [Geostatistics](user-guide/geostatistics-models.md) | How spatial covariance and uncertainty enter interpolation. |
| Effects of a geographic intervention | [Geo-causal models](user-guide/geo-causal-models.md) | How to define the comparison and assess spillover. |

## Validation

Predicting tomorrow's trips and predicting trips in an unseen zone are different
tasks. A random split does not answer both. Choose a temporal, spatial, or grouped
holdout before comparing models, and fit transformations on training data only.

[Spatial validation](user-guide/spatial-cv-best-practices.md) explains blocked and
buffered splits. [Geo evaluation](user-guide/geo-evaluation-standard.md) covers
reproducible split manifests. The [benchmark reports](benchmarks/index.md) show
how specific models performed under stated conditions.

## API and integration reference

Use the [Python API](reference/python-api.md) for public methods, the
[feature schema](feature_schema.md) for typed inputs, and
[model artifacts](model_artifact.md) for persistence. The
[CLI reference](reference/cli.md) covers command-line workflows.

Returning from an older version? Read the [0.3 migration guide](migration-v0.3.md).
For tools and coding assistants, <a href={useBaseUrl('/docs/llms.txt')}>llms.txt</a> indexes the
maintained documentation.
