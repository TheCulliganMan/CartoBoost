# Benchmark Methodology

CartoBoost benchmark claims must be tied to real commands, fixed comparable
settings, and recorded artifacts. A benchmark page should let a reader see what
was run, which data were used, which split was evaluated, which models were
compared, and which metric supports the interpretation.

## Required Fields

Each benchmark report should name:

- command and working directory;
- data source and whether the data are real, synthetic, or generated acceptance
  data;
- sample size and task definitions;
- train/test split or CV fold construction;
- model roster and comparable estimator settings;
- comparability audit covering shared rows or horizons, metric roster,
  feature-access policy, skipped requested models, and whether any model
  selection used holdout labels;
- metric table with timing fields;
- artifact paths for JSON, JSONL, markdown, and plots;
- limitations that affect interpretation.

The manifest validator in `benchmarks/runners/manifest.py` also defines the
official geographic benchmark families. A family is claim-ready only when its
tasks declare leakage-safe split manifests, required external competitors,
diagnostic outputs, leaderboard JSON/markdown, fit/predict timing, and peak
memory. CI runs `scripts/check_release_gates.py` to block drift between these
manifests, benchmark dependencies, workflow gates, and release artifact
attestation. The maintained family roster is:

| Family | Required evidence |
| --- | --- |
| NYC TLC zone/lane demand | TLC source hashes, spatial/grouped splits, LightGBM/XGBoost/CatBoost/sklearn baselines, residual Moran's I, block error, regional calibration and interval width. |
| METR-LA / PEMS graph forecasting | Fixed sensor graph hashes, rolling-origin graph splits, PyTorch Geometric/DCRNN baselines, horizon error, graph-distance residual decay. |
| EPA air-quality interpolation | EPA source hashes, buffered monitor holdouts, PySAL/PyKrige/GSTools baselines, variogram and spatial residual diagnostics. |
| California housing sanity | Fixed public sample, strong tabular baselines, spatial block error as a sanity diagnostic only. |
| Synthetic spatial fields | Generator manifest hash, block splits, kriging/spatial-regression baselines, variogram and residual autocorrelation diagnostics. |
| Synthetic graph diffusion | Generator manifest hash, rolling graph splits, graph neural baselines, horizon and graph-distance diagnostics. |
| Synthetic geo-causal lift panels | Generator manifest hash, rolling panel splits, placebo summaries, and known-effect error metrics. |

## v0.3 Acceptance Gates

The maintained beta evidence uses native validation and real comparable
baselines. Stable structured regression claims require duration and fare
out-of-time and spatial-holdout comparisons; lane-demand forecasting requires
at least three leakage-safe rolling origins and an explicit comparison with the
strongest completed external learned baseline. The forecasting artifact gate is
run with:

```sh
PYTHONPATH=python uv run python scripts/check_forecasting_quality_gate.py \
  --artifact docs/assets/nyc_taxi_benchmarks/forecasting_library_benchmark_real.json
```

The release firewall also checks wheel installation, artifact round trips,
import and scale budgets, synchronized versions, benchmark provenance, and
protected publishing. Synthetic smoke runs remain diagnostics for execution
coverage only and cannot replace the maintained real-data artifacts.

Classification reports should include logloss, ROC-AUC or PR-AUC, Brier score,
ECE, fit time, prediction time, and save/load probability drift. Ranking
reports should include NDCG, MAP, MRR, fit time, prediction time, and save/load score drift.
Categorical reports should state the number of categories, chosen
encoding strategy, unknown-category rate, and whether the saved model
round-tripped predictions within tolerance. Unsupported export checks should
assert loud `NotImplementedError` failures for categorical regressor export and
classifier/ranker portable-weight or ONNX export.

## Interpretation Rules

Do not use stale artifacts after changing benchmark-affecting code. If feature
generation, fitting, prediction, metric computation, or split construction
changes, rerun the affected benchmark before updating public claims.

Do not frame benchmark pages around process labels such as cleanup or
provenance. Lead with the current-code result, then show command, data, split,
model roster, metrics, timing, artifact paths, and limits.

## Release Gates

Run the release audit locally before packaging:

```sh
uv run --group dev python scripts/check_release_gates.py
```

The audit validates the official geo benchmark manifest contract, required
benchmark dependency coverage, external install metadata for PyTorch Geometric
baselines, CI commands for public API/serialization/performance coverage, and
distribution provenance attestation in the PyPI workflow.

The stable root estimators are audited separately:

```sh
uv run --group dev python scripts/check_public_api_contract.py
```

That audit requires each stable root estimator to expose `fit`, `predict`,
`score`, `save`, `load`, `get_params`, `set_params`, and typed metadata. It also
runs save/load prediction-drift checks for representative registry entries
covering the native booster, classifier, and ranker contracts.

Model artifact compatibility is audited separately:

```sh
uv run --group dev python scripts/check_artifact_compatibility.py
```

That audit saves representative stable estimator artifacts, checks explicit
schema/version markers including nested artifacts, verifies save/load prediction
drift, and mutates artifact versions to prove unsupported versions fail loudly.

The maintained docs examples are audited separately:

```sh
uv run --group dev python scripts/check_docs_examples.py
```

That audit executes the model-choice, leakage-safe geo split, and conformal
interval examples that public docs use as stable contracts.

Official geo evidence is classified separately:

```sh
uv run --group dev python scripts/check_official_geo_evidence.py
```

That audit distinguishes deferred automatic-selector evidence from synthetic gates,
incomplete manifests, and real non-selector benchmark artifacts. The current
audit passes while the final selector evidence requirement remains explicitly
deferred.

The NYC TLC quality harness does not run an automatic geo selector in v0.3.
The selector is not shipped in v0.3, so no selector benchmark is executed by
the release gate.

Low-level Rust performance thresholds are audited separately:

```sh
uv run --group dev python scripts/check_performance_thresholds.py
```

That audit reads `benches/benchmark_summary.json` and fails when a maintained
training, prediction, serialization, or data-loading benchmark exceeds its
recorded `max_mean_ns` threshold.

The deferred selector gate reports its status and is not part of the v0.3
stable release gate:

```sh
PYTHONPATH=python uv run --group dev python scripts/run_autogeo_benchmark_gate.py \
  --output-dir target/autogeo-gate \
  --sample-size 90 \
  --n-splits 5
```

The command writes a small JSON status artifact and exits successfully to make
the deferral explicit. Real public quality claims still require the maintained
real-data artifacts and a future native selector.
