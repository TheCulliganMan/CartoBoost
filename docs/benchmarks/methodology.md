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

## v0.2 Modeling Gates

For the v0.2 spatial boosting release, the maintained benchmark set should
include:

- binary spatial classification versus dummy and tabular baselines;
- grouped ranking versus baseline scoring;
- native categorical versus one-hot preprocessing;
- random CV versus buffered spatial CV leakage comparison;
- regression benchmark showing no more than 5 percent slowdown on the existing
  regressor workload.

The deterministic smoke harness for these release gates is:

```sh
PYTHONPATH=python uv run --group dev python scripts/run_v02_modeling_benchmarks.py \
  --output-dir target/v02-benchmarks \
  --seed 42 \
  --sample-size 240 \
  --n-estimators 24
```

To turn the regression fit-speed guard from a current-code repeatability check
into a before/after slowdown check, pass a prior artifact from the same harness:

```sh
PYTHONPATH=python uv run --group dev python scripts/run_v02_modeling_benchmarks.py \
  --output-dir target/v02-benchmarks \
  --regression-baseline-json target/v02-baseline/v02_modeling_benchmark.json
```

It writes:

- `target/v02-benchmarks/v02_modeling_benchmark.json`
- `target/v02-benchmarks/v02_modeling_benchmark.jsonl`
- `target/v02-benchmarks/v02_modeling_benchmark.md`

The output is synthetic taxi-shaped smoke evidence. It proves that the release
gates execute and fail loudly when they should; it does not replace real NYC
TLC benchmark artifacts for public quality claims. When no
`--regression-baseline-json` is supplied, the regression guard records
`evidence_kind=current_code_repeatability` and should not be interpreted as a
historical slowdown comparison.

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

The stable model registry is audited separately:

```sh
uv run --group dev python scripts/check_public_api_contract.py
```

That audit requires every registered stable model to expose `fit`, `predict`,
`score`, `save`, `load`, `get_params`, `set_params`, and typed metadata. It also
runs save/load prediction-drift checks for representative registry entries
covering boosters, AutoGeoModel, stacking, NNGP residuals, and conformal
interval wrappers.

Model artifact compatibility is audited separately:

```sh
uv run --group dev python scripts/check_artifact_compatibility.py
```

That audit saves representative stable registry artifacts, checks explicit
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

That audit distinguishes final AutoGeoModel acceptance evidence from synthetic
gates, incomplete manifests, and real non-AutoGeo benchmark artifacts. The
current audit is intentionally allowed to pass while `acceptance_passed` is
false so CI can block evidence misclassification without pretending the final
3-of-5 real benchmark claim is complete.

The NYC TLC quality harness accepts `autogeo` in `--models` and records the
selector's chosen family, candidate evaluations, and train-only inner validation
metadata inside the same scorecard as the fixed CartoBoost and external
baseline rows. Those rows count toward final acceptance only after the real TLC
artifact is regenerated with leakage-safe split manifests and the official
evidence audit recognizes the public result.

Low-level Rust performance thresholds are audited separately:

```sh
uv run --group dev python scripts/check_performance_thresholds.py
```

That audit reads `benches/benchmark_summary.json` and fails when a maintained
training, prediction, serialization, or data-loading benchmark exceeds its
recorded `max_mean_ns` threshold.

CI also runs the deterministic AutoGeoModel benchmark gate:

```sh
PYTHONPATH=python uv run --group dev python scripts/run_autogeo_benchmark_gate.py \
  --output-dir target/autogeo-gate \
  --sample-size 90 \
  --n-splits 5
```

That gate emits JSON, JSONL, and Markdown scorecards for official-style
synthetic workloads. It requires leakage-safe split hashes, runtime, memory,
baseline comparisons, residual Moran's I for spatial rows, interval coverage
and width, regional diagnostics, and save/load prediction drift. It is a CI
regression gate for benchmark plumbing and AutoGeoModel behavior; real public
quality claims still require the maintained real-data artifacts.
