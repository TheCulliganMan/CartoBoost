# CartoBoost workspace audit — 2026-11-24

This repository-wide audit covers the 12 Cargo crates, Python interfaces,
CLI, WASM/browser integration, packaging, dependencies, tests, and build tooling.
Changes are based on `d223db7414664b57e74da4a74f1627a5950074b2` and are
uncommitted. No release, push, or deployment was performed.

## Correctness and compatibility

The reported native NumPy bug was real. A Fortran-contiguous array could pass
`as_slice()` and then be interpreted using row-major indexing. For example,
`[[1, 2], [3, 4]]` has Fortran memory order `[1, 3, 2, 4]`; treating that memory
as row-major changes the observations. The native adapter now borrows C-order
buffers and copies other layouts in logical row order. Regression tests cover
Fortran, reversed, and strided inputs, fitting, predictions, contributions, and
geospatial coordinates. The same nine layout/adapter tests pass with NumPy
1.26.4 and 2.5.5.

Public estimator signatures and valid artifact formats were preserved. New
native classifier array margins and metric bindings are additive. Correctness
fixes intentionally change previously incorrect results; invalid inputs that
were truncated, silently altered, or could panic now fail explicitly.

| Area | Audited and implemented changes |
| --- | --- |
| Core data, trees, prediction, explanations | Validated dataset deserialization and sparse normalization; checked schema and dimension bounds; removed repeated forest scans, dense-row copies, SHAP scratch allocation, and graph penalty cloning. |
| Objectives, classifier, ranker, metrics | Checked class IDs, group sizes, objective dimensions and finite values; reused ranking sorts; moved weighted/graph ROC-AUC and graph MRR into Rust; preserved ties and each API's empty-group conventions. |
| Artifacts | Explicitly flush writes and propagate errors; serialize borrowed forests; verify schema hashes and weights envelope versions; checked loaders validate tree feature indices, linear leaf shapes, fuzzy/periodic parameters, classifier dimensions and schema layout. |
| Forecasting and probability | Materialize only requested rolling folds; preserve original Python row indices; correct conformal quantile rank and reject nonfinite residuals; propagate native training errors; validate forecast artifacts and interval/quantile inputs. |
| Geo core and geostatistics | Partial nearest-neighbor selection and KD medians; buffered spatial CV avoids a full CPU distance matrix; reject duplicate panel times, leakage across cutoffs, and invalid solver results. ResidualNNGP fits an independent estimator instead of mutating the caller's object. |
| Spatial economics and geo causal | Normalize extreme finite spatial weights safely; fail clearly for singular/nonfinite systems; review fitting, effects, validation and persistence paths. |
| Geo spatiotemporal and neural forecasting | Backtests slice actual folds and all covariates/masks/weights; check horizon arithmetic and prediction shape; reject duplicate panel cells and changing static covariates; reduce graph-distance diagnostics from quadratic storage to per-source storage. |
| Neural and graph | Backpropagate through original weights; stabilize seeded embedding reductions; preserve full-width integer IDs; validate network artifacts; fix weighted self-loop alignment, pair-node collisions and custom transformer configuration. OOF embeddings now fit residual baselines independently per fold, including weighted, sparse and induced-graph inputs. |
| Accelerator | Improve CPU dense cache locality; reuse activations/exponentials; batch neural dense execution; check tensor/CSR overflow and malformed dimensions. Optional backend source was reviewed and compiled where supported. |
| PyO3 and Python adapters | Correct NumPy layouts; avoid numeric-to-object boxing; use array classifier margins; restore documented graph exports; preserve categorical plus sparse schemas, fitted ranker selectors and legacy classifier labels. Cache zero-depth constants from Rust instead of duplicating loss mathematics in Python. |
| Python persistence and integrations | Restrict legacy forecast pickle reconstruction, including cached extension opcodes; avoid copying refit inputs when native persistence exists; preserve DuckDB/Polars SHAP feature/data alignment; remove the deprecated Plotly append call. |
| CLI | Reject missing/duplicate options and unterminated CSV quotes; correctly escape all JSON controls and CSV carriage returns. |
| WASM and docs UI | Validate hidden network artifacts; recover failed forecast workers, settle pending requests and revoke Blob URLs; handle stale initialization; bundle MapLibre's real ESM worker and resolve its SSR export. |

The OOF leakage regression perturbs only a held-out fold's targets and verifies
that the embedding features for those held-out rows remain exactly unchanged.
The previous implementation changed those features by up to 1.003928 in the
deterministic reproduction.

## Measured performance

Measurements are local synthetic microbenchmarks on an Apple M5/macOS ARM64
host, Rust 1.99.0 and Python 3.12.13. They establish improvements in these
operations, not end-to-end application or real-data quality claims. Baselines
were captured before changing the measured hot paths. Correctness checks use
exact parity where applicable and numeric tolerances for metric accumulation.

Rust values below are Criterion mean estimates in milliseconds: 10 samples,
one-second warmup, two-second requested measurement. Prediction measurements
were noisy; checked axis prediction had no statistically significant change
(`p=0.62`) and is not counted as a demonstrated improvement.

| Operation and fixture | Before ms | After ms | Ratio |
| --- | ---: | ---: | ---: |
| CPU dense, 256 × 128 × 128 | 5.1570 | 0.7548 | 6.83× |
| CPU dense, 32 × 32 × 32 | 0.0333 | 0.0132 | 2.52× |
| CPU softmax, 1,024 × 32 | 0.4953 | 0.2074 | 2.39× |
| CPU tanh training, 64 × 32 × 32 | 1.6310 | 1.0677 | 1.53× |
| Native rolling folds, last 3 of 8 lanes × 2,400 hours | 2451.1136 | 0.9130 | 2684.75× |
| Nearest 16 neighbors from 10,000 points | 0.4341 | 0.0480 | 9.05× |
| Fuzzy prediction, 1,024 × 8, 100 trees | 7.1309 | 0.9716 | 7.34× |
| Linear-leaf prediction, same dimensions | 4.3518 | 1.0225 | 4.26× |
| Combined ranking metrics, 4,096 rows, 64 groups | 0.5892 | 0.3218 | 1.83× |
| Schema validation, 4,096 features | 24.2216 | 0.1326 | 182.70× |
| Schema validation, 64 features | 0.0100 | 0.0023 | 4.41× |
| Checked axis prediction, same prediction dimensions | 1.8076 | 1.6101 | No significant change |

The large rolling-fold gain comes from selecting the requested final origins
before constructing folds, rather than constructing thousands and discarding
nearly all of them. It does not accelerate fitting a model by that ratio.

Python comparisons use alternating before/after executions and report medians:

| Operation | Before ms | After ms | Ratio |
| --- | ---: | ---: | ---: |
| Numeric fit adapter, 100,000 × 8 float64, 7 runs | 27.256 | 0.190 | 143.33× |
| Python rolling splitter, 4,800 rows/1,200 times/4 series, last 3, 6 runs | 253.604 | 1.416 | 179.04× |
| Weighted ROC-AUC, 8,192 rows, 3 runs | 426.851 | 3.924 | 108.79× |
| Graph ROC-AUC, 8,192 rows, 3 runs | 27.659 | 0.401 | 69.02× |
| Graph MRR, 8,192 rows, 3 runs | 67.275 | 0.805 | 83.60× |

The numeric adapter preserves the original float64 buffer. ROC-AUC now uses
O(n log n) sorting and graph MRR uses O(n) rank counting instead of quadratic
Python work. Fixing OOF leakage adds necessary per-fold model fitting; that
correctness change is not presented as a speed improvement.

Reproduce the Rust comparison with the same added benchmark fixtures on the
before and after implementations:

```sh
cargo bench -p cartoboost-core -p cartoboost-geostats -p cartoboost-accelerator \
  --bench prediction --bench training --bench data_loading --bench forecasting \
  --bench geostats --bench accelerator -- \
  'axis_checked|fuzzy_dataset|linear_dataset|rolling_origin_last_3|geostats_neighbors_10000_k16|ranking_metrics|schema_validate|cpu_' \
  --save-baseline audit-before --sample-size 10 --warm-up-time 1 --measurement-time 2
# On the changed implementation, replace --save-baseline with --baseline.
```

Local Python comparison scripts and raw measurements are retained in
`/tmp/cartoboost-audit-numpy-adapter.py`,
`/tmp/cartoboost-python-splitter-benchmark.py`,
`/tmp/cartoboost-metric-speed-audit.py` and their adjacent JSON/log outputs.
Frozen original Python sources are retained alongside them. Rust mean estimates
and confidence intervals were copied to
`/tmp/cartoboost-audit-rust-benchmarks.json` before Cargo cleanup.

## Validation

The final full Python run passed **1,053 tests with no skips**, with **78.75%**
coverage (the required minimum is 35%). The Rust workspace passed **896 tests**
across 38 test/doc-test groups. Rust 1.85 CPU compilation, all benchmark target
builds, fuzz-harness compilation, TypeScript and production WASM/site builds
passed. NumPy 2.5.5 also passed the nine layout/adapter tests in a separate
minimal environment.

The required pre-commit run passed Cargo formatting, Clippy with warnings denied,
Ruff formatting/checks, Python type checks and lockfile consistency. Production
Webpack emits two upstream MapLibre dynamic-URL warnings; the real worker is
explicitly emitted and configured, and the build succeeds.

The maintained synthetic validation scripts passed 9 splitter phases, 4 lane
phases and 8 v1 phases. Segmentation proof generation now actually selects
the named native algorithm; affected images were regenerated. The diagonal
render was visually inspected.

Public API registry checks, nine artifact compatibility roundtrips, executable
docs examples, release/version gates, maintained geo evidence checks and
performance-manifest checks passed. These checks do not constitute rerunning
the maintained real NYC benchmark datasets. A bounded 10,000-row/two-thread/
four-estimator scale smoke passed; it is not the fixed-host million-row release
qualification. Cold package import measured 14.8 ms against its 350 ms gate.

The full validation commands include:

```sh
PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --workspace --locked
PYO3_PYTHON="$PWD/.venv/bin/python" cargo +1.85.0 check --workspace --no-default-features --locked
cargo bench --workspace --no-run --locked
cargo check --manifest-path fuzz/Cargo.toml --all-targets
uv run --no-sync --group dev pytest -q -rs --cov=python/cartoboost --cov-report=term --cov-fail-under=35
UV_NO_SYNC=1 uv run --group dev pre-commit run --all-files
npm run typecheck
npm run build
npm run test:ui
```

Optional SHAP, Plotly, ONNX, libpysal and spreg integrations were installed for
testing without adding required runtime dependencies. `UV_NO_SYNC` preserves
that local test environment. The base package remains NumPy-only.

## Dependencies and workspace

- npm's dependency audit went from 54 findings to zero. Docusaurus, MapLibre,
  wasm-pack and affected transitive dependencies were updated and built.
- The installed Python environment, including tested optional integrations,
  has no known findings in the final pip-audit scan. Development lockfile
  updates address filelock, Pillow and virtualenv advisories.
- Cargo dependency checks cover both the main workspace and standalone fuzz
  package. Existing informational advisory exceptions remain explicit.
  The fuzz package now declares the repository's MIT license; a narrow
  `libfuzzer-sys` exception recognizes its bundled NCSA-licensed runtime.
- Shared edition/license/MSRV and Rayon/tempfile declarations are centralized;
  unused workspace dependency declarations were removed. CPU builds check on
  Rust 1.85; optional WebGPU dependencies require a newer compiler and retain
  a separate stable/default-feature CI job.
- Removed 63 verified uncompiled duplicate Rust files, totaling 44,760 lines.
  Active module/include paths and references were checked before removal and
  workspace builds/tests pass afterward.
- Removed the unused stale pnpm lockfile and placeholder build-policy file;
  npm remains the documented and CI-used package manager.
- Updated stale tests to the existing v0.3 public contract while retaining
  native algorithm coverage. No deliberately retired public API was restored.
- Validation reports and the scale smoke artifact are retained under
  `/tmp/cartoboost-audit-validation-evidence` and
  `/tmp/cartoboost-audit-scale-performance.json` before removing build output.
- Final `cargo clean` removed 37,579 files (14.9 GiB reported); the standalone
  fuzz clean removed another 548 files (216.2 MiB). Generated test, formatting,
  Hypothesis, coverage and site-build caches were removed. Development
  environments and the rebuilt distributable WASM assets were retained.

## Boundaries and remaining tradeoffs

- Execution evidence covers macOS ARM64 CPU and Node WASM. CUDA, ROCm,
  DirectML, other operating systems and other Python interpreter versions were
  not executed locally. GPU source review/compilation is not hardware evidence.
- The production site builds, its real worker asset is emitted, and 17
  WASM/worker/sample tests pass. Live browser/map verification was unavailable:
  the Mac was locked and the browser inventory contained no usable browser.
- Generic composite artifacts can import recorded third-party estimator
  loaders and remain trusted-source-only. Restricted forecast fit-data loading
  does not turn all arbitrary artifacts into a safe untrusted interchange.
- Direct Serde deserialization or mutation of public Rust model structs can
  bypass checked loader validation. Use checked model loaders for persisted
  models; broader adversarial resource limits are not established by this pass.
- Existing infallible linear-leaf fitting retains its constant-leaf fallback
  on solver failure. Changing that policy needs a compatible fallible API and
  an explicit model-behavior decision. Exhaustive periodic split search also
  remains expensive; its search semantics were preserved.
- No claims about improved real-data prediction quality or release readiness
  are made from these local microbenchmarks. Real benchmark refreshes and
  the release platform matrix remain necessary before publishing.
