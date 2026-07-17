import {DeepModelWasmExample, ForecastModelExample, MarketStructureWasmExample} from '@site/src/components/ModelingLabClient';
import {MarketStructureExplorerSample} from '@site/src/components/MarketStructureExplorer';

# Graph and Market Structure Forecasting

`MarketStructureForecaster` is the learned sparse-kernel model for connected,
directional markets. Open the [Modeling Lab](../../../modeling-lab), select
**Deep**, then run **Learned market structure explorer** to inspect forecasts,
inbound/outbound volume, components, and the retained kernels interactively.

## Graph Spatiotemporal Forecasting

Use `DCRNNForecaster`, `GraphWaveNetForecaster`, or `STAEformerForecaster`
when each forecast series is a node in a known directed graph and neighboring
nodes can lead, lag, or diffuse signal into one another. This is for sensor
networks, route flows, road segments, zone flows, equipment networks, or other
panels where the graph is part of the modeling claim.

Do not use a graph forecaster only because node ids exist. The edges should
represent known movement, influence, adjacency, or dependency available at the
forecast cutoff.

## Paper Graph Transformer Profiles

Five native graph-transformer profiles extend the directed graph surface. They
all fit in Rust, make direct multi-horizon predictions, and save/load as native
JSON artifacts. Choose the mechanism you need, then compare it with DCRNN,
GraphWaveNet, STAEformer, and seasonal-naive baselines on the identical
time-ordered split.

| Model | Use when | Native components |
| --- | --- | --- |
| `STGormerForecaster` | Spatial and temporal behavior differs by node or time regime. | Time2Vec-style temporal features, in/out-degree and path features, three causal spatial-temporal attention stages, and independent spatial/temporal routed MoE states. |
| `STGformerForecaster` | A large graph needs retained high-order propagation in one efficient block. | Retained propagation orders, shared-QKV scaling-normalized linear space-time interaction at every order, recursive pointwise interaction. |
| `LSTTNForecaster` | Long history, recurring periods, and immediate dynamics all matter. | 75%-masked patch pretraining, a frozen four-layer Transformer, four-stage dilated long trend, independent forward/backward/adaptive daily and weekly graph diffusion, an eight-layer causal Graph WaveNet short branch, and two-stage feature fusion. |
| `SpatialTemporalGraphGatedTransformerForecaster` | Graph signal should be filtered through stable temporal gates. | Graph convolution, causal temporal attention, GRU reset/update gates. |
| `SpatialShiftGraphonMoEForecaster` | Graph relationships may change between the training and deployment periods. | Input-conditioned graphon experts and softmax graphon mixing. |

`LSTTNForecaster` defaults to a long-horizon hourly configuration: four weeks
of history, a daily period, one week of recent context, and a one-week forecast.
Its native pretraining stage follows the paper's masked sub-series Transformer:
a learned strided patch convolution and learned temporal positions feed four
multi-head Transformer encoder layers; 75% of whole patches are withheld; and
a learned mask token, encoder-to-decoder projection, one Transformer decoder
layer, and linear patch head reconstruct only the masked values with L1 loss.
The pretrained patch encoder is frozen during forecast fitting.

The forecast path contains each spatial and temporal mechanism described by
LSTTN. Four stride-two, three-tap dilated convolution stages use dilations 1,
2, 4, and 8 with GELU and max pooling for long trend. Previous-day and
previous-week frozen Transformer states each enter their own order-two graph
convolution with row-normalized forward diffusion, independently normalized
backward diffusion, and a separate rank-10 learned adaptive adjacency. The
short branch consumes signal and time-of-day channels through the complete
four-block, two-layer Graph WaveNet stack: causal gated temporal convolutions,
structural and adaptive order-two graph diffusion, residual/skip paths, batch
normalization, and the two output projections. Two MLP stages fuse long trend,
daily periodicity, weekly periodicity, and the short state into direct
multi-horizon predictions. Forecast fitting uses every valid rolling origin in
32-window batches, zero-masked inverse-scale MAE, gradient-norm clipping at 3,
and the paper's milestone schedule.

When `GraphTemporalFrame.covariates` is supplied, LSTTN uses its first feature
as the normalized time-of-day channel paired with the traffic signal in Graph
WaveNet. Without covariates it derives the channel from the configured
`periodicity` and the absolute row origin. Covariates must have one consistent,
finite feature width; malformed or empty feature axes are rejected.

Learned patch positions cover the complete configured history. LSTTN rejects a
configuration that cannot supply a complete previous-week patch instead of
silently inserting a zero periodic feature. Frozen representations are cached
only while their estimated size is at most 512 MiB; larger panels use bounded
per-window reconstruction, so cache size does not grow without limit. These
mechanisms depend on the supplied directed CSR graph, not on H3: nodes may be
H3 cells, S2 cells, administrative regions, road sensors, routes, or any other
geography with stable node ids and explicit directed edges.

For another sampling frequency, set `lookback`, `periodicity`,
`recent_window`, and `horizon`
explicitly. All four are measured in frame rows rather than fixed wall-clock
units. For example, an hourly freight-network forecast can use:

```python
model = LSTTNForecaster(
    lookback=24 * 28,
    periodicity=24,
    recent_window=24 * 7,
    horizon=24 * 7,
)
```

Here `horizon` is also expressed in frame rows, so this predicts seven days
when the input frame has hourly timestamps. CartoBoost does not assume a
five-minute sampling interval.

The embedded browser debugger uses an explicitly compact interpretation
fixture: 192 hourly history rows across eight real H3 cells and a six-hour
holdout. It runs the native Rust LSTTN forecast four times in WASM: the full
directed graph, self edges only, recent values replaced by prior-period values,
and a trend-preserving history with recurring variation removed. Those reruns
measure how graph context, immediate dynamics, and recurring history change
each cell's forecast. They are interpretation checks, not long-history
benchmark evidence.

`SpatialShiftGraphonMoEForecaster` derives recurring source environments by
partitioning a traffic cycle into contiguous rank-coherent graph relations. Its
episodic training pass holds out the environment's expert, freezes expert
graphon gradients, trains the mixup router against the remaining experts, and
uses binary Gumbel-Softmax graphon sampling during native fitting.

```python
from cartoboost.forecasting import GraphTemporalFrame, STGormerForecaster

model = STGormerForecaster(
    lookback=12,
    attention_heads=4,
    experts=4,
    horizon=3,
).fit(frame)
forecast = model.predict(3)
report = model.metadata_["architecture_report"]
```

`STGformerForecaster`, `LSTTNForecaster`,
`SpatialTemporalGraphGatedTransformerForecaster`, and
`SpatialShiftGraphonMoEForecaster` use the same `GraphTemporalFrame` contract.
The generic deep facade also routes these profiles through
`SpatioTemporalGraphForecaster(backbone=...)`.

For a directional market lane panel, call
`MarketPanelFrame.as_graph_temporal_frame(...)` with an explicit CSR adjacency,
then fit any of these native graph profiles. The adapter preserves the observed
target exactly and rejects unavailable lane values; it never invents a graph or
imputes targets.

### LSTTN H3 WASM Debugger

<DeepModelWasmExample model="LSTTNForecaster" />

Use the map controls to move through forecast horizons and switch among:

- forecast and observed holdout volume;
- forecast rate relative to the latest observation;
- absolute holdout error;
- signed directed-graph sensitivity;
- signed recent-pulse sensitivity; and
- signed recurring-rhythm sensitivity.

Click an H3 cell to see its history, forecast, observed holdout, ranked native
counterfactual sensitivities, and the fitted architecture report. Toggle 3D
height to read magnitude and graph edges to follow the directed paths available
at the forecast cutoff. Counterfactual sensitivities are not additive feature
attributions: LSTTN's pathways interact, so read each number as the result of
one controlled native rerun.

The Modeling Lab exposes separate runnable Rust/WASM entries for
`STGormerForecaster`, `STGformerForecaster`, `LSTTNForecaster`,
`SpatialTemporalGraphGatedTransformerForecaster`, and
`SpatialShiftGraphonMoEForecaster`. Select **Deep** and choose the model by
name to run the same native profile against the directed taxi-shaped graph.

These profiles are implementation surfaces, not a claim that the included
synthetic checks reproduce published benchmark rankings. Validate any accuracy
claim with your production or real traffic data and a leakage-safe temporal
holdout. The original papers are [STGormer](https://arxiv.org/abs/2408.10822),
[STGformer](https://arxiv.org/abs/2410.00385),
[LSTTN](https://arxiv.org/abs/2403.16495) and the
[authors' reference implementation](https://github.com/GeoX-Lab/LSTTN),
[STGGT](https://doi.org/10.1002/ett.5021), and
[spatial-shift graphon MoE](https://arxiv.org/abs/2410.00373).

For a fixed-origin traffic-graph evaluation on real DCRNN-format inputs, use
the maintained runner. It requires both the original HDF5 time series and its
matching adjacency pickle, records SHA-256 source hashes in the output, and
does not synthesize missing sensors or graph edges.

```bash
uv run --with h5py -- python -m benchmarks.runners.traffic_graph_forecasting \
  --data-h5 /path/to/metr-la.h5 \
  --adjacency-pickle /path/to/adj_mx.pkl \
  --source-url https://github.com/liyaguang/DCRNN \
  --output target/metr-la-stgformer.json \
  --models dcrnn,graph_wavenet,staeformer,stgormer,stgformer,lsttn,spatial_temporal_graph_gated_transformer,spatial_shift_graphon_moe \
  --cutoffs 10000,15000,20000,25000,30000 \
  --horizon 12 --lookback 12 --lsttn-lookback 4032 --periodicity 288 --recent-window 12
```

Use the same input files, origins, horizon, and estimator budget for every
model in a comparison. For `lsttn`, set `--lsttn-lookback` to at least fourteen
periods; the runner rejects a shorter context rather than silently disabling
its long-history branch.

The runner also accepts the established PEMS-BAY layout used by the PyTorch
Geometric Temporal loader: `--node-values-npy` for a `[time, node, feature]`
array and `--adjacency-npy` for its matching dense adjacency. Its default
`--target-feature 0` matches that loader's speed target; select another source
feature only when the dataset definition explicitly identifies it as the
forecast target.

## METR-LA 168-Hour Metal Results

On the registered [METR-LA archive](https://anl.app.box.com/shared/static/plgsv3te0akmqluiuqva34su60nn93c2),
STAEformer produced the lowest RMSE and the only positive R², while LSTTN
produced the lowest MAE. The MAE difference between LSTTN and STAEformer was
0.0079, so this one-origin run supports an MAE tie in practical terms rather
than a broad accuracy-win claim for either model.

| Model | RMSE | MAE | R² | Fit | Predict | Execution |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `staeformer` | **20.3177** | 14.6713 | **0.0271** | 16,286.223s | 8.445s | Metal forecast head; CPU feature graph and training |
| `graph_wavenet` | 21.7086 | 18.1118 | -0.1107 | 293.396s | 0.200s | Metal forecast head; CPU feature graph and training |
| `lsttn` | 21.7977 | **14.6634** | -0.1198 | 1,306.264s | 65.714s | Full Metal graph training and inference; CPU orchestration |
| `dcrnn` | 39.0141 | 30.9860 | -2.5873 | 721.803s | 0.101s | Metal forecast head; CPU feature graph and training |

The source archive SHA-256 is
`82319afd4a2ef3327c0551bf0e1f2adb14395a8ecf65cd393d4fe24cf574d34e`.
All 34,272 five-minute timestamps and all 207 sensors were loaded, then each
block of 12 consecutive rows was averaged to produce 2,856 hourly rows. The
derived values SHA-256 is
`d7e30ab900ae9d293e543551e14c79df0bd61ed5b8310a81aaa94d16a07019a3`;
the 1,722-edge adjacency SHA-256 is
`e87344b0a792e1eb5ff884e892ffbfbdd642fbb66c56fd739fd271a55279f825`.
The fixed origin trained on the first 2,284 hourly rows and evaluated the next
168 hours for all 207 sensors. The remaining 404 hourly rows are outside this
declared origin and were not used for fitting or scoring.

The exact command was:

```bash
uv run --group dev python -m benchmarks.runners.traffic_graph_forecasting \
  --node-values-npy /tmp/cartoboost-metr-la/metr_la_hourly.npy \
  --adjacency-npy /tmp/cartoboost-metr-la/adj_mat.npy \
  --source-url https://anl.app.box.com/shared/static/plgsv3te0akmqluiuqva34su60nn93c2 \
  --source-artifact-sha256 82319afd4a2ef3327c0551bf0e1f2adb14395a8ecf65cd393d4fe24cf574d34e \
  --source-time-rows-before-preprocessing 34272 \
  --preprocessing mean_12_consecutive_5min_rows_to_hourly \
  --output docs/assets/model_benchmarks/metr_la_hourly_168h_metal.json \
  --models lsttn,dcrnn,graph_wavenet,staeformer \
  --cutoffs 2284 --frequency hourly --horizon 168 \
  --lookback 168 --lsttn-lookback 672 --periodicity 24 --recent-window 168 \
  --hidden-size 4 --attention-heads 2 --graph-order 2 --experts 2 \
  --epochs 1 --learning-rate 0.003 --weight-decay 0.00001 --backend metal
```

The [machine-readable artifact](../../assets/model_benchmarks/metr_la_hourly_168h_metal.json)
records the exact invocation, source hashes, split, settings, per-model backend
scope, quality metrics, and wall-clock timings. This is a full-scale execution
and device-path verification with one small hidden layer and one training epoch,
not a tuned state-of-the-art comparison. It uses one forecast origin, and three
of the four models have negative R². Those limits prevent a general accuracy
claim; the result establishes that LSTTN's complete training and inference graph
runs on Metal at 207-sensor, four-week-history, one-week-horizon scale.

## Interactive Example

<ForecastModelExample title="Graph-style multi-route panel forecast" model="neural_panel" sample="spatial" />

The embedded browser example uses the same panel forecasting surface and a
multi-location demand panel. Use it as a quick shape check for panel behavior.
Evaluate graph-specific quality in Python with your own adjacency and the
same rolling-origin split used by the baselines.

<MarketStructureWasmExample />

This browser exercise uses 56 directed taxi-shaped lanes over 126 daily dates
(7,056 lane-day observations) and runs the native relationship learner in
WASM. It is intentionally separate from the real TLC benchmark evidence below:
the fixture is an interactive scale check, while the maintained taxi benchmark
uses real records and reports the quality claim.

## Python Example

```python
import numpy as np
from cartoboost.forecasting import (
    DCRNNForecaster,
    GraphTemporalFrame,
    GraphWaveNetForecaster,
    STAEformerForecaster,
)

target = np.array(
    [
        [12.0, 8.0, 5.0, 4.0],
        [14.0, 9.0, 6.0, 5.0],
        [18.0, 12.0, 7.0, 5.5],
        [21.0, 16.0, 9.0, 6.0],
        [19.0, 17.0, 11.0, 7.0],
        [16.0, 15.0, 12.0, 8.0],
        [14.0, 12.0, 10.0, 8.5],
        [13.0, 10.0, 8.0, 7.5],
    ],
    dtype=float,
)

frame = GraphTemporalFrame(
    node_ids=["sensor_a", "sensor_b", "sensor_c", "sensor_d"],
    timestamps=list(range(target.shape[0])),
    target=target,
    indptr=[0, 2, 3, 4, 4],
    indices=[1, 2, 2, 3],
    data=[0.7, 0.3, 1.0, 1.0],
    horizon=2,
    frequency="hourly",
)

model = DCRNNForecaster(
    diffusion_steps=2,
    hidden_size=8,
    epochs=160,
    learning_rate=0.03,
    backend="cpu",
)
model.fit(frame)

forecast = model.predict(2)
metrics = model.backtest(frame=frame, train_size=6)
model.save("graph-forecast.json")

attention_model = STAEformerForecaster(
    lookback=4,
    attention_heads=2,
    hidden_size=8,
    backend="cpu",
)
attention_model.fit(frame)
attention_forecast = attention_model.predict(2)
attention_model.save("graph-attention-forecast.json")

wave_model = GraphWaveNetForecaster(
    lookback=4,
    dilation_depth=2,
    hidden_size=8,
    backend="cpu",
)
wave_model.fit(frame)
wave_forecast = wave_model.predict(2)
```

`forecast`, `attention_forecast`, and `wave_forecast` are numeric arrays with
shape `[horizon, node]`. `backtest` returns horizon-level MAE, RMSE, and WAPE
for the supplied cutoff.

`backend="cpu"` is the default. `backend="auto"` is accepted as a CPU-resolving
alias. On Apple-platform wheels built with native Metal support,
`backend="metal"` runs LSTTN's scalar computation graph, reverse-mode
gradients, AdamW updates, and forecast evaluation on Metal; Rust on the CPU
still validates data and orchestrates graph execution. The DCRNN decoder head,
GraphWaveNet dilated decoder head, and STAEformer attention decoder head use the
shared Metal affine kernel while their feature graphs and training remain on
the CPU. On Linux or WSL wheels built with ROCm support,
`backend="rocm"` routes the same decoder head through the shared HIP affine
kernel. On Windows or Linux wheels built with CUDA support, `backend="cuda"`
routes the same decoder head through the shared CUDA affine kernel. If the
requested accelerator is unavailable, construction fails with the available
backend list.

## Inputs

| Input | Meaning |
| --- | --- |
| `node_ids` | Stable ids for the graph nodes. |
| `timestamps` | Regular time steps for the panel. |
| `target` | Matrix shaped `[time, node]`. |
| `indptr`, `indices`, `data` | Directed CSR adjacency for the graph. |
| `horizon` | Maintained forecast horizon for the frame. |
| `frequency` | Frequency label such as `"hourly"` or `"daily"`. |
| `covariates` | Optional node-time features shaped `[time, node, feature]`. |

Prediction before `fit`, invalid CSR arrays, non-finite targets, missing graph
edges, or incompatible shapes should be treated as data errors.

## When To Use

- The graph topology is stable and known before the forecast cutoff.
- Neighboring nodes plausibly move before or after the target node.
- You can compare against panel-only baselines on the same rolling-origin split.
- You need horizon-by-node diagnostics, not only one aggregate score.

## Use When

| Need | Better first choice |
| --- | --- |
| Transparent last-value or seasonal baseline. | `NaiveForecaster` or `SeasonalNaiveForecaster` |
| Shared lag and calendar features across many panels. | `CartoBoostLagForecaster` |
| Direct neural panel forecasts without explicit adjacency. | `NeuralPanelForecaster` |
| Directed graph diffusion across panel nodes. | `DCRNNForecaster` |

## Validation

Use rolling-origin validation. Keep the graph fixed to information available at
the cutoff and compare against seasonal naive, `CartoBoostLagForecaster`, and
`NeuralPanelForecaster` when the panel is large enough.

Report:

| Metric | Why it matters |
| --- | --- |
| MAE, RMSE, WAPE by horizon | Shows whether graph signal helps near and far horizons. |
| Error by node | Finds nodes where graph diffusion helps or hurts. |
| Error by graph distance | Checks whether upstream/downstream structure explains residuals. |
| Baseline table | Prevents a graph model from replacing a simpler panel model without evidence. |

## Limitations

- `DCRNNForecaster` is not a replacement for validating the graph itself.
- If the graph changes over time, keep only edges known at the cutoff.
- Do not fill missing adjacency with an empty graph or silently fall back to a
  panel-only model.
- `STAEformerForecaster` uses a deterministic Rust spatiotemporal attention
  feature encoder with a trained native decoder. Benchmark claims still need
  the same rolling-origin split and baseline comparisons as DCRNN claims.

## Learned Market Structure

`MarketStructureForecaster` is an explainable sparse smoother for a daily
directional market panel with two caller-named targets. It learns candidate
relationships from shared endpoints, reverse direction, geography, and
train-only residual correlation; optional versioned expert priors can add,
weight, or prohibit individual relationships. Each retained edge has a
train-only weekly multiplier, so recurring temporal structure can change the
strength of a relationship without making the graph dense. A native GraphSAGE
kernel is fit on the sparse provisional graph and static lane state before the
final relationship ranking; explanations identify its contribution as
`neural_kernel`. Forecasts combine that graph signal with weekly and
known-future calendar components.

## Market Structure Explorer

<MarketStructureExplorerSample />

The explorer uses the two caller-supplied target names directly: a fare,
price, cost, demand, or another named measure appears with that exact label in
the signal selector. It aggregates directional lanes into their pickup market,
so each surface shows the spatial shape of a market rather than repeated lane
markers. Switch between a smoothed 2D kernel with contour rings and an
extruded 3D grid. Click a market to inspect its inbound/outbound volume,
forecast path, and strongest learned relationships; select a kernel arc to
follow a connected market. The component accepts artifact-backed nodes and
relationships, so an application can use the same interaction with its own
selected targets and geography.

```python
import math
import numpy as np

from cartoboost.forecasting import MarketPanelFrame, MarketStructureForecaster

# Twelve directional lanes across airport and Manhattan-style markets. The
# caller owns the two target names; neither is assigned package semantics.
markets = {
    "132": (-73.7865, 40.6470),  # airport_a
    "138": (-73.8729, 40.7738),  # airport_b
    "161": (-73.9777, 40.7580),  # midtown
    "230": (-73.9842, 40.7598),  # theatre_district
    "48": (-73.9904, 40.7623),   # west_midtown
    "79": (-73.9864, 40.7277),   # village
}
directions = [
    ("132", "161"), ("161", "132"),
    ("132", "230"), ("230", "132"),
    ("138", "161"), ("161", "138"),
    ("138", "230"), ("230", "138"),
    ("161", "230"), ("230", "161"),
    ("48", "79"), ("79", "48"),
]
lane_ids = [f"{origin}:{destination}" for origin, destination in directions]
origin_ids = [origin for origin, _ in directions]
destination_ids = [destination for _, destination in directions]
coordinates = [
    [*markets[origin], *markets[destination]]
    for origin, destination in directions
]

def known_calendar(day: int) -> list[float]:
    return [math.sin(2 * math.pi * day / 7), float(day % 28 == 0)]

days = 84
timestamps = list(range(days))
calendar = np.array([known_calendar(day) for day in timestamps])
lane_offset = np.linspace(-0.18, 0.18, len(lane_ids))
airport_lane = np.array([origin in {"132", "138"} for origin in origin_ids], dtype=float)
primary = np.array([
    np.exp(1.25 + lane_offset + 0.24 * airport_lane
           + 0.16 * math.sin(day / 9) + 0.09 * math.cos(day / 21))
    for day in timestamps
])
secondary = np.array([
    np.maximum(1, np.rint(22 + 44 * airport_lane + 9 * np.sin(day / 7 + lane_offset * 4)))
    for day in timestamps
])

frame = MarketPanelFrame(
    lane_ids=lane_ids,
    timestamps=timestamps,
    target_names=["primary_measure", "supporting_measure"],
    primary=primary,
    secondary=secondary,
    origin_ids=origin_ids,
    destination_ids=destination_ids,
    coordinates=coordinates,
    calendar=calendar,
    # Ordered, caller-owned parent keys, from most specific to broadest.
    hierarchy_groups=[
        [f"origin_parent:5:{origin}", f"market_family:{'airport' if origin in {'132', '138'} else 'city'}"]
        for origin in origin_ids
    ],
    horizon=7,
    frequency="daily",
)
model = MarketStructureForecaster(top_k=8).fit(frame)
future_calendar = np.array([known_calendar(days + step) for step in range(1, 8)])
forecast_rows = model.predict(7, future_calendar=future_calendar)
weekly_rows = model.weekly_rollups(7, future_calendar=future_calendar)
current_explanations = model.nowcast()
explorer_payload = model.explorer_payload(7)

# Analyst-facing evidence: one row per directional lane, then its sparse kernel.
for row in current_explanations:
    print(
        row["lane_id"],
        row["shift"],
        round(row["smoothed_primary"], 3),
        round(row["uncertainty"], 3),
        [(edge["target_lane_id"], round(edge["weight"], 2)) for edge in row["top_relationships"][:3]],
    )
```

This is a 1,008-observation directional panel—not a two-lane toy. The browser
example above uses the larger 7,056-observation panel. In either environment,
select a lane in the explorer to see the current smoothed value, inbound and
outbound supporting volume, its forecast path, component-based shift rationale,
and the retained directed kernel edges.

Predictions retain generic `primary` and `secondary` fields; a fitted
lane-local coupling lets the supporting target consume primary co-movement
without changing the primary smoother. `nowcast()` adds
the smoothed primary value, uncertainty, seasonal, local/mix, and residual components, a
`market`, `local_or_mix`, or `no_shift` classification, the top sparse edges,
and any matching expert label. Use rolling-origin evaluation with a fixed
distance baseline and a fixed-graph forecaster before adopting the smoother.

For intermittent panels, `primary` may contain `NaN` for an unobserved
lane-time value. It is not interpreted as zero and is not filled. Supply
ordered `hierarchy_groups` when a lane may have no direct observations: for
example, a fine spatial cell can list its parent cells from finest to broadest.
The model partially pools sparse lane estimates toward the first parent with
observed support; a fully unobserved lane uses that explicit parent state and
receives `support: "hierarchy"` plus `observed_primary: null` in its nowcast.
If neither lane nor an supplied parent has observations, fitting fails.

`weekly_rollups()` is a native aggregation of the daily forecast path, not a
separately fitted weekly model: it averages `primary` and its interval bounds,
and sums `secondary`, with the number of contributing days on each row.

`explorer_payload()` is the portable interpretability contract for analyst
tools: it returns lane endpoint geometry, current explanations, forecast rows,
and retained learned kernels in one native JSON-compatible payload. The browser
export exposes the same behavior as `runMarketStructureExplorer(request)`, so a
WASM app can render and evaluate the same evidence as a Python notebook.

The supplied explorer includes `marketExplorerDataFromPayload(payload)`, which
turns that contract into clickable endpoint markets without recreating model
logic in the UI. Pass its `nodes` and `edges` directly to
`MarketStructureExplorer`. This gives a Python-backed web notebook and a
browser-only WASM app the same outbound/inbound measures, forecast-change map,
and retained directed kernels.

Primary intervals combine learned asymmetric quantile-head tails with a
per-lane train-only residual-calibration floor. Benchmark artifacts report
realized coverage and mean interval width. On the 36-lane taxi holdout, the
resulting primary interval covered 96.83% of 252 observed values with mean
width 2.5792; this is observed holdout evidence, not a claim of universal
calibration.

Run the maintained real-taxi structure benchmark with a local TLC cache (or
allow the documented TLC inputs to be fetched):

```bash
uv run --group dev python scripts/forecasting_library_benchmark.py \
  --source nyc-taxi --year 2024 --months 1,2,3,4 --lanes 128 --horizon 14 \
  --market-structure-splits \
  --output target/nyc_taxi_market_structure.json
```

The artifact reports separate primary and secondary MAE, RMSE, and WAPE for
the learned structure model, a distance-only baseline, a sparse fixed-graph
baseline, and a last-value baseline, along with relationship and shift counts.
Dense-panel comparators are included only when every required primary history
value is observed; otherwise the artifact records their omission rather than
filling observations. The taxi effective-fare target is a proxy only; it is
not evidence about any other market.

On locally cached January–April 2024 Yellow Taxi records, this command used
12,429,495 cleaned trips aggregated into 128 directional lanes, 109 training
days, and a 14-day final holdout. Primary observations were available for
99.10% of training lane-days and 92.91% of holdout lane-days. The native fit
and forecast took 63.91 seconds; total data loading and evaluation time was
255.13 seconds. The table shows the primary effective-fare-per-mile result on
the 1,665 observed holdout values; lower is better.

| Model | MAE | RMSE | WAPE |
| --- | ---: | ---: | ---: |
| `market_structure` | 0.5301 | 0.6883 | 0.0341 |
| `last_value` | 1.3678 | 1.9532 | 0.0880 |
| `inverse_distance_last_value` | 3.1159 | 3.8419 | 0.2005 |
| `fixed_graph_last_observed` | 3.1766 | 3.8815 | 0.2044 |

The larger sparse-panel run supports the claim that learned structure improves
the available non-imputing spatial baselines. It does not compare against the
dense-only lag, neural-panel, or DCRNN primary models, so it does not establish
replacement of those models.

The supporting trip-count target was complete for the same 1,792 holdout
lane-days, allowing the dense panel models to remain in that comparison.

| Model | MAE | RMSE | WAPE |
| --- | ---: | ---: | ---: |
| `cartoboost_lag` | 35.7115 | 74.2022 | 0.2103 |
| `cartoboost_neural_panel` | 56.4735 | 86.8781 | 0.3325 |
| `market_structure` | 43.2418 | 80.8679 | 0.2546 |
| `fixed_graph_last_observed` | 68.2106 | 119.4271 | 0.4016 |
| `last_value` | 74.5993 | 108.3420 | 0.4392 |
| `inverse_distance_last_value` | 103.4633 | 131.1571 | 0.6092 |

For this supporting target, the learned structure improves the spatial and
neural-panel baselines, while `cartoboost_lag` remains stronger. That
limitation is reported directly in the artifact rather than hidden by a
blended score.

On the locally cached January–March 2024 Yellow Taxi records, the corresponding
`--months 1,2,3 --lanes 36 --horizon 7` dense-panel run used 36 daily
directional lanes and a seven-day final holdout. The table shows the primary
effective-fare-per-mile result; lower is better.

| Model | MAE | RMSE | WAPE |
| --- | ---: | ---: | ---: |
| `cartoboost_lag` | 0.4889 | 0.7209 | 0.0333 |
| `cartoboost_neural_panel` | 0.8390 | 1.1162 | 0.0572 |
| `market_structure` | 0.4720 | 0.7068 | 0.0322 |
| `fixed_graph_dcrnn` | 0.9256 | 1.1476 | 0.0631 |
| `last_value` | 1.2883 | 1.7164 | 0.0878 |
| `inverse_distance_last_value` | 2.8525 | 3.6585 | 0.1945 |

The learned structure has the lowest primary-target MAE, RMSE, and WAPE in
this dense holdout. This is one fixed taxi split, so it is evidence for this
configuration rather than a universal replacement claim.

With `--rolling-origin-folds 3`, the same 36-lane setup evaluates three
leakage-safe daily cutoffs. The market model averaged primary MAE 0.4204 and
supporting-target MAE 31.2169 across the 70-, 77-, and 84-day cutoffs. The
artifact records the full per-cutoff metrics and learned-edge stability.

The same 36-lane run includes a controlled composition-shift analogue: the
final recorded primary value for `PU132->DO230` was multiplied by three and an
aligned synthetic mix feature was set only on that lane. The model labeled the
affected lane `local_or_mix` and raised zero `market` alerts on the other 35
lanes. This validates rejection of a controlled local composition signal; it
is not a claim that the taxi records contain a real provider-mix field.

The artifact also stores every retained relationship and explanation. Across
three train-only daily cutoffs (63, 70, and 77 days), all 288 retained edges
were present at each cutoff and consecutive edge-set Jaccard overlap averaged
0.9363. This is stability evidence for this taxi panel, not a guarantee that a
different market has the same relationship persistence.
