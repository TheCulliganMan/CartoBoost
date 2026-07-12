import {ForecastModelExample} from '@site/src/components/ModelingLabClient';
import {MarketStructureExplorerSample} from '@site/src/components/MarketStructureExplorer';

# Graph Spatiotemporal Forecasting

Use `DCRNNForecaster`, `GraphWaveNetForecaster`, or `STAEformerForecaster`
when each forecast series is a node in a known directed graph and neighboring
nodes can lead, lag, or diffuse signal into one another. This is for sensor
networks, route flows, road segments, zone flows, equipment networks, or other
panels where the graph is part of the modeling claim.

Do not use a graph forecaster only because node ids exist. The edges should
represent known movement, influence, adjacency, or dependency available at the
forecast cutoff.

## Interactive Example

<ForecastModelExample title="Graph-style panel forecast sanity check" model="neural_panel" sample="spatial" />

The embedded browser example uses the same panel forecasting surface and a
multi-location demand panel. Use it as a quick shape check for panel behavior.
Evaluate graph-specific quality in Python with your own adjacency and the
same rolling-origin split used by the baselines.

## Public Contract

```python
import numpy as np
from cartoboost.preview.forecasting import (
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
alias. On Apple-platform wheels built with native
Metal support, `backend="metal"` routes the DCRNN decoder head, GraphWaveNet
dilated decoder head, and STAEformer attention decoder head through the shared
Metal affine kernel. On Linux or WSL wheels built with ROCm support,
`backend="rocm"` routes the same decoder head through the shared HIP affine
kernel. On Windows or Linux wheels built with CUDA support, `backend="cuda"`
routes the same decoder head through the shared CUDA affine kernel. Diffusion
state updates, dilated temporal graph features, attention feature generation,
graph validation, and training remain deterministic Rust code. If the requested
accelerator is unavailable, construction fails with the available backend list.

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

Click a point to inspect inbound/outbound volume, the forecast path, and its
strongest learned relationships. Select a kernel arc to follow the connected
market. The component accepts artifact-backed nodes and relationships, so an
application can use the same interaction with its own selected targets and
geography.

```python
from cartoboost.preview.forecasting import MarketPanelFrame, MarketStructureForecaster

frame = MarketPanelFrame(
    lane_ids=["132:138", "132:161", "138:132"],
    timestamps=list(range(21)),
    target_names=["benchmark", "supporting_measure"],
    primary=daily_primary_matrix,
    secondary=daily_secondary_matrix,
    origin_ids=["132", "132", "138"],
    destination_ids=["138", "161", "132"],
    coordinates=lane_endpoint_coordinates,
    # Ordered, caller-owned parent keys, from most specific to broadest.
    hierarchy_groups=[["origin_parent:5:132"], ["origin_parent:5:132"], ["origin_parent:5:138"]],
)
model = MarketStructureForecaster(top_k=8).fit(frame)
forecast_rows = model.predict(7)
weekly_rows = model.weekly_rollups(7)
current_explanations = model.nowcast()
```

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

Primary intervals use per-lane empirical residual radii and a train-only
three-origin rolling calibration multiplier. Benchmark artifacts report
realized coverage and mean interval width. On the 36-lane taxi holdout, the
resulting primary interval covered 84.52% of 252 observed values with mean
width 3.2859; this is the observed evidence for the conservative interval
setting, not a claim of universal calibration.

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
and forecast took 0.83 seconds; total data loading and evaluation time was
100.47 seconds. The table shows the primary effective-fare-per-mile result on
the 1,665 observed holdout values; lower is better.

| Model | MAE | RMSE | WAPE |
| --- | ---: | ---: | ---: |
| `market_structure` | 0.9513 | 1.1974 | 0.0612 |
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
| `market_structure` | 60.8855 | 87.8150 | 0.3585 |
| `fixed_graph_last_observed` | 68.2106 | 119.4271 | 0.4016 |
| `last_value` | 74.5993 | 108.3420 | 0.4392 |
| `inverse_distance_last_value` | 103.4633 | 131.1571 | 0.6092 |

For this supporting target, the learned structure again improves the spatial
baselines but trails the lag and neural-panel models. That limitation is
reported directly in the artifact rather than hidden by a blended score.

On the locally cached January–March 2024 Yellow Taxi records, the corresponding
`--months 1,2,3 --lanes 36 --horizon 7` dense-panel run used 36 daily
directional lanes and a seven-day final holdout. The table shows the primary
effective-fare-per-mile result; lower is better.

| Model | MAE | RMSE | WAPE |
| --- | ---: | ---: | ---: |
| `cartoboost_lag` | 0.4889 | 0.7209 | 0.0333 |
| `cartoboost_neural_panel` | 0.8390 | 1.1162 | 0.0572 |
| `market_structure` | 0.8675 | 1.1290 | 0.0591 |
| `fixed_graph_dcrnn` | 0.9256 | 1.1476 | 0.0631 |
| `last_value` | 1.2883 | 1.7164 | 0.0878 |
| `inverse_distance_last_value` | 2.8525 | 3.6585 | 0.1945 |

The learned structure improves the distance-only and fixed-graph smoothers on
this holdout, while `cartoboost_lag` is the stronger primary-target baseline.
Treat that as a limitation of this initial model, not a reason to replace the
lag model without further validation.

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
