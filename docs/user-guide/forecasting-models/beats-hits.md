# N-BEATS And N-HiTS

`NBeatsForecaster` and `NHiTSForecaster` are deterministic neural forecasting
experts for regular forecast windows. Use them when you want a neural baseline
for a clean, evenly spaced series or panel, and compare them against seasonal
naive, local statistical models, and `CartoBoostLagForecaster` on the same
rolling-origin split.

## When To Use

- The series is regular enough to form fixed history windows.
- You want a neural window model, not an interpretable component model.
- The training set is large enough that a small neural expert is meaningful.
- You can report rolling-origin metrics against simpler baselines.

Use `NBeatsForecaster` first when recent history should be projected directly.
Use `NHiTSForecaster` when pooled history windows may be useful for smoother or
longer-horizon structure.

## Public Contract

```python
from cartoboost.forecasting import ForecastFrame, NBeatsForecaster, NHiTSForecaster

frame = ForecastFrame.from_pandas(
    hourly_demand,
    timestamp_col="timestamp",
    target_col="demand",
    series_id_col="series_id",
    freq="h",
)

nbeats = NBeatsForecaster(
    input_size=24,
    hidden_size=32,
    epochs=80,
    learning_rate=0.01,
)
nbeats.fit(frame)
nbeats_forecast = nbeats.predict(12)

nhits = NHiTSForecaster(
    input_size=48,
    hidden_size=32,
    epochs=80,
    learning_rate=0.01,
    pooling_size=2,
)
nhits.fit(frame)
nhits_forecast = nhits.predict(12)
```

Set `backend="auto"` for ordinary runs. On Apple-platform wheels built with the
native Metal feature, `backend="metal"` routes the deterministic dense
inference layers through CartoBoost's shared Metal backend. On Linux or WSL
wheels built with ROCm support, `backend="rocm"` routes the same dense
inference layers through CartoBoost's shared HIP backend. On builds with
CUDA support, `backend="cuda"` routes the same dense inference layers through
CartoBoost's shared CUDA backend. On builds with WebGPU enabled,
`backend="webgpu"` routes the same dense inference layers through the shared
WebGPU backend. Invalid or unavailable accelerator requests fail instead of
silently falling back to CPU.

## Use When

| Situation | Better first choice |
| --- | --- |
| You need an explainable level/trend/seasonality decomposition. | `ThetaForecaster`, `ETSForecaster`, or `PiecewiseLinearSeasonalForecaster` |
| You need a guarded default over several forecast families. | `AutoForecaster` |
| You have many aligned panels and want tabular lag features. | `CartoBoostLagForecaster` |
| You want a compact neural expert for fixed windows. | `NBeatsForecaster` or `NHiTSForecaster` |

## Parameters

| Parameter | Applies to | Notes |
| --- | --- | --- |
| `input_size` | Both | Number of historical observations used for each training window. |
| `hidden_size` | Both | Width of the internal neural representation. |
| `epochs` | Both | Number of deterministic training passes. |
| `learning_rate` | Both | Optimization step size. |
| `pooling_size` | `NHiTSForecaster` | Pooling factor for compressed history windows. |
| `backend` | Both | `"auto"`, `"cpu"`, or an available accelerator such as `"metal"`, `"rocm"`, `"cuda"`, or `"webgpu"` for backend-dispatched dense prediction kernels. |

## Validation

These models can look strong when a random split leaks nearby time windows. Use
rolling-origin validation and keep the baseline table visible:

```python
from cartoboost.forecasting import RollingOriginBacktester, RollingOriginSplitter

splitter = RollingOriginSplitter(horizon=12, step=12, min_train_size=96)
backtester = RollingOriginBacktester(splitter=splitter)

nbeats_result = backtester.evaluate(NBeatsForecaster(input_size=24), frame)
nhits_result = backtester.evaluate(NHiTSForecaster(input_size=48, pooling_size=2), frame)
```

Report RMSE, MAE, WAPE, horizon metrics, train time, prediction time, and the
same seasonal naive and `CartoBoostLagForecaster` comparison used for the rest
of the forecasting family.
