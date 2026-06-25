# Installation

CartoBoost is published on PyPI as `cartoboost`.

Install CartoBoost when you need to compare temporal-spatial regressors,
forecasting models, graph/neural variants, or reproducible benchmark
workflows in a Python environment. The core package is enough for NumPy and
pandas-style modeling; optional extras add integrations without changing the
modeling contracts.

## Install From PyPI

```sh
uv add cartoboost
```

The published wheels target CPython 3.10, 3.11, 3.12, and 3.13 on:

- Linux x86_64 and aarch64 with manylinux2014 compatibility.
- macOS x86_64 and arm64.
- Windows x86_64 and arm64.

If no compatible wheel exists, `uv` may try to build from source, which requires
the project build toolchain.

## Optional Extras

Install optional integrations only when the scientific workflow needs them:

```sh
uv add "cartoboost[explain]"
uv add "cartoboost[h3]"
uv add "cartoboost[holidays]"
uv add "cartoboost[s2]"
uv add "cartoboost[duckdb]"
uv add "cartoboost[optuna]"
uv add "cartoboost[polars]"
uv add "cartoboost[onnx]"
uv add "cartoboost[visualization]"
```

| Extra | Adds | Use when |
| --- | --- | --- |
| `explain` | SHAP explanations. | You need feature-attribution diagnostics for a fitted regressor. |
| `h3` | Optional H3 latitude/longitude encoder. | Spatial cells are part of the tested feature design. |
| `holidays` | Country holiday calendar expansion for the piecewise linear seasonal forecaster. | You need Prophet-style `add_country_holidays` behavior. |
| `s2` | Optional S2 latitude/longitude encoder. | S2 cells match the existing geography pipeline. |
| `duckdb` | DuckDB relation/query-result input support. | Taxi training data already lives in DuckDB queries. |
| `optuna` | Hyperparameter tuning examples and workflows. | You are tuning under a fixed validation protocol. |
| `polars` | Polars input support. | Data preparation uses Polars tables. |
| `onnx` | ONNX export for the supported dense axis-tree subset. | Deployment requires ONNX and the model stays inside the supported subset. |
| `visualization` | Matplotlib, GeoPandas, Shapely, and PyDeck plotting helpers. | You need diagnostic plots, static spatial plots, or interactive taxi route maps. |

## Verify The Install

```sh
python -c "import cartoboost; print(cartoboost.__version__)"
cartoboost --help
```

Python usage should work immediately after install:

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(n_estimators=10, max_depth=2)
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ImportError` during import | Reinstall CartoBoost in a clean Python environment. |
| `uv` tries to compile from source | Use CPython 3.10-3.13 on a supported platform, or install the project build toolchain before building. |
| `cartoboost` command not found | Make sure the Python environment where `cartoboost` was installed is active. |
