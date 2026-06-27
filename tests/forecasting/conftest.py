from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest


class RecordingForecastFrame:
    def __init__(
        self,
        rows: list[tuple[str, str, float]],
        freq: str,
        **kwargs: Any,
    ) -> None:
        self.rows = rows
        self.freq = freq
        self.kwargs = kwargs


@pytest.fixture
def install_fake_native(monkeypatch: pytest.MonkeyPatch):
    def install(class_name: str):
        import cartoboost

        calls: list[tuple[Any, ...]] = []

        class RecordingModel:
            def __init__(self, **params: Any) -> None:
                calls.append(("init", params))

            def fit(self, frame: RecordingForecastFrame):
                calls.append(("fit", frame))
                return self

            def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                calls.append(("predict", args, kwargs))
                return {"args": args, "kwargs": kwargs}

            def predict_with_known_future(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                calls.append(("predict_with_known_future", args, kwargs))
                return {"args": args, "kwargs": kwargs}

            def predict_for_lanes(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                calls.append(("predict_for_lanes", args, kwargs))
                return {"args": args, "kwargs": kwargs}

            def components_json(self, *args: Any, **kwargs: Any) -> str:
                calls.append(("components_json", args, kwargs))
                return json.dumps({"args": list(args), "kwargs": kwargs}, default=str)

            def history_components_json(self, *args: Any, **kwargs: Any) -> str:
                calls.append(("history_components_json", args, kwargs))
                return json.dumps(
                    {
                        "model": class_name,
                        "columns": [
                            "series_id",
                            "timestamp",
                            "index",
                            "actual",
                            "fitted",
                            "residual",
                            "trend",
                            "trend_movement",
                            "components",
                        ],
                        "records": [
                            {
                                "series_id": "PULocationID=132",
                                "timestamp": "1970-01-03T00:00:00",
                                "index": 2,
                                "actual": 14.0,
                                "fitted": 13.5,
                                "residual": 0.5,
                                "trend": 12.0,
                                "trend_movement": 1.0,
                                "components": {
                                    "seasonal_total": 1.5,
                                    "weekly": 1.25,
                                    "events": {"airport_surge": 0.25},
                                },
                            }
                        ],
                    }
                )

            def samples_json(self, *args: Any, **kwargs: Any) -> str:
                calls.append(("samples_json", args, kwargs))
                return json.dumps({"args": list(args), "kwargs": kwargs})

            def quantiles_json(self, *args: Any, **kwargs: Any) -> str:
                calls.append(("quantiles_json", args, kwargs))
                return json.dumps({"args": list(args), "kwargs": kwargs})

            def to_json(self) -> str:
                calls.append(("to_json", (), {}))
                return json.dumps({"model": class_name})

            @classmethod
            def from_json(cls, value: str):
                calls.append(("from_json", (value,), {}))
                return cls()

        native = SimpleNamespace(ForecastFrame=RecordingForecastFrame)
        setattr(native, class_name, RecordingModel)
        monkeypatch.setattr(cartoboost, "_native", native, raising=False)
        return SimpleNamespace(calls=calls, frame_class=RecordingForecastFrame)

    return install
