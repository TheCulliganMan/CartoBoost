from __future__ import annotations

import numpy as np
import pytest
from cartoboost.foundation import (
    ChronosAdapter,
    FoundationAdapterUnavailable,
    FoundationForecastFeatures,
    TabPFNAdapter,
    TimesFMAdapter,
)


def test_foundation_adapter_cache_includes_external_metadata(tmp_path):
    adapter = ChronosAdapter(
        model_hash="sha256:test-model",
        explicitly_enabled=True,
        backend=lambda values: np.asarray(values, dtype=float) + 1.0,
    )
    output = adapter.predict([1.0, 2.0, 3.0])
    path = adapter.cache_output(tmp_path / "chronos.json", [1.0, 2.0, 3.0], output)
    cache = adapter.load_cache(path)

    assert cache["metadata"]["adapter"] == "chronos"
    assert cache["metadata"]["model_hash"] == "sha256:test-model"
    assert cache["metadata"]["explicitly_enabled"] is True
    assert cache["metadata"]["auto_geo_enabled"] is True
    assert cache["output_shape"] == [3]
    np.testing.assert_array_equal(cache["output"], np.array([2.0, 3.0, 4.0]))


def test_foundation_features_benchmark_compare_with_without_features():
    report = FoundationForecastFeatures.benchmark_with_without_features(
        [10.0, 12.0, 14.0],
        [8.0, 10.0, 16.0],
        [9.5, 12.2, 13.8],
    )

    assert report["with_foundation_rmse"] < report["without_foundation_rmse"]
    assert report["rmse_delta"] > 0.0


def test_missing_optional_dependency_gives_clear_skip_reason():
    adapter = TimesFMAdapter()
    reason = adapter.missing_dependency_skip_reason()

    if reason is not None:
        assert "timesfm" in reason
        assert "cartoboost[foundation]" in reason
        with pytest.raises(FoundationAdapterUnavailable, match="timesfm"):
            adapter.predict([1.0, 2.0])


def test_tabpfn_backend_can_be_explicit_without_core_dependency():
    adapter = TabPFNAdapter(
        explicitly_enabled=True,
        backend=lambda values: np.asarray(values, dtype=float).sum(axis=1),
    )

    np.testing.assert_array_equal(adapter.predict([[1.0, 2.0], [3.0, 4.0]]), np.array([3.0, 7.0]))
