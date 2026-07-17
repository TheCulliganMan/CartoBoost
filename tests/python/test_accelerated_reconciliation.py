from __future__ import annotations

import numpy as np
import pytest
from cartoboost import _native
from cartoboost.forecasting.ensemble import reconcile_hierarchy


def test_native_reconciliation_exposes_backend_selection() -> None:
    if not hasattr(_native, "forecast_hierarchy_reconcile_value"):
        pytest.skip("local extension predates the reconciliation binding")
    result = reconcile_hierarchy(
        [("total", None), ("a", "total"), ("b", "total")],
        [[0.0, 0.0], [2.0, 3.0], [4.0, 5.0]],
        backend="cpu",
    )
    np.testing.assert_allclose(result["values"], [[6.0, 8.0], [2.0, 3.0], [4.0, 5.0]])
    assert result["backend_requested"] == "cpu"
    assert result["backend_selected"] == "cpu"
