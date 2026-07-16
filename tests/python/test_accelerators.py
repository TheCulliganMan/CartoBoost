from __future__ import annotations

from cartoboost.accelerators import workload_decision


def test_workload_decision_reports_actual_threshold_execution() -> None:
    small = workload_decision("cpu", "dense", 128, 16_384)
    assert small["selected"] == "cpu"
    assert small["executed"] == "cpu"
    assert small["accelerated"] is False


def test_workload_decision_rejects_unknown_operation() -> None:
    try:
        workload_decision("cpu", "missing", 1, 1)
    except ValueError as error:
        assert "unknown accelerator operation" in str(error)
    else:
        raise AssertionError("unknown operation should fail")
