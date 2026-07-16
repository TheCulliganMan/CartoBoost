from __future__ import annotations

import numpy as np
from cartoboost.accelerators import (
    affine_scores,
    csr_diffusion,
    dense_layer,
    pairwise_squared_distances,
    workload_decision,
)


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


def test_dense_layer_exposes_native_tensor_dispatch() -> None:
    output = dense_layer(
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, -1.0], [0.5, 3.0]],
        [1.0, -2.0],
        backend="cpu",
    )

    np.testing.assert_allclose(output, [[4.0, 3.0], [9.0, 7.0]], rtol=1e-6, atol=1e-6)


def test_pairwise_distances_expose_native_geo_dispatch() -> None:
    output = pairwise_squared_distances(
        [[0.0, 0.0], [3.0, 4.0]],
        [[0.0, 4.0], [3.0, 0.0]],
        backend="cpu",
    )

    np.testing.assert_allclose(output, [[16.0, 9.0], [9.0, 16.0]], rtol=1e-6, atol=1e-6)


def test_affine_and_csr_graph_kernels_are_public_to_python() -> None:
    scores = affine_scores(
        [[2.0, 5.0], [4.0, 1.0]],
        [1.0, 1.0],
        [2.0, -0.5],
        [3.0, -1.0],
        backend="cpu",
    )
    np.testing.assert_allclose(scores, [3.0, 5.0])

    diffused = csr_diffusion(
        [0, 1, 3],
        [1, 0, 1],
        [1.0, 0.25, 0.75],
        [[2.0, 4.0], [6.0, 8.0]],
        channels=2,
        backend="cpu",
    )
    np.testing.assert_allclose(diffused, [[6.0, 8.0], [5.0, 7.0]])
