from __future__ import annotations

import numpy as np
from cartoboost.accelerators import (
    adamw_step,
    affine_scores,
    available_backends,
    csr_diffusion,
    csr_diffusion_backward,
    csr_row_softmax,
    csr_row_softmax_backward,
    dense_layer,
    dispatch_report,
    graph_smooth,
    layer_norm,
    pair_sigmoid_scores,
    pairwise_squared_distances,
    workload_decision,
)


def test_vector_dispatch_report_runs_on_every_available_backend() -> None:
    for backend in available_backends("vector_dispatch"):
        report = dispatch_report(backend, length=64)
        assert report["selected"] == backend
        assert report["accelerated"] is (backend != "cpu")
        assert abs(report["checksum"] - report["expected_checksum"]) < 1.0e-4


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


def test_graph_smoothing_runs_on_every_available_csr_backend() -> None:
    nodes = 4_096
    indptr = np.arange(nodes + 1, dtype=np.uint32)
    indices = (np.arange(nodes, dtype=np.uint32) + 1) % nodes
    weights = np.ones(nodes, dtype=np.float32)
    values = np.sin(np.arange(nodes) * 0.03)
    expected = graph_smooth(
        indptr,
        indices,
        weights,
        values,
        smoothing=0.75,
        iterations=4,
        backend="cpu",
    )
    for backend in available_backends("csr_diffusion"):
        actual = graph_smooth(
            indptr,
            indices,
            weights,
            values,
            smoothing=0.75,
            iterations=4,
            backend=backend,
        )
        np.testing.assert_allclose(actual, expected, rtol=1.0e-4, atol=1.0e-4)


def test_optimizer_normalization_and_sparse_attention_are_public() -> None:
    normalized = layer_norm(
        [[1.0, 3.0], [2.0, 6.0]],
        [1.0, 1.0],
        [0.0, 0.0],
        backend="cpu",
    )
    np.testing.assert_allclose(normalized, [[-1.0, 1.0], [-1.0, 1.0]], atol=1e-4)

    softmax = csr_row_softmax([0, 2, 3], [0.0, 0.0, 4.0], backend="cpu")
    np.testing.assert_allclose(softmax, [0.5, 0.5, 1.0])

    scores = pair_sigmoid_scores(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
        [[0, 1], [0, 2]],
        backend="cpu",
    )
    np.testing.assert_allclose(scores, [0.73105858, 0.26894142])

    parameters, first, second = adamw_step(
        [1.0, -1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.5, -0.25],
        step=1,
        learning_rate=0.01,
        weight_decay=0.01,
        backend="cpu",
    )
    assert parameters.shape == first.shape == second.shape == (2,)
    assert parameters[0] < 1.0
    assert parameters[1] > -1.0


def test_sparse_backward_kernels_match_reference_formulas() -> None:
    input_grad, edge_grad = csr_diffusion_backward(
        [0, 1, 3],
        [1, 0, 1],
        [1.0, 0.25, 0.75],
        [[2.0, 4.0], [6.0, 8.0]],
        [[1.0, 2.0], [3.0, 4.0]],
        channels=2,
        backend="cpu",
    )
    np.testing.assert_allclose(input_grad, [[0.75, 1.0], [3.25, 5.0]])
    np.testing.assert_allclose(edge_grad, [22.0, 22.0, 50.0])

    weights = csr_row_softmax([0, 2], [0.0, 1.0], backend="cpu")
    gradient = csr_row_softmax_backward([0, 2], weights, [2.0, -1.0], backend="cpu")
    expected = weights * (np.array([2.0, -1.0]) - np.dot(weights, [2.0, -1.0]))
    np.testing.assert_allclose(gradient, expected, atol=1e-6)
