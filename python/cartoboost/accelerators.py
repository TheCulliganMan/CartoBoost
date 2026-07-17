"""Runtime accelerator inventory and operation-level capabilities."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native

_ALL_OPERATIONS = [
    "vector_dispatch",
    "affine",
    "dense",
    "pair_scoring",
    "pairwise_distance",
    "csr_diffusion",
    "csr_diffusion_backward",
    "csr_row_softmax",
    "csr_row_softmax_backward",
    "adamw",
    "layer_norm",
    "scalar_graph",
    "scalar_graph_training",
    "tanh_mlp_training",
]


def capabilities() -> dict[str, Any]:
    """Return every canonical backend and the kernels it can dispatch.

    ``available`` reflects this process's build and runtime device discovery;
    ``operations`` is the backend contract and remains useful when a device is
    not present on the current host.
    """

    function = getattr(_native, "accelerator_capabilities_value", None)
    if function is None:
        return {
            "backends": [
                {
                    "backend": "cpu",
                    "available": True,
                    "operations": list(_ALL_OPERATIONS),
                }
            ]
        }
    return dict(json.loads(function()))


def dispatch_report(
    backend: str | None = None,
    *,
    length: int = 4096,
) -> dict[str, Any]:
    """Run the backend's vector-dispatch probe and return its execution report."""

    if length <= 0:
        raise ValueError("length must be positive")
    function = getattr(_native, "accelerator_dispatch_report_value", None)
    if function is None:
        if backend not in {None, "cpu"}:
            raise RuntimeError("native accelerator support is unavailable")
        raise RuntimeError("vector dispatch reporting requires the native extension")
    return dict(json.loads(function(backend, int(length))))


def affine_scores(
    features: ArrayLike,
    means: ArrayLike,
    weights: ArrayLike,
    intercepts: ArrayLike,
    *,
    backend: str | None = None,
) -> NDArray[np.float64]:
    """Execute centered affine scoring through the shared backend contract."""

    feature_array = np.asarray(features, dtype=np.float64)
    mean_array = np.asarray(means, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    intercept_array = np.asarray(intercepts, dtype=np.float64)
    function = getattr(_native, "accelerator_affine_scores_value", None)
    if function is None:
        if backend not in {None, "cpu"}:
            raise RuntimeError("native accelerator support is unavailable")
        return (feature_array - mean_array) @ weight_array + intercept_array
    return np.asarray(
        function(
            feature_array.tolist(),
            mean_array.tolist(),
            weight_array.tolist(),
            intercept_array.tolist(),
            backend,
        ),
        dtype=np.float64,
    )


def available_backends(operation: str | None = None) -> list[str]:
    """List runtime-available backends, optionally filtered by operation."""

    rows = capabilities()["backends"]
    return [
        str(row["backend"])
        for row in rows
        if bool(row["available"]) and (operation is None or operation in row["operations"])
    ]


def workload_decision(
    backend: str | None,
    operation: str,
    workload_size: int,
    minimum_accelerated_size: int,
) -> dict[str, Any]:
    """Report the device that a threshold-gated operation will actually use.

    This performs capability selection only. It does not benchmark, calibrate,
    allocate device buffers, or launch a kernel, so it is safe to call on an
    inference path.
    """

    if operation not in _ALL_OPERATIONS:
        raise ValueError(f"unknown accelerator operation {operation!r}")
    if workload_size < 0 or minimum_accelerated_size < 0:
        raise ValueError("workload sizes must be non-negative")
    function = getattr(_native, "accelerator_workload_decision_value", None)
    if function is None:
        return {
            "requested": "cpu" if backend is None else str(backend),
            "selected": "cpu",
            "executed": "cpu",
            "operation": operation,
            "workload_size": int(workload_size),
            "minimum_accelerated_size": int(minimum_accelerated_size),
            "accelerated": False,
            "fallback_reason": None,
        }
    return dict(
        json.loads(
            function(
                backend,
                operation,
                int(workload_size),
                int(minimum_accelerated_size),
            )
        )
    )


def dense_layer(
    features: ArrayLike,
    weights: ArrayLike,
    biases: ArrayLike,
    *,
    backend: str | None = None,
) -> NDArray[np.float32]:
    """Execute a batched dense layer on the requested accelerator.

    ``weights`` uses the native row-major ``[input_width, output_width]``
    layout. Explicit accelerator requests are dispatched directly without
    calibration; ``auto`` uses the shared backend selector.
    """

    feature_array = np.asarray(features, dtype=np.float32)
    weight_array = np.asarray(weights, dtype=np.float32)
    bias_array = np.asarray(biases, dtype=np.float32)
    if feature_array.ndim != 2 or weight_array.ndim != 2 or bias_array.ndim != 1:
        raise ValueError("features and weights must be matrices and biases must be a vector")
    if feature_array.shape[1] != weight_array.shape[0]:
        raise ValueError("feature width must equal the first weights dimension")
    if weight_array.shape[1] != bias_array.shape[0]:
        raise ValueError("weights output width must equal biases length")
    function = getattr(_native, "accelerator_dense_layer_value", None)
    if function is None:
        if backend not in {None, "cpu"}:
            raise RuntimeError("native accelerator support is unavailable")
        return feature_array @ weight_array + bias_array
    output = function(
        feature_array.tolist(),
        weight_array.reshape(-1).tolist(),
        bias_array.tolist(),
        backend,
    )
    return np.asarray(output, dtype=np.float32)


def csr_diffusion(
    indptr: ArrayLike,
    indices: ArrayLike,
    weights: ArrayLike,
    values: ArrayLike,
    *,
    channels: int,
    backend: str | None = None,
) -> NDArray[np.float32]:
    """Apply a CSR graph operator to contiguous node-channel values."""

    indptr_array = np.asarray(indptr, dtype=np.uint32).reshape(-1)
    indices_array = np.asarray(indices, dtype=np.uint32).reshape(-1)
    weight_array = np.asarray(weights, dtype=np.float32).reshape(-1)
    value_array = np.asarray(values, dtype=np.float32)
    flat_values = value_array.reshape(-1)
    function = getattr(_native, "accelerator_csr_diffusion_value", None)
    if function is None:
        if backend not in {None, "cpu"}:
            raise RuntimeError("native accelerator support is unavailable")
        rows = indptr_array.shape[0] - 1
        output = np.zeros((rows, channels), dtype=np.float32)
        nodes = flat_values.reshape(-1, channels)
        for row in range(rows):
            for offset in range(int(indptr_array[row]), int(indptr_array[row + 1])):
                output[row] += weight_array[offset] * nodes[indices_array[offset]]
        return output
    output = function(
        indptr_array.tolist(),
        indices_array.tolist(),
        weight_array.tolist(),
        flat_values.tolist(),
        int(channels),
        backend,
    )
    return np.asarray(output, dtype=np.float32).reshape(indptr_array.shape[0] - 1, channels)


def graph_smooth(
    indptr: ArrayLike,
    indices: ArrayLike,
    weights: ArrayLike,
    values: ArrayLike,
    *,
    smoothing: float,
    iterations: int,
    backend: str | None = None,
) -> NDArray[np.float64]:
    """Smooth scalar node values with thresholded backend CSR iterations."""

    indptr_array = np.asarray(indptr, dtype=np.uint64).reshape(-1)
    indices_array = np.asarray(indices, dtype=np.uint64).reshape(-1)
    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    if indptr_array.size != value_array.size + 1:
        raise ValueError("indptr length must equal node count plus one")
    if indices_array.size != weight_array.size:
        raise ValueError("indices and weights must have the same length")
    if (
        value_array.size == 0
        or indptr_array[0] != 0
        or indptr_array[-1] != indices_array.size
        or np.any(indptr_array[1:] < indptr_array[:-1])
        or np.any(indices_array >= value_array.size)
        or np.any(~np.isfinite(weight_array))
        or np.any(weight_array < 0.0)
        or np.any(~np.isfinite(value_array))
    ):
        raise ValueError("graph smoothing requires valid finite non-negative CSR inputs")
    if smoothing < 0.0 or not np.isfinite(smoothing):
        raise ValueError("smoothing must be finite and non-negative")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    function = getattr(_native, "accelerator_graph_smooth_value", None)
    if function is None:
        if backend not in {None, "cpu"}:
            raise RuntimeError("native accelerator support is unavailable")
        current = value_array.copy()
        degree = np.asarray(
            [
                np.sum(weight_array[int(indptr_array[row]) : int(indptr_array[row + 1])])
                for row in range(value_array.size)
            ],
            dtype=np.float64,
        )
        for _ in range(iterations):
            neighbor_sum = csr_diffusion(
                indptr_array,
                indices_array,
                weight_array,
                current[:, None],
                channels=1,
                backend="cpu",
            ).reshape(-1)
            current = np.where(
                degree > 0.0,
                (value_array + smoothing * neighbor_sum) / (1.0 + smoothing * degree),
                current,
            )
        return current
    return np.asarray(
        function(
            indptr_array.tolist(),
            indices_array.tolist(),
            weight_array.tolist(),
            value_array.tolist(),
            float(smoothing),
            int(iterations),
            backend,
        ),
        dtype=np.float64,
    )


def csr_diffusion_backward(
    indptr: ArrayLike,
    indices: ArrayLike,
    weights: ArrayLike,
    values: ArrayLike,
    output_grad: ArrayLike,
    *,
    channels: int,
    backend: str | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Backpropagate through CSR diffusion into node values and edge weights."""

    arrays = (
        np.asarray(indptr, dtype=np.uint32).reshape(-1),
        np.asarray(indices, dtype=np.uint32).reshape(-1),
        np.asarray(weights, dtype=np.float32).reshape(-1),
        np.asarray(values, dtype=np.float32).reshape(-1),
        np.asarray(output_grad, dtype=np.float32).reshape(-1),
    )
    function = getattr(_native, "accelerator_csr_diffusion_backward_value", None)
    if function is None:
        if backend not in {None, "cpu"}:
            raise RuntimeError("native accelerator support is unavailable")
        indptr_array, indices_array, weight_array, value_array, grad_array = arrays
        nodes = value_array.reshape(-1, channels)
        gradients = grad_array.reshape(-1, channels)
        input_grad = np.zeros_like(nodes)
        edge_grad = np.zeros_like(weight_array)
        for row in range(indptr_array.shape[0] - 1):
            for offset in range(int(indptr_array[row]), int(indptr_array[row + 1])):
                node = int(indices_array[offset])
                input_grad[node] += weight_array[offset] * gradients[row]
                edge_grad[offset] = np.dot(gradients[row], nodes[node])
        return input_grad, edge_grad
    input_grad, edge_grad = function(
        *(array.tolist() for array in arrays),
        int(channels),
        backend,
    )
    return (
        np.asarray(input_grad, dtype=np.float32).reshape(-1, channels),
        np.asarray(edge_grad, dtype=np.float32),
    )


def csr_row_softmax(
    indptr: ArrayLike,
    logits: ArrayLike,
    *,
    backend: str | None = None,
) -> NDArray[np.float32]:
    """Apply independent softmax normalization to each CSR row."""

    indptr_array = np.asarray(indptr, dtype=np.uint32).reshape(-1)
    logits_array = np.asarray(logits, dtype=np.float32).reshape(-1)
    function = getattr(_native, "accelerator_csr_row_softmax_value", None)
    if function is None:
        if backend not in {None, "cpu"}:
            raise RuntimeError("native accelerator support is unavailable")
        output = np.empty_like(logits_array)
        for row in range(indptr_array.shape[0] - 1):
            start, end = int(indptr_array[row]), int(indptr_array[row + 1])
            shifted = logits_array[start:end] - np.max(logits_array[start:end])
            exponentials = np.exp(shifted)
            output[start:end] = exponentials / np.sum(exponentials)
        return output
    return np.asarray(
        function(indptr_array.tolist(), logits_array.tolist(), backend),
        dtype=np.float32,
    )


def csr_row_softmax_backward(
    indptr: ArrayLike,
    weights: ArrayLike,
    output_grad: ArrayLike,
    *,
    backend: str | None = None,
) -> NDArray[np.float32]:
    """Backpropagate through independently normalized CSR softmax rows."""

    indptr_array = np.asarray(indptr, dtype=np.uint32).reshape(-1)
    weight_array = np.asarray(weights, dtype=np.float32).reshape(-1)
    grad_array = np.asarray(output_grad, dtype=np.float32).reshape(-1)
    function = getattr(_native, "accelerator_csr_row_softmax_backward_value", None)
    if function is None:
        if backend not in {None, "cpu"}:
            raise RuntimeError("native accelerator support is unavailable")
        output = np.empty_like(weight_array)
        for row in range(indptr_array.shape[0] - 1):
            start, end = int(indptr_array[row]), int(indptr_array[row + 1])
            dot = np.dot(weight_array[start:end], grad_array[start:end])
            output[start:end] = weight_array[start:end] * (grad_array[start:end] - dot)
        return output
    return np.asarray(
        function(
            indptr_array.tolist(),
            weight_array.tolist(),
            grad_array.tolist(),
            backend,
        ),
        dtype=np.float32,
    )


def layer_norm(
    values: ArrayLike,
    gamma: ArrayLike,
    beta: ArrayLike,
    *,
    backend: str | None = None,
) -> NDArray[np.float32]:
    """Apply affine layer normalization to the last matrix dimension."""

    value_array = np.asarray(values, dtype=np.float32)
    gamma_array = np.asarray(gamma, dtype=np.float32).reshape(-1)
    beta_array = np.asarray(beta, dtype=np.float32).reshape(-1)
    if value_array.ndim != 2 or value_array.shape[1] != gamma_array.shape[0]:
        raise ValueError("values must be a matrix whose width equals gamma length")
    if beta_array.shape != gamma_array.shape:
        raise ValueError("beta and gamma must have equal lengths")
    function = getattr(_native, "accelerator_layer_norm_value", None)
    if function is None:
        if backend not in {None, "cpu"}:
            raise RuntimeError("native accelerator support is unavailable")
        mean = np.mean(value_array, axis=1, keepdims=True)
        variance = np.mean((value_array - mean) ** 2, axis=1, keepdims=True)
        return (value_array - mean) / np.sqrt(variance + 1e-5) * gamma_array + beta_array
    output = function(
        value_array.reshape(-1).tolist(),
        gamma_array.tolist(),
        beta_array.tolist(),
        value_array.shape[0],
        value_array.shape[1],
        backend,
    )
    return np.asarray(output, dtype=np.float32).reshape(value_array.shape)


def pair_sigmoid_scores(
    embeddings: ArrayLike,
    pairs: ArrayLike,
    *,
    backend: str | None = None,
) -> NDArray[np.float64]:
    """Score node pairs using sigmoid(dot(source, target))."""

    embedding_array = np.asarray(embeddings, dtype=np.float32)
    pair_array = np.asarray(pairs, dtype=np.int64)
    if embedding_array.ndim != 2 or pair_array.ndim != 2 or pair_array.shape[1] != 2:
        raise ValueError("embeddings must be a matrix and pairs must have shape [n, 2]")
    pair_rows = [(int(row[0]), int(row[1])) for row in pair_array]
    function = getattr(_native, "accelerator_pair_sigmoid_scores_value", None)
    if function is None:
        if backend not in {None, "cpu"}:
            raise RuntimeError("native accelerator support is unavailable")
        logits = np.sum(
            embedding_array[pair_array[:, 0]] * embedding_array[pair_array[:, 1]],
            axis=1,
        )
        return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    return np.asarray(function(embedding_array.tolist(), pair_rows, backend), dtype=np.float64)


def adamw_step(
    parameters: ArrayLike,
    first_moment: ArrayLike,
    second_moment: ArrayLike,
    gradients: ArrayLike,
    *,
    step: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    backend: str | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Apply one backend AdamW update and return parameters and moment state."""

    arrays = [
        np.asarray(values, dtype=np.float32).reshape(-1)
        for values in (parameters, first_moment, second_moment, gradients)
    ]
    function = getattr(_native, "accelerator_adamw_step_value", None)
    if function is None:
        if backend not in {None, "cpu"}:
            raise RuntimeError("native accelerator support is unavailable")
        params, first, second, grads = arrays
        first = 0.9 * first + 0.1 * grads
        second = 0.999 * second + 0.001 * grads * grads
        corrected_first = first / (1.0 - 0.9**step)
        corrected_second = second / (1.0 - 0.999**step)
        params = params * (1.0 - learning_rate * weight_decay) - learning_rate * corrected_first / (
            np.sqrt(corrected_second) + 1e-8
        )
        return params, first, second
    output = function(
        *(values.tolist() for values in arrays),
        int(step),
        float(learning_rate),
        float(weight_decay),
        backend,
    )
    return tuple(np.asarray(values, dtype=np.float32) for values in output)  # type: ignore[return-value]


def scalar_graph(
    initial_values: ArrayLike,
    opcodes: ArrayLike,
    left: ArrayLike,
    right: ArrayLike,
    *,
    backend: str | None = None,
) -> NDArray[np.float32]:
    """Evaluate a validated topologically ordered scalar computation graph."""

    function = getattr(_native, "accelerator_scalar_graph_value", None)
    arguments = (
        np.asarray(initial_values, dtype=np.float32).reshape(-1),
        np.asarray(opcodes, dtype=np.uint8).reshape(-1),
        np.asarray(left, dtype=np.uint32).reshape(-1),
        np.asarray(right, dtype=np.uint32).reshape(-1),
    )
    if function is None:
        if backend not in {None, "cpu"}:
            raise RuntimeError("native accelerator support is unavailable")
        raise RuntimeError("scalar graph evaluation requires the native extension")
    return np.asarray(function(*(value.tolist() for value in arguments), backend), dtype=np.float32)


def scalar_graph_train_step(
    initial_values: ArrayLike,
    opcodes: ArrayLike,
    left: ArrayLike,
    right: ArrayLike,
    parameter_ids: ArrayLike,
    *,
    loss: int,
    parameters: ArrayLike,
    first_moment: ArrayLike,
    second_moment: ArrayLike,
    step: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    backend: str | None = None,
) -> tuple[float, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Run scalar-graph reverse mode and one fused AdamW parameter update."""

    function = getattr(_native, "accelerator_scalar_graph_train_step_value", None)
    if function is None:
        raise RuntimeError("scalar graph training requires the native extension")
    output = function(
        np.asarray(initial_values, dtype=np.float32).reshape(-1).tolist(),
        np.asarray(opcodes, dtype=np.uint8).reshape(-1).tolist(),
        np.asarray(left, dtype=np.uint32).reshape(-1).tolist(),
        np.asarray(right, dtype=np.uint32).reshape(-1).tolist(),
        np.asarray(parameter_ids, dtype=np.uint32).reshape(-1).tolist(),
        int(loss),
        np.asarray(parameters, dtype=np.float32).reshape(-1).tolist(),
        np.asarray(first_moment, dtype=np.float32).reshape(-1).tolist(),
        np.asarray(second_moment, dtype=np.float32).reshape(-1).tolist(),
        int(step),
        float(learning_rate),
        float(weight_decay),
        backend,
    )
    return (
        float(output[0]),
        np.asarray(output[1], dtype=np.float32),
        np.asarray(output[2], dtype=np.float32),
        np.asarray(output[3], dtype=np.float32),
    )


def train_tanh_mlp(
    inputs: ArrayLike,
    targets: ArrayLike,
    *,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    parameters: ArrayLike,
    backend: str | None = None,
) -> NDArray[np.float32]:
    """Train the fused single-hidden-layer tanh MLP kernel."""

    function = getattr(_native, "accelerator_train_tanh_mlp_value", None)
    if function is None:
        raise RuntimeError("fused tanh MLP training requires the native extension")
    output = function(
        np.asarray(inputs, dtype=np.float32).tolist(),
        np.asarray(targets, dtype=np.float32).reshape(-1).tolist(),
        int(hidden_size),
        int(epochs),
        float(learning_rate),
        np.asarray(parameters, dtype=np.float32).reshape(-1).tolist(),
        backend,
    )
    return np.asarray(output, dtype=np.float32)


def pairwise_squared_distances(
    left: ArrayLike,
    right: ArrayLike | None = None,
    *,
    backend: str | None = None,
) -> NDArray[np.float32]:
    """Compute a complete squared-Euclidean distance matrix on a backend."""

    left_array = np.asarray(left, dtype=np.float32)
    right_array = left_array if right is None else np.asarray(right, dtype=np.float32)
    if left_array.ndim != 2 or right_array.ndim != 2:
        raise ValueError("left and right must be matrices")
    if left_array.shape[1] != right_array.shape[1]:
        raise ValueError("left and right must have the same feature width")
    function = getattr(_native, "accelerator_pairwise_squared_distances_value", None)
    if function is None:
        if backend not in {None, "cpu"}:
            raise RuntimeError("native accelerator support is unavailable")
        delta = left_array[:, None, :] - right_array[None, :, :]
        return np.sum(delta * delta, axis=2, dtype=np.float32)
    output = function(left_array.tolist(), right_array.tolist(), backend)
    return np.asarray(output, dtype=np.float32)


__all__ = [
    "adamw_step",
    "affine_scores",
    "available_backends",
    "capabilities",
    "csr_diffusion",
    "csr_diffusion_backward",
    "csr_row_softmax",
    "csr_row_softmax_backward",
    "dense_layer",
    "dispatch_report",
    "graph_smooth",
    "layer_norm",
    "pair_sigmoid_scores",
    "pairwise_squared_distances",
    "scalar_graph",
    "scalar_graph_train_step",
    "train_tanh_mlp",
    "workload_decision",
]
