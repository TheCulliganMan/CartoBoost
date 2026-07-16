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
    "affine_scores",
    "available_backends",
    "capabilities",
    "csr_diffusion",
    "dense_layer",
    "pairwise_squared_distances",
    "workload_decision",
]
