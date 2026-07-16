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


__all__ = ["available_backends", "capabilities", "dense_layer", "workload_decision"]
