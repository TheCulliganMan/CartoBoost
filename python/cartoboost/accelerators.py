"""Runtime accelerator inventory and operation-level capabilities."""

from __future__ import annotations

import json
from typing import Any

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


__all__ = ["available_backends", "capabilities", "workload_decision"]
