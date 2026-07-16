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


__all__ = ["available_backends", "capabilities"]
