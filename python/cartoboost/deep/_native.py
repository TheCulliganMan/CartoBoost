from __future__ import annotations

import json
from typing import Any

from cartoboost import _native


def dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def loads(value: str) -> Any:
    return json.loads(value)


def require_native(name: str) -> Any:
    fn = getattr(_native, name, None)
    if fn is None:
        raise NotImplementedError(f"native deep binding {name!r} is not available")
    return fn


def available_backends() -> list[str]:
    fn = getattr(_native, "deep_available_backends_value", None)
    if fn is None:
        return ["cpu"]
    return list(fn())


def available_deep_backends() -> list[str]:
    return available_backends()


def backend_dispatch_report(backend: str | None = None, len: int = 4096) -> dict[str, Any]:
    fn = require_native("deep_backend_dispatch_report_value")
    return loads(fn(backend, int(len)))
