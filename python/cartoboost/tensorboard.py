from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_training_history(
    model: Any,
    log_dir: str | Path | None,
    *,
    run_name: str | None = None,
) -> None:
    """Write native training scalars to TensorBoard when explicitly requested."""
    if log_dir is None:
        return
    writer_cls = _summary_writer_class()
    history = _training_history(model)
    path = Path(log_dir)
    if run_name:
        path = path / run_name
    writer = writer_cls(str(path))
    try:
        for row in history:
            writer.add_scalar(str(row["name"]), float(row["value"]), int(row["iteration"]))
    finally:
        writer.close()


def _summary_writer_class() -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter
    except ImportError:
        pass
    try:
        from tensorboardX import SummaryWriter

        return SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging requires an optional writer. Install with "
            "`pip install cartoboost[tensorboard]` or install `tensorboardX`."
        ) from exc


def _training_history(model: Any) -> list[dict[str, Any]]:
    payload = getattr(model, "training_history_json", "[]")
    if callable(payload):
        payload = payload()
    rows = json.loads(payload or "[]")
    if not isinstance(rows, list):
        raise ValueError("native training history must be a list")
    return rows
