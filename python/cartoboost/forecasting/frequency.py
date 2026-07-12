from __future__ import annotations

from numbers import Integral
from typing import Any

_PANDAS_INSTALL_HINT = (
    "Forecasting dataframe utilities require pandas. Install it with "
    "`pip install pandas` or add pandas to your environment."
)


def require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(_PANDAS_INSTALL_HINT) from exc
    return pd


def normalize_frequency(freq: str | None) -> str | None:
    """Return pandas' canonical frequency string for an explicit frequency."""

    if freq is None:
        return None
    if not isinstance(freq, str) or not freq.strip():
        raise ValueError("frequency must be a non-empty string")
    # Pandas 3 rejects several legacy uppercase aliases (notably ``H``), while
    # the native boundary intentionally uses stable uppercase aliases.  Handle
    # those aliases before asking pandas to parse the value so saved artifacts
    # remain reloadable across pandas versions.
    normalized = freq.strip()
    lower = normalized.lower()
    if lower in {"h", "1h", "hour", "hourly"}:
        return "H"
    if lower in {"d", "1d", "day", "daily"}:
        return "D"
    if lower in {"w", "1w"} or lower.startswith("w-"):
        return "W"
    pd = require_pandas()
    try:
        return _native_frequency_alias(pd.tseries.frequencies.to_offset(normalized).freqstr)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid frequency {freq!r}") from exc


def _native_frequency_alias(freq: str) -> str:
    normalized = str(freq)
    lower = normalized.lower()
    if lower in {"h", "1h", "hour", "hourly"}:
        return "H"
    if lower in {"d", "1d", "day", "daily"}:
        return "D"
    if lower == "w" or lower == "1w" or lower.startswith("w-"):
        return "W"
    return normalized


def _pandas_frequency_alias(freq: str) -> str:
    """Map native stable aliases to spellings accepted by current pandas."""

    return "h" if freq == "H" else freq


def infer_frequency(timestamps: Any) -> str | None:
    """Infer a regular pandas frequency from parseable timestamps."""

    pd = require_pandas()
    values = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise")).sort_values()
    if values.hasnans:
        raise ValueError("timestamps must be parseable and non-null")
    if len(values) < 2:
        return None
    if len(values) == 2:
        delta = values[1] - values[0]
        if delta <= pd.Timedelta(0):
            return None
        try:
            return _native_frequency_alias(pd.tseries.frequencies.to_offset(delta).freqstr)
        except ValueError:
            return None
    inferred = pd.infer_freq(values)
    return normalize_frequency(inferred) if inferred is not None else None


def validate_regular_frequency(
    timestamps: Any,
    freq: str,
    *,
    label: str = "series",
) -> str:
    """Validate that timestamps exactly follow freq after sorting."""

    pd = require_pandas()
    normalized = normalize_frequency(freq)
    assert normalized is not None
    values = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise")).sort_values()
    if values.hasnans:
        raise ValueError("timestamps must be parseable and non-null")
    if len(values) <= 1:
        return normalized
    pandas_freq = _pandas_frequency_alias(normalized)
    expected = pd.date_range(start=values[0], periods=len(values), freq=pandas_freq)
    # Pandas 3 may preserve microsecond-resolution input while constructing a
    # nanosecond-resolution expected range.  Compare on one explicit unit so a
    # valid regular panel is not rejected solely because of dtype metadata.
    actual_ns = values.astype("datetime64[ns]")
    expected_ns = expected.astype("datetime64[ns]")
    if not actual_ns.equals(expected_ns):
        raise ValueError(f"{label} timestamps are irregular for frequency {normalized!r}")
    return normalized


def next_timestamps(last_timestamp: Any, horizon: int, freq: str) -> list[Any]:
    """Return the next horizon timestamps after last_timestamp."""

    validate_horizon(horizon)
    pd = require_pandas()
    normalized = normalize_frequency(freq)
    assert normalized is not None
    pandas_freq = _pandas_frequency_alias(normalized)
    start = pd.Timestamp(last_timestamp) + pd.tseries.frequencies.to_offset(pandas_freq)
    return list(pd.date_range(start=start, periods=horizon, freq=pandas_freq))


def validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool):
        raise ValueError("horizon must be a positive integer")
    if not isinstance(horizon, Integral):
        raise ValueError("horizon must be a positive integer")
    try:
        value = int(horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError("horizon must be a positive integer") from exc
    if value != horizon:
        raise ValueError("horizon must be a positive integer")
    if value <= 0:
        raise ValueError("horizon must be a positive integer")
    return value
