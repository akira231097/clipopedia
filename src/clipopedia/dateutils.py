"""Small date helpers.

Dates are compared throughout the pipeline as ``YYYYMMDD`` integers (e.g.
``20260617``). Integers sort chronologically, survive a round-trip through a
vector-store metadata field, and avoid timezone ambiguity.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta


def to_numeric(d: date) -> int:
    """Convert a date to its ``YYYYMMDD`` integer form."""
    return d.year * 10000 + d.month * 100 + d.day


def from_numeric(n: int) -> date:
    return date(n // 10000, (n % 10000) // 100, n % 100)


def today_utc() -> date:
    return datetime.now(UTC).date()


def parse_iso_date(value: str | None) -> date | None:
    """Best-effort parse of an ISO-8601 date/datetime string."""
    if not value:
        return None
    cleaned = value.strip().strip('"').strip("'").rstrip(",")
    try:
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        try:
            return date.fromisoformat(cleaned[:10])
        except (ValueError, TypeError):
            return None


def window_range(window_days: int, *, anchor: date | None = None) -> tuple[int, int]:
    """Return the ``(start, end)`` numeric range for the last ``window_days``."""
    end = anchor or today_utc()
    start = end - timedelta(days=max(1, window_days))
    return to_numeric(start), to_numeric(end)
