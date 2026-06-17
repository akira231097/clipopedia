"""Turn a parsed time constraint into a concrete multi-bucket search plan.

A single query like "what did guests say about AI agents *lately*" has two
competing intents: semantic relevance ("AI agents") and recency ("lately").
Serving both from one search degrades both. Instead we expand the query into a
small set of weighted *buckets* — each a separate filtered search — and fuse
their results. A pure-recency bucket guarantees fresh material even when the
semantic search would have surfaced older, more on-topic clips.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..config import Settings
from ..dateutils import to_numeric, window_range
from ..models import TimeFilter, TimeMode


@dataclass
class SearchBucket:
    label: str
    include_date_clause: bool = False
    # Inclusive ``(start, end)`` as YYYYMMDD ints; ``None`` ends are open.
    date_range: tuple[int | None, int | None] | None = None
    weight: float = 1.0
    sort_by_date: bool = False
    limit: int | None = None


def _num(d: date | None) -> int | None:
    return to_numeric(d) if d else None


def make_search_plan(time_filter: TimeFilter, settings: Settings) -> list[SearchBucket]:
    """Expand a :class:`TimeFilter` into weighted search buckets."""
    if not time_filter.has_time_constraint:
        return [SearchBucket("all", weight=1.0)]

    mode = time_filter.mode
    start = _num(time_filter.start_date)
    end = _num(time_filter.end_date)

    if mode is TimeMode.latest:
        recent = window_range(settings.recent_window_days)
        return [
            # Guaranteed-fresh bucket, ranked purely by date.
            SearchBucket(
                "recency_first",
                include_date_clause=True,
                date_range=recent,
                weight=2.0,
                sort_by_date=True,
                limit=30,
            ),
            # Semantic relevance inside the recent window.
            SearchBucket(
                "recent_semantic",
                include_date_clause=True,
                date_range=recent,
                weight=settings.recency_boost,
            ),
            # Backstop so a thin recent window never returns empty.
            SearchBucket("all", weight=0.6),
        ]

    if mode is TimeMode.oldest:
        # Ascending sort is applied after fusion; one broad bucket is enough.
        return [SearchBucket("all", weight=1.0)]

    if mode is TimeMode.relative_recent:
        rng = (start, end) if (start or end) else window_range(settings.recent_window_days)
        return [
            SearchBucket("recent", include_date_clause=True, date_range=rng, weight=settings.recency_boost),
            SearchBucket("all", weight=0.5),
        ]

    # Explicit ranges: on / before / after / between.
    if mode is TimeMode.on and start:
        rng = (start, start)
    elif mode is TimeMode.before and end:
        rng = (None, end)
    elif mode is TimeMode.after and start:
        rng = (start, None)
    elif mode is TimeMode.between and (start or end):
        rng = (start, end)
    else:
        return [SearchBucket("all", weight=1.0)]

    strict = mode in (TimeMode.on, TimeMode.between)
    if strict:
        return [SearchBucket("in_range", include_date_clause=True, date_range=rng, weight=1.0)]
    # Soft before/after: prefer the range but keep a low-weight global backstop.
    return [
        SearchBucket("in_range", include_date_clause=True, date_range=rng, weight=1.1),
        SearchBucket("all", weight=0.4),
    ]


def date_range_to_filter(date_range: tuple[int | None, int | None] | None) -> dict | None:
    """Translate a bucket date range into a metadata filter clause."""
    if not date_range:
        return None
    start, end = date_range
    clause: dict[str, int] = {}
    if start is not None:
        clause["$gte"] = start
    if end is not None:
        clause["$lte"] = end
    return {"pdnumeric": clause} if clause else None
