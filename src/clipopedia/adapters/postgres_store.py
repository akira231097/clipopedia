"""Resolve chunk ids to full :class:`Clip` records from PostgreSQL.

The vector store holds only ids + light metadata; the source of truth for clip
content (transcript text, titles, media URLs, timestamps) is a relational store.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime

import psycopg
from psycopg.rows import dict_row

from ..models import Clip

_SELECT = """
    SELECT chunk_id, episode_id, show_title, episode_title, text,
           guests, hosts, speakers, topics, published_date,
           start_ms, end_ms, duration_ms, audio_url, video_url
    FROM {table}
    WHERE chunk_id = ANY(%s)
"""


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _as_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _row_to_clip(row: dict) -> Clip:
    return Clip(
        chunk_id=row["chunk_id"],
        episode_id=row.get("episode_id", ""),
        show_title=row.get("show_title") or "",
        episode_title=row.get("episode_title") or "",
        text=row.get("text") or "",
        guests=_as_list(row.get("guests")),
        hosts=_as_list(row.get("hosts")),
        speakers=_as_list(row.get("speakers")),
        topics=_as_list(row.get("topics")),
        published_date=_as_date(row.get("published_date")),
        start_ms=row.get("start_ms"),
        end_ms=row.get("end_ms"),
        duration_ms=row.get("duration_ms"),
        audio_url=row.get("audio_url"),
        video_url=row.get("video_url"),
    )


class PostgresMetadataStore:
    def __init__(self, dsn: str, table: str = "clips") -> None:
        self._dsn = dsn
        self._table = table

    async def fetch(self, chunk_ids: list[str]) -> dict[str, Clip]:
        if not chunk_ids:
            return {}

        def _call() -> dict[str, Clip]:
            query = _SELECT.format(table=self._table)
            with psycopg.connect(self._dsn) as conn, conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, (chunk_ids,))
                rows = cur.fetchall()
            return {row["chunk_id"]: _row_to_clip(row) for row in rows}

        return await asyncio.to_thread(_call)
