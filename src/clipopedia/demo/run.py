"""Offline end-to-end demo.

Builds the in-memory backend over the synthetic corpus and runs full queries
through the real retrieval pipeline — analysis, HyDE, hybrid search, fusion,
diversity caps, rerank, scoring, and selection — printing a readable trace. No
network, no API keys.
"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.table import Table

from ..config import get_settings
from ..factory import build_demo_backend
from ..models import ClipSelection, QueryAnalysis

_DEFAULT_QUERIES = [
    "best clip on AI agents and reliability",
    "anything recent on founder burnout",
    "how should I think about startup pricing",
    "hey there!",
]


def _render(console: Console, query: str, analysis: QueryAnalysis, selection: ClipSelection | None) -> None:
    console.rule(f"[bold cyan]Query[/]: {query}")
    console.print(
        f"[dim]cleaned[/]: {analysis.cleaned_query}  "
        f"[dim]small_talk[/]: {analysis.is_small_talk}  "
        f"[dim]time[/]: {analysis.time_filter.mode.value}"
    )
    console.print(
        f"[dim]entities[/]: guests={analysis.guests or '—'} "
        f"hosts={analysis.hosts or '—'} show={analysis.show or '—'}  "
        f"[dim]hyde_docs[/]: {len(analysis.hyde_documents)}"
    )
    if selection is None:
        console.print("[yellow]→ No clip (small talk or no confident match).[/]\n")
        return

    clip = selection.chunk.clip
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("Show", clip.show_title)
    table.add_row("Episode", clip.episode_title)
    table.add_row("Guest", ", ".join(clip.guests) or "—")
    table.add_row("Published", clip.published_date.isoformat() if clip.published_date else "—")
    rerank = selection.chunk.rerank_score
    table.add_row("Rerank score", f"{rerank:.3f}" if rerank is not None else "—")
    table.add_row("Final score", f"{selection.chunk.final_score:.3f}")
    table.add_row("Stage", selection.chunk.retrieval_stage)
    table.add_row("Why", selection.reason or "—")
    table.add_row("Link", clip.audio_url or "—")
    console.print(table)
    console.print(f"[green]Excerpt[/]: {clip.text[:220]}…\n")


async def run_demo(query: str | None = None) -> None:
    console = Console()
    settings = get_settings()
    console.print("[bold]Clip'O'pedia[/] — offline demo (deterministic, no API keys)\n")
    backend = await build_demo_backend(settings)

    queries = [query] if query else _DEFAULT_QUERIES
    for q in queries:
        analysis, selection = await backend.pipeline.run(q)
        _render(console, q, analysis, selection)


def main(query: str | None = None) -> None:
    asyncio.run(run_demo(query))
