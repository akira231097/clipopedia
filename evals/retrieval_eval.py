"""Offline retrieval evaluation for Clip'O'pedia.

Runs the REAL retrieval pipeline (query analysis -> HyDE -> hybrid search ->
double RRF -> diversity caps -> rerank -> scoring -> selection) against the
bundled synthetic demo corpus, using the deterministic in-memory backend
(no API keys, no network). It reports retrieval quality on a hand-labeled
query set plus end-to-end latency percentiles.

IMPORTANT: these numbers measure the *pipeline architecture* on a small
synthetic corpus with deterministic stand-in models (hash embedder, Jaccard
reranker, rule-based LLM). They are NOT production accuracy claims — they exist
to make the system measurable and the harness reproducible.

Run:  python evals/retrieval_eval.py
"""

from __future__ import annotations

import asyncio
import statistics
import time

from clipopedia.config import get_settings
from clipopedia.factory import build_demo_backend

# (query, set of relevant chunk_ids) — labeled by topic against demo/corpus.py
LABELED: list[tuple[str, set[str]]] = [
    ("best clip on AI agents and reliable tool use", {"clip-000", "clip-004"}),
    ("how do you actually evaluate whether an AI agent works", {"clip-004"}),
    ("the risk of autonomous agents doing the wrong thing at scale", {"clip-008"}),
    ("anything recent on founder burnout", {"clip-002"}),
    ("how should I think about startup pricing", {"clip-007"}),
    ("the term sheet detail that matters most when fundraising", {"clip-001"}),
    ("how to pitch investors a compelling narrative", {"clip-010"}),
    ("advice on delegation and scaling a team past ten people", {"clip-006"}),
    ("why is climate hardware so hard to build", {"clip-003"}),
    ("making remote work culture actually work", {"clip-005"}),
    ("tips for deep work and protecting focus", {"clip-011"}),
    ("supply chain resilience after the shortages", {"clip-012"}),
    ("the science of longevity and healthspan", {"clip-009"}),
    ("how to change habits so they actually stick", {"clip-013"}),
]

LATENCY_ITERS = 25  # repeats per query for stable percentiles


def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


async def main() -> None:
    settings = get_settings()
    backend = await build_demo_backend(settings)
    await backend.pipeline.run("warmup query")  # warm caches / JIT paths

    hit1 = hit3 = 0
    rr: list[float] = []
    recall5: list[float] = []
    select_correct = 0
    select_total = 0

    for query, relevant in LABELED:
        analysis = await backend.pipeline.analyze(query)
        chunks = await backend.pipeline.retrieve(query, analysis)
        ranked = [c.clip.chunk_id for c in chunks]

        if ranked and ranked[0] in relevant:
            hit1 += 1
        if any(cid in relevant for cid in ranked[:3]):
            hit3 += 1
        first = next((i + 1 for i, cid in enumerate(ranked) if cid in relevant), None)
        rr.append(1.0 / first if first else 0.0)
        top5 = set(ranked[:5])
        recall5.append(len(top5 & relevant) / len(relevant))

        _, selection = await backend.pipeline.run(query)
        select_total += 1
        if selection is not None and selection.chunk.clip.chunk_id in relevant:
            select_correct += 1

    latencies: list[float] = []
    for query, _ in LABELED:
        for _ in range(LATENCY_ITERS):
            t0 = time.perf_counter()
            await backend.pipeline.run(query)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    n = len(LABELED)
    print("Clip'O'pedia — offline retrieval evaluation")
    print(f"corpus: 14 clips | queries: {n} | latency samples: {len(latencies)}")
    print("-" * 56)
    print(f"Hit@1            {hit1 / n:.3f}")
    print(f"Hit@3            {hit3 / n:.3f}")
    print(f"Recall@5         {statistics.mean(recall5):.3f}")
    print(f"MRR              {statistics.mean(rr):.3f}")
    print(f"Selection acc.   {select_correct / select_total:.3f}")
    print("-" * 56)
    print(f"Latency p50      {_pct(latencies, 50):.2f} ms")
    print(f"Latency p95      {_pct(latencies, 95):.2f} ms")
    print(f"Latency p99      {_pct(latencies, 99):.2f} ms")
    print(f"Latency mean     {statistics.mean(latencies):.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())
