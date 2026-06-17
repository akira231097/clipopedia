"""Cross-encoder reranking via the Cohere Rerank API."""

from __future__ import annotations

import asyncio

import cohere


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0") -> None:
        self._client = cohere.Client(api_key)
        self.model = model

    async def rerank(
        self, *, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        if not documents:
            return []

        def _call() -> list[tuple[int, float]]:
            result = self._client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
            )
            return [(r.index, float(r.relevance_score)) for r in result.results]

        return await asyncio.to_thread(_call)
