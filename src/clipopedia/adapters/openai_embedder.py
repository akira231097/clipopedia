"""Dense embeddings via the OpenAI Embeddings API."""

from __future__ import annotations

import asyncio

from openai import OpenAI

# `text-embedding-3-large` has 3072 dimensions.
_DEFAULT_DIM = 3072


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-large", dimension: int = _DEFAULT_DIM) -> None:
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        def _call() -> list[list[float]]:
            resp = self._client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in resp.data]

        return await asyncio.to_thread(_call)
