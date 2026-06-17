"""Sparse (BM25) encoding via ``pinecone-text``.

A BM25 model can be fit on your corpus offline and loaded from JSON; absent
that, the library's default English parameters work out of the box.
"""

from __future__ import annotations

import asyncio

from pinecone_text.sparse import BM25Encoder

from ..models import SparseVector


class Bm25SparseEncoder:
    def __init__(self, model_path: str | None = None) -> None:
        if model_path:
            self._bm25 = BM25Encoder()
            self._bm25.load(model_path)
        else:
            self._bm25 = BM25Encoder.default()

    async def encode(self, texts: list[str]) -> list[SparseVector]:
        def _call() -> list[SparseVector]:
            out: list[SparseVector] = []
            for text in texts:
                enc = self._bm25.encode_queries(text)
                out.append(SparseVector(indices=enc["indices"], values=enc["values"]))
            return out

        return await asyncio.to_thread(_call)
