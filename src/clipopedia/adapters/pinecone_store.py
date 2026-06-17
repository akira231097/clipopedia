"""Hybrid search over a single Pinecone index.

The index must be created with the ``dotproduct`` metric — the only metric that
can store and query dense + sparse vectors together. Pinecone expresses the
dense/sparse trade-off by *scaling the vectors* before the query: dense values
are multiplied by ``alpha`` and sparse values by ``1 - alpha``. We translate our
provider-agnostic metadata filter into Pinecone's ``$and``/``$in``/``$gte``
syntax here so the rest of the pipeline never sees vendor-specific shapes.
"""

from __future__ import annotations

import asyncio

from pinecone import Pinecone

from ..models import SparseVector, VectorMatch


def _to_pinecone_filter(flt: dict | None) -> dict | None:
    if not flt:
        return None
    clauses: list[dict] = []
    if "guests" in flt:
        clauses.append({"guests": {"$in": flt["guests"]}})
    if "hosts" in flt:
        clauses.append({"hosts": {"$in": flt["hosts"]}})
    if "show" in flt:
        clauses.append({"show": {"$eq": flt["show"]}})
    if "pdnumeric" in flt:
        clauses.append({"pdnumeric": flt["pdnumeric"]})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _normalize_metadata(meta: dict) -> dict:
    """Map stored field names to the pipeline's conventions."""
    out = dict(meta)
    if "episodeId" in meta and "episode_id" not in meta:
        out["episode_id"] = meta["episodeId"]
    return out


class PineconeVectorStore:
    def __init__(self, api_key: str, index_name: str, namespace: str = "default", alpha: float = 0.7) -> None:
        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(index_name)
        self.namespace = namespace
        self.alpha = alpha

    async def hybrid_query(
        self,
        *,
        dense: list[float],
        sparse: SparseVector | None,
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> list[VectorMatch]:
        def _call() -> list[VectorMatch]:
            scaled_dense = [v * self.alpha for v in dense]
            kwargs: dict = {
                "vector": scaled_dense,
                "top_k": top_k,
                "include_metadata": True,
                "namespace": self.namespace,
            }
            if sparse and sparse.indices:
                kwargs["sparse_vector"] = {
                    "indices": sparse.indices,
                    "values": [v * (1 - self.alpha) for v in sparse.values],
                }
            pinecone_filter = _to_pinecone_filter(metadata_filter)
            if pinecone_filter:
                kwargs["filter"] = pinecone_filter
            result = self._index.query(**kwargs)
            return [
                VectorMatch(
                    chunk_id=m["id"],
                    score=float(m["score"]),
                    metadata=_normalize_metadata(m.get("metadata", {}) or {}),
                )
                for m in result.get("matches", [])
            ]

        return await asyncio.to_thread(_call)
