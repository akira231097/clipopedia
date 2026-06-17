"""The hybrid-RAG retrieval pipeline.

End-to-end flow for a single query:

    analyze ─► embed (dense + sparse, original + HyDE) ─► weight by HyDE sim
            ─► per-bucket hybrid search ─► RRF fuse query vectors
            ─► RRF fuse buckets ─► episode/diversity caps ─► hydrate metadata
            ─► duration filter ─► cross-encoder rerank ─► signal scoring
            ─► relevance floor ─► LLM selection

The pipeline depends only on the :mod:`clipopedia.ports` protocols, so the exact
same code runs against real services or the in-memory demo fakes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ..config import Settings
from ..models import ClipSelection, QueryAnalysis, RetrievedChunk, VectorMatch
from ..ports import (
    Embedder,
    LanguageModel,
    MetadataStore,
    Reranker,
    SparseEncoder,
    VectorStore,
)
from .fusion import reciprocal_rank_fusion
from .gazetteer import Gazetteer
from .hyde import compute_query_weights
from .query_analysis import analyze_query
from .scoring import (
    apply_relevance_floor,
    enforce_episode_cap_and_bucket_quota,
    score_and_rank,
)
from .selection import select_best_clip
from .time_planning import SearchBucket, date_range_to_filter, make_search_plan

logger = logging.getLogger(__name__)

# Clips longer than this are skipped — too long to be a shareable "clip".
_MAX_CLIP_DURATION_MS = 600_000  # 10 minutes


@dataclass
class RetrievalPipeline:
    embedder: Embedder
    sparse_encoder: SparseEncoder
    vector_store: VectorStore
    metadata_store: MetadataStore
    reranker: Reranker
    llm: LanguageModel
    gazetteer: Gazetteer
    settings: Settings

    async def analyze(self, raw_query: str) -> QueryAnalysis:
        return await analyze_query(self.llm, raw_query, self.gazetteer, self.settings)

    def _metadata_filter(self, analysis: QueryAnalysis, bucket: SearchBucket) -> dict | None:
        flt: dict = {}
        if analysis.guests:
            flt["guests"] = analysis.guests
        if analysis.hosts:
            flt["hosts"] = analysis.hosts
        if analysis.show:
            flt["show"] = analysis.show
        if bucket.include_date_clause:
            date_clause = date_range_to_filter(bucket.date_range)
            if date_clause:
                flt.update(date_clause)
        return flt or None

    async def _search_bucket(
        self,
        bucket: SearchBucket,
        analysis: QueryAnalysis,
        dense: list[list[float]],
        sparse: list,
        weights: list[float],
    ) -> list[VectorMatch]:
        md_filter = self._metadata_filter(analysis, bucket)
        tasks = [
            self.vector_store.hybrid_query(
                dense=dense[i],
                sparse=sparse[i] if i < len(sparse) else None,
                top_k=self.settings.pinecone_top_k,
                metadata_filter=md_filter,
            )
            for i in range(len(dense))
        ]
        per_query = await asyncio.gather(*tasks)
        fused = reciprocal_rank_fusion(
            per_query, weights=weights, k=self.settings.rrf_k, top_k=self.settings.rerank_top_n
        )
        if bucket.sort_by_date:
            fused.sort(key=lambda m: m.metadata.get("pdnumeric", 0), reverse=True)
            if bucket.limit:
                fused = fused[: bucket.limit]
        for m in fused:
            m.metadata["bucket"] = bucket.label
        return fused

    async def retrieve(self, raw_query: str, analysis: QueryAnalysis) -> list[RetrievedChunk]:
        # 1. Embed the original query plus every HyDE document.
        queries = [raw_query] + analysis.hyde_documents
        dense, sparse = await asyncio.gather(
            self.embedder.embed(queries), self.sparse_encoder.encode(queries)
        )
        weights = compute_query_weights(
            dense,
            original_weight=self.settings.original_query_weight,
            hyde_high=self.settings.hyde_weight_max,
            hyde_low=self.settings.hyde_weight_min,
        )

        # 2. Run each time bucket and fuse across buckets.
        plan = make_search_plan(analysis.time_filter, self.settings)
        per_bucket = await asyncio.gather(
            *(self._search_bucket(b, analysis, dense, sparse, weights) for b in plan)
        )
        combined = reciprocal_rank_fusion(
            per_bucket,
            weights=[b.weight for b in plan],
            k=self.settings.rrf_k,
            top_k=self.settings.rerank_top_n,
        )
        combined = enforce_episode_cap_and_bucket_quota(
            combined,
            per_episode_cap=self.settings.per_episode_cap,
            min_per_bucket=self.settings.min_per_bucket,
        )

        # 3. Safety recall: if an entity filter zeroed everything out, retry open.
        if not combined and (analysis.guests or analysis.hosts or analysis.show):
            logger.warning("Entity filter returned nothing; running unfiltered safety recall")
            safety = await self.vector_store.hybrid_query(
                dense=dense[0],
                sparse=sparse[0] if sparse else None,
                top_k=self.settings.pinecone_top_k,
                metadata_filter=None,
            )
            combined = safety[: self.settings.rerank_top_n]
            for m in combined:
                m.metadata.setdefault("bucket", "safety_recall")

        # 4. Hydrate chunk ids into full Clip records.
        ids = [m.chunk_id for m in combined]
        clips = await self.metadata_store.fetch(ids)
        chunks: list[RetrievedChunk] = []
        for m in combined:
            clip = clips.get(m.chunk_id)
            if clip is None:
                continue
            bucket = m.metadata.get("bucket", "all")
            chunks.append(
                RetrievedChunk(
                    clip=clip,
                    hybrid_score=m.score,
                    bucket=bucket,
                    retrieval_stage=bucket,
                )
            )

        # 5. Drop over-long clips.
        chunks = [
            c
            for c in chunks
            if c.clip.duration_ms is None or c.clip.duration_ms < _MAX_CLIP_DURATION_MS
        ]
        if not chunks:
            return []

        # 6. Cross-encoder rerank.
        order = await self.reranker.rerank(
            query=raw_query,
            documents=[c.clip.text for c in chunks],
            top_n=min(len(chunks), self.settings.rerank_top_n),
        )
        reranked: list[RetrievedChunk] = []
        for idx, score in order:
            if 0 <= idx < len(chunks):
                chunks[idx].rerank_score = score
                reranked.append(chunks[idx])
        chunks = reranked or chunks

        # 7. Signal-based scoring + relevance floor.
        chunks = score_and_rank(chunks, analysis, self.settings)
        chunks = apply_relevance_floor(
            chunks, self.settings.relevance_floor, min_keep=self.settings.final_top_k
        )
        return chunks

    async def run(self, raw_query: str) -> tuple[QueryAnalysis, ClipSelection | None]:
        """Analyze, retrieve, and select. Returns the analysis and chosen clip."""
        analysis = await self.analyze(raw_query)
        if analysis.is_small_talk:
            return analysis, None
        chunks = await self.retrieve(raw_query, analysis)
        selection = await select_best_clip(self.llm, raw_query, chunks, self.settings)
        return analysis, selection
