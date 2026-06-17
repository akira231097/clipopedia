"""Composition root.

This is the *only* place that knows which concrete adapters exist. Everything
else depends on the ports. Picking a backend ("demo" vs "live") is a single
switch here; no pipeline or orchestration code changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, get_settings
from .ports import (
    Embedder,
    LanguageModel,
    MediaStore,
    MessageSource,
    MetadataStore,
    Reranker,
    SocialClient,
    SparseEncoder,
    VectorStore,
    VisionModel,
)
from .retrieval.gazetteer import Gazetteer
from .retrieval.pipeline import RetrievalPipeline


@dataclass
class Backend:
    """A fully-wired set of adapters plus the assembled retrieval pipeline."""

    settings: Settings
    embedder: Embedder
    sparse_encoder: SparseEncoder
    vector_store: VectorStore
    metadata_store: MetadataStore
    reranker: Reranker
    llm: LanguageModel
    vision: VisionModel
    gazetteer: Gazetteer
    pipeline: RetrievalPipeline
    message_source: MessageSource | None = None
    social_client: SocialClient | None = None
    media_store: MediaStore | None = None


def _assemble_pipeline(
    settings: Settings,
    embedder: Embedder,
    sparse_encoder: SparseEncoder,
    vector_store: VectorStore,
    metadata_store: MetadataStore,
    reranker: Reranker,
    llm: LanguageModel,
    gazetteer: Gazetteer,
) -> RetrievalPipeline:
    return RetrievalPipeline(
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        vector_store=vector_store,
        metadata_store=metadata_store,
        reranker=reranker,
        llm=llm,
        gazetteer=gazetteer,
        settings=settings,
    )


async def build_demo_backend(settings: Settings | None = None) -> Backend:
    """Wire the deterministic, no-network backend used by the demo and tests."""
    settings = settings or get_settings()
    from .adapters.memory import (
        ConsoleSocialClient,
        EchoVisionModel,
        FakeEmbedder,
        FakeReranker,
        FakeSparseEncoder,
        InMemoryMessageSource,
        InMemoryMetadataStore,
        InMemoryVectorStore,
        NullMediaStore,
        RuleBasedLanguageModel,
    )
    from .demo.corpus import build_demo_corpus, build_demo_gazetteer, demo_mentions

    clips = build_demo_corpus()
    embedder = FakeEmbedder()
    sparse = FakeSparseEncoder()
    vector_store = await InMemoryVectorStore.from_clips(
        clips, embedder, sparse, alpha=settings.hybrid_alpha
    )
    metadata_store = InMemoryMetadataStore(clips)
    reranker = FakeReranker()
    llm = RuleBasedLanguageModel()
    gazetteer = build_demo_gazetteer(clips)

    pipeline = _assemble_pipeline(
        settings, embedder, sparse, vector_store, metadata_store, reranker, llm, gazetteer
    )
    return Backend(
        settings=settings,
        embedder=embedder,
        sparse_encoder=sparse,
        vector_store=vector_store,
        metadata_store=metadata_store,
        reranker=reranker,
        llm=llm,
        vision=EchoVisionModel(),
        gazetteer=gazetteer,
        pipeline=pipeline,
        message_source=InMemoryMessageSource(demo_mentions()),
        social_client=ConsoleSocialClient(),
        media_store=NullMediaStore(),
    )


async def build_live_backend(settings: Settings | None = None) -> Backend:
    """Wire the real external-service adapters (requires the ``live`` extras)."""
    settings = settings or get_settings()
    from pathlib import Path

    from .adapters.bm25_encoder import Bm25SparseEncoder
    from .adapters.cohere_reranker import CohereReranker
    from .adapters.gemini_llm import GeminiLanguageModel, GeminiVisionModel
    from .adapters.openai_embedder import OpenAIEmbedder
    from .adapters.pinecone_store import PineconeVectorStore
    from .adapters.postgres_store import PostgresMetadataStore
    from .adapters.s3_media import S3MediaStore
    from .adapters.sqs_source import SqsMessageSource
    from .adapters.throttle import AsyncMinInterval, ThrottledLanguageModel, ThrottledVisionModel
    from .adapters.twitter_client import TwitterSocialClient

    embedder = OpenAIEmbedder(settings.openai_api_key, settings.embedding_model)
    sparse = Bm25SparseEncoder()
    vector_store = PineconeVectorStore(
        settings.pinecone_api_key,
        settings.pinecone_index,
        settings.pinecone_namespace,
        settings.hybrid_alpha,
    )
    metadata_store = PostgresMetadataStore(settings.database_url)
    reranker = CohereReranker(settings.cohere_api_key, settings.rerank_model)

    # Share one limiter across the language and vision models so every hosted
    # model call (several per mention) is spaced under the provider quota.
    limiter = AsyncMinInterval(settings.min_seconds_between_llm_calls)
    llm = ThrottledLanguageModel(
        GeminiLanguageModel(settings.gemini_api_key, settings.reasoning_model), limiter
    )
    vision = ThrottledVisionModel(
        GeminiVisionModel(settings.gemini_api_key, settings.reasoning_model), limiter
    )

    # Load the canonical entity catalog so fuzzy resolution actually has names
    # to match against (falls back to an empty gazetteer if the file is absent).
    entities = Path(settings.entities_path)
    if entities.exists():
        gazetteer = Gazetteer.from_json(
            entities,
            wratio_threshold=settings.fuzzy_ratio_threshold,
            token_set_min_score=settings.token_set_min_score,
        )
    else:
        gazetteer = Gazetteer(
            wratio_threshold=settings.fuzzy_ratio_threshold,
            token_set_min_score=settings.token_set_min_score,
        )

    pipeline = _assemble_pipeline(
        settings, embedder, sparse, vector_store, metadata_store, reranker, llm, gazetteer
    )
    return Backend(
        settings=settings,
        embedder=embedder,
        sparse_encoder=sparse,
        vector_store=vector_store,
        metadata_store=metadata_store,
        reranker=reranker,
        llm=llm,
        vision=vision,
        gazetteer=gazetteer,
        pipeline=pipeline,
        message_source=SqsMessageSource(
            settings.mentions_queue_url, settings.aws_region, settings.bot_handle
        ),
        social_client=TwitterSocialClient(
            settings.twitter_consumer_key,
            settings.twitter_consumer_secret,
            settings.twitter_access_token,
            settings.twitter_access_token_secret,
            settings.twitter_bearer_token,
        ),
        media_store=S3MediaStore(settings.clip_bucket, settings.aws_region),
    )


async def build_backend(settings: Settings | None = None) -> Backend:
    """Build whichever backend ``CLIPOPEDIA_BACKEND`` selects."""
    settings = settings or get_settings()
    if settings.backend == "live":
        return await build_live_backend(settings)
    return await build_demo_backend(settings)
