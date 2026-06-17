"""Ports — the abstract interfaces the pipeline depends on.

This is the "ports and adapters" (hexagonal) seam of the system. The retrieval
and orchestration layers depend only on these ``Protocol`` types, never on a
concrete vendor SDK. Two families of adapters implement them:

* ``adapters/`` — real services (OpenAI, Pinecone, Cohere, Gemini, SQS, X, …)
* ``adapters/memory.py`` — deterministic, in-process fakes used by the demo
  and the test suite.

Because everything is a ``Protocol``, swapping a backend is a wiring change in
``factory.py`` — no pipeline code changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    BotReply,
    Clip,
    MediaItem,
    Mention,
    PublishResult,
    SparseVector,
    VectorMatch,
)


@runtime_checkable
class Embedder(Protocol):
    """Produces dense embeddings for a batch of texts."""

    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class SparseEncoder(Protocol):
    """Produces sparse (lexical / BM25-style) vectors for a batch of texts."""

    async def encode(self, texts: list[str]) -> list[SparseVector]: ...


@runtime_checkable
class VectorStore(Protocol):
    """A hybrid (dense + sparse) vector index."""

    async def hybrid_query(
        self,
        *,
        dense: list[float],
        sparse: SparseVector | None,
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> list[VectorMatch]: ...


@runtime_checkable
class MetadataStore(Protocol):
    """Resolves chunk ids to fully-hydrated :class:`Clip` records."""

    async def fetch(self, chunk_ids: list[str]) -> dict[str, Clip]: ...


@runtime_checkable
class Reranker(Protocol):
    """A cross-encoder reranker. Returns ``(index, score)`` pairs, best first."""

    async def rerank(
        self, *, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]: ...


@runtime_checkable
class LanguageModel(Protocol):
    """A chat-completion model used for analysis, selection, and replies."""

    async def complete(self, *, system: str, user: str, json_mode: bool = False) -> str: ...


@runtime_checkable
class VisionModel(Protocol):
    """Describes an image / video / GIF in natural language."""

    async def describe(self, media: MediaItem) -> str: ...


@runtime_checkable
class MessageSource(Protocol):
    """A source of inbound mentions (e.g. an SQS queue)."""

    async def poll(self) -> Mention | None: ...

    async def ack(self, mention: Mention) -> None: ...


@runtime_checkable
class SocialClient(Protocol):
    """Posts replies back to the social platform.

    ``media`` is optional clip bytes to attach (e.g. a short MP4); when present
    the client uploads it and includes it in the reply.
    """

    async def publish_reply(self, reply: BotReply, media: bytes | None = None) -> PublishResult: ...


@runtime_checkable
class MediaStore(Protocol):
    """Fetches a stored media clip (e.g. an MP4 in object storage)."""

    async def fetch_clip(self, ref: str) -> bytes | None: ...
