"""Domain models shared across the pipeline.

These are deliberately provider-agnostic: nothing here knows about Pinecone,
OpenAI, SQS, or X. Adapters translate vendor payloads into these types at the
edges, so the retrieval and orchestration code only ever deals with clean,
validated objects.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Incoming message types
# --------------------------------------------------------------------------- #
class MediaKind(StrEnum):
    image = "image"
    video = "video"
    gif = "gif"


class MediaItem(BaseModel):
    url: str
    kind: MediaKind = MediaKind.image
    # Natural-language description produced by a vision model, if extracted.
    description: str | None = None


class Mention(BaseModel):
    """A single inbound social mention to respond to."""

    id: str
    text: str
    author_handle: str = ""
    author_name: str = ""
    media: list[MediaItem] = Field(default_factory=list)
    # Text of a replied-to or quoted post, used as extra context.
    referenced_text: str | None = None
    # Opaque token the message source uses to acknowledge / delete the message.
    ack_token: str | None = None


# --------------------------------------------------------------------------- #
# Query understanding
# --------------------------------------------------------------------------- #
class TimeMode(StrEnum):
    none = "none"
    latest = "latest"
    oldest = "oldest"
    on = "on"
    before = "before"
    after = "after"
    between = "between"
    relative_recent = "relative_recent"


class TimeFilter(BaseModel):
    has_time_constraint: bool = False
    mode: TimeMode = TimeMode.none
    start_date: date | None = None
    end_date: date | None = None
    anchor_date: date | None = None
    approx_window_days: int | None = None
    sort_preference: str | None = None  # "latest" | "oldest"
    gating: str | None = None           # "soft" | "hard"
    recall_ratio: float | None = None


class QueryAnalysis(BaseModel):
    """Structured understanding of a user's request."""

    cleaned_query: str
    is_small_talk: bool = False
    intent: str = "search"
    complexity: str = "simple"  # "simple" | "complex"
    guests: list[str] = Field(default_factory=list)
    hosts: list[str] = Field(default_factory=list)
    show: str | None = None
    topics: list[str] = Field(default_factory=list)
    hyde_documents: list[str] = Field(default_factory=list)
    time_filter: TimeFilter = Field(default_factory=TimeFilter)


# --------------------------------------------------------------------------- #
# Retrieval types
# --------------------------------------------------------------------------- #
class SparseVector(BaseModel):
    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


class VectorMatch(BaseModel):
    """A raw hit from the vector store, before metadata enrichment."""

    chunk_id: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Clip(BaseModel):
    """A retrievable segment of a podcast episode."""

    chunk_id: str
    episode_id: str
    show_title: str = ""
    episode_title: str = ""
    text: str = ""
    guests: list[str] = Field(default_factory=list)
    hosts: list[str] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    published_date: date | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    duration_ms: int | None = None
    audio_url: str | None = None
    video_url: str | None = None


class RetrievedChunk(BaseModel):
    """A clip plus every score it has accumulated along the pipeline."""

    clip: Clip
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0           # fused score after RRF
    rerank_score: float | None = None   # cross-encoder relevance
    final_score: float = 0.0            # after recency + metadata scoring
    bucket: str = "all"                 # which time bucket produced it
    retrieval_stage: str = "primary"    # primary | recency_first | safety_recall

    @property
    def id(self) -> str:
        return self.clip.chunk_id


class SelectionScores(BaseModel):
    relevance: float = 0.0
    depth: float = 0.0
    completeness: float = 0.0
    authority: float = 0.0
    temporal_fit: float = 0.0
    coherence: float = 0.0
    query_coverage: float = 0.0
    overall: float = 0.0


class ClipSelection(BaseModel):
    """The final clip chosen by the LLM selector, with its rationale."""

    chunk: RetrievedChunk
    reason: str = ""
    completeness_level: str = ""
    scores: SelectionScores = Field(default_factory=SelectionScores)


# --------------------------------------------------------------------------- #
# Outgoing reply types
# --------------------------------------------------------------------------- #
class BotReply(BaseModel):
    text: str
    in_reply_to: str
    clip_link: str | None = None
    video_ref: str | None = None  # storage reference for an optional video clip


class PublishResult(BaseModel):
    success: bool
    reply_id: str | None = None
    error: str | None = None
    error_type: str | None = None  # rate_limit | forbidden | unknown
