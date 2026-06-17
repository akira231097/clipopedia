"""Application settings.

All configuration is read from the environment (and an optional ``.env`` file)
via ``pydantic-settings``. Secrets default to empty strings so the package can
be imported and the demo can run without any credentials present.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, environment-driven configuration.

    Provider keys use their conventional names (``OPENAI_API_KEY`` …). App-level
    knobs are prefixed ``CLIPOPEDIA_`` to avoid colliding with anything else in
    the environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Runtime ------------------------------------------------------------
    backend: Literal["demo", "live"] = Field(
        default="demo",
        validation_alias=AliasChoices("CLIPOPEDIA_BACKEND", "BACKEND"),
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("CLIPOPEDIA_LOG_LEVEL", "LOG_LEVEL"),
    )
    poll_interval_seconds: float = 10.0
    min_seconds_between_llm_calls: float = 6.0  # stay under provider rate limits

    # ---- Providers / credentials -------------------------------------------
    openai_api_key: str = ""
    gemini_api_key: str = ""
    cohere_api_key: str = ""
    pinecone_api_key: str = ""

    # Hybrid search uses a single Pinecone index created with the `dotproduct`
    # metric, which is the only metric that can hold dense + sparse vectors.
    pinecone_index: str = "clip-hybrid"
    pinecone_namespace: str = "default"

    # Canonical entity catalog used for fuzzy resolution in the live backend.
    entities_path: str = "data/entities.example.json"

    database_url: str = ""

    aws_region: str = "us-east-1"
    mentions_queue_url: str = ""
    clip_bucket: str = ""

    twitter_consumer_key: str = ""
    twitter_consumer_secret: str = ""
    twitter_access_token: str = ""
    twitter_access_token_secret: str = ""
    twitter_bearer_token: str = ""
    bot_handle: str = "@clipopedia"

    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_project: str = "clipopedia"

    # ---- Model selection ----------------------------------------------------
    embedding_model: str = "text-embedding-3-large"
    reasoning_model: str = "gemini-2.5-flash"
    rerank_model: str = "rerank-english-v3.0"

    # ---- Retrieval tuning (see docs/RAG_PIPELINE.md) ------------------------
    hybrid_alpha: float = 0.7          # weight on dense vs. sparse (0.3) scores
    pinecone_top_k: int = 100          # candidates pulled per query vector
    rerank_top_n: int = 60             # candidates handed to the cross-encoder
    llm_selector_input_k: int = 50     # candidates handed to the LLM selector
    final_top_k: int = 3               # clips returned to the caller

    # HyDE embedding-fusion weights.
    original_query_weight: float = 1.25
    hyde_weight_max: float = 1.10
    hyde_weight_min: float = 0.85

    # Recency strategy.
    recent_window_days: int = 45
    recent_window_days_max: int = 365
    recency_boost: float = 1.25
    relevance_floor: float = 0.35

    # Diversity controls.
    per_episode_cap: int = 15
    min_per_bucket: int = 7

    # Fuzzy entity matching thresholds.
    fuzzy_ratio_threshold: int = 85
    token_set_min_score: int = 75

    # Reciprocal-rank-fusion constant.
    rrf_k: int = 60


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
