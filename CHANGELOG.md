# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-17

Initial public release of Clip'O'pedia — a mention-driven, hybrid-RAG assistant
that listens for questions and replies with the single most relevant podcast clip.

### Added

- **Hexagonal ports-and-adapters core.** The retrieval and orchestration layers
  depend only on the `Protocol` interfaces in `ports.py` (`Embedder`,
  `SparseEncoder`, `VectorStore`, `MetadataStore`, `Reranker`, `LanguageModel`,
  `VisionModel`, `MessageSource`, `SocialClient`, `MediaStore`). `factory.py` is
  the single composition root; `CLIPOPEDIA_BACKEND=demo|live` selects the backend.
- **Fully offline, deterministic demo** (`python -m clipopedia demo`). In-memory
  fakes in `adapters/memory.py` — a hashing embedder, real hybrid `dotproduct`
  scoring with metadata filters, a Jaccard reranker, and a rule-based JSON LLM —
  run the entire pipeline with no network and no API keys. The test suite runs
  against the same fakes (no patched SDKs).
- **Hybrid retrieval** combining dense embeddings (OpenAI `text-embedding-3-large`)
  and sparse BM25 (`pinecone-text`) in a single Pinecone `dotproduct` index, with
  a tunable `HYBRID_ALPHA` dense/sparse trade-off.
- **HyDE with weighted fusion.** The analyzer generates hypothetical transcript-style
  answers; each is weighted by cosine similarity to the original-query embedding,
  decaying from `hyde_weight_max` to `hyde_weight_min`, with the original query
  always weighted highest.
- **Double Reciprocal Rank Fusion** (`fusion.py`): an inner fusion across per-query
  vectors and an outer fusion across time buckets, combining by rank for robustness
  to differing score distributions.
- **Time-bucket planning** (`time_planning.py`) that decomposes recency-aware
  queries (`latest`, `before`, `after`, `between`, `relative_recent`, ...) into
  separate weighted recency and relevance searches.
- **Diversity and safety controls** (`scoring.py`): per-episode cap, minimum
  per-bucket quota, and an unfiltered safety-recall search when an entity filter
  zeroes everything out.
- **Cross-encoder reranking** via Cohere `rerank-english-v3.0`, followed by
  explainable multiplicative recency and metadata-agreement boosts and a relevance
  floor that never returns fewer than `final_top_k` clips.
- **LLM clip selection** (`selection.py`) that picks the single best clip and
  returns a rationale.
- **Fuzzy gazetteer** (`gazetteer.py`) using RapidFuzz (`WRatio` + `token_set_ratio`)
  to reconcile noisy entity surface forms to canonical catalog names.
- **LangGraph orchestration** (`orchestration/`): an `extract_context → analyze →
  (small_talk | search_clips → generate_reply) → publish` state machine, with
  multimodal context extraction via a vision model.
- **Long-running bot worker** (`bot.py`): a poll → run-graph → acknowledge loop
  that only acks a message after a reply is successfully published, so transient
  failures are retried rather than dropped.
- **Live adapters** for OpenAI, Pinecone, Cohere, Gemini (OpenAI-compatible
  endpoint, including vision), PostgreSQL (`psycopg` 3), AWS SQS/S3 (`boto3`), and
  X/Twitter (`tweepy`), plus a shared rate-limiting throttle for hosted model calls.
- **Environment-driven configuration** via `pydantic-settings` (`config.py`); every
  retrieval threshold is overridable and no secrets are hard-coded.
- **CLI** (`clipopedia demo` / `clipopedia run`) with Rich-rendered demo traces.
- **Packaging and deployment**: `pyproject.toml` with `live`/`dev` extras, a
  Dockerfile, `docker-compose.yml`, an AWS ECS Fargate task-definition template
  (secrets via Secrets Manager), and GitHub Actions CI on Python 3.11 and 3.12.

[1.0.0]: https://github.com/akira231097/clipopedia/releases/tag/v1.0.0