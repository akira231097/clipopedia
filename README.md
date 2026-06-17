<h1 align="center">Clip'O'pedia 🎧</h1>

<p align="center">
  <em>A mention-driven, hybrid-RAG assistant that listens for questions and replies with the single best podcast clip.</em>
</p>

<p align="center">
  <a href="https://github.com/akira231097/clipopedia/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/akira231097/clipopedia/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="LangGraph" src="https://img.shields.io/badge/orchestration-LangGraph-1c3c3c">
  <img alt="Hybrid RAG" src="https://img.shields.io/badge/RAG-hybrid%20%2B%20HyDE%20%2B%20rerank-8a2be2">
  <img alt="Lint" src="https://img.shields.io/badge/lint-ruff-black">
</p>

---

## Overview

Tag the bot under a post — *"what's a good clip on founder burnout?"* — and it finds the most relevant moment across a large library of podcast episodes and replies with a link. Under the hood it is a production-shaped Retrieval-Augmented Generation pipeline: **multimodal context extraction → query understanding → HyDE → hybrid (dense + sparse) search → reciprocal-rank fusion → cross-encoder reranking → LLM selection**, orchestrated as a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine and deployed as a long-running worker.

This is a clean-room **reference implementation** built to demonstrate the architecture of a clip-recommendation assistant. It ships with a fully **offline demo** (deterministic fakes, no API keys) so you can run the entire pipeline in seconds, plus real adapters for OpenAI, Pinecone, Cohere, Gemini, PostgreSQL, AWS SQS/S3, and the X API. Because every layer depends only on abstract interfaces, swapping the whole backend from offline-fakes to real services is a single environment variable.

## Try it

Run the **offline** pipeline locally — no API keys, no network:

```bash
pip install -e . gradio
python app.py                 # interactive web demo (Gradio)
# or
python -m clipopedia demo     # readable CLI trace
```

🚀 **One-click live demo** — deploy the offline app to Hugging Face Spaces in ~3 minutes: see [deploy/HF_SPACE.md](deploy/HF_SPACE.md).
📐 **Architecture diagram + design deep-dive:** [docs/DESIGN.md](docs/DESIGN.md).

## Key features

- **Hybrid retrieval** — dense embeddings for meaning + sparse/BM25 for exact names and jargon, fused with a tunable `alpha`.
- **HyDE** — embeds *hypothetical answers*, not just the terse query, with fusion weights that decay by how on-topic each hypothetical is.
- **Double reciprocal rank fusion** — fuses per-query result lists, then fuses across time buckets, combining by rank rather than raw score.
- **Time-bucket planning** — decomposes a "lately" query into separate weighted recency and relevance searches instead of one compromised ranking.
- **Cross-encoder reranking** — re-scores the shortlist with full query↔passage attention ("retrieve wide, rerank narrow").
- **Explainable scoring** — multiplicative recency and metadata-agreement boosts plus a relevance floor, so the final order traces back to concrete signals.
- **LLM selection** — a final model picks the single best clip and justifies it across relevance, depth, completeness, and more.
- **Fuzzy gazetteer** — reconciles "lena ortiz" / "the long game show" to canonical catalog names before filtering.
- **Hexagonal architecture** — the pipeline depends only on `Protocol` ports; demo vs. live is one env var, no pipeline code changes.
- **Runs fully offline** — deterministic in-memory fakes power both the demo and the test suite, with no network and no credentials.

## Evaluation

The retrieval pipeline is measured by a reproducible offline harness ([`evals/retrieval_eval.py`](evals/retrieval_eval.py)) over a hand-labeled query set against the bundled 14-clip synthetic corpus. These numbers measure the **pipeline architecture** with deterministic stand-in models (hash embedder, Jaccard reranker, rule-based LLM) — they are *not* production accuracy claims; they make the system measurable and the harness reproducible.

| Metric | Score |
| --- | --- |
| Hit@1 | 0.93 |
| Hit@3 | 1.00 |
| Recall@5 | 1.00 |
| MRR | 0.96 |
| Selection accuracy | 0.93 |
| End-to-end latency — p50 / p95 / p99 | **2.4 / 5.9 / 10.7 ms** |

```bash
python evals/retrieval_eval.py   # reproduce
```

## Architecture

A single stateless worker turns inbound social mentions into clip recommendations. A producer (outside this repo) drops one JSON message per mention onto a queue; the worker polls it, runs each mention through a LangGraph state machine, and replies. A message is **acknowledged only after its reply is successfully published**, so transient failures are retried rather than dropped.

```
              ┌──────────────┐      poll        ┌───────────────────────────┐
  X mention ─▶│  Queue (SQS) │ ───────────────▶ │  BotRunner: poll→graph→ack │
              └──────────────┘                  └─────────────┬─────────────┘
                                                              │ LangGraph
                          ┌───────────────────────────────────▼───────────────────────────────┐
                          │ extract_context → analyze ─┬─(small talk)─▶ small_talk ───────────┐ │
                          │                            └─(content)────▶ search_clips           │ │
                          │                                              │                     │ │
                          │                                       generate_reply ──▶ publish ◀─┘ │
                          └──────────────────────────────────────────────┬────────────────────┘
                                                                          │ uses
              ┌───────────────────────────── Retrieval pipeline ─────────▼──────────────────────────────┐
              │ analyze ▶ embed (dense+sparse, query+HyDE) ▶ weight by HyDE sim ▶ per-bucket hybrid search │
              │ ▶ RRF fuse query vectors ▶ RRF fuse buckets ▶ episode/diversity caps ▶ hydrate metadata    │
              │ ▶ duration filter ▶ cross-encoder rerank ▶ signal scoring ▶ relevance floor ▶ LLM selection │
              └───────────────────────────────────────────────────────────────────────────────────────────┘
                  │              │                │                │                │
            Embedder/Sparse  VectorStore     MetadataStore      Reranker        LanguageModel/Vision
            (OpenAI/BM25)    (Pinecone)      (PostgreSQL)       (Cohere)        (Gemini)
```

Everything in the pipeline depends only on the `Protocol` interfaces in [`ports.py`](src/clipopedia/ports.py). Two adapter families implement them: deterministic **in-memory fakes** ([`adapters/memory.py`](src/clipopedia/adapters/memory.py)) that power the demo and tests, and **live adapters** for OpenAI, Pinecone, Cohere, Gemini, PostgreSQL, AWS SQS/S3, and Tweepy. [`factory.py`](src/clipopedia/factory.py) is the single composition root; `CLIPOPEDIA_BACKEND=demo|live` selects the family. The retrieval design is documented in [docs/RAG_PIPELINE.md](docs/RAG_PIPELINE.md) and the component/deployment view in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tech stack

| Layer | Technology |
|---|---|
| Language / runtime | Python 3.11+, asyncio |
| Domain models & config | Pydantic v2, pydantic-settings |
| Orchestration | LangGraph state machine |
| Dense embeddings | OpenAI `text-embedding-3-large` |
| Sparse encoding | BM25 via `pinecone-text` |
| Vector search | Pinecone (single `dotproduct` hybrid index) |
| Reranking | Cohere `rerank-english-v3.0` |
| LLM & vision | Google Gemini (OpenAI-compatible endpoint) |
| Metadata store | PostgreSQL (`psycopg` 3) |
| Ingest queue / media | AWS SQS + S3 (`boto3`) |
| Social client | Tweepy / X API |
| Fuzzy matching | RapidFuzz |
| CLI / output | Rich |
| Tooling | pytest, ruff, mypy |
| Packaging / deploy | Docker, AWS ECS Fargate, GitHub Actions |

## Project structure

```
.
├── src/clipopedia/
│   ├── config.py            # env-driven settings (pydantic-settings)
│   ├── models.py            # provider-agnostic domain models (pydantic)
│   ├── ports.py             # the Protocol interfaces everything depends on
│   ├── factory.py           # composition root: wires demo vs live backends
│   ├── bot.py               # poll → run graph → ack loop
│   ├── cli.py               # `clipopedia demo` / `clipopedia run`
│   ├── jsonparse.py         # tolerant JSON extraction for LLM output
│   ├── dateutils.py         # date ↔ YYYYMMDD helpers
│   ├── textutils.py         # tokenisation for the offline fakes
│   ├── retrieval/           # the RAG pipeline
│   │   ├── pipeline.py      #   orchestrates the whole retrieval flow
│   │   ├── query_analysis.py#   LLM entity/intent/time/HyDE extraction
│   │   ├── hyde.py          #   HyDE similarity weighting
│   │   ├── fusion.py        #   reciprocal rank fusion
│   │   ├── time_planning.py #   query → weighted time buckets
│   │   ├── scoring.py       #   recency boost, diversity caps, signal scoring
│   │   ├── selection.py     #   final LLM clip selection
│   │   └── gazetteer.py     #   fuzzy entity resolution
│   ├── orchestration/       # LangGraph state, nodes, graph
│   ├── adapters/            # memory fakes + real service adapters
│   └── demo/                # synthetic corpus + runnable example
├── tests/                   # unit + end-to-end tests on the in-memory backend
├── docs/                    # ARCHITECTURE.md, RAG_PIPELINE.md
├── deploy/                  # AWS ECS Fargate task-definition template
├── data/                    # example entity catalog (fictional)
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

## Getting started

### Prerequisites

- Python 3.11 or 3.12
- (Live backend only) accounts/keys for OpenAI, Pinecone, Cohere, Gemini, plus PostgreSQL and AWS access

### Quickstart (offline, no API keys)

```bash
git clone https://github.com/akira231097/clipopedia.git
cd clipopedia
pip install -e ".[dev]"      # core + dev tooling; no heavy vendor SDKs needed
python -m clipopedia demo    # runs the full pipeline on a synthetic corpus
```

The demo traces each query through the pipeline — extracted entities, number of HyDE documents, the chosen clip, and the scores behind the choice:

```
──────────── Query: best clip on AI agents and reliability ────────────
cleaned: best clip on AI agents and reliability   small_talk: False   time: none
entities: guests=—  hosts=—  show=—   hyde_docs: 3
  Show          Frontier Notes
  Episode       Dr. Lena Ortiz on ai safety
  Guest         Dr. Lena Ortiz
  Rerank score  0.083
  Final score   0.104          (rerank × recency boost)
  Stage         all
  Why           Top-ranked clip; best lexical and semantic match for the request.
```

Try your own: `python -m clipopedia demo --query "how should I price my startup"`.

### Running against real services

```bash
pip install -e ".[live,dev]"   # adds OpenAI / Pinecone / Cohere / langgraph / boto3 / …
cp .env.example .env           # fill in your own keys; .env is git-ignored
# set CLIPOPEDIA_BACKEND=live in .env
python -m clipopedia run       # polls the queue and replies to mentions
```

See [.env.example](.env.example) for every configuration knob and [deploy/](deploy/) for an AWS ECS Fargate task-definition template (secrets injected from AWS Secrets Manager, never baked into the image).

### Tests & quality

```bash
pytest          # unit + end-to-end tests on the in-memory backend
ruff check .    # lint
mypy src        # types
```

CI runs the suite on Python 3.11 and 3.12 and smoke-tests the demo on every push.

## How it works

The interesting engineering lives in [`retrieval/pipeline.py`](src/clipopedia/retrieval/pipeline.py):

- **HyDE with weighted fusion** — instead of embedding only the terse query, an LLM writes a few *hypothetical* transcript-style answers. Each is scored by cosine similarity to the original-query embedding and assigned a fusion weight that decays from `hyde_weight_max` to `hyde_weight_min`, while the original query always keeps the largest weight. Rich, on-topic texts land closer to real relevant chunks in embedding space.
- **Hybrid search** — every query is embedded both dense (`text-embedding-3-large`) and sparse (BM25). On Pinecone a single `dotproduct` index stores both, and the dense/sparse trade-off is expressed by scaling the vectors (`dense × alpha`, `sparse × (1 − alpha)`) before the query.
- **Double Reciprocal Rank Fusion** — results are fused twice: first across the per-query lists (weighted by the HyDE weights), then across the time buckets (weighted by each bucket's weight). RRF combines by rank, so it is robust to the wildly different score distributions of dense, sparse, and per-HyDE searches.
- **Time-bucket planning** — a query like *"what did guests say about agents lately"* has two competing intents. The time filter is expanded into weighted buckets — a pure-recency bucket ranked by date, a recent-semantic bucket, and a low-weight global backstop — so recency and relevance are served by separate searches.
- **Diversity & safety** — a per-episode cap stops one chatty episode from dominating; a minimum-per-bucket quota keeps the recency bucket represented; and if an entity filter zeroes everything out, an unfiltered safety-recall search runs as a backstop.
- **Rerank then select** — the fused shortlist is re-scored by a Cohere cross-encoder, combined with explainable recency and metadata-agreement boosts, filtered by a relevance floor (never returning fewer than `final_top_k`), and finally handed to an LLM that picks the single best clip with a rationale.
- **Ports & adapters** — the same pipeline code runs against real services or in-memory fakes because it only ever talks to the `Protocol` interfaces. The fakes are non-trivial: a hashing embedder so shared vocabulary lands close in cosine space, real hybrid scoring with metadata filters, a Jaccard reranker, and a rule-based "LLM" that emits valid JSON for the analysis and selection prompts — which is exactly why the demo and tests exercise the real code paths.

Every threshold (`HYBRID_ALPHA`, `RERANK_TOP_N`, `FINAL_TOP_K`, `RECENT_WINDOW_DAYS`, `RECENCY_BOOST`, `RELEVANCE_FLOOR`, `PER_EPISODE_CAP`, the HyDE weights, the fuzzy-match thresholds) lives in [`config.py`](src/clipopedia/config.py) and is environment-overridable.

## Notes / limitations

- This is a **reference implementation**. The demo corpus, entity catalog, and example mentions are entirely fictional and intentionally small — just rich enough to produce sensible, differentiated results.
- The **live backend assumes external infrastructure** you provide: a populated Pinecone hybrid index, a PostgreSQL `clips` table, an SQS queue fed by an upstream mention producer (outside this repo), and an S3 bucket of media clips.
- Vision uses Gemini's OpenAI-compatible endpoint, which accepts still images only; video/GIF media is skipped rather than sent unsupported.
- All configuration is environment-driven via `pydantic-settings`; secrets are never hard-coded — locally from a git-ignored `.env`, and in production from AWS Secrets Manager. The committed `.env.example` and ECS template contain placeholders only.

## License

[MIT](LICENSE) — use it, learn from it, build on it.