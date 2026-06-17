# Clip'O'pedia — Design Notes

Clip'O'pedia is a mention-driven, hybrid-RAG assistant: tag it under a social
post with a question and it replies with the single most relevant podcast clip.
This document explains two design choices that make the codebase interesting —
the **hexagonal ports-and-adapters seam** that lets the entire retrieval
pipeline run fully offline and deterministically, and the **double reciprocal
rank fusion** at the core of retrieval.

```mermaid
flowchart TD
    X["X / Twitter mention"] --> Q[("Queue - SQS")]
    Q --> BOT["BotRunner: poll then run graph then ack"]

    subgraph Graph["LangGraph state machine"]
        EC["extract_context"] --> AN["analyze"]
        AN -->|small talk| ST["small_talk"]
        AN -->|content| SR["search_clips"]
        SR --> GR["generate_reply"]
        ST --> PB["publish"]
        GR --> PB
    end

    BOT --> EC
    EC -.->|describe media| VM["VisionModel - Gemini"]
    PB --> XOUT["Reply on X"]

    subgraph Pipeline["RetrievalPipeline"]
        A1["analyze query + HyDE"] --> A2["embed dense + sparse: query + HyDE docs"]
        A2 --> A3["weight by HyDE cosine sim"]
        A3 --> A4["per-bucket hybrid search"]
        A4 --> A5["RRF #1: fuse query vectors"]
        A5 --> A6["RRF #2: fuse time buckets"]
        A6 --> A7["episode cap + bucket quota"]
        A7 --> A8["hydrate metadata + duration filter"]
        A8 --> A9["cross-encoder rerank"]
        A9 --> A10["signal scoring + relevance floor"]
        A10 --> A11["LLM selection"]
    end

    SR -.uses.-> Pipeline

    A2 --> P_EMB["Embedder + SparseEncoder"]
    A4 --> P_VS[("VectorStore")]
    A8 --> P_MD[("MetadataStore")]
    A9 --> P_RK["Reranker"]
    A11 --> P_LM["LanguageModel"]

    subgraph Ports["ports.py - Protocol interfaces"]
        P_EMB
        P_VS
        P_MD
        P_RK
        P_LM
    end

    Ports -.demo backend.-> FAKES["In-memory fakes: FakeEmbedder, InMemoryVectorStore, FakeReranker, RuleBasedLanguageModel"]
    Ports -.live backend.-> LIVE["OpenAI, Pinecone, PostgreSQL, Cohere, Gemini"]
    FAKES --> FACT{{"factory.py composition root"}}
    LIVE --> FACT
    FACT -->|CLIPOPEDIA_BACKEND=demo or live| BOT
```

## Hexagonal architecture: one pipeline, two worlds

**The problem.** A production RAG pipeline talks to half a dozen paid, networked,
non-deterministic services: OpenAI for dense embeddings, Pinecone for hybrid
vector search, Cohere for reranking, Gemini for analysis/selection/vision,
PostgreSQL for clip metadata, and AWS SQS/S3 for ingestion and media. If the
orchestration logic imports those SDKs directly, the project becomes impossible
to run without a wallet full of API keys, and its tests degrade into a pile of
mocks that assert against patched SDK calls rather than real behavior.

**The approach.** Every capability the pipeline needs is expressed as a
`typing.Protocol` in [`ports.py`](../src/clipopedia/ports.py): `Embedder`,
`SparseEncoder`, `VectorStore`, `MetadataStore`, `Reranker`, `LanguageModel`,
`VisionModel`, `MessageSource`, `SocialClient`, `MediaStore`. The retrieval and
orchestration layers depend *only* on these abstract types — there is not a
single `import openai` or `import pinecone` anywhere in `retrieval/` or
`orchestration/`. Two adapter families implement the ports. The live family
([`adapters/`](../src/clipopedia/adapters/)) wraps the real SDKs and bridges
their synchronous calls onto the event loop with `asyncio.to_thread`. The demo
family ([`adapters/memory.py`](../src/clipopedia/adapters/memory.py)) is a set of
deterministic in-process fakes.

[`factory.py`](../src/clipopedia/factory.py) is the single **composition root** —
the only module that imports concrete adapters at all. `build_demo_backend`
wires the fakes; `build_live_backend` wires the real services; `build_backend`
picks between them based on one environment variable, `CLIPOPEDIA_BACKEND`.
Because the wiring is the only thing that changes, swapping the entire backend
from offline fakes to real infrastructure touches zero lines of pipeline code.

**The pay-off — a fully offline, deterministic pipeline.** The fakes are not
trivial stubs that return canned data; they are faithful enough to exercise the
*real* code paths. `FakeEmbedder` hashes tokens into a fixed-dimension vector so
that texts sharing vocabulary land close in cosine space. `InMemoryVectorStore`
does genuine hybrid `dotproduct` scoring with metadata filtering, mirroring how
Pinecone blends `dense × alpha` and `sparse × (1 − alpha)`. `FakeReranker` is a
Jaccard cross-encoder; `RuleBasedLanguageModel` emits valid JSON for both the
analysis and selection prompts. The result: `python -m clipopedia demo` runs the
entire pipeline — analysis, HyDE, hybrid search, double fusion, diversity caps,
rerank, scoring, selection — in seconds with no network and no credentials, and
the test suite uses these same fakes instead of patched SDKs, so tests verify
actual behavior.

**The trade-off.** The fakes are an approximation, not a simulator. They prove
the pipeline is *wired correctly* and that every stage handles real data shapes,
but they cannot reproduce the *quality* of GPT-class embeddings or a true
cross-encoder. Offline determinism guarantees the plumbing, not the relevance
numbers — relevance has to be validated against the live backend.

## Double reciprocal rank fusion

**The problem.** A single query produces many candidate rankings that must be
merged. HyDE means we search with the original query *plus* several hypothetical
answer documents, each yielding its own ranked list. Time-bucket planning means a
"lately" query is split into a pure-recency bucket, a recent-semantic bucket, and
a low-weight global backstop — each, again, its own ranking. These lists have
wildly different and incomparable score distributions: dense cosine, sparse BM25,
and per-HyDE scores do not live on the same scale, so you cannot just add raw
scores.

**The approach.** [`fusion.py`](../src/clipopedia/retrieval/fusion.py) implements
weighted **Reciprocal Rank Fusion**, which combines lists by *rank* rather than
score: `score(item) = Σ_lists weight_list / (k + rank_in_list)`. Because only
ordinal position matters, RRF is immune to mismatched score scales. The pipeline
([`pipeline.py`](../src/clipopedia/retrieval/pipeline.py)) applies it **twice**.
The *inner* fusion merges the per-query-vector lists *within* one time bucket,
weighted by the HyDE weights — so the original query (largest weight) dominates
and each hypothetical contributes in proportion to its cosine similarity to the
original (computed in [`hyde.py`](../src/clipopedia/retrieval/hyde.py)). The
*outer* fusion then merges the per-bucket results, weighted by each bucket's
planned weight, so recency and relevance are reconciled by rank rather than by a
single compromised ranking.

**The trade-off.** RRF deliberately discards score magnitude. Two clips ranked #1
and #2 in a list are treated as one rank apart regardless of whether the gap in
true relevance was a hair or a chasm. That robustness to scale is exactly what we
want for *fusion*, but it would be wrong as a *final* ordering — which is why
fusion only produces the rerank shortlist. The Cohere cross-encoder then re-scores
with full query-passage attention, and explainable multiplicative recency and
metadata-agreement boosts plus a relevance floor produce the order the LLM
selector finally sees. Fuse by rank to be robust; rerank by score to be precise.