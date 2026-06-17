# The retrieval pipeline

This is a deep dive into [`retrieval/pipeline.py`](../src/clipopedia/retrieval/pipeline.py)
and the modules it orchestrates. The goal of the pipeline is to take a short,
noisy question and return the single most relevant podcast clip — balancing
semantic relevance, lexical precision, recency, and diversity.

## 1. Query analysis
[`query_analysis.py`](../src/clipopedia/retrieval/query_analysis.py)

One LLM call turns the raw mention into a structured `QueryAnalysis`:

- a cleaned query and a small-talk flag,
- named **guests / hosts / show** and **topics**,
- a **time filter** (`latest`, `before`, `between`, …), and
- a handful of **HyDE documents** (see below).

Extracted entity surface forms ("lena ortiz", "the long game show") are then
reconciled to canonical catalog names by the **gazetteer**
([`gazetteer.py`](../src/clipopedia/retrieval/gazetteer.py)) using RapidFuzz —
combining `WRatio` with `token_set_ratio` so reordered names and dropped
honorifics still match. Without this step, a metadata filter built from the
LLM's spelling would silently match nothing.

The whole stage is defensive: any parse failure falls back to a minimal analysis
that still drives a reasonable semantic search.

## 2. HyDE: embed hypothetical answers
[`hyde.py`](../src/clipopedia/retrieval/hyde.py)

A terse query like *"AI agents reliability?"* embeds poorly — real relevant
transcript passages are long and discursive. **HyDE** (Hypothetical Document
Embeddings) asks the LLM to write a few passages that *would* answer the question
and embeds those too.

Not every hypothetical is equally good, so each is scored by cosine similarity to
the original query embedding and assigned a fusion weight that decays from
`hyde_weight_max` (most on-topic) down to `hyde_weight_min`. The original query
always keeps the largest weight (`original_query_weight`). These weights feed
directly into rank fusion in step 5.

## 3. Dense + sparse embeddings
[`adapters`](../src/clipopedia/adapters/)

Every query (original + HyDE docs) is embedded two ways, concurrently:

- **Dense** — `text-embedding-3-large` → semantic similarity.
- **Sparse** — BM25 → exact lexical matches (names, jargon, acronyms).

Dense catches *meaning*; sparse catches *the exact word the user typed*. Hybrid
search needs both.

## 4. Time-bucket planning
[`time_planning.py`](../src/clipopedia/retrieval/time_planning.py)

A query like *"what did guests say about agents **lately**"* has two competing
intents: relevance and recency. Serving both from one search degrades both.
Instead the time filter is expanded into weighted **buckets**, each a separate
filtered search:

- `latest` → a pure-recency bucket (ranked by date), a recent-semantic bucket,
  and a low-weight global backstop.
- explicit ranges (`between`, `before`, `after`) → a date-clamped bucket, plus a
  soft backstop unless the request is strict.
- no constraint → a single global bucket.

## 5. Hybrid search + double rank fusion
[`fusion.py`](../src/clipopedia/retrieval/fusion.py)

For each bucket, every query vector runs a hybrid (dense + sparse) search,
weighted by `hybrid_alpha`. Results are fused **twice** with
[Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormack/cormacksigir09-rrf.pdf):

1. across the per-query result lists (weighted by the HyDE weights from step 2),
2. across the buckets (weighted by each bucket's weight from step 4).

RRF combines lists by *rank* rather than raw score, so it's robust to the wildly
different score distributions of dense, sparse, and per-HyDE searches.

## 6. Diversity caps
[`scoring.py`](../src/clipopedia/retrieval/scoring.py)

A per-episode cap stops a single chatty episode from filling the results, while a
minimum-per-bucket quota guarantees the recency bucket keeps representation even
when semantic scores favour older clips. If an entity filter zeroed everything
out, an unfiltered **safety-recall** search runs as a backstop.

## 7. Metadata hydration
[`adapters`](../src/clipopedia/adapters/)

The vector store holds ids + light metadata. The shortlist of ids is hydrated
into full `Clip` records (transcript text, titles, timestamps, media URLs) from
the metadata store. Over-long segments (≥ 10 min) are dropped — a "clip" should
be shareable.

## 8. Cross-encoder reranking

The fused shortlist is re-scored by a cross-encoder reranker
(`rerank-english-v3.0`), which reads each query↔passage pair jointly with full
attention. This is far more accurate than the bi-encoder vector scores, but too
expensive to run over the whole corpus — hence "retrieve wide, rerank narrow".

## 9. Signal scoring
[`scoring.py`](../src/clipopedia/retrieval/scoring.py)

The reranker score is combined with explainable, multiplicative signals:

- **recency** — clips inside the recent window get a `recency_boost`,
- **metadata agreement** — a clip whose guest/host/show matches the query gets a
  boost.

A **relevance floor** then drops weak matches, while always keeping at least
`final_top_k` so the bot is never empty-handed.

## 10. LLM selection
[`selection.py`](../src/clipopedia/retrieval/selection.py)

Finally, the top candidates go to an LLM that picks the single best clip and
scores it across relevance, depth, completeness, authority, temporal fit,
coherence, and query coverage — returning a short rationale. If the model is
unavailable or returns nothing usable, the top reranked clip is used.

## Tuning

Every threshold lives in [`config.py`](../src/clipopedia/config.py) and is
environment-overridable: `HYBRID_ALPHA`, `RERANK_TOP_N`, `FINAL_TOP_K`,
`RECENT_WINDOW_DAYS`, `RECENCY_BOOST`, `RELEVANCE_FLOOR`, `PER_EPISODE_CAP`, the
HyDE weights, and the fuzzy-match thresholds.
