"""In-memory, deterministic adapters for the offline demo and tests.

These implement the same ports as the real services, so the *entire* retrieval
pipeline can run with no network and no API keys. The fakes are intentionally
simple but not trivial:

* :class:`FakeEmbedder` hashes tokens into a fixed-width vector — texts that
  share words land close together in cosine space, so search is meaningful.
* :class:`FakeSparseEncoder` builds a bag-of-words sparse vector.
* :class:`InMemoryVectorStore` does real hybrid (dense + sparse) scoring and
  honours metadata filters.
* :class:`RuleBasedLanguageModel` returns valid JSON for the analysis and
  selection prompts using lightweight heuristics — no model required.
"""

from __future__ import annotations

import logging
import math

from ..dateutils import to_numeric
from ..models import (
    BotReply,
    Clip,
    MediaItem,
    Mention,
    PublishResult,
    SparseVector,
    VectorMatch,
)
from ..textutils import stable_hash, tokenize

_DENSE_DIM = 256
_SPARSE_SPACE = 1 << 20


# --------------------------------------------------------------------------- #
# Embedding / encoding
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    """Deterministic hashing embedder. Shared vocabulary → similar vectors."""

    def __init__(self, dimension: int = _DENSE_DIM) -> None:
        self.dimension = dimension

    def _vectorize(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for tok in tokenize(text, drop_stopwords=True):
            vec[stable_hash(tok, self.dimension)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]


class FakeSparseEncoder:
    """Deterministic bag-of-words sparse encoder (term frequencies)."""

    async def encode(self, texts: list[str]) -> list[SparseVector]:
        out: list[SparseVector] = []
        for text in texts:
            counts: dict[int, float] = {}
            for tok in tokenize(text, drop_stopwords=True):
                idx = stable_hash(tok, _SPARSE_SPACE)
                counts[idx] = counts.get(idx, 0.0) + 1.0
            out.append(SparseVector(indices=list(counts), values=list(counts.values())))
        return out


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _sparse_cosine(a: SparseVector, b: SparseVector) -> float:
    da = dict(zip(a.indices, a.values, strict=False))
    db = dict(zip(b.indices, b.values, strict=False))
    dot = sum(val * db.get(idx, 0.0) for idx, val in da.items())
    na = math.sqrt(sum(v * v for v in da.values()))
    nb = math.sqrt(sum(v * v for v in db.values()))
    return dot / (na * nb) if na and nb else 0.0


def _passes_filter(metadata: dict, flt: dict | None) -> bool:
    if not flt:
        return True
    if "guests" in flt:
        want = {g.lower() for g in flt["guests"]}
        have = {g.lower() for g in metadata.get("guests", [])}
        if not want & have:
            return False
    if "hosts" in flt:
        want = {h.lower() for h in flt["hosts"]}
        have = {h.lower() for h in metadata.get("hosts", [])}
        if not want & have:
            return False
    if "show" in flt and metadata.get("show", "").lower() != str(flt["show"]).lower():
        return False
    if "pdnumeric" in flt:
        clause = flt["pdnumeric"]
        pd = metadata.get("pdnumeric")
        if pd is None:
            return False
        if "$gte" in clause and pd < clause["$gte"]:
            return False
        if "$lte" in clause and pd > clause["$lte"]:
            return False
    return True


class InMemoryVectorStore:
    """Hybrid vector index backed by Python lists."""

    def __init__(self, alpha: float = 0.7) -> None:
        self.alpha = alpha
        self._dense: dict[str, list[float]] = {}
        self._sparse: dict[str, SparseVector] = {}
        self._metadata: dict[str, dict] = {}

    def add(self, chunk_id: str, dense: list[float], sparse: SparseVector, metadata: dict) -> None:
        self._dense[chunk_id] = dense
        self._sparse[chunk_id] = sparse
        self._metadata[chunk_id] = metadata

    @classmethod
    async def from_clips(
        cls,
        clips: list[Clip],
        embedder: FakeEmbedder,
        sparse_encoder: FakeSparseEncoder,
        *,
        alpha: float = 0.7,
    ) -> InMemoryVectorStore:
        store = cls(alpha=alpha)
        texts = [_clip_index_text(c) for c in clips]
        dense = await embedder.embed(texts)
        sparse = await sparse_encoder.encode(texts)
        for clip, d, s in zip(clips, dense, sparse, strict=True):
            store.add(clip.chunk_id, d, s, _clip_metadata(clip))
        return store

    async def hybrid_query(
        self,
        *,
        dense: list[float],
        sparse: SparseVector | None,
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> list[VectorMatch]:
        scored: list[VectorMatch] = []
        for cid, meta in self._metadata.items():
            if not _passes_filter(meta, metadata_filter):
                continue
            d_score = _cosine(dense, self._dense[cid])
            s_score = _sparse_cosine(sparse, self._sparse[cid]) if sparse else 0.0
            hybrid = self.alpha * d_score + (1 - self.alpha) * s_score
            if hybrid <= 0:
                continue
            scored.append(VectorMatch(chunk_id=cid, score=hybrid, metadata=dict(meta)))
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]


def _clip_index_text(clip: Clip) -> str:
    parts = [clip.text, clip.episode_title, clip.show_title]
    parts += clip.guests + clip.hosts + clip.topics
    return " ".join(p for p in parts if p)


def _clip_metadata(clip: Clip) -> dict:
    return {
        "episode_id": clip.episode_id,
        "guests": clip.guests,
        "hosts": clip.hosts,
        "show": clip.show_title,
        "pdnumeric": to_numeric(clip.published_date) if clip.published_date else 0,
    }


# --------------------------------------------------------------------------- #
# Metadata store
# --------------------------------------------------------------------------- #
class InMemoryMetadataStore:
    def __init__(self, clips: list[Clip]) -> None:
        self._by_id = {c.chunk_id: c for c in clips}

    async def fetch(self, chunk_ids: list[str]) -> dict[str, Clip]:
        return {cid: self._by_id[cid] for cid in chunk_ids if cid in self._by_id}


# --------------------------------------------------------------------------- #
# Reranker
# --------------------------------------------------------------------------- #
class FakeReranker:
    """Lexical-overlap reranker (Jaccard over content tokens)."""

    async def rerank(
        self, *, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        q = set(tokenize(query, drop_stopwords=True))
        scored: list[tuple[int, float]] = []
        for i, doc in enumerate(documents):
            d = set(tokenize(doc, drop_stopwords=True))
            score = len(q & d) / len(q | d) if q and d else 0.0
            scored.append((i, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_n]


# --------------------------------------------------------------------------- #
# Language / vision models
# --------------------------------------------------------------------------- #
_GREETINGS = {"hi", "hello", "hey", "yo", "sup", "thanks", "thank", "gm", "lol", "haha"}


class RuleBasedLanguageModel:
    """A no-network stand-in that returns valid JSON for the pipeline prompts.

    It inspects the system prompt to decide which task it is being asked to do
    (query analysis vs. clip selection) and responds with heuristics.
    """

    async def complete(self, *, system: str, user: str, json_mode: bool = False) -> str:
        if system.startswith("You select"):
            return self._select(user)
        if system.startswith("You analyze"):
            return self._analyze(user)
        # Reply / small-talk generation: return empty so callers use their
        # deterministic template fallbacks (keeps the demo output stable).
        return ""

    def _analyze(self, user: str) -> str:
        import json

        tokens = tokenize(user, drop_stopwords=True)
        is_small_talk = (
            len(tokens) <= 2 and any(t in _GREETINGS for t in tokenize(user))
        )
        lowered = user.lower()
        wants_latest = any(w in lowered for w in ("latest", "recent", "lately", "new"))
        topics = list(dict.fromkeys(tokens))[:5]
        hyde = [
            user,
            "In this episode the guest explains " + " ".join(topics[:4]),
            "A practical discussion about " + " ".join(topics[:4]),
        ]
        payload = {
            "cleaned_query": user.strip(),
            "is_small_talk": is_small_talk,
            "intent": "smalltalk" if is_small_talk else "search",
            "complexity": "complex" if len(tokens) > 12 else "simple",
            "guests": [],
            "hosts": [],
            "show": None,
            "topics": topics,
            "hyde_documents": hyde,
            "time_filter": {
                "has_time_constraint": wants_latest,
                "mode": "latest" if wants_latest else "none",
                "start_date": None,
                "end_date": None,
                "sort_preference": "latest" if wants_latest else None,
                "gating": "soft",
            },
        }
        return json.dumps(payload)

    def _select(self, user: str) -> str:
        import json

        # The candidates are already ranked, so pick the top one and explain.
        return json.dumps(
            {
                "index": 0,
                "reason": "Top-ranked clip; best lexical and semantic match for the request.",
                "completeness_level": "high",
                "scores": {
                    "relevance": 0.9,
                    "depth": 0.8,
                    "completeness": 0.85,
                    "authority": 0.8,
                    "temporal_fit": 0.7,
                    "coherence": 0.85,
                    "query_coverage": 0.88,
                    "overall": 0.85,
                },
            }
        )


class EchoVisionModel:
    async def describe(self, media: MediaItem) -> str:
        return f"[demo vision] a {media.kind.value} relevant to the user's request"


# --------------------------------------------------------------------------- #
# Messaging / publishing
# --------------------------------------------------------------------------- #
class InMemoryMessageSource:
    """A seedable queue of mentions for running the bot loop offline."""

    def __init__(self, mentions: list[Mention] | None = None) -> None:
        self._pending = list(mentions or [])
        self.acked: list[str] = []

    async def poll(self) -> Mention | None:
        return self._pending.pop(0) if self._pending else None

    async def ack(self, mention: Mention) -> None:
        self.acked.append(mention.id)


class ConsoleSocialClient:
    """Prints the reply instead of posting it. Records what it 'published'."""

    def __init__(self) -> None:
        self.published: list[BotReply] = []

    async def publish_reply(self, reply: BotReply, media: bytes | None = None) -> PublishResult:
        self.published.append(reply)
        if media:
            logging.getLogger(__name__).info("Would attach %d bytes of media", len(media))
        return PublishResult(success=True, reply_id=f"demo-{len(self.published)}")


class NullMediaStore:
    async def fetch_clip(self, ref: str) -> bytes | None:
        return None
