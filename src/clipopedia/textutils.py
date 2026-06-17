"""Tiny tokenisation helpers used by the offline (fake) adapters."""

from __future__ import annotations

import hashlib
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A short stop-word list keeps the demo's lexical scoring focused on content
# words without pulling in a heavyweight NLP dependency.
_STOPWORDS = frozenset({
    "a", "an", "and", "the", "of", "to", "in", "on", "for", "with", "about",
    "what", "who", "whom", "how", "why", "when", "where", "is", "are", "was",
    "were", "do", "does", "did", "this", "that", "these", "those", "it", "its",
    "as", "at", "by", "from", "i", "you", "he", "she", "they", "we", "me", "my",
    "your", "our", "their", "said", "say", "says", "talk", "talks", "talked",
    "clip", "clips", "episode", "podcast", "show",
})


def tokenize(text: str, *, drop_stopwords: bool = False) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


def stable_hash(token: str, modulo: int) -> int:
    """Deterministic, process-independent hash (unlike the salted builtin)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo
