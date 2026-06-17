"""Fuzzy entity resolution.

LLM entity extraction is fuzzy: it will return "lena ortiz", "Dr. Ortiz", or
"the long game show" for entities that are canonically spelled differently in
our catalog ("Dr. Lena Ortiz", "The Long Game"). The gazetteer reconciles
extracted surface forms against known guests, hosts, and shows using RapidFuzz,
so a metadata filter built from them actually matches what is stored.

Two signals are combined:

* ``WRatio``        — robust general string similarity.
* ``token_set_ratio`` — order-independent overlap, good for reordered names
                        and dropped honorifics ("Dr.", "the … show").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz, utils


@dataclass
class Gazetteer:
    """Known canonical entity names to resolve extracted surface forms against."""

    guests: list[str] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)
    shows: list[str] = field(default_factory=list)
    wratio_threshold: int = 85
    token_set_min_score: int = 75

    @classmethod
    def from_json(cls, path: str | Path, **kwargs) -> Gazetteer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            guests=data.get("guests", []),
            hosts=data.get("hosts", []),
            shows=data.get("shows", []),
            **kwargs,
        )

    def _best_match(self, surface: str, choices: list[str]) -> str | None:
        if not surface or not choices:
            return None
        best_name: str | None = None
        best_score = 0.0
        # default_process lowercases, trims, and strips non-alphanumerics, so
        # "lena ortiz" matches "Dr. Lena Ortiz" and "the long game show" matches
        # "The Long Game".
        for choice in choices:
            score = max(
                fuzz.WRatio(surface, choice, processor=utils.default_process),
                fuzz.token_set_ratio(surface, choice, processor=utils.default_process),
            )
            if score > best_score:
                best_score, best_name = score, choice
        if best_name is None:
            return None
        # Accept on either a strong overall ratio or strong token overlap.
        if (
            fuzz.WRatio(surface, best_name, processor=utils.default_process)
            >= self.wratio_threshold
            or fuzz.token_set_ratio(surface, best_name, processor=utils.default_process)
            >= self.token_set_min_score
        ):
            return best_name
        return None

    def _resolve_many(self, surfaces: list[str], choices: list[str]) -> list[str]:
        resolved: list[str] = []
        for surface in surfaces:
            match = self._best_match(surface, choices)
            if match and match not in resolved:
                resolved.append(match)
        return resolved

    def resolve_guests(self, surfaces: list[str]) -> list[str]:
        return self._resolve_many(surfaces, self.guests)

    def resolve_hosts(self, surfaces: list[str]) -> list[str]:
        return self._resolve_many(surfaces, self.hosts)

    def resolve_show(self, surface: str | None) -> str | None:
        return self._best_match(surface or "", self.shows)
