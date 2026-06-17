"""LLM-driven query understanding.

A single LLM call turns a noisy mention into a structured :class:`QueryAnalysis`:
cleaned query, small-talk flag, named entities, topics, a temporal filter, and a
handful of HyDE documents. Extracted entities are then reconciled against the
catalog via the :class:`Gazetteer` so downstream metadata filters actually match.

The whole thing is defensive: any parsing failure falls back to a minimal
analysis that still drives a reasonable semantic search.
"""

from __future__ import annotations

import logging

from ..config import Settings
from ..dateutils import parse_iso_date
from ..jsonparse import extract_json
from ..models import QueryAnalysis, TimeFilter, TimeMode
from ..ports import LanguageModel
from .gazetteer import Gazetteer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You analyze a user's request to a podcast-clip recommendation bot.
Return ONLY a JSON object with these fields:
  cleaned_query   (string)  the request rewritten as a clear search query
  is_small_talk   (bool)    true if it is a greeting/chit-chat, not a content request
  intent          (string)  "search" | "smalltalk" | "follow_up"
  complexity      (string)  "simple" | "complex"
  guests          (string[]) people the user wants to hear FROM (interviewees)
  hosts           (string[]) show hosts / creators named
  show            (string|null) a podcast/show name if specified
  topics          (string[]) key topics/themes
  hyde_documents  (string[]) 3-5 short, first-person transcript-style passages that
                             would perfectly answer the request (one per array item)
  time_filter     (object)  {
      has_time_constraint: bool,
      mode: "none"|"latest"|"oldest"|"on"|"before"|"after"|"between"|"relative_recent",
      start_date: "YYYY-MM-DD"|null,
      end_date: "YYYY-MM-DD"|null,
      sort_preference: "latest"|"oldest"|null,
      gating: "soft"|"hard"|null
  }
Do not include any prose outside the JSON object."""


def _coerce_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _parse_time_filter(raw: object) -> TimeFilter:
    if not isinstance(raw, dict) or not raw.get("has_time_constraint"):
        return TimeFilter()
    try:
        mode = TimeMode(str(raw.get("mode", "none")).lower())
    except ValueError:
        mode = TimeMode.none
    return TimeFilter(
        has_time_constraint=True,
        mode=mode,
        start_date=parse_iso_date(raw.get("start_date")),
        end_date=parse_iso_date(raw.get("end_date")),
        anchor_date=parse_iso_date(raw.get("anchor_date")),
        approx_window_days=raw.get("approx_window_days"),
        sort_preference=raw.get("sort_preference"),
        gating=raw.get("gating"),
    )


def _fallback(raw_query: str) -> QueryAnalysis:
    return QueryAnalysis(cleaned_query=raw_query, hyde_documents=[raw_query])


async def analyze_query(
    llm: LanguageModel,
    raw_query: str,
    gazetteer: Gazetteer,
    settings: Settings,
) -> QueryAnalysis:
    """Analyze a raw query and reconcile entities against the catalog."""
    try:
        raw = await llm.complete(system=_SYSTEM_PROMPT, user=raw_query, json_mode=True)
    except Exception as exc:  # noqa: BLE001 - never let analysis crash the bot
        logger.warning("Query analysis LLM call failed (%s); using fallback", exc)
        return _fallback(raw_query)

    data = extract_json(raw)
    if not data:
        logger.warning("Could not parse query analysis JSON; using fallback")
        return _fallback(raw_query)

    hyde = _coerce_str_list(data.get("hyde_documents"))
    analysis = QueryAnalysis(
        cleaned_query=str(data.get("cleaned_query") or raw_query),
        is_small_talk=bool(data.get("is_small_talk", False)),
        intent=str(data.get("intent", "search")),
        complexity=str(data.get("complexity", "simple")),
        guests=_coerce_str_list(data.get("guests")),
        hosts=_coerce_str_list(data.get("hosts")),
        show=(str(data["show"]).strip() if data.get("show") else None),
        topics=_coerce_str_list(data.get("topics")),
        hyde_documents=hyde or [raw_query],
        time_filter=_parse_time_filter(data.get("time_filter")),
    )

    # Reconcile fuzzy surface forms to canonical catalog names.
    analysis.guests = gazetteer.resolve_guests(analysis.guests) or analysis.guests
    analysis.hosts = gazetteer.resolve_hosts(analysis.hosts) or analysis.hosts
    resolved_show = gazetteer.resolve_show(analysis.show)
    if resolved_show:
        analysis.show = resolved_show

    logger.info(
        "Analyzed query: intent=%s guests=%s hosts=%s show=%s hyde=%d time=%s",
        analysis.intent,
        analysis.guests,
        analysis.hosts,
        analysis.show,
        len(analysis.hyde_documents),
        analysis.time_filter.mode.value,
    )
    return analysis
