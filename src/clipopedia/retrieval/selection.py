"""Final clip selection.

The reranker produces a good ordering, but the *best* clip depends on subtler
judgments — does it actually answer the question, is it self-contained, does it
start mid-sentence? We hand the top candidates to an LLM and ask it to pick one
and justify the choice across several axes. If the model is unavailable or
returns nothing usable, we fall back to the top reranked chunk so the bot always
has an answer.
"""

from __future__ import annotations

import logging

from ..config import Settings
from ..jsonparse import extract_json
from ..models import ClipSelection, RetrievedChunk, SelectionScores
from ..ports import LanguageModel

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You select the single best podcast clip to answer a user's request.
You are given the user's query and a numbered list of candidate clips.
Pick the one clip that best answers the request: relevant, self-contained, and
substantive. Return ONLY a JSON object:
  {
    "index": <int>,                  // index of the chosen clip
    "reason": "<one sentence>",      // why it is the best answer
    "completeness_level": "high"|"medium"|"low",
    "scores": {                       // 0.0 - 1.0 each
      "relevance": <float>, "depth": <float>, "completeness": <float>,
      "authority": <float>, "temporal_fit": <float>, "coherence": <float>,
      "query_coverage": <float>, "overall": <float>
    }
  }"""


def _render_candidates(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks):
        clip = c.clip
        guests = ", ".join(clip.guests) or "—"
        published = clip.published_date.isoformat() if clip.published_date else "—"
        snippet = clip.text.replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        lines.append(
            f"[{i}] show='{clip.show_title}' episode='{clip.episode_title}' "
            f"guests='{guests}' published={published}\n    {snippet}"
        )
    return "\n".join(lines)


def _scores_from(raw: object) -> SelectionScores:
    if not isinstance(raw, dict):
        return SelectionScores()
    allowed = SelectionScores.model_fields.keys()
    clean = {}
    for key in allowed:
        try:
            clean[key] = float(raw.get(key, 0.0))
        except (TypeError, ValueError):
            clean[key] = 0.0
    return SelectionScores(**clean)


async def select_best_clip(
    llm: LanguageModel,
    raw_query: str,
    chunks: list[RetrievedChunk],
    settings: Settings,
) -> ClipSelection | None:
    """Choose the single best clip from ``chunks`` (already ranked)."""
    if not chunks:
        return None

    candidates = chunks[: settings.llm_selector_input_k]
    user_prompt = (
        f"User request:\n{raw_query}\n\nCandidate clips:\n{_render_candidates(candidates)}"
    )

    try:
        raw = await llm.complete(system=_SYSTEM_PROMPT, user=user_prompt, json_mode=True)
        data = extract_json(raw) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Clip selection LLM call failed (%s); using top candidate", exc)
        data = {}

    index = data.get("index")
    if not isinstance(index, int) or not (0 <= index < len(candidates)):
        logger.info("No valid LLM selection; falling back to the top reranked clip")
        return ClipSelection(chunk=candidates[0], reason="Top-ranked fallback selection.")

    return ClipSelection(
        chunk=candidates[index],
        reason=str(data.get("reason", "")),
        completeness_level=str(data.get("completeness_level", "")),
        scores=_scores_from(data.get("scores")),
    )
