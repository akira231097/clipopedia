"""Graph nodes.

Each node is a small async function over :class:`ConversationState`. They are
built by :func:`make_nodes`, which closes over a wired :class:`Backend`, so the
graph itself stays free of any service knowledge and the nodes are trivially
unit-testable with the in-memory fakes.
"""

from __future__ import annotations

import logging

from ..factory import Backend
from ..models import BotReply, Mention
from ..retrieval.selection import select_best_clip
from .response import craft_reply, craft_small_talk
from .state import ConversationState

logger = logging.getLogger(__name__)


def make_nodes(backend: Backend) -> dict:
    pipeline = backend.pipeline
    vision = backend.vision
    llm = backend.llm
    settings = backend.settings
    social = backend.social_client
    media_store = backend.media_store

    async def extract_context(state: ConversationState) -> dict:
        mention: Mention = state["mention"]
        parts = [mention.text]
        if mention.referenced_text:
            parts.append(mention.referenced_text)
        for item in mention.media:
            try:
                description = await vision.describe(item)
                if description:
                    parts.append(description)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Vision describe failed: %s", exc)
        return {"processed_query": "\n".join(p for p in parts if p)}

    async def analyze(state: ConversationState) -> dict:
        query = state.get("processed_query") or state["mention"].text
        analysis = await pipeline.analyze(query)
        return {"analysis": analysis, "is_small_talk": analysis.is_small_talk}

    async def small_talk(state: ConversationState) -> dict:
        return {"reply": await craft_small_talk(llm, state["mention"])}

    async def search_clips(state: ConversationState) -> dict:
        query = state.get("processed_query") or state["mention"].text
        chunks = await pipeline.retrieve(query, state["analysis"])
        selection = await select_best_clip(llm, query, chunks, settings)
        return {"selection": selection}

    async def generate_reply(state: ConversationState) -> dict:
        selection = state.get("selection")
        if not selection:
            return {
                "reply": BotReply(
                    text="I couldn't find a clip for that yet — try another angle?",
                    in_reply_to=state["mention"].id,
                )
            }
        return {"reply": await craft_reply(llm, state["mention"], selection, settings)}

    async def publish(state: ConversationState) -> dict:
        reply = state.get("reply")
        if not reply or social is None:
            return {"publish_result": None}
        media: bytes | None = None
        if reply.video_ref and media_store is not None:
            media = await media_store.fetch_clip(reply.video_ref)
        return {"publish_result": await social.publish_reply(reply, media)}

    return {
        "extract_context": extract_context,
        "analyze": analyze,
        "small_talk": small_talk,
        "search_clips": search_clips,
        "generate_reply": generate_reply,
        "publish": publish,
    }
