"""The shared state object threaded through the graph.

Each node returns a partial dict; LangGraph merges it into the running state.
``total=False`` lets nodes populate only the keys they own.
"""

from __future__ import annotations

from typing import TypedDict

from ..models import BotReply, ClipSelection, Mention, PublishResult, QueryAnalysis


class ConversationState(TypedDict, total=False):
    mention: Mention
    processed_query: str
    analysis: QueryAnalysis
    is_small_talk: bool
    selection: ClipSelection | None
    reply: BotReply | None
    publish_result: PublishResult | None
    error: str
