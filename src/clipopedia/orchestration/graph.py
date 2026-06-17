"""Assemble the LangGraph state machine.

    extract_context ─► analyze ─┬─(small talk)─► small_talk ─────────┐
                                └─(content)────► search_clips ─► generate_reply ─► publish ─► END

``langgraph`` is only imported here, so the offline demo (which calls the
retrieval pipeline directly) never needs it installed.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from ..factory import Backend
from .nodes import make_nodes
from .state import ConversationState


def _route_after_analyze(state: ConversationState) -> str:
    return "small_talk" if state.get("is_small_talk") else "search_clips"


def build_graph(backend: Backend):
    """Build and compile the conversation graph for a wired backend."""
    nodes = make_nodes(backend)
    graph = StateGraph(ConversationState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point("extract_context")
    graph.add_edge("extract_context", "analyze")
    graph.add_conditional_edges(
        "analyze",
        _route_after_analyze,
        {"small_talk": "small_talk", "search_clips": "search_clips"},
    )
    graph.add_edge("small_talk", "publish")
    graph.add_edge("search_clips", "generate_reply")
    graph.add_edge("generate_reply", "publish")
    graph.add_edge("publish", END)
    return graph.compile()
