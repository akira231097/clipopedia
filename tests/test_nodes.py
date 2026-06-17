"""Exercise the graph nodes directly (no langgraph dependency required)."""

from clipopedia.config import get_settings
from clipopedia.factory import build_demo_backend
from clipopedia.models import Mention
from clipopedia.orchestration.nodes import make_nodes


async def test_search_path_publishes_reply():
    backend = await build_demo_backend(get_settings(refresh=True))
    nodes = make_nodes(backend)
    state: dict = {"mention": Mention(id="42", text="@clipopedia best clip on AI agents")}

    state.update(await nodes["extract_context"](state))
    state.update(await nodes["analyze"](state))
    assert state["is_small_talk"] is False

    state.update(await nodes["search_clips"](state))
    assert state["selection"] is not None

    state.update(await nodes["generate_reply"](state))
    assert state["reply"].text
    assert len(state["reply"].text) <= 240

    state.update(await nodes["publish"](state))
    assert state["publish_result"].success is True
    assert backend.social_client.published  # ConsoleSocialClient recorded it


async def test_small_talk_path():
    backend = await build_demo_backend(get_settings(refresh=True))
    nodes = make_nodes(backend)
    state: dict = {"mention": Mention(id="7", text="hey gm")}

    state.update(await nodes["extract_context"](state))
    state.update(await nodes["analyze"](state))
    assert state["is_small_talk"] is True

    state.update(await nodes["small_talk"](state))
    assert "clip" in state["reply"].text.lower()
