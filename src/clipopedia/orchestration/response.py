"""Turn a selected clip (or small talk) into a short, postable reply."""

from __future__ import annotations

from ..config import Settings
from ..models import BotReply, ClipSelection, Mention
from ..ports import LanguageModel

_MAX_REPLY_CHARS = 240

_REPLY_SYSTEM = (
    "You are a witty, helpful podcast-clip bot. Given a user's request and the clip "
    "we picked, write ONE reply under 240 characters recommending it. Name the show "
    "and guest. No hashtags, no surrounding quotes."
)

_SMALLTALK_SYSTEM = (
    "You are a friendly podcast-clip bot. Reply to this casual message in one short, "
    "warm sentence and invite the person to ask for a clip. No hashtags."
)


def _template_reply(selection: ClipSelection) -> str:
    clip = selection.chunk.clip
    guest = clip.guests[0] if clip.guests else "a great guest"
    return f'Try this: {guest} on "{clip.show_title}" — {clip.episode_title}.'


async def craft_reply(
    llm: LanguageModel, mention: Mention, selection: ClipSelection, settings: Settings
) -> BotReply:
    clip = selection.chunk.clip
    link = clip.audio_url or clip.video_url
    context = (
        f"User asked: {mention.text}\n"
        f"Clip: show='{clip.show_title}', guest='{', '.join(clip.guests)}', "
        f"about='{selection.reason or clip.episode_title}'"
    )
    try:
        text = (await llm.complete(system=_REPLY_SYSTEM, user=context)).strip().strip('"')
    except Exception:  # noqa: BLE001
        text = ""
    if not text:
        text = _template_reply(selection)
    return BotReply(
        text=text[:_MAX_REPLY_CHARS],
        in_reply_to=mention.id,
        clip_link=link,
        video_ref=clip.video_url,
    )


async def craft_small_talk(llm: LanguageModel, mention: Mention) -> BotReply:
    try:
        text = (await llm.complete(system=_SMALLTALK_SYSTEM, user=mention.text)).strip().strip('"')
    except Exception:  # noqa: BLE001
        text = ""
    if not text:
        text = "Hey! Tag me with a topic or a guest and I'll dig up the perfect podcast clip."
    return BotReply(text=text[:_MAX_REPLY_CHARS], in_reply_to=mention.id)
