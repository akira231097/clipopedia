"""Inbound mentions from an AWS SQS queue.

A producer (outside this repo) writes a JSON message per mention onto the queue.
This adapter receives one at a time with long polling, parses it into a
:class:`Mention`, and exposes ``ack`` to delete it once handled.
"""

from __future__ import annotations

import asyncio
import json
import logging

import boto3

from ..models import MediaItem, MediaKind, Mention

logger = logging.getLogger(__name__)


def _parse_media(raw: list | None) -> list[MediaItem]:
    items: list[MediaItem] = []
    for entry in raw or []:
        if isinstance(entry, str):
            items.append(MediaItem(url=entry))
        elif isinstance(entry, dict) and entry.get("url"):
            kind = entry.get("type", "image")
            try:
                kind_enum = MediaKind(kind)
            except ValueError:
                kind_enum = MediaKind.image
            items.append(MediaItem(url=entry["url"], kind=kind_enum))
    return items


def _parse_mention(body: dict, bot_handle: str) -> Mention:
    tweet = body.get("tweet", body)
    text = (tweet.get("text") or "").replace(bot_handle, "").strip()
    author = tweet.get("author", {}) or {}
    referenced = None
    for ref in tweet.get("referenced_tweets", []) or []:
        if ref.get("text"):
            referenced = ref["text"]
            break
    return Mention(
        id=str(tweet.get("id", "")),
        text=text,
        author_handle=author.get("username", ""),
        author_name=author.get("name", ""),
        media=_parse_media(tweet.get("media_urls") or tweet.get("media")),
        referenced_text=referenced,
    )


class SqsMessageSource:
    def __init__(self, queue_url: str, region: str, bot_handle: str = "@clipopedia") -> None:
        self._sqs = boto3.client("sqs", region_name=region)
        self.queue_url = queue_url
        self.bot_handle = bot_handle

    async def poll(self) -> Mention | None:
        def _receive() -> dict:
            return self._sqs.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=10,
            )

        resp = await asyncio.to_thread(_receive)
        messages = resp.get("Messages", [])
        if not messages:
            return None
        msg = messages[0]
        try:
            body = json.loads(msg["Body"])
        except (ValueError, KeyError):
            logger.warning("Skipping unparseable SQS message")
            return None
        mention = _parse_mention(body, self.bot_handle)
        mention.ack_token = msg.get("ReceiptHandle")
        return mention

    async def ack(self, mention: Mention) -> None:
        if not mention.ack_token:
            return

        def _delete() -> None:
            self._sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=mention.ack_token)

        await asyncio.to_thread(_delete)
