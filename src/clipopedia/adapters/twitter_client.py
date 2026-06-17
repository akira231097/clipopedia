"""Post replies to X (Twitter) via Tweepy.

Errors are classified so the orchestrator can decide whether to retry (rate
limits), back off (forbidden), or drop the message (everything else).
"""

from __future__ import annotations

import asyncio
import io

import tweepy

from ..models import BotReply, PublishResult


class TwitterSocialClient:
    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        access_token: str,
        access_token_secret: str,
        bearer_token: str,
    ) -> None:
        # v2 client posts tweets; v1.1 API is still required for media upload.
        self._client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
        )
        self._api = tweepy.API(
            tweepy.OAuth1UserHandler(
                consumer_key, consumer_secret, access_token, access_token_secret
            )
        )

    def _upload_media(self, reply: BotReply, media: bytes) -> str:
        filename = (reply.video_ref or "clip.mp4").split("/")[-1]
        uploaded = self._api.media_upload(
            filename=filename,
            file=io.BytesIO(media),
            chunked=True,
            media_category="tweet_video",
        )
        return uploaded.media_id

    async def publish_reply(self, reply: BotReply, media: bytes | None = None) -> PublishResult:
        text = reply.text
        if reply.clip_link:
            text = f"{text}\n{reply.clip_link}"

        def _post() -> str:
            media_ids = [self._upload_media(reply, media)] if media else None
            resp = self._client.create_tweet(
                text=text,
                in_reply_to_tweet_id=reply.in_reply_to,
                media_ids=media_ids,
            )
            return str(resp.data["id"])

        try:
            reply_id = await asyncio.to_thread(_post)
            return PublishResult(success=True, reply_id=reply_id)
        except tweepy.TooManyRequests as exc:
            return PublishResult(success=False, error=str(exc), error_type="rate_limit")
        except tweepy.Forbidden as exc:
            return PublishResult(success=False, error=str(exc), error_type="forbidden")
        except Exception as exc:  # noqa: BLE001
            return PublishResult(success=False, error=str(exc), error_type="unknown")
