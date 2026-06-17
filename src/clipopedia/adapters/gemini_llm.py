"""Google Gemini via its OpenAI-compatible endpoint.

Using the OpenAI-compatible surface keeps the dependency footprint small (just
the ``openai`` client) and the call sites identical to any other chat model,
while still running on Gemini under the hood.
"""

from __future__ import annotations

import asyncio

from openai import OpenAI

from ..models import MediaItem, MediaKind

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiLanguageModel:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self._client = OpenAI(api_key=api_key, base_url=_BASE_URL)
        self.model = model

    async def complete(self, *, system: str, user: str, json_mode: bool = False) -> str:
        def _call() -> str:
            kwargs: dict = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

        return await asyncio.to_thread(_call)


class GeminiVisionModel:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self._client = OpenAI(api_key=api_key, base_url=_BASE_URL)
        self.model = model

    async def describe(self, media: MediaItem) -> str:
        # The OpenAI-compatible endpoint only accepts an `image_url` content
        # part, so still images are supported here; video/GIF would need the
        # native Gemini file API and are skipped rather than sent unsupported.
        if media.kind is not MediaKind.image:
            return ""

        def _call() -> str:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this media in one sentence for search context.",
                            },
                            {"type": "image_url", "image_url": {"url": media.url}},
                        ],
                    }
                ],
            )
            return resp.choices[0].message.content or ""

        return await asyncio.to_thread(_call)
