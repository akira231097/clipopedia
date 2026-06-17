"""Client-side rate limiting for LLM / vision calls.

Hosted models enforce per-minute quotas. Handling one mention fans out into
several model calls (query analysis → clip selection → reply, plus a vision call
per attached media item), so spacing requests *per mention* is not enough — the
throttle has to sit on the individual model call.

:class:`AsyncMinInterval` enforces a minimum spacing between calls behind a lock
(so concurrent callers serialize), and the ``Throttled*`` wrappers apply it to
any :class:`LanguageModel` / :class:`VisionModel` without the pipeline knowing.
"""

from __future__ import annotations

import asyncio
import time

from ..models import MediaItem
from ..ports import LanguageModel, VisionModel


class AsyncMinInterval:
    """Ensures at least ``min_interval`` seconds elapse between acquisitions."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            delay = self.min_interval - (time.monotonic() - self._last)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


class ThrottledLanguageModel:
    def __init__(self, inner: LanguageModel, limiter: AsyncMinInterval) -> None:
        self._inner = inner
        self._limiter = limiter

    async def complete(self, *, system: str, user: str, json_mode: bool = False) -> str:
        await self._limiter.wait()
        return await self._inner.complete(system=system, user=user, json_mode=json_mode)


class ThrottledVisionModel:
    def __init__(self, inner: VisionModel, limiter: AsyncMinInterval) -> None:
        self._inner = inner
        self._limiter = limiter

    async def describe(self, media: MediaItem) -> str:
        await self._limiter.wait()
        return await self._inner.describe(media)
