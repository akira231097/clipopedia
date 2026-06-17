"""The continuous bot loop: poll → run the graph → acknowledge.

Failure handling mirrors a real deployment: a message is only acknowledged
(removed from the queue) once a reply is successfully published, so transient
failures are retried instead of silently dropped.
"""

from __future__ import annotations

import asyncio
import logging

from .factory import Backend
from .models import Mention

logger = logging.getLogger(__name__)


class BotRunner:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.settings = backend.settings
        # Imported here so the demo path never requires langgraph.
        from .orchestration.graph import build_graph

        self.graph = build_graph(backend)

    async def process(self, mention: Mention) -> bool:
        """Run a single mention through the graph. Returns True if published.

        Provider rate limiting is handled per model call by the throttled
        LLM/vision wrappers (see ``adapters/throttle.py``), not here — one
        mention triggers several model calls, so gating the loop is not enough.
        """
        final = await self.graph.ainvoke({"mention": mention})
        result = final.get("publish_result")
        return bool(result and result.success)

    async def run_forever(self) -> None:
        source = self.backend.message_source
        if source is None:
            raise RuntimeError("No message source configured for this backend")

        logger.info("Clip'O'pedia bot started (backend=%s)", self.settings.backend)
        while True:
            try:
                mention = await source.poll()
                if mention is None:
                    await asyncio.sleep(self.settings.poll_interval_seconds)
                    continue
                published = await self.process(mention)
                if published:
                    await source.ack(mention)
                    logger.info("Replied to mention %s", mention.id)
                else:
                    logger.warning("No reply published for %s; leaving on queue", mention.id)
            except asyncio.CancelledError:
                logger.info("Bot loop cancelled; shutting down")
                raise
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                logger.exception("Error in bot loop: %s", exc)
                await asyncio.sleep(self.settings.poll_interval_seconds)
