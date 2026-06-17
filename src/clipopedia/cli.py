"""Command-line entry point: ``clipopedia demo`` and ``clipopedia run``."""

from __future__ import annotations

import argparse
import asyncio

from . import __version__
from .config import get_settings
from .logging_setup import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipopedia",
        description="A mention-driven hybrid-RAG assistant that recommends podcast clips.",
    )
    parser.add_argument("--version", action="version", version=f"clipopedia {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run the offline retrieval demo (no API keys).")
    demo.add_argument("--query", "-q", default=None, help="A single query to run.")

    sub.add_parser("run", help="Run the live bot loop (set CLIPOPEDIA_BACKEND=live).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level)

    if args.command == "demo":
        from .demo.run import run_demo

        asyncio.run(run_demo(args.query))
        return 0

    if args.command == "run":
        from .bot import BotRunner
        from .factory import build_backend

        async def _go() -> None:
            backend = await build_backend(settings)
            await BotRunner(backend).run_forever()

        asyncio.run(_go())
        return 0

    return 1
