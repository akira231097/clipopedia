"""Clip'O'pedia — interactive offline demo (Gradio).

Runs the REAL retrieval pipeline against the bundled synthetic corpus using the
deterministic in-memory backend: no API keys, no network. Ask a question and it
finds the single most relevant podcast clip.

Local:  pip install -r requirements.txt gradio && python app.py
Deploy: Hugging Face Spaces (SDK: Gradio) — this file is the entry point.
"""

from __future__ import annotations

import asyncio

import gradio as gr

from clipopedia.config import get_settings
from clipopedia.factory import build_demo_backend

# Build the offline backend once at startup.
_backend = asyncio.run(build_demo_backend(get_settings()))

EXAMPLES = [
    "best clip on AI agents and reliable tool use",
    "anything recent on founder burnout",
    "how should I think about startup pricing",
    "the risk of autonomous agents at scale",
    "tips for deep work and protecting focus",
    "hey there!",
]


async def _answer(question: str) -> str:
    analysis, selection = await _backend.pipeline.run(question)
    if selection is None:
        return (
            "**No clip selected.** This was interpreted as small talk or had no "
            f"confident match.\n\n_cleaned query:_ `{analysis.cleaned_query}`"
        )
    clip = selection.chunk.clip
    rerank = selection.chunk.rerank_score
    return (
        f"### 🎧 {clip.show_title} — *{clip.episode_title}*\n"
        f"**Guest:** {', '.join(clip.guests) or '—'}  \n"
        f"**Published:** {clip.published_date.isoformat() if clip.published_date else '—'}  \n"
        f"**Scores:** rerank `{rerank:.3f}`  ·  final `{selection.chunk.final_score:.3f}`  ·  stage `{selection.chunk.retrieval_stage}`  \n\n"
        f"> {clip.text}\n\n"
        f"**Why this clip:** {selection.reason or '—'}"
    )


def answer(question: str) -> str:
    if not question or not question.strip():
        return "_Ask something about podcasts — try one of the examples below._"
    return asyncio.run(_answer(question.strip()))


with gr.Blocks(title="Clip'O'pedia") as demo:
    gr.Markdown(
        "# 🎧 Clip'O'pedia\n"
        "A mention-driven **hybrid-RAG** assistant that finds the single best podcast clip "
        "for your question — query analysis → HyDE → dense+sparse search → double RRF → "
        "rerank → scoring → LLM selection.\n\n"
        "_This demo runs fully offline on a small synthetic corpus with deterministic "
        "stand-in models (no API keys)._"
    )
    inp = gr.Textbox(label="Ask about podcasts", placeholder="e.g. best clip on AI agents and reliability")
    out = gr.Markdown()
    btn = gr.Button("Find the clip", variant="primary")
    btn.click(answer, inputs=inp, outputs=out)
    inp.submit(answer, inputs=inp, outputs=out)
    gr.Examples(EXAMPLES, inputs=inp)


if __name__ == "__main__":
    demo.launch()
