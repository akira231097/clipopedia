"""Tolerant JSON extraction for LLM output.

Even in "JSON mode", models occasionally wrap output in Markdown fences or add a
sentence of preamble. This helper recovers the first balanced JSON object so the
pipeline degrades gracefully instead of crashing on a stray character.
"""

from __future__ import annotations

import json
from typing import Any


def extract_json(text: str | None) -> dict[str, Any] | None:
    """Return the first JSON object found in ``text``, or ``None``."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip a ```json … ``` fence.
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to scanning for the first balanced { … } span.
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        return None
    return None
