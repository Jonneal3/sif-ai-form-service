from __future__ import annotations

import json
from typing import Any


def normalize_step_id(step_id: str) -> str:
    """
    Canonicalize step ids to match the Next.js side:
      - underscores -> hyphens
      - preserve leading `step-` prefix
    """
    t = str(step_id or "").strip()
    if not t:
        return t
    return t.replace("_", "-")


def sse(event: str, data: Any) -> str:
    # SSE format reminder:
    #   event: <name>\n
    #   data: <json>\n\n
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_comment(text: str) -> str:
    # SSE "comment" lines start with ":" and are ignored by EventSource clients.
    t = str(text or "").replace("\n", " ").replace("\r", " ")
    return f": {t}\n\n"


def sse_padding(bytes_hint: int = 2048) -> str:
    n = max(0, int(bytes_hint))
    return f": {' ' * n}\n\n"


