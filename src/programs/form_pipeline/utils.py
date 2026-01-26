from __future__ import annotations

import json
from typing import Any


def _normalize_step_id(step_id: str) -> str:
    """
    Canonicalize step ids to match the frontend:
      - underscores -> hyphens
      - preserve leading `step-` prefix
    """

    t = str(step_id or "").strip()
    if not t:
        return t
    return t.replace("_", "-")


def _compact_json(obj: Any) -> str:
    """
    Stable, compact JSON used for `context_json` sent to the model.
    """

    try:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=True, sort_keys=True)
    except Exception:
        return json.dumps(str(obj), separators=(",", ":"), ensure_ascii=True)


__all__ = ["_normalize_step_id", "_compact_json"]

