"""
Form pipeline helper: JSON + id normalization utilities.

Used by the orchestrator (and validation) to:
- normalize step ids into frontend-canonical `step-...` hyphen format
- build stable, compact JSON strings for model inputs / cache keys
"""

from __future__ import annotations

import json
from typing import Any


def normalize_step_id(step_id: str) -> str:
    """
    Canonicalize step ids to match the frontend:
      - underscores -> hyphens
      - preserve leading `step-` prefix
    """
    t = str(step_id or "").strip()
    if not t:
        return t
    return t.replace("_", "-")


def compact_json(obj: Any) -> str:
    """
    Stable, compact JSON used for context strings sent to models.
    """
    try:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=True, sort_keys=True)
    except Exception:
        return json.dumps(str(obj), separators=(",", ":"), ensure_ascii=True)


# Back-compat aliases (older code/scripts use underscore-prefixed helpers).
_normalize_step_id = normalize_step_id
_compact_json = compact_json


__all__ = ["compact_json", "normalize_step_id", "_compact_json", "_normalize_step_id"]

