from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    t = str(s).strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t, flags=re.IGNORECASE)
    return t.strip()


def _best_effort_parse_json(text: str) -> Any:
    if not text:
        return None
    t = _strip_code_fences(str(text))
    try:
        return json.loads(t)
    except Exception:
        return None


def parse_jsonl_steps(text: Any) -> List[Dict[str, Any]]:
    """
    Parse a JSONL string (one JSON object per line) into a list of dicts.
    Invalid lines are ignored.
    """
    out: List[Dict[str, Any]] = []
    raw = str(text or "")
    if not raw.strip():
        return out
    for line in raw.splitlines():
        t = line.strip()
        if not t:
            continue
        obj = _best_effort_parse_json(t)
        if isinstance(obj, dict):
            out.append(obj)
    return out


__all__ = ["parse_jsonl_steps"]

