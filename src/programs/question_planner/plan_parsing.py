from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Set


def normalize_plan_key(raw: Any) -> str:
    t = str(raw or "").strip().lower()
    if not t:
        return ""
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    t = re.sub(r"_+", "_", t)
    return t[:48]


def derive_step_id_from_key(key: str) -> str:
    return f"step-{key.replace('_', '-')}"


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
        pass
    m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def extract_plan_items(text: Any, *, max_items: int, asked_step_ids: Set[str]) -> List[Dict[str, Any]]:
    """
    Parse the planner output into normalized plan items.

    Accepts common shapes:
    - list[dict]
    - { plan: [...] }
    - { question_keys/items: [...] } (legacy-ish)
    """
    parsed = _best_effort_parse_json(str(text or ""))
    if isinstance(parsed, list):
        raw_items = parsed
    elif isinstance(parsed, dict):
        raw_items = parsed.get("plan")
        if not isinstance(raw_items, list):
            raw_items = parsed.get("question_keys")
        if not isinstance(raw_items, list):
            raw_items = parsed.get("items")
        if not isinstance(raw_items, list):
            raw_items = []
    else:
        raw_items = []

    out: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        key = normalize_plan_key(item.get("key"))
        if not key or key in seen_keys:
            continue
        step_id = derive_step_id_from_key(key)
        if step_id in asked_step_ids:
            continue
        seen_keys.add(key)
        normalized = dict(item)
        normalized["key"] = key
        out.append(normalized)
        if len(out) >= int(max_items):
            break
    return out


__all__ = ["normalize_plan_key", "derive_step_id_from_key", "extract_plan_items"]

