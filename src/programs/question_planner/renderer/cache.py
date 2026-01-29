"""
Render-output cache key helpers.
"""

from __future__ import annotations

from typing import List

from programs.common.hashing import short_hash


def render_cache_key(
    *,
    session_id: str,
    schema_version: str,
    plan_json: str,
    render_context_json: str,
    allowed_mini_types: List[str],
) -> str:
    sid = str(session_id or "").strip()
    if not sid:
        return ""
    sv = str(schema_version or "").strip() or "0"
    plan_h = short_hash(plan_json, n=12)
    ctx_h = short_hash(render_context_json, n=12)
    allowed_h = short_hash(
        ",".join(sorted([str(x).strip().lower() for x in (allowed_mini_types or []) if str(x).strip()])),
        n=10,
    )
    return f"render_out:{sid}:{sv}:{plan_h}:{ctx_h}:{allowed_h}"


__all__ = ["render_cache_key"]
