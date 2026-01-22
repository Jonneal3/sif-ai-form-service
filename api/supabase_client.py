"""
Supabase hydration helpers (minimal).

This service can optionally enrich incoming requests using Supabase so callers
don't need to send full grounding/context every time.

Scope (intentionally small):
- Fetch `instances` row by `instanceId`
- Fetch instance subcategories with joined category/subcategory labels
- Resolve a selected subcategory UUID into plain-English labels
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from supabase import Client, create_client


def _env(name: str) -> str:
    return str(os.getenv(name) or "").strip()


@lru_cache(maxsize=1)
def get_supabase_client() -> Optional[Client]:
    url = _env("NEXT_PUBLIC_SUPABASE_URL") or _env("SUPABASE_URL")
    key = _env("SUPABASE_SERVICE_ROLE_KEY") or _env("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def _safe_first(data: Any) -> Optional[Dict[str, Any]]:
    if isinstance(data, list) and data:
        first = data[0]
        return first if isinstance(first, dict) else None
    return None


def fetch_instance(instance_id: str) -> Optional[Dict[str, Any]]:
    client = get_supabase_client()
    if not client or not instance_id:
        return None
    try:
        result = client.table("instances").select("*").eq("id", str(instance_id)).execute()
        return _safe_first(getattr(result, "data", None) or result.data)  # type: ignore[attr-defined]
    except Exception:
        return None


def fetch_instance_subcategories(instance_id: str) -> List[Dict[str, Any]]:
    """
    Returns list of:
      { "id": <category_subcategory_id>, "subcategory": <label>, "category_name": <label> }
    """
    client = get_supabase_client()
    if not client or not instance_id:
        return []
    try:
        result = (
            client.table("instance_subcategories")
            .select(
                "category_subcategory_id, categories_subcategories ( id, subcategory, description, slug, categories ( name ) )"
            )
            .eq("instance_id", str(instance_id))
            .execute()
        )
        rows = getattr(result, "data", None) or result.data  # type: ignore[attr-defined]
        if not isinstance(rows, list):
            return []
        out: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            subcat = row.get("categories_subcategories")
            if not isinstance(subcat, dict):
                continue
            category = subcat.get("categories") if isinstance(subcat.get("categories"), dict) else {}
            out.append(
                {
                    "id": row.get("category_subcategory_id") or subcat.get("id"),
                    "subcategory": subcat.get("subcategory"),
                    "description": subcat.get("description"),
                    "slug": subcat.get("slug"),
                    "category_name": category.get("name"),
                }
            )
        return out
    except Exception:
        return []


def resolve_selected_service(
    *,
    selected_subcategory_id: Optional[str],
    instance_subcategories: List[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns: (category_name, subcategory_name, subcategory_id)
    """
    sid = str(selected_subcategory_id or "").strip()
    if not sid:
        return None, None, None
    for row in instance_subcategories:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id") or "").strip()
        if rid and rid == sid:
            category_name = str(row.get("category_name") or "").strip() or None
            subcategory_name = str(row.get("subcategory") or "").strip() or None
            return category_name, subcategory_name, sid
    return None, None, sid

