from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def is_first_batch(payload: Dict[str, Any]) -> bool:
    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}
    batch_number = current_batch.get("batchNumber") or current_batch.get("batch_number")
    if batch_number is not None:
        try:
            return int(batch_number) <= 1
        except Exception:
            pass

    form_state = payload.get("formState") if isinstance(payload.get("formState"), dict) else {}
    batch_index = (
        form_state.get("batchIndex")
        or form_state.get("batch_index")
        or form_state.get("batchNumber")
        or form_state.get("batch_number")
    )
    if batch_index is not None:
        try:
            return int(batch_index) <= 1
        except Exception:
            pass

    batch_id = str(payload.get("batchId") or payload.get("batch_id") or "").strip()
    return batch_id == "ContextCore"


def normalize_plan_key(raw: Any) -> str:
    t = str(raw or "").strip()
    if not t:
        return ""
    if t.startswith("step-"):
        t = t[len("step-") :]
    t = t.replace("-", "_")
    t = re.sub(r"[^a-zA-Z0-9_]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t.lower()


def step_id_from_key(key: str) -> str:
    t = normalize_plan_key(key)
    if not t:
        return ""
    return f"step-{t.replace('_', '-')}"


def parse_produced_form_plan_json(text: Any) -> List[Dict[str, Any]]:
    from flow_planner import _best_effort_parse_json  # local import to avoid cycles at import-time

    raw = str(text or "").strip()
    if not raw:
        return []
    obj = _best_effort_parse_json(raw)
    if not isinstance(obj, list):
        return []

    try:
        from app.schemas.ui_steps import FormPlanItem
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        item_key = normalize_plan_key(item.get("key"))
        if not item_key:
            continue
        normalized = dict(item)
        normalized["key"] = item_key
        try:
            FormPlanItem.model_validate(normalized)
        except Exception:
            continue
        out.append(normalized)
    return out


def deterministic_upload_plan_items(required_uploads: Any) -> List[Dict[str, Any]]:
    uploads = required_uploads if isinstance(required_uploads, list) else []
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for up in uploads:
        if not isinstance(up, dict):
            continue
        step_id = str(up.get("stepId") or up.get("step_id") or up.get("id") or "").strip()
        if not step_id:
            continue
        key = normalize_plan_key(step_id)
        if not key or key in seen:
            continue
        seen.add(key)
        role = str(up.get("role") or "").strip() or None
        item: Dict[str, Any] = {
            "key": key,
            "goal": "Upload a required reference",
            "why": "Required input for the experience",
            "component_hint": "file_upload",
            "priority": "critical",
            "importance_weight": 0.15,
            "expected_metric_gain": 0.05,
            "deterministic": True,
        }
        if role:
            item["role"] = role
        items.append(item)
    return items


def fallback_attribute_family_plan(attribute_families: Any, limit: int = 8) -> List[Dict[str, Any]]:
    families = attribute_families if isinstance(attribute_families, list) else []
    out: List[Dict[str, Any]] = []
    for fam in families:
        if not isinstance(fam, dict):
            continue
        family = normalize_plan_key(fam.get("family"))
        if not family:
            continue
        goal = str(fam.get("goal") or "").strip() or f"Capture {family.replace('_', ' ')}"
        out.append(
            {
                "key": family,
                "goal": goal[:120],
                "why": "Improves estimate and/or output quality",
                "component_hint": "choice",
                "priority": "medium",
                "importance_weight": 0.1,
                "expected_metric_gain": 0.06,
                "deterministic": True,
            }
        )
        if len(out) >= limit:
            break
    return out


def finalize_form_plan(
    *,
    payload: Dict[str, Any],
    context: Dict[str, Any],
    produced_form_plan_json: Any,
    max_items: int = 12,
) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    """
    Returns (final_plan, did_generate_or_patch).

    - If a plan already exists in context, returns (None, False).
    - If not first batch, returns (None, False).
    - Otherwise returns a merged plan (deterministic uploads + produced/fallback).
    """
    existing = context.get("form_plan")
    if isinstance(existing, list) and existing:
        return None, False
    if not is_first_batch(payload):
        return None, False

    deterministic = deterministic_upload_plan_items(context.get("required_uploads"))
    produced = parse_produced_form_plan_json(produced_form_plan_json)
    if not produced:
        produced = fallback_attribute_family_plan(context.get("attribute_families"))

    merged: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in deterministic + produced:
        if not isinstance(item, dict):
            continue
        key = normalize_plan_key(item.get("key"))
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(dict(item, key=key))
        if max_items and len(merged) >= max_items:
            break
    if not merged:
        return [], True
    return merged, True
