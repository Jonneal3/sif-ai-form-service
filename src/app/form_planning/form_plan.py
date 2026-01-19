from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


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


def build_shared_form_plan(
    *,
    context: Dict[str, Any],
    batch_policy: Optional[Dict[str, Any]],
    form_plan_items: Optional[List[Dict[str, Any]]] = None,
    token_budget_total: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a compact plan object for the UI.

    This includes form-level strategy (constraints, batch flow, and stop conditions).
    It can optionally embed `nextBatchGuide` for round-tripping when the backend relies on the
    frontend to persist the plan across calls (client-side sessions).
    """

    def _map_allowed_component_types(raw: Any) -> Optional[List[str]]:
        if not isinstance(raw, list):
            return None
        out: List[str] = []
        seen: set[str] = set()
        for t in raw:
            s = str(t or "").strip()
            if not s:
                continue
            s = s.lower()
            if s == "multiple_choice":
                s = "choice"
            if s == "text_input":
                s = "text"
            if s in {"file_upload", "file_picker"}:
                s = "upload"
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out or None

    policy = batch_policy if isinstance(batch_policy, dict) else (context.get("batch_policy") if isinstance(context, dict) else None)
    if not policy:
        try:
            from app.form_planning.guides import default_form_skeleton

            policy = default_form_skeleton(
                goal_intent=str((context or {}).get("goal_intent") or "pricing"),
                max_calls=None,
            )
        except Exception:
            policy = None
    phases = policy.get("phases") if isinstance(policy, dict) else None
    stop_conditions = policy.get("stopConditions") if isinstance(policy, dict) else None

    batches: List[Dict[str, Any]] = []
    batch_order: List[str] = []
    max_steps_per_batch: Optional[int] = None
    max_steps_total: Optional[int] = None

    if isinstance(phases, list):
        for p in phases:
            if not isinstance(p, dict):
                continue
            batch_id = str(p.get("id") or "").strip()
            if not batch_id:
                continue
            batch_order.append(batch_id)
            max_steps = p.get("maxSteps")
            try:
                max_steps_int = int(max_steps) if max_steps is not None else None
            except Exception:
                max_steps_int = None
            if isinstance(max_steps_int, int) and max_steps_int > 0:
                max_steps_per_batch = max(max_steps_per_batch or 0, max_steps_int)
                max_steps_total = (max_steps_total or 0) + max_steps_int
            batches.append(
                {
                    "batchId": batch_id,
                    "purpose": str(p.get("purpose") or "").strip() or None,
                    "maxSteps": max_steps_int,
                    "allowedComponentTypes": _map_allowed_component_types(p.get("allowedMiniTypes")),
                    "rigidity": p.get("rigidity"),
                }
            )

    max_batches = None
    if isinstance(policy, dict):
        max_batches = policy.get("maxCalls")
    if max_batches is None and isinstance(context, dict):
        info = context.get("batch_info")
        if isinstance(info, dict):
            max_batches = info.get("max_batches")
    try:
        max_batches_int = int(max_batches) if max_batches is not None else None
    except Exception:
        max_batches_int = None

    if token_budget_total is None and isinstance(context, dict):
        bs = context.get("batch_state")
        if isinstance(bs, dict):
            raw = bs.get("tokens_total_budget") or bs.get("tokensTotalBudget")
            try:
                token_budget_total = int(raw) if raw is not None else None
            except Exception:
                token_budget_total = None
    if isinstance(token_budget_total, int) and token_budget_total <= 0:
        token_budget_total = None

    required_complete = None
    satiety_target = None
    if isinstance(stop_conditions, dict):
        required_complete = stop_conditions.get("requiredKeysComplete")
        satiety_target = stop_conditions.get("satietyTarget")

    # Plan items (question backlog) are optional for the UI renderer, but required for
    # continuity when the backend does not persist sessions server-side.
    plan_items_out = form_plan_items if isinstance(form_plan_items, list) and form_plan_items else None
    plan_keys: Optional[List[str]] = None
    if plan_items_out:
        keys: List[str] = []
        for item in plan_items_out:
            if not isinstance(item, dict):
                continue
            k = str(item.get("key") or "").strip()
            if k:
                keys.append(k)
        plan_keys = keys or None

    return {
        "v": 1,
        "constraints": {
            "maxBatches": max_batches_int,
            "maxStepsTotal": max_steps_total,
            "maxStepsPerBatch": max_steps_per_batch,
            "tokenBudgetTotal": token_budget_total,
        },
        "flow": {
            "batchOrder": batch_order or None,
            "withinBatchStepOrder": "easy_to_deep",
            "priority": {"levels": ["critical", "optional"], "neverSkipCritical": True},
        },
        "batches": batches,
        "stop": {"requiredComplete": required_complete, "satietyTarget": satiety_target},
        "keys": plan_keys,
        "nextBatchGuide": plan_items_out,
    }
