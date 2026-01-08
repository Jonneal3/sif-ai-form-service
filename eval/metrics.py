from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _normalize_step_id(step_id: str) -> str:
    t = str(step_id or "").strip()
    if not t:
        return t
    return t.replace("_", "-")


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


@dataclass(frozen=True)
class EvalMetrics:
    ok: bool
    num_steps: int
    within_max_steps: bool
    ids_all_normalized: bool
    ids_all_prefixed_step: bool
    no_steps_in_already_asked: bool
    types_all_allowed: bool
    has_min_step_when_needed: bool
    errors: Tuple[str, ...]


def compute_metrics(payload: Dict[str, Any], result: Dict[str, Any]) -> EvalMetrics:
    errors: List[str] = []

    mini_steps = result.get("miniSteps") if isinstance(result, dict) else None
    if not isinstance(mini_steps, list):
        mini_steps = []
        errors.append("miniSteps_missing_or_not_list")

    # Pull constraints from payload (accept both camelCase and snake_case).
    max_steps_raw = payload.get("maxSteps") or payload.get("max_steps") or "4"
    try:
        max_steps = int(str(max_steps_raw))
    except Exception:
        max_steps = 4
        errors.append("maxSteps_not_int")

    allowed_raw = payload.get("allowedMiniTypes") or payload.get("allowed_mini_types") or []
    allowed_types: List[str]
    if isinstance(allowed_raw, list):
        allowed_types = [str(x).strip() for x in allowed_raw if str(x).strip()]
    else:
        allowed_types = [s.strip() for s in str(allowed_raw).split(",") if s.strip()]
    allowed_set = set(allowed_types)

    already_raw = payload.get("alreadyAskedKeys") or payload.get("alreadyAskedKeysJson") or payload.get("already_asked_keys") or []
    already: List[str] = []
    if isinstance(already_raw, list):
        already = [_normalize_step_id(str(x)) for x in already_raw if str(x or "").strip()]
    elif isinstance(already_raw, str):
        parsed = _safe_json_loads(already_raw)
        if isinstance(parsed, list):
            already = [_normalize_step_id(str(x)) for x in parsed if str(x or "").strip()]
    already_set = set(already)

    # Determine whether a step is required based on form plan + already asked.
    form_plan_raw = payload.get("formPlan") or payload.get("form_plan") or []
    plan_items: List[Dict[str, Any]] = form_plan_raw if isinstance(form_plan_raw, list) else []
    planned_ids: List[str] = []
    for it in plan_items:
        if not isinstance(it, dict):
            continue
        key = str(it.get("key") or "").strip()
        if not key:
            continue
        planned_ids.append(_normalize_step_id(f"step-{key}").replace("--", "-"))
    unasked_planned = [pid for pid in planned_ids if pid and pid not in already_set]
    needed_min_one = len(unasked_planned) > 0

    ids_all_normalized = True
    ids_all_prefixed_step = True
    no_steps_in_already = True
    types_all_allowed = True if allowed_set else True

    for s in mini_steps:
        if not isinstance(s, dict):
            errors.append("miniStep_not_object")
            continue
        sid = _normalize_step_id(str(s.get("id") or ""))
        if not sid:
            errors.append("miniStep_missing_id")
            continue
        if "_" in sid:
            ids_all_normalized = False
        if not sid.startswith("step-"):
            ids_all_prefixed_step = False
        if sid in already_set:
            no_steps_in_already = False
        stype = str(s.get("type") or "").strip()
        if allowed_set and stype and stype not in allowed_set:
            types_all_allowed = False

    within_max_steps = len(mini_steps) <= max_steps if max_steps >= 0 else True
    if not within_max_steps:
        errors.append("exceeds_maxSteps")

    has_min_step_when_needed = (len(mini_steps) >= 1) if needed_min_one else True
    if needed_min_one and not has_min_step_when_needed:
        errors.append("needed_min_one_step_but_zero")

    ok = bool(result.get("ok")) and len([e for e in errors if "missing_or_not_list" not in e]) == 0
    return EvalMetrics(
        ok=ok,
        num_steps=len(mini_steps),
        within_max_steps=within_max_steps,
        ids_all_normalized=ids_all_normalized,
        ids_all_prefixed_step=ids_all_prefixed_step,
        no_steps_in_already_asked=no_steps_in_already,
        types_all_allowed=types_all_allowed,
        has_min_step_when_needed=has_min_step_when_needed,
        errors=tuple(errors),
    )


