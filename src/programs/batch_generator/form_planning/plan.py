from __future__ import annotations

from typing import Any, Dict, List, Optional


def _as_str(x: Any, *, max_len: int = 2000) -> str:
    s = str(x or "")
    return s[:max_len]


def _normalize_key(key: Any) -> str:
    k = _as_str(key, max_len=120).strip()
    return k


def _as_float(x: Any, *, default: float = 0.1) -> float:
    try:
        f = float(x)
    except Exception:
        return default
    if f < 0:
        return default
    return f


def _as_int(x: Any, *, default: int) -> int:
    try:
        n = int(x)
    except Exception:
        return default
    return n


def _extract_widget_form_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    fp = state.get("formPlan") or state.get("form_plan") or payload.get("formPlan") or payload.get("form_plan")
    return fp if isinstance(fp, dict) else {}


def _answered_step_ids(payload: Dict[str, Any], context: Dict[str, Any]) -> set[str]:
    asked = set()
    for raw in (
        context.get("asked_step_ids"),
        context.get("already_asked_keys"),
        (payload.get("state") or {}).get("askedStepIds") if isinstance(payload.get("state"), dict) else None,
        payload.get("askedStepIds"),
        payload.get("alreadyAskedKeys"),
    ):
        if isinstance(raw, list):
            for sid in raw:
                s = _as_str(sid, max_len=120).strip()
                if s:
                    asked.add(s)
    return asked


def _answered_keys(context: Dict[str, Any]) -> set[str]:
    known = context.get("known_answers")
    if not isinstance(known, dict):
        return set()
    out: set[str] = set()
    for k, v in known.items():
        kk = _normalize_key(k)
        if not kk:
            continue
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        out.add(kk)
    return out


def build_deterministic_form_plan_items_for_batch(
    *,
    payload: Dict[str, Any],
    context: Dict[str, Any],
    batch_number: int,
    max_items: int,
) -> List[Dict[str, Any]]:
    """
    Best-effort deterministic `form_plan` prompt scaffold.

    Priority order:
    1) Widget `state.formPlan.nextBatchGuide[]` (richest per-batch hints).
    2) Widget `state.formPlan.keys[]` (falls back to generic goals).
    """
    if not isinstance(payload, dict) or not isinstance(context, dict):
        return []

    fp = _extract_widget_form_plan(payload)
    if not fp:
        return []

    max_items = max(1, min(12, _as_int(max_items, default=4)))

    answered_keys = _answered_keys(context)
    asked_step_ids = _answered_step_ids(payload, context)

    candidates: list[dict] = []

    guide = fp.get("nextBatchGuide")
    if isinstance(guide, list) and guide:
        for it in guide:
            if not isinstance(it, dict):
                continue
            key = _normalize_key(it.get("key"))
            if not key:
                continue
            if key in answered_keys:
                continue
            step_id = f"step-{key.replace('_', '-')}"
            if step_id in asked_step_ids:
                continue
            candidates.append(
                {
                    "key": key,
                    "goal": _as_str(it.get("goal"), max_len=200) or key.replace("_", " ").title(),
                    "why": _as_str(it.get("why"), max_len=240),
                    "priority": _as_str(it.get("priority"), max_len=40) or "medium",
                    "component_hint": _as_str(it.get("component_hint") or it.get("componentHint"), max_len=40)
                    or "choice",
                    "importance_weight": _as_float(it.get("importance_weight") or it.get("importanceWeight"), default=0.1),
                    "expected_metric_gain": _as_float(it.get("expected_metric_gain") or it.get("expectedMetricGain"), default=0.1),
                }
            )
            if len(candidates) >= max_items:
                break

    if not candidates:
        keys = fp.get("keys")
        if isinstance(keys, list):
            for raw in keys:
                key = _normalize_key(raw)
                if not key:
                    continue
                if key in answered_keys:
                    continue
                step_id = f"step-{key.replace('_', '-')}"
                if step_id in asked_step_ids:
                    continue
                candidates.append(
                    {
                        "key": key,
                        "goal": key.replace("_", " ").title(),
                        "why": "",
                        "priority": "medium" if batch_number > 1 else "high",
                        "component_hint": "choice",
                        "importance_weight": 0.1,
                        "expected_metric_gain": 0.1,
                    }
                )
                if len(candidates) >= max_items:
                    break

    return candidates[:max_items]

