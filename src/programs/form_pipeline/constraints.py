from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple


def _as_int(value: Any) -> Optional[int]:
    try:
        n = int(value)
    except Exception:
        return None
    return n if n > 0 else None


def _get_int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def extract_token_budget(batch_state: Any) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(batch_state, dict):
        return None, None
    total_raw = batch_state.get("tokensTotalBudget")
    used_raw = batch_state.get("tokensUsedSoFar")
    try:
        total = int(total_raw) if total_raw is not None else None
    except Exception:
        total = None
    try:
        used = int(used_raw) if used_raw is not None else None
    except Exception:
        used = None
    return total, used


def extract_form_state_subset(payload: Dict[str, Any], batch_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Modern shape: top-level `formState` / `form_state`, or `currentBatch`.
    """

    form_state: Any = payload.get("formState") or payload.get("form_state") or {}
    if not isinstance(form_state, dict):
        form_state = {}

    batch_index = form_state.get("batchIndex") or form_state.get("batch_index") or form_state.get("batchNumber") or form_state.get("batch_number")
    max_batches = form_state.get("maxBatches") or form_state.get("max_batches") or form_state.get("maxCalls") or form_state.get("max_calls")
    calls_remaining = form_state.get("callsRemaining") or form_state.get("calls_remaining")
    if max_batches is None and isinstance(batch_state, dict):
        max_batches = batch_state.get("maxCalls")
    if calls_remaining is None and isinstance(batch_state, dict):
        calls_remaining = batch_state.get("callsRemaining")

    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}
    if batch_index is None and isinstance(current_batch, dict):
        batch_index = current_batch.get("batchNumber") or current_batch.get("batch_number")

    subset: Dict[str, Any] = {}
    if batch_index is not None:
        try:
            subset["batch_index"] = int(batch_index)
        except Exception:
            pass
    if max_batches is not None:
        try:
            subset["max_batches"] = int(max_batches)
        except Exception:
            pass
    if calls_remaining is not None:
        try:
            subset["calls_remaining"] = int(calls_remaining)
        except Exception:
            pass
    return subset


def resolve_backend_max_calls(*, default_max_calls: int = 2) -> int:
    """
    Backend-owned call cap.
    """

    try:
        from programs.form_pipeline.planning import DEFAULT_CONSTRAINTS

        default_max_calls = int((DEFAULT_CONSTRAINTS or {}).get("maxBatches") or default_max_calls)
    except Exception:
        default_max_calls = 2

    return max(1, min(10, _get_int_env("AI_FORM_MAX_BATCH_CALLS", default_max_calls)))


def build_batch_constraints(*, payload: Dict[str, Any], batch_state: Dict[str, Any], max_batches: int) -> Dict[str, Any]:
    """
    Build backend constraints we share with the frontend (max calls, step limits, token budget).
    """

    default_min_steps_per_batch = 2
    default_max_steps_per_batch = 4
    default_token_budget_total = 3000
    default_default_steps_per_batch = 8
    try:
        from programs.form_pipeline.planning import DEFAULT_CONSTRAINTS

        default_min_steps_per_batch = int((DEFAULT_CONSTRAINTS or {}).get("minStepsPerBatch") or default_min_steps_per_batch)
        default_max_steps_per_batch = int((DEFAULT_CONSTRAINTS or {}).get("maxStepsPerBatch") or default_max_steps_per_batch)
        default_token_budget_total = int((DEFAULT_CONSTRAINTS or {}).get("tokenBudgetTotal") or default_token_budget_total)
        default_default_steps_per_batch = int((DEFAULT_CONSTRAINTS or {}).get("defaultStepsPerBatch") or default_default_steps_per_batch)
    except Exception:
        pass

    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}
    min_steps_per_batch = (
        _as_int(payload.get("minStepsPerBatch"))
        or _as_int(payload.get("min_steps_per_batch"))
        or _as_int(current_batch.get("minStepsPerBatch"))
        or _as_int(current_batch.get("min_steps_per_batch"))
        or _as_int(os.getenv("AI_FORM_MIN_STEPS_PER_BATCH"))
        or default_min_steps_per_batch
    )
    max_steps_per_batch = (
        _as_int(payload.get("maxSteps"))
        or _as_int(payload.get("max_steps"))
        or _as_int(current_batch.get("maxSteps"))
        or _as_int(current_batch.get("max_steps"))
        or _as_int(os.getenv("AI_FORM_MAX_STEPS_PER_BATCH"))
        or default_max_steps_per_batch
    )
    default_steps_per_batch = (
        _as_int(payload.get("defaultStepsPerBatch"))
        or _as_int(payload.get("default_steps_per_batch"))
        or _as_int(current_batch.get("defaultStepsPerBatch"))
        or _as_int(current_batch.get("default_steps_per_batch"))
        or _as_int(os.getenv("AI_FORM_DEFAULT_STEPS_PER_BATCH"))
        or default_default_steps_per_batch
    )
    if min_steps_per_batch < 1:
        min_steps_per_batch = default_min_steps_per_batch
    if max_steps_per_batch < min_steps_per_batch:
        max_steps_per_batch = min_steps_per_batch
    if default_steps_per_batch < min_steps_per_batch:
        default_steps_per_batch = min_steps_per_batch
    if default_steps_per_batch > max_steps_per_batch:
        default_steps_per_batch = max_steps_per_batch

    max_steps_total = _as_int(batch_state.get("max_steps_total")) or _as_int(batch_state.get("maxStepsTotal")) or max_steps_per_batch * max_batches
    token_budget_total = (
        _as_int(batch_state.get("tokensTotalBudget"))
        or _as_int(batch_state.get("token_budget_total"))
        or _as_int(os.getenv("AI_FORM_TOKEN_BUDGET_TOTAL"))
        or default_token_budget_total
    )
    # Keep budgets in a sane default range. This is a *soft* product constraint:
    # callers can still override via env/payload, but we avoid huge or tiny budgets by default.
    token_budget_total = max(3000, min(int(token_budget_total or default_token_budget_total), 5000))
    return {
        "maxBatches": max_batches,
        "maxStepsTotal": max_steps_total,
        "minStepsPerBatch": min_steps_per_batch,
        "maxStepsPerBatch": max_steps_per_batch,
        "defaultStepsPerBatch": default_steps_per_batch,
        "tokenBudgetTotal": token_budget_total,
    }


__all__ = [
    "build_batch_constraints",
    "extract_form_state_subset",
    "extract_token_budget",
    "resolve_backend_max_calls",
]

