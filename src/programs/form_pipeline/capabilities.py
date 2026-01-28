from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional


_CAP_KEYS = ("image_preview", "recommendations", "pricing_preview", "finalization")


def _norm_key(x: Any) -> str:
    t = str(x or "").strip().lower()
    if not t:
        return ""
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    t = re.sub(r"_+", "_", t)
    return t


def _is_non_empty_answer(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        # Booleans are meaningful answers.
        return True
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, tuple, set, dict)):
        return len(v) > 0
    return bool(str(v).strip())


def _answered_count_from_step_data(step_data_so_far: Dict[str, Any]) -> int:
    n = 0
    for k, v in (step_data_so_far or {}).items():
        if str(k) == "__capabilities":
            continue
        if _is_non_empty_answer(v):
            n += 1
    return n


def _answered_count_from_answered_qa(answered_qa: Optional[Iterable[Any]]) -> int:
    if not answered_qa:
        return 0
    n = 0
    for item in answered_qa:
        if not isinstance(item, dict):
            continue
        if _is_non_empty_answer(item.get("answer")):
            n += 1
    return n


def _has_answer(step_data_so_far: Dict[str, Any], key: str) -> bool:
    """
    Best-effort presence check for a semantic answer key.

    NOTE: This service is multi-vertical and does not have a fixed ontology of question keys.
    Keep this deterministic and conservative; UI should treat capabilities as "available"
    signals, not guarantees of perfect domain readiness.
    """
    want = _norm_key(key)
    if not want:
        return False
    for k, v in (step_data_so_far or {}).items():
        nk = _norm_key(k)
        if not nk:
            continue
        if nk == want or nk.endswith(f"_{want}") or nk.startswith(f"{want}_") or want in nk:
            if _is_non_empty_answer(v):
                return True
    return False


def compute_capabilities(
    *,
    step_data_so_far: Dict[str, Any],
    answered_qa: Optional[Iterable[Any]] = None,
    previous_caps: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    """
    Compute backend-owned capability flags.

    Requirements:
    - Deterministic and derived only from known answers + rules.
    - Monotonic: once true, never flips back to false (merged with `previous_caps`).
    - Session-safe: callers can persist the returned flags inside stepDataSoFar["__capabilities"].
    """
    prev = previous_caps if isinstance(previous_caps, dict) else {}

    # Prefer stepDataSoFar (canonical "known answers") and backstop with answeredQA.
    answered_n = max(_answered_count_from_step_data(step_data_so_far), _answered_count_from_answered_qa(answered_qa))

    # A simple, deterministic completeness proxy for generic flows.
    # 6 answers => "complete enough" for most mid-flow experiences.
    target = 6
    completeness = min(1.0, float(answered_n) / float(target))

    computed: Dict[str, bool] = {}
    computed["image_preview"] = answered_n >= 3
    computed["recommendations"] = computed["image_preview"] and completeness >= 0.7

    # Pricing readiness is best-effort: look for common budget/timeline keys.
    has_budget = _has_answer(step_data_so_far, "budget") or _has_answer(step_data_so_far, "price") or _has_answer(step_data_so_far, "cost")
    has_timeline = (
        _has_answer(step_data_so_far, "timeline")
        or _has_answer(step_data_so_far, "date")
        or _has_answer(step_data_so_far, "start_date")
        or _has_answer(step_data_so_far, "deadline")
    )
    computed["pricing_preview"] = bool(has_budget and has_timeline)

    # Finalization becomes available once we're "complete enough".
    computed["finalization"] = completeness >= 1.0 or answered_n >= 8

    # Enforce monotonicity + stable key set.
    out: Dict[str, bool] = {}
    for k in _CAP_KEYS:
        out[k] = bool(prev.get(k) is True or computed.get(k) is True)
    return out


__all__ = ["compute_capabilities"]

