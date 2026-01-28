from __future__ import annotations


def planner_cache_key(*, session_id: str, services_fingerprint: str, use_case_key: str) -> str:
    """
    Cache key for the full planner plan.

    Note: this is intentionally string-only and stable. TTL behavior lives in the caller.
    """
    sid = str(session_id or "").strip()
    if not sid:
        return ""
    svc = str(services_fingerprint or "").strip() or "none"
    uc = str(use_case_key or "").strip().lower() or "none"
    return f"question_plan:{sid}:{svc}:{uc}"


__all__ = ["planner_cache_key"]

