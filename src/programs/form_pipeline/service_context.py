from __future__ import annotations

import json
from typing import Any, Dict, Tuple


def derive_industry_and_service_strings(
    payload: Dict[str, Any],
    *,
    max_len: int = 120,
) -> Tuple[str, str]:
    """
    Derive short, plain-English industry/service strings.

    Modern shape: top-level `industry` and `service` (plus `vertical` alias for industry).
    """

    industry = str(payload.get("industry") or payload.get("vertical") or "").strip()[:max_len]
    service = str(payload.get("service") or "").strip()[:max_len]
    return industry, service


def _coerce_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (dict, list)):
        try:
            raw = json.dumps(raw, ensure_ascii=True)
        except Exception:
            raw = str(raw)
    return str(raw).strip()


def extract_service_summary(payload: Dict[str, Any], *, max_len: int = 1200) -> str:
    """
    Extract a plain-text service summary from a request payload.

    Accepted keys (new + back-compat):
      - `service_summary` / `serviceSummary` (preferred)
      - `services_summary` / `servicesSummary` (legacy)
    """

    text = _coerce_text(
        payload.get("service_summary")
        or payload.get("serviceSummary")
        or payload.get("services_summary")
        or payload.get("servicesSummary")
    )
    if text:
        return text[: int(max_len or 0) or 1200]
    return ""


def extract_company_summary(payload: Dict[str, Any], *, max_len: int = 1200) -> str:
    """
    Extract a plain-text company summary from a request payload.

    Accepted keys:
      - `company_summary` / `companySummary`
    """

    text = _coerce_text(payload.get("company_summary") or payload.get("companySummary"))
    if text:
        return text[: int(max_len or 0) or 1200]
    return ""


def infer_goal_intent(*, services_summary: str = "", explicit_goal_intent: str = "") -> str:
    """
    Decide between "pricing" vs "visual" intent.

    Prefer explicit `goal_intent` from payload; otherwise default to "pricing".
    """

    t = str(explicit_goal_intent or "").strip().lower()
    if t in {"pricing", "visual"}:
        return t
    return "pricing"


__all__ = [
    "derive_industry_and_service_strings",
    "extract_company_summary",
    "extract_service_summary",
    "infer_goal_intent",
]

