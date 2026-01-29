"""
Form pipeline helper: allowed UI step type policy.

Used by the orchestrator to:
- parse allowed step types from payload/currentBatch
- filter caller-provided types to schema-known values
- enforce backend-owned allowed type policy (currently: `multiple_choice`)
- validate that emitted renderer steps match the allowed type set
"""

from __future__ import annotations

from typing import Any, Dict, List

from schemas.ui_steps import UI_STEP_TYPE_VALUES


def _normalize_allowed_mini_types(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    return [s.strip().lower() for s in str(raw or "").split(",") if s.strip()]


_KNOWN_TYPES: set[str] = set([t.lower() for t in (UI_STEP_TYPE_VALUES or []) if str(t).strip()])


def extract_allowed_mini_types_from_payload(payload: Dict[str, Any]) -> List[str]:
    raw = payload.get("allowedMiniTypes") or payload.get("allowed_mini_types")
    types = _normalize_allowed_mini_types(raw)
    if types:
        return [t for t in types if (not _KNOWN_TYPES) or (t in _KNOWN_TYPES)]
    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}
    if isinstance(current_batch, dict):
        raw_component_types = current_batch.get("allowedComponentTypes") or current_batch.get("allowed_component_types")
        if raw_component_types:
            types = _normalize_allowed_mini_types(raw_component_types)
            # Back-compat mapping: some clients send "text" but schema uses "text_input".
            types = ["text_input" if t == "text" else t for t in types]
            return [t for t in types if (not _KNOWN_TYPES) or (t in _KNOWN_TYPES)]
    return []


DEFAULT_ALLOWED_MINI_TYPES: List[str] = [
    # Backend-owned policy: keep step types minimal.
    # All "choice styling" should be expressed via fields on `multiple_choice`
    # (e.g. allow_multiple, UI hints) rather than new step `type`s.
    "multiple_choice",
]


def ensure_allowed_mini_types(allowed: List[str]) -> List[str]:
    values = [str(x).strip().lower() for x in (allowed or []) if str(x).strip()]
    if _KNOWN_TYPES:
        values = [t for t in values if t in _KNOWN_TYPES]
    # Enforce backend-owned type policy: never allow callers to widen the set.
    policy = {str(x).strip().lower() for x in (DEFAULT_ALLOWED_MINI_TYPES or []) if str(x).strip()}
    if policy:
        values = [t for t in values if t in policy]
    return values or list(DEFAULT_ALLOWED_MINI_TYPES)


def prefer_structured_allowed_mini_types(raw: Any) -> List[str]:
    types = [t.strip().lower() for t in _normalize_allowed_mini_types(raw) if str(t or "").strip()]
    if not types:
        return types
    if _KNOWN_TYPES:
        types = [t for t in types if t in _KNOWN_TYPES]
    structured = {"choice", "multiple_choice"}
    has_structured = any(t in structured for t in types)
    if not has_structured:
        return types
    return [t for t in types if t not in {"text", "text_input"}]


def allowed_type_matches(step_type: str, allowed: set[str]) -> bool:
    if not allowed:
        return True
    t = str(step_type or "").strip().lower()
    if not t:
        return False
    if t in allowed:
        return True
    if t == "choice":
        return "choice" in allowed or "multiple_choice" in allowed
    if t == "multiple_choice":
        return "multiple_choice" in allowed or "choice" in allowed
    # Do NOT implicitly widen types. If you want a variant, it must be explicitly allowed.
    return False


__all__ = [
    "DEFAULT_ALLOWED_MINI_TYPES",
    "allowed_type_matches",
    "ensure_allowed_mini_types",
    "extract_allowed_mini_types_from_payload",
    "prefer_structured_allowed_mini_types",
]
