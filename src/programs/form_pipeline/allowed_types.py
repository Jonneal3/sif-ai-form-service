from __future__ import annotations

from typing import Any, Dict, List


def _normalize_allowed_mini_types(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [s.strip() for s in str(raw or "").split(",") if s.strip()]


def _normalize_allowed_component_types(raw: Any) -> List[str]:
    """
    Back-compat adapter.

    Some clients send `allowedComponentTypes` with values like:
      ["choice", "slider", "text"]
    while DSPy signatures use `text_input`.
    """

    values = _normalize_allowed_mini_types(raw)
    mapped: List[str] = []
    for v in values:
        t = str(v or "").strip().lower()
        if not t:
            continue
        if t == "text":
            t = "text_input"
        mapped.append(t)
    return mapped


def extract_allowed_mini_types_from_payload(payload: Dict[str, Any]) -> List[str]:
    raw = payload.get("allowedMiniTypes") or payload.get("allowed_mini_types")
    types = _normalize_allowed_mini_types(raw)
    if types:
        return types
    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}
    raw_component_types = None
    if isinstance(current_batch, dict):
        raw_component_types = current_batch.get("allowedComponentTypes") or current_batch.get("allowed_component_types")
    if raw_component_types:
        return _normalize_allowed_component_types(raw_component_types)
    return []


DEFAULT_ALLOWED_MINI_TYPES: List[str] = [
    "multiple_choice",
    "yes_no",
    "slider",
    "rating",
    "file_upload",
    "segmented_choice",
    "chips_multi",
    "searchable_select",
    "gallery",
]


def ensure_allowed_mini_types(allowed: List[str]) -> List[str]:
    values = [str(x).strip().lower() for x in (allowed or []) if str(x).strip()]
    return values or list(DEFAULT_ALLOWED_MINI_TYPES)


def prefer_structured_allowed_mini_types(raw: Any) -> List[str]:
    types = [t.strip().lower() for t in _normalize_allowed_mini_types(raw) if str(t or "").strip()]
    if not types:
        return types
    structured = {"choice", "multiple_choice", "segmented_choice", "chips_multi", "yes_no", "slider", "rating", "range_slider"}
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
    if t in ["text", "text_input"]:
        return "text" in allowed or "text_input" in allowed
    if t in ["slider", "rating", "range_slider"]:
        return "slider" in allowed or "rating" in allowed or "range_slider" in allowed
    if t in ["upload", "file_upload", "file_picker"]:
        return "upload" in allowed or "file_upload" in allowed or "file_picker" in allowed
    if t in ["gallery"]:
        return "gallery" in allowed
    return False


__all__ = [
    "extract_allowed_mini_types_from_payload",
    "ensure_allowed_mini_types",
    "prefer_structured_allowed_mini_types",
    "allowed_type_matches",
    "DEFAULT_ALLOWED_MINI_TYPES",
]

