from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from programs.form_pipeline.utils import _normalize_step_id


def extract_session_id(payload: Dict[str, Any]) -> str:
    """
    Extract a stable session identifier (if provided).

    Modern shape: top-level `sessionId` or `session_id`.
    """

    for key in ("sessionId", "session_id"):
        v = payload.get(key)
        if v:
            return str(v)[:120]
    return ""


def extract_answered_qa(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Expected shape: [{ stepId, question, answer }]
    """

    answered_qa_raw = payload.get("answeredQA") or payload.get("answered_qa")

    answered_qa: List[Dict[str, str]] = []
    if isinstance(answered_qa_raw, list):
        for item in answered_qa_raw:
            if not isinstance(item, dict):
                continue
            step_id = _normalize_step_id(str(item.get("stepId") or item.get("step_id") or item.get("id") or "").strip())
            question = str(item.get("question") or item.get("q") or "").strip()
            answer = item.get("answer") or item.get("a")
            if answer is None:
                answer_text = ""
            elif isinstance(answer, (dict, list)):
                try:
                    answer_text = json.dumps(answer, ensure_ascii=True)
                except Exception:
                    answer_text = str(answer)
            else:
                answer_text = str(answer)
            answer_text = answer_text.strip()

            if not step_id or not step_id.startswith("step-"):
                continue
            if not question and not answer_text:
                continue
            answered_qa.append({"stepId": step_id, "question": question[:200], "answer": answer_text[:300]})
            if len(answered_qa) >= 24:
                break

    return answered_qa


def extract_asked_step_ids(payload: Dict[str, Any], *, answered_qa: Optional[List[Dict[str, str]]] = None) -> List[str]:
    """
    Asked step ids are derived from answered Q/A (preferred), but we also accept an explicit
    `askedStepIds[]` list so clients can dedupe even when they don't send `answeredQA`.
    """

    normalized: List[str] = []
    if isinstance(answered_qa, list) and answered_qa:
        for item in answered_qa:
            if not isinstance(item, dict):
                continue
            sid = _normalize_step_id(str(item.get("stepId") or item.get("step_id") or item.get("id") or "").strip())
            if sid and sid.startswith("step-") and sid not in normalized:
                normalized.append(sid)

    explicit = payload.get("askedStepIds") or payload.get("asked_step_ids")
    if isinstance(explicit, list):
        for raw in explicit:
            sid = _normalize_step_id(str(raw or "").strip())
            if sid and sid.startswith("step-") and sid not in normalized:
                normalized.append(sid)
    return normalized


def _normalize_use_case(raw: Any) -> str:
    t = str(raw or "").strip().lower()
    if not t:
        return "scene"
    t = t.replace("_", " ").replace("-", " ").strip()
    if "tryon" in t or "try on" in t:
        return "tryon"
    if "scene placement" in t or "placement" in t:
        return "scene_placement"
    if "scene" in t:
        return "scene"
    return t.replace(" ", "_")


def extract_use_case(payload: Dict[str, Any]) -> str:
    """
    Single modern shape (top-level). Accept camelCase + snake_case.
    """

    raw = payload.get("useCase") or payload.get("use_case")
    return _normalize_use_case(raw)


__all__ = [
    "extract_answered_qa",
    "extract_asked_step_ids",
    "extract_session_id",
    "extract_use_case",
]

