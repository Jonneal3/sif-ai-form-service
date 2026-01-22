from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _normalize_step_id(step_id: Any) -> str:
    s = str(step_id or "").strip()
    return s


def _extract_required_upload_steps(required_uploads: Any) -> List[Dict[str, Any]]:
    """
    Convert `required_uploads` (best-effort) into deterministic upload UI steps.
    """
    out: list[dict] = []
    if not isinstance(required_uploads, list):
        return out
    for idx, u in enumerate(required_uploads):
        if not isinstance(u, dict):
            continue
        sid = _normalize_step_id(u.get("stepId") or u.get("step_id") or u.get("id"))
        role = str(u.get("role") or "").strip()
        if not sid:
            sid = f"step-upload-{idx+1}"
        out.append(
            {
                "id": sid,
                "type": "file_upload",
                "question": str(u.get("question") or u.get("label") or "Upload a file").strip(),
                "required": True,
                "role": role or None,
            }
        )
    return out


def wrap_last_step_with_upload_composite(
    *,
    payload: Dict[str, Any],
    emitted_steps: List[Dict[str, Any]],
    required_uploads: Optional[Any],
) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    """
    If uploads are required and the emitted steps do not already include an upload step,
    wrap the final emitted step into a `composite` step with deterministic upload blocks.
    """
    if not isinstance(emitted_steps, list) or not emitted_steps:
        return None, False

    upload_steps = _extract_required_upload_steps(required_uploads)
    if not upload_steps:
        return None, False

    existing_ids = set()
    for s in emitted_steps:
        if isinstance(s, dict):
            sid = _normalize_step_id(s.get("id"))
            if sid:
                existing_ids.add(sid)
            stype = str(s.get("type") or "").strip().lower()
            if stype in ("file_upload", "upload", "file_picker"):
                return None, False

    uploads_to_add = [u for u in upload_steps if _normalize_step_id(u.get("id")) not in existing_ids]
    if not uploads_to_add:
        return None, False

    last = emitted_steps[-1] if isinstance(emitted_steps[-1], dict) else {}
    last_id = _normalize_step_id(last.get("id")) or "step-final"

    composite = {
        "id": f"{last_id}-composite",
        "type": "composite",
        "question": last.get("question") or None,
        "required": True,
        "blocks": [last] + uploads_to_add,
    }
    return [*emitted_steps[:-1], composite], True

