from __future__ import annotations

from typing import Any, Dict, List, Optional


def _normalize_step_id(step_id: Any) -> str:
    return str(step_id or "").strip()


def build_deterministic_placements(
    *,
    payload: Dict[str, Any],
    final_form_plan: List[Dict[str, Any]],
    emitted_mini_steps: List[Dict[str, Any]],
    required_uploads: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """
    Provide a minimal deterministic placements payload so a client can inject uploads
    without requiring the model to output them.
    """
    if not isinstance(required_uploads, list) or not required_uploads:
        return None

    emitted_ids: set[str] = set()
    last_step_id: Optional[str] = None
    for s in emitted_mini_steps if isinstance(emitted_mini_steps, list) else []:
        if not isinstance(s, dict):
            continue
        sid = _normalize_step_id(s.get("id"))
        if sid:
            emitted_ids.add(sid)
            last_step_id = sid

    placements: list[dict] = []
    for idx, u in enumerate(required_uploads):
        if not isinstance(u, dict):
            continue
        step_id = _normalize_step_id(u.get("stepId") or u.get("step_id") or u.get("id")) or f"step-upload-{idx+1}"
        if step_id in emitted_ids:
            continue
        placements.append(
            {
                "kind": "upload",
                "stepId": step_id,
                "role": str(u.get("role") or "").strip() or None,
                "insertAfterStepId": last_step_id,
                "required": True,
            }
        )

    if not placements:
        return None

    return {"version": 1, "placements": placements}

