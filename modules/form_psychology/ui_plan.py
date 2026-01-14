from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules.form_psychology.form_plan import is_first_batch, step_id_from_key


def _step_id_from_upload(required_upload: Dict[str, Any]) -> str:
    raw = (
        required_upload.get("stepId")
        or required_upload.get("step_id")
        or required_upload.get("id")
        or ""
    )
    return str(raw or "").strip()


def build_ui_plan(
    *,
    payload: Dict[str, Any],
    final_form_plan: List[Dict[str, Any]],
    emitted_mini_steps: List[Dict[str, Any]],
    required_uploads: Any,
) -> Optional[Dict[str, Any]]:
    """
    Build a tiny placement spec for deterministic/structural steps.

    Current behavior:
    - For first batch only, place required upload step(s) immediately AFTER the last emitted mini step
      (so they appear "in the plan order" without consuming LLM budget inside the batch).
    - If no emitted steps are present, place uploads at START.
    - Other structural types can be added later via deterministic plan items.
    """
    if not is_first_batch(payload):
        return None
    if not isinstance(final_form_plan, list) or not final_form_plan:
        return None

    try:
        from modules.schemas.ui_steps import UIPlan, UIPlacement
    except Exception:
        return None

    placements: List[Any] = []
    anchor_step_id = None
    if isinstance(emitted_mini_steps, list) and emitted_mini_steps:
        anchor_step_id = str(emitted_mini_steps[-1].get("id") or "").strip() or None

    uploads = required_uploads if isinstance(required_uploads, list) else []
    seen_ids: set[str] = set()
    for up in uploads:
        if not isinstance(up, dict):
            continue
        step_id = _step_id_from_upload(up)
        if not step_id:
            continue
        if step_id in seen_ids:
            continue
        seen_ids.add(step_id)
        role = str(up.get("role") or "").strip() or None
        if anchor_step_id:
            placements.append(
                UIPlacement(
                    id=step_id,
                    type="upload",
                    role=role,
                    position="after",
                    anchor_step_id=anchor_step_id,
                    deterministic=True,
                )
            )
        else:
            placements.append(
                UIPlacement(
                    id=step_id,
                    type="upload",
                    role=role,
                    position="start",
                    anchor_step_id=None,
                    deterministic=True,
                )
            )

    # Future: derive placements from deterministic items in final_form_plan.
    # For now, we only place uploads via required_uploads (authoritative).
    plan = UIPlan(v=1, placements=placements)
    if not plan.placements:
        return None
    return plan.model_dump(by_alias=True)


def derive_deterministic_step_ids(final_form_plan: List[Dict[str, Any]]) -> List[str]:
    """
    Convenience helper for UIs that want to precompute deterministic step ids
    from plan keys (e.g., key=upload_scene -> step-upload-scene).
    """
    out: List[str] = []
    for item in final_form_plan or []:
        if not isinstance(item, dict):
            continue
        if not item.get("deterministic"):
            continue
        sid = step_id_from_key(item.get("key"))
        if sid:
            out.append(sid)
    return out

