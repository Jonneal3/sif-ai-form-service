from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.form_planning.form_plan import is_first_batch


def _normalize_id_fragment(text: str) -> str:
    t = str(text or "").strip().lower()
    t = t.replace("step-", "")
    t = t.replace("-", "_")
    t = re.sub(r"[^a-z0-9_]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


def _build_upload_step(step_id: str, role: Optional[str]) -> Dict[str, Any]:
    rid = str(role or "").strip() or None
    label = "Upload a photo"
    if rid == "sceneImage":
        label = "Upload a photo of the space"
    elif rid == "userImage":
        label = "Upload a photo of yourself"
    elif rid == "productImage":
        label = "Upload a photo of the item"
    return {
        "id": step_id,
        "type": "upload",
        "question": label,
        "required": True,
        "max_files": 1,
        "upload_role": rid,
        "allow_skip": False,
    }


def wrap_last_step_with_upload_composite(
    *,
    payload: Dict[str, Any],
    emitted_steps: List[Dict[str, Any]],
    required_uploads: Any,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Wrap the last emitted AI step with deterministic upload step(s) inside a single composite step.

    Key conventions:
    - Each block's `id` is the nested step id (e.g. `step-project-type`, `step-upload-scene`) so the
      frontend stores answers under those ids inside the composite answer object.
    - This lets the backend flatten answers without needing the composite schema at runtime.
    """
    if not is_first_batch(payload):
        return emitted_steps, False
    if not isinstance(emitted_steps, list) or not emitted_steps:
        return emitted_steps, False
    uploads = required_uploads if isinstance(required_uploads, list) else []
    uploads = [u for u in uploads if isinstance(u, dict)]
    if not uploads:
        return emitted_steps, False

    anchor = emitted_steps[-1]
    anchor_id = str(anchor.get("id") or "").strip()
    if not anchor_id:
        return emitted_steps, False

    composite_id = f"step-composite-{_normalize_id_fragment(anchor_id)}"
    blocks: List[Dict[str, Any]] = []

    blocks.append(
        {
            "id": f"{composite_id}-md",
            "kind": "markdown",
            "text": "If you have a reference photo, add it here.",
            "source": "deterministic",
        }
    )

    blocks.append(
        {
            "id": anchor_id,
            "step": anchor,
            "source": "ai",
        }
    )

    for up in uploads:
        step_id = str(up.get("stepId") or up.get("step_id") or up.get("id") or "").strip()
        if not step_id:
            continue
        role = str(up.get("role") or "").strip() or None
        blocks.append(
            {
                "id": step_id,
                "step": _build_upload_step(step_id, role),
                "source": "deterministic",
            }
        )

    if len(blocks) < 2:
        return emitted_steps, False

    composite_step: Dict[str, Any] = {
        "id": composite_id,
        "type": "composite",
        "question": str(anchor.get("question") or "A couple quick things"),
        "blocks": blocks,
        "required": False,
    }

    # Best-effort validation (fail-open to avoid breaking production if schema drifts).
    try:
        from app.schemas.ui_steps import CompositeUI

        CompositeUI.model_validate(composite_step)
    except Exception:
        pass

    out = list(emitted_steps[:-1])
    out.append(composite_step)
    return out, True
