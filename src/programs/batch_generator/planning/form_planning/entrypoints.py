from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _resolve_copy_pack_id(payload: Dict[str, Any]) -> str:
    for key in ("copyPackId", "copy_pack_id", "copyPack", "copy_pack"):
        val = payload.get(key)
        if val:
            return str(val).strip()
    request = payload.get("request")
    if isinstance(request, dict):
        for key in ("copyPackId", "copy_pack_id", "copyPack", "copy_pack"):
            val = request.get(key)
            if val:
                return str(val).strip()
    return "default_v1"


def apply_planning_rules(
    *,
    payload: Dict[str, Any],
    context: Dict[str, Any],
    batch_number: int,
    extracted_allowed_mini_types: List[str],
    extracted_max_steps: int,
) -> Tuple[Dict[str, Any], List[str], int, Dict[str, Any], str]:
    """
    Single integration point for `planning/form_planning/`.

    Returns:
    - context (possibly modified)
    - allowed_mini_types (possibly modified)
    - max_steps (possibly modified)
    - lint_config (for post-generation linting/sanitizing)
    - copy_pack_id (resolved)
    """
    if not isinstance(context, dict):
        context = {}

    # 1) Copy pack (style + lint rules)
    copy_pack_id = _resolve_copy_pack_id(payload)
    lint_config: Dict[str, Any] = {}
    try:
        from programs.batch_generator.planning.form_planning.copywriting.compiler import compile_pack, load_pack

        pack = load_pack(copy_pack_id)
        style_snippet_json, lint_config = compile_pack(pack)
        if style_snippet_json:
            context = dict(context)
            context["copy_style"] = style_snippet_json
            context["copy_context"] = style_snippet_json
    except Exception:
        # Best-effort: copy packs are not required for the pipeline to run.
        lint_config = {}

    # 2) Flow guide (allowed types + max steps + skeleton passed to the model)
    allowed = list(extracted_allowed_mini_types or [])
    max_steps = int(extracted_max_steps or 0)
    try:
        from programs.batch_generator.planning.form_planning.flow import apply_flow_guide

        context, allowed, max_steps = apply_flow_guide(
            payload=payload,
            context=context,
            batch_number=batch_number,
            extracted_allowed_mini_types=allowed,
            extracted_max_steps=max_steps,
        )
    except Exception:
        pass

    return context, allowed, max_steps, (lint_config if isinstance(lint_config, dict) else {}), copy_pack_id


__all__ = ["apply_planning_rules"]

