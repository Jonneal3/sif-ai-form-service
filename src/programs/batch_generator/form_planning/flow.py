from __future__ import annotations

from typing import Any, Dict, List, Tuple

from programs.batch_generator.form_planning.batch_ordering import resolve_stage
from programs.batch_generator.form_planning.components_allowed import allowed_components
from programs.batch_generator.form_planning.question_tonality import get_question_hints


def _as_str(x: Any, *, max_len: int = 200) -> str:
    return str(x or "")[:max_len]


def _as_int(x: Any, *, default: int) -> int:
    try:
        n = int(x)
    except Exception:
        return default
    return n


def _extract_goal_intent(context: Dict[str, Any]) -> str:
    return _as_str(context.get("goal_intent") or context.get("goalIntent") or "", max_len=80).strip().lower()


def _extract_use_case(context: Dict[str, Any]) -> str:
    return _as_str(context.get("use_case") or context.get("useCase") or "", max_len=80).strip().lower()


def _resolve_total_batches(context: Dict[str, Any]) -> int:
    batch_constraints = context.get("batch_constraints") if isinstance(context.get("batch_constraints"), dict) else {}
    n = batch_constraints.get("maxBatches")
    if n is None:
        info = context.get("batch_info") if isinstance(context.get("batch_info"), dict) else {}
        n = info.get("max_batches") or info.get("maxBatches") or info.get("maxCalls")
    if n is None:
        try:
            from programs.batch_generator.form_planning.static_constraints import DEFAULT_CONSTRAINTS

            n = (DEFAULT_CONSTRAINTS or {}).get("maxBatches")
        except Exception:
            n = 2
    return max(1, _as_int(n, default=2))


def flow_guide_for_batch(*, context: Dict[str, Any], batch_number: int) -> Dict[str, Any]:
    """
    A hardcoded flow "skeleton" we can pass to the model (and also use for runtime defaults).
    """
    use_case = _extract_use_case(context)
    goal_intent = _extract_goal_intent(context)

    total_batches = _resolve_total_batches(context)
    batch_index = max(0, _as_int(batch_number, default=1) - 1)
    stage = resolve_stage(batch_index=batch_index, total_batches=total_batches)

    allowed = allowed_components(stage)
    question_hints = get_question_hints(stage)

    # Early = bias toward structured components and remove text when structured types exist.
    prefer_structured_inputs = stage == "early"

    guide: Dict[str, Any] = {
        "v": 1,
        "stage": stage,
        "batchNumber": int(batch_number or 1),
        "batchIndex": int(batch_index),
        "totalBatches": int(total_batches),
        "rules": {
            "preferStructuredInputs": prefer_structured_inputs,
            "allowedMiniTypesDefault": allowed,
            "questionHints": question_hints,
        },
    }
    if use_case:
        guide["useCase"] = use_case
    if goal_intent:
        guide["goalIntent"] = goal_intent
    return guide


def apply_flow_guide(
    *,
    payload: Dict[str, Any],
    context: Dict[str, Any],
    batch_number: int,
    extracted_allowed_mini_types: List[str],
    extracted_max_steps: int,
) -> Tuple[Dict[str, Any], List[str], int]:
    """
    Apply the flow guide:
    - Always set `context["flow_guide"]` so DSPy sees the skeleton.
    - Set `context["prefer_structured_inputs"]` per batch stage.
    - Provide defaults for allowed types + max steps if missing.
    """
    if not isinstance(context, dict):
        context = {}
    guide = flow_guide_for_batch(context=context, batch_number=batch_number)
    context = dict(context)
    context["flow_guide"] = guide
    prefer_structured = bool((guide.get("rules") or {}).get("preferStructuredInputs"))
    context["prefer_structured_inputs"] = prefer_structured

    allowed = list(extracted_allowed_mini_types or [])
    if not allowed:
        allowed = list((guide.get("rules") or {}).get("allowedMiniTypesDefault") or [])

    max_steps = int(extracted_max_steps or 0)
    if max_steps <= 0:
        max_steps = 4
    # Keep early batches short by default.
    if guide.get("stage") == "early":
        max_steps = max(1, min(max_steps, 3))
    elif guide.get("stage") == "middle":
        max_steps = max(1, min(max_steps, 4))
    return context, allowed, max_steps
