"""
Form pipeline helper: batch flow policy + deterministic suffix planning.

Used by the orchestrator to:
- pick stage (early/middle/late/single) per batch
- provide default allowed mini types + max steps per call
- attach a `flow_guide` skeleton into the planner context
- (optionally) append deterministic "upload/gallery" plan items when required uploads exist

DEPRECATED (runtime):
The current orchestrator does not use this module. It is kept as a reference implementation
for future multi-batch flow policy experiments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from programs.form_pipeline.constraints import DEFAULT_CONSTRAINTS


def resolve_stage(*, batch_index: int, total_batches: int) -> str:
    """
    Returns a stage label: early / middle / late / single

    - `batch_index` is 0-based.
    - `total_batches` is the planned max batches/calls.
    """
    try:
        idx = int(batch_index)
    except Exception:
        idx = 0
    try:
        total = int(total_batches)
    except Exception:
        total = 1

    # Special-case: a single batch should not inherit "early" caps.
    if total <= 1:
        return "single"
    if idx <= 0:
        return "early"
    if idx < total - 1:
        return "middle"
    return "late"


FLOW_COMPONENTS: dict[str, list[str]] = {
    # Backend-owned policy: only allow minimal step types.
    # Richness should come from better options + hints, not more component types.
    "early": ["multiple_choice", "slider"],
    "single": ["multiple_choice", "slider"],
    "middle": ["multiple_choice", "slider"],
    "late": ["multiple_choice", "slider"],
}


def allowed_components(stage: str) -> List[str]:
    return list(FLOW_COMPONENTS.get(str(stage or "").strip().lower(), FLOW_COMPONENTS["early"]))


QUESTION_HINTS: dict[str, dict[str, str]] = {
    "early": {"length": "short", "tone": "simple, broad"},
    "single": {"length": "medium", "tone": "simple, quantifying"},
    "middle": {"length": "medium", "tone": "more specific, quantifying"},
    "late": {"length": "long", "tone": "detailed, pointed"},
}


def get_question_hints(stage: str) -> Dict[str, Any]:
    return dict(QUESTION_HINTS.get(str(stage or "").strip().lower(), QUESTION_HINTS["early"]))


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
        n = (DEFAULT_CONSTRAINTS or {}).get("maxBatches")
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
    prefer_structured_inputs = stage in {"early", "single"}

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
    _ = payload  # reserved for future use (caller payload overrides)

    if not isinstance(context, dict):
        context = {}
    guide = flow_guide_for_batch(context=context, batch_number=batch_number)
    context = dict(context)
    context["flow_guide"] = guide
    prefer_structured = bool((guide.get("rules") or {}).get("preferStructuredInputs"))
    context["prefer_structured_inputs"] = prefer_structured

    stage = str(guide.get("stage") or "").strip().lower() or "early"
    stage_allowed = allowed_components(stage)

    allowed = list(extracted_allowed_mini_types or [])
    if not allowed:
        allowed = list((guide.get("rules") or {}).get("allowedMiniTypesDefault") or [])
    # Enforce stage-specific allowed types from `allowed_components()`.
    stage_allowed_set = set([str(t).strip().lower() for t in (stage_allowed or []) if str(t).strip()])
    if stage_allowed:
        allowed = [t for t in allowed if str(t).strip().lower() in stage_allowed_set]
        if not allowed:
            allowed = list(stage_allowed)

    max_steps = int(extracted_max_steps or 0)
    constraints = context.get("batch_constraints") if isinstance(context.get("batch_constraints"), dict) else {}
    min_steps_per_batch = _as_int(constraints.get("minStepsPerBatch"), default=2)
    max_steps_per_batch = _as_int(constraints.get("maxStepsPerBatch"), default=4)
    default_steps_per_batch = _as_int(constraints.get("defaultStepsPerBatch"), default=8)
    if min_steps_per_batch < 1:
        min_steps_per_batch = 2
    if max_steps_per_batch < min_steps_per_batch:
        max_steps_per_batch = min_steps_per_batch
    if default_steps_per_batch < min_steps_per_batch:
        default_steps_per_batch = min_steps_per_batch
    if default_steps_per_batch > max_steps_per_batch:
        default_steps_per_batch = max_steps_per_batch

    if max_steps <= 0:
        max_steps = default_steps_per_batch
    max_steps = max(min_steps_per_batch, min(max_steps, max_steps_per_batch))

    # Keep early batches short by default (while respecting the configured range).
    if stage == "early":
        max_steps = max(min_steps_per_batch, min(max_steps, 5))
    elif stage == "middle":
        max_steps = max(min_steps_per_batch, min(max_steps, 4))
    return context, allowed, max_steps


def _as_required_upload_step_ids(required_uploads: Any) -> List[str]:
    """
    Extract step ids from `required_uploads` while preserving list order.
    """
    if not isinstance(required_uploads, list):
        return []
    ids: List[str] = []
    for item in required_uploads:
        if not isinstance(item, dict):
            continue
        raw = item.get("stepId") or item.get("step_id") or item.get("id")
        sid = str(raw or "").strip()
        if not sid or not sid.startswith("step-"):
            continue
        if sid not in ids:
            ids.append(sid)
    return ids


def _key_from_step_id(step_id: str) -> str:
    """
    Convert `step-foo-bar` -> `foo_bar` so the pipeline’s id derivation
    (`step-` + key with `_` -> `-`) round-trips to the same id.
    """
    t = str(step_id or "").strip()
    if t.startswith("step-"):
        t = t[len("step-") :]
    return t.replace("-", "_").strip("_")


def build_deterministic_suffix_plan_items(*, context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build deterministic plan items for required upload flows.

    NOTE: these are "plan items" (keys + hints), not UI steps.
    """
    ctx = context if isinstance(context, dict) else {}
    required_uploads = ctx.get("required_uploads")
    upload_step_ids = _as_required_upload_step_ids(required_uploads)

    out: List[Dict[str, Any]] = []

    upload_keys: List[str] = []
    if upload_step_ids:
        upload_keys = [_key_from_step_id(sid) for sid in upload_step_ids if _key_from_step_id(sid)]

    if upload_keys:
        for key in upload_keys:
            out.append(
                {
                    "key": key,
                    "deterministic": True,
                    "type_hint": "file_upload",
                    "intent": "Upload an image to continue.",
                    "question": "Upload an image.",
                    "required": True,
                }
            )

        out.append(
            {
                "key": "gallery",
                "deterministic": True,
                "type_hint": "gallery",
                "intent": "Review uploaded images.",
                "question": "Review your images.",
                "required": False,
            }
        )
    return out


__all__ = [
    "DEFAULT_CONSTRAINTS",
    "FLOW_COMPONENTS",
    "QUESTION_HINTS",
    "allowed_components",
    "apply_flow_guide",
    "build_deterministic_suffix_plan_items",
    "flow_guide_for_batch",
    "get_question_hints",
    "resolve_stage",
]

