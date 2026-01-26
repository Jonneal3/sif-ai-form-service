from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import dspy


# Hardcoded, backend-owned defaults for form constraints.
# Keep this section intentionally logic-light.
DEFAULT_CONSTRAINTS = {
    "maxBatches": 3,
    # Keep batches short to reduce variance and improve completion.
    # Use a range so different stages can clamp within it.
    "minStepsPerBatch": 2,
    "maxStepsPerBatch": 4,
    "tokenBudgetTotal": 3000,
}


def resolve_stage(*, batch_index: int, total_batches: int) -> str:
    """
    Returns a stage label: early / middle / late

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

    if total <= 1 or idx <= 0:
        return "early"
    if idx < total - 1:
        return "middle"
    return "late"


FLOW_COMPONENTS: dict[str, list[str]] = {
    # Early = easiest, mostly structured.
    "early": ["multiple_choice"],
    # Middle = add quantifiers/controls.
    "middle": ["multiple_choice", "yes_no", "slider", "range_slider"],
    # Late = allow detail and uploads.
    "late": ["multiple_choice", "yes_no", "slider", "range_slider", "file_upload"],
}


def allowed_components(stage: str) -> List[str]:
    return list(FLOW_COMPONENTS.get(str(stage or "").strip().lower(), FLOW_COMPONENTS["early"]))


QUESTION_HINTS: dict[str, dict[str, str]] = {
    "early": {"length": "short", "tone": "simple, broad"},
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
    - Always set `context[\"flow_guide\"]` so DSPy sees the skeleton.
    - Set `context[\"prefer_structured_inputs\"]` per batch stage.
    - Provide defaults for allowed types + max steps if missing.
    """
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
    # This prevents clients/demos from widening component types beyond the backend-owned flow.
    if stage_allowed:
        allowed = [t for t in allowed if str(t).strip().lower() in set(stage_allowed)]
        if not allowed:
            allowed = list(stage_allowed)

    max_steps = int(extracted_max_steps or 0)
    constraints = context.get("batch_constraints") if isinstance(context.get("batch_constraints"), dict) else {}
    min_steps_per_batch = _as_int(constraints.get("minStepsPerBatch"), default=2)
    max_steps_per_batch = _as_int(constraints.get("maxStepsPerBatch"), default=4)
    if min_steps_per_batch < 1:
        min_steps_per_batch = 2
    if max_steps_per_batch < min_steps_per_batch:
        max_steps_per_batch = min_steps_per_batch

    if max_steps <= 0:
        max_steps = max_steps_per_batch
    # Clamp within the configured range first.
    max_steps = max(min_steps_per_batch, min(max_steps, max_steps_per_batch))

    # Keep early batches short by default (while respecting the configured range).
    if stage == "early":
        max_steps = max(min_steps_per_batch, min(max_steps, 3))
    elif stage == "middle":
        max_steps = max(min_steps_per_batch, min(max_steps, 4))
    return context, allowed, max_steps


def _as_required_upload_step_ids(required_uploads: Any) -> List[str]:
    """
    Extract step ids from `required_uploads` while preserving list order.

    Expected input shape (best-effort):
      [{ "stepId": "step-upload-..." }, ...]
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
    Backend-owned suffix that ensures the form always ends in a predictable way:
      upload -> gallery -> confirmation

    These are "plan items" (keys + hints), not UI steps.
    """
    ctx = context if isinstance(context, dict) else {}
    required_uploads = ctx.get("required_uploads")
    upload_step_ids = _as_required_upload_step_ids(required_uploads)

    upload_keys: List[str] = []
    if upload_step_ids:
        upload_keys = [_key_from_step_id(sid) for sid in upload_step_ids if _key_from_step_id(sid)]
    else:
        upload_keys = ["upload_reference"]

    out: List[Dict[str, Any]] = []

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
    out.append(
        {
            "key": "confirmation",
            "deterministic": True,
            "type_hint": "confirmation",
            "intent": "Finish the form.",
            "question": "All set. Submit when ready.",
            "required": False,
        }
    )

    return out


def load_pack(pack_id: str) -> Dict[str, Any]:
    pid = (pack_id or "").strip() or "default_v1"
    if pid not in {"default_v1"}:
        pid = "default_v1"
    return {
        "pack_id": pid,
        "pack_version": "1",
        "style": {
            "tone": "direct, friendly, professional",
            "question_rules": [
                "Ask one thing at a time.",
                "Use concrete nouns; avoid generic filler.",
                "Avoid parenthetical enumerations when options are present.",
                "Keep questions under ~12 words when possible.",
            ],
            "option_rules": [
                "Use parallel phrasing across options.",
                "Avoid overly broad location lists unless the service is outdoor-specific.",
                "Include 'Not sure' only when it reduces drop-off.",
            ],
        },
        "lint": {
            "require_question_mark": True,
            "max_question_chars": 120,
            "banned_question_substrings": ["(install, replace, repair)"],
        },
    }


def compile_pack(pack: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    style = pack.get("style") if isinstance(pack.get("style"), dict) else {}
    lint = pack.get("lint") if isinstance(pack.get("lint"), dict) else {}
    lint_config: Dict[str, Any] = {
        "pack_id": str(pack.get("pack_id") or "").strip() or "default_v1",
        "pack_version": str(pack.get("pack_version") or "").strip() or "1",
        **lint,
    }
    style_snippet_json = json.dumps(style, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return style_snippet_json, lint_config


def _strip_parenthetical_enumeration(q: str) -> str:
    # Remove trailing "(a, b, c)" style enumerations which duplicate the options list.
    return re.sub(r"\s*\([^)]{0,80}\)\s*$", "", q).strip()


def sanitize_steps(steps: List[dict], lint_config: Dict[str, Any]) -> List[dict]:
    out: List[dict] = []
    require_qmark = bool(lint_config.get("require_question_mark") is True)
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        s = dict(step)
        q = str(s.get("question") or "").strip()
        if q:
            q = _strip_parenthetical_enumeration(q)
            if require_qmark and not q.endswith("?"):
                q = q.rstrip(".").strip()
                if q and not q.endswith("?"):
                    q = f"{q}?"
            s["question"] = q
        out.append(s)
    return out


def apply_reassurance(steps: List[dict], lint_config: Dict[str, Any]) -> List[dict]:
    # Keep this minimal; the UI already carries trust cues.
    return steps


def lint_steps(steps: List[dict], lint_config: Dict[str, Any]) -> Tuple[bool, List[dict], List[str]]:
    violations: List[dict] = []
    bad_ids: List[str] = []

    banned_substrings = lint_config.get("banned_question_substrings") or []
    if not isinstance(banned_substrings, list):
        banned_substrings = []
    max_chars = lint_config.get("max_question_chars")
    try:
        max_chars_i = int(max_chars)
    except Exception:
        max_chars_i = 140
    require_qmark = bool(lint_config.get("require_question_mark") is True)

    for step in steps or []:
        if not isinstance(step, dict):
            continue
        sid = str(step.get("id") or "").strip()
        q = str(step.get("question") or "").strip()
        if not sid:
            violations.append({"code": "missing_id", "message": "Step is missing id"})
            continue
        if not q:
            violations.append({"code": "missing_question", "message": f"{sid}: missing question"})
            bad_ids.append(sid)
            continue
        if require_qmark and not q.endswith("?"):
            violations.append({"code": "question_no_qmark", "message": f"{sid}: question should end with '?'"})
        if len(q) > max_chars_i:
            violations.append({"code": "question_too_long", "message": f"{sid}: question too long ({len(q)} chars)"})
        q_lower = q.lower()
        for sub in banned_substrings:
            t = str(sub or "").strip().lower()
            if t and t in q_lower:
                violations.append({"code": "banned_phrase", "message": f"{sid}: contains banned phrase '{sub}'"})

    ok = len(violations) == 0
    return ok, violations, bad_ids


class _MustHaveCopySignature(dspy.Signature):
    context_json: str = dspy.InputField(desc="Compact JSON context")
    mini_steps_jsonl: str = dspy.InputField(desc="The UI steps JSONL for this batch")

    must_have_copy_json: str = dspy.OutputField(desc="JSON string of required copy fields")


class MustHaveCopyModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(_MustHaveCopySignature)

    def forward(self, *, context_json: str, mini_steps_jsonl: str):  # type: ignore[override]
        return self.prog(context_json=context_json, mini_steps_jsonl=mini_steps_jsonl)


__all__ = [
    "DEFAULT_CONSTRAINTS",
    "FLOW_COMPONENTS",
    "QUESTION_HINTS",
    "MustHaveCopyModule",
    "allowed_components",
    "apply_flow_guide",
    "apply_reassurance",
    "build_deterministic_suffix_plan_items",
    "compile_pack",
    "flow_guide_for_batch",
    "get_question_hints",
    "lint_steps",
    "load_pack",
    "resolve_stage",
    "sanitize_steps",
]

