from __future__ import annotations

from typing import Any, Dict, List, Optional


def _best_effort_json(text: Any) -> Any:
    from sif_ai_form_service.pipeline.form_pipeline import _best_effort_parse_json  # local import to avoid cycles at import-time

    return _best_effort_parse_json(str(text or ""))


def default_form_psychology_guide(*, goal_intent: str, use_case: str) -> Dict[str, Any]:
    """
    Compact "form psychology" guide for the UI + DSPy context.
    Keep this small: it's guidance, not a full rubric.
    """
    goal = str(goal_intent or "pricing").strip().lower()
    use = str(use_case or "scene").strip().lower()
    stages: List[Dict[str, Any]] = [
        {"id": "scope", "goal": "Clarify scope and intent", "rules": ["Prefer choice over text early"]},
        {"id": "quantifiers", "goal": "Add measurements, budget, timeline", "rules": ["Use slider/range when possible"]},
        {"id": "constraints", "goal": "Capture constraints that change outcome", "rules": ["Avoid vague 'anything else'"]},
        {"id": "preferences", "goal": "Capture style/material prefs if impactful", "rules": ["Keep options grounded"]},
        {"id": "commitment", "goal": "Collect final required details", "rules": ["Minimize friction; bundle deterministic UI"]},
    ]
    if goal != "pricing":
        stages[1]["goal"] = "Add specificity for visuals"
    if "tryon" in use:
        stages[0]["rules"] = ["Ask for subject photo early (deterministic)"]
    return {"v": 1, "approach": "escalation_ladder", "stages": stages}


def default_form_skeleton(*, goal_intent: str, max_calls: int | None = None) -> Dict[str, Any]:
    """
    Default form "skeleton"/guide for multi-batch planning.

    This is intentionally simple: it describes the batch phases, per-batch constraints, and when to stop.
    External payloads still call this a `batchPolicy`, but internally it's closer to a form skeleton.
    """
    _ = goal_intent  # reserved for future specialization
    phases: List[Dict[str, Any]] = [
        {
            "id": "ContextCore",
            "purpose": "Quick, low-friction context capture (scope + intent).",
            "maxSteps": 5,
            "allowedMiniTypes": ["choice"],
            "rigidity": 0.95,
            "focusKeys": [],
        },
        {
            "id": "Details",
            "purpose": "Quantify scope, size, budget, timeline, and constraints.",
            "maxSteps": 10,
            "allowedMiniTypes": ["choice", "slider"],
            "rigidity": 0.6,
            "focusKeys": [],
        },
        {
            "id": "Preferences",
            "purpose": "Capture style/material preferences that change the output.",
            "maxSteps": 8,
            "allowedMiniTypes": ["choice", "slider"],
            "rigidity": 0.4,
            "focusKeys": [],
        },
    ]
    if isinstance(max_calls, int) and max_calls > 0:
        phases = phases[: max(1, min(len(phases), max_calls))]
    return {
        "v": 1,
        # For N batches: planner should set maxCalls to len(phases) or a hard cap.
        "maxCalls": len(phases),
        # Back-compat (older consumers)
        "maxStepsPerCall": {p["id"]: p.get("maxSteps") for p in phases},
        "allowedMiniTypes": {p["id"]: p.get("allowedMiniTypes") for p in phases},
        # Preferred (N-batch)
        "phases": phases,
        "deterministic": {
            "uploads": "composite_or_ui_plan",
        },
        "stopConditions": {
            "requiredKeysComplete": True,
            "satietyTarget": 1.0,
        },
    }


def skeleton_for_phase(form_skeleton: Dict[str, Any], phase: str) -> Dict[str, Any]:
    allowed = form_skeleton.get("allowedMiniTypes") if isinstance(form_skeleton, dict) else None
    max_steps = form_skeleton.get("maxStepsPerCall") if isinstance(form_skeleton, dict) else None
    phases = form_skeleton.get("phases") if isinstance(form_skeleton, dict) else None
    if isinstance(phases, list):
        for p in phases:
            if not isinstance(p, dict):
                continue
            if str(p.get("id") or "").strip() == str(phase or "").strip():
                return {
                    "allowedMiniTypes": p.get("allowedMiniTypes"),
                    "maxSteps": p.get("maxSteps"),
                    "maxCalls": form_skeleton.get("maxCalls"),
                    "rigidity": p.get("rigidity"),
                    "purpose": p.get("purpose"),
                    "focusKeys": p.get("focusKeys"),
                }
    return {
        "allowedMiniTypes": (allowed.get(phase) if isinstance(allowed, dict) else None),
        "maxSteps": (max_steps.get(phase) if isinstance(max_steps, dict) else None),
        "maxCalls": form_skeleton.get("maxCalls") if isinstance(form_skeleton, dict) else None,
    }


def skeleton_for_batch(form_skeleton: Dict[str, Any], batch_number: int) -> Dict[str, Any]:
    """
    Resolve the active phase using 1-based batch_number.
    Falls back to ContextCore/PersonalGuide for older skeleton/policy shapes.
    """
    phases = form_skeleton.get("phases") if isinstance(form_skeleton, dict) else None
    if isinstance(phases, list) and phases:
        idx = max(0, int(batch_number) - 1)
        idx = min(idx, len(phases) - 1)
        phase_id = str((phases[idx] or {}).get("id") or "").strip() or "ContextCore"
        c = skeleton_for_phase(form_skeleton, phase_id)
        return dict(c, phaseId=phase_id)
    # Back-compat
    phase_id = "ContextCore" if int(batch_number) <= 1 else "PersonalGuide"
    c = skeleton_for_phase(form_skeleton, phase_id)
    return dict(c, phaseId=phase_id)


def normalize_form_skeleton(form_skeleton: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(form_skeleton, dict) or not form_skeleton:
        return None
    # Shallow validation / normalization only (fail-open).
    v = form_skeleton.get("v") or 1
    try:
        v = int(v)
    except Exception:
        v = 1
    out = dict(form_skeleton)
    out["v"] = v
    phases = out.get("phases")
    if isinstance(phases, list):
        cleaned = []
        for p in phases:
            if not isinstance(p, dict):
                continue
            pid = str(p.get("id") or "").strip()
            if not pid:
                continue
            rigidity = p.get("rigidity")
            try:
                rigidity_f = float(rigidity) if rigidity is not None else None
            except Exception:
                rigidity_f = None
            if rigidity_f is not None:
                rigidity_f = max(0.0, min(1.0, rigidity_f))
            cleaned.append(
                {
                    "id": pid,
                    "purpose": str(p.get("purpose") or "").strip() or None,
                    "maxSteps": p.get("maxSteps"),
                    "allowedMiniTypes": p.get("allowedMiniTypes"),
                    "rigidity": rigidity_f,
                    "focusKeys": p.get("focusKeys") if isinstance(p.get("focusKeys"), list) else None,
                }
            )
        out["phases"] = cleaned
    return out


def parse_form_skeleton_json(text: Any) -> Optional[Dict[str, Any]]:
    obj = _best_effort_json(text)
    if not isinstance(obj, dict):
        return None
    return normalize_form_skeleton(obj)


def parse_psychology_plan_json(text: Any) -> Optional[Dict[str, Any]]:
    obj = _best_effort_json(text)
    if not isinstance(obj, dict):
        return None
    v = obj.get("v") or 1
    try:
        obj["v"] = int(v)
    except Exception:
        obj["v"] = 1
    return obj


# Back-compat aliases (older code/tests use "policy" terminology).
default_psychology_plan = default_form_psychology_guide
default_batch_policy = default_form_skeleton
policy_for_phase = skeleton_for_phase
policy_for_batch = skeleton_for_batch
normalize_policy = normalize_form_skeleton
parse_batch_policy_json = parse_form_skeleton_json
