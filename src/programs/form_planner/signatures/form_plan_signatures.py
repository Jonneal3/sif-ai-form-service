from __future__ import annotations

import dspy


class FormPlanJSON(dspy.Signature):
    """
    Generate a form-level plan as JSON.

    This is intentionally separate from batch generation:
    - Form planning decides *what to collect over time* (priorities, families, rationale).
    - Batch generation decides *what fits in the current batch* given constraints/state.

    Expected shapes:
    - `form_plan_json`: JSON array of plan items (what information we want to collect over time).
    - `batch_policy_json`: JSON object describing the batch skeleton (how to group/sequence collection).

    `batch_policy_json` should be shaped like:
    {
      "v": 1,
      "phases": [
        {
          "phaseId": "ContextCore",
          "title": "Quick Basics",
          "purpose": "Low-friction, high-signal context capture.",
          "maxSteps": 4,
          "allowedMiniTypes": ["choice","chips_multi","slider","text"],
          "rigidity": 0.9,
          "focusKeys": ["project_type","area_location","timeline_urgency"]
        }
      ],
      "stop": {"requiredKeysComplete": [], "satietyTarget": 0.85}
    }

    Hard rules:
    - Output MUST be JSON ONLY in `form_plan_json` and `batch_policy_json` (no prose, no markdown).
    - `form_plan_json` MUST parse as a JSON array of objects.
    - `batch_policy_json` MUST parse as a JSON object (or `{}` if unknown).
    """

    context_json: str = dspy.InputField(
        desc=(
            "Compact planning context as JSON string. Includes platform goal, business context, "
            "industry/service (+ optional grounding/RAG), current form state, and any constraints/policies."
        )
    )
    current_phase_id: str = dspy.InputField(desc="Stable phase/batch identifier for the current request")

    form_plan_json: str = dspy.OutputField(
        desc=(
            "JSON array string of plan items. Each item should include: "
            "`key`, `goal`, `why`, `priority`, `component_hint`, `importance_weight`, `expected_metric_gain`."
        )
    )
    batch_policy_json: str = dspy.OutputField(
        desc=(
            "JSON object string describing the batch skeleton/policy (or `{}` if not applicable). "
            "Prefer `phases: [{phaseId,title,purpose,maxSteps,allowedMiniTypes,rigidity,focusKeys}]`."
        )
    )


__all__ = ["FormPlanJSON"]
