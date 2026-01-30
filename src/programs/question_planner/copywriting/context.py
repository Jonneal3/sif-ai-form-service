from __future__ import annotations

import json
from typing import Any, Dict


DEFAULT_COPY_CONTEXT: Dict[str, Any] = {
    # -----------------------------
    # Voice & emotional grounding
    # -----------------------------

    # "Brand voice" presets are intentionally coarse; the planner should still write naturally.
    # Suggested values: "friendly_expert", "luxury_minimal", "playful", "clinical", "bold_direct".
    "brand_voice": "friendly_expert",

    # User state at the time we ask questions.
    # Suggested values: "curious", "cautious", "overwhelmed", "urgent", "skeptical".
    "user_state": "curious",

    # Where we are on the commitment ladder (1..6).
    # 1=safety, 2=momentum, 3=diagnosis, 4=reflection, 5=identity, 6=payoff.
    "commitment_stage": 2,

    # -----------------------------
    # Persuasion & trust mechanics
    # -----------------------------

    # Should we explicitly preempt common objections in copy?
    # (e.g. "no spam", "takes <1 minute")
    "objection_preemption": True,

    # How much explanation the user tolerates for *why* we ask a question.
    # Suggested values: "none", "low", "medium".
    "justification_tolerance": "low",

    # Whether to preserve explicit user agency on higher-friction questions.
    # Suggested values: "none", "implicit", "explicit".
    "escape_hatch_policy": "implicit",

    # Risk tolerance for asking higher-friction / identity-adjacent questions.
    # Suggested values: "low", "medium", "high".
    "risk_posture": "low",

    # Sensitivity awareness for copy tone.
    # Suggested values: "low", "medium", "high".
    # High = avoid playfulness, humor, or clever phrasing.
    "sensitivity_level": "low",

    # -----------------------------
    # Gamification & momentum
    # -----------------------------

    # Whether the product can be playful in microcopy (NOT in sensitive moments).
    "playfulness_level": "low",  # "low" | "medium" | "high"

    # How to format progress cues (if any) in question text.
    "progress_style": "none",  # "none" | "step_of_total"

    # Narrative framing for progress (use sparingly; keep copy natural).
    # Suggested values: "momentum", "relief", "clarity", "reward".
    "progress_narrative": "momentum",

    # -----------------------------
    # Question strategy
    # -----------------------------

    # Strategy hints: the planner should implicitly assign each question a "job".
    # Suggested values per question:
    # "warmup", "classification", "constraint", "personalization",
    # "prioritization", "confidence_building", "commitment_lock", "motivation".
    "question_intent_palette": [
        "warmup",
        "classification",
        "constraint",
        "personalization",
        "prioritization",
        "confidence_building",
        "commitment_lock",
        "motivation",
    ],

    # Answer-effort preference for early questions (tap > short_text > long_text).
    # Suggested values: "tap", "short_text", "long_text", "thinking", "emotional".
    "answer_effort_preference": "tap",

    # Keep questions short unless a narrowing/constraints question requires more detail.
    "max_question_words_soft": 18,

    # Required phrasing constraints
    "must_be_a_question": True,
}


def normalize_copy_context(raw: Any) -> Dict[str, Any]:
    """
    Normalize a caller-provided copy context to a safe, compact dict.
    """
    if raw is None:
        return dict(DEFAULT_COPY_CONTEXT)

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        raw = parsed if isinstance(parsed, dict) else None

    if not isinstance(raw, dict):
        return dict(DEFAULT_COPY_CONTEXT)

    out = dict(DEFAULT_COPY_CONTEXT)
    for k, v in raw.items():
        if k in out:
            out[k] = v
    return out


def build_copy_context(*, payload: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build `copy_context` for the planner.

    This is NOT a scoring heuristic; it's a prompt-conditioning context
    that helps the planner produce better question wording (copy).
    """
    raw = (
        payload.get("copyContext")
        or payload.get("copy_context")
        or payload.get("promptContext")
        or payload.get("prompt_context")
    )
    base = normalize_copy_context(raw)

    # -----------------------------
    # Commitment progression
    # -----------------------------
    try:
        asked = ctx.get("asked_step_ids") if isinstance(ctx.get("asked_step_ids"), list) else []
        asked_count = len([x for x in asked if str(x or "").strip()])
    except Exception:
        asked_count = 0

    if asked_count >= 6:
        base["commitment_stage"] = max(int(base.get("commitment_stage") or 2), 4)
    elif asked_count >= 3:
        base["commitment_stage"] = max(int(base.get("commitment_stage") or 2), 3)

    stage = int(base.get("commitment_stage") or 2)

    # -----------------------------
    # Adaptive tolerance rules
    # -----------------------------

    # As commitment increases, we can tolerate more effort and risk.
    if stage >= 5:
        base["risk_posture"] = (
            "medium" if base.get("risk_posture") == "low" else base["risk_posture"]
        )
        base["answer_effort_preference"] = (
            "short_text"
            if base.get("answer_effort_preference") == "tap"
            else base["answer_effort_preference"]
        )
        base["progress_narrative"] = "reward"
    elif stage >= 4:
        base["progress_narrative"] = "clarity"

    # High sensitivity suppresses playfulness automatically.
    if base.get("sensitivity_level") == "high":
        base["playfulness_level"] = "low"
        base["escape_hatch_policy"] = "explicit"

    # -----------------------------
    # Goal-aware tuning
    # -----------------------------
    goal_intent = str(
        payload.get("goalIntent")
        or payload.get("goal_intent")
        or ctx.get("goal_intent")
        or ""
    ).strip().lower()

    if goal_intent == "visual":
        # Visual flows benefit from momentum and confidence.
        base["progress_narrative"] = "momentum"
    elif goal_intent == "pricing":
        # Pricing flows benefit from clarity and reassurance.
        if stage >= 3:
            base["progress_narrative"] = "clarity"
            base["justification_tolerance"] = "medium"

    return base


__all__ = ["DEFAULT_COPY_CONTEXT", "normalize_copy_context", "build_copy_context"]
