from __future__ import annotations

"""
Back-compat module.

Historically we used the word "policy" for what is effectively a lightweight form skeleton / batch guide.
New code should prefer `app.form_planning.guides`.
"""

from app.form_planning.guides import (  # noqa: F401
    default_batch_policy,
    default_form_psychology_guide,
    default_form_skeleton,
    default_psychology_plan,
    normalize_form_skeleton,
    normalize_policy,
    parse_batch_policy_json,
    parse_form_skeleton_json,
    parse_psychology_plan_json,
    policy_for_batch,
    policy_for_phase,
    skeleton_for_batch,
    skeleton_for_phase,
)
