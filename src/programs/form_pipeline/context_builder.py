from __future__ import annotations

import random
from typing import Any, Dict

from programs.form_pipeline.constraints import (
    build_batch_constraints,
    extract_form_state_subset,
    resolve_backend_max_calls,
)
from programs.form_pipeline.payload_extractors import extract_answered_qa, extract_asked_step_ids, extract_use_case
from programs.form_pipeline.service_context import (
    derive_industry_and_service_strings,
    extract_company_summary,
    extract_service_summary,
    infer_goal_intent,
)


def build_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the minimal context dict used by the form pipeline.

    Assumes a single modern payload shape (top-level fields, minimal camel/snake aliasing).
    """

    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}

    required_uploads_raw = payload.get("requiredUploads") or payload.get("required_uploads") or current_batch.get("requiredUploads") or []
    required_uploads = required_uploads_raw if isinstance(required_uploads_raw, list) else []

    answered_qa = extract_answered_qa(payload)
    asked_step_ids = extract_asked_step_ids(payload, answered_qa=answered_qa)

    batch_state_raw = payload.get("batchState") or payload.get("batch_state") or {}
    batch_state = batch_state_raw if isinstance(batch_state_raw, dict) else {}

    use_case = extract_use_case(payload)

    # Frontend-provided sources of truth (preferred).
    service_summary = extract_service_summary(payload)
    company_summary = extract_company_summary(payload)
    if service_summary:
        service_summary = service_summary[:1200]
    if company_summary:
        company_summary = company_summary[:1200]

    # Back-compat internal naming: most prompts still refer to `services_summary`.
    services_summary = service_summary[:600] if service_summary else ""

    industry, service = derive_industry_and_service_strings(payload)
    goal_intent = infer_goal_intent(
        services_summary=services_summary,
        explicit_goal_intent=str(payload.get("goalIntent") or payload.get("goal_intent") or ""),
    )

    model_batch = extract_form_state_subset(payload, batch_state)
    backend_max_calls = resolve_backend_max_calls()
    model_batch = dict(model_batch)
    model_batch["max_batches"] = backend_max_calls
    batch_constraints = build_batch_constraints(payload=payload, batch_state=batch_state, max_batches=backend_max_calls)

    # Choice option bounds (UI-only hinting)
    choice_option_min = 4
    choice_option_max = 10
    try:
        raw_min = payload.get("choiceOptionMin") or payload.get("choice_option_min")
        raw_max = payload.get("choiceOptionMax") or payload.get("choice_option_max")
        if raw_min is not None:
            choice_option_min = max(2, min(12, int(raw_min)))
        if raw_max is not None:
            choice_option_max = max(choice_option_min, min(12, int(raw_max)))
    except Exception:
        choice_option_min, choice_option_max = 4, 10

    choice_option_target = None
    try:
        raw_target = payload.get("choiceOptionTarget") or payload.get("choice_option_target")
        if raw_target is not None:
            choice_option_target = int(raw_target)
    except Exception:
        choice_option_target = None
    if not isinstance(choice_option_target, int) or not (choice_option_min <= choice_option_target <= choice_option_max):
        choice_option_target = random.randint(choice_option_min, choice_option_max)

    ctx: Dict[str, Any] = {
        # Service context
        "industry": industry,
        "service": service,
        "services_summary": services_summary or "",
        "service_summary": service_summary or "",
        "company_summary": company_summary or "",
        # Back-compat internal alias
        "grounding_summary": services_summary or "",
        # Memory for dedupe/continuity
        "answered_qa": answered_qa,
        "asked_step_ids": asked_step_ids,
        "already_asked_keys": asked_step_ids,
        # Server-side flow/enforcement
        "use_case": use_case,
        "goal_intent": goal_intent,
        "required_uploads": required_uploads,
        "batch_info": model_batch,
        "batch_constraints": batch_constraints,
        "batch_state": batch_state,
        "choice_option_min": choice_option_min,
        "choice_option_max": choice_option_max,
        "choice_option_target": choice_option_target,
        "prefer_structured_inputs": False,
    }

    if services_summary:
        ctx["vertical_context"] = services_summary

    return ctx


__all__ = ["build_context"]

