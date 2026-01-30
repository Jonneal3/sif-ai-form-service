"""
Form pipeline helper: build the model-facing `context` dict from request payload.

Used by the orchestrator to create the compact planner/render context:
- service context (industry/service + summaries)
- memory (answered_qa, asked_step_ids)
- batch constraints + UI hint bounds (choice option count targets)

This module consolidates the old `service_context.py` helpers as well, since
they are only used here.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, Tuple

from programs.form_pipeline.constraints import (
    build_batch_constraints,
    extract_form_state_subset,
    resolve_backend_max_calls,
)
from api.payload_extractors import extract_answered_qa, extract_asked_step_ids
from programs.question_planner.copywriting import build_copy_context


def derive_industry_and_service_strings(
    payload: Dict[str, Any],
    *,
    max_len: int = 120,
) -> Tuple[str, str]:
    """
    Derive short, plain-English industry/service strings.

    Modern shape: top-level `industry` and `service` (plus `vertical` alias for industry).
    """
    industry = str(payload.get("industry") or payload.get("vertical") or "").strip()
    service = str(payload.get("service") or "").strip()

    # Widget/contract shape: nested instanceContext objects.
    if (not industry or not service) and isinstance(payload.get("instanceContext"), dict):
        ctx = payload.get("instanceContext") or {}
        if isinstance(ctx, dict):
            if not industry:
                raw_ind = ctx.get("industry") or (
                    ctx.get("categories")[0] if isinstance(ctx.get("categories"), list) and ctx.get("categories") else None
                )
                if isinstance(raw_ind, dict):
                    industry = str(raw_ind.get("name") or raw_ind.get("label") or raw_ind.get("id") or "").strip()
                else:
                    industry = str(raw_ind or "").strip()
            if not service:
                raw_svc = ctx.get("service") or (
                    ctx.get("subcategories")[0] if isinstance(ctx.get("subcategories"), list) and ctx.get("subcategories") else None
                )
                if isinstance(raw_svc, dict):
                    service = str(raw_svc.get("name") or raw_svc.get("label") or raw_svc.get("id") or "").strip()
                else:
                    service = str(raw_svc or "").strip()

    # Snake_case alias
    if (not industry or not service) and isinstance(payload.get("instance_context"), dict):
        ctx = payload.get("instance_context") or {}
        if isinstance(ctx, dict):
            if not industry:
                raw_ind = ctx.get("industry") or (
                    ctx.get("categories")[0] if isinstance(ctx.get("categories"), list) and ctx.get("categories") else None
                )
                if isinstance(raw_ind, dict):
                    industry = str(raw_ind.get("name") or raw_ind.get("label") or raw_ind.get("id") or "").strip()
                else:
                    industry = str(raw_ind or "").strip()
            if not service:
                raw_svc = ctx.get("service") or (
                    ctx.get("subcategories")[0] if isinstance(ctx.get("subcategories"), list) and ctx.get("subcategories") else None
                )
                if isinstance(raw_svc, dict):
                    service = str(raw_svc.get("name") or raw_svc.get("label") or raw_svc.get("id") or "").strip()
                else:
                    service = str(raw_svc or "").strip()

    industry = industry[:max_len]
    service = service[:max_len]
    return industry, service


def _coerce_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (dict, list)):
        try:
            raw = json.dumps(raw, ensure_ascii=True)
        except Exception:
            raw = str(raw)
    return str(raw).strip()


def extract_service_summary(payload: Dict[str, Any], *, max_len: int = 1200) -> str:
    """
    Extract a plain-text service summary from a request payload.

    Accepted keys (new + back-compat):
      - `service_summary` / `serviceSummary` (preferred)
      - `services_summary` / `servicesSummary` (legacy)
    """
    text = _coerce_text(
        payload.get("service_summary")
        or payload.get("serviceSummary")
        or payload.get("services_summary")
        or payload.get("servicesSummary")
    )
    if text:
        return text[: int(max_len or 0) or 1200]
    return ""


def extract_company_summary(payload: Dict[str, Any], *, max_len: int = 1200) -> str:
    """
    Extract a plain-text company summary from a request payload.

    Accepted keys:
      - `company_summary` / `companySummary`
    """
    text = _coerce_text(payload.get("company_summary") or payload.get("companySummary"))
    if text:
        return text[: int(max_len or 0) or 1200]
    return ""


def infer_goal_intent(*, services_summary: str = "", explicit_goal_intent: str = "") -> str:
    """
    Decide between "pricing" vs "visual" intent.

    Prefer explicit `goal_intent` from payload; otherwise default to "pricing".
    """
    _ = services_summary  # reserved for future heuristic scoring
    t = str(explicit_goal_intent or "").strip().lower()
    if t in {"pricing", "visual"}:
        return t
    return "pricing"


def build_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the minimal context dict used by the form pipeline.

    Assumes a single modern payload shape (top-level fields, minimal camel/snake aliasing).
    """
    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}

    required_uploads_raw = (
        payload.get("requiredUploads") or payload.get("required_uploads") or current_batch.get("requiredUploads") or []
    )
    required_uploads = required_uploads_raw if isinstance(required_uploads_raw, list) else []

    answered_qa = extract_answered_qa(payload)
    asked_step_ids = extract_asked_step_ids(payload, answered_qa=answered_qa)

    batch_state_raw = payload.get("batchState") or payload.get("batch_state") or {}
    batch_state = batch_state_raw if isinstance(batch_state_raw, dict) else {}

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

    # Copywriting/Form-Intelligence prompt conditioning for the planner.
    #
    # This is NOT used as a heuristic scoring system. It's a compact prompt context
    # that shapes question wording (tone, reassurance, momentum, etc.).
    try:
        ctx["copy_context"] = build_copy_context(payload=payload, ctx=ctx)
    except Exception:
        ctx["copy_context"] = {}

    return ctx


__all__ = [
    "build_context",
    # exported for reuse/testing (formerly in service_context.py)
    "derive_industry_and_service_strings",
    "extract_company_summary",
    "extract_service_summary",
    "infer_goal_intent",
]
