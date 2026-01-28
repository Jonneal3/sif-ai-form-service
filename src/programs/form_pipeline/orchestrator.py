"""
Form pipeline orchestrator (Planner -> Renderer).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from programs.common.dspy_runtime import configure_dspy, extract_dspy_usage, make_dspy_lm_for_module
from programs.common.env import env_bool, env_float, env_int
from programs.common.hashing import short_hash
from programs.common.ttl_cache import ttl_cache_get, ttl_cache_set
from programs.form_pipeline.allowed_types import (
    allowed_type_matches,
    ensure_allowed_mini_types,
    extract_allowed_mini_types_from_payload,
    prefer_structured_allowed_mini_types,
)
from programs.form_pipeline.capabilities import compute_capabilities
from programs.form_pipeline.context_builder import build_context
from programs.form_pipeline.constraints import extract_token_budget
from programs.form_pipeline.payload_extractors import extract_session_id
from programs.form_pipeline.planning import sanitize_steps
from programs.form_pipeline.utils import _compact_json
from programs.question_planner.cache import planner_cache_key
from programs.question_planner.plan_parsing import derive_step_id_from_key, extract_plan_items, normalize_plan_key
from programs.question_planner.program import QuestionPlannerProgram
from programs.renderer.cache import render_cache_key
from programs.renderer.jsonl import parse_jsonl_steps
from programs.renderer.program import RendererProgram

from programs.form_pipeline.validation import (
    _extract_required_upload_ids,
    _looks_like_upload_step_id,
    _reject_banned_option_sets,
    _validate_mini,
)


# Suppress Pydantic serialization warnings from LiteLLM
warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
    module="pydantic",
)


_PLANNER_PLAN_CACHE: dict[str, tuple[float, str]] = {}
_RENDER_OUTPUT_CACHE: dict[str, tuple[float, List[Dict[str, Any]]]] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _best_effort_contract_schema_version() -> str:
    try:
        p_new = _repo_root() / "shared" / "ai-form-ui-contract" / "schema" / "schema_version.txt"
        if p_new.exists():
            v = p_new.read_text(encoding="utf-8").strip()
            return v or "0"
        p_old = _repo_root() / "shared" / "ai-form-contract" / "schema" / "schema_version.txt"
        if p_old.exists():
            v = p_old.read_text(encoding="utf-8").strip()
            return v or "0"
    except Exception:
        pass
    return "0"


def _make_dspy_lm() -> Optional[Dict[str, str]]:
    """
    Return a LiteLLM model string for DSPy v3 (provider-prefixed), or None if not configured.
    """
    # Legacy behavior: use the planner env prefix and keep the small-model guard on.
    return make_dspy_lm_for_module(module_env_prefix="DSPY_PLANNER", allow_small_models=False)


def _configure_dspy(lm: Any) -> bool:
    return configure_dspy(lm)


# Back-compat for scripts importing cache key helpers from this module.
def _planner_cache_key(*, session_id: str, services_fingerprint: str, use_case_key: str) -> str:
    return planner_cache_key(session_id=session_id, services_fingerprint=services_fingerprint, use_case_key=use_case_key)


def _render_cache_key(
    *,
    session_id: str,
    schema_version: str,
    plan_json: str,
    render_context_json: str,
    allowed_mini_types: List[str],
) -> str:
    return render_cache_key(
        session_id=session_id,
        schema_version=schema_version,
        plan_json=plan_json,
        render_context_json=render_context_json,
        allowed_mini_types=allowed_mini_types,
    )


def _include_response_meta(payload: Dict[str, Any]) -> bool:
    if os.getenv("AI_FORM_INCLUDE_META") == "true":
        return True
    req = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    return bool(req.get("includeMeta") is True or str(req.get("includeMeta") or "").lower() == "true")


def _print_lm_history_if_available(lm: Any, n: int = 1) -> None:
    try:
        inspect_fn = getattr(lm, "inspect_history", None)
        if not callable(inspect_fn):
            return
        with contextlib.redirect_stdout(sys.stderr):
            inspect_fn(n=n)
    except Exception:
        return


def _resolve_max_plan_items(ctx: Dict[str, Any]) -> int:
    constraints = ctx.get("batch_constraints") if isinstance(ctx.get("batch_constraints"), dict) else {}
    raw = constraints.get("maxStepsTotal") or constraints.get("max_steps_total")
    try:
        n = int(raw) if raw is not None else 0
    except Exception:
        n = 0
    n = max(4, min(30, int(n or 12)))
    return n


def _select_ui_types() -> Dict[str, Any]:
    from schemas.ui_steps import (
        BudgetCardsUI,
        ColorPickerUI,
        CompositeUI,
        ConfirmationUI,
        DatePickerUI,
        DesignerUI,
        FileUploadUI,
        GalleryUI,
        IntroUI,
        LeadCaptureUI,
        MultipleChoiceUI,
        PricingUI,
        RatingUI,
        SearchableSelectUI,
        TextInputUI,
    )

    return {
        "BudgetCardsUI": BudgetCardsUI,
        "ColorPickerUI": ColorPickerUI,
        "CompositeUI": CompositeUI,
        "ConfirmationUI": ConfirmationUI,
        "DatePickerUI": DatePickerUI,
        "DesignerUI": DesignerUI,
        "FileUploadUI": FileUploadUI,
        "GalleryUI": GalleryUI,
        "IntroUI": IntroUI,
        "LeadCaptureUI": LeadCaptureUI,
        "MultipleChoiceUI": MultipleChoiceUI,
        "PricingUI": PricingUI,
        "RatingUI": RatingUI,
        "SearchableSelectUI": SearchableSelectUI,
        "TextInputUI": TextInputUI,
    }


def _build_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    return build_context(payload)


def next_steps_jsonl(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate the next UI steps as `miniSteps[]` via Planner -> Renderer.
    """

    request_id = f"next_steps_{int(time.time() * 1000)}"
    start_time = time.time()
    t_planner_ms = 0
    t_renderer_ms = 0
    t_post_ms = 0

    schema_version = payload.get("schemaVersion") or payload.get("schema_version") or _best_effort_contract_schema_version()

    planner_lm_cfg = make_dspy_lm_for_module(module_env_prefix="DSPY_PLANNER", allow_small_models=False)
    renderer_lm_cfg = make_dspy_lm_for_module(module_env_prefix="DSPY_RENDERER", allow_small_models=True)
    if not planner_lm_cfg or not renderer_lm_cfg:
        return {"ok": False, "error": "DSPy LM not configured", "requestId": request_id, "schemaVersion": str(schema_version or "0")}

    try:
        import dspy  # type: ignore
    except Exception:
        return {"ok": False, "error": "DSPy import failed", "requestId": request_id, "schemaVersion": str(schema_version or "0")}

    # Token budget guard (best-effort).
    # We treat the caller-provided budget as *soft*: allow a small overage instead of hard-failing
    # exactly at 0, since token accounting is approximate and may drift between client/server.
    batch_state_raw = payload.get("batchState") or payload.get("batch_state") or {}
    tokens_total, tokens_used = extract_token_budget(batch_state_raw)
    token_budget_total: Optional[int] = None
    token_budget_used: Optional[int] = None
    token_budget_remaining: Optional[int] = None
    token_budget_soft_exceeded = False
    if isinstance(tokens_total, int) and tokens_total > 0:
        used_i = tokens_used if isinstance(tokens_used, int) and tokens_used >= 0 else 0
        remaining = int(tokens_total) - int(used_i)
        token_budget_total = int(tokens_total)
        token_budget_used = int(used_i)
        token_budget_remaining = int(remaining)
        if remaining <= 0:
            # Allow a small overage window; beyond that, stop early.
            allowed_overage = env_int("AI_FORM_TOKEN_BUDGET_ALLOWED_OVERAGE", 750)
            if remaining < -int(allowed_overage):
                return {
                    "ok": False,
                    "error": "Token budget exhausted",
                    "requestId": request_id,
                    "schemaVersion": str(schema_version or "0"),
                }
            token_budget_soft_exceeded = True

    default_timeout = env_float("DSPY_LLM_TIMEOUT_SEC", 20.0)
    default_temperature = env_float("DSPY_TEMPERATURE", 0.7)
    default_max_tokens = env_int("DSPY_NEXT_STEPS_MAX_TOKENS", 2000)

    planner_timeout = env_float("DSPY_PLANNER_TIMEOUT_SEC", default_timeout)
    planner_temperature = env_float("DSPY_PLANNER_TEMPERATURE", default_temperature)
    planner_max_tokens = env_int("DSPY_PLANNER_MAX_TOKENS", default_max_tokens)

    renderer_timeout = env_float("DSPY_RENDERER_TIMEOUT_SEC", default_timeout)
    renderer_temperature = env_float("DSPY_RENDERER_TEMPERATURE", default_temperature)
    renderer_max_tokens = env_int("DSPY_RENDERER_MAX_TOKENS", default_max_tokens)

    planner_lm = dspy.LM(
        model=planner_lm_cfg["model"],
        temperature=planner_temperature,
        max_tokens=planner_max_tokens,
        timeout=planner_timeout,
        num_retries=0,
    )
    renderer_lm = dspy.LM(
        model=renderer_lm_cfg["model"],
        temperature=renderer_temperature,
        max_tokens=renderer_max_tokens,
        timeout=renderer_timeout,
        num_retries=0,
    )
    track_usage = False

    # Build context (copy packs removed)
    ctx = _build_context(payload)
    lint_config: Dict[str, Any] = {}

    step_data_so_far_raw = payload.get("stepDataSoFar") or payload.get("step_data_so_far") or {}
    step_data_so_far = step_data_so_far_raw if isinstance(step_data_so_far_raw, dict) else {}
    previous_caps = step_data_so_far.get("__capabilities") if isinstance(step_data_so_far.get("__capabilities"), dict) else {}
    capabilities = compute_capabilities(step_data_so_far=step_data_so_far, answered_qa=ctx.get("answered_qa"), previous_caps=previous_caps)

    # Require some explicit service context. We intentionally do not default industry/service
    # to "General", and the planner needs at least a hint of what vertical this is for.
    if not str(ctx.get("services_summary") or "").strip() and not str(ctx.get("industry") or "").strip() and not str(
        ctx.get("service") or ""
    ).strip():
        return {
            "ok": False,
            "error": "Missing service context (provide serviceSummary/service_summary or industry/service).",
            "requestId": request_id,
            "schemaVersion": str(schema_version or "0"),
        }

    # Extract batch_number (1-based)
    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}
    raw_batch_number = (
        current_batch.get("batchNumber")
        or current_batch.get("batch_number")
        or payload.get("batchNumber")
        or payload.get("batch_number")
        or 1
    )
    try:
        batch_number = int(raw_batch_number)
    except Exception:
        batch_number = 1

    # Extract per-call limits + allowed types
    max_steps_raw = (
        payload.get("maxStepsThisCall")
        or payload.get("max_steps_this_call")
        or payload.get("maxSteps")
        or payload.get("max_steps")
        or (current_batch.get("maxSteps") if isinstance(current_batch, dict) else None)
    )
    # If the caller doesn't specify a per-call cap, let `apply_flow_guide()` pick a backend default.
    if max_steps_raw is None:
        max_steps = 0
    else:
        try:
            max_steps = int(str(max_steps_raw))
        except Exception:
            max_steps = 0
        if max_steps < 1:
            max_steps = 0

    allowed_mini_types = ensure_allowed_mini_types(extract_allowed_mini_types_from_payload(payload))

    # Flow guide (stage defaults for allowed types + max steps)
    try:
        from programs.form_pipeline.planning import apply_flow_guide  # type: ignore

        ctx, allowed_mini_types, max_steps = apply_flow_guide(
            payload=payload,
            context=ctx,
            batch_number=batch_number,
            extracted_allowed_mini_types=allowed_mini_types,
            extracted_max_steps=max_steps,
        )
    except Exception:
        pass

    if ctx.get("prefer_structured_inputs"):
        allowed_mini_types = prefer_structured_allowed_mini_types(allowed_mini_types)

    asked_ids = set([str(x).strip() for x in (ctx.get("asked_step_ids") or []) if str(x).strip()])
    session_id = extract_session_id(payload)
    services_key_material = str(ctx.get("services_summary") or ctx.get("grounding_summary") or "").strip()
    if not services_key_material:
        services_key_material = str(ctx.get("service") or "").strip()
    if not services_key_material:
        services_key_material = f"{str(ctx.get('industry') or '').strip()}::{str(ctx.get('service') or '').strip()}"
    services_hash = short_hash(services_key_material, n=10)
    # Cache should vary by use_case, but the planner doesn't need it in the prompt.
    use_case_key = str(ctx.get("use_case") or "").strip().lower() or "none"
    cache_key = _planner_cache_key(session_id=session_id, services_fingerprint=services_hash, use_case_key=use_case_key)
    # IMPORTANT:
    # - noCache should mainly affect renderer output caching (debugging).
    # - planner plan determinism must be preserved per-session; otherwise the user sees duplicates/reshuffles.
    disable_render_cache = bool(payload.get("noCache") is True or str(payload.get("noCache") or "").lower() == "true")
    disable_planner_cache = False
    if os.getenv("AI_FORM_DEBUG") == "true":
        print(f"[FormPipeline] requestId={request_id} plannerCacheKey={cache_key}", flush=True)

    planner_context_json = _compact_json(
        {
            "services_summary": str(ctx.get("services_summary") or ctx.get("grounding_summary") or "").strip(),
            "service_summary": str(ctx.get("service_summary") or "").strip(),
            "company_summary": str(ctx.get("company_summary") or "").strip(),
            "industry": str(ctx.get("industry") or "").strip(),
            "service": str(ctx.get("service") or "").strip(),
            "answered_qa": ctx.get("answered_qa") if isinstance(ctx.get("answered_qa"), list) else [],
            "asked_step_ids": sorted(list(asked_ids)),
            "allowed_mini_types_hint": list(allowed_mini_types or []),
            "choice_option_min": ctx.get("choice_option_min"),
            "choice_option_max": ctx.get("choice_option_max"),
            "choice_option_target": ctx.get("choice_option_target"),
            "batch_constraints": ctx.get("batch_constraints") if isinstance(ctx.get("batch_constraints"), dict) else {},
            "required_uploads": ctx.get("required_uploads") if isinstance(ctx.get("required_uploads"), list) else [],
        }
    )

    # Planner (cached per session)
    _t0 = time.time()
    raw_plan = ""
    planner_cache_hit = False
    if cache_key and not disable_planner_cache:
        cached = ttl_cache_get(_PLANNER_PLAN_CACHE, cache_key)
        if cached:
            raw_plan = cached
            planner_cache_hit = True

    planner_module = QuestionPlannerProgram(demo_pack=(os.getenv("DSPY_PLANNER_DEMO_PACK") or "").strip())
    plan_pred: Optional[Any] = None
    if not raw_plan:
        track_usage = _configure_dspy(planner_lm) or track_usage
        plan_pred = planner_module(
            planner_context_json=planner_context_json,
            max_steps=int(_resolve_max_plan_items(ctx)),
            allowed_mini_types=allowed_mini_types,
        )
        raw_plan = str(getattr(plan_pred, "question_plan_json", "") or "")
        if cache_key and raw_plan.strip() and not disable_planner_cache:
            ttl_cache_set(_PLANNER_PLAN_CACHE, cache_key, raw_plan, ttl_sec=int(os.getenv("AI_FORM_PLANNER_CACHE_TTL_SEC") or "900"))
    t_planner_ms = int((time.time() - _t0) * 1000)

    # Parse the full plan without filtering asked steps; we filter per-call later to ensure we can
    # always fill `max_steps` while still keeping deterministic ordering.
    #
    # Reserve known internal keys if needed (currently none).
    reserved_suffix_keys: set[str] = set()
    full_plan_items = extract_plan_items(raw_plan, max_items=int(_resolve_max_plan_items(ctx)), asked_step_ids=set())
    full_plan_items = [x for x in full_plan_items if normalize_plan_key(x.get("key")) not in reserved_suffix_keys]

    # If we hit cache but it only contained reserved suffix keys (or was otherwise unusable), re-plan once.
    if planner_cache_hit and not full_plan_items:
        try:
            plan_pred = planner_module(
                planner_context_json=planner_context_json,
                max_steps=int(_resolve_max_plan_items(ctx)),
                allowed_mini_types=allowed_mini_types,
            )
            raw_plan_retry = str(getattr(plan_pred, "question_plan_json", "") or "")
            retry_items = extract_plan_items(raw_plan_retry, max_items=int(_resolve_max_plan_items(ctx)), asked_step_ids=set())
            retry_items = [x for x in retry_items if normalize_plan_key(x.get("key")) not in reserved_suffix_keys]
            if retry_items:
                raw_plan = raw_plan_retry
                planner_cache_hit = False
                full_plan_items = retry_items
                if cache_key and raw_plan.strip() and not disable_planner_cache:
                    ttl_cache_set(_PLANNER_PLAN_CACHE, cache_key, raw_plan, ttl_sec=int(os.getenv("AI_FORM_PLANNER_CACHE_TTL_SEC") or "900"))
        except Exception:
            pass

    plan_sequence: List[Dict[str, Any]] = []
    plan_sequence = list(full_plan_items)

    merged_plan_items: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in plan_sequence:
        if not isinstance(item, dict):
            continue
        key = normalize_plan_key(item.get("key"))
        if not key or key in seen_keys:
            continue
        sid = derive_step_id_from_key(key)
        normalized = dict(item)
        normalized["key"] = key
        merged_plan_items.append(normalized)
        seen_keys.add(key)

    # Slice next items for this batch
    sliced: List[Dict[str, Any]] = []
    for item in merged_plan_items:
        key = normalize_plan_key(item.get("key"))
        if not key:
            continue
        sid = derive_step_id_from_key(key)
        if sid in asked_ids:
            continue
        sliced.append(item)
        if len(sliced) >= int(max_steps):
            break

    # Only accept renderer outputs that match planned ids (prevents hallucinated steps like confirmation).
    planned_ids: set[str] = set()
    for item in sliced:
        if isinstance(item, dict):
            k = normalize_plan_key(item.get("key"))
            if k:
                planned_ids.add(derive_step_id_from_key(k))

    # Do NOT widen allowed types based on planner hints.
    # If the planner emits a `type_hint` that is not allowed by policy, it will be ignored downstream.
    allowed_mini_types = [str(x).strip().lower() for x in allowed_mini_types if str(x).strip()]

    renderer_module = RendererProgram(demo_pack=(os.getenv("DSPY_RENDERER_DEMO_PACK") or "").strip())
    render_context_json = _compact_json(
        {
            "services_summary": str(ctx.get("services_summary") or ctx.get("grounding_summary") or "").strip(),
            "choice_option_min": ctx.get("choice_option_min"),
            "choice_option_max": ctx.get("choice_option_max"),
            "choice_option_target": ctx.get("choice_option_target"),
            "required_uploads": ctx.get("required_uploads") if isinstance(ctx.get("required_uploads"), list) else [],
        }
    )
    render_cache_enabled = env_bool("AI_FORM_RENDER_CACHE", False)
    render_cache_hit = False
    pred: Optional[Any] = None
    raw_jsonl = ""
    parsed_steps: List[Dict[str, Any]] = []

    _t0 = time.time()
    plan_json_for_render = _compact_json({"plan": sliced})
    render_cache_key = (
        _render_cache_key(
            session_id=session_id,
            schema_version=str(schema_version or "0"),
            plan_json=plan_json_for_render,
            render_context_json=render_context_json,
            allowed_mini_types=allowed_mini_types,
        )
        if (render_cache_enabled and not disable_render_cache)
        else ""
    )
    if os.getenv("AI_FORM_DEBUG") == "true" and render_cache_key:
        print(f"[FormPipeline] requestId={request_id} renderCacheKey={render_cache_key}", flush=True)
    cached_emitted = ttl_cache_get(_RENDER_OUTPUT_CACHE, render_cache_key) if render_cache_key else None

    # Renderer output cache is always *post-validation* output (miniSteps[]), never raw JSONL.
    # This preserves schema enforcement even when cached.
    if isinstance(cached_emitted, list) and cached_emitted:
        render_cache_hit = True

    if not render_cache_hit:
        track_usage = _configure_dspy(renderer_lm) or track_usage
        pred = renderer_module(
            question_plan_json=plan_json_for_render,
            render_context_json=render_context_json,
            max_steps=len(sliced),
            allowed_mini_types=allowed_mini_types,
        )
        if os.getenv("AI_FORM_DEBUG") == "true":
            _print_lm_history_if_available(renderer_lm, n=1)

        raw_jsonl = str(getattr(pred, "mini_steps_jsonl", "") or "")
        parsed_steps = parse_jsonl_steps(raw_jsonl)
        # Best-effort retry: if the renderer returns empty output, try once more.
        if (not raw_jsonl.strip() or not parsed_steps) and sliced:
            try:
                pred = renderer_module(
                    question_plan_json=plan_json_for_render,
                    render_context_json=render_context_json,
                    max_steps=len(sliced),
                    allowed_mini_types=allowed_mini_types,
                )
                raw_jsonl = str(getattr(pred, "mini_steps_jsonl", "") or "")
                parsed_steps = parse_jsonl_steps(raw_jsonl)
            except Exception:
                pass
    t_renderer_ms = int((time.time() - _t0) * 1000)

    ui_types = _select_ui_types()
    allowed_set = set([str(x).strip().lower() for x in allowed_mini_types if str(x).strip()])
    required_upload_ids = _extract_required_upload_ids(ctx.get("required_uploads"))

    emitted: List[Dict[str, Any]] = []
    taken_ids: set[str] = set(asked_ids)
    _t0 = time.time()
    if render_cache_hit and isinstance(cached_emitted, list):
        # Best-effort: cached output was validated before insertion; still normalize list shape.
        emitted = [x for x in cached_emitted if isinstance(x, dict)]
        for x in emitted:
            sid = str(x.get("id") or "").strip()
            if sid:
                taken_ids.add(sid)
    else:
        for s in parsed_steps:
            if not isinstance(s, dict):
                continue
            sid = str(s.get("id") or "").strip()
            if not sid or sid in taken_ids:
                continue
            if planned_ids and sid not in planned_ids:
                continue
            if not allowed_type_matches(str(s.get("type") or ""), allowed_set):
                continue
            if _looks_like_upload_step_id(sid) and required_upload_ids and sid not in required_upload_ids:
                # If required upload ids exist, only allow those upload ids.
                continue
            validated = _validate_mini(s, ui_types)
            if not validated:
                continue
            validated = _reject_banned_option_sets(validated)
            if not validated:
                continue
            emitted.append(validated)
            taken_ids.add(sid)

    # Renderer backstop for deterministic suffix items.
    # If the renderer fails to emit required suffix steps, inject minimal validated steps.
    if sliced and len(emitted) < len(sliced):
        for plan_item in sliced:
            if not isinstance(plan_item, dict):
                continue
            if plan_item.get("deterministic") is not True:
                continue
            key = normalize_plan_key(plan_item.get("key"))
            if not key:
                continue
            sid = derive_step_id_from_key(key)
            if not sid or sid in taken_ids:
                continue
            if len(emitted) >= len(sliced):
                break

            t = str(plan_item.get("type_hint") or "").strip().lower()
            if not t:
                continue
            if not allowed_type_matches(t, allowed_set):
                continue
            if _looks_like_upload_step_id(sid) and required_upload_ids and sid not in required_upload_ids:
                continue

            candidate = {
                "id": sid,
                "type": t,
                "question": str(plan_item.get("question") or plan_item.get("intent") or "").strip() or "Continue.",
                "required": bool(plan_item.get("required") is True),
            }
            validated = _validate_mini(candidate, ui_types)
            if not validated:
                continue
            validated = _reject_banned_option_sets(validated)
            if not validated:
                continue
            emitted.append(validated)
            taken_ids.add(sid)

    # Final copy sanitation (question marks, remove duplicated enumerations, etc.)
    emitted = sanitize_steps(emitted, lint_config)
    t_post_ms = int((time.time() - _t0) * 1000)

    # Cache renderer output (validated miniSteps only).
    if render_cache_key and (not disable_render_cache) and (not render_cache_hit) and emitted:
        ttl_sec = env_int("AI_FORM_RENDER_CACHE_TTL_SEC", 600)
        ttl_cache_set(_RENDER_OUTPUT_CACHE, render_cache_key, emitted, ttl_sec=ttl_sec)

    meta: Dict[str, Any] = {
        "requestId": request_id,
        "schemaVersion": str(schema_version or "0"),
        "miniSteps": emitted,
        "__capabilities": dict(capabilities),
        # Convenience echo to help clients persist monotonic flags without extra storage.
        "stepDataSoFar": {**step_data_so_far, "__capabilities": dict(capabilities)},
    }

    if _include_response_meta(payload):
        meta["debugContext"] = {
            "industry": ctx.get("industry"),
            "service": ctx.get("service"),
            "useCase": ctx.get("use_case"),
            "goalIntent": ctx.get("goal_intent"),
            "servicesSummaryLen": len(str(ctx.get("services_summary") or ctx.get("grounding_summary") or "")),
            "companySummaryLen": len(str(ctx.get("company_summary") or "")),
            "allowedMiniTypes": allowed_mini_types,
            "maxSteps": max_steps,
            "plannerModel": planner_lm_cfg.get("modelName"),
            "rendererModel": renderer_lm_cfg.get("modelName"),
            "plannerCacheHit": planner_cache_hit,
            "renderCacheHit": render_cache_hit,
            "plannedItems": len(sliced),
            "renderedJsonlLines": len(parsed_steps),
            "emittedSteps": len(emitted),
            "tokenBudgetTotal": token_budget_total,
            "tokenBudgetUsed": token_budget_used,
            "tokenBudgetRemaining": token_budget_remaining,
            "tokenBudgetSoftExceeded": bool(token_budget_soft_exceeded),
            "capabilities": dict(capabilities),
        }

    if track_usage:
        lm_usage_by_module: Dict[str, Any] = {}
        usage_planner = extract_dspy_usage(plan_pred) if plan_pred is not None else None
        usage_renderer = extract_dspy_usage(pred) if pred is not None else None
        if usage_planner:
            lm_usage_by_module["planner"] = usage_planner
        if usage_renderer:
            lm_usage_by_module["renderer"] = usage_renderer
            # Back-compat: keep `lmUsage` as renderer usage.
            meta["lmUsage"] = usage_renderer
        if lm_usage_by_module:
            meta["lmUsageByModule"] = lm_usage_by_module

    latency_ms = int((time.time() - start_time) * 1000)
    if env_bool("AI_FORM_LOG_LATENCY", False):
        try:
            print(
                json.dumps(
                    {
                        "event": "step3_latency",
                        "requestId": request_id,
                        "plannerMs": int(t_planner_ms),
                        "rendererMs": int(t_renderer_ms),
                        "postProcessingMs": int(t_post_ms),
                        "totalMs": int(latency_ms),
                        "plannerModel": planner_lm_cfg.get("modelName"),
                        "rendererModel": renderer_lm_cfg.get("modelName"),
                        "plannerCacheHit": bool(planner_cache_hit),
                        "renderCacheHit": bool(render_cache_hit),
                        "plannedItems": int(len(sliced)),
                        "renderedJsonlLines": int(len(parsed_steps)),
                        "emittedSteps": int(len(emitted)),
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception:
            pass
    if os.getenv("AI_FORM_DEBUG") == "true":
        print(
            (
                f"[FormPipeline] requestId={request_id} latencyMs={latency_ms} steps={len(emitted)} "
                f"plannerModel={planner_lm_cfg.get('modelName') or planner_lm_cfg.get('model')} "
                f"rendererModel={renderer_lm_cfg.get('modelName') or renderer_lm_cfg.get('model')} "
                f"plannerCacheHit={planner_cache_hit} renderCacheHit={render_cache_hit}"
            ),
            flush=True,
        )

    return meta


__all__ = [
    "next_steps_jsonl",
    "_build_context",
    "_compact_json",
    "_configure_dspy",
    "_make_dspy_lm",
]

