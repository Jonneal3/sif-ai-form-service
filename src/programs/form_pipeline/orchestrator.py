"""
Form pipeline orchestrator (Planner -> Renderer).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from programs.form_pipeline.allowed_types import (
    allowed_type_matches,
    ensure_allowed_mini_types,
    extract_allowed_mini_types_from_payload,
    prefer_structured_allowed_mini_types,
)
from programs.form_pipeline.context import apply_copy_pack, build_context, extract_session_id, extract_token_budget
from programs.form_pipeline.grounding_summary import GroundingSummaryProgram
from programs.form_pipeline.planning import build_deterministic_suffix_plan_items
from programs.form_pipeline.utils import _compact_json
from programs.question_planner.program import QuestionPlannerProgram
from programs.renderer.program import RendererProgram

from programs.form_pipeline.validation import (
    _best_effort_parse_json,
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


def _planner_cache_get(cache_key: str) -> Optional[str]:
    if not cache_key:
        return None
    rec = _PLANNER_PLAN_CACHE.get(cache_key)
    if not rec:
        return None
    expires_at, value = rec
    if time.time() >= float(expires_at):
        _PLANNER_PLAN_CACHE.pop(cache_key, None)
        return None
    return value


def _planner_cache_set(cache_key: str, value: str, *, ttl_sec: int) -> None:
    if not cache_key or not value:
        return
    ttl = max(60, min(3600, int(ttl_sec or 0)))
    _PLANNER_PLAN_CACHE[cache_key] = (time.time() + ttl, str(value))


def _make_dspy_lm() -> Optional[Dict[str, str]]:
    """
    Return a LiteLLM model string for DSPy v3 (provider-prefixed), or None if not configured.
    """

    provider = (os.getenv("DSPY_PROVIDER") or "groq").lower()
    locked_model = os.getenv("DSPY_MODEL_LOCK") or "openai/gpt-oss-20b"
    requested_model = os.getenv("DSPY_MODEL") or locked_model
    model = requested_model

    is_gpt_oss = "gpt-oss" in model.lower()
    if not is_gpt_oss and ("8b" in model.lower() or "8-b" in model.lower() or "instant" in model.lower()):
        model = locked_model

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            return None
        model_str = f"groq/{model}" if model.startswith("openai/") else f"groq/{model}"
        return {"provider": "groq", "model": model_str, "modelName": model}

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return {"provider": "openai", "model": f"openai/{model}", "modelName": model}

    return None


def _configure_dspy(lm: Any) -> bool:
    try:
        import dspy  # type: ignore
    except Exception:
        return False

    telemetry_on = os.getenv("AI_FORM_TOKEN_TELEMETRY") == "true" or os.getenv("AI_FORM_DEBUG") == "true"
    track_usage = os.getenv("DSPY_TRACK_USAGE") == "true" or telemetry_on
    try:
        dspy.settings.configure(lm=lm, track_usage=track_usage)
        return track_usage
    except Exception:
        return False


def _extract_dspy_usage(prediction: Any) -> Optional[Dict[str, Any]]:
    try:
        get_usage = getattr(prediction, "get_lm_usage", None)
        if callable(get_usage):
            usage = get_usage()
            if isinstance(usage, dict) and usage:
                return usage
    except Exception:
        return None
    return None


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


def _normalize_plan_key(raw: Any) -> str:
    t = str(raw or "").strip().lower()
    if not t:
        return ""
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    t = re.sub(r"_+", "_", t)
    return t[:48]


def _derive_step_id_from_key(key: str) -> str:
    return f"step-{key.replace('_', '-')}"


def _extract_plan_items(text: Any, *, max_items: int, asked_step_ids: set[str]) -> List[Dict[str, Any]]:
    parsed = _best_effort_parse_json(str(text or ""))
    if isinstance(parsed, list):
        raw_items = parsed
    elif isinstance(parsed, dict):
        raw_items = parsed.get("plan")
        if not isinstance(raw_items, list):
            raw_items = parsed.get("question_keys")
        if not isinstance(raw_items, list):
            raw_items = parsed.get("items")
        if not isinstance(raw_items, list):
            raw_items = []
    else:
        raw_items = []

    out: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        key = _normalize_plan_key(item.get("key"))
        if not key or key in seen_keys:
            continue
        step_id = _derive_step_id_from_key(key)
        if step_id in asked_step_ids:
            continue
        seen_keys.add(key)
        normalized = dict(item)
        normalized["key"] = key
        out.append(normalized)
        if len(out) >= max_items:
            break
    return out


def _resolve_max_plan_items(ctx: Dict[str, Any]) -> int:
    constraints = ctx.get("batch_constraints") if isinstance(ctx.get("batch_constraints"), dict) else {}
    raw = constraints.get("maxStepsTotal") or constraints.get("max_steps_total")
    try:
        n = int(raw) if raw is not None else 0
    except Exception:
        n = 0
    n = max(4, min(30, int(n or 12)))
    return n


def _parse_jsonl_steps(text: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    raw = str(text or "")
    if not raw.strip():
        return out
    for line in raw.splitlines():
        t = line.strip()
        if not t:
            continue
        obj = _best_effort_parse_json(t)
        if isinstance(obj, dict):
            out.append(obj)
    return out


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

    schema_version = payload.get("schemaVersion") or payload.get("schema_version") or _best_effort_contract_schema_version()

    lm_cfg = _make_dspy_lm()
    if not lm_cfg:
        return {"ok": False, "error": "DSPy LM not configured", "requestId": request_id, "schemaVersion": str(schema_version or "0")}

    try:
        import dspy  # type: ignore
    except Exception:
        return {"ok": False, "error": "DSPy import failed", "requestId": request_id, "schemaVersion": str(schema_version or "0")}

    # Token budget guard (best-effort)
    batch_state_raw = payload.get("batchState") or payload.get("batch_state") or {}
    tokens_total, tokens_used = extract_token_budget(batch_state_raw)
    if isinstance(tokens_total, int) and tokens_total > 0:
        used = tokens_used if isinstance(tokens_used, int) and tokens_used > 0 else 0
        if tokens_total - used <= 0:
            return {"ok": False, "error": "Token budget exhausted", "requestId": request_id, "schemaVersion": str(schema_version or "0")}

    llm_timeout = float(os.getenv("DSPY_LLM_TIMEOUT_SEC") or "20")
    temperature = float(os.getenv("DSPY_TEMPERATURE") or "0.7")
    default_max_tokens = int(os.getenv("DSPY_NEXT_STEPS_MAX_TOKENS") or "2000")
    max_tokens = default_max_tokens

    lm = dspy.LM(
        model=lm_cfg["model"],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=llm_timeout,
        num_retries=0,
    )
    track_usage = _configure_dspy(lm)

    # Build context and apply copy pack style
    ctx = _build_context(payload)
    ctx, lint_config, copy_pack_id = apply_copy_pack(payload, context=ctx)

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
        or "4"
    )
    try:
        max_steps = int(str(max_steps_raw))
        if max_steps < 1:
            max_steps = 4
    except Exception:
        max_steps = 4

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

    # If grounding is missing, generate a short one (cheap).
    if not str(ctx.get("grounding_summary") or "").strip():
        grounding_ctx = {
            "industry": ctx.get("industry"),
            "service": ctx.get("service"),
            "instance_subcategories": ctx.get("instance_subcategories"),
            "platform_goal": ctx.get("platform_goal"),
            "goal_intent": ctx.get("goal_intent"),
            "use_case": ctx.get("use_case"),
            "answered_qa": ctx.get("answered_qa") if isinstance(ctx.get("answered_qa"), list) else [],
        }
        grounding_context_json = _compact_json(grounding_ctx)
        try:
            grounding_prog = GroundingSummaryProgram()
            gs_pred = grounding_prog(grounding_context_json=grounding_context_json)
            generated = str(getattr(gs_pred, "grounding_summary", "") or "").strip()
        except Exception:
            generated = ""
        if generated:
            ctx["grounding_summary"] = generated[:600]
            ctx["vertical_context"] = generated[:600]

    asked_ids = set([str(x).strip() for x in (ctx.get("asked_step_ids") or []) if str(x).strip()])
    session_id = extract_session_id(payload)
    services_hash = hashlib.sha256(str(ctx.get("service") or "").encode("utf-8")).hexdigest()[:8] if str(ctx.get("service") or "") else "none"
    platform_goal_for_cache = str(ctx.get("platform_goal") or "")[:200]
    goal_hash = hashlib.sha256(platform_goal_for_cache.encode("utf-8")).hexdigest()[:10] if platform_goal_for_cache else "none"
    cache_key = f"question_plan:{session_id}:{services_hash}:{goal_hash}" if session_id else ""

    planner_context_json = _compact_json(
        {
            "vertical_context": {
                "industry": ctx.get("industry"),
                "service": ctx.get("service"),
                "instance_subcategories": ctx.get("instance_subcategories"),
                "grounding_summary": ctx.get("grounding_summary") or "",
            },
            "goal_context": {
                "use_case": ctx.get("use_case"),
                "goal_intent": ctx.get("goal_intent"),
                "platform_goal": ctx.get("platform_goal"),
            },
            "memory_context": {
                "known_answers": ctx.get("known_answers") if isinstance(ctx.get("known_answers"), dict) else {},
                "asked_step_ids": sorted(list(asked_ids)),
                "answered_qa": ctx.get("answered_qa") if isinstance(ctx.get("answered_qa"), list) else [],
            },
            "constraints": {
                "max_steps": int(_resolve_max_plan_items(ctx)),
                "allowed_mini_types": allowed_mini_types,
                "choice_option_min": ctx.get("choice_option_min"),
                "choice_option_max": ctx.get("choice_option_max"),
                "choice_option_target": ctx.get("choice_option_target"),
            },
        }
    )

    # Planner (cached per session)
    raw_plan = ""
    if cache_key:
        cached = _planner_cache_get(cache_key)
        if cached:
            raw_plan = cached

    planner_module = QuestionPlannerProgram(demo_pack=(os.getenv("DSPY_PLANNER_DEMO_PACK") or "").strip())
    plan_pred: Optional[Any] = None
    if not raw_plan:
        plan_pred = planner_module(
            planner_context_json=planner_context_json,
            max_steps=int(_resolve_max_plan_items(ctx)),
            allowed_mini_types=allowed_mini_types,
        )
        raw_plan = str(getattr(plan_pred, "question_plan_json", "") or "")
        if cache_key and raw_plan.strip():
            _planner_cache_set(cache_key, raw_plan, ttl_sec=int(os.getenv("AI_FORM_PLANNER_CACHE_TTL_SEC") or "900"))

    full_plan_items = _extract_plan_items(raw_plan, max_items=int(_resolve_max_plan_items(ctx)), asked_step_ids=asked_ids)

    # Merge in a deterministic suffix so the form always ends predictably.
    suffix_plan_items = build_deterministic_suffix_plan_items(context=ctx)
    merged_plan_items: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in list(full_plan_items) + list(suffix_plan_items):
        if not isinstance(item, dict):
            continue
        key = _normalize_plan_key(item.get("key"))
        if not key or key in seen_keys:
            continue
        sid = _derive_step_id_from_key(key)
        if sid in asked_ids:
            continue
        normalized = dict(item)
        normalized["key"] = key
        merged_plan_items.append(normalized)
        seen_keys.add(key)

    # Slice next items for this batch
    sliced: List[Dict[str, Any]] = []
    for item in merged_plan_items:
        key = _normalize_plan_key(item.get("key"))
        if not key:
            continue
        sid = _derive_step_id_from_key(key)
        if sid in asked_ids:
            continue
        sliced.append(item)
        if len(sliced) >= int(max_steps):
            break

    # Ensure deterministic suffix types are renderable even in early/middle stages.
    forced_types: set[str] = set()
    for item in sliced:
        if not isinstance(item, dict):
            continue
        t = str(item.get("type_hint") or "").strip().lower()
        if t:
            forced_types.add(t)
    if forced_types:
        allowed_mini_types = sorted(set([str(x).strip().lower() for x in allowed_mini_types if str(x).strip()]) | forced_types)

    renderer_module = RendererProgram(demo_pack=(os.getenv("DSPY_RENDERER_DEMO_PACK") or "").strip())
    render_context_json = _compact_json(
        {
            "grounding_summary": ctx.get("grounding_summary") or "",
            "copy_style": ctx.get("copy_style") if isinstance(ctx.get("copy_style"), str) else "",
            "choice_option_min": ctx.get("choice_option_min"),
            "choice_option_max": ctx.get("choice_option_max"),
            "choice_option_target": ctx.get("choice_option_target"),
            "required_uploads": ctx.get("required_uploads") if isinstance(ctx.get("required_uploads"), list) else [],
            "asked_step_ids": sorted(list(asked_ids)),
        }
    )
    pred = renderer_module(
        question_plan_json=_compact_json({"plan": sliced}),
        render_context_json=render_context_json,
        max_steps=len(sliced),
        allowed_mini_types=allowed_mini_types,
    )
    if os.getenv("AI_FORM_DEBUG") == "true":
        _print_lm_history_if_available(lm, n=1)

    raw_jsonl = str(getattr(pred, "mini_steps_jsonl", "") or "")
    parsed_steps = _parse_jsonl_steps(raw_jsonl)

    ui_types = _select_ui_types()
    allowed_set = set([str(x).strip().lower() for x in allowed_mini_types if str(x).strip()])
    required_upload_ids = _extract_required_upload_ids(ctx.get("required_uploads"))

    emitted: List[Dict[str, Any]] = []
    taken_ids: set[str] = set(asked_ids)
    for s in parsed_steps:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id") or "").strip()
        if not sid or sid in taken_ids:
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
            key = _normalize_plan_key(plan_item.get("key"))
            if not key:
                continue
            sid = _derive_step_id_from_key(key)
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

            candidate: Dict[str, Any] = {
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

    meta: Dict[str, Any] = {"requestId": request_id, "schemaVersion": str(schema_version or "0"), "miniSteps": emitted}

    if _include_response_meta(payload):
        meta["copyPackId"] = copy_pack_id
        meta["debugContext"] = {
            "industry": ctx.get("industry"),
            "service": ctx.get("service"),
            "useCase": ctx.get("use_case"),
            "goalIntent": ctx.get("goal_intent"),
            "platformGoalPreview": str(ctx.get("platform_goal") or "")[:120],
            "groundingSummaryLen": len(str(ctx.get("grounding_summary") or "")),
            "allowedMiniTypes": allowed_mini_types,
            "maxSteps": max_steps,
        }

    if track_usage:
        usage = _extract_dspy_usage(pred)
        if usage:
            meta["lmUsage"] = usage

    latency_ms = int((time.time() - start_time) * 1000)
    if os.getenv("AI_FORM_DEBUG") == "true":
        print(
            f"[FormPipeline] requestId={request_id} latencyMs={latency_ms} steps={len(emitted)} model={lm_cfg.get('modelName') or lm_cfg.get('model')}",
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

