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
from programs.form_pipeline.context_builder import build_context
from programs.form_pipeline.constraints import extract_token_budget
from programs.form_pipeline.payload_extractors import extract_session_id
from programs.form_pipeline.planning import build_deterministic_suffix_plan_items, sanitize_steps
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
_RENDER_OUTPUT_CACHE: dict[str, tuple[float, List[Dict[str, Any]]]] = {}

_AUGMENTED_PLAN_CACHE: dict[str, tuple[float, List[Dict[str, Any]]]] = {}


def _take_next_unasked_plan_items(
    items: List[Dict[str, Any]],
    *,
    asked_step_ids: set[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Take the next `limit` plan items, skipping those already asked.
    """
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = _normalize_plan_key(item.get("key"))
        if not key:
            continue
        sid = _derive_step_id_from_key(key)
        if sid in asked_step_ids:
            continue
        out.append(item)
        if len(out) >= int(limit):
            break
    return out


def _build_initial_image_trigger_plan_item(
    plan_items: List[Dict[str, Any]],
    *,
    after_n_keys: int,
) -> Optional[Dict[str, Any]]:
    """
    Build a deterministic plan item that signals the frontend to call image generation.

    We intentionally do not execute any function here; we only emit a `functionCall` hint.
    """
    n = max(2, min(6, int(after_n_keys or 0)))
    if n < 2:
        n = 3

    trigger_key = "initial_image_trigger"
    keys_in_plan = [_normalize_plan_key(x.get("key")) for x in plan_items if isinstance(x, dict)]
    keys_in_plan = [k for k in keys_in_plan if k]
    if trigger_key in set(keys_in_plan):
        return None

    if len(keys_in_plan) < 3:
        # Not enough signal to justify generating a preview; skip quietly.
        return None

    trigger_after = keys_in_plan[: min(n, len(keys_in_plan))]
    function_call = {
        "name": "generateInitialImage",
        "triggerAfterStepKeys": trigger_after,
    }
    return {
        "key": trigger_key,
        "question": "Preview your design so far",
        # Render this as a composite so the frontend can keep it sticky/shared.
        "type_hint": "composite",
        # Mark deterministic so renderer backstops will inject it if missing.
        "deterministic": True,
        "functionCall": function_call,
        # Provide a deterministic, renderer-optional template.
        # The backend also backstops this into emitted miniSteps to avoid model drift.
        "blocks": [
            {"type": "question", "content": "Preview your design so far"},
            {"type": "designer", "functionCall": function_call},
        ],
        "metadata": {"shared": True},
    }


def _augment_plan_items_for_function_calls(
    plan_items: List[Dict[str, Any]],
    *,
    cache_key: str,
    ttl_sec: int,
) -> List[Dict[str, Any]]:
    """
    Deterministically insert function-call hint plan items (cached per session).
    """
    cached = _ttl_cache_get(_AUGMENTED_PLAN_CACHE, cache_key) if cache_key else None
    if isinstance(cached, list) and cached:
        return [x for x in cached if isinstance(x, dict)]

    after_n = _env_int("AI_FORM_INITIAL_IMAGE_TRIGGER_AFTER_N", 3)
    trigger_item = _build_initial_image_trigger_plan_item(plan_items, after_n_keys=after_n)
    if not trigger_item:
        if cache_key:
            _ttl_cache_set(_AUGMENTED_PLAN_CACHE, cache_key, list(plan_items), ttl_sec=ttl_sec)
        return list(plan_items)

    # Insert after N planned questions (example: "trigger image at step 4" => after 3 keys).
    insert_at = min(max(0, int(after_n)), len(plan_items))
    augmented = list(plan_items[:insert_at]) + [trigger_item] + list(plan_items[insert_at:])
    if cache_key:
        _ttl_cache_set(_AUGMENTED_PLAN_CACHE, cache_key, augmented, ttl_sec=ttl_sec)
    return augmented


def _composite_trigger_step_from_plan_item(plan_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Deterministic transform for a composite function-call trigger step.
    """
    if not isinstance(plan_item, dict):
        return None
    key = _normalize_plan_key(plan_item.get("key"))
    if not key:
        return None
    sid = _derive_step_id_from_key(key)
    question = str(plan_item.get("question") or plan_item.get("intent") or "").strip() or "Preview your design so far"
    fc = plan_item.get("functionCall")
    if not isinstance(fc, dict) or not fc:
        return None
    # Keep functionCall both at the top-level (back-compat) and inside the designer block.
    return {
        "id": sid,
        "type": "composite",
        "question": question,
        "blocks": [
            {"type": "question", "content": question},
            {"type": "designer", "functionCall": dict(fc)},
        ],
        "metadata": {"shared": True},
        "functionCall": dict(fc),
    }


def _is_initial_image_trigger_plan_item(plan_item: Dict[str, Any]) -> bool:
    return _normalize_plan_key(plan_item.get("key")) == "initial_image_trigger"


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


def _ttl_cache_get(cache: dict[str, tuple[float, Any]], cache_key: str) -> Any:
    if not cache_key:
        return None
    rec = cache.get(cache_key)
    if not rec:
        return None
    expires_at, value = rec
    if time.time() >= float(expires_at):
        cache.pop(cache_key, None)
        return None
    return value


def _ttl_cache_set(cache: dict[str, tuple[float, Any]], cache_key: str, value: Any, *, ttl_sec: int) -> None:
    if not cache_key:
        return
    ttl = max(60, min(3600, int(ttl_sec or 0)))
    cache[cache_key] = (time.time() + ttl, value)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _prefixed_model(provider: str, model_name: str) -> str:
    p = str(provider or "").strip().lower()
    m = str(model_name or "").strip()
    if not p:
        return m
    if m.startswith(f"{p}/"):
        return m
    return f"{p}/{m}"


def _make_dspy_lm_for_module(*, module_env_prefix: str, allow_small_models: bool) -> Optional[Dict[str, str]]:
    """
    Resolve the DSPy LM config for a module using env overrides.

    Env resolution order (example for module_env_prefix=\"DSPY_PLANNER\"):
      - DSPY_PLANNER_PROVIDER / DSPY_PROVIDER
      - DSPY_PLANNER_MODEL_LOCK / DSPY_MODEL_LOCK / default
      - DSPY_PLANNER_MODEL / DSPY_MODEL / DSPY_PLANNER_MODEL_LOCK
    """

    prefix = str(module_env_prefix or "").strip().upper()
    provider = (os.getenv(f"{prefix}_PROVIDER") or os.getenv("DSPY_PROVIDER") or "groq").lower()
    locked_model = os.getenv(f"{prefix}_MODEL_LOCK") or os.getenv("DSPY_MODEL_LOCK") or "openai/gpt-oss-20b"
    requested_model = os.getenv(f"{prefix}_MODEL") or os.getenv("DSPY_MODEL") or locked_model
    model_name = str(requested_model or locked_model).strip()

    # Safety guard: keep planner on a strong model unless explicitly allowed.
    if not allow_small_models:
        is_gpt_oss = "gpt-oss" in model_name.lower()
        if not is_gpt_oss and ("8b" in model_name.lower() or "8-b" in model_name.lower() or "instant" in model_name.lower()):
            model_name = str(locked_model or model_name).strip()

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            return None
        return {"provider": "groq", "model": _prefixed_model("groq", model_name), "modelName": model_name}

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return {"provider": "openai", "model": _prefixed_model("openai", model_name), "modelName": model_name}

    return None


def _make_dspy_lm() -> Optional[Dict[str, str]]:
    """
    Return a LiteLLM model string for DSPy v3 (provider-prefixed), or None if not configured.
    """
    # Legacy behavior: use global env vars and keep the small-model guard on.
    return _make_dspy_lm_for_module(module_env_prefix="DSPY_PLANNER", allow_small_models=False)


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


def _short_hash(text: str, *, n: int = 10) -> str:
    t = str(text or "")
    if not t:
        return "none"
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[: max(6, min(24, int(n or 10)))]


def _planner_cache_key(*, session_id: str, services_fingerprint: str, use_case_key: str) -> str:
    sid = str(session_id or "").strip()
    if not sid:
        return ""
    svc = str(services_fingerprint or "").strip() or "none"
    uc = str(use_case_key or "").strip().lower() or "none"
    return f"question_plan:{sid}:{svc}:{uc}"


def _render_cache_key(
    *,
    session_id: str,
    schema_version: str,
    plan_json: str,
    render_context_json: str,
    allowed_mini_types: List[str],
) -> str:
    sid = str(session_id or "").strip()
    if not sid:
        return ""
    sv = str(schema_version or "").strip() or "0"
    plan_h = _short_hash(plan_json, n=12)
    ctx_h = _short_hash(render_context_json, n=12)
    allowed_h = _short_hash(",".join(sorted([str(x).strip().lower() for x in (allowed_mini_types or []) if str(x).strip()])), n=10)
    return f"render_out:{sid}:{sv}:{plan_h}:{ctx_h}:{allowed_h}"


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
    t_planner_ms = 0
    t_renderer_ms = 0
    t_post_ms = 0

    schema_version = payload.get("schemaVersion") or payload.get("schema_version") or _best_effort_contract_schema_version()

    planner_lm_cfg = _make_dspy_lm_for_module(module_env_prefix="DSPY_PLANNER", allow_small_models=False)
    renderer_lm_cfg = _make_dspy_lm_for_module(module_env_prefix="DSPY_RENDERER", allow_small_models=True)
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
            allowed_overage = _env_int("AI_FORM_TOKEN_BUDGET_ALLOWED_OVERAGE", 750)
            if remaining < -int(allowed_overage):
                return {
                    "ok": False,
                    "error": "Token budget exhausted",
                    "requestId": request_id,
                    "schemaVersion": str(schema_version or "0"),
                }
            token_budget_soft_exceeded = True

    default_timeout = _env_float("DSPY_LLM_TIMEOUT_SEC", 20.0)
    default_temperature = _env_float("DSPY_TEMPERATURE", 0.7)
    default_max_tokens = _env_int("DSPY_NEXT_STEPS_MAX_TOKENS", 2000)

    planner_timeout = _env_float("DSPY_PLANNER_TIMEOUT_SEC", default_timeout)
    planner_temperature = _env_float("DSPY_PLANNER_TEMPERATURE", default_temperature)
    planner_max_tokens = _env_int("DSPY_PLANNER_MAX_TOKENS", default_max_tokens)

    renderer_timeout = _env_float("DSPY_RENDERER_TIMEOUT_SEC", default_timeout)
    renderer_temperature = _env_float("DSPY_RENDERER_TEMPERATURE", default_temperature)
    renderer_max_tokens = _env_int("DSPY_RENDERER_MAX_TOKENS", default_max_tokens)

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
    services_hash = _short_hash(services_key_material, n=10)
    # Cache should vary by use_case, but the planner doesn't need it in the prompt.
    use_case_key = str(ctx.get("use_case") or "").strip().lower() or "none"
    cache_key = _planner_cache_key(session_id=session_id, services_fingerprint=services_hash, use_case_key=use_case_key)
    disable_cache = bool(payload.get("noCache") is True or str(payload.get("noCache") or "").lower() == "true")
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
    if cache_key and not disable_cache:
        cached = _planner_cache_get(cache_key)
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
        if cache_key and raw_plan.strip() and not disable_cache:
            _planner_cache_set(cache_key, raw_plan, ttl_sec=int(os.getenv("AI_FORM_PLANNER_CACHE_TTL_SEC") or "900"))
    t_planner_ms = int((time.time() - _t0) * 1000)

    # Parse the full plan without filtering asked steps; we filter per-call later to ensure we can
    # always fill `max_steps` while still keeping deterministic ordering.
    full_plan_items = _extract_plan_items(raw_plan, max_items=int(_resolve_max_plan_items(ctx)), asked_step_ids=set())

    # Deterministically insert function-call hints (cached per sessionId+services_fingerprint+use_case).
    # This ensures retries or repeated requests do not change ordering/keys.
    augmented_ttl = _env_int("AI_FORM_PLANNER_CACHE_TTL_SEC", 900)
    full_plan_items = _augment_plan_items_for_function_calls(full_plan_items, cache_key=cache_key, ttl_sec=augmented_ttl)

    # Merge in a deterministic suffix so the form always ends predictably.
    suffix_plan_items = build_deterministic_suffix_plan_items(context=ctx)

    # If we're effectively running a single-batch flow, reserve room for the suffix in this batch.
    batch_constraints = ctx.get("batch_constraints") if isinstance(ctx.get("batch_constraints"), dict) else {}
    try:
        max_batches = int(batch_constraints.get("maxBatches") or 0)
    except Exception:
        max_batches = 0
    suffix_in_this_batch = max_batches <= 1

    plan_sequence: List[Dict[str, Any]] = []
    if suffix_in_this_batch and suffix_plan_items:
        reserved = len([x for x in suffix_plan_items if isinstance(x, dict)])
        n_planner = max(0, int(max_steps) - int(reserved))
        # Reserve space for suffix, but still skip asked steps deterministically.
        plan_sequence = (
            _take_next_unasked_plan_items(list(full_plan_items), asked_step_ids=asked_ids, limit=n_planner)
            + list(suffix_plan_items)
        )
    else:
        plan_sequence = list(full_plan_items) + list(suffix_plan_items)

    merged_plan_items: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in plan_sequence:
        if not isinstance(item, dict):
            continue
        key = _normalize_plan_key(item.get("key"))
        if not key or key in seen_keys:
            continue
        sid = _derive_step_id_from_key(key)
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

    # Optional validation: warn if functionCall.triggerAfterStepKeys references unknown keys.
    # Do not break the flow.
    try:
        known_keys = set([_normalize_plan_key(x.get("key")) for x in merged_plan_items if isinstance(x, dict)])
        for item in merged_plan_items:
            if not isinstance(item, dict):
                continue
            fc = item.get("functionCall")
            if not isinstance(fc, dict):
                continue
            after_keys = fc.get("triggerAfterStepKeys")
            if not isinstance(after_keys, list):
                continue
            bad = [k for k in after_keys if _normalize_plan_key(k) not in known_keys]
            if bad:
                print(
                    f"[FormPipeline] warn functionCall.triggerAfterStepKeys missing keys={bad} stepKey={item.get('key')}",
                    flush=True,
                )
    except Exception:
        pass

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
            "services_summary": str(ctx.get("services_summary") or ctx.get("grounding_summary") or "").strip(),
            "choice_option_min": ctx.get("choice_option_min"),
            "choice_option_max": ctx.get("choice_option_max"),
            "choice_option_target": ctx.get("choice_option_target"),
            "required_uploads": ctx.get("required_uploads") if isinstance(ctx.get("required_uploads"), list) else [],
        }
    )
    render_cache_enabled = _env_bool("AI_FORM_RENDER_CACHE", False)
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
        if (render_cache_enabled and not disable_cache)
        else ""
    )
    if os.getenv("AI_FORM_DEBUG") == "true" and render_cache_key:
        print(f"[FormPipeline] requestId={request_id} renderCacheKey={render_cache_key}", flush=True)
    cached_emitted = _ttl_cache_get(_RENDER_OUTPUT_CACHE, render_cache_key) if render_cache_key else None

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
        parsed_steps = _parse_jsonl_steps(raw_jsonl)
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

            # Special-case deterministic composite trigger steps: they need `blocks`.
            if t == "composite" and _is_initial_image_trigger_plan_item(plan_item):
                composite = _composite_trigger_step_from_plan_item(plan_item)
                if not composite:
                    continue
                candidate = composite
            else:
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

    # Backstop: ensure functionCall metadata is preserved for deterministic trigger steps.
    # Even if the renderer forgets to copy it, we re-attach it from the plan.
    try:
        plan_fc_by_id: Dict[str, Dict[str, Any]] = {}
        for pi in sliced:
            if not isinstance(pi, dict):
                continue
            fc = pi.get("functionCall")
            if not isinstance(fc, dict) or not fc:
                continue
            k = _normalize_plan_key(pi.get("key"))
            if not k:
                continue
            plan_fc_by_id[_derive_step_id_from_key(k)] = dict(fc)
        if plan_fc_by_id and emitted:
            for step in emitted:
                if not isinstance(step, dict):
                    continue
                sid = str(step.get("id") or "").strip()
                if sid and sid in plan_fc_by_id and not isinstance(step.get("functionCall"), dict):
                    step["functionCall"] = plan_fc_by_id[sid]
    except Exception:
        pass

    # Backstop: enforce the initial-image trigger step shape as a composite with blocks + metadata.shared.
    # This makes the step deterministic even if the renderer emits it as a non-composite.
    try:
        trigger_plan = None
        for pi in sliced:
            if isinstance(pi, dict) and _is_initial_image_trigger_plan_item(pi):
                trigger_plan = pi
                break
        if isinstance(trigger_plan, dict) and emitted:
            trigger_key = _normalize_plan_key(trigger_plan.get("key"))
            trigger_id = _derive_step_id_from_key(trigger_key) if trigger_key else ""
            forced = _composite_trigger_step_from_plan_item(trigger_plan)
            if trigger_id and forced:
                for i, step in enumerate(emitted):
                    if not isinstance(step, dict):
                        continue
                    if str(step.get("id") or "").strip() == trigger_id:
                        # Preserve position; replace shape deterministically.
                        validated = _validate_mini(forced, ui_types)
                        if validated:
                            emitted[i] = validated
                        break
    except Exception:
        pass

    # Cache renderer output (validated miniSteps only).
    if render_cache_key and (not disable_cache) and (not render_cache_hit) and emitted:
        ttl_sec = _env_int("AI_FORM_RENDER_CACHE_TTL_SEC", 600)
        _ttl_cache_set(_RENDER_OUTPUT_CACHE, render_cache_key, emitted, ttl_sec=ttl_sec)

    meta: Dict[str, Any] = {"requestId": request_id, "schemaVersion": str(schema_version or "0"), "miniSteps": emitted}

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
        }

    if track_usage:
        lm_usage_by_module: Dict[str, Any] = {}
        usage_planner = _extract_dspy_usage(plan_pred) if plan_pred is not None else None
        usage_renderer = _extract_dspy_usage(pred) if pred is not None else None
        if usage_planner:
            lm_usage_by_module["planner"] = usage_planner
        if usage_renderer:
            lm_usage_by_module["renderer"] = usage_renderer
            # Back-compat: keep `lmUsage` as renderer usage.
            meta["lmUsage"] = usage_renderer
        if lm_usage_by_module:
            meta["lmUsageByModule"] = lm_usage_by_module

    latency_ms = int((time.time() - start_time) * 1000)
    if _env_bool("AI_FORM_LOG_LATENCY", False):
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

