from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from programs.form_pipeline.utils import _normalize_step_id


DEFAULT_PLATFORM_GOAL = "Collect scope details to produce a quick estimate."


def _as_str(x: Any, *, max_len: int) -> str:
    return str(x or "")[:max_len]


def extract_platform_goal(payload: Dict[str, Any], *, default: str = DEFAULT_PLATFORM_GOAL) -> str:
    """
    Extract `platform_goal` from both top-level and nested places.

    Why: the widget often sends context under `state.context`, and we still want the model
    to know what it is supposed to do.
    """

    # Top-level (preferred)
    raw = payload.get("platformGoal") or payload.get("platform_goal")
    goal = _as_str(raw, max_len=600).strip()
    if goal:
        return goal

    # Nested widget shapes
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    state_context = state.get("context") if isinstance(state.get("context"), dict) else {}
    raw2 = (
        state_context.get("platformGoal")
        or state_context.get("platform_goal")
        or state.get("platformGoal")
        or state.get("platform_goal")
    )
    goal = _as_str(raw2, max_len=600).strip()
    if goal:
        return goal

    return str(default or "").strip()


def extract_instance_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    instance_context = payload.get("instanceContext") if isinstance(payload.get("instanceContext"), dict) else {}
    if not instance_context:
        instance_context = payload.get("instance_context") if isinstance(payload.get("instance_context"), dict) else {}
    return instance_context if isinstance(instance_context, dict) else {}


def extract_instance_categories_subcategories(
    payload: Dict[str, Any], *, instance_context: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
    - instance_categories
    - instance_subcategories
    - instance_subcategories (alias list, kept for API parity)
    """

    instance_categories_raw = (instance_context or {}).get("categories") or []
    instance_categories = instance_categories_raw if isinstance(instance_categories_raw, list) else []

    instance_subcategories_raw = (instance_context or {}).get("subcategories") or []
    instance_subcategories = instance_subcategories_raw if isinstance(instance_subcategories_raw, list) else []

    # Back-compat: if callers still send top-level arrays, accept them.
    if not instance_categories:
        legacy = payload.get("instanceCategories") or payload.get("instance_categories") or []
        instance_categories = legacy if isinstance(legacy, list) else []
    if not instance_subcategories:
        legacy = payload.get("instanceSubcategories") or payload.get("instance_subcategories") or []
        instance_subcategories = legacy if isinstance(legacy, list) else []

    return instance_categories, instance_subcategories, instance_subcategories


def derive_industry_and_service_strings(
    payload: Dict[str, Any],
    *,
    instance_context: Dict[str, Any],
    instance_categories: List[Dict[str, Any]],
    instance_subcategories: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Derive short, plain-English industry/service strings for the model.
    """

    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    state_context = state.get("context") if isinstance(state.get("context"), dict) else {}

    industry_names = [str(c.get("name") or "").strip() for c in instance_categories if isinstance(c, dict) and c.get("name")]
    service_names = [str(s.get("name") or "").strip() for s in instance_subcategories if isinstance(s, dict) and s.get("name")]

    if not industry_names:
        ind = (instance_context or {}).get("industry")
        if isinstance(ind, dict) and ind.get("name"):
            industry_names = [str(ind.get("name") or "").strip()]
    if not service_names:
        svc = (instance_context or {}).get("service")
        if isinstance(svc, dict) and svc.get("name"):
            service_names = [str(svc.get("name") or "").strip()]

    industry = str(
        ", ".join(industry_names)
        if industry_names
        else (
            payload.get("industry")
            or payload.get("vertical")
            or state_context.get("industry")
            or state_context.get("categoryName")
            or "General"
        )
    )[:120]

    service = str(
        ", ".join(service_names)
        if service_names
        else (
            payload.get("service")
            or payload.get("subcategoryName")
            or state_context.get("subcategoryName")
            or ""
        )
    )[:120]

    return industry, service


def extract_grounding_summary(payload: Dict[str, Any], *, max_len: int = 800) -> str:
    """
    Extract grounding / RAG summary text from a request payload.

    This mirrors the legacy search order so behavior stays stable.
    """

    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    state_context = state.get("context") if isinstance(state.get("context"), dict) else {}
    instance_context = payload.get("instanceContext") if isinstance(payload.get("instanceContext"), dict) else {}
    if not instance_context:
        instance_context = payload.get("instance_context") if isinstance(payload.get("instance_context"), dict) else {}

    def _coerce(raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, (dict, list)):
            try:
                raw = json.dumps(raw, ensure_ascii=True)
            except Exception:
                raw = str(raw)
        return str(raw).strip()

    keys = (
        # canonical + preferred
        "grounding_summary",
        "groundingSummary",
        # common alternates used in callers
        "grounding_preview",
        "groundingPreview",
        "grounding",
        "rag_summary",
        "ragSummary",
    )

    for container in (payload, instance_context, state, state_context):
        if not isinstance(container, dict) or not container:
            continue
        for key in keys:
            text = _coerce(container.get(key))
            if text:
                return text[: int(max_len or 0) or 800]

    return ""


def extract_session_id(payload: Dict[str, Any]) -> str:
    for key in ("sessionId", "session_id"):
        v = payload.get(key)
        if v:
            return str(v)[:120]
    session = payload.get("session")
    if isinstance(session, dict):
        for key in ("id", "sessionId", "session_id"):
            v = session.get(key)
            if v:
                return str(v)[:120]
    return ""


def extract_known_answers(payload: Dict[str, Any]) -> Dict[str, Any]:
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    state_answers = state.get("answers") if isinstance(state.get("answers"), dict) else {}
    known_answers_raw = payload.get("stepDataSoFar") or payload.get("knownAnswers") or state_answers or {}
    return known_answers_raw if isinstance(known_answers_raw, dict) else {}


def extract_asked_step_ids(payload: Dict[str, Any], *, known_answers: Dict[str, Any]) -> List[str]:
    """
    Normalize asked step ids and backfill from known answers when missing.
    """

    already_asked = payload.get("askedStepIds") or payload.get("alreadyAskedKeys") or payload.get("alreadyAskedKeysJson") or []
    if not already_asked:
        form_state = payload.get("formState") or payload.get("form_state") or {}
        if not isinstance(form_state, dict):
            form_state = {}
        if not form_state:
            state_raw = payload.get("state")
            if isinstance(state_raw, dict):
                nested = state_raw.get("formState") or state_raw.get("form_state")
                if isinstance(nested, dict):
                    form_state = nested
                else:
                    form_state = state_raw
        if isinstance(form_state, dict):
            already_asked = (
                form_state.get("askedStepIds")
                or form_state.get("asked_step_ids")
                or form_state.get("alreadyAskedKeys")
                or form_state.get("already_asked_keys")
                or []
            )

    normalized: List[str] = []
    if isinstance(already_asked, list):
        for x in already_asked:
            t = str(x or "").strip()
            if not t:
                continue
            sid = _normalize_step_id(t)
            if not sid.startswith("step-"):
                continue
            normalized.append(sid)

    if not normalized and isinstance(known_answers, dict) and known_answers:
        for k in list(known_answers.keys()):
            sid = _normalize_step_id(str(k or "").strip())
            if sid and sid.startswith("step-") and sid not in normalized:
                normalized.append(sid)

    return normalized


def extract_answered_qa(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Expected shape: [{ stepId, question, answer }]
    """

    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    answered_qa_raw = payload.get("answeredQA") or payload.get("answered_qa")
    if answered_qa_raw is None and state:
        answered_qa_raw = state.get("answeredQA") or state.get("answered_qa")

    answered_qa: List[Dict[str, str]] = []
    if isinstance(answered_qa_raw, list):
        for item in answered_qa_raw:
            if not isinstance(item, dict):
                continue
            step_id = _normalize_step_id(str(item.get("stepId") or item.get("step_id") or item.get("id") or "").strip())
            question = str(item.get("question") or item.get("q") or "").strip()
            answer = item.get("answer") or item.get("a")
            if answer is None:
                answer_text = ""
            elif isinstance(answer, (dict, list)):
                try:
                    answer_text = json.dumps(answer, ensure_ascii=True)
                except Exception:
                    answer_text = str(answer)
            else:
                answer_text = str(answer)
            answer_text = answer_text.strip()

            if not step_id or not step_id.startswith("step-"):
                continue
            if not question and not answer_text:
                continue
            answered_qa.append({"stepId": step_id, "question": question[:200], "answer": answer_text[:300]})
            if len(answered_qa) >= 24:
                break

    return answered_qa


def _as_int(value: Any) -> Optional[int]:
    try:
        n = int(value)
    except Exception:
        return None
    return n if n > 0 else None


def _get_int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def extract_token_budget(batch_state: Any) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(batch_state, dict):
        return None, None
    total_raw = batch_state.get("tokensTotalBudget")
    used_raw = batch_state.get("tokensUsedSoFar")
    try:
        total = int(total_raw) if total_raw is not None else None
    except Exception:
        total = None
    try:
        used = int(used_raw) if used_raw is not None else None
    except Exception:
        used = None
    return total, used


def extract_form_state_subset(payload: Dict[str, Any], batch_state: Dict[str, Any]) -> Dict[str, Any]:
    form_state: Any = payload.get("formState") or payload.get("form_state") or {}
    if not isinstance(form_state, dict):
        form_state = {}
    if not form_state:
        state_raw = payload.get("state")
        if isinstance(state_raw, dict):
            nested = state_raw.get("formState") or state_raw.get("form_state")
            if isinstance(nested, dict):
                form_state = nested
            else:
                form_state = state_raw

    batch_index = (
        form_state.get("batchIndex")
        or form_state.get("batch_index")
        or form_state.get("batchNumber")
        or form_state.get("batch_number")
    )
    max_batches = (
        form_state.get("maxBatches")
        or form_state.get("max_batches")
        or form_state.get("maxCalls")
        or form_state.get("max_calls")
    )
    calls_remaining = form_state.get("callsRemaining") or form_state.get("calls_remaining")
    if max_batches is None and isinstance(batch_state, dict):
        max_batches = batch_state.get("maxCalls")
    if calls_remaining is None and isinstance(batch_state, dict):
        calls_remaining = batch_state.get("callsRemaining")

    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}
    if batch_index is None and isinstance(current_batch, dict):
        batch_index = current_batch.get("batchNumber") or current_batch.get("batch_number")

    subset: Dict[str, Any] = {}
    if batch_index is not None:
        try:
            subset["batch_index"] = int(batch_index)
        except Exception:
            pass
    if max_batches is not None:
        try:
            subset["max_batches"] = int(max_batches)
        except Exception:
            pass
    if calls_remaining is not None:
        try:
            subset["calls_remaining"] = int(calls_remaining)
        except Exception:
            pass
    return subset


def resolve_backend_max_calls(*, default_max_calls: int = 2) -> int:
    """
    Backend-owned call cap.
    """

    try:
        from programs.form_pipeline.planning import DEFAULT_CONSTRAINTS

        default_max_calls = int((DEFAULT_CONSTRAINTS or {}).get("maxBatches") or default_max_calls)
    except Exception:
        default_max_calls = 2

    return max(1, min(10, _get_int_env("AI_FORM_MAX_BATCH_CALLS", default_max_calls)))


def build_batch_constraints(*, payload: Dict[str, Any], batch_state: Dict[str, Any], max_batches: int) -> Dict[str, Any]:
    """
    Build the backend constraints we share with the frontend (max calls, step limits, token budget).
    """

    default_min_steps_per_batch = 2
    default_max_steps_per_batch = 4
    default_token_budget_total = 3000
    try:
        from programs.form_pipeline.planning import DEFAULT_CONSTRAINTS

        default_min_steps_per_batch = int((DEFAULT_CONSTRAINTS or {}).get("minStepsPerBatch") or default_min_steps_per_batch)
        default_max_steps_per_batch = int((DEFAULT_CONSTRAINTS or {}).get("maxStepsPerBatch") or default_max_steps_per_batch)
        default_token_budget_total = int((DEFAULT_CONSTRAINTS or {}).get("tokenBudgetTotal") or default_token_budget_total)
    except Exception:
        default_min_steps_per_batch = 2
        default_max_steps_per_batch = 4
        default_token_budget_total = 3000

    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}
    min_steps_per_batch = (
        _as_int(payload.get("minStepsPerBatch"))
        or _as_int(payload.get("min_steps_per_batch"))
        or _as_int(current_batch.get("minStepsPerBatch"))
        or _as_int(current_batch.get("min_steps_per_batch"))
        or _as_int(os.getenv("AI_FORM_MIN_STEPS_PER_BATCH"))
        or default_min_steps_per_batch
    )
    max_steps_per_batch = (
        _as_int(payload.get("maxSteps"))
        or _as_int(payload.get("max_steps"))
        or _as_int(current_batch.get("maxSteps"))
        or _as_int(current_batch.get("max_steps"))
        or _as_int(os.getenv("AI_FORM_MAX_STEPS_PER_BATCH"))
        or default_max_steps_per_batch
    )
    if min_steps_per_batch < 1:
        min_steps_per_batch = default_min_steps_per_batch
    if max_steps_per_batch < min_steps_per_batch:
        max_steps_per_batch = min_steps_per_batch

    max_steps_total = _as_int(batch_state.get("max_steps_total")) or _as_int(batch_state.get("maxStepsTotal")) or max_steps_per_batch * max_batches
    token_budget_total = (
        _as_int(batch_state.get("tokensTotalBudget"))
        or _as_int(batch_state.get("token_budget_total"))
        or _as_int(os.getenv("AI_FORM_TOKEN_BUDGET_TOTAL"))
        or default_token_budget_total
    )
    return {
        "maxBatches": max_batches,
        "maxStepsTotal": max_steps_total,
        "minStepsPerBatch": min_steps_per_batch,
        "maxStepsPerBatch": max_steps_per_batch,
        "tokenBudgetTotal": token_budget_total,
    }


def resolve_copy_pack_id(payload: Dict[str, Any]) -> str:
    for key in ("copyPackId", "copy_pack_id", "copyPack", "copy_pack"):
        val = payload.get(key)
        if val:
            return str(val).strip()
    request = payload.get("request")
    if isinstance(request, dict):
        for key in ("copyPackId", "copy_pack_id", "copyPack", "copy_pack"):
            val = request.get(key)
            if val:
                return str(val).strip()
    return "default_v1"


def apply_copy_pack(payload: Dict[str, Any], *, context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Load the copy pack (style + lint rules) and attach to context.
    """

    copy_pack_id = resolve_copy_pack_id(payload)
    lint_config: Dict[str, Any] = {}
    try:
        from programs.form_pipeline.planning import compile_pack, load_pack

        pack = load_pack(copy_pack_id)
        style_snippet_json, lint_config = compile_pack(pack)
        if style_snippet_json:
            context = dict(context)
            context["copy_style"] = style_snippet_json
            context["copy_context"] = style_snippet_json
    except Exception:
        lint_config = {}

    return context, (lint_config if isinstance(lint_config, dict) else {}), copy_pack_id


ATTRIBUTE_FAMILIES_SCENE = [
    {"family": "scene_setting", "goal": "Where/what environment should appear?"},
    {"family": "subject_scope", "goal": "What exactly is being designed and roughly how big/type?"},
    {"family": "materials_finishes", "goal": "Visible materials and finish levels (matte/satin/gloss/etc.)."},
    {"family": "color_palette", "goal": "Color families and contrast (avoid primary-color filler)."},
    {"family": "texture_pattern", "goal": "Texture and pattern density (smooth/rough, solid/subtle/bold)."},
    {"family": "key_features", "goal": "Major visible components/features that change the design."},
    {"family": "layout_composition", "goal": "Arrangement/framing of elements in the scene."},
    {"family": "lighting_time", "goal": "Lighting and time-of-day (day/dusk/night, warm/cool)."},
    {"family": "style_direction", "goal": "Concrete style direction (tie to materials/palette, no abstract art)."},
    {"family": "visual_constraints", "goal": "Only constraints that change what the render can show."},
    {"family": "reference_inputs", "goal": "Reference uploads/inspiration links if needed."},
]

ATTRIBUTE_FAMILIES_TRYON = [
    {"family": "subject_pose_view", "goal": "Pose, angle, and framing of the try-on subject."},
    {"family": "item_type", "goal": "What item is being tried on (category/type)."},
    {"family": "fit_silhouette", "goal": "How the item fits or drapes (slim/regular/oversized)."},
    {"family": "size_proportion", "goal": "Length/coverage proportions (crop/regular/long)."},
    {"family": "materials_finishes", "goal": "Visible materials and finish levels (matte/satin/gloss/etc.)."},
    {"family": "color_palette", "goal": "Color families and contrast (avoid primary-color filler)."},
    {"family": "pattern_texture", "goal": "Pattern/texture density (solid/subtle/bold)."},
    {"family": "styling_accessories", "goal": "Styling pairings or accessories that change the look."},
    {"family": "background_setting", "goal": "Backdrop/environment for the try-on context."},
    {"family": "lighting_time", "goal": "Lighting and time-of-day (day/dusk/night, warm/cool)."},
    {"family": "visual_constraints", "goal": "Only constraints that change what the render can show."},
    {"family": "reference_inputs", "goal": "Reference uploads or base photos for try-on."},
]

ATTRIBUTE_FAMILIES_PRICING = [
    {"family": "project_type", "goal": "Type of work or outcome needed (install, replace, repair)."},
    {"family": "area_location", "goal": "Which area/room/location the work applies to."},
    {"family": "area_size", "goal": "Approx size/measurements to estimate (sq ft, rooms, length)."},
    {"family": "material_preference", "goal": "Material or product preferences that affect pricing."},
    {"family": "condition_constraints", "goal": "Existing condition or constraints that change scope."},
    {"family": "timeline_urgency", "goal": "Desired timing or urgency for the work."},
    {"family": "budget_range", "goal": "Budget range or target spend."},
    {"family": "style_finish", "goal": "Style, pattern, or finish preferences if relevant."},
    {"family": "reference_inputs", "goal": "Reference photos or inspiration if available."},
]


def _normalize_use_case(raw: Any) -> str:
    t = str(raw or "").strip().lower()
    if not t:
        return "scene"
    t = t.replace("_", " ").replace("-", " ").strip()
    if "tryon" in t or "try on" in t:
        return "tryon"
    if "scene placement" in t or "placement" in t:
        return "scene_placement"
    if "scene" in t:
        return "scene"
    return t.replace(" ", "_")


def extract_use_case(payload: Dict[str, Any]) -> str:
    raw = payload.get("useCase") or payload.get("use_case") or payload.get("instanceUseCase") or payload.get("instance_use_case")
    if not raw:
        instance = payload.get("instance") if isinstance(payload.get("instance"), dict) else {}
        if isinstance(instance, dict):
            raw = instance.get("use_case") or instance.get("useCase")
    return _normalize_use_case(raw)


def infer_goal_intent(platform_goal: str, business_context: str) -> str:
    text = f"{platform_goal} {business_context}".lower()
    if "visual_only" in text or "visual-only" in text or "visual only" in text:
        return "visual"
    return "pricing"


def select_attribute_families(use_case: str, goal_intent: str) -> List[Dict[str, str]]:
    if goal_intent == "pricing":
        return list(ATTRIBUTE_FAMILIES_PRICING)
    if use_case == "tryon":
        return list(ATTRIBUTE_FAMILIES_TRYON)
    return list(ATTRIBUTE_FAMILIES_SCENE)


def summarize_instance_subcategories(instance_subcategories: List[Dict[str, Any]], limit: int = 8) -> str:
    if not instance_subcategories:
        return ""
    seen: set[str] = set()
    names: List[str] = []
    for item in instance_subcategories:
        if not isinstance(item, dict):
            continue
        label = str(item.get("subcategory") or "").strip()
        if not label or label.lower() in seen:
            continue
        seen.add(label.lower())
        names.append(label)
        if len(names) >= limit:
            break
    return ", ".join(names)


def build_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the context dict that we send to the planner/renderer.
    """

    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    state_context = state.get("context") if isinstance(state.get("context"), dict) else {}

    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}
    required_uploads_raw = (
        payload.get("requiredUploads")
        or payload.get("required_uploads")
        or current_batch.get("requiredUploads")
        or current_batch.get("required_uploads")
        or []
    )
    required_uploads = required_uploads_raw if isinstance(required_uploads_raw, list) else []

    known_answers = extract_known_answers(payload)
    asked_step_ids = extract_asked_step_ids(payload, known_answers=known_answers)
    answered_qa = extract_answered_qa(payload)

    batch_state_raw = payload.get("batchState") or payload.get("batch_state") or {}
    batch_state = batch_state_raw if isinstance(batch_state_raw, dict) else {}

    items_raw = payload.get("items") or []
    items = items_raw if isinstance(items_raw, list) else []

    instance_context = extract_instance_context(payload)
    instance_categories, instance_subcategories, instance_subcategories_alias = extract_instance_categories_subcategories(
        payload, instance_context=instance_context
    )
    industry, service = derive_industry_and_service_strings(
        payload,
        instance_context=instance_context,
        instance_categories=instance_categories,
        instance_subcategories=instance_subcategories,
    )
    subcategory_summary = summarize_instance_subcategories(instance_subcategories_alias)

    use_case = extract_use_case(payload)
    platform_goal = extract_platform_goal(payload)
    business_context = str(payload.get("businessContext") or payload.get("business_context") or state_context.get("businessContext") or "")[:200]
    goal_intent = infer_goal_intent(platform_goal, business_context)

    grounding_summary = extract_grounding_summary(payload)
    if grounding_summary:
        grounding_summary = grounding_summary[:600]

    attribute_families = select_attribute_families(use_case, goal_intent)

    model_batch = extract_form_state_subset(payload, batch_state)
    backend_max_calls = resolve_backend_max_calls()
    model_batch = dict(model_batch)
    model_batch["max_batches"] = backend_max_calls
    batch_constraints = build_batch_constraints(payload=payload, batch_state=batch_state, max_batches=backend_max_calls)

    # options count bounds
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

    context: Dict[str, Any] = {
        "platform_goal": platform_goal,
        "business_context": business_context,
        "industry": industry,
        "service": service,
        "use_case": use_case,
        "goal_intent": goal_intent,
        "required_uploads": required_uploads,
        "personalization_summary": str(payload.get("personalizationSummary") or payload.get("personalization_summary") or "")[:1200],
        "known_answers": known_answers,
        "asked_step_ids": asked_step_ids,
        "already_asked_keys": asked_step_ids,  # legacy name
        "batch_info": model_batch,
        "form_plan": [],  # intentionally deprecated
        "batch_constraints": batch_constraints,
        "psychology_plan": payload.get("psychologyPlan") or payload.get("psychology_plan") or {},
        "batch_state": batch_state,
        "items": items,
        "instance_context": instance_context,
        "instance_categories": instance_categories,
        "instance_subcategories": instance_subcategories_alias,
        "instance_subcategory_summary": subcategory_summary,
        "attribute_families": attribute_families,
        "grounding_summary": grounding_summary or "",
        "answered_qa": answered_qa,
        "choice_option_min": choice_option_min,
        "choice_option_max": choice_option_max,
        "choice_option_target": choice_option_target,
        "prefer_structured_inputs": False,
    }

    if grounding_summary:
        context["vertical_context"] = grounding_summary

    return context


__all__ = [
    "DEFAULT_PLATFORM_GOAL",
    "apply_copy_pack",
    "build_batch_constraints",
    "build_context",
    "derive_industry_and_service_strings",
    "extract_answered_qa",
    "extract_asked_step_ids",
    "extract_form_state_subset",
    "extract_grounding_summary",
    "extract_instance_categories_subcategories",
    "extract_instance_context",
    "extract_known_answers",
    "extract_platform_goal",
    "extract_session_id",
    "extract_token_budget",
    "extract_use_case",
    "infer_goal_intent",
    "resolve_backend_max_calls",
    "resolve_copy_pack_id",
    "select_attribute_families",
    "summarize_instance_subcategories",
]

