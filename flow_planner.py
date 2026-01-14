"""
Minimal DSPy Flow Planner.

Ported from `sif-widget/dspy/flow_planner.py` into this standalone service repo so we can run DSPy
without spawning a local Python subprocess from Next.js.

---

### How to read this file (DSPy beginner notes)

This module is the **bridge between HTTP** (FastAPI) and **DSPy** (prompted programs).

Data flow:
1. `api/index.py` receives a POST body (payload dict)
2. `api/index.py` calls `next_steps_jsonl(payload)`
3. `next_steps_jsonl` creates a DSPy LM + configures DSPy settings
4. `next_steps_jsonl` creates a DSPy Module: `FlowPlannerModule`
5. DSPy sends a request to the provider (Groq/OpenAI) via LiteLLM
6. We parse the model output, validate it with Pydantic, and return a clean JSON structure

Key DSPy concepts used here:
- **Signature** (`NextStepsJSONL`): describes *inputs* and *outputs* in a declarative way.
- **Predict** (`dspy.Predict(Signature)`): turns that signature into a callable LLM-backed function.
- **Demos**: examples attached to the module's predictor that guide the model (few-shot).

DSPy map for this repo:
- Signature: `modules/signatures/json_signatures.py` → `NextStepsJSONL`
- Predictor: created inside `modules/flow_planner_module.py` via `dspy.Predict(NextStepsJSONL)`
- Module: `modules/flow_planner_module.py` → `FlowPlannerModule`
- Pipeline (future): would be multiple Modules chained in `flow_planner.py`

Why some fields are strings:
- LLM outputs are text. For reliability we ask DSPy to output JSON/JSONL **as strings**,
  then we parse + validate with Pydantic. This makes failures detectable and recoverable.

Token/step constraints:
- Hard cap token length via `dspy.LM(max_tokens=...)` (provider-enforced).
- Keep `max_steps` in the signature as a soft/content constraint, and enforce step limits in runtime parsing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

# Suppress Pydantic serialization warnings from LiteLLM
# These warnings occur when LiteLLM serializes LLM response objects (Message, StreamingChoices)
# and are harmless - they don't affect functionality
warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
    module="pydantic",
)


try:
    BaseExceptionGroup
except NameError:  # pragma: no cover - Python < 3.11
    BaseExceptionGroup = Exception

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

_BANNED_OPTION_SETS = [
    {"red", "blue", "green"},
    {"circle", "square", "triangle"},
]
_BANNED_OPTION_TERMS = {"abstract"}


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _strip_code_fences(s: str) -> str:
    import re

    if not s:
        return s
    t = s.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t, flags=re.IGNORECASE)
    return t.strip()


def _best_effort_parse_json(text: str) -> Any:
    if not text:
        return None
    t = _strip_code_fences(str(text))
    parsed = _safe_json_loads(t)
    if parsed is not None:
        return parsed
    # Fallback: find first array/object block
    import re

    m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", t)
    if not m:
        return None
    return _safe_json_loads(m.group(0))


def _normalize_step_id(step_id: str) -> str:
    """
    Canonicalize step ids to match the Next.js side:
      - underscores -> hyphens
      - preserve leading `step-` prefix
    """
    t = str(step_id or "").strip()
    if not t:
        return t
    return t.replace("_", "-")


def _compact_json(obj: Any) -> str:
    try:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=True, sort_keys=True)
    except Exception:
        return json.dumps(str(obj), separators=(",", ":"), ensure_ascii=True)


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


def _infer_goal_intent(platform_goal: str, business_context: str) -> str:
    text = f"{platform_goal} {business_context}".lower()
    if "visual_only" in text or "visual-only" in text or "visual only" in text:
        return "visual"
    return "pricing"


def _select_attribute_families(use_case: str, goal_intent: str) -> list[dict]:
    if goal_intent == "pricing":
        return ATTRIBUTE_FAMILIES_PRICING
    if use_case == "tryon":
        return ATTRIBUTE_FAMILIES_TRYON
    return ATTRIBUTE_FAMILIES_SCENE




def _extract_use_case(payload: Dict[str, Any]) -> str:
    raw = (
        payload.get("useCase")
        or payload.get("use_case")
        or payload.get("instanceUseCase")
        or payload.get("instance_use_case")
    )
    if not raw:
        instance = payload.get("instance") if isinstance(payload.get("instance"), dict) else {}
        if isinstance(instance, dict):
            raw = instance.get("use_case") or instance.get("useCase")
    return _normalize_use_case(raw)


def _extract_grounding_summary(payload: Dict[str, Any]) -> str:
    for key in (
        "grounding_summary",
        "groundingSummary",
        "grounding_preview",
        "groundingPreview",
        "grounding",
    ):
        raw = payload.get(key)
        if raw:
            if isinstance(raw, (dict, list)):
                try:
                    raw = json.dumps(raw, ensure_ascii=True)
                except Exception:
                    raw = str(raw)
            return str(raw)[:300]
    return ""


def _extract_service_anchor_terms(industry: str, service: str, grounding: str) -> list[str]:
    try:
        from modules.grounding.keywords import extract_service_anchor_terms

        return extract_service_anchor_terms(industry, service, grounding)
    except Exception:
        return []


def _summarize_instance_subcategories(instance_subcategories: list[dict], limit: int = 8) -> str:
    if not instance_subcategories:
        return ""
    seen: set[str] = set()
    names: list[str] = []
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


def _normalize_option_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _option_token_set(step: Dict[str, Any]) -> set[str]:
    options = step.get("options")
    if not isinstance(options, list):
        return set()
    tokens: set[str] = set()
    for opt in options:
        if isinstance(opt, dict):
            label = opt.get("label") or opt.get("value") or ""
        else:
            label = str(opt or "")
        norm = _normalize_option_label(label)
        if not norm:
            continue
        parts = norm.split()
        if len(parts) == 1:
            tokens.add(parts[0])
    return tokens


def _has_banned_option_set(step: Dict[str, Any]) -> bool:
    options = step.get("options")
    if not isinstance(options, list) or not options:
        return False
    tokens = _option_token_set(step)
    for banned in _BANNED_OPTION_SETS:
        if banned.issubset(tokens) and len(tokens) <= len(banned) + 1:
            return True
    for opt in options:
        if isinstance(opt, dict):
            label = str(opt.get("label") or "")
            value = str(opt.get("value") or "")
            combined = f"{label} {value}".lower()
        else:
            combined = str(opt or "").lower()
        if any(term in combined for term in _BANNED_OPTION_TERMS):
            return True
    return False


def _anchor_options(anchor_terms: list[str], limit: int = 4) -> list[dict]:
    options: list[dict] = []
    seen_values: set[str] = set()
    for term in anchor_terms:
        label = str(term or "").strip()
        if not label:
            continue
        value = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        if not value or value in seen_values:
            continue
        seen_values.add(value)
        options.append({"label": label.title(), "value": value})
        if len(options) >= limit:
            break
    return options


def _apply_banned_option_policy(step: Dict[str, Any], anchor_terms: list[str]) -> Optional[Dict[str, Any]]:
    if not _has_banned_option_set(step):
        return step
    if not anchor_terms:
        return None
    if str(step.get("type") or "").lower() not in {
        "choice",
        "multiple_choice",
        "segmented_choice",
        "chips_multi",
        "yes_no",
        "image_choice_grid",
        "searchable_select",
    }:
        return None
    options = _anchor_options(anchor_terms, limit=4)
    if len(options) < 2:
        return None
    if len(options) < 3:
        options.append({"label": "Not sure", "value": "not_sure"})
    step = dict(step)
    step["options"] = options[:5]
    print("[FlowPlanner] ⚠️ Rewrote banned filler options using service anchors", flush=True)
    return step


def _normalize_allowed_mini_types(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [s.strip() for s in str(raw or "").split(",") if s.strip()]


def _allowed_type_matches(step_type: str, allowed: set[str]) -> bool:
    if not allowed:
        return True
    t = str(step_type or "").strip().lower()
    if not t:
        return False
    if t in allowed:
        return True
    if t in ["choice", "multiple_choice", "segmented_choice", "chips_multi", "yes_no", "image_choice_grid"]:
        return "choice" in allowed or "multiple_choice" in allowed
    if t in ["text", "text_input"]:
        return "text" in allowed or "text_input" in allowed
    if t in ["slider", "rating", "range_slider"]:
        return "slider" in allowed or "rating" in allowed or "range_slider" in allowed
    if t in ["upload", "file_upload", "file_picker"]:
        return "upload" in allowed or "file_upload" in allowed or "file_picker" in allowed
    return False


def _extract_required_upload_ids(required_uploads: Any) -> set[str]:
    ids: set[str] = set()
    if not isinstance(required_uploads, list):
        return ids
    for item in required_uploads:
        if not isinstance(item, dict):
            continue
        raw = item.get("stepId") or item.get("step_id") or item.get("id")
        sid = _normalize_step_id(str(raw or ""))
        if sid:
            ids.add(sid)
    return ids


def _looks_like_upload_step_id(step_id: str) -> bool:
    t = str(step_id or "").lower()
    return "upload" in t or "file" in t


def _resolve_copy_pack_id(payload: Dict[str, Any]) -> str:
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


def _summarize_violation_codes(violations: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in violations or []:
        code = str(v.get("code") or "").strip()
        if not code:
            continue
        counts[code] = counts.get(code, 0) + 1
    return counts


def _extract_must_have_copy_needed(batch_state: Any) -> Dict[str, Any]:
    if isinstance(batch_state, dict):
        raw = batch_state.get("mustHaveCopyNeeded")
        if isinstance(raw, dict):
            budget = bool(raw.get("budget") or raw.get("budgetNeeded") or raw.get("needsBudget"))
            uploads_raw = raw.get("uploads") or raw.get("uploadIds") or []
            uploads = [str(x).strip() for x in uploads_raw if str(x or "").strip()] if isinstance(uploads_raw, list) else []
            return {"budget": budget, "uploads": uploads}
        if isinstance(raw, bool):
            return {"budget": raw, "uploads": []}
    return {"budget": False, "uploads": []}


def _extract_token_budget(batch_state: Any) -> tuple[Optional[int], Optional[int]]:
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


def _extract_form_state_subset(payload: Dict[str, Any], batch_state: Dict[str, Any]) -> Dict[str, Any]:
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
    calls_remaining = (
        form_state.get("callsRemaining")
        or form_state.get("calls_remaining")
    )
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


def _build_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    required_uploads_raw = payload.get("requiredUploads") or payload.get("required_uploads") or []
    required_uploads = required_uploads_raw if isinstance(required_uploads_raw, list) else []

    known_answers_raw = payload.get("stepDataSoFar") or payload.get("knownAnswers") or {}
    known_answers = known_answers_raw if isinstance(known_answers_raw, dict) else {}

    already_asked = payload.get("alreadyAskedKeys") or payload.get("alreadyAskedKeysJson") or []
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
            already_asked = form_state.get("alreadyAskedKeys") or form_state.get("already_asked_keys") or []
    normalized_already: list[str] = []
    if isinstance(already_asked, list):
        for x in already_asked:
            t = str(x or "").strip()
            if not t:
                continue
            normalized_already.append(_normalize_step_id(t))

    form_plan_raw = payload.get("formPlan") or payload.get("form_plan") or []
    form_plan = form_plan_raw if isinstance(form_plan_raw, list) else []
    batch_policy = payload.get("batchPolicy") or payload.get("batch_policy") or {}
    psychology_plan = payload.get("psychologyPlan") or payload.get("psychology_plan") or {}

    batch_state_raw = payload.get("batchState") or payload.get("batch_state") or {}
    batch_state = batch_state_raw if isinstance(batch_state_raw, dict) else {}

    items_raw = payload.get("items") or []
    items = items_raw if isinstance(items_raw, list) else []

    instance_subcategories_raw = payload.get("instanceSubcategories") or payload.get("instance_subcategories") or []
    instance_subcategories = instance_subcategories_raw if isinstance(instance_subcategories_raw, list) else []

    industry = str(payload.get("industry") or payload.get("vertical") or "General")[:80]
    service = str(payload.get("service") or payload.get("subcategoryName") or "")[:80]
    use_case = _extract_use_case(payload)
    platform_goal = str(payload.get("platformGoal") or payload.get("platform_goal") or "")[:600]
    business_context = str(payload.get("businessContext") or payload.get("business_context") or "")[:200]
    goal_intent = _infer_goal_intent(platform_goal, business_context)
    grounding_summary = _extract_grounding_summary(payload)
    subcategory_summary = _summarize_instance_subcategories(instance_subcategories)
    combined_grounding = grounding_summary
    if subcategory_summary:
        combined_grounding = f"{combined_grounding} {subcategory_summary}".strip()
    if combined_grounding:
        combined_grounding = combined_grounding[:300]
    service_anchor_terms = _extract_service_anchor_terms(industry, service, combined_grounding)
    attribute_families = _select_attribute_families(use_case, goal_intent)

    model_batch = _extract_form_state_subset(payload, batch_state)
    context = {
        "platform_goal": platform_goal,
        "business_context": business_context,
        "industry": industry,
        "service": service,
        "use_case": use_case,
        "goal_intent": goal_intent,
        "required_uploads": required_uploads,
        "personalization_summary": str(payload.get("personalizationSummary") or payload.get("personalization_summary") or "")[:1200],
        "known_answers": known_answers,
        "already_asked_keys": normalized_already,
        "batch_info": model_batch,
        "form_plan": form_plan,
        "batch_policy": batch_policy if isinstance(batch_policy, dict) else {},
        "psychology_plan": psychology_plan if isinstance(psychology_plan, dict) else {},
        "batch_state": batch_state,
        "items": items,
        "instance_subcategories": instance_subcategories,
        "attribute_families": attribute_families,
        "service_anchor_terms": service_anchor_terms,
    }
    if combined_grounding:
        context["grounding_summary"] = combined_grounding
    # Planner/batch metadata for downstream control
    if payload.get("batchNumber") is not None:
        context["batch_number"] = payload.get("batchNumber")
    return context


def _extract_rigidity(payload: Dict[str, Any], context: Dict[str, Any]) -> float:
    """
    Rigidity controls how strictly we follow the plan for this batch.
    1.0 => only allow steps from the planned `items` list
    0.0 => allow full exploration
    """
    try:
        from modules.form_psychology.policy import normalize_policy, policy_for_batch, policy_for_phase

        policy = payload.get("batchPolicy") if isinstance(payload.get("batchPolicy"), dict) else {}
        policy = normalize_policy(policy) or {}
        batch_number = payload.get("batchNumber")
        if isinstance(batch_number, int) and batch_number > 0:
            r = policy_for_batch(policy, batch_number).get("rigidity")
        else:
            phase = str(payload.get("batchId") or "ContextCore")
            r = policy_for_phase(policy, phase).get("rigidity")
        if r is None:
            return 1.0
        return max(0.0, min(1.0, float(r)))
    except Exception:
        return 1.0


def _allowed_item_ids_from_context(context: Dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    items = context.get("items")
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            sid = str(it.get("id") or "").strip()
            if sid:
                ids.add(sid)
    return ids


def _exploration_budget(max_steps_limit: Optional[int], rigidity: float) -> int:
    if not isinstance(max_steps_limit, int) or max_steps_limit <= 0:
        return 0
    return max(0, min(max_steps_limit, int(round((1.0 - max(0.0, min(1.0, rigidity))) * max_steps_limit))))


def _load_signature_types() -> tuple[Any, Dict[str, Any]]:
    from modules.schemas.ui_steps import (
        BudgetCardsUI,
        ColorPickerUI,
        CompositeUI,
        ConfirmationUI,
        DatePickerUI,
        DesignerUI,
        FileUploadUI,
        GenericUI,
        IntroUI,
        LeadCaptureUI,
        MultipleChoiceUI,
        PricingUI,
        RatingUI,
        SearchableSelectUI,
        TextInputUI,
    )
    from modules.signatures.json_signatures import NextStepsJSONL

    ui_types = {
        "BudgetCardsUI": BudgetCardsUI,
        "ColorPickerUI": ColorPickerUI,
        "CompositeUI": CompositeUI,
        "ConfirmationUI": ConfirmationUI,
        "DatePickerUI": DatePickerUI,
        "DesignerUI": DesignerUI,
        "FileUploadUI": FileUploadUI,
        "GenericUI": GenericUI,
        "IntroUI": IntroUI,
        "LeadCaptureUI": LeadCaptureUI,
        "MultipleChoiceUI": MultipleChoiceUI,
        "PricingUI": PricingUI,
        "RatingUI": RatingUI,
        "SearchableSelectUI": SearchableSelectUI,
        "TextInputUI": TextInputUI,
    }
    return NextStepsJSONL, ui_types


def _clean_options(options: Any) -> list:
    """
    Clean up placeholder values in options.
    Detects and removes options with placeholder values like '<<max_depth>>'.
    """
    if not isinstance(options, list):
        return []

    cleaned = []
    placeholder_patterns = ["<<max_depth>>", "<<max_depth", "max_depth>>", "<max_depth>", "max_depth"]
    removed_count = 0

    for opt in options:
        if isinstance(opt, dict):
            label = str(opt.get("label") or "")
            value = str(opt.get("value") or "")
            # Check if label or value contains placeholder patterns
            is_placeholder = any(
                pattern.lower() in label.lower() or pattern.lower() in value.lower()
                for pattern in placeholder_patterns
            )
            if is_placeholder:
                removed_count += 1
                print(f"[FlowPlanner] 🧹 Removed placeholder option: label='{label}', value='{value}'", flush=True)
            elif label and value:
                cleaned.append(opt)
        elif isinstance(opt, str):
            # Handle simple string options (legacy format)
            is_placeholder = any(pattern.lower() in opt.lower() for pattern in placeholder_patterns)
            if is_placeholder:
                removed_count += 1
                print(f"[FlowPlanner] 🧹 Removed placeholder option: '{opt}'", flush=True)
            else:
                cleaned.append(opt)

    if removed_count > 0:
        print(f"[FlowPlanner] 🧹 Cleaned {removed_count} placeholder option(s), {len(cleaned)} valid option(s) remaining", flush=True)

    return cleaned


def _validate_mini(obj: Any, ui_types: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    t = str(obj.get("type") or obj.get("componentType") or "").lower()
    try:
        if t in ["text", "text_input"]:
            out = ui_types["TextInputUI"].model_validate(obj).model_dump(by_alias=True)
            out["id"] = _normalize_step_id(str(out.get("id") or ""))
            return out
        if t in ["choice", "multiple_choice", "segmented_choice", "chips_multi", "yes_no", "image_choice_grid"]:
            # Repair common LLM failure modes: missing options or placeholder values.
            obj = dict(obj)
            step_id = str(obj.get("id") or "")
            if "options" not in obj or not obj.get("options"):
                print(f"[FlowPlanner] ⚠️ Step '{step_id}': Missing options, using fallback", flush=True)
                obj["options"] = [{"label": "Not sure", "value": "not_sure"}]
            else:
                original_count = len(obj.get("options", []))
                # Clean up placeholder values
                cleaned_options = _clean_options(obj.get("options"))
                if not cleaned_options:
                    # If all options were placeholders, use fallback
                    print(f"[FlowPlanner] ⚠️ Step '{step_id}': All {original_count} option(s) were placeholders, using fallback", flush=True)
                    obj["options"] = [{"label": "Not sure", "value": "not_sure"}]
                elif len(cleaned_options) < original_count:
                    print(f"[FlowPlanner] ✅ Step '{step_id}': Cleaned options ({original_count} -> {len(cleaned_options)})", flush=True)
                    obj["options"] = cleaned_options
                else:
                    obj["options"] = cleaned_options
            out = ui_types["MultipleChoiceUI"].model_validate(obj).model_dump(by_alias=True)
            out["id"] = _normalize_step_id(step_id)
            return out
        if t in ["slider", "rating", "range_slider"]:
            out = ui_types["RatingUI"].model_validate(obj).model_dump(by_alias=True)
            out["id"] = _normalize_step_id(str(out.get("id") or ""))
            return out
        if t in ["budget_cards"]:
            out = ui_types["BudgetCardsUI"].model_validate(obj).model_dump(by_alias=True)
            out["id"] = _normalize_step_id(str(out.get("id") or ""))
            return out
        if t in ["upload", "file_upload", "file_picker"]:
            out = ui_types["FileUploadUI"].model_validate(obj).model_dump(by_alias=True)
            out["id"] = _normalize_step_id(str(out.get("id") or ""))
            return out
        if t in ["intro"]:
            out = ui_types["IntroUI"].model_validate(obj).model_dump(by_alias=True)
            out["id"] = _normalize_step_id(str(out.get("id") or ""))
            return out
        if t in ["date_picker"]:
            out = ui_types["DatePickerUI"].model_validate(obj).model_dump(by_alias=True)
            out["id"] = _normalize_step_id(str(out.get("id") or ""))
            return out
        if t in ["color_picker"]:
            out = ui_types["ColorPickerUI"].model_validate(obj).model_dump(by_alias=True)
            out["id"] = _normalize_step_id(str(out.get("id") or ""))
            return out
        if t in ["searchable_select"]:
            # Repair common LLM failure modes: missing options or placeholder values.
            obj = dict(obj)
            step_id = str(obj.get("id") or "")
            if "options" not in obj or not obj.get("options"):
                print(f"[FlowPlanner] ⚠️ Step '{step_id}': Missing options, using fallback", flush=True)
                obj["options"] = [{"label": "Not sure", "value": "not_sure"}]
            else:
                original_count = len(obj.get("options", []))
                # Clean up placeholder values
                cleaned_options = _clean_options(obj.get("options"))
                if not cleaned_options:
                    # If all options were placeholders, use fallback
                    print(f"[FlowPlanner] ⚠️ Step '{step_id}': All {original_count} option(s) were placeholders, using fallback", flush=True)
                    obj["options"] = [{"label": "Not sure", "value": "not_sure"}]
                elif len(cleaned_options) < original_count:
                    print(f"[FlowPlanner] ✅ Step '{step_id}': Cleaned options ({original_count} -> {len(cleaned_options)})", flush=True)
                    obj["options"] = cleaned_options
                else:
                    obj["options"] = cleaned_options
            out = ui_types["SearchableSelectUI"].model_validate(obj).model_dump(by_alias=True)
            out["id"] = _normalize_step_id(step_id)
            return out
        if t in ["lead_capture"]:
            out = ui_types["LeadCaptureUI"].model_validate(obj).model_dump(by_alias=True)
            out["id"] = _normalize_step_id(str(out.get("id") or ""))
            return out
        if t in ["pricing"]:
            out = ui_types["PricingUI"].model_validate(obj).model_dump(by_alias=True)
            out["id"] = _normalize_step_id(str(out.get("id") or ""))
            return out
        if t in ["confirmation"]:
            out = ui_types["ConfirmationUI"].model_validate(obj).model_dump(by_alias=True)
            out["id"] = _normalize_step_id(str(out.get("id") or ""))
            return out
        if t in ["designer"]:
            out = ui_types["DesignerUI"].model_validate(obj).model_dump(by_alias=True)
            out["id"] = _normalize_step_id(str(out.get("id") or ""))
            return out
        if t in ["composite"]:
            # Repair common LLM failure mode: missing blocks.
            if "blocks" not in obj or not obj.get("blocks"):
                obj = dict(obj)
                obj["blocks"] = []
            out = ui_types["CompositeUI"].model_validate(obj).model_dump(by_alias=True)
            out["id"] = _normalize_step_id(str(out.get("id") or ""))
            return out

        # Fallback for other UI types found in ComponentType union
        return ui_types["GenericUI"].model_validate(obj).model_dump(by_alias=True)
    except Exception:
        return None


def _prepare_predictor(payload: Dict[str, Any]) -> Dict[str, Any]:
    import time as _time

    request_id = f"next_steps_{int(_time.time() * 1000)}"
    start_time = _time.time()
    schema_version = (
        payload.get("schemaVersion")
        or payload.get("schema_version")
        or _best_effort_contract_schema_version()
    )

    lm_cfg = _make_dspy_lm()
    if not lm_cfg:
        return {
            "error": "DSPy LM not configured",
            "request_id": request_id,
            "schema_version": str(schema_version) if schema_version else "0",
        }

    try:
        import dspy  # type: ignore
    except Exception:
        return {
            "error": "DSPy import failed",
            "request_id": request_id,
            "schema_version": str(schema_version) if schema_version else "0",
        }

    llm_timeout = float(os.getenv("DSPY_LLM_TIMEOUT_SEC") or "20")
    temperature = float(os.getenv("DSPY_TEMPERATURE") or "0.7")
    default_max_tokens = int(os.getenv("DSPY_NEXT_STEPS_MAX_TOKENS") or "2000")
    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}
    request_flags = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    max_tokens_override = (
        (current_batch or {}).get("maxTokens")
        or (request_flags or {}).get("maxTokens")
        or payload.get("maxTokensThisCall")
        or payload.get("max_tokens_this_call")
    )
    max_tokens = default_max_tokens
    if max_tokens_override is not None:
        try:
            max_tokens = int(max_tokens_override)
            if max_tokens < 1:
                max_tokens = default_max_tokens
        except Exception:
            max_tokens = default_max_tokens
    else:
        batch_state_raw = payload.get("batchState") or payload.get("batch_state") or {}
        tokens_total, tokens_used = _extract_token_budget(batch_state_raw)
        if isinstance(tokens_total, int) and tokens_total > 0:
            used = tokens_used if isinstance(tokens_used, int) and tokens_used > 0 else 0
            remaining = tokens_total - used
            if remaining <= 0:
                return {
                    "error": "Token budget exhausted",
                    "request_id": request_id,
                    "schema_version": str(schema_version) if schema_version else "0",
                }
            max_tokens = min(default_max_tokens, remaining)
            if max_tokens < 1:
                return {
                    "error": "Token budget exhausted",
                    "request_id": request_id,
                    "schema_version": str(schema_version) if schema_version else "0",
                }

    lm = dspy.LM(
        model=lm_cfg["model"],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=llm_timeout,
        num_retries=0,
    )
    track_usage = _configure_dspy(lm)
    if track_usage:
        print("[FlowPlanner] ✅ DSPy LM usage tracking enabled", flush=True)

    NextStepsJSONL, ui_types = _load_signature_types()
    from modules.flow_planner_module import FlowPlannerModule

    module = FlowPlannerModule()

    try:
        from examples.registry import as_dspy_examples, load_examples_pack

        compiled_override = os.getenv("DSPY_NEXT_STEPS_COMPILED") or ""
        compiled_default = str(Path(__file__).resolve().parent / "compiled" / "next_steps_compiled.jsonl")
        if compiled_override and os.path.exists(compiled_override):
            demo_pack = compiled_override
            print(f"[FlowPlanner] ✅ Using compiled demos: {demo_pack}", flush=True)
        elif os.path.exists(compiled_default):
            demo_pack = compiled_default
            print(f"[FlowPlanner] ✅ Using compiled demos: {demo_pack}", flush=True)
        else:
            demo_pack = _default_next_steps_demo_pack()
        demos = as_dspy_examples(
            load_examples_pack(demo_pack),
            input_keys=[
                "context_json",
                "batch_id",
                "max_steps",
                "allowed_mini_types",
            ],
        )
        if demos:
            setattr(module.prog, "demos", demos)
    except Exception:
        pass

    batch_id_raw = payload.get("batchId") or payload.get("batch_id")
    if not batch_id_raw:
        return {
            "error": "Missing batchId in request payload",
            "request_id": request_id,
            "schema_version": str(schema_version) if schema_version else "0",
        }
    batch_id = str(batch_id_raw)[:40]
    copy_pack_id = _resolve_copy_pack_id(payload)
    style_snippet_json = ""
    lint_config: Dict[str, Any] = {}
    try:
        from modules.copywriting.compiler import compile_pack, load_pack

        pack = load_pack(copy_pack_id)
        style_snippet_json, lint_config = compile_pack(pack)
    except Exception as e:
        print(f"[FlowPlanner] ⚠️ Copy pack load failed: {e}", flush=True)

    context = _build_context(payload)
    if style_snippet_json:
        context["copy_style"] = style_snippet_json
    context_json = _compact_json(context)
    must_have_copy_needed = _extract_must_have_copy_needed(context.get("batch_state"))
    copy_context = dict(context)
    copy_context["must_have_copy_needed"] = must_have_copy_needed
    copy_context_json = _compact_json(copy_context)

    max_steps_raw = (
        payload.get("maxStepsThisCall")
        or payload.get("max_steps_this_call")
        or payload.get("maxSteps")
        or payload.get("max_steps")
        or "4"
    )
    try:
        max_steps = int(str(max_steps_raw))
        if max_steps < 1:
            max_steps = 4
    except Exception:
        max_steps = 4
    allowed_mini_types = _normalize_allowed_mini_types(payload.get("allowedMiniTypes") or payload.get("allowed_mini_types") or [])
    already_asked_keys = set(context.get("already_asked_keys") or [])
    required_upload_ids = _extract_required_upload_ids(context.get("required_uploads"))

    inputs = {
        "context_json": context_json,
        "batch_id": batch_id,
        "max_steps": max_steps,
        "allowed_mini_types": allowed_mini_types,
    }

    return {
        "request_id": request_id,
        "start_time": start_time,
        "schema_version": str(schema_version) if schema_version else "0",
        "module": module,
        "inputs": inputs,
        "context": context,
        "context_json": context_json,
        "copy_context_json": copy_context_json,
        "must_have_copy_needed": must_have_copy_needed,
        "copy_needed": bool(must_have_copy_needed.get("budget")) or bool(must_have_copy_needed.get("uploads")),
        "max_steps": max_steps,
        "allowed_mini_types": allowed_mini_types,
        "already_asked_keys": already_asked_keys,
        "required_upload_ids": required_upload_ids,
        "service_anchor_terms": context.get("service_anchor_terms") or [],
        "lm_cfg": lm_cfg,
        "track_usage": track_usage,
        "ui_types": ui_types,
        "lint_config": lint_config,
        "style_snippet_json": style_snippet_json,
        "copy_pack_id": lint_config.get("pack_id") or copy_pack_id,
        "copy_pack_version": lint_config.get("pack_version") or "",
    }


def next_steps_jsonl(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Single-call NEXT STEPS generator.

    - DSPy decides which questions to ask next based on batch_state + known_answers.
    - Returns a meta JSON object with validated mini steps.
    """
    prep = _prepare_predictor(payload)
    error = prep.get("error")
    if error:
        return {
            "ok": False,
            "error": error,
            "requestId": prep.get("request_id"),
            "schemaVersion": prep.get("schema_version", "0"),
        }

    module = prep["module"]
    inputs = prep["inputs"]
    ui_types = prep["ui_types"]
    request_id = prep["request_id"]
    start_time = prep["start_time"]
    lm_cfg = prep["lm_cfg"]
    track_usage = prep.get("track_usage", False)
    copy_needed = prep.get("copy_needed", False)
    copy_context_json = prep.get("copy_context_json", "")
    max_steps = prep.get("max_steps", 4)
    max_steps_limit = max_steps if isinstance(max_steps, int) and max_steps > 0 else None
    allowed_set = set(prep.get("allowed_mini_types") or [])
    already_asked_keys = prep.get("already_asked_keys") or set()
    required_upload_ids = prep.get("required_upload_ids") or set()
    service_anchor_terms = prep.get("service_anchor_terms") or []
    lint_config = prep.get("lint_config") or {}
    context = prep.get("context") or {}
    context = prep.get("context") or {}
    apply_reassurance = None
    lint_steps = None
    sanitize_steps = None
    if lint_config:
        try:
            from modules.copywriting.linter import (
                apply_reassurance as _apply_reassurance,
                lint_steps as _lint_steps,
                sanitize_steps as _sanitize_steps,
            )

            apply_reassurance = _apply_reassurance
            lint_steps = _lint_steps
            sanitize_steps = _sanitize_steps
        except Exception:
            pass

    # Stage 0: Form Planner (one-time). When no plan exists yet, run a dedicated planner module first,
    # then run the batch generator with that plan. This keeps responsibilities separated.
    context = prep.get("context") or {}
    created_plan = None
    created_batch_policy = None
    created_psychology_plan = None
    if not (isinstance(context.get("form_plan"), list) and context.get("form_plan")):
        try:
            from modules.form_psychology.form_plan import is_first_batch, parse_produced_form_plan_json
            from modules.form_psychology.policy import (
                default_batch_policy,
                default_psychology_plan,
                parse_batch_policy_json,
                parse_psychology_plan_json,
                policy_for_phase,
            )

            if is_first_batch(payload) and hasattr(module, "plan_form"):
                plan_pred = module.plan_form(
                    context_json=str(inputs.get("context_json") or ""),
                    batch_id=str(inputs.get("batch_id") or ""),
                )
                created_plan = parse_produced_form_plan_json(getattr(plan_pred, "form_plan_json", "") or "")
                created_batch_policy = parse_batch_policy_json(getattr(plan_pred, "batch_policy_json", "") or "")
                created_psychology_plan = parse_psychology_plan_json(getattr(plan_pred, "psychology_plan_json", "") or "")

                # If planner didn't produce anything valid, fall back to deterministic defaults.
                if not created_plan:
                    created_plan = None
                if not created_batch_policy:
                    created_batch_policy = default_batch_policy(goal_intent=str(context.get("goal_intent") or "pricing"))
                if not created_psychology_plan:
                    created_psychology_plan = default_psychology_plan(
                        goal_intent=str(context.get("goal_intent") or "pricing"),
                        use_case=str(context.get("use_case") or "scene"),
                    )

                if created_plan:
                    context = dict(context)
                    context["form_plan"] = created_plan
                    context["batch_policy"] = created_batch_policy or {}
                    context["psychology_plan"] = created_psychology_plan or {}
                    prep["context"] = context
                    inputs = dict(inputs)
                    inputs["context_json"] = _compact_json(context)
                    # Apply planner-driven constraints immediately for this call when available.
                    if created_batch_policy:
                        phase = str(inputs.get("batch_id") or "ContextCore")
                        constraints = policy_for_phase(created_batch_policy, phase)
                        if isinstance(constraints.get("allowedMiniTypes"), list) and constraints.get("allowedMiniTypes"):
                            inputs["allowed_mini_types"] = constraints["allowedMiniTypes"]
                            prep["allowed_mini_types"] = constraints["allowedMiniTypes"]
                        if constraints.get("maxSteps") is not None:
                            try:
                                inputs["max_steps"] = int(constraints["maxSteps"])
                                prep["max_steps"] = int(constraints["maxSteps"])
                            except Exception:
                                pass
                    prep["inputs"] = inputs
                    # Refresh local caps/filters using possibly-updated inputs.
                    try:
                        max_steps = int(inputs.get("max_steps"))
                        max_steps_limit = max_steps if max_steps > 0 else None
                    except Exception:
                        pass
                    allowed_set = set(inputs.get("allowed_mini_types") or prep.get("allowed_mini_types") or [])
        except Exception:
            pass

    pred = module(**inputs)

    # Log the raw DSPy response for debugging
    print(f"[FlowPlanner] Raw DSPy response fields: {list(pred.__dict__.keys()) if hasattr(pred, '__dict__') else 'N/A'}", flush=True)
    raw_mini_steps = getattr(pred, "mini_steps_jsonl", None) or ""
    produced_form_plan_json = getattr(pred, "produced_form_plan_json", None)
    print(f"[FlowPlanner] Raw mini_steps_jsonl (first 500 chars): {str(raw_mini_steps)[:500]}", flush=True)
    emitted: list[dict] = []
    seen_ids: set[str] = set()
    rigidity = _extract_rigidity(payload, prep.get("context") or {})
    allowed_item_ids = _allowed_item_ids_from_context(prep.get("context") or {})
    exploration_left = _exploration_budget(max_steps_limit, rigidity)
    # 6) DSPy output is an object that exposes fields defined in the Signature.
    #    Here we read the *string* `mini_steps_jsonl` and parse line-by-line.
    raw_lines = raw_mini_steps

    if raw_lines:
        for line in str(raw_lines).splitlines():
            line = line.strip()
            if not line:
                continue
            if max_steps_limit and len(emitted) >= max_steps_limit:
                break
            obj = _best_effort_parse_json(line)
            v = _validate_mini(obj, ui_types)
            if v:
                sid = str(v.get("id") or "")
                stype = str(v.get("type") or "")
                if sid:
                    if sid in already_asked_keys:
                        print(f"[FlowPlanner] ⚠️ Skipping already asked step: {sid}", flush=True)
                        continue
                    if sid in seen_ids:
                        continue
                    if allowed_item_ids and sid not in allowed_item_ids:
                        if exploration_left <= 0:
                            continue
                        exploration_left -= 1
                    if not _allowed_type_matches(stype, allowed_set):
                        print(f"[FlowPlanner] ⚠️ Skipping disallowed step type '{stype}' for {sid or 'unknown'}", flush=True)
                        continue
                    v = _apply_banned_option_policy(v, service_anchor_terms)
                    if not v:
                        print(f"[FlowPlanner] ⚠️ Skipping step with banned filler options: {sid or 'unknown'}", flush=True)
                        continue
                if sid in required_upload_ids and stype.lower() not in ["upload", "file_upload", "file_picker"]:
                    print(f"[FlowPlanner] ⚠️ Skipping upload step with non-upload type: {sid} ({stype})", flush=True)
                    continue
                if _looks_like_upload_step_id(sid) and stype.lower() in ["text", "text_input"]:
                    print(f"[FlowPlanner] ⚠️ Skipping upload-like id with text type: {sid} ({stype})", flush=True)
                    continue
                if _looks_like_upload_step_id(sid) and stype.lower() in ["text", "text_input"]:
                    print(f"[FlowPlanner] ⚠️ Skipping upload-like id with text type: {sid} ({stype})", flush=True)
                    continue
                if sid:
                    seen_ids.add(sid)
                if max_steps_limit and len(emitted) >= max_steps_limit:
                    break
                emitted.append(v)

    # Log the validated/inflated output
    print(f"[FlowPlanner] Validated steps count: {len(emitted)}", flush=True)
    for i, step in enumerate(emitted):
        print(f"[FlowPlanner] Step {i+1}: id={step.get('id')}, type={step.get('type')}, question={step.get('question', '')[:60]}...", flush=True)

    violations: list[dict] = []
    lint_failed = False
    if sanitize_steps and emitted:
        emitted = sanitize_steps(emitted, lint_config)
    if apply_reassurance and emitted:
        emitted = apply_reassurance(emitted, lint_config)
    if lint_steps:
        try:
            ok, violations, _bad_ids = lint_steps(emitted, lint_config)
            if not ok:
                lint_failed = True
                print(f"[FlowPlanner] ❌ Copy lint failed ({len(violations)} violations)", flush=True)
                for v in violations[:20]:
                    print(f"[FlowPlanner]   - {v}", flush=True)
        except Exception as e:
            print(f"[FlowPlanner] ⚠️ Copy lint failed to run: {e}", flush=True)

    latency_ms = int((time.time() - start_time) * 1000)
    meta = {
        "requestId": request_id,
        "miniSteps": emitted,
        "copyPackId": prep.get("copy_pack_id"),
        "copyPackVersion": prep.get("copy_pack_version"),
        "lintFailed": lint_failed,
        "lintViolationCodes": _summarize_violation_codes(violations),
    }
    # Pass through session info from payload if available
    session_info = payload.get("session")
    if isinstance(session_info, dict):
        meta["session"] = session_info
    final_plan_for_ui = None
    existing_batch_policy = payload.get("batchPolicy") if isinstance(payload.get("batchPolicy"), dict) else None
    existing_psychology_plan = payload.get("psychologyPlan") if isinstance(payload.get("psychologyPlan"), dict) else None
    try:
        from modules.form_psychology.form_plan import finalize_form_plan

        final_plan, did_generate = finalize_form_plan(
            payload=payload,
            context=context,
            produced_form_plan_json=produced_form_plan_json,
        )
        if did_generate and final_plan is not None:
            meta["formPlan"] = final_plan
            final_plan_for_ui = final_plan
    except Exception:
        pass
    # Prefer the dedicated planner's outputs when available, otherwise fill defaults.
    if created_batch_policy and not existing_batch_policy:
        meta["batchPolicy"] = created_batch_policy
    if created_psychology_plan and not existing_psychology_plan:
        meta["psychologyPlan"] = created_psychology_plan
    if not existing_batch_policy or not existing_psychology_plan:
        try:
            from modules.form_psychology.policy import (
                default_batch_policy,
                default_psychology_plan,
                normalize_policy,
            )

            context_for_policy = prep.get("context") or {}
            goal_intent = str(context_for_policy.get("goal_intent") or "pricing")
            use_case = str(context_for_policy.get("use_case") or "scene")
            if not existing_batch_policy:
                meta["batchPolicy"] = normalize_policy(default_batch_policy(goal_intent=goal_intent))
            if not existing_psychology_plan:
                meta["psychologyPlan"] = default_psychology_plan(goal_intent=goal_intent, use_case=use_case)
        except Exception:
            pass
    if copy_needed and hasattr(module, "generate_copy"):
        try:
            copy_pred = module.generate_copy(
                context_json=copy_context_json,
                mini_steps_jsonl=str(raw_mini_steps or ""),
            )
            copy_json = getattr(copy_pred, "must_have_copy_json", None) or ""
            parsed_copy = _parse_must_have_copy(copy_json)
            if parsed_copy:
                meta["mustHaveCopy"] = parsed_copy
            if track_usage:
                copy_usage = _extract_dspy_usage(copy_pred)
                if copy_usage:
                    meta["lmUsageCopy"] = copy_usage
        except Exception:
            pass
    if track_usage:
        usage = _extract_dspy_usage(pred)
        if usage:
            meta["lmUsage"] = usage

    # Optional deterministic wrapping: combine last AI step + uploads into one composite UI step.
    # This avoids spending LLM budget on deterministic UI while still keeping ordering in one step.
    wrapped_steps = None
    try:
        from modules.form_psychology.composite import wrap_last_step_with_upload_composite

        wrapped_steps, did_wrap = wrap_last_step_with_upload_composite(
            payload=payload,
            emitted_steps=meta.get("miniSteps") or [],
            required_uploads=context.get("required_uploads"),
        )
        if did_wrap and wrapped_steps is not None:
            meta["miniSteps"] = wrapped_steps
    except Exception:
        did_wrap = False

    # If we did not wrap uploads into a composite step, provide uiPlan placements instead.
    if not did_wrap and final_plan_for_ui is not None:
        try:
            from modules.form_psychology.ui_plan import build_ui_plan

            ui_plan = build_ui_plan(
                payload=payload,
                final_form_plan=final_plan_for_ui,
                emitted_mini_steps=meta.get("miniSteps") or [],
                required_uploads=context.get("required_uploads"),
            )
            if ui_plan is not None:
                meta["uiPlan"] = ui_plan
        except Exception:
            pass

    # Log the final response meta
    print(
        f"[FlowPlanner] Final response: requestId={meta['requestId']}, latencyMs={latency_ms}, steps={len(meta['miniSteps'])}, model={lm_cfg.get('modelName') or lm_cfg.get('model')}",
        flush=True,
    )

    return meta


async def stream_next_steps_jsonl(payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
    prep = _prepare_predictor(payload)
    error = prep.get("error")
    if error:
        yield {
            "event": "error",
            "data": {
                "message": error,
                "type": "PlannerSetupError",
                "requestId": prep.get("request_id"),
                "schemaVersion": prep.get("schema_version", "0"),
            },
        }
        return

    module = prep["module"]
    inputs = prep["inputs"]
    ui_types = prep["ui_types"]
    request_id = prep["request_id"]
    start_time = prep["start_time"]
    lm_cfg = prep["lm_cfg"]
    track_usage = prep.get("track_usage", False)
    copy_needed = prep.get("copy_needed", False)
    copy_context_json = prep.get("copy_context_json", "")
    max_steps = prep.get("max_steps", 4)
    max_steps_limit = max_steps if isinstance(max_steps, int) and max_steps > 0 else None
    allowed_set = set(prep.get("allowed_mini_types") or [])
    already_asked_keys = prep.get("already_asked_keys") or set()
    required_upload_ids = prep.get("required_upload_ids") or set()
    service_anchor_terms = prep.get("service_anchor_terms") or []
    lint_config = prep.get("lint_config") or {}
    apply_reassurance = None
    lint_steps = None
    sanitize_step = None
    if lint_config:
        try:
            from modules.copywriting.linter import (
                apply_reassurance as _apply_reassurance,
                lint_steps as _lint_steps,
                sanitize_step as _sanitize_step,
            )

            apply_reassurance = _apply_reassurance
            lint_steps = _lint_steps
            sanitize_step = _sanitize_step
        except Exception:
            pass

    try:
        import dspy  # type: ignore
        from dspy.streaming import StatusMessage, StreamListener, StreamResponse
    except Exception as e:
        yield {
            "event": "error",
            "data": {"message": f"DSPy streaming unavailable: {e}", "type": "StreamingSetupError"},
        }
        return

    # Stage 0: Form Planner (one-time). Run before streaming generation when no plan exists yet.
    created_batch_policy = None
    created_psychology_plan = None
    context = prep.get("context") or {}
    if not (isinstance(context.get("form_plan"), list) and context.get("form_plan")):
        try:
            from modules.form_psychology.form_plan import is_first_batch, parse_produced_form_plan_json
            from modules.form_psychology.policy import (
                default_batch_policy,
                default_psychology_plan,
                parse_batch_policy_json,
                parse_psychology_plan_json,
                policy_for_phase,
            )

            if is_first_batch(payload) and hasattr(module, "plan_form"):
                plan_pred = module.plan_form(
                    context_json=str(inputs.get("context_json") or ""),
                    batch_id=str(inputs.get("batch_id") or ""),
                )
                created_plan = parse_produced_form_plan_json(getattr(plan_pred, "form_plan_json", "") or "")
                created_batch_policy = parse_batch_policy_json(getattr(plan_pred, "batch_policy_json", "") or "")
                created_psychology_plan = parse_psychology_plan_json(getattr(plan_pred, "psychology_plan_json", "") or "")

                if not created_batch_policy:
                    created_batch_policy = default_batch_policy(goal_intent=str(context.get("goal_intent") or "pricing"))
                if not created_psychology_plan:
                    created_psychology_plan = default_psychology_plan(
                        goal_intent=str(context.get("goal_intent") or "pricing"),
                        use_case=str(context.get("use_case") or "scene"),
                    )

                if created_plan:
                    context = dict(context)
                    context["form_plan"] = created_plan
                    context["batch_policy"] = created_batch_policy or {}
                    context["psychology_plan"] = created_psychology_plan or {}
                    prep["context"] = context
                    inputs = dict(inputs)
                    inputs["context_json"] = _compact_json(context)
                    if created_batch_policy:
                        phase = str(inputs.get("batch_id") or "ContextCore")
                        constraints = policy_for_phase(created_batch_policy, phase)
                        if isinstance(constraints.get("allowedMiniTypes"), list) and constraints.get("allowedMiniTypes"):
                            inputs["allowed_mini_types"] = constraints["allowedMiniTypes"]
                            allowed_set = set(constraints["allowedMiniTypes"])
                        if constraints.get("maxSteps") is not None:
                            try:
                                max_steps = int(constraints["maxSteps"])
                                inputs["max_steps"] = max_steps
                                max_steps_limit = max_steps if max_steps > 0 else None
                            except Exception:
                                pass
                    prep["inputs"] = inputs
        except Exception:
            pass

    rigidity = _extract_rigidity(payload, prep.get("context") or {})
    allowed_item_ids = _allowed_item_ids_from_context(prep.get("context") or {})
    exploration_left = _exploration_budget(max_steps_limit, rigidity)
    buffer = ""
    emitted: list[dict] = []
    seen_ids: set[str] = set()
    had_chunks = False
    final_pred = None
    reached_cap = False
    lint_violations: list[dict] = []

    max_retries, backoff_ms = _dspy_stream_retry_config()
    stream_error: BaseException | None = None
    attempt = 0

    while True:
        stream_error = None
        listener = StreamListener(signature_field_name="mini_steps_jsonl")
        stream_predict = dspy.streamify(
            module,
            stream_listeners=[listener],
            include_final_prediction_in_output_stream=True,
        )

        stream = None
        try:
            stream = stream_predict(**inputs)
            async for item in stream:
                if isinstance(item, StreamResponse):
                    if item.signature_field_name != "mini_steps_jsonl":
                        continue
                    had_chunks = True
                    buffer += str(item.chunk or "")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        obj = _best_effort_parse_json(line)
                        v = _validate_mini(obj, ui_types)
                        if v:
                            if sanitize_step:
                                v = sanitize_step(v, lint_config)
                            if apply_reassurance:
                                v = apply_reassurance([v], lint_config)[0]
                            if lint_steps:
                                try:
                                    ok, violations, _bad_ids = lint_steps([v], lint_config)
                                    if not ok:
                                        lint_violations.extend(violations)
                                except Exception:
                                    pass
                            sid = str(v.get("id") or "")
                            stype = str(v.get("type") or "")
                            if sid:
                                if sid in already_asked_keys:
                                    print(f"[FlowPlanner] ⚠️ Skipping already asked step: {sid}", flush=True)
                                    continue
                                if sid in seen_ids:
                                    continue
                                if allowed_item_ids and sid not in allowed_item_ids:
                                    if exploration_left <= 0:
                                        continue
                                    exploration_left -= 1
                            if not _allowed_type_matches(stype, allowed_set):
                                print(
                                    f"[FlowPlanner] ⚠️ Skipping disallowed step type '{stype}' for {sid or 'unknown'}",
                                    flush=True,
                                )
                                continue
                            v = _apply_banned_option_policy(v, service_anchor_terms)
                            if not v:
                                print(
                                    f"[FlowPlanner] ⚠️ Skipping step with banned filler options: {sid or 'unknown'}",
                                    flush=True,
                                )
                                continue
                            if sid in required_upload_ids and stype.lower() not in ["upload", "file_upload", "file_picker"]:
                                print(f"[FlowPlanner] ⚠️ Skipping upload step with non-upload type: {sid} ({stype})", flush=True)
                                continue
                            if _looks_like_upload_step_id(sid) and stype.lower() in ["text", "text_input"]:
                                print(f"[FlowPlanner] ⚠️ Skipping upload-like id with text type: {sid} ({stype})", flush=True)
                                continue
                            if sid:
                                seen_ids.add(sid)
                            if max_steps_limit and len(emitted) >= max_steps_limit:
                                reached_cap = True
                                break
                            emitted.append(v)
                            yield {"event": "mini_step", "data": v}
                            if max_steps_limit and len(emitted) >= max_steps_limit:
                                reached_cap = True
                                break
                    if reached_cap:
                        break
                elif isinstance(item, dspy.Prediction):
                    final_pred = item
                elif isinstance(item, StatusMessage):
                    continue
        except (asyncio.CancelledError, GeneratorExit):
            try:
                if stream is not None and hasattr(stream, "aclose"):
                    await stream.aclose()
            except Exception:
                pass
            return
        except BaseExceptionGroup as e:
            stream_error = e
            import traceback
            print(f"[FlowPlanner] ⚠️ Stream error (BaseExceptionGroup): {e}", flush=True)
            print(f"[FlowPlanner] Traceback: {traceback.format_exc()}", flush=True)
        except Exception as e:
            stream_error = e
            import traceback
            print(f"[FlowPlanner] ⚠️ Stream error: {e}", flush=True)
            print(f"[FlowPlanner] Traceback: {traceback.format_exc()}", flush=True)
        finally:
            try:
                if stream is not None and hasattr(stream, "aclose"):
                    await stream.aclose()
            except Exception:
                pass

        if stream_error is None:
            break
        if emitted or attempt >= max_retries:
            break
        attempt += 1
        buffer = ""
        had_chunks = False
        final_pred = None
        reached_cap = False
        lint_violations = []
        if backoff_ms:
            await asyncio.sleep(backoff_ms / 1000)

    if stream_error is not None:
        print(f"[FlowPlanner] ⚠️ Stream completed with error: {stream_error}", flush=True)
        
        # If we already emitted valid steps, don't fail - just log the error and continue
        if emitted:
            print(f"[FlowPlanner] ✅ {len(emitted)} steps already emitted, ignoring stream error", flush=True)
            # Extract sub-exceptions from BaseExceptionGroup for logging
            if isinstance(stream_error, BaseExceptionGroup):
                try:
                    exceptions = stream_error.exceptions
                    for exc in exceptions:
                        print(f"[FlowPlanner] Sub-exception: {exc}", flush=True)
                except Exception:
                    pass
            # Don't yield error if we have valid steps - continue to meta event below
            stream_error = None  # Clear error so we continue to meta event
        else:
            # No steps emitted, check if it's a rate limit error
            error_str = str(stream_error)
            is_rate_limit = "RateLimitError" in error_str or "rate limit" in error_str.lower() or "429" in error_str
            if is_rate_limit:
                import re
                # Try to extract wait time from Groq error message
                wait_match = re.search(r"try again in ([\d.]+[smh])", error_str, re.IGNORECASE)
                wait_time = wait_match.group(1) if wait_match else None
                error_msg = f"API rate limit reached. Please try again in {wait_time or 'a few minutes'}."
                if wait_time:
                    error_msg += f" (Wait time: {wait_time})"
                yield {
                    "event": "error",
                    "data": {
                        "message": error_msg,
                        "type": "RateLimitError",
                        "retryAfter": wait_time,
                        "originalError": error_str[:500]  # Truncate for safety
                    }
                }
                return
            # For non-rate-limit errors with no steps, try fallback
            if not had_chunks and final_pred is None:
                try:
                    fallback_result = await asyncio.to_thread(next_steps_jsonl, payload)
                except Exception as e:
                    yield {"event": "error", "data": {"message": str(e), "type": type(e).__name__}}
                    return
                if not isinstance(fallback_result, dict):
                    yield {"event": "error", "data": {"message": "DSPy fallback failed", "type": "FallbackError"}}
                    return
                if fallback_result.get("error"):
                    yield {
                        "event": "error",
                        "data": {"message": str(fallback_result.get("error")), "type": "FallbackError"},
                    }
                    return
                fallback_steps = fallback_result.get("miniSteps") or []
                if isinstance(fallback_steps, list):
                    for step in fallback_steps:
                        if isinstance(step, dict):
                            yield {"event": "mini_step", "data": step}
                yield {"event": "meta", "data": fallback_result}
                return

    if not reached_cap and buffer.strip():
        obj = _best_effort_parse_json(buffer.strip())
        v = _validate_mini(obj, ui_types)
        if v:
            if sanitize_step:
                v = sanitize_step(v, lint_config)
            if apply_reassurance:
                v = apply_reassurance([v], lint_config)[0]
            if lint_steps:
                try:
                    ok, violations, _bad_ids = lint_steps([v], lint_config)
                    if not ok:
                        lint_violations.extend(violations)
                except Exception:
                    pass
            sid = str(v.get("id") or "")
            stype = str(v.get("type") or "")
            should_emit = True
            if sid:
                if sid in already_asked_keys:
                    print(f"[FlowPlanner] ⚠️ Skipping already asked step: {sid}", flush=True)
                    should_emit = False
                elif sid in seen_ids:
                    should_emit = False
            if should_emit and not _allowed_type_matches(stype, allowed_set):
                print(f"[FlowPlanner] ⚠️ Skipping disallowed step type '{stype}' for {sid or 'unknown'}", flush=True)
                should_emit = False
            if should_emit:
                v = _apply_banned_option_policy(v, service_anchor_terms)
                if not v:
                    print(f"[FlowPlanner] ⚠️ Skipping step with banned filler options: {sid or 'unknown'}", flush=True)
                    should_emit = False
            if should_emit and sid in required_upload_ids and stype.lower() not in ["upload", "file_upload", "file_picker"]:
                print(f"[FlowPlanner] ⚠️ Skipping upload step with non-upload type: {sid} ({stype})", flush=True)
                should_emit = False
            if should_emit and _looks_like_upload_step_id(sid) and stype.lower() in ["text", "text_input"]:
                print(f"[FlowPlanner] ⚠️ Skipping upload-like id with text type: {sid} ({stype})", flush=True)
                should_emit = False
            if should_emit and (not sid or sid not in seen_ids):
                if sid:
                    seen_ids.add(sid)
                if max_steps_limit and len(emitted) >= max_steps_limit:
                    reached_cap = True
                    should_emit = False
            if should_emit:
                emitted.append(v)
                yield {"event": "mini_step", "data": v}

    if not reached_cap and final_pred is not None and (not had_chunks or not emitted):
        raw_mini_steps = getattr(final_pred, "mini_steps_jsonl", None) or ""
        for line in str(raw_mini_steps).splitlines():
            line = line.strip()
            if not line:
                continue
            obj = _best_effort_parse_json(line)
            v = _validate_mini(obj, ui_types)
            if v:
                if sanitize_step:
                    v = sanitize_step(v, lint_config)
                if apply_reassurance:
                    v = apply_reassurance([v], lint_config)[0]
                if lint_steps:
                    try:
                        ok, violations, _bad_ids = lint_steps([v], lint_config)
                        if not ok:
                            lint_violations.extend(violations)
                    except Exception:
                        pass
                sid = str(v.get("id") or "")
                stype = str(v.get("type") or "")
                if sid and sid in already_asked_keys:
                    print(f"[FlowPlanner] ⚠️ Skipping already asked step: {sid}", flush=True)
                    continue
                if not _allowed_type_matches(stype, allowed_set):
                    print(f"[FlowPlanner] ⚠️ Skipping disallowed step type '{stype}' for {sid or 'unknown'}", flush=True)
                    continue
                v = _apply_banned_option_policy(v, service_anchor_terms)
                if not v:
                    print(f"[FlowPlanner] ⚠️ Skipping step with banned filler options: {sid or 'unknown'}", flush=True)
                    continue
                if sid in required_upload_ids and stype.lower() not in ["upload", "file_upload", "file_picker"]:
                    print(f"[FlowPlanner] ⚠️ Skipping upload step with non-upload type: {sid} ({stype})", flush=True)
                    continue
                if sid and sid in seen_ids:
                    continue
                if sid:
                    seen_ids.add(sid)
                if max_steps_limit and len(emitted) >= max_steps_limit:
                    reached_cap = True
                    break
                emitted.append(v)
                yield {"event": "mini_step", "data": v}
                if max_steps_limit and len(emitted) >= max_steps_limit:
                    reached_cap = True
                    break

    # If we had an error but already emitted steps, clear the error so we can send meta
    if stream_error is not None and emitted:
        print(f"[FlowPlanner] ✅ Ignoring stream error since {len(emitted)} steps were already emitted", flush=True)
        stream_error = None

    latency_ms = int((time.time() - start_time) * 1000)
    lint_failed = bool(lint_violations)
    lint_codes = _summarize_violation_codes(lint_violations)
    meta = {
        "requestId": request_id,
        "miniSteps": emitted,
        "copyPackId": prep.get("copy_pack_id"),
        "copyPackVersion": prep.get("copy_pack_version"),
        "lintFailed": lint_failed,
        "lintViolationCodes": lint_codes,
    }
    # Pass through session info from payload if available
    session_info = payload.get("session")
    if isinstance(session_info, dict):
        meta["session"] = session_info
    if lint_steps:
        try:
            ok, violations, _bad_ids = lint_steps(emitted, lint_config)
            if not ok:
                meta["lintFailed"] = True
                meta["lintViolationCodes"] = _summarize_violation_codes(violations)
                print(f"[FlowPlanner] ❌ Copy lint failed ({len(violations)} violations)", flush=True)
        except Exception as e:
            print(f"[FlowPlanner] ⚠️ Copy lint failed to run: {e}", flush=True)
    if copy_needed and hasattr(module, "generate_copy"):
        mini_steps_for_copy = ""
        if final_pred is not None:
            mini_steps_for_copy = str(getattr(final_pred, "mini_steps_jsonl", "") or "")
        if not mini_steps_for_copy and emitted:
            mini_steps_for_copy = "\n".join(
                _compact_json(step) for step in emitted if isinstance(step, dict)
            )
        try:
            copy_pred = module.generate_copy(
                context_json=copy_context_json,
                mini_steps_jsonl=mini_steps_for_copy,
            )
            copy_json = getattr(copy_pred, "must_have_copy_json", None) or ""
            parsed_copy = _parse_must_have_copy(copy_json)
            if parsed_copy:
                meta["mustHaveCopy"] = parsed_copy
            if track_usage:
                copy_usage = _extract_dspy_usage(copy_pred)
                if copy_usage:
                    meta["lmUsageCopy"] = copy_usage
        except Exception:
            pass
    if track_usage and final_pred is not None:
        usage = _extract_dspy_usage(final_pred)
        if usage:
            meta["lmUsage"] = usage
    try:
        from modules.form_psychology.form_plan import finalize_form_plan

        produced_form_plan_json = ""
        if final_pred is not None:
            produced_form_plan_json = getattr(final_pred, "produced_form_plan_json", "") or ""
        final_plan, did_generate = finalize_form_plan(
            payload=payload,
            context=prep.get("context") or {},
            produced_form_plan_json=produced_form_plan_json,
        )
        if did_generate and final_plan is not None:
            meta["formPlan"] = final_plan
            try:
                from modules.form_psychology.ui_plan import build_ui_plan

                ui_plan = build_ui_plan(
                    payload=payload,
                    final_form_plan=final_plan,
                    emitted_mini_steps=emitted,
                    required_uploads=(prep.get("context") or {}).get("required_uploads"),
                )
                if ui_plan is not None:
                    meta["uiPlan"] = ui_plan
            except Exception:
                pass
    except Exception:
        pass
    if created_batch_policy and "batchPolicy" not in meta:
        meta["batchPolicy"] = created_batch_policy
    if created_psychology_plan and "psychologyPlan" not in meta:
        meta["psychologyPlan"] = created_psychology_plan
    print(
        f"[FlowPlanner] Streamed response: requestId={meta['requestId']}, latencyMs={latency_ms}, steps={len(meta['miniSteps'])}, model={lm_cfg.get('modelName') or lm_cfg.get('model')}",
        flush=True,
    )
    yield {"event": "meta", "data": meta}


def _default_next_steps_demo_pack() -> str:
    """
    Prefer shared contract demos when present; otherwise fall back to repo-local examples.
    """
    env_pack = (os.getenv("DSPY_NEXT_STEPS_DEMO_PACK") or "").strip()
    if env_pack:
        return env_pack
    repo_root = Path(__file__).resolve().parent
    shared = repo_root / "shared" / "ai-form-contract" / "demos" / "next_steps_examples.jsonl"
    if shared.exists():
        return str(shared)
    return "next_steps_examples.jsonl"


def _best_effort_contract_schema_version() -> str:
    try:
        repo_root = Path(__file__).resolve().parent
        p = repo_root / "shared" / "ai-form-contract" / "schema" / "schema_version.txt"
        if p.exists():
            v = p.read_text(encoding="utf-8").strip()
            return v or "0"
    except Exception:
        pass
    return "0"


def _make_dspy_lm() -> Optional[Dict[str, str]]:
    """
    Return a LiteLLM model string for DSPy v3 (provider-prefixed), or None if not configured.
    """
    provider = (os.getenv("DSPY_PROVIDER") or "groq").lower()
    # Default to GPT OSS 20B (good balance of quality and cost)
    # Can also use: openai/gpt-oss-120b (larger, more capable)
    locked_model = os.getenv("DSPY_MODEL_LOCK") or "openai/gpt-oss-20b"
    requested_model = os.getenv("DSPY_MODEL") or locked_model
    model = requested_model

    # Block 8B/instant models by default (JSON reliability / rate limit stability)
    # But allow GPT OSS models (they're high quality)
    is_gpt_oss = "gpt-oss" in model.lower()
    if not is_gpt_oss and ("8b" in model.lower() or "8-b" in model.lower() or "instant" in model.lower()):
        if os.getenv("AI_FORM_DEBUG") == "true":
            sys.stderr.write(
                f"[DSPy] 🚫 BLOCKED: Requested DSPY_MODEL='{model}' (8B/instant). Forcing lock='{locked_model}'.\n"
            )
        model = locked_model

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            return None
        # For Groq, GPT OSS models use the format: groq/openai/gpt-oss-20b
        # LiteLLM expects: groq/openai/gpt-oss-20b or groq/openai/gpt-oss-120b
        if model.startswith("openai/"):
            model_str = f"groq/{model}"
        else:
            model_str = f"groq/{model}"
        return {"provider": "groq", "model": model_str, "modelName": model}

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return {"provider": "openai", "model": f"openai/{model}", "modelName": model}

    return None


def _get_int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _dspy_stream_retry_config() -> tuple[int, int]:
    max_retries = max(0, _get_int_env("AI_FORM_DSPY_MAX_RETRIES", 0))
    backoff_ms = max(0, _get_int_env("AI_FORM_DSPY_RETRY_BACKOFF_MS", 250))
    return max_retries, backoff_ms


def _print_lm_history_if_available(lm: Any, n: int = 1) -> None:
    try:
        inspect_fn = getattr(lm, "inspect_history", None)
        if not callable(inspect_fn):
            return
        with contextlib.redirect_stdout(sys.stderr):
            inspect_fn(n=n)
    except Exception:
        return


def _configure_dspy(lm: Any) -> bool:
    try:
        import dspy  # type: ignore
    except Exception:
        return False

    telemetry_on = os.getenv("AI_FORM_TOKEN_TELEMETRY") == "true" or os.getenv("AI_FORM_DEBUG") == "true"
    track_usage = os.getenv("DSPY_TRACK_USAGE") == "true" or telemetry_on

    try:
        settings = getattr(dspy, "settings", None)
        settings_cfg = getattr(settings, "configure", None)
        if callable(settings_cfg):
            settings_cfg(lm=lm, track_usage=track_usage)
            return track_usage
    except Exception:
        return False
    return track_usage


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


def _parse_must_have_copy(text: Any) -> Dict[str, Any]:
    obj = _best_effort_parse_json(str(text or ""))
    if not isinstance(obj, dict):
        return {}
    try:
        from modules.schemas.ui_steps import StepCopy
    except Exception:
        return {}
    out: Dict[str, Any] = {}
    for key, val in obj.items():
        k = str(key or "").strip()
        if not k or not isinstance(val, dict):
            continue
        try:
            out[k] = StepCopy.model_validate(val).model_dump()
        except Exception:
            continue
    return out
