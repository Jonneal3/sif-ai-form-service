"""
Form generation pipeline (DSPy + validation + deterministic placements).

Note: this module lives under `programs.batch_generator` because it is the orchestrator that
produces batch steps; the old AI "form planner" module has been removed.

---

### How to read this file (DSPy beginner notes)

This module is the **bridge between HTTP** (FastAPI) and **DSPy** (prompted programs).

Data flow:
1. `api/routes/form.py` receives a POST body (payload dict)
2. `api/routes/form.py` calls `next_steps_jsonl(payload)`
3. `next_steps_jsonl` creates a DSPy LM + configures DSPy settings
4. `next_steps_jsonl` creates a DSPy Module: `FlowPlannerModule`
5. DSPy sends a request to the provider (Groq/OpenAI) via LiteLLM
6. We parse the model output, validate it with Pydantic, and return a clean JSON structure

Key DSPy concepts used here:
- **Signature** (`NextStepsJSONL`): describes *inputs* and *outputs* in a declarative way.
- **Predict** (`dspy.Predict(Signature)`): turns that signature into a callable LLM-backed function.
- **Demos**: examples attached to the module's predictor that guide the model (few-shot).

DSPy map for this repo:
    - Signature: `src/programs/batch_generator/signatures/json_signatures.py` → `NextStepsJSONL`
- Predictor: created inside `src/app/dspy/batch_generator_module.py` via `dspy.Predict(NextStepsJSONL)`
- Module: `src/app/dspy/flow_planner_module.py` → `FlowPlannerModule`
- Pipeline (future): would be multiple Modules chained in `src/app/pipeline/form_pipeline.py`

Why some fields are strings:
- LLM outputs are text. For reliability we ask DSPy to output JSON/JSONL **as strings**,
  then we parse + validate with Pydantic. This makes failures detectable and recoverable.

Token/step constraints:
- Hard cap token length via `dspy.LM(max_tokens=...)` (provider-enforced).
- Keep `max_steps` in the signature as a soft/content constraint, and enforce step limits in runtime parsing.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

# Suppress Pydantic serialization warnings from LiteLLM
# These warnings occur when LiteLLM serializes LLM response objects (Message, StreamingChoices)
# and are harmless - they don't affect functionality
warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
    module="pydantic",
)

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
    # Heuristic: find first array/object block
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
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    for key in (
        "grounding_summary",
        "groundingSummary",
        "grounding_preview",
        "groundingPreview",
        "grounding",
    ):
        raw = payload.get(key)
        if raw is None and state:
            raw = state.get(key)
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
        from grounding.keywords import extract_service_anchor_terms

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


def _slug_option_value(label: str) -> str:
    base = _normalize_option_label(label).replace(" ", "_").strip("_")
    return base or "option"


def _coerce_options(options: Any) -> list[dict]:
    """
    Normalize option arrays into the canonical object form:
      [{ "label": str, "value": str }, ...]
    """
    if not isinstance(options, list):
        return []

    out: list[dict] = []
    seen: dict[str, int] = {}
    for opt in options:
        label: str
        value: str

        if isinstance(opt, str):
            label = opt.strip()
            value = _slug_option_value(label)
        elif isinstance(opt, dict):
            raw_label = opt.get("label")
            raw_value = opt.get("value")
            label = str(raw_label if raw_label is not None else (raw_value if raw_value is not None else "")).strip()
            value = str(raw_value if raw_value is not None else _slug_option_value(label)).strip()
        else:
            continue

        if not label:
            continue
        if not value:
            value = _slug_option_value(label)

        if value in seen:
            seen[value] += 1
            value = f"{value}_{seen[value]}"
        else:
            seen[value] = 1

        out.append({"label": label, "value": value})

    return out


def _canonicalize_step_output(step: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure step objects returned over the wire match the shared UI contract as closely as possible.
    """
    if not isinstance(step, dict):
        return step
    out = dict(step)

    def _default_metric_gain_for_step(s: Dict[str, Any]) -> float:
        step_type = str(s.get("type") or "").strip().lower()
        base = 0.1
        if step_type in {
            "choice",
            "multiple_choice",
            "segmented_choice",
            "chips_multi",
            "yes_no",
            "image_choice_grid",
            "searchable_select",
        }:
            base = 0.12
        elif step_type in {"slider", "rating", "range_slider", "budget_cards"}:
            base = 0.1
        elif step_type in {"text", "text_input"}:
            base = 0.08
        elif step_type in {"upload", "file_upload", "file_picker"}:
            base = 0.15
        elif step_type in {"intro", "confirmation", "pricing", "designer", "composite"}:
            base = 0.05

        required = s.get("required")
        if required is True:
            base = min(0.25, base + 0.03)
        if required is False:
            base = max(0.03, base - 0.02)
        return float(base)

    # Strip legacy keys that can confuse the frontend.
    for k in (
        "stepId",
        "step_id",
        "stepID",
        "component_hint",
        "componentHint",
        "componentType",
        "component_type",
        "allowMultiple",
        # We no longer use/emit backend "phase policy" blobs; keep responses lean.
        "batch_phase_policy",
        "batchPhasePolicy",
    ):
        out.pop(k, None)

    # Canonicalize allow_multiple.
    if "allow_multiple" not in out:
        raw = step.get("allow_multiple")
        if raw is None:
            raw = step.get("allowMultiple")
        if raw is None:
            raw = step.get("multi_select")
        if raw is None:
            raw = step.get("multiSelect")
        if raw is not None:
            out["allow_multiple"] = bool(raw)

    # Canonicalize options.
    if isinstance(step.get("options"), list):
        out["options"] = _coerce_options(step.get("options"))

    # Ensure `metricGain` is always present and numeric for downstream scoring.
    mg = out.get("metricGain")
    if mg is None:
        mg = out.get("metric_gain")
    try:
        mg_val = float(mg) if mg is not None else None
    except Exception:
        mg_val = None
    if mg_val is None:
        out["metricGain"] = _default_metric_gain_for_step(out)
        out.pop("metric_gain", None)
    else:
        out["metricGain"] = float(mg_val)
        out.pop("metric_gain", None)

    return out


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

def _normalize_allowed_component_types(raw: Any) -> list[str]:
    """
    Frontend/back-compat adapter.

    Some clients send `allowedComponentTypes` with values like:
      ["choice", "slider", "text"]
    while the DSPy signature expects `allowed_mini_types` with values like:
      ["choice", "slider", "text_input"]
    """
    values = _normalize_allowed_mini_types(raw)
    mapped: list[str] = []
    for v in values:
        t = str(v or "").strip().lower()
        if not t:
            continue
        if t == "text":
            t = "text_input"
        mapped.append(t)
    return mapped


def _prefer_structured_allowed_mini_types(raw: Any) -> list[str]:
    types = [t.strip().lower() for t in _normalize_allowed_mini_types(raw) if str(t or "").strip()]
    if not types:
        return types
    structured = {"choice", "multiple_choice", "segmented_choice", "chips_multi", "yes_no", "slider", "rating", "range_slider"}
    has_structured = any(t in structured for t in types)
    if not has_structured:
        return types
    return [t for t in types if t not in {"text", "text_input"}]

def _extract_allowed_mini_types_from_payload(payload: Dict[str, Any]) -> list[str]:
    raw = payload.get("allowedMiniTypes") or payload.get("allowed_mini_types")
    types = _normalize_allowed_mini_types(raw)
    if types:
        return types
    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}
    raw_component_types = None
    if isinstance(current_batch, dict):
        raw_component_types = current_batch.get("allowedComponentTypes") or current_batch.get("allowed_component_types")
    if raw_component_types:
        return _normalize_allowed_component_types(raw_component_types)
    return []


_DEFAULT_ALLOWED_MINI_TYPES: list[str] = [
    # Structured first: tends to produce more reliable downstream UX.
    "multiple_choice",
    "yes_no",
    "slider",
    "rating",
    "file_upload",
    # Helpful structured variants some clients use.
    "segmented_choice",
    "chips_multi",
    "searchable_select",
]


def _ensure_allowed_mini_types(allowed: list[str]) -> list[str]:
    # If caller didn't provide constraints, give DSPy a sane default rather than an empty list.
    values = [str(x).strip().lower() for x in (allowed or []) if str(x).strip()]
    return values or list(_DEFAULT_ALLOWED_MINI_TYPES)


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


def _resolve_backend_max_calls(*, use_case: str, goal_intent: str) -> int:
    default_max_calls = 2
    try:
        from programs.batch_generator.form_planning.static_constraints import DEFAULT_MAX_BATCH_CALLS

        default_max_calls = int(DEFAULT_MAX_BATCH_CALLS)
    except Exception:
        default_max_calls = 2

    env_default = max(1, min(10, _get_int_env("AI_FORM_MAX_BATCH_CALLS", default_max_calls)))
    try:
        from programs.batch_generator.form_planning.static_constraints import resolve_max_calls

        return resolve_max_calls(use_case=use_case, goal_intent=goal_intent, default_max_calls=env_default)
    except Exception:
        return env_default


def _as_int(value: Any) -> Optional[int]:
    try:
        n = int(value)
    except Exception:
        return None
    return n if n > 0 else None


def _as_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except Exception:
        return None
    return f if f >= 0 else None


def _build_batch_constraints(*, payload: Dict[str, Any], batch_state: Dict[str, Any], max_batches: int) -> Dict[str, Any]:
    """
    Build the backend constraints we share with the frontend (max calls, step limits, token budget).
    """
    default_max_steps_per_batch = 5
    default_token_budget_total = 3000
    try:
        from programs.batch_generator.form_planning.static_constraints import (
            DEFAULT_MAX_STEPS_PER_BATCH,
            DEFAULT_TOKEN_BUDGET_TOTAL,
        )

        default_max_steps_per_batch = int(DEFAULT_MAX_STEPS_PER_BATCH)
        default_token_budget_total = int(DEFAULT_TOKEN_BUDGET_TOTAL)
    except Exception:
        default_max_steps_per_batch = 5
        default_token_budget_total = 3000

    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}
    max_steps_per_batch = (
        _as_int(payload.get("maxSteps"))
        or _as_int(payload.get("max_steps"))
        or _as_int(current_batch.get("maxSteps"))
        or _as_int(current_batch.get("max_steps"))
        or _as_int(os.getenv("AI_FORM_MAX_STEPS_PER_BATCH"))
        or default_max_steps_per_batch
    )
    max_steps_total = (
        _as_int(batch_state.get("max_steps_total"))
        or _as_int(batch_state.get("maxStepsTotal"))
        or max_steps_per_batch * max_batches
    )
    token_budget_total = (
        _as_int(batch_state.get("tokensTotalBudget"))
        or _as_int(batch_state.get("token_budget_total"))
        or _as_int(os.getenv("AI_FORM_TOKEN_BUDGET_TOTAL"))
        or default_token_budget_total
    )
    return {
        "maxBatches": max_batches,
        "maxStepsTotal": max_steps_total,
        "maxStepsPerBatch": max_steps_per_batch,
        "tokenBudgetTotal": token_budget_total,
    }


def _extract_max_batches_from_context(context: Dict[str, Any]) -> Optional[int]:
    info = context.get("batch_info") if isinstance(context, dict) else None
    if not isinstance(info, dict):
        return None
    raw = info.get("max_batches") or info.get("maxCalls") or info.get("max_calls")
    try:
        n = int(raw) if raw is not None else None
    except Exception:
        return None
    return n if isinstance(n, int) and n > 0 else None



def _build_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    state_answers = state.get("answers") if isinstance(state.get("answers"), dict) else {}

    required_uploads_raw = payload.get("requiredUploads") or payload.get("required_uploads") or []
    required_uploads = required_uploads_raw if isinstance(required_uploads_raw, list) else []

    # Memory: treat accumulated answers as the source of truth.
    # - Widget shape: `state.answers`
    # - Legacy/service shape: `stepDataSoFar` / `knownAnswers`
    known_answers_raw = payload.get("stepDataSoFar") or payload.get("knownAnswers") or state_answers or {}
    known_answers = known_answers_raw if isinstance(known_answers_raw, dict) else {}

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
    normalized_already: list[str] = []
    if isinstance(already_asked, list):
        for x in already_asked:
            t = str(x or "").strip()
            if not t:
                continue
            sid = _normalize_step_id(t)
            # `askedStepIds` should track question step ids that were shown (answered or not).
            # Avoid mixing in non-step identifiers or plan keys.
            if not sid.startswith("step-"):
                continue
            normalized_already.append(sid)

    # Optional: richer memory to help the model avoid re-asking semantically similar questions.
    # Expected shape: [{ stepId, question, answer }] (strings).
    answered_qa_raw = payload.get("answeredQA") or payload.get("answered_qa")
    if answered_qa_raw is None and state:
        answered_qa_raw = state.get("answeredQA") or state.get("answered_qa")
    answered_qa: list[dict] = []
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

    # Deprecated: do not rely on client-provided or server-generated form-level plan items.
    # The batch policy + current context should be sufficient to generate steps.
    form_plan: list[dict] = []
    psychology_plan = payload.get("psychologyPlan") or payload.get("psychology_plan") or {}

    batch_state_raw = payload.get("batchState") or payload.get("batch_state") or {}
    batch_state = batch_state_raw if isinstance(batch_state_raw, dict) else {}

    items_raw = payload.get("items") or []
    items = items_raw if isinstance(items_raw, list) else []

    instance_subcategories_raw = payload.get("instanceSubcategories") or payload.get("instance_subcategories") or []
    instance_subcategories = instance_subcategories_raw if isinstance(instance_subcategories_raw, list) else []

    # Plain-English context anchors (critical when answers contain UUIDs).
    state_context = state.get("context") if isinstance(state.get("context"), dict) else {}

    industry = str(payload.get("industry") or payload.get("vertical") or state_context.get("industry") or state_context.get("categoryName") or "General")[:80]
    service = str(payload.get("service") or payload.get("subcategoryName") or state_context.get("subcategoryName") or "")[:80]
    use_case = _extract_use_case(payload)
    platform_goal = str(payload.get("platformGoal") or payload.get("platform_goal") or "")[:600]
    business_context = str(payload.get("businessContext") or payload.get("business_context") or state_context.get("businessContext") or "")[:200]
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

    # Backend-owned call cap. We intentionally do NOT trust `formState.maxBatches` as authoritative.
    backend_max_calls = _resolve_backend_max_calls(use_case=use_case, goal_intent=goal_intent)
    model_batch = dict(model_batch)
    model_batch["max_batches"] = backend_max_calls
    batch_constraints = _build_batch_constraints(payload=payload, batch_state=batch_state, max_batches=backend_max_calls)

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
        "asked_step_ids": normalized_already,
        # Deprecated: legacy name kept for compatibility.
        "already_asked_keys": normalized_already,
        "batch_info": model_batch,
        "form_plan": form_plan,
        "batch_constraints": batch_constraints,
        "psychology_plan": psychology_plan if isinstance(psychology_plan, dict) else {},
        "batch_state": batch_state,
        "items": items,
        "instance_subcategories": instance_subcategories,
        "attribute_families": attribute_families,
        "service_anchor_terms": service_anchor_terms,
        "answered_qa": answered_qa,
        "choice_option_min": choice_option_min,
        "choice_option_max": choice_option_max,
        "choice_option_target": choice_option_target,
        "prefer_structured_inputs": True,
    }
    if combined_grounding:
        # "Grounding" = vertical-specific facts/anchors to prevent invention.
        # Use clearer key names while preserving backward compatibility.
        context["vertical_context"] = combined_grounding
        context["grounding_summary"] = combined_grounding
    # Planner/batch metadata for downstream control
    if payload.get("batchNumber") is not None:
        context["batch_number"] = payload.get("batchNumber")
    return context


def _extract_rigidity(payload: Dict[str, Any], context: Dict[str, Any]) -> float:
    """Rigidity is fixed at 0.0 now that AI plans drive the flow."""
    return 0.0




def _synthesize_form_plan_items_for_batch(*, context: Dict[str, Any], batch_number: int, max_items: int) -> list[dict]:
    """
    Provide a lightweight `form_plan` list for the LLM prompt when none is present.

    This is an internal prompt aid only (not emitted to clients). It uses the already-selected
    attribute families and splits them into batches in a predictable way.
    """
    if not isinstance(context, dict):
        return []
    families = context.get("attribute_families") if isinstance(context.get("attribute_families"), list) else []
    if not families:
        return []

    normalized_families: list[dict] = []
    for f in families:
        if not isinstance(f, dict):
            continue
        fam = str(f.get("family") or "").strip()
        goal = str(f.get("goal") or "").strip()
        if not fam:
            continue
        normalized_families.append({"family": fam, "goal": goal})
    if not normalized_families:
        return []

    # Simple split: batch 1 gets the first half, later batches get the remainder.
    split_idx = max(1, int(round(len(normalized_families) * 0.5)))
    if int(batch_number) <= 1:
        selected = normalized_families[:split_idx]
    elif int(batch_number) >= 2:
        selected = normalized_families[split_idx:]
    else:
        selected = normalized_families

    # Cap to the per-call max to keep prompts small.
    selected = selected[: max(1, int(max_items or 1))]
    out: list[dict] = []
    for idx, f in enumerate(selected):
        fam = f["family"]
        out.append(
            {
                "key": fam,
                "goal": f.get("goal") or fam.replace("_", " ").strip().title(),
                "why": "",
                "priority": "critical" if idx == 0 else "optional",
                "component_hint": "choice",
                "importance_weight": 0.2 if idx == 0 else 0.1,
                "expected_metric_gain": 0.15 if idx == 0 else 0.1,
            }
        )
    return out


def _allowed_item_ids_from_context(context: Dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    items = context.get("items")
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            # Item ids can arrive in mixed formats (underscores vs hyphens).
            # Normalize to our canonical `step-...` id style so runtime filtering
            # doesn't accidentally discard valid model output.
            raw_id = it.get("id") or it.get("stepId") or it.get("step_id") or ""
            sid = _normalize_step_id(str(raw_id or "")).strip()
            if sid:
                ids.add(sid)
    return ids


def _ensure_items_from_form_plan(context: Dict[str, Any]) -> None:
    """
    If we have a `form_plan` but no `items`, synthesize `items` so DSPy has concrete ids/keys.

    This also enables the runtime's allowed-item filtering when rigidity > 0.
    """
    if not isinstance(context, dict):
        return
    items = context.get("items")
    if isinstance(items, list) and items:
        return
    form_plan = context.get("form_plan")
    if not isinstance(form_plan, list) or not form_plan:
        return
    synthesized: list[dict] = []
    for it in form_plan:
        if not isinstance(it, dict):
            continue
        key = str(it.get("key") or "").strip()
        if not key:
            continue
        synthesized.append(
            {
                "id": f"step-{key.replace('_', '-')}",
                "key": key,
                "goal": it.get("goal") or "",
                "why": it.get("why") or "",
                "priority": it.get("priority") or "medium",
                "component_hint": it.get("component_hint") or "choice",
                "importance_weight": it.get("importance_weight") or 0.1,
                "expected_metric_gain": it.get("expected_metric_gain") or 0.1,
            }
        )
    if synthesized:
        context["items"] = synthesized


def _exploration_budget(max_steps_limit: Optional[int], rigidity: float) -> int:
    if not isinstance(max_steps_limit, int) or max_steps_limit <= 0:
        return 0
    return max(0, min(max_steps_limit, int(round((1.0 - max(0.0, min(1.0, rigidity))) * max_steps_limit))))


def _load_signature_types() -> tuple[Any, Dict[str, Any]]:
    from schemas.ui_steps import (
        BudgetCardsUI,
        ColorPickerUI,
        CompositeUI,
        ConfirmationUI,
        DatePickerUI,
        DesignerUI,
        FileUploadUI,
        IntroUI,
        LeadCaptureUI,
        MultipleChoiceUI,
        PricingUI,
        RatingUI,
        SearchableSelectUI,
        TextInputUI,
    )
    from programs.batch_generator.signatures.next_steps_jsonl import BatchNextStepsJSONL

    ui_types = {
        "BudgetCardsUI": BudgetCardsUI,
        "ColorPickerUI": ColorPickerUI,
        "CompositeUI": CompositeUI,
        "ConfirmationUI": ConfirmationUI,
        "DatePickerUI": DatePickerUI,
        "DesignerUI": DesignerUI,
        "FileUploadUI": FileUploadUI,
        "IntroUI": IntroUI,
        "LeadCaptureUI": LeadCaptureUI,
        "MultipleChoiceUI": MultipleChoiceUI,
        "PricingUI": PricingUI,
        "RatingUI": RatingUI,
        "SearchableSelectUI": SearchableSelectUI,
        "TextInputUI": TextInputUI,
    }
    return BatchNextStepsJSONL, ui_types


def _batch_context_summary(context: Dict[str, Any]) -> str:
    """
    Short, non-PII summary that can be embedded into each batch plan.
    """
    if not isinstance(context, dict):
        return ""
    service = str(context.get("service") or "").strip()
    industry = str(context.get("industry") or "").strip()
    goal_intent = str(context.get("goal_intent") or "").strip()
    business_context = str(context.get("business_context") or "").strip()
    asked = context.get("asked_step_ids") if isinstance(context.get("asked_step_ids"), list) else []
    asked = [str(x).strip() for x in asked if str(x).strip()]
    asked_preview = ", ".join(asked[:6])

    parts: list[str] = []
    if industry and service:
        parts.append(f"{industry} / {service}")
    elif service:
        parts.append(service)
    elif industry:
        parts.append(industry)
    if goal_intent:
        parts.append(f"goal={goal_intent}")
    if business_context:
        parts.append(business_context[:80])
    if asked_preview:
        parts.append(f"already asked: {asked_preview}")
    return " | ".join(parts)[:240]


def _fill_missing_batches(*, batches: list[dict], max_batches: int, default_max_steps: int) -> list[dict]:
    """
    Ensure we always return `max_batches` batch objects even if the planner returns fewer phases.
    """
    if max_batches <= 0:
        return batches
    out = [b for b in (batches or []) if isinstance(b, dict)]
    seen = {str(b.get("batchId") or "") for b in out}
    for idx in range(max_batches):
        batch_id = f"batch-{idx + 1}"
        if batch_id in seen:
            continue
        out.append({"batchId": batch_id, "maxSteps": int(default_max_steps or 5)})
        seen.add(batch_id)
    # Keep stable ordering batch-1..batch-n
    def _sort_key(b: dict) -> int:
        raw = str(b.get("batchId") or "")
        if raw.startswith("batch-"):
            try:
                return int(raw.split("-", 1)[1])
            except Exception:
                return 10_000
        return 10_000

    out.sort(key=_sort_key)
    return out[:max_batches]


def _clean_options(options: Any) -> list:
    """
    Clean up placeholder values in options.
    Detects and removes options with placeholder values like '<<max_depth>>'.
    """
    if not isinstance(options, list):
        return []

    cleaned: list[Any] = []
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
            else:
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

    return _coerce_options(cleaned)


def _validate_mini(obj: Any, ui_types: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    # The model may emit legacy shapes like:
    #   { "stepId": "...", "component_hint": "segmented_choice", ... }
    # Normalize into the shared UI contract fields (`id`, `type`) before validation.
    if "id" not in obj:
        step_id = obj.get("stepId") or obj.get("step_id") or obj.get("stepID")
        if step_id:
            obj = dict(obj)
            obj["id"] = step_id
    if "type" not in obj:
        component_hint = obj.get("component_hint") or obj.get("componentHint") or obj.get("componentType") or obj.get("component_type")
        if component_hint:
            obj = dict(obj)
            obj["type"] = component_hint

    t = str(obj.get("type") or obj.get("componentType") or obj.get("component_hint") or "").lower()
    try:
        if t in ["text", "text_input"]:
            out = ui_types["TextInputUI"].model_validate(obj).model_dump(by_alias=True)
            step_id = _normalize_step_id(str(out.get("id") or "").strip())
            if not step_id:
                step_id = _fallback_step_id(step_type=t, question=str(out.get("question") or ""))
            out["id"] = step_id
            return _canonicalize_step_output(out)
        if t in ["choice", "multiple_choice", "segmented_choice", "chips_multi", "yes_no", "image_choice_grid"]:
            obj = dict(obj)
            step_id = str(obj.get("id") or obj.get("stepId") or obj.get("step_id") or "").strip()
            if "options" not in obj or not obj.get("options"):
                return None
            original_count = len(obj.get("options", []))
            cleaned_options = _clean_options(obj.get("options"))
            if not cleaned_options:
                return None
            if len(cleaned_options) < original_count:
                print(
                    f"[FlowPlanner] ✅ Step '{step_id}': Cleaned options ({original_count} -> {len(cleaned_options)})",
                    flush=True,
            )
            obj["options"] = cleaned_options
            out = ui_types["MultipleChoiceUI"].model_validate(obj).model_dump(by_alias=True)
            out_id = _normalize_step_id(step_id)
            if not out_id:
                out_id = _fallback_step_id(step_type=t, question=str(out.get("question") or ""), options=cleaned_options)
            out["id"] = out_id
            return _canonicalize_step_output(out)
        if t in ["slider", "rating", "range_slider"]:
            out = ui_types["RatingUI"].model_validate(obj).model_dump(by_alias=True)
            step_id = _normalize_step_id(str(out.get("id") or "").strip())
            if not step_id:
                step_id = _fallback_step_id(step_type=t, question=str(out.get("question") or ""))
            out["id"] = step_id
            return _canonicalize_step_output(out)
        if t in ["budget_cards"]:
            out = ui_types["BudgetCardsUI"].model_validate(obj).model_dump(by_alias=True)
            step_id = _normalize_step_id(str(out.get("id") or "").strip())
            if not step_id:
                step_id = _fallback_step_id(step_type=t, question=str(out.get("question") or ""))
            out["id"] = step_id
            return _canonicalize_step_output(out)
        if t in ["upload", "file_upload", "file_picker"]:
            out = ui_types["FileUploadUI"].model_validate(obj).model_dump(by_alias=True)
            step_id = _normalize_step_id(str(out.get("id") or "").strip())
            if not step_id:
                step_id = _fallback_step_id(step_type=t, question=str(out.get("question") or ""))
            out["id"] = step_id
            return _canonicalize_step_output(out)
        if t in ["intro"]:
            out = ui_types["IntroUI"].model_validate(obj).model_dump(by_alias=True)
            step_id = _normalize_step_id(str(out.get("id") or "").strip())
            if not step_id:
                step_id = _fallback_step_id(step_type=t, question=str(out.get("title") or out.get("question") or ""))
            out["id"] = step_id
            return _canonicalize_step_output(out)
        if t in ["date_picker"]:
            out = ui_types["DatePickerUI"].model_validate(obj).model_dump(by_alias=True)
            step_id = _normalize_step_id(str(out.get("id") or "").strip())
            if not step_id:
                step_id = _fallback_step_id(step_type=t, question=str(out.get("question") or ""))
            out["id"] = step_id
            return _canonicalize_step_output(out)
        if t in ["color_picker"]:
            out = ui_types["ColorPickerUI"].model_validate(obj).model_dump(by_alias=True)
            step_id = _normalize_step_id(str(out.get("id") or "").strip())
            if not step_id:
                step_id = _fallback_step_id(step_type=t, question=str(out.get("question") or ""))
            out["id"] = step_id
            return _canonicalize_step_output(out)
        if t in ["searchable_select"]:
            obj = dict(obj)
            step_id = str(obj.get("id") or obj.get("stepId") or obj.get("step_id") or "").strip()
            if "options" not in obj or not obj.get("options"):
                return None
            original_count = len(obj.get("options", []))
            cleaned_options = _clean_options(obj.get("options"))
            if not cleaned_options:
                return None
            if len(cleaned_options) < original_count:
                print(
                    f"[FlowPlanner] ✅ Step '{step_id}': Cleaned options ({original_count} -> {len(cleaned_options)})",
                    flush=True,
            )
            obj["options"] = cleaned_options
            out = ui_types["SearchableSelectUI"].model_validate(obj).model_dump(by_alias=True)
            out_id = _normalize_step_id(step_id)
            if not out_id:
                out_id = _fallback_step_id(step_type=t, question=str(out.get("question") or ""), options=cleaned_options)
            out["id"] = out_id
            return _canonicalize_step_output(out)
        if t in ["lead_capture"]:
            out = ui_types["LeadCaptureUI"].model_validate(obj).model_dump(by_alias=True)
            step_id = _normalize_step_id(str(out.get("id") or "").strip())
            if not step_id:
                step_id = _fallback_step_id(step_type=t, question=str(out.get("question") or ""))
            out["id"] = step_id
            return _canonicalize_step_output(out)
        if t in ["pricing"]:
            out = ui_types["PricingUI"].model_validate(obj).model_dump(by_alias=True)
            step_id = _normalize_step_id(str(out.get("id") or "").strip())
            if not step_id:
                step_id = _fallback_step_id(step_type=t, question=str(out.get("question") or ""))
            out["id"] = step_id
            return _canonicalize_step_output(out)
        if t in ["confirmation"]:
            out = ui_types["ConfirmationUI"].model_validate(obj).model_dump(by_alias=True)
            step_id = _normalize_step_id(str(out.get("id") or "").strip())
            if not step_id:
                step_id = _fallback_step_id(step_type=t, question=str(out.get("question") or ""))
            out["id"] = step_id
            return _canonicalize_step_output(out)
        if t in ["designer"]:
            out = ui_types["DesignerUI"].model_validate(obj).model_dump(by_alias=True)
            step_id = _normalize_step_id(str(out.get("id") or "").strip())
            if not step_id:
                step_id = _fallback_step_id(step_type=t, question=str(out.get("question") or ""))
            out["id"] = step_id
            return _canonicalize_step_output(out)
        if t in ["composite"]:
            if "blocks" not in obj or not obj.get("blocks"):
                return None
            out = ui_types["CompositeUI"].model_validate(obj).model_dump(by_alias=True)
            step_id = _normalize_step_id(str(out.get("id") or "").strip())
            if not step_id:
                step_id = _fallback_step_id(step_type=t, question=str(out.get("question") or ""))
            out["id"] = step_id
            return _canonicalize_step_output(out)

        return None
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
        import dspy
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

    _, ui_types = _load_signature_types()
    from programs.batch_generator.batch_steps_module import BatchStepsModule

    module = BatchStepsModule()

    try:
        from core.demos import as_dspy_examples, load_jsonl_records

        demo_pack = _default_next_steps_demo_pack()
        demos = as_dspy_examples(
            load_jsonl_records(demo_pack),
            input_keys=[
                "context_json",
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
    context = _build_context(payload)
    batch_id = str(batch_id_raw)[:40]
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

    copy_pack_id = _resolve_copy_pack_id(payload)
    style_snippet_json = ""
    lint_config: Dict[str, Any] = {}
    try:
        from core.copywriting.compiler import compile_pack, load_pack

        pack = load_pack(copy_pack_id)
        style_snippet_json, lint_config = compile_pack(pack)
    except Exception as e:
        print(f"[FlowPlanner] ⚠️ Copy pack load failed: {e}", flush=True)

    if style_snippet_json:
        context["copy_style"] = style_snippet_json
        context["copy_context"] = style_snippet_json
    context["batch_id_raw"] = str(batch_id_raw)[:80]
    context["batch_phase_id"] = batch_id
    # Give the batch generator an explicit definition of what this phase means (purpose, focus keys, limits).
    # If the client didn't provide a prompt-level `form_plan`, synthesize one so the model
    # has stable target ids/keys for this batch (helps prevent empty/invalid outputs).
    if isinstance(context, dict) and not context.get("form_plan"):
        try:
            max_items = int((payload.get("maxSteps") or (payload.get("currentBatch") or {}).get("maxSteps") or 4))
            synthesized = []
            try:
                from programs.batch_generator.form_planning.plan import (
                    build_deterministic_form_plan_items_for_batch,
                )

                synthesized = build_deterministic_form_plan_items_for_batch(
                    payload=payload,
                    context=context,
                    batch_number=batch_number,
                    max_items=max_items,
                )
            except Exception:
                synthesized = []
            if not synthesized:
                synthesized = _synthesize_form_plan_items_for_batch(
                    context=context,
                    batch_number=batch_number,
                    max_items=max_items,
                )
            if synthesized:
                context["form_plan"] = synthesized
        except Exception:
            pass

    _ensure_items_from_form_plan(context)
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
        or ((payload.get("currentBatch") or {}).get("maxSteps") if isinstance(payload.get("currentBatch"), dict) else None)
        or "4"
    )
    try:
        max_steps = int(str(max_steps_raw))
        if max_steps < 1:
            max_steps = 4
    except Exception:
        max_steps = 4
    allowed_mini_types = _extract_allowed_mini_types_from_payload(payload)
    allowed_mini_types = _ensure_allowed_mini_types(allowed_mini_types)
    if context.get("prefer_structured_inputs"):
        allowed_mini_types = _prefer_structured_allowed_mini_types(allowed_mini_types)
    already_asked_keys = set(context.get("already_asked_keys") or [])
    required_upload_ids = _extract_required_upload_ids(context.get("required_uploads"))

    inputs = {
        "context_json": context_json,
        "max_steps": max_steps,
        "allowed_mini_types": allowed_mini_types,
    }

    batch_constraints_for_session = context.get("batch_constraints") or {}
    return {
        "request_id": request_id,
        "start_time": start_time,
        "schema_version": str(schema_version) if schema_version else "0",
        "module": module,
        "inputs": inputs,
        "context": context,
        "batch_constraints": batch_constraints_for_session if isinstance(batch_constraints_for_session, dict) else {},
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
    apply_reassurance = None
    lint_steps = None
    sanitize_steps = None
    if lint_config:
        try:
            from core.copywriting.linter import (
                apply_reassurance as _apply_reassurance,
                lint_steps as _lint_steps,
                sanitize_steps as _sanitize_steps,
            )

            apply_reassurance = _apply_reassurance
            lint_steps = _lint_steps
            sanitize_steps = _sanitize_steps
        except Exception:
            pass

    # Generate miniSteps for the current batch.

    pred = module(**inputs)

    # Log the raw DSPy response for debugging
    print(f"[FlowPlanner] Raw DSPy response fields: {list(pred.__dict__.keys()) if hasattr(pred, '__dict__') else 'N/A'}", flush=True)
    raw_mini_steps = getattr(pred, "mini_steps_jsonl", None) or ""
    print(f"[FlowPlanner] Raw mini_steps_jsonl (first 500 chars): {str(raw_mini_steps)[:500]}", flush=True)
    emitted: list[dict] = []
    seen_ids: set[str] = set()
    rigidity = _extract_rigidity(payload, prep.get("context") or {})
    allowed_item_ids = _allowed_item_ids_from_context(prep.get("context") or {})
    exploration_left = _exploration_budget(max_steps_limit, rigidity)
    # 6) DSPy output is an object that exposes fields defined in the Signature.
    #    Here we read the *string* `mini_steps_jsonl` and parse line-by-line.
    raw_lines = raw_mini_steps

    def _iter_candidates(parsed: Any) -> list[Any]:
        if isinstance(parsed, list):
            return list(parsed)
        if isinstance(parsed, dict):
            for k in ("miniSteps", "mini_steps", "steps", "items"):
                v = parsed.get(k)
                if isinstance(v, list):
                    return list(v)
            return [parsed]
        return []

    def _maybe_accept(candidate: Any) -> None:
        nonlocal exploration_left
        if max_steps_limit and len(emitted) >= max_steps_limit:
            return
        v = _validate_mini(candidate, ui_types)
        if not v:
            return
        # Keep batch ids numeric/stable (no "ContextCore"/"Details" naming conventions).
        raw_batch_id = payload.get("batchId") or payload.get("batch_id")
        if raw_batch_id:
            v = dict(v)
            v["batch_phase_id"] = str(raw_batch_id)
        sid = str(v.get("id") or "")
        stype = str(v.get("type") or "")
        if sid:
            if sid in already_asked_keys:
                print(f"[FlowPlanner] ⚠️ Skipping already asked step: {sid}", flush=True)
                return
            if sid in seen_ids:
                return
            if allowed_item_ids and sid not in allowed_item_ids:
                if exploration_left <= 0:
                    return
                exploration_left -= 1
            if not _allowed_type_matches(stype, allowed_set):
                print(f"[FlowPlanner] ⚠️ Skipping disallowed step type '{stype}' for {sid or 'unknown'}", flush=True)
                return
            v = _apply_banned_option_policy(v, service_anchor_terms)
            if not v:
                print(f"[FlowPlanner] ⚠️ Skipping step with banned filler options: {sid or 'unknown'}", flush=True)
                return
        if sid in required_upload_ids and stype.lower() not in ["upload", "file_upload", "file_picker"]:
            print(f"[FlowPlanner] ⚠️ Skipping upload step with non-upload type: {sid} ({stype})", flush=True)
            return
        if _looks_like_upload_step_id(sid) and stype.lower() in ["text", "text_input"]:
            print(f"[FlowPlanner] ⚠️ Skipping upload-like id with text type: {sid} ({stype})", flush=True)
            return
        if sid:
            seen_ids.add(sid)
        emitted.append(v)

    if raw_lines:
        # Preferred path: strict JSONL (one JSON object per line).
        for line in str(raw_lines).splitlines():
            line = line.strip()
            if not line:
                continue
            if max_steps_limit and len(emitted) >= max_steps_limit:
                break
            parsed = _best_effort_parse_json(line)
            for candidate in _iter_candidates(parsed):
                _maybe_accept(candidate)
                if max_steps_limit and len(emitted) >= max_steps_limit:
                    break

        # Fallback path: some models return a single JSON array/object (possibly pretty-printed).
        if not emitted:
            parsed_all = _best_effort_parse_json(str(raw_lines))
            for candidate in _iter_candidates(parsed_all):
                _maybe_accept(candidate)
                if max_steps_limit and len(emitted) >= max_steps_limit:
                    break

    # Log the validated/inflated output
    print(f"[FlowPlanner] Validated steps count: {len(emitted)}", flush=True)
    for i, step in enumerate(emitted):
        question_preview = str(step.get("question") or step.get("label") or "")[:60]
        print(
            f"[FlowPlanner] Step {i+1}: id={step.get('id')}, type={step.get('type')}, question={question_preview}...",
            flush=True,
        )

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
    batch_constraints_for_session = prep.get("batch_constraints") or {}
    # Preserve stable key order for clients that rely on predictable JSON field ordering.
    # (Python dict preserves insertion order; JSONResponse will serialize in that order.)
    meta: Dict[str, Any] = {
        "requestId": request_id,
        "schemaVersion": prep.get("schema_version", "0"),
    }
    if _include_form_plan(payload):
        meta["formPlan"] = _build_form_plan_snapshot(
            constraints=batch_constraints_for_session if isinstance(batch_constraints_for_session, dict) else {},
        )
    meta["miniSteps"] = emitted
    include_meta = _include_response_meta(payload)
    if include_meta:
        meta["copyPackId"] = prep.get("copy_pack_id")
        meta["copyPackVersion"] = prep.get("copy_pack_version")
        meta["lintFailed"] = lint_failed
        meta["lintViolationCodes"] = _summarize_violation_codes(violations)
        # Pass through session info from payload if available
        session_info = payload.get("session")
        if isinstance(session_info, dict):
            meta["session"] = session_info
    # Intentionally do not include deprecated form-level plan snapshots in responses.
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
            if include_meta and track_usage:
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
        from programs.batch_generator.form_planning.composite import wrap_last_step_with_upload_composite

        wrapped_steps, did_wrap = wrap_last_step_with_upload_composite(
            payload=payload,
            emitted_steps=meta.get("miniSteps") or [],
            required_uploads=context.get("required_uploads"),
        )
        if did_wrap and wrapped_steps is not None:
            meta["miniSteps"] = wrapped_steps
    except Exception:
        did_wrap = False

    # If we did not wrap uploads into a composite step, provide deterministic placement info.
    # This lets the frontend inject deterministic/structural steps (uploads, CTAs, etc) without LLM output.
    if not did_wrap:
        try:
            from programs.batch_generator.form_planning.ui_plan import build_deterministic_placements

            ui_plan = build_deterministic_placements(
                payload=payload,
                final_form_plan=[],
                emitted_mini_steps=meta.get("miniSteps") or [],
                required_uploads=context.get("required_uploads"),
            )
            if ui_plan is not None:
                meta["deterministicPlacements"] = ui_plan
        except Exception:
            pass

    # Log the final response meta
    print(
        f"[FlowPlanner] Final response: requestId={meta['requestId']}, latencyMs={latency_ms}, steps={len(meta['miniSteps'])}, model={lm_cfg.get('modelName') or lm_cfg.get('model')}",
        flush=True,
    )
    try:
        fp = meta.get("formPlan")
        if fp is None:
            print("[FlowPlanner] Response formPlan: <NULL>", flush=True)
        elif isinstance(fp, dict):
            keys = ",".join(list(fp.keys())[:12])
            print(f"[FlowPlanner] Response formPlan keys: {keys}", flush=True)
        else:
            print(f"[FlowPlanner] Response formPlan: <{type(fp).__name__}>", flush=True)
    except Exception:
        pass

    return meta

def _default_next_steps_demo_pack() -> str:
    """
    Prefer env override; otherwise use the vendored shared contract demos.
    """
    env_pack = (os.getenv("DSPY_NEXT_STEPS_DEMO_PACK") or "").strip()
    if env_pack:
        return env_pack
    shared = _repo_root() / "shared" / "ai-form-contract" / "demos" / "next_steps_examples.jsonl"
    if shared.exists():
        return str(shared)
    return ""


def _best_effort_contract_schema_version() -> str:
    try:
        p = _repo_root() / "shared" / "ai-form-contract" / "schema" / "schema_version.txt"
        if p.exists():
            v = p.read_text(encoding="utf-8").strip()
            return v or "0"
    except Exception:
        pass
    return "0"


def _repo_root() -> Path:
    """
    Resolve the repository root when running from a `src/` layout.

    `src/app/pipeline/form_pipeline.py` lives under the `src/` layout,
    so the repo root is 3 parents up: pipeline -> app -> src -> repo root.
    """
    return Path(__file__).resolve().parents[3]


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
        import dspy
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


def _include_response_meta(payload: Dict[str, Any]) -> bool:
    """
    Control extra response metadata (copy pack ids, lint status, session passthrough, etc).

    Default is off to keep responses lean. Enable with:
    - env `AI_FORM_INCLUDE_META=true`
    - request flag `{"request": {"includeMeta": true}}`
    """
    if os.getenv("AI_FORM_INCLUDE_META") == "true":
        return True
    req = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    return bool(req.get("includeMeta") is True or str(req.get("includeMeta") or "").lower() == "true")


def _include_form_plan(payload: Dict[str, Any]) -> bool:
    """
    Whether to include `formPlan` in the response.

    Default is YES. Some clients do not send `currentBatch`, but still need `formPlan`
    on every call to keep the UI and backend in sync.
    """
    req = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    if req.get("excludeFormPlan") is True or str(req.get("excludeFormPlan") or "").lower() == "true":
        return False
    return True


def _build_form_plan_snapshot(
    *,
    constraints: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a minimal `formPlan` snapshot that exposes the backend constraints we enforce.
    """
    try:
        max_batches = int(constraints.get("maxBatches") or 0)
    except Exception:
        max_batches = 0
    if max_batches <= 0:
        max_batches = 1
    try:
        max_steps_per_batch = int(constraints.get("maxStepsPerBatch") or 0)
    except Exception:
        max_steps_per_batch = 0
    if max_steps_per_batch <= 0:
        max_steps_per_batch = 5

    batches = [{"batchId": f"batch-{i + 1}", "maxSteps": max_steps_per_batch} for i in range(max_batches)]
    batches = _fill_missing_batches(batches=batches, max_batches=max_batches, default_max_steps=max_steps_per_batch)

    # Keep the payload small and widget-friendly: batches + constraints only.
    constraints_obj = {
        "maxBatches": constraints.get("maxBatches"),
        "maxStepsTotal": constraints.get("maxStepsTotal"),
        "maxStepsPerBatch": constraints.get("maxStepsPerBatch"),
        "tokenBudgetTotal": constraints.get("tokenBudgetTotal"),
    }
    # Backward-compatible shape: some clients expect `v:1` + `form.constraints`.
    # Newer clients expect `version:2` + top-level `constraints`.
    return {
        "v": 1,
        "version": 2,
        "batches": batches,
        "constraints": constraints_obj,
        "form": {"constraints": constraints_obj},
        "stop": None,
        "notes": None,
    }


def _parse_must_have_copy(text: Any) -> Dict[str, Any]:
    obj = _best_effort_parse_json(str(text or ""))
    if not isinstance(obj, dict):
        return {}
    try:
        from core.schemas.ui_steps import StepCopy
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
