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
- Signature: `src/programs/batch_generator/signatures/signature.py` → `BatchNextStepsJSONL`
- Program wrapper: `src/programs/batch_generator/engine/dspy_program.py` → `BatchStepsProgram`
- Orchestrator entrypoint: `src/programs/batch_generator/orchestrator.py` → `next_steps_jsonl(...)`

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

# Keep orchestrator smaller by importing shared helpers.
from programs.batch_generator.engine.parse_validate import (
    _best_effort_parse_json,
    _extract_required_upload_ids,
    _looks_like_upload_step_id,
    _reject_banned_option_sets,
    _validate_mini,
)
from programs.batch_generator.engine.utils import _compact_json, _normalize_step_id

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

"""
Note: JSON parsing + step validation helpers live in:
- `programs.batch_generator.engine.parse_validate`
- `programs.batch_generator.engine.utils`
"""


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
    """
    Standardize where we look for RAG/grounding text.

    We accept multiple shapes for back-compat:
    - top-level keys (service callers)
    - `state.*` and `state.context.*` (widget callers)
    - `instanceContext.*` (preferred shared place when present)
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
                # Keep it short to avoid blowing up prompts and leaking too much content.
                return text[:800]

    return ""


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
    # End-of-flow UI (used by the last-batch finisher).
    "gallery",
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
    # Be strict: only `choice` is treated as an alias for `multiple_choice`.
    # Other choice-like variants must be explicitly allowed.
    if t == "choice":
        return "choice" in allowed or "multiple_choice" in allowed
    if t == "multiple_choice":
        return "multiple_choice" in allowed or "choice" in allowed
    if t in ["text", "text_input"]:
        return "text" in allowed or "text_input" in allowed
    if t in ["slider", "rating", "range_slider"]:
        return "slider" in allowed or "rating" in allowed or "range_slider" in allowed
    if t in ["upload", "file_upload", "file_picker"]:
        return "upload" in allowed or "file_upload" in allowed or "file_picker" in allowed
    if t in ["gallery"]:
        return "gallery" in allowed
    return False


def _extract_batch_number_1based(payload: Dict[str, Any]) -> int:
    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}
    raw = (
        current_batch.get("batchNumber")
        or current_batch.get("batch_number")
        or payload.get("batchNumber")
        or payload.get("batch_number")
        or 1
    )
    try:
        n = int(raw)
    except Exception:
        n = 1
    return max(1, n)


def _extract_total_batches_from_context(context: Dict[str, Any]) -> int:
    if not isinstance(context, dict):
        return 1
    constraints = context.get("batch_constraints") if isinstance(context.get("batch_constraints"), dict) else {}
    raw = constraints.get("maxBatches")
    try:
        n = int(raw) if raw is not None else None
    except Exception:
        n = None
    if not isinstance(n, int) or n <= 0:
        n = _extract_max_batches_from_context(context)
    if not isinstance(n, int) or n <= 0:
        n = 1
    return max(1, min(10, n))


def _is_last_batch(payload: Dict[str, Any], context: Dict[str, Any]) -> bool:
    batch_number = _extract_batch_number_1based(payload)
    total_batches = _extract_total_batches_from_context(context)
    return batch_number >= total_batches


def _step_type_is_upload(step: Dict[str, Any]) -> bool:
    t = str((step or {}).get("type") or "").strip().lower()
    return t in {"upload", "file_upload", "file_picker"}


def _step_type_is_gallery(step: Dict[str, Any]) -> bool:
    t = str((step or {}).get("type") or "").strip().lower()
    return t == "gallery"


def _ensure_unique_step_id(*, preferred: str, taken: set[str]) -> str:
    base = _normalize_step_id(str(preferred or "").strip()) or "step"
    if base not in taken:
        return base
    i = 2
    while f"{base}-{i}" in taken:
        i += 1
    return f"{base}-{i}"


def _deterministic_finish_last_batch(
    *,
    steps: list[dict],
    ui_types: Dict[str, Any],
    required_upload_ids: set[str],
    blocked_step_ids: set[str],
    max_steps_limit: Optional[int],
) -> list[dict]:
    """
    Last-batch invariant:
    - The final 2 steps must be Upload → Gallery.
    - If we exceed max steps, trim earlier (non-required) steps first.
    """
    out = [s for s in (steps or []) if isinstance(s, dict)]

    taken_ids: set[str] = {str(s.get("id") or "").strip() for s in out if str(s.get("id") or "").strip()}
    taken_ids |= {str(x or "").strip() for x in (blocked_step_ids or set()) if str(x or "").strip()}

    # Pull an existing gallery step (if present) so we can re-append it at the end.
    gallery_step: Optional[dict] = None
    for idx in range(len(out) - 1, -1, -1):
        if _step_type_is_gallery(out[idx]):
            gallery_step = out.pop(idx)
            break

    # Pull one upload step (if present) so we can re-append it right before gallery.
    upload_step: Optional[dict] = None
    preferred_upload_idx: Optional[int] = None
    if required_upload_ids:
        for idx in range(len(out) - 1, -1, -1):
            s = out[idx]
            if not _step_type_is_upload(s):
                continue
            sid = str(s.get("id") or "").strip()
            if sid and sid in required_upload_ids:
                preferred_upload_idx = idx
                break
    if preferred_upload_idx is None:
        for idx in range(len(out) - 1, -1, -1):
            if _step_type_is_upload(out[idx]):
                preferred_upload_idx = idx
                break
    if preferred_upload_idx is not None:
        upload_step = out.pop(preferred_upload_idx)

    # If missing, synthesize a valid upload step.
    if not upload_step:
        preferred_id = "step-upload"
        if required_upload_ids:
            preferred_id = sorted([str(x) for x in required_upload_ids if str(x or "").strip()])[0] or preferred_id
        upload_id = _ensure_unique_step_id(preferred=preferred_id, taken=taken_ids)
        taken_ids.add(upload_id)
        candidate = {
            "id": upload_id,
            "type": "upload",
            "question": "Upload your files",
            "subtext": "Add any files that help.",
            "required": True,
        }
        upload_step = _validate_mini(candidate, ui_types) or candidate

    # If missing, synthesize a valid gallery step.
    if not gallery_step:
        gallery_id = _ensure_unique_step_id(preferred="step-gallery", taken=taken_ids)
        taken_ids.add(gallery_id)
        candidate = {
            "id": gallery_id,
            "type": "gallery",
            "question": "Review your uploads",
            "subtext": "Make sure everything looks right.",
            "required": True,
        }
        gallery_step = _validate_mini(candidate, ui_types) or candidate

    # Append in the required order.
    out.append(upload_step)
    out.append(gallery_step)

    # Enforce max step count by trimming earlier steps first.
    if isinstance(max_steps_limit, int) and max_steps_limit > 0:
        if max_steps_limit < 2:
            max_steps_limit = 2
        if len(out) > max_steps_limit:
            keep_non_required = max(0, max_steps_limit - 2)
            core = out[:-2][:keep_non_required]
            out = core + out[-2:]

    print("[FlowPlanner] Enforced last-batch finisher: Upload -> Gallery", flush=True)
    return out


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
    # Kept for compatibility with existing call sites; we intentionally ignore `use_case`/`goal_intent`
    # when using a single fixed constraint set.
    default_max_calls = 2
    try:
        from programs.batch_generator.planning.form_planning.static_constraints import DEFAULT_CONSTRAINTS

        default_max_calls = int((DEFAULT_CONSTRAINTS or {}).get("maxBatches") or default_max_calls)
    except Exception:
        default_max_calls = 2

    return max(1, min(10, _get_int_env("AI_FORM_MAX_BATCH_CALLS", default_max_calls)))


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
    default_min_steps_per_batch = 2
    default_max_steps_per_batch = 4
    default_token_budget_total = 3000
    try:
        from programs.batch_generator.planning.form_planning.static_constraints import DEFAULT_CONSTRAINTS

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
        "minStepsPerBatch": min_steps_per_batch,
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

    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else {}
    required_uploads_raw = (
        payload.get("requiredUploads")
        or payload.get("required_uploads")
        or current_batch.get("requiredUploads")
        or current_batch.get("required_uploads")
        or []
    )
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
    # If the client didn't send asked step ids, infer them from known answers to avoid re-asking.
    # This is a best-effort backstop for older clients.
    if not normalized_already and isinstance(known_answers, dict) and known_answers:
        for k in list(known_answers.keys()):
            sid = _normalize_step_id(str(k or "").strip())
            if sid and sid.startswith("step-") and sid not in normalized_already:
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

    instance_context = payload.get("instanceContext") if isinstance(payload.get("instanceContext"), dict) else {}
    instance_context = instance_context or (payload.get("instance_context") if isinstance(payload.get("instance_context"), dict) else {})

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

    # Plain-English context anchors (critical when answers contain UUIDs).
    state_context = state.get("context") if isinstance(state.get("context"), dict) else {}

    # Multi-value support: if arrays are present, derive a short comma-separated industry/service string
    # for the LLM's high-level context (while keeping full objects in `instance_*` / `instance_context`).
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
        ", ".join(industry_names) if industry_names else (
            payload.get("industry")
            or payload.get("vertical")
            or state_context.get("industry")
            or state_context.get("categoryName")
            or "General"
        )
    )[:120]
    service = str(
        ", ".join(service_names) if service_names else (
            payload.get("service")
            or payload.get("subcategoryName")
            or state_context.get("subcategoryName")
            or ""
        )
    )[:120]
    use_case = _extract_use_case(payload)
    platform_goal = str(payload.get("platformGoal") or payload.get("platform_goal") or "")[:600]
    business_context = str(payload.get("businessContext") or payload.get("business_context") or state_context.get("businessContext") or "")[:200]
    goal_intent = _infer_goal_intent(platform_goal, business_context)
    grounding_summary = _extract_grounding_summary(payload)
    subcategory_summary = _summarize_instance_subcategories(instance_subcategories)
    if grounding_summary:
        grounding_summary = grounding_summary[:600]
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
        "instance_context": instance_context,
        "instance_categories": instance_categories,
        "instance_subcategories": instance_subcategories,
        "instance_subcategory_summary": subcategory_summary,
        "attribute_families": attribute_families,
        # Always include this key so prompts/debugging are stable.
        "grounding_summary": grounding_summary or "",
        "answered_qa": answered_qa,
        "choice_option_min": choice_option_min,
        "choice_option_max": choice_option_max,
        "choice_option_target": choice_option_target,
        # Set by the hardcoded flow guide in `_prepare_predictor`.
        "prefer_structured_inputs": False,
    }
    if grounding_summary:
        # Back-compat: some prompts/demos still reference `vertical_context`.
        # Keep it aligned with `grounding_summary` (RAG output only).
        context["vertical_context"] = grounding_summary
    if os.getenv("AI_FORM_DEBUG") == "true":
        gs = str(context.get("grounding_summary") or "")
        if not gs:
            print("[FlowPlanner] ⚠️ No grounding_summary provided", flush=True)
        print(f"[FlowPlanner] grounding_summary_len={len(gs)}", flush=True)
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
        GalleryUI,
        IntroUI,
        LeadCaptureUI,
        MultipleChoiceUI,
        PricingUI,
        RatingUI,
        SearchableSelectUI,
        TextInputUI,
    )
    from programs.batch_generator.signatures.signature import BatchNextStepsJSONL

    ui_types = {
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
    from programs.batch_generator.engine.dspy_program import BatchStepsProgram

    # Some clients (e.g. the consolidated /v1/api/form endpoint) do not send a batch id.
    # Default to the first batch.
    batch_id_raw = payload.get("batchId") or payload.get("batch_id") or "batch-1"
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

    context = _build_context(payload)

    # Attach demos matched to the current use-case + batch position (derived from constraints).
    # This prevents a single generic demo pack from training the model into bland/repetitive outputs.
    module = BatchStepsProgram(demo_pack=_select_next_steps_demo_pack(context=context, batch_number=batch_number))

    context["batch_id_raw"] = str(batch_id_raw)[:80]
    context["batch_phase_id"] = batch_id
    # Give the batch generator an explicit definition of what this phase means (purpose, focus keys, limits).
    # If the client didn't provide a prompt-level `form_plan`, synthesize one so the model
    # has stable target ids/keys for this batch (helps prevent empty/invalid outputs).
    if isinstance(context, dict) and not context.get("form_plan"):
        try:
            max_items = int((payload.get("maxSteps") or (payload.get("currentBatch") or {}).get("maxSteps") or 4))
            synthesized = []
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
    must_have_copy_needed = _extract_must_have_copy_needed(context.get("batch_state"))

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

    # Apply form-planning rules (copy pack + flow guide) in one explicit place.
    lint_config: Dict[str, Any] = {}
    copy_pack_id = "default_v1"
    try:
        from programs.batch_generator.planning.form_planning.entrypoints import apply_planning_rules

        context, allowed_mini_types, max_steps, lint_config, copy_pack_id = apply_planning_rules(
            payload=payload,
            context=context,
            batch_number=batch_number,
            extracted_allowed_mini_types=allowed_mini_types,
            extracted_max_steps=max_steps,
        )
    except Exception:
        lint_config = {}
        copy_pack_id = _resolve_copy_pack_id(payload)

    style_snippet_json = context.get("copy_style") if isinstance(context.get("copy_style"), str) else ""

    if context.get("prefer_structured_inputs"):
        allowed_mini_types = _prefer_structured_allowed_mini_types(allowed_mini_types)

    context_json = _compact_json(context)
    copy_context = dict(context)
    copy_context["must_have_copy_needed"] = must_have_copy_needed
    copy_context_json = _compact_json(copy_context)
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
        "lm": lm,
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
    lm = prep.get("lm")
    track_usage = prep.get("track_usage", False)
    copy_needed = prep.get("copy_needed", False)
    copy_context_json = prep.get("copy_context_json", "")
    max_steps = prep.get("max_steps", 4)
    max_steps_limit = max_steps if isinstance(max_steps, int) and max_steps > 0 else None
    allowed_set = set(prep.get("allowed_mini_types") or [])
    already_asked_keys = prep.get("already_asked_keys") or set()
    required_upload_ids = prep.get("required_upload_ids") or set()
    lint_config = prep.get("lint_config") or {}
    context = prep.get("context") or {}
    apply_reassurance = None
    lint_steps = None
    sanitize_steps = None
    if lint_config:
        try:
            from programs.batch_generator.planning.form_planning.copywriting.linter import (
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
    if os.getenv("AI_FORM_DEBUG") == "true":
        _print_lm_history_if_available(lm, n=1)

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
        # Keep batch ids stable (phase ids may be semantic, e.g. "ContextCore"/"Details").
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
            v = _reject_banned_option_sets(v)
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

    # Deterministic invariant: on the last batch, always end with Upload -> Gallery.
    if _is_last_batch(payload, context):
        emitted = _deterministic_finish_last_batch(
            steps=emitted,
            ui_types=ui_types,
            required_upload_ids=required_upload_ids,
            blocked_step_ids=already_asked_keys,
            max_steps_limit=max_steps_limit,
        )

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
    meta["miniSteps"] = emitted
    include_meta = _include_response_meta(payload)
    if include_meta:
        meta["copyPackId"] = prep.get("copy_pack_id")
        meta["copyPackVersion"] = prep.get("copy_pack_version")
        meta["lintFailed"] = lint_failed
        meta["lintViolationCodes"] = _summarize_violation_codes(violations)
        # Lightweight debug context to confirm what the model actually saw.
        # (Helps diagnose "RAG not applied" / "wrong allowed types" issues.)
        try:
            ctx = prep.get("context") if isinstance(prep.get("context"), dict) else {}
            fg = ctx.get("flow_guide") if isinstance(ctx.get("flow_guide"), dict) else {}
            meta["debugContext"] = {
                "industry": ctx.get("industry"),
                "service": ctx.get("service"),
                "useCase": ctx.get("use_case"),
                "goalIntent": ctx.get("goal_intent"),
                "groundingSummaryLen": len(str(ctx.get("grounding_summary") or "")),
                "groundingSummaryPreview": str(ctx.get("grounding_summary") or "").replace("\n", " ")[:160],
                "stage": fg.get("stage"),
                "allowedMiniTypes": prep.get("allowed_mini_types"),
                "maxSteps": prep.get("max_steps"),
            }
        except Exception:
            pass
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

    # Note: we intentionally do not do deterministic composite wrapping here.
    # Frontend renders `miniSteps[]` as-is.

    # Log the final response meta
    print(
        f"[FlowPlanner] Final response: requestId={meta['requestId']}, latencyMs={latency_ms}, steps={len(meta['miniSteps'])}, model={lm_cfg.get('modelName') or lm_cfg.get('model')}",
        flush=True,
    )
    return meta

def _select_next_steps_demo_pack(*, context: Dict[str, Any], batch_number: int) -> str:
    """
    Select a next-steps demo pack based on:
    - use_case (scene / scene_placement / tryon)
    - total_batches (derived from backend constraints)
    - batch_index (0-based)

    Preference order:
    1) explicit env override `DSPY_NEXT_STEPS_DEMO_PACK`
    2) optimized artifact for the current pack (if present)
    3) base pack for the current (use_case,total_batches,batch_index)
    4) legacy single-pack fallback
    """
    env_pack = (os.getenv("DSPY_NEXT_STEPS_DEMO_PACK") or "").strip()
    if env_pack:
        return env_pack

    use_case = str((context or {}).get("use_case") or "scene").strip().lower() or "scene"
    if use_case not in {"scene", "scene_placement", "tryon"}:
        use_case = "scene"

    constraints = (context or {}).get("batch_constraints") if isinstance((context or {}).get("batch_constraints"), dict) else {}
    total_batches_raw = constraints.get("maxBatches")
    try:
        total_batches = int(total_batches_raw) if total_batches_raw is not None else 0
    except Exception:
        total_batches = 0
    if total_batches <= 0:
        try:
            from programs.batch_generator.planning.form_planning.static_constraints import DEFAULT_CONSTRAINTS

            total_batches = int((DEFAULT_CONSTRAINTS or {}).get("maxBatches") or 3)
        except Exception:
            total_batches = 3
    total_batches = max(1, min(10, total_batches))

    try:
        idx0 = max(0, int(batch_number) - 1)
    except Exception:
        idx0 = 0
    batch_index = max(0, min(total_batches - 1, idx0))

    base_dir = _repo_root() / "src" / "programs" / "batch_generator" / "examples" / use_case / f"b{total_batches}"
    base_pack = base_dir / f"batch_{batch_index}.jsonl"
    optimized_pack = base_dir / f"batch_{batch_index}.optimized.jsonl"

    if optimized_pack.exists():
        return str(optimized_pack)
    if base_pack.exists():
        return str(base_pack)

    return _default_next_steps_demo_pack()


def _default_next_steps_demo_pack() -> str:
    """
    Prefer env override; otherwise use repo-local canonical demos for next-steps.
    """
    env_pack = (os.getenv("DSPY_NEXT_STEPS_DEMO_PACK") or "").strip()
    if env_pack:
        return env_pack
    local = (
        _repo_root()
        / "src"
        / "programs"
        / "batch_generator"
        / "examples"
        / "next_steps_examples.jsonl"
    )
    if local.exists():
        return str(local)
    shared_new = _repo_root() / "shared" / "ai-form-ui-contract" / "demos" / "next_steps_examples.jsonl"
    if shared_new.exists():
        return str(shared_new)
    shared_old = _repo_root() / "shared" / "ai-form-contract" / "demos" / "next_steps_examples.jsonl"
    if shared_old.exists():
        return str(shared_old)
    return ""


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
