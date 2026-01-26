from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from programs.form_pipeline.utils import _normalize_step_id

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
    if not s:
        return s
    t = str(s).strip()
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
    m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", t)
    if not m:
        return None
    return _safe_json_loads(m.group(0))


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
    if not isinstance(step, dict):
        return step
    out = dict(step)

    # Copy normalization:
    # Some model outputs use `question`, others use `title`. The widget expects
    # consistent fields, so we ensure both exist whenever either is present.
    raw_title = out.get("title")
    raw_question = out.get("question")
    title = str(raw_title or "").strip() if raw_title is not None else ""
    question = str(raw_question or "").strip() if raw_question is not None else ""
    if title and not question:
        out["question"] = title
    elif question and not title:
        out["title"] = question

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

    for k in (
        "stepId",
        "step_id",
        "stepID",
        "component_hint",
        "componentHint",
        "componentType",
        "component_type",
        "allowMultiple",
        "batch_phase_policy",
        "batchPhasePolicy",
    ):
        out.pop(k, None)

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

    if isinstance(step.get("options"), list):
        out["options"] = _coerce_options(step.get("options"))

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


def _reject_banned_option_sets(step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Hard guardrail:
    - If a step contains a known "toy" option set (colors/shapes/abstract), drop it.
    """
    if not _has_banned_option_set(step):
        return step
    return None


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


def _clean_options(options: Any) -> list:
    if not isinstance(options, list):
        return []
    cleaned: list[Any] = []
    placeholder_patterns = ["<<max_depth>>", "<<max_depth", "max_depth>>", "<max_depth>", "max_depth"]
    for opt in options:
        if isinstance(opt, dict):
            label = str(opt.get("label") or "")
            value = str(opt.get("value") or "")
            is_placeholder = any(
                pattern.lower() in label.lower() or pattern.lower() in value.lower() for pattern in placeholder_patterns
            )
            if not is_placeholder:
                cleaned.append(opt)
        elif isinstance(opt, str):
            is_placeholder = any(pattern.lower() in opt.lower() for pattern in placeholder_patterns)
            if not is_placeholder:
                cleaned.append(opt)
    return _coerce_options(cleaned)


def _fallback_step_id(*, step_type: str, question: str, options: Optional[list[dict]] = None) -> str:
    """
    Deterministic backstop id when the model forgets to emit an id.
    """
    t = re.sub(r"[^a-z0-9]+", "-", str(step_type or "").lower()).strip("-") or "step"
    q = re.sub(r"[^a-z0-9]+", "-", str(question or "").lower()).strip("-")
    q = "-".join([p for p in q.split("-") if p][:6])
    base = f"step-{t}"
    if q:
        base = f"{base}-{q}"
    if options:
        try:
            opt0 = str((options[0] or {}).get("value") or (options[0] or {}).get("label") or "")
            opt0 = re.sub(r"[^a-z0-9]+", "-", opt0.lower()).strip("-")
            if opt0:
                base = f"{base}-{opt0}"
        except Exception:
            pass
    return base[:64]


def _validate_mini(obj: Any, ui_types: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    if "id" not in obj:
        step_id = obj.get("stepId") or obj.get("step_id") or obj.get("stepID")
        if step_id:
            obj = dict(obj)
            obj["id"] = step_id
    if "type" not in obj:
        component_hint = obj.get("component_hint") or obj.get("componentHint") or obj.get("componentType") or obj.get(
            "component_type"
        )
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
            cleaned_options = _clean_options(obj.get("options"))
            if not cleaned_options:
                return None
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
            cleaned_options = _clean_options(obj.get("options"))
            if not cleaned_options:
                return None
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
        if t in ["gallery"]:
            out = ui_types["GalleryUI"].model_validate(obj).model_dump(by_alias=True)
            step_id = _normalize_step_id(str(out.get("id") or "").strip())
            if not step_id:
                step_id = _fallback_step_id(step_type=t, question=str(out.get("question") or ""))
            out["id"] = step_id
            return _canonicalize_step_output(out)
        return None
    except Exception:
        return None


__all__ = [
    "_best_effort_parse_json",
    "_reject_banned_option_sets",
    "_extract_required_upload_ids",
    "_looks_like_upload_step_id",
    "_validate_mini",
]

