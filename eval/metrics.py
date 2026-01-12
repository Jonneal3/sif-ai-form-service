"""
Deterministic metrics for offline DSPy evaluation.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

from flow_planner import (
    _best_effort_parse_json,
    _load_signature_types,
    _normalize_step_id,
    _validate_mini,
)


def _coerce_int(value: Any, default: int) -> int:
    try:
        n = int(value)
        return n if n > 0 else default
    except Exception:
        return default


def _parse_context_from_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    context_json = inputs.get("context_json") or ""
    parsed = _best_effort_parse_json(context_json)
    return parsed if isinstance(parsed, dict) else {}


def _extract_copy_limit(context: Dict[str, Any]) -> int | None:
    raw = context.get("copy_style")
    if not raw:
        return None
    parsed = _best_effort_parse_json(raw)
    if not isinstance(parsed, dict):
        return None
    limits = parsed.get("limits")
    if not isinstance(limits, dict):
        return None
    try:
        limit = int(limits.get("question_max_chars") or 0)
    except Exception:
        return None
    return limit if limit > 0 else None


def _count_keywords(text: str, keywords: List[str]) -> int:
    hay = str(text or "").lower()
    return sum(1 for k in keywords if k in hay)


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _tokenize(text: str) -> List[str]:
    t = _normalize_text(text)
    return [x for x in t.split() if x]


def _has_banned_option_set(step: Dict[str, Any]) -> bool:
    options = step.get("options") if isinstance(step.get("options"), list) else None
    if not options:
        return False
    tokens: set[str] = set()
    for opt in options:
        if isinstance(opt, dict):
            label = f"{opt.get('label') or ''} {opt.get('value') or ''}"
        else:
            label = str(opt or "")
        normalized = _normalize_text(label)
        parts = normalized.split()
        if "abstract" in parts:
            return True
        if len(parts) == 1:
            tokens.add(parts[0])
    banned_sets = [
        {"red", "blue", "green"},
        {"circle", "square", "triangle"},
    ]
    for banned in banned_sets:
        if banned.issubset(tokens) and len(tokens) <= len(banned) + 1:
            return True
    return False


def _parse_steps(prediction_jsonl: str, ui_types: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    parsed_steps: List[Dict[str, Any]] = []
    parse_failures = 0
    for line in str(prediction_jsonl or "").splitlines():
        t = line.strip()
        if not t:
            continue
        obj = _best_effort_parse_json(t)
        v = _validate_mini(obj, ui_types)
        if v:
            parsed_steps.append(v)
        else:
            parse_failures += 1
    return parsed_steps, parse_failures


def score_prediction(example_inputs: Dict[str, Any], prediction_jsonl: str) -> Tuple[float, Dict[str, int]]:
    """
    Returns (score, details) where details is a dict of failure counts.
    """
    _, ui_types = _load_signature_types()
    details: Dict[str, int] = {
        "empty_output": 0,
        "parse_failures": 0,
        "invalid_type": 0,
        "exceeds_max_steps": 0,
        "duplicate_ids": 0,
        "already_asked": 0,
        "known_answer": 0,
        "missing_id": 0,
        "options_missing": 0,
        "options_bad": 0,
        "question_too_long": 0,
        "visual_missing": 0,
        "vague_terms": 0,
        "banned_option_set": 0,
        "disallowed_family": 0,
        "generic_shape_size": 0,
        "service_relevance_fail": 0,
    }

    steps, parse_failures = _parse_steps(prediction_jsonl, ui_types)
    if not steps and not parse_failures:
        details["empty_output"] = 1
        return 0.0, details
    if parse_failures:
        details["parse_failures"] = parse_failures
        return 0.0, details

    max_steps = _coerce_int(example_inputs.get("max_steps"), 4)
    allowed = example_inputs.get("allowed_mini_types") or []
    allowed_set = {str(x).strip().lower() for x in allowed if str(x).strip()}
    context = _parse_context_from_inputs(example_inputs)
    already_asked = {
        _normalize_step_id(str(x).strip())
        for x in (context.get("already_asked_keys") or [])
        if str(x).strip()
    }
    known_answers = context.get("known_answers")
    known_answer_ids = (
        {_normalize_step_id(str(k)) for k in known_answers.keys()}
        if isinstance(known_answers, dict)
        else set()
    )
    question_limit = _extract_copy_limit(context)
    allowed_families = {
        str(f.get("family")).strip()
        for f in (context.get("attribute_families") or [])
        if isinstance(f, dict) and str(f.get("family") or "").strip()
    }
    service_anchor_set: set[str] = set()
    for term in (context.get("service_anchor_terms") or []):
        for tok in _tokenize(term):
            service_anchor_set.add(tok)
    steps_with_anchor_overlap = 0

    seen_ids: set[str] = set()
    visual_keywords = [
        "color",
        "tone",
        "palette",
        "material",
        "texture",
        "finish",
        "pattern",
        "shape",
        "size",
        "scale",
        "lighting",
        "light",
        "shadow",
        "background",
        "environment",
        "scene",
        "composition",
        "layout",
        "contrast",
        "detail",
        "surface",
        "gloss",
        "matte",
    ]
    vague_keywords = [
        "style",
        "constraints",
        "additional info",
        "details",
        "notes",
        "other",
        "misc",
        "general",
    ]
    generic_question_patterns = [
        re.compile(r"\bwhat\s+shape\s+do\s+you\s+want\b"),
        re.compile(r"\bwhat\s+size\s+do\s+you\s+need\b"),
    ]
    for step in steps:
        step_id = _normalize_step_id(str(step.get("id") or ""))
        step_type = str(step.get("type") or "").strip().lower()
        if allowed_set and step_type not in allowed_set:
            details["invalid_type"] += 1
        if step_id:
            if step_id in seen_ids:
                details["duplicate_ids"] += 1
            seen_ids.add(step_id)
            if step_id in already_asked:
                details["already_asked"] += 1
            if step_id in known_answer_ids:
                details["known_answer"] += 1
        else:
            details["missing_id"] += 1
        if question_limit and len(str(step.get("question") or "")) > question_limit:
            details["question_too_long"] += 1

        option_types = {
            "multiple_choice",
            "choice",
            "segmented_choice",
            "chips_multi",
            "yes_no",
            "image_choice_grid",
            "searchable_select",
        }
        if step_type in option_types:
            options = step.get("options") if isinstance(step.get("options"), list) else None
            if options is None:
                details["options_missing"] += 1
            if isinstance(options, list):
                if len(options) < 3 or len(options) > 6:
                    details["options_bad"] += 1
                for opt in options:
                    if not isinstance(opt, dict):
                        details["options_bad"] += 1
                        break
                    label = str(opt.get("label") or "")
                    value = str(opt.get("value") or "")
                    if not label or not value:
                        details["options_bad"] += 1
                        break
            if _has_banned_option_set(step):
                details["banned_option_set"] += 1

        question = str(step.get("question") or "")
        option_labels = ""
        if isinstance(step.get("options"), list):
            option_labels = " ".join(
                str(opt.get("label") or "")
                for opt in step.get("options")
                if isinstance(opt, dict)
            )
        combined = f"{question} {option_labels}".strip()
        combined_tokens = set(_tokenize(combined))
        if service_anchor_set and combined_tokens.intersection(service_anchor_set):
            steps_with_anchor_overlap += 1
        if any(p.search(_normalize_text(question)) for p in generic_question_patterns):
            if not (service_anchor_set and combined_tokens.intersection(service_anchor_set)):
                details["generic_shape_size"] += 1
        if _count_keywords(combined, visual_keywords) == 0:
            details["visual_missing"] += 1
        if _count_keywords(combined, vague_keywords) > 0:
            details["vague_terms"] += 1
        blueprint = step.get("blueprint") if isinstance(step.get("blueprint"), dict) else {}
        family = str(blueprint.get("family") or "").strip()
        if family and allowed_families and family not in allowed_families:
            details["disallowed_family"] += 1

    if max_steps and len(steps) > max_steps:
        details["exceeds_max_steps"] = len(steps) - max_steps
    if service_anchor_set and steps:
        required_overlap = max(1, int(math.ceil(len(steps) * 0.6)))
        if steps_with_anchor_overlap < required_overlap:
            details["service_relevance_fail"] = 1

    if (
        details["banned_option_set"]
        or details["disallowed_family"]
        or details["generic_shape_size"]
        or details["service_relevance_fail"]
    ):
        return 0.0, details

    score = 1.0
    if details["invalid_type"]:
        score -= 0.8
    if details["exceeds_max_steps"]:
        score -= 0.2
    if details["duplicate_ids"]:
        score -= 0.2
    if details["already_asked"]:
        score -= 0.2
    if details["known_answer"]:
        score -= 0.2
    if details["missing_id"]:
        score -= 0.4
    if details["options_missing"]:
        score -= 0.6
    if details["options_bad"]:
        score -= 0.1
    if details["question_too_long"]:
        score -= 0.05
    if details["visual_missing"]:
        score -= 0.2
    if details["vague_terms"]:
        score -= 0.1

    score = max(0.0, min(1.0, score))
    return score, details
