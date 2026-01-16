"""
Deterministic lint rules for copy style enforcement.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


RISK_SCORE = {"low": 1, "medium": 2, "high": 3, "very_high": 4}


def _lower(text: Any) -> str:
    return str(text or "").strip().lower()


def _limits_from_config(lint_config: Dict[str, Any]) -> Dict[str, int]:
    limits = lint_config.get("limits") or {}
    return {
        "question_max_chars": int(limits.get("question_max_chars") or 140),
        "option_label_max_chars": int(limits.get("option_label_max_chars") or 48),
        "option_count_min": int(limits.get("option_count_min") or 4),
        "option_count_max": int(limits.get("option_count_max") or 10),
        "placeholder_max_chars": int(limits.get("placeholder_max_chars") or 60),
    }


def _risk_levels(lint_config: Dict[str, Any]) -> Dict[str, List[str]]:
    raw = lint_config.get("risk_levels") or {}
    if isinstance(raw, dict) and raw:
        return {k: [str(x).lower() for x in (v or [])] for k, v in raw.items()}
    return {
        "low": ["multiple_choice", "choice", "rating", "slider"],
        "medium": ["text", "text_input"],
        "high": ["lead_capture"],
        "very_high": ["upload", "file_upload", "file_picker"],
    }


def _risk_score_for_step(step: Dict[str, Any], lint_config: Dict[str, Any]) -> int:
    step_type = _lower(step.get("type"))
    levels = _risk_levels(lint_config)
    for label, types in levels.items():
        if step_type in types:
            return RISK_SCORE.get(label, 1)
    # Elevate text inputs that clearly request contact info
    if step_type in {"text", "text_input"}:
        combined = f"{_lower(step.get('id'))} {_lower(step.get('question'))}"
        if "email" in combined or "phone" in combined:
            return RISK_SCORE["high"]
    return RISK_SCORE["low"]


def _contains_banned(text: str, banned_phrases: List[str]) -> str:
    hay = _lower(text)
    for phrase in banned_phrases:
        p = _lower(phrase)
        if p and p in hay:
            return phrase
    return ""


def _has_markdown_fence(text: str) -> bool:
    return "```" in text or "~~~" in text


def _strip_markdown_fences(text: str) -> str:
    return str(text or "").replace("```", "").replace("~~~", "").strip()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return f"{text[: max_chars - 3].rstrip()}..."


def _remove_banned(text: str, banned_phrases: List[str]) -> str:
    cleaned = str(text or "")
    for phrase in banned_phrases:
        p = str(phrase or "").strip()
        if not p:
            continue
        cleaned = re.sub(re.escape(p), "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _fix_excessive_punct(text: str) -> str:
    return re.sub(r"([!?])\1{2,}", r"\1\1", str(text or ""))


def _fallback_question(step_type: str, templates: Dict[str, Any]) -> str:
    if step_type in {"lead_capture"}:
        prompt = str(templates.get("lead_capture_prompt") or "").strip()
        if prompt:
            return prompt
        return "How should we contact you?"
    if step_type in {"upload", "file_upload", "file_picker"}:
        return "Please upload a file."
    if step_type in {"text", "text_input"}:
        return "Share a short detail."
    if step_type in {
        "multiple_choice",
        "choice",
        "segmented_choice",
        "chips_multi",
        "yes_no",
        "image_choice_grid",
        "searchable_select",
    }:
        return "Which option fits best?"
    if step_type in {"rating", "slider", "range_slider"}:
        return "How would you rate this?"
    return "Please share your answer."


def _fallback_options() -> List[Dict[str, str]]:
    return [
        {"label": "Not sure yet", "value": "not_sure"},
        {"label": "I'm flexible", "value": "flexible"},
        {"label": "Other / depends", "value": "other"},
        {"label": "Prefer not to say", "value": "prefer_not_to_say"},
    ]


def _step_has_reassurance(step: Dict[str, Any], lint_config: Dict[str, Any]) -> bool:
    phrases = lint_config.get("reassurance_phrases") or []
    combined = " ".join(
        [
            str(step.get("question") or ""),
            str(step.get("subtext") or ""),
            str(step.get("humanism") or ""),
        ]
    ).lower()
    for phrase in phrases:
        p = _lower(phrase)
        if p and p in combined:
            return True
    return False


def apply_reassurance(steps: List[Dict[str, Any]], lint_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not steps or not lint_config:
        return steps
    templates = lint_config.get("templates") or {}
    privacy = str(templates.get("privacy_reassurance") or "").strip()
    upload = str(templates.get("upload_reassurance") or privacy).strip()
    for step in steps:
        if not isinstance(step, dict):
            continue
        risk = _risk_score_for_step(step, lint_config)
        if risk < RISK_SCORE["high"]:
            continue
        if _step_has_reassurance(step, lint_config):
            continue
        step_type = _lower(step.get("type"))
        reassurance = upload if step_type in {"upload", "file_upload", "file_picker"} else privacy
        if not reassurance:
            continue
        if not step.get("subtext"):
            step["subtext"] = reassurance
            continue
        if not step.get("humanism"):
            step["humanism"] = reassurance
            continue
        question = str(step.get("question") or "").strip()
        if question and _lower(reassurance) not in _lower(question):
            step["question"] = f"{question} {reassurance}".strip()
    return steps


def sanitize_step(step: Dict[str, Any], lint_config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(step, dict):
        return step
    limits = _limits_from_config(lint_config)
    banned_phrases = [str(x) for x in (lint_config.get("banned_phrases") or []) if str(x).strip()]
    templates = lint_config.get("templates") or {}
    step_type = _lower(step.get("type"))

    question = _strip_markdown_fences(step.get("question"))
    question = _remove_banned(question, banned_phrases)
    question = _fix_excessive_punct(question)
    question = question.strip()
    if not question:
        question = _fallback_question(step_type, templates)
    question = _truncate(question, limits["question_max_chars"])
    step["question"] = question

    if step_type in {"text", "text_input"}:
        placeholder = _strip_markdown_fences(step.get("placeholder"))
        placeholder = _remove_banned(placeholder, banned_phrases)
        placeholder = _truncate(placeholder, limits["placeholder_max_chars"])
        if placeholder:
            step["placeholder"] = placeholder

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
        options = step.get("options")
        if not isinstance(options, list) or not options:
            options = _fallback_options()
        cleaned: List[Dict[str, str]] = []
        for opt in options:
            if not isinstance(opt, dict):
                opt = {}
            label = _strip_markdown_fences(opt.get("label"))
            label = _remove_banned(label, banned_phrases).strip()
            if not label:
                label = "Option"
            label = _truncate(label, limits["option_label_max_chars"])
            value = _lower(opt.get("value") or label)
            value = re.sub(r"\s+", "_", value).strip("_")
            if not value:
                value = "option"
            cleaned.append({"label": label, "value": value})
        if len(cleaned) < limits["option_count_min"]:
            cleaned += _fallback_options()[: limits["option_count_min"] - len(cleaned)]
        if len(cleaned) > limits["option_count_max"]:
            cleaned = cleaned[: limits["option_count_max"]]
        step["options"] = cleaned

    if any(
        _has_markdown_fence(str(text or ""))
        for text in [step.get("subtext"), step.get("humanism")]
    ):
        step["subtext"] = _strip_markdown_fences(step.get("subtext"))
        step["humanism"] = _strip_markdown_fences(step.get("humanism"))

    apply_reassurance([step], lint_config)
    return step


def sanitize_steps(steps: List[Dict[str, Any]], lint_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not steps:
        return steps
    return [sanitize_step(dict(step), lint_config) if isinstance(step, dict) else step for step in steps]


def lint_steps(steps: List[Dict[str, Any]], lint_config: Dict[str, Any]) -> Tuple[bool, List[Dict[str, str]], List[str]]:
    violations: List[Dict[str, str]] = []
    bad_ids: List[str] = []
    if not steps:
        return True, violations, bad_ids

    limits = _limits_from_config(lint_config)
    banned_phrases = [str(x) for x in (lint_config.get("banned_phrases") or []) if str(x).strip()]
    option_types = {
        "multiple_choice",
        "choice",
        "segmented_choice",
        "chips_multi",
        "yes_no",
        "image_choice_grid",
        "searchable_select",
    }

    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "")
        question = str(step.get("question") or "")
        before_count = len(violations)

        if question and len(question) > limits["question_max_chars"]:
            violations.append(
                {
                    "step_id": step_id,
                    "code": "question_too_long",
                    "message": f"Question exceeds {limits['question_max_chars']} characters",
                }
            )

        banned_hit = _contains_banned(question, banned_phrases)
        if banned_hit:
            violations.append(
                {
                    "step_id": step_id,
                    "code": "question_banned_phrase",
                    "message": f"Question contains banned phrase '{banned_hit}'",
                }
            )

        if re.search(r"[!?]{3,}", question):
            violations.append(
                {
                    "step_id": step_id,
                    "code": "excessive_punctuation",
                    "message": "Question uses excessive punctuation",
                }
            )

        if any(
            _has_markdown_fence(str(text or ""))
            for text in [
                question,
                step.get("subtext"),
                step.get("humanism"),
                step.get("placeholder"),
            ]
        ):
            violations.append(
                {
                    "step_id": step_id,
                    "code": "markdown_fence",
                    "message": "Markdown/code fences are not allowed",
                }
            )

        step_type = _lower(step.get("type"))
        options = step.get("options") if isinstance(step.get("options"), list) else None
        if options is None and step_type in option_types:
            violations.append(
                {
                    "step_id": step_id,
                    "code": "options_missing",
                    "message": "Options are required for this step type",
                }
            )
        if options is not None:
            option_count = len(options)
            if option_count < limits["option_count_min"] or option_count > limits["option_count_max"]:
                violations.append(
                    {
                        "step_id": step_id,
                        "code": "options_count",
                        "message": f"Options count must be {limits['option_count_min']}-{limits['option_count_max']}",
                    }
                )
            for opt in options:
                if not isinstance(opt, dict):
                    violations.append(
                        {
                            "step_id": step_id,
                            "code": "option_shape",
                            "message": "Option must be an object with label/value",
                        }
                    )
                    continue
                label = str(opt.get("label") or "")
                value = str(opt.get("value") or "")
                if not label or not value:
                    violations.append(
                        {
                            "step_id": step_id,
                            "code": "option_missing_fields",
                            "message": "Each option must include label and value",
                        }
                    )
                if label and len(label) > limits["option_label_max_chars"]:
                    violations.append(
                        {
                            "step_id": step_id,
                            "code": "option_label_too_long",
                            "message": f"Option label exceeds {limits['option_label_max_chars']} characters",
                        }
                    )
                banned_label = _contains_banned(label, banned_phrases)
                if banned_label:
                    violations.append(
                        {
                            "step_id": step_id,
                            "code": "option_label_banned_phrase",
                            "message": f"Option label contains banned phrase '{banned_label}'",
                        }
                    )
                if value and " " in value:
                    violations.append(
                        {
                            "step_id": step_id,
                            "code": "option_value_spaces",
                            "message": "Option values must not contain spaces",
                        }
                    )
                if _has_markdown_fence(label):
                    violations.append(
                        {
                            "step_id": step_id,
                            "code": "markdown_fence",
                            "message": "Markdown/code fences are not allowed",
                        }
                    )

        if step_type in {"text", "text_input"}:
            placeholder = str(step.get("placeholder") or "")
            if placeholder and len(placeholder) > limits["placeholder_max_chars"]:
                violations.append(
                    {
                        "step_id": step_id,
                        "code": "placeholder_too_long",
                        "message": f"Placeholder exceeds {limits['placeholder_max_chars']} characters",
                    }
                )

        if _risk_score_for_step(step, lint_config) >= RISK_SCORE["high"]:
            if not _step_has_reassurance(step, lint_config):
                violations.append(
                    {
                        "step_id": step_id,
                        "code": "missing_reassurance",
                        "message": "High-risk step requires reassurance text",
                    }
                )

        if len(violations) > before_count and step_id and step_id not in bad_ids:
            bad_ids.append(step_id)

    return len(violations) == 0, violations, bad_ids
