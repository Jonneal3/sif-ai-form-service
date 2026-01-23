from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _strip_parenthetical_enumeration(q: str) -> str:
    # Remove trailing "(a, b, c)" style enumerations which duplicate the options list.
    return re.sub(r"\s*\([^)]{0,80}\)\s*$", "", q).strip()


def sanitize_steps(steps: List[dict], lint_config: Dict[str, Any]) -> List[dict]:
    out: List[dict] = []
    require_qmark = bool(lint_config.get("require_question_mark") is True)
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        s = dict(step)
        q = str(s.get("question") or "").strip()
        if q:
            q = _strip_parenthetical_enumeration(q)
            if require_qmark and not q.endswith("?"):
                q = q.rstrip(".").strip()
                if q and not q.endswith("?"):
                    q = f"{q}?"
            s["question"] = q
        out.append(s)
    return out


def apply_reassurance(steps: List[dict], lint_config: Dict[str, Any]) -> List[dict]:
    # Keep this minimal; the UI already carries trust cues.
    return steps


def lint_steps(steps: List[dict], lint_config: Dict[str, Any]) -> Tuple[bool, List[dict], List[str]]:
    violations: List[dict] = []
    bad_ids: List[str] = []

    banned_substrings = lint_config.get("banned_question_substrings") or []
    if not isinstance(banned_substrings, list):
        banned_substrings = []
    max_chars = lint_config.get("max_question_chars")
    try:
        max_chars_i = int(max_chars)
    except Exception:
        max_chars_i = 140
    require_qmark = bool(lint_config.get("require_question_mark") is True)

    for step in steps or []:
        if not isinstance(step, dict):
            continue
        sid = str(step.get("id") or "").strip()
        q = str(step.get("question") or "").strip()
        if not sid:
            violations.append({"code": "missing_id", "message": "Step is missing id"})
            continue
        if not q:
            violations.append({"code": "missing_question", "message": f"{sid}: missing question"})
            bad_ids.append(sid)
            continue
        if require_qmark and not q.endswith("?"):
            violations.append({"code": "question_no_qmark", "message": f"{sid}: question should end with '?'"})
        if len(q) > max_chars_i:
            violations.append({"code": "question_too_long", "message": f"{sid}: question too long ({len(q)} chars)"})
        q_lower = q.lower()
        for sub in banned_substrings:
            t = str(sub or "").strip().lower()
            if t and t in q_lower:
                violations.append({"code": "banned_phrase", "message": f"{sid}: contains banned phrase '{sub}'"})

    ok = len(violations) == 0
    return ok, violations, bad_ids


__all__ = ["sanitize_steps", "apply_reassurance", "lint_steps"]

