"""
Renderer helper: post-process emitted UI steps copy.

Used by the form pipeline orchestrator after validation to:
- strip meta/instruction-y prefixes from questions
- remove redundant parenthetical enumerations
- enforce trailing question marks where appropriate
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _strip_parenthetical_enumeration(q: str) -> str:
    # Remove trailing "(a, b, c)" style enumerations which duplicate the options list.
    return re.sub(r"\s*\([^)]{0,80}\)\s*$", "", q).strip()


def _strip_meta_instruction_prefix(q: str) -> str:
    """
    Models sometimes echo planner intents like "Ask user ..." into the user-facing question.
    Strip obvious instruction-y prefixes so the UI copy reads naturally.
    """
    t = str(q or "").strip()
    if not t:
        return t
    t = re.sub(r"^(ask\s+(the\s+)?user\s+)", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^(ask\s+user\s+)", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^(ask\s+about\s+)", "", t, flags=re.IGNORECASE).strip()
    # "whether they ..." -> "Do you ..."
    t = re.sub(r"^whether\s+(you|they)\s+", "Do you ", t, flags=re.IGNORECASE).strip()
    return t


def _step_priority_score(step: dict) -> int:
    """
    Lower score == earlier in the list.

    Heuristic ordering to match expected UX:
    - prefer design/style/features early
    - push logistics + measurements later
    """
    sid = str(step.get("id") or "").strip().lower()
    q = str(step.get("question") or "").strip().lower()
    text = f"{sid} {q}"
    tokens = set([t for t in re.split(r"[^a-z0-9]+", text) if t])

    # Strong signals: these should usually come first.
    if "style" in tokens or "design" in tokens or "aesthetic" in tokens:
        return 0
    if "feature" in tokens or "features" in tokens:
        return 1

    # Still "creative" / preference-setting.
    if tokens.intersection(
        {
            "color",
            "palette",
            "material",
            "materials",
            "finish",
            "finishes",
            "look",
            "theme",
            "vibe",
            "lighting",
            "layout",
            "shape",
            "location",
        }
    ):
        return 10

    # Moderate: functional intent.
    if "usage" in tokens or "use" in tokens:
        return 25

    # Logistics later.
    if tokens.intersection({"budget", "price", "pricing", "cost", "permit", "permits", "photo", "photos", "upload", "file"}):
        return 70

    # Measurements later.
    if tokens.intersection({"size", "sqft", "square", "footage", "depth", "dimensions", "length", "width", "ft", "feet"}):
        return 80

    return 40


def _order_steps_by_plan(steps: List[dict], *, plan_step_ids: Optional[List[str]]) -> List[dict]:
    """
    Renderer responsibility: enforce output ordering.

    Ordering is determined by:
    - heuristic UX priority score (design/style/features first)
    - then plan order (if provided)
    - then original order (stable)
    """
    order: Dict[str, int] = {}
    if plan_step_ids:
        for i, sid in enumerate(plan_step_ids):
            t = str(sid or "").strip()
            if t and t not in order:
                order[t] = i

    indexed = list(enumerate(steps or []))
    indexed.sort(
        key=lambda pair: (
            _step_priority_score(pair[1] or {}),
            order.get(str((pair[1] or {}).get("id") or "").strip(), 10**9),
            pair[0],
        )
    )
    return [s for _, s in indexed]


def sanitize_steps(steps: List[dict], lint_config: Dict[str, Any], *, plan_step_ids: Optional[List[str]] = None) -> List[dict]:
    """
    Final copy sanitation for emitted UI steps.
    """
    steps = _order_steps_by_plan(list(steps or []), plan_step_ids=plan_step_ids)
    out: List[dict] = []
    require_qmark = bool(lint_config.get("require_question_mark") is True)
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        s = dict(step)
        step_type = str(s.get("type") or "").strip().lower()
        q = str(s.get("question") or "").strip()
        if q:
            q = _strip_meta_instruction_prefix(q)
            q = _strip_parenthetical_enumeration(q)
            # Only enforce question marks on actual question-like steps.
            # Confirmation/intro/designer/etc. read better as statements.
            enforce_qmark = require_qmark and step_type not in {
                "confirmation",
                "intro",
                "designer",
                "pricing",
                "gallery",
                "file_upload",
                "upload",
                "file_picker",
            }
            if enforce_qmark and not q.endswith("?"):
                q = q.rstrip(".").strip()
                if q and not q.endswith("?"):
                    q = f"{q}?"
            s["question"] = q
        out.append(s)
    return out


__all__ = ["sanitize_steps"]
