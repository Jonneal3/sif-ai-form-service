from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from programs.question_planner.plan_parsing import derive_step_id_from_key, normalize_plan_key
from programs.question_planner.renderer.validation import _coerce_options


_SPECIAL_OPTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bnot\s+sure\b", re.IGNORECASE),
    re.compile(r"\bno\s+preference\b", re.IGNORECASE),
    re.compile(r"\bno\s+strong\s+preference\b", re.IGNORECASE),
    re.compile(r"\bother\b", re.IGNORECASE),
)

_MULTI_SELECT_STRONG_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(select|choose|pick|check)\s+all\s+that\s+apply\b", re.IGNORECASE),
    re.compile(r"\b(select|choose|pick|check)\s+any\s+(that\s+apply)?\b", re.IGNORECASE),
    re.compile(r"\b(check|select)\s+all\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+of\s+(these|the\s+following)\b", re.IGNORECASE),
)
_MULTI_SELECT_UP_TO_PATTERN = re.compile(r"\b(?:up\s+to|at\s+most|no\s+more\s+than)\s+(?P<n>\d+)\b", re.IGNORECASE)

# Conservative noun/verb hints for inferring multi-select when the planner forgets to set allow_multiple.
_MULTI_SELECT_NOUN_HINTS: tuple[str, ...] = (
    "materials",
    "features",
    "elements",
    "plant types",
    "plants",
    "seating options",
    "amenities",
    "add-ons",
    "addons",
    "extras",
)
_MULTI_SELECT_VERB_HINTS: tuple[str, ...] = (
    "include",
    "add",
    "highlight",
    "focus",
    "avoid",
    "favor",
    "would you like",
    "do you want",
    "are you interested",
)


def _is_special_option_label(label: str) -> bool:
    t = str(label or "").strip()
    if not t:
        return False
    return any(p.search(t) for p in _SPECIAL_OPTION_PATTERNS)


def _option_label(opt: Any) -> str:
    if isinstance(opt, str):
        return opt
    if isinstance(opt, dict):
        label = opt.get("label")
        if label is None:
            label = opt.get("value")
        return str(label or "")
    return ""


def _normalize_option_hints(option_hints: Any) -> list:
    if not isinstance(option_hints, list):
        return []
    out: list = []
    for opt in option_hints:
        if isinstance(opt, (str, dict)):
            if str(_option_label(opt) or "").strip():
                out.append(opt)
    return out


def _enforce_option_count(
    option_hints: list,
    *,
    choice_option_min: Optional[int],
    choice_option_max: Optional[int],
    choice_option_target: Optional[int],
) -> list:
    """
    Keep option counts within the UI's min/max/target guidance.

    We prefer to preserve any "special" options like "Not sure yet" / "Other" when trimming.
    When we must pad, we add conservative generic options rather than hallucinating.
    """
    opts = list(option_hints or [])

    def _as_int(x: Any) -> Optional[int]:
        try:
            return int(x)
        except Exception:
            return None

    opt_min = _as_int(choice_option_min)
    opt_max = _as_int(choice_option_max)
    opt_target = _as_int(choice_option_target)

    if opt_min is not None:
        opt_min = max(1, min(12, opt_min))
    if opt_max is not None:
        opt_max = max(1, min(12, opt_max))
    if opt_min is not None and opt_max is not None and opt_max < opt_min:
        opt_max = opt_min
    if opt_target is not None and opt_min is not None and opt_target < opt_min:
        opt_target = opt_min
    if opt_target is not None and opt_max is not None and opt_target > opt_max:
        opt_target = opt_max

    # Trim to max first.
    if opt_max is not None and len(opts) > opt_max:
        special = [o for o in opts if _is_special_option_label(_option_label(o))]
        core = [o for o in opts if o not in special]
        keep: list = []
        # Keep as much core as we can, reserving space for specials.
        reserve = min(len(special), opt_max)
        core_limit = max(0, opt_max - reserve)
        keep.extend(core[:core_limit])
        keep.extend(special[: max(0, opt_max - len(keep))])
        opts = keep[:opt_max]

    # Best-effort trim toward target (do not pad to reach target).
    if opt_target is not None and len(opts) > opt_target:
        special = [o for o in opts if _is_special_option_label(_option_label(o))]
        core = [o for o in opts if o not in special]
        keep: list = []
        reserve = min(len(special), opt_target)
        core_limit = max(0, opt_target - reserve)
        keep.extend(core[:core_limit])
        keep.extend(special[: max(0, opt_target - len(keep))])
        opts = keep[:opt_target]

    # Pad to min with generic options (only if needed).
    if opt_min is not None and len(opts) < opt_min:
        existing_norm = {re.sub(r"\s+", " ", _option_label(o).strip().lower()) for o in opts}
        for label in ("Not sure yet", "Other", "No strong preference"):
            if len(opts) >= opt_min:
                break
            norm = re.sub(r"\s+", " ", label.strip().lower())
            if norm in existing_norm:
                continue
            opts.append(label)
            existing_norm.add(norm)

    return opts


def _infer_multi_select(*, key: str, question: str) -> tuple[bool, Optional[int]]:
    """
    Best-effort inference for when a step should allow multiple selections.

    Intentionally conservative:
    - Only triggers on strong phrasing ("select all that apply", "which of these...")
      or on noun+verb cues that strongly imply a list of inclusions (materials/features/etc.).
    """
    q = str(question or "").strip().lower()
    if not q:
        return False, None

    if any(p.search(q) for p in _MULTI_SELECT_STRONG_PATTERNS):
        m = _MULTI_SELECT_UP_TO_PATTERN.search(q)
        if m:
            try:
                n = int(m.group("n"))
                if n > 0:
                    return True, max(1, min(10, n))
            except Exception:
                pass
        return True, None

    if any(n in q for n in _MULTI_SELECT_NOUN_HINTS) and any(v in q for v in _MULTI_SELECT_VERB_HINTS):
        m = _MULTI_SELECT_UP_TO_PATTERN.search(q)
        if m:
            try:
                n = int(m.group("n"))
                if n > 0:
                    return True, max(1, min(10, n))
            except Exception:
                pass
        return True, None

    # Key-based backstop for common plural preference buckets.
    k = str(key or "").strip().lower()
    if re.search(r"(materials|features|amenities|extras|add_ons|addons|must_avoid|avoid)$", k):
        return True, None

    return False, None


def render_plan_items_to_mini_steps(
    plan_items: List[Dict[str, Any]],
    *,
    choice_option_min: Optional[int] = None,
    choice_option_max: Optional[int] = None,
    choice_option_target: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert planner plan items into raw UI step dicts.

    Output is intentionally "model-free": the orchestrator still validates/coerces
    into schema objects via `_validate_mini`.
    """
    steps: List[Dict[str, Any]] = []
    for item in plan_items or []:
        if not isinstance(item, dict):
            continue
        key = normalize_plan_key(item.get("key"))
        if not key:
            continue
        step_id = derive_step_id_from_key(key)

        question = str(item.get("question") or item.get("intent") or "").strip()
        if not question:
            continue

        raw_option_hints = _normalize_option_hints(item.get("option_hints"))
        raw_option_hints = _enforce_option_count(
            raw_option_hints,
            choice_option_min=choice_option_min,
            choice_option_max=choice_option_max,
            choice_option_target=choice_option_target,
        )
        options = _coerce_options(raw_option_hints)
        if not options:
            # Hard backstop: choice steps require options for schema validation.
            options = _coerce_options(["Not sure yet"])

        step: Dict[str, Any] = {
            "id": step_id,
            "type": "multiple_choice",
            "question": question,
            "options": options,
        }

        allow_multiple = item.get("allow_multiple")
        if allow_multiple is None:
            allow_multiple = item.get("allowMultiple")
        if allow_multiple is None:
            allow_multiple = item.get("multi_select")
        if allow_multiple is None:
            allow_multiple = item.get("multiSelect")
        if allow_multiple is not None:
            # Frontend contract uses `multi_select` (snake_case). Keep older keys as input-only.
            step["multi_select"] = bool(allow_multiple)
        else:
            inferred_multi, inferred_max = _infer_multi_select(key=key, question=question)
            if inferred_multi:
                step["multi_select"] = True
                if inferred_max is not None:
                    step["max_selections"] = int(inferred_max)

        allow_other = item.get("allow_other")
        if allow_other is None:
            allow_other = item.get("allowOther")
        if allow_other is not None:
            step["allow_other"] = bool(allow_other)

        other_label = item.get("other_label")
        if other_label is None:
            other_label = item.get("otherLabel")
        if str(other_label or "").strip():
            step["other_label"] = str(other_label).strip()

        other_placeholder = item.get("other_placeholder")
        if other_placeholder is None:
            other_placeholder = item.get("otherPlaceholder")
        if str(other_placeholder or "").strip():
            step["other_placeholder"] = str(other_placeholder).strip()

        other_requires_text = item.get("other_requires_text")
        if other_requires_text is None:
            other_requires_text = item.get("otherRequiresText")
        if other_requires_text is not None:
            step["other_requires_text"] = bool(other_requires_text)

        if item.get("required") is True:
            step["required"] = True

        # Pass through optional function calls unchanged.
        if isinstance(item.get("functionCall"), dict):
            step["functionCall"] = item["functionCall"]

        steps.append(step)

    return steps


__all__ = ["render_plan_items_to_mini_steps"]
