from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class PlanQualityResult:
    """
    Result of `score_question_plan`.

    `score` is a 0..100 weighted aggregate of `breakdown`.
    `breakdown` keys are stable, and each is also 0..100:
      - question_progression_psychology: Early-step guardrails / progressive disclosure
        (avoid invasive + high-friction asks up front).
      - sequencing: Early-to-late progression (early = shorter/simpler; later can be more detailed).
      - service_alignment: Per-question alignment to `services_summary` (plus credit for core intake topics).
      - goal_adherence: Plan contains at least one question aligned to goal intent ("pricing" vs "visual").
      - intake_breadth: Plan covers multiple distinct goal-aligned topics (configurable via topic lexicon).
      - novelty: Non-redundancy proxy (penalizes near-duplicate questions by string similarity).
      - engagement: Low-friction wording proxy (penalizes very long / essay-style prompts).
      - min_step_schema_adherence: Plan items include the minimum required fields to deterministically
        render into schema-valid UI miniSteps (honors allowed type policy + option_hints requirements).
      - ui_option_breadth: Option "richness" for choice-family steps, based on how many `option_hints`
        the planner provides relative to configured min/max/target bounds.

    Topic-based scoring (`goal_adherence`, `intake_breadth`, and the "core intake" boost inside
    `service_alignment`) uses a caller-provided lexicon so topic tokens are not hardcoded here.
    Provide it inside `planner_context_json` as `topicLexicon` (or `topic_lexicon`) with shape:
      { "<topic_name>": ["token1", "token2", ...], ... }
    """

    score: float
    breakdown: Dict[str, float]
    notes: List[str]


def _safe_json_loads(text: Any) -> Any:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    # Best-effort: allow the model to wrap JSON with prose (avoid hard failure).
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _extract_plan_items(question_plan_json: Any) -> List[Dict[str, Any]]:
    parsed = _safe_json_loads(question_plan_json)
    items: list[Any] = []
    if isinstance(parsed, dict):
        raw = parsed.get("plan")
        items = raw if isinstance(raw, list) else []
    elif isinstance(parsed, list):
        items = parsed
    out: List[Dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            out.append(it)
    return out


def _tokens(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(str(text or ""))]


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _cap01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / float(len(xs) or 1)


def _get_example_field(example: Any, key: str) -> Any:
    if isinstance(example, dict):
        return example.get(key)
    return getattr(example, key, None)


def _extract_services_text(planner_context_json: Any) -> Tuple[str, str]:
    """
    Returns (services_summary, goal_intent).

    We only rely on fields the orchestrator actually emits into planner_context_json.
    """
    parsed = _safe_json_loads(planner_context_json)
    if not isinstance(parsed, dict):
        return ("", "")
    services = str(parsed.get("services_summary") or parsed.get("grounding_summary") or "").strip()
    # goal_intent is not currently forwarded into planner_context_json by orchestrator;
    # keep it future-proof anyway.
    goal = str(parsed.get("goal_intent") or parsed.get("goalIntent") or "").strip().lower()
    return (services, goal)


def _extract_topic_lexicon(planner_context_json: Any) -> Dict[str, set[str]]:
    """
    Optional topic lexicon for topic detection. Kept external to the metric so you can
    tune it without code changes.
    """
    parsed = _safe_json_loads(planner_context_json)
    if not isinstance(parsed, dict):
        return {}
    raw = parsed.get("topicLexicon") or parsed.get("topic_lexicon") or parsed.get("qualityTopicLexicon")
    if not isinstance(raw, dict):
        return {}

    out: Dict[str, set[str]] = {}
    for k, v in raw.items():
        topic = str(k or "").strip()
        if not topic:
            continue
        toks: set[str] = set()
        if isinstance(v, list):
            for t in v:
                s = str(t or "").strip().lower()
                if s:
                    toks.add(s)
        elif isinstance(v, str):
            for part in re.split(r"[\s,]+", v.strip()):
                p = str(part or "").strip().lower()
                if p:
                    toks.add(p)
        if toks:
            out[topic] = toks
    return out


def _extract_allowed_mini_types_hint(planner_context_json: Any) -> List[str]:
    parsed = _safe_json_loads(planner_context_json)
    if not isinstance(parsed, dict):
        return []
    raw = (
        parsed.get("allowed_mini_types_hint")
        or parsed.get("allowedMiniTypesHint")
        or parsed.get("allowed_mini_types")
        or parsed.get("allowedMiniTypes")
    )
    if isinstance(raw, list):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [s.strip().lower() for s in raw.split(",") if s.strip()]
    return []


def _extract_choice_option_bounds(planner_context_json: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Pull choice option count guidance from planner context, if present.
    """
    parsed = _safe_json_loads(planner_context_json)
    if not isinstance(parsed, dict):
        return (None, None, None)

    def _as_int(x: Any) -> Optional[int]:
        try:
            return int(x)
        except Exception:
            return None

    opt_min = _as_int(parsed.get("choice_option_min") or parsed.get("choiceOptionMin"))
    opt_max = _as_int(parsed.get("choice_option_max") or parsed.get("choiceOptionMax"))
    opt_target = _as_int(parsed.get("choice_option_target") or parsed.get("choiceOptionTarget"))
    return (opt_min, opt_max, opt_target)


# --- Heuristic detectors (cheap + objective-ish) ---------------------------------

_INVASIVE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bssn\b|\bsocial security\b", re.IGNORECASE),
    re.compile(r"\bdriver'?s license\b|\bpassport\b", re.IGNORECASE),
    re.compile(r"\bdate of birth\b|\bdob\b", re.IGNORECASE),
    re.compile(r"\bcredit card\b|\bcard number\b|\bcvv\b|\bbank\b|\brouting\b", re.IGNORECASE),
]

# Mildly invasive / higher-friction (often OK later, but avoid in first 1-2 steps)
_HIGH_FRICTION_EARLY_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\baddress\b|\bstreet\b|\bzip\b|\bpostal\b", re.IGNORECASE),
    re.compile(r"\bphone\b|\bemail\b|\bcontact\b", re.IGNORECASE),
    re.compile(r"\bupload\b|\battach\b|\bphoto\b|\bimage\b|\bdocument\b", re.IGNORECASE),
    re.compile(r"\bmeasure(?:ment)?s?\b|\bdimensions?\b|\bexact\b", re.IGNORECASE),
]

def _topic_hits(text: str, lexicon: Dict[str, set[str]]) -> set[str]:
    if not lexicon:
        return set()
    ts = _token_set(text)
    hits: set[str] = set()
    for topic, kws in lexicon.items():
        if ts.intersection(kws):
            hits.add(topic)
    return hits


def _string_sim(a: str, b: str) -> float:
    return float(SequenceMatcher(a=a.strip().lower(), b=b.strip().lower()).ratio())


# --- Scoring ---------------------------------------------------------------------

def score_question_plan(
    *,
    planner_context_json: str,
    question_plan_json: str,
    early_steps: int = 2,
) -> PlanQualityResult:
    """
    Cheap, heuristic-only scoring for Question Planner output.

    Returns a 0..100 score with component breakdown (each component is also 0..100).
    """
    items = _extract_plan_items(question_plan_json)
    questions = [str((it or {}).get("question") or "").strip() for it in items if isinstance(it, dict)]
    questions = [q for q in questions if q]

    services_text, goal_intent = _extract_services_text(planner_context_json)
    services_tokens = _token_set(services_text)
    goal_intent = str(goal_intent or "").strip().lower()
    if goal_intent not in {"pricing", "visual"}:
        goal_intent = "pricing"
    topic_lexicon = _extract_topic_lexicon(planner_context_json)
    has_topics = bool(topic_lexicon)

    notes: List[str] = []
    if not questions:
        return PlanQualityResult(
            score=0.0,
            breakdown={
                "question_progression_psychology": 0.0,
                "sequencing": 0.0,
                "service_alignment": 0.0,
                "goal_adherence": 0.0,
                "intake_breadth": 0.0,
                "novelty": 0.0,
                "engagement": 0.0,
                "min_step_schema_adherence": 0.0,
                "ui_option_breadth": 0.0,
            },
            notes=["empty plan/questions"],
        )

    # 0) UI option breadth: do choice-family steps have enough selectable options?
    #
    # This is intentionally separate from schema adherence: you can be schema-valid with
    # a single option, but still have a "thin" UI surface area.
    def _ui_option_breadth() -> float:
        opt_min, opt_max, opt_target = _extract_choice_option_bounds(planner_context_json)

        def _as_int(x: Any) -> Optional[int]:
            try:
                return int(x)
            except Exception:
                return None

        # Keep aligned with renderer bounds (defaults + caps).
        min_req = max(2, _as_int(opt_min) if _as_int(opt_min) and _as_int(opt_min) > 0 else 3)
        max_req = max(min_req, _as_int(opt_max) if _as_int(opt_max) and _as_int(opt_max) > 0 else 12)
        # Target is the "good enough" breadth point.
        target = _as_int(opt_target)
        if target is None or target <= 0:
            target = 6
        target = max(min_req, min(max_req, int(target)))

        allowed_types_hint = _extract_allowed_mini_types_hint(planner_context_json)
        if not allowed_types_hint:
            allowed_types_hint = ["multiple_choice"]
        default_type = next((t for t in allowed_types_hint if t), "multiple_choice")

        choice_family = {
            "multiple_choice",
            "choice",
            "segmented_choice",
            "chips_multi",
            "yes_no",
            "image_choice_grid",
            "searchable_select",
        }

        def _normalize_option_hints(raw: Any) -> List[Any]:
            if not isinstance(raw, list):
                return []
            out: List[Any] = []
            for opt in raw:
                if isinstance(opt, str):
                    if opt.strip():
                        out.append(opt)
                elif isinstance(opt, dict):
                    label = opt.get("label")
                    if label is None:
                        label = opt.get("value")
                    if str(label or "").strip():
                        out.append(opt)
            return out

        per_step_scores: List[float] = []
        missing_option_hints = 0

        for it in items:
            if not isinstance(it, dict):
                continue

            type_hint = str(it.get("type_hint") or it.get("typeHint") or "").strip().lower()
            intended_type = type_hint or str(default_type or "").strip().lower()
            if intended_type not in choice_family:
                continue

            option_hints = it.get("option_hints")
            if option_hints is None:
                option_hints = it.get("optionHints")
            if option_hints is None:
                option_hints = it.get("answer_hints")

            normalized_hints = _normalize_option_hints(option_hints)
            hint_count = len(normalized_hints)
            if hint_count <= 0:
                missing_option_hints += 1
                per_step_scores.append(0.0)
                continue

            # Score rises quickly up to `target`, then flattens; penalize very large lists.
            ratio = float(hint_count) / float(target or 1)
            score = _cap01(ratio) ** 0.5  # sqrt to reward early gains
            if hint_count > max_req:
                score *= _cap01(float(max_req) / float(hint_count))

            per_step_scores.append(_cap01(score))

        if missing_option_hints:
            notes.append("ui_option_breadth: some choice steps have no option_hints")

        return _cap01(_mean(per_step_scores)) if per_step_scores else 0.0

    ui_option_breadth = _ui_option_breadth()

    # 0) Min-step schema adherence: can the plan be deterministically rendered into
    # schema-valid `miniSteps[]` given the allowed type policy?
    def _min_step_schema_adherence() -> float:
        allowed_types_hint = _extract_allowed_mini_types_hint(planner_context_json)
        if not allowed_types_hint:
            allowed_types_hint = ["multiple_choice"]
        allowed_set = set([t for t in allowed_types_hint if t])
        if not allowed_set:
            allowed_types_hint = ["multiple_choice"]
            allowed_set = {"multiple_choice"}
        opt_min, opt_max, opt_target = _extract_choice_option_bounds(planner_context_json)

        try:
            from programs.form_pipeline.allowed_types import allowed_type_matches
            from programs.question_planner.renderer.plan_to_steps import render_plan_items_to_mini_steps
            from programs.question_planner.renderer.validation import _reject_banned_option_sets, _validate_mini
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
                SliderUI,
                SearchableSelectUI,
                TextInputUI,
            )

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
                "SliderUI": SliderUI,
                "SearchableSelectUI": SearchableSelectUI,
                "TextInputUI": TextInputUI,
            }
        except Exception:
            notes.append("schema adherence check unavailable (imports failed)")
            return 0.0

        def _normalize_option_hints(raw: Any) -> List[Any]:
            if not isinstance(raw, list):
                return []
            out: List[Any] = []
            for opt in raw:
                if isinstance(opt, str):
                    if opt.strip():
                        out.append(opt)
                elif isinstance(opt, dict):
                    label = opt.get("label")
                    if label is None:
                        label = opt.get("value")
                    if str(label or "").strip():
                        out.append(opt)
            return out

        choice_family = {
            "multiple_choice",
            "choice",
            "segmented_choice",
            "chips_multi",
            "yes_no",
            "image_choice_grid",
            "searchable_select",
        }

        per_item_scores: List[float] = []
        missing_option_hints = 0
        invalid_type_hints = 0
        unrenderable = 0

        for it in items:
            if not isinstance(it, dict):
                continue

            item_score = 1.0

            key = str(it.get("key") or "").strip()
            question = str(it.get("question") or "").strip()
            if not key or not question:
                item_score = 0.0
                unrenderable += 1
                per_item_scores.append(item_score)
                continue

            type_hint = str(it.get("type_hint") or it.get("typeHint") or "").strip().lower()
            default_type = next((t for t in allowed_types_hint if t), "multiple_choice")
            intended_type = type_hint or default_type
            if type_hint and not allowed_type_matches(type_hint, allowed_set):
                invalid_type_hints += 1
                item_score *= 0.5

            option_hints = it.get("option_hints")
            if option_hints is None:
                option_hints = it.get("optionHints")
            if option_hints is None:
                option_hints = it.get("answer_hints")
            normalized_hints = _normalize_option_hints(option_hints)
            hint_count = len(normalized_hints)
            if intended_type in choice_family:
                if hint_count <= 0:
                    missing_option_hints += 1
                    item_score *= 0.25
                else:
                    # Keep this aligned with runtime guidance (soft); do not hard-fail here.
                    # If bounds are missing, default to the eval heuristic range.
                    min_req = max(2, int(opt_min) if isinstance(opt_min, int) and opt_min > 0 else 3)
                    max_req = max(min_req, int(opt_max) if isinstance(opt_max, int) and opt_max > 0 else 12)
                    if hint_count < min_req:
                        item_score *= _cap01(hint_count / float(min_req))
                    elif hint_count > max_req:
                        item_score *= _cap01(float(max_req) / float(hint_count))

            rendered = render_plan_items_to_mini_steps(
                [it],
                choice_option_min=opt_min,
                choice_option_max=opt_max,
                choice_option_target=opt_target,
            )
            if not rendered:
                item_score = 0.0
                unrenderable += 1
                per_item_scores.append(item_score)
                continue

            step = rendered[0] if isinstance(rendered[0], dict) else None
            if not isinstance(step, dict):
                item_score = 0.0
                unrenderable += 1
                per_item_scores.append(item_score)
                continue

            if allowed_set and not allowed_type_matches(str(step.get("type") or ""), allowed_set):
                item_score *= 0.0
                invalid_type_hints += 1

            validated = _validate_mini(step, ui_types)
            if not validated:
                item_score *= 0.0
            else:
                validated = _reject_banned_option_sets(validated)
                if not validated:
                    item_score *= 0.0

            per_item_scores.append(_cap01(item_score))

        if missing_option_hints:
            notes.append("missing option_hints for choice steps")
        if invalid_type_hints:
            notes.append("plan includes invalid type_hint for allowed types")
        if unrenderable:
            notes.append("some plan items are not renderable into valid miniSteps")

        return _cap01(_mean(per_item_scores)) if per_item_scores else 0.0

    min_step_schema_adherence = _min_step_schema_adherence()

    # 1) Question progression psychology: safety + friction checks on early steps
    #    (first `early_steps` questions).
    early_qs = questions[: max(1, int(early_steps))]
    invasive_hits = 0
    friction_hits = 0
    for q in early_qs:
        if any(p.search(q) for p in _INVASIVE_PATTERNS):
            invasive_hits += 1
        if any(p.search(q) for p in _HIGH_FRICTION_EARLY_PATTERNS):
            friction_hits += 1
    question_progression_psychology = 1.0
    if invasive_hits:
        question_progression_psychology -= 0.75 * min(1.0, invasive_hits / float(len(early_qs) or 1))
        notes.append("invasive question detected early")
    if friction_hits:
        question_progression_psychology -= 0.35 * min(1.0, friction_hits / float(len(early_qs) or 1))
        notes.append("high-friction question detected early")
    question_progression_psychology = _cap01(question_progression_psychology)

    # 2) Sequencing: reward early questions being shorter/simpler than late questions.
    #    Also penalize multi-part questions ("and/or") in the early steps.
    lengths = [len(_tokens(q)) for q in questions]
    first = lengths[: max(1, min(len(lengths), int(early_steps)))]
    last = lengths[-max(1, min(len(lengths), int(early_steps))) :]
    # If early is longer than late, that's a sequencing smell.
    sequencing = 1.0
    if _mean(first) > _mean(last) + 2.0:
        sequencing -= 0.35
        notes.append("early questions longer than later ones")
    # Penalize multi-part "and/or" early.
    multipart_early = sum(1 for q in early_qs if re.search(r"\b(and|or)\b", q, re.IGNORECASE))
    if multipart_early:
        sequencing -= 0.15 * min(1.0, multipart_early / float(len(early_qs) or 1))
    sequencing = _cap01(sequencing)

    # 3) Service alignment: per-question alignment to the provided services/context summary.
    #    We also grant some credit for "core intake" topics even if the summary is short/noisy.
    def alignment_for(q: str) -> float:
        q_tokens = _token_set(q)
        overlap = len(q_tokens.intersection(services_tokens))
        denom = max(6, len(q_tokens))
        j = overlap / float(denom)
        generic_ok = 1.0 if (_topic_hits(q, topic_lexicon) if has_topics else set()) else 0.0
        # Allow generic core-intake questions even when the summary is short.
        return _cap01(0.7 * j + 0.3 * generic_ok)

    service_alignment = _cap01(_mean([alignment_for(q) for q in questions]))

    # 4) Goal adherence: does the plan include at least one goal-aligned topic?
    all_topics: set[str] = set()
    if has_topics:
        for q in questions:
            all_topics |= _topic_hits(q, topic_lexicon)
    goal_topics = {"style", "color", "material", "finish", "lighting", "constraints"} if goal_intent == "visual" else {"budget", "timeline", "scope", "constraints"}
    goal_adherence = 1.0 if (all_topics.intersection(goal_topics) if has_topics else set()) else 0.0
    if has_topics and goal_adherence < 1.0:
        notes.append("missing obvious goal-aligned topic")
    if not has_topics:
        notes.append("topicLexicon missing; skipping goal/intake topic checks")

    # 5) Intake breadth: reward covering multiple distinct goal-aligned topics.
    # Normalize by a small denominator so plans don't need to hit "everything" to score well.
    intake_breadth = (
        _cap01(len(all_topics.intersection(goal_topics)) / 3.0) if has_topics else 0.0
    )  # full credit at 3+ topics

    # 6) Novelty: a non-redundancy proxy. Penalize near-duplicate questions.
    sims: List[float] = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            sims.append(_string_sim(questions[i], questions[j]))
    max_sim = max(sims) if sims else 0.0
    novelty = _cap01(1.0 - max(0.0, max_sim - 0.75) / 0.25)  # 1 until ~0.75 similarity, then declines
    if max_sim >= 0.9:
        notes.append("questions appear redundant")

    # 7) Engagement: likely-to-answer proxy based on length + "essay prompt" phrasing.
    def engagement_for(q: str) -> float:
        t = q.strip()
        w = len(_tokens(t))
        score = 1.0
        if w > 18:
            score -= 0.25
        if w > 28:
            score -= 0.25
        if re.search(r"\bplease describe\b|\btell us about\b|\bwrite\b|\bexplain\b", t, re.IGNORECASE):
            score -= 0.2
        if re.search(r"\bincluding\b|\bsuch as\b", t, re.IGNORECASE) and w > 18:
            score -= 0.1
        return _cap01(score)

    engagement = _cap01(_mean([engagement_for(q) for q in questions]))

    breakdown01 = {
        "question_progression_psychology": question_progression_psychology,
        "sequencing": sequencing,
        "service_alignment": service_alignment,
        "goal_adherence": goal_adherence,
        "intake_breadth": intake_breadth,
        "novelty": novelty,
        "engagement": engagement,
        "min_step_schema_adherence": min_step_schema_adherence,
        "ui_option_breadth": ui_option_breadth,
    }

    # Weighted score out of 100 (bias toward goal + service fit).
    # Each value is "points available" for that component; total must be 100.
    weights_points = {
        "min_step_schema_adherence": 15,
        "goal_adherence": 25,
        "service_alignment": 25,
        "question_progression_psychology": 15,
        "intake_breadth": 5,
        "sequencing": 5,
        "engagement": 5,
        "novelty": 5,
    }

    scored_keys = list(weights_points.keys())
    if not has_topics:
        # Do not penalize score when topic lexicon is missing; renormalize remaining components back to 100.
        scored_keys = [k for k in scored_keys if k not in {"goal_adherence", "intake_breadth"}]
    denom_points = float(sum(weights_points[k] for k in scored_keys)) or 1.0
    raw_points = sum(float(breakdown01[k]) * float(weights_points[k]) for k in scored_keys)
    score = (raw_points / denom_points) * 100.0
    score = max(0.0, min(100.0, float(score)))

    breakdown = {k: round(v * 100.0, 4) for k, v in breakdown01.items()}
    return PlanQualityResult(score=round(score, 4), breakdown=breakdown, notes=notes[:8])


def question_planner_quality_metric(example: Any, pred: Any, trace: Any = None) -> float | bool:
    """
    DSPy-compatible metric for the Question Planner.

    Inputs expected:
      - example.planner_context_json (or dict key) : JSON string
      - pred.question_plan_json : JSON string
    """
    planner_context_json = str(_get_example_field(example, "planner_context_json") or "")
    question_plan_json = str(getattr(pred, "question_plan_json", None) or "")

    result = score_question_plan(planner_context_json=planner_context_json, question_plan_json=question_plan_json)
    if trace is not None:
        # Strict-ish gate for compiling/bootstrapping.
        if not _extract_topic_lexicon(planner_context_json):
            return False
        return bool(
            result.score >= 70.0
            and float(result.breakdown.get("question_progression_psychology", 0.0)) >= 75.0
            and float(result.breakdown.get("goal_adherence", 0.0)) >= 50.0
            and float(result.breakdown.get("min_step_schema_adherence", 0.0)) >= 85.0
        )
    return float(result.score)


__all__ = [
    "PlanQualityResult",
    "score_question_plan",
    "question_planner_quality_metric",
]
