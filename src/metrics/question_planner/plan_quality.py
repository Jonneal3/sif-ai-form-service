from __future__ import annotations

"""
Question Planner plan-quality scoring.

IMPORTANT: This module intentionally contains NO regex-based heuristic detector lists
for "visual seeds", "operational questions", "narrowing patterns", etc.

We use an LLM judge (demo-guided + service_summary grounded) to score plans, because:
- it's more flexible than hand-maintained keyword/regex heuristics
- it better matches product intent across industries/services
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_RATE_LIMIT_WAIT_RE = re.compile(r"Please try again in\\s+([0-9.]+)s", re.IGNORECASE)


_USAGE_ACCUMULATOR: Dict[str, Dict[str, float]] = {
    "planner": {
        "calls": 0.0,
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
    },
    "judge": {
        "calls": 0.0,
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
    },
}


def reset_metric_usage() -> None:
    for bucket in _USAGE_ACCUMULATOR.values():
        bucket["calls"] = 0.0
        bucket["prompt_tokens"] = 0.0
        bucket["completion_tokens"] = 0.0
        bucket["total_tokens"] = 0.0


def _usage_totals(usage: Any) -> Dict[str, float]:
    """
    Normalize DSPy/LiteLLM usage payloads into prompt/completion/total token counts.

    `prediction.get_lm_usage()` can be either:
      - {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
      - { "<model>": {"prompt_tokens": ..., ...}, "<model2>": {...}, ... }
    """
    if not isinstance(usage, dict) or not usage:
        return {}

    if any(k in usage for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
        return {
            "prompt_tokens": float(usage.get("prompt_tokens") or 0.0),
            "completion_tokens": float(usage.get("completion_tokens") or 0.0),
            "total_tokens": float(usage.get("total_tokens") or 0.0),
        }

    totals = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0}
    for v in usage.values():
        if not isinstance(v, dict):
            continue
        totals["prompt_tokens"] += float(v.get("prompt_tokens") or 0.0)
        totals["completion_tokens"] += float(v.get("completion_tokens") or 0.0)
        totals["total_tokens"] += float(v.get("total_tokens") or 0.0)
    return totals


def _accumulate_usage(bucket_name: str, usage: Any) -> None:
    bucket = _USAGE_ACCUMULATOR.get(bucket_name)
    if not isinstance(bucket, dict):
        return
    totals = _usage_totals(usage)
    if not totals:
        return
    bucket["calls"] += 1.0
    bucket["prompt_tokens"] += float(totals.get("prompt_tokens") or 0.0)
    bucket["completion_tokens"] += float(totals.get("completion_tokens") or 0.0)
    bucket["total_tokens"] += float(totals.get("total_tokens") or 0.0)


def get_metric_usage_summary() -> Dict[str, Any]:
    planner = _USAGE_ACCUMULATOR.get("planner") or {}
    judge = _USAGE_ACCUMULATOR.get("judge") or {}
    total_calls = float(planner.get("calls") or 0.0) + float(judge.get("calls") or 0.0)
    total_prompt = float(planner.get("prompt_tokens") or 0.0) + float(judge.get("prompt_tokens") or 0.0)
    total_completion = float(planner.get("completion_tokens") or 0.0) + float(judge.get("completion_tokens") or 0.0)
    total_tokens = float(planner.get("total_tokens") or 0.0) + float(judge.get("total_tokens") or 0.0)

    def _as_ints(x: Dict[str, Any]) -> Dict[str, int]:
        return {
            "calls": int(float(x.get("calls") or 0.0)),
            "prompt_tokens": int(float(x.get("prompt_tokens") or 0.0)),
            "completion_tokens": int(float(x.get("completion_tokens") or 0.0)),
            "total_tokens": int(float(x.get("total_tokens") or 0.0)),
        }

    return {
        "total": {
            "calls": int(total_calls),
            "prompt_tokens": int(total_prompt),
            "completion_tokens": int(total_completion),
            "total_tokens": int(total_tokens),
        },
        "planner": _as_ints(planner),
        "judge": _as_ints(judge),
    }


@dataclass(frozen=True)
class PlanQualityResult:
    """
    Result of `score_question_plan`.

    `score` is 0..100. `breakdown` keys are stable and also 0..100:
      - ordering
      - progression
      - personalization
      - service_alignment
      - goal_adherence
      - question_variety
      - redundancy
      - intent_disambiguation
      - question_engagement
      - simplicity
      - question_length
    """

    score: float
    breakdown: Dict[str, float]
    notes: List[str]
    groups: Dict[str, float] = field(default_factory=dict)


BREAKDOWN_KEYS: Tuple[str, ...] = (
    "ordering",
    "progression",
    "personalization",
    "service_alignment",
    "goal_adherence",
    "question_variety",
    "redundancy",
    "intent_disambiguation",
    "question_engagement",
    "simplicity",
    "question_length",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_json_loads(text: Any) -> Any:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of the first complete JSON object from a larger string.

    This mitigates judge outputs that include extra prose/markers or multiple blobs.
    """
    s = str(text or "")
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = s[start : i + 1]
                try:
                    parsed = json.loads(blob)
                except Exception:
                    return None
                return parsed if isinstance(parsed, dict) else None

    return None


def _strict_json_loads(text: Any) -> Any:
    # In this file, "strict" just means "json.loads only".
    return _safe_json_loads(text)


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _cap01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / float(len(xs) or 1)


def _tokens(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(str(text or ""))]


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _get_example_field(example: Any, key: str) -> Any:
    if isinstance(example, dict):
        return example.get(key)
    return getattr(example, key, None)


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


def _extract_services_summary(planner_context_json: Any) -> str:
    parsed = _safe_json_loads(planner_context_json)
    if not isinstance(parsed, dict):
        return ""
    return str(parsed.get("services_summary") or parsed.get("grounding_summary") or "").strip()


def _extract_context_subset(planner_context_json: str) -> Dict[str, Any]:
    """
    Keep the LLM-judge context tight and stable.
    """
    parsed = _safe_json_loads(planner_context_json)
    if not isinstance(parsed, dict):
        return {"services_summary": str(planner_context_json or "").strip()}
    out: Dict[str, Any] = {
        "services_summary": parsed.get("services_summary") or parsed.get("grounding_summary"),
        "service_summary": parsed.get("service_summary") or parsed.get("serviceSummary"),
        "company_summary": parsed.get("company_summary") or parsed.get("companySummary"),
        "answered_qa": parsed.get("answered_qa") or parsed.get("answeredQA"),
        "asked_step_ids": parsed.get("asked_step_ids") or parsed.get("askedStepIds"),
        "batch_constraints": parsed.get("batch_constraints") or parsed.get("batchConstraints"),
        "choice_option_min": parsed.get("choice_option_min") or parsed.get("choiceOptionMin"),
        "choice_option_target": parsed.get("choice_option_target") or parsed.get("choiceOptionTarget"),
        "choice_option_max": parsed.get("choice_option_max") or parsed.get("choiceOptionMax"),
        "allowed_mini_types_hint": parsed.get("allowed_mini_types_hint") or parsed.get("allowedMiniTypesHint"),
        "copy_context": parsed.get("copy_context") or parsed.get("copyContext"),
    }
    # Drop null-ish values for a tighter prompt.
    return {k: v for k, v in out.items() if v is not None and v != "" and v != [] and v != {}}


def _extract_plan_items(question_plan_json: Any) -> List[Dict[str, Any]]:
    parsed = _safe_json_loads(question_plan_json)
    if not isinstance(parsed, dict):
        return []
    raw = parsed.get("plan")
    if not isinstance(raw, list):
        return []
    return [it for it in raw if isinstance(it, dict)]


def _compact_option_labels(option_hints: Any, *, max_items: int = 8) -> List[str]:
    """
    Reduce option payload size for LLM judging while preserving semantic intent.
    """
    if not isinstance(option_hints, list):
        return []
    out: List[str] = []
    for opt in option_hints:
        if len(out) >= int(max_items):
            break
        if isinstance(opt, str):
            label = opt.strip()
        elif isinstance(opt, dict):
            label = str(opt.get("label") or opt.get("value") or "").strip()
        else:
            label = ""
        if label:
            out.append(label)
    return out


def _compact_plan_for_judge(plan_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compress a candidate/reference plan for scoring.

    The judge does not need full renderer hints; sending them repeatedly explodes token usage.
    We keep:
      - key, question
      - type_hint (if present)
      - allow_multiple/allow_other
      - option labels (labels only, capped)
    """
    raw = plan_obj.get("plan")
    if not isinstance(raw, list):
        return {"plan": []}

    compact_items: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        question = str(item.get("question") or "").strip()
        if not key or not question:
            continue

        type_hint = str(item.get("type_hint") or item.get("typeHint") or "").strip()
        allow_multiple = bool(item.get("allow_multiple") is True or item.get("allowMultiple") is True)
        allow_other = bool(item.get("allow_other") is True or item.get("allowOther") is True)

        option_hints = item.get("option_hints")
        if option_hints is None:
            option_hints = item.get("optionHints")
        option_labels = _compact_option_labels(option_hints)

        ci: Dict[str, Any] = {
            "key": key,
            "question": question,
        }
        if type_hint:
            ci["type_hint"] = type_hint
        if allow_multiple:
            ci["allow_multiple"] = True
        if allow_other:
            ci["allow_other"] = True
        if option_labels:
            ci["option_labels"] = option_labels
            ci["option_count"] = len(option_labels)

        compact_items.append(ci)

    return {"plan": compact_items}


def _is_strict_question_plan_schema_valid(*, planner_context_json: Any, question_plan_json: Any) -> bool:
    """
    Hard gate for the optimizer metric:
    - STRICT JSON only
    - requires `plan[]`
    - requires choice-family steps to provide a real `option_hints` list
    """
    parsed = _strict_json_loads(question_plan_json)
    if not isinstance(parsed, dict):
        return False
    raw_plan = parsed.get("plan")
    if not isinstance(raw_plan, list) or not raw_plan:
        return False

    allowed_types_hint = _extract_allowed_mini_types_hint(planner_context_json)
    if not allowed_types_hint:
        allowed_types_hint = ["multiple_choice"]
    allowed_set = set([t for t in allowed_types_hint if t]) or {"multiple_choice"}
    default_type = next((t for t in allowed_types_hint if t), "multiple_choice")

    try:
        from programs.form_pipeline.allowed_types import allowed_type_matches
    except Exception:
        allowed_type_matches = None

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

    for item in raw_plan:
        if not isinstance(item, dict):
            return False
        key = str(item.get("key") or "").strip()
        question = str(item.get("question") or "").strip()
        if not key or not question:
            return False

        type_hint = str(item.get("type_hint") or item.get("typeHint") or "").strip().lower()
        intended_type = type_hint or default_type
        if type_hint and callable(allowed_type_matches) and not allowed_type_matches(type_hint, allowed_set):
            return False

        if intended_type in choice_family:
            option_hints = item.get("option_hints")
            if option_hints is None:
                option_hints = item.get("optionHints")
            if option_hints is None:
                option_hints = item.get("answer_hints")
            if option_hints is None:
                option_hints = item.get("options")
            if len(_normalize_option_hints(option_hints)) < 2:
                return False

    return True


def _validate_question_plan_schema(*, planner_context_json: Any, question_plan_json: Any) -> Tuple[bool, str]:
    """
    Like `_is_strict_question_plan_schema_valid`, but returns a reason for debugging/logging.

    Reasons are intentionally coarse and stable (so logs stay readable).
    """
    parsed = _strict_json_loads(question_plan_json)
    if not isinstance(parsed, dict):
        return False, "invalid_json"
    raw_plan = parsed.get("plan")
    if not isinstance(raw_plan, list) or not raw_plan:
        return False, "missing_plan_array"

    allowed_types_hint = _extract_allowed_mini_types_hint(planner_context_json)
    if not allowed_types_hint:
        allowed_types_hint = ["multiple_choice"]
    allowed_set = set([t for t in allowed_types_hint if t]) or {"multiple_choice"}
    default_type = next((t for t in allowed_types_hint if t), "multiple_choice")

    try:
        from programs.form_pipeline.allowed_types import allowed_type_matches
    except Exception:
        allowed_type_matches = None

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

    for item in raw_plan:
        if not isinstance(item, dict):
            return False, "invalid_plan_item"
        key = str(item.get("key") or "").strip()
        question = str(item.get("question") or "").strip()
        if not key or not question:
            return False, "missing_key_or_question"

        type_hint = str(item.get("type_hint") or item.get("typeHint") or "").strip().lower()
        intended_type = type_hint or default_type
        if type_hint and callable(allowed_type_matches) and not allowed_type_matches(type_hint, allowed_set):
            return False, "disallowed_type"

        if intended_type in choice_family:
            option_hints = item.get("option_hints")
            if option_hints is None:
                option_hints = item.get("optionHints")
            if option_hints is None:
                option_hints = item.get("answer_hints")
            if option_hints is None:
                option_hints = item.get("options")
            if len(_normalize_option_hints(option_hints)) < 2:
                return False, "missing_or_empty_option_hints"

    return True, "ok"


def _default_reference_demos_path() -> Path:
    return _repo_root() / "src" / "programs" / "question_planner" / "data" / "examples" / "demo_examples.json"


def _load_reference_demos() -> List[Dict[str, Any]]:
    """
    Loads curated demo examples that include a `planner_context_json` and `question_plan_json`.

    Format supported:
      - JSON array of flat records: {planner_context_json: {...|str}, question_plan_json: {...|str}, ...}
      - JSON array of {"inputs":{...},"outputs":{...}} records (legacy)
    """
    env_path = os.getenv("QP_METRIC_DEMOS_PATH") or os.getenv("DSPY_PLANNER_METRIC_DEMOS_PATH") or ""
    path = Path(env_path) if env_path.strip() else _default_reference_demos_path()
    if not path.exists():
        return []

    raw = _safe_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []

    out: List[Dict[str, Any]] = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        inputs = rec.get("inputs") if isinstance(rec.get("inputs"), dict) else None
        outputs = rec.get("outputs") if isinstance(rec.get("outputs"), dict) else None
        if inputs is not None:
            ctx = inputs.get("planner_context_json") or inputs.get("context") or ""
            plan = (outputs or {}).get("question_plan_json") if isinstance(outputs, dict) else None
        else:
            ctx = rec.get("planner_context_json") or rec.get("context") or ""
            plan = rec.get("question_plan_json") or rec.get("plan") or ""

        ctx_obj = _safe_json_loads(ctx) if isinstance(ctx, str) else ctx
        plan_obj = _safe_json_loads(plan) if isinstance(plan, str) else plan
        if not isinstance(ctx_obj, dict) or not isinstance(plan_obj, dict):
            continue
        if not isinstance(plan_obj.get("plan"), list):
            continue

        out.append(
            {
                "services_summary": str(ctx_obj.get("services_summary") or ctx_obj.get("grounding_summary") or "").strip(),
                "planner_context_json": ctx_obj,
                "question_plan_json": plan_obj,
            }
        )
    return out


@lru_cache(maxsize=1)
def _cached_reference_demos() -> List[Dict[str, Any]]:
    return _load_reference_demos()


def _select_reference_demos(*, services_summary: str, k: int = 3) -> List[Dict[str, Any]]:
    demos = _cached_reference_demos()
    if not demos:
        return []

    target = _token_set(services_summary)
    if not target:
        return demos[:k]

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for d in demos:
        ss = str(d.get("services_summary") or "").strip()
        sset = _token_set(ss)
        inter = len(target.intersection(sset))
        union = len(target.union(sset)) or 1
        j = inter / float(union)
        scored.append((j, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[: max(0, int(k or 0))]]


def _best_effort_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    root = _repo_root()
    load_dotenv(root / ".env", override=False)
    load_dotenv(root / ".env.local", override=False)


@lru_cache(maxsize=1)
def _make_metric_lm() -> Any:
    """
    Build a dedicated LM for the plan-quality metric judge.

    IMPORTANT: We do NOT configure global DSPy settings here because optimizer runs
    also configure a planner LM globally. Instead, we run the judge under a
    `dspy.settings.context(lm=...)` override.
    """
    try:
        import dspy  # type: ignore
    except Exception:
        raise RuntimeError("DSPy is not installed/available")

    _best_effort_load_dotenv()

    from programs.common.dspy_runtime import make_dspy_lm_for_module
    from programs.common.env import env_float, env_int

    cfg = make_dspy_lm_for_module(module_env_prefix="DSPY_PLANNER_METRIC", allow_small_models=True)
    if not cfg:
        cfg = make_dspy_lm_for_module(module_env_prefix="DSPY_PLANNER", allow_small_models=True)
    if not cfg:
        raise RuntimeError(
            "DSPy metric LM is not configured. Set `DSPY_PROVIDER` + API key (e.g. `OPENAI_API_KEY` or `GROQ_API_KEY`)."
        )

    default_timeout = env_float("DSPY_LLM_TIMEOUT_SEC", 20.0)
    timeout = env_float("DSPY_PLANNER_METRIC_TIMEOUT_SEC", default_timeout)
    max_tokens = env_int("DSPY_PLANNER_METRIC_MAX_TOKENS", 800)

    # Deterministic judge.
    temperature = env_float("DSPY_PLANNER_METRIC_TEMPERATURE", 0.0)
    num_retries = env_int("DSPY_PLANNER_METRIC_NUM_RETRIES", 1)

    return dspy.LM(
        model=cfg["model"],
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        timeout=float(timeout),
        num_retries=max(0, int(num_retries)),
    )


def _maybe_sleep_for_rate_limit(err: Exception) -> bool:
    """
    Best-effort Groq/LiteLLM rate-limit backoff.
    Returns True if we slept and the caller should retry.
    """
    msg = str(err or "")
    m = _RATE_LIMIT_WAIT_RE.search(msg)
    if m:
        try:
            secs = float(m.group(1))
        except Exception:
            secs = 1.0
        time.sleep(max(0.25, min(20.0, secs + 0.25)))
        return True

    lowered = (type(err).__name__ + " " + msg).lower()
    if "rate_limit" in lowered or "ratelimit" in lowered:
        time.sleep(1.0)
        return True
    return False


def _llm_judge_plan_quality(
    *,
    context_json: Dict[str, Any],
    candidate_plan_json: Dict[str, Any],
    reference_demos: List[Dict[str, Any]],
) -> PlanQualityResult:
    """
    Use an LLM judge to score the candidate plan.
    """
    import dspy  # type: ignore

    class _JudgeSignature(dspy.Signature):  # type: ignore
        """
        Grade a question plan for quality.

        Return JSON ONLY with:
        {
          "score": <0-100>,
          "breakdown": {<metric>: <0-100>, ...},
          "notes": [<short strings>]
        }
        """

        planner_context_json: str = dspy.InputField(desc="Compact JSON string of planner context (service + memory + constraints).")
        reference_demos_json: str = dspy.InputField(desc="Compact JSON string of a small set of reference demos.")
        candidate_plan_json: str = dspy.InputField(desc="Candidate question plan JSON string (object with plan[]).")
        grading_json: str = dspy.OutputField(desc="JSON ONLY. No prose, no markdown, no code fences.")

    judge = dspy.Predict(_JudgeSignature)

    rubric = {
        "breakdown_keys": list(BREAKDOWN_KEYS),
        # Keep the rubric compact; it's repeated many times during optimizer runs.
        "global_guidance": [
            "Stay aligned to services_summary/service_summary and the platform goal.",
            "Frontload visual seeds; avoid budget/timeline/scope/contact.",
            "Simple ≠ generic: early questions must be high-signal and tailored to the service.",
            "Avoid broad 'overall style/vibe' openers unless the options materially constrain the design.",
            "Prefer concise, answerable questions; avoid essay prompts.",
            "Avoid redundancy; ensure variety of decision functions.",
            "If service is ambiguous/multi-service, disambiguate early.",
        ],
        "notes": "Return short notes only (top issues).",
    }

    metric_lm = _make_metric_lm()
    desired_track_usage = os.getenv("DSPY_TRACK_USAGE") == "true" or os.getenv("AI_FORM_TOKEN_TELEMETRY") == "true"

    pred: Any = None
    last_err: Optional[Exception] = None
    max_attempts = int(os.getenv("DSPY_PLANNER_METRIC_RATE_LIMIT_RETRIES") or "3")
    for attempt in range(max(1, max_attempts)):
        try:
            with dspy.settings.context(lm=metric_lm, track_usage=bool(desired_track_usage)):
                pred = judge(
                    planner_context_json=_compact_json({"context": context_json, "rubric": rubric}),
                    reference_demos_json=_compact_json(reference_demos),
                    candidate_plan_json=_compact_json(_compact_plan_for_judge(candidate_plan_json)),
                )
            last_err = None
            break
        except Exception as e:
            last_err = e
            if attempt < max_attempts - 1 and _maybe_sleep_for_rate_limit(e):
                continue
            break
    if pred is None:
        raise last_err or RuntimeError("LLM judge failed")

    # Usage tracking (best-effort; depends on provider/adapter).
    try:
        from programs.common.dspy_runtime import extract_dspy_usage

        _accumulate_usage("judge", extract_dspy_usage(pred))
    except Exception:
        pass
    raw = str(getattr(pred, "grading_json", "") or "").strip()
    if os.getenv("QP_METRIC_DEBUG_FAILS") == "true":
        if not raw or raw[0] not in "{[":
            head = raw.replace("\n", " ")[:240]
            print(f"[QPMetric] judge_non_json head={head!r}", flush=True)
    parsed = _safe_json_loads(raw)
    if not isinstance(parsed, dict):
        parsed = _extract_first_json_object(raw) or parsed
    if not isinstance(parsed, dict):
        return PlanQualityResult(
            score=0.0,
            breakdown={k: 0.0 for k in BREAKDOWN_KEYS},
            notes=["LLM judge returned non-JSON output"],
            groups={},
        )

    score = float(parsed.get("score") or 0.0)
    score = max(0.0, min(100.0, score))

    breakdown_in = parsed.get("breakdown") if isinstance(parsed.get("breakdown"), dict) else {}
    breakdown: Dict[str, float] = {}
    for k in BREAKDOWN_KEYS:
        try:
            v = float(breakdown_in.get(k)) if k in breakdown_in else 0.0
        except Exception:
            v = 0.0
        breakdown[k] = max(0.0, min(100.0, v))

    notes = parsed.get("notes") if isinstance(parsed.get("notes"), list) else []
    notes_out = [str(x) for x in notes if str(x or "").strip()][:8]

    groups01: Dict[str, float] = {
        "structure": _cap01(
            _mean([breakdown["ordering"], breakdown["progression"], breakdown["personalization"]]) / 100.0
        ),
        "goal": _cap01(_mean([breakdown["service_alignment"], breakdown["goal_adherence"]]) / 100.0),
        "planning_quality": _cap01(
            _mean([breakdown["question_variety"], breakdown["redundancy"], breakdown["intent_disambiguation"]]) / 100.0
        ),
        "copywriting": _cap01(
            _mean([breakdown["question_engagement"], breakdown["simplicity"], breakdown["question_length"]]) / 100.0
        ),
    }
    groups = {k: round(v * 100.0, 4) for k, v in groups01.items()}

    return PlanQualityResult(score=round(score, 4), breakdown=breakdown, notes=notes_out, groups=groups)


def score_question_plan(*, planner_context_json: str, question_plan_json: str) -> PlanQualityResult:
    """
    LLM-judged scoring for a question plan.
    """
    plan_obj = _safe_json_loads(question_plan_json)
    if not isinstance(plan_obj, dict) or not isinstance(plan_obj.get("plan"), list):
        return PlanQualityResult(
            score=0.0,
            breakdown={k: 0.0 for k in BREAKDOWN_KEYS},
            notes=["invalid question_plan_json (expected JSON object with plan[])"],
            groups={},
        )

    context_subset = _extract_context_subset(planner_context_json)
    services_summary = str(context_subset.get("services_summary") or "").strip()

    ref = _select_reference_demos(services_summary=services_summary, k=int(os.getenv("QP_METRIC_K_DEMOS") or 3))
    # Keep reference demos compact: service_summary + plan only.
    ref_compact = [
        {
            "services_summary": d.get("services_summary"),
            "question_plan_json": _compact_plan_for_judge(d.get("question_plan_json") if isinstance(d.get("question_plan_json"), dict) else (_safe_json_loads(d.get("question_plan_json")) or {})),
        }
        for d in ref
    ]

    try:
        return _llm_judge_plan_quality(
            context_json=context_subset,
            candidate_plan_json=plan_obj,
            reference_demos=ref_compact,
        )
    except Exception as e:
        return PlanQualityResult(
            score=0.0,
            breakdown={k: 0.0 for k in BREAKDOWN_KEYS},
            notes=[f"LLM judge error: {type(e).__name__}"],
            groups={},
        )


def score_question_plan_optimizer(*, planner_context_json: str, question_plan_json: str) -> float:
    """
    Optimizer-focused score (0..100).

    Keeps the same signal as `score_question_plan`, but:
    - applies strict JSON/schema gate
    - returns a scalar
    """
    ok, _reason = _validate_question_plan_schema(
        planner_context_json=planner_context_json,
        question_plan_json=question_plan_json,
    )
    if not ok:
        return 0.0
    res = score_question_plan(planner_context_json=planner_context_json, question_plan_json=question_plan_json)
    return float(res.score)


def question_planner_quality_metric(example: Any, pred: Any, trace: Any = None) -> float | bool:
    """
    DSPy-compatible metric for the Question Planner.

    Inputs expected:
      - example.planner_context_json (or dict key) : JSON string
      - pred.question_plan_json : JSON string
    """
    planner_context_json = str(_get_example_field(example, "planner_context_json") or "")
    question_plan_json = str(getattr(pred, "question_plan_json", None) or "")

    # Track planner-call usage (best-effort).
    try:
        from programs.common.dspy_runtime import extract_dspy_usage

        _accumulate_usage("planner", extract_dspy_usage(pred))
    except Exception:
        pass

    ok, reason = _validate_question_plan_schema(
        planner_context_json=planner_context_json,
        question_plan_json=question_plan_json,
    )
    if not ok:
        if os.getenv("QP_METRIC_DEBUG_FAILS") == "true":
            head = (question_plan_json or "").replace("\n", " ")[:240]
            print(f"[QPMetric] schema_fail reason={reason} head={head!r}", flush=True)
        return False if trace is not None else 0.0

    score = float(score_question_plan_optimizer(planner_context_json=planner_context_json, question_plan_json=question_plan_json))
    if trace is not None:
        return bool(score >= float(os.getenv("QP_OPTIMIZER_ACCEPT_SCORE") or 70.0))
    # DSPy prints metrics as percentages under the assumption that float metrics are 0..1.
    # Keep internal scoring in 0..100, but return 0..1 here for sane logs/selection behavior.
    return max(0.0, min(1.0, score / 100.0))


__all__ = [
    "PlanQualityResult",
    "score_question_plan",
    "score_question_plan_optimizer",
    "question_planner_quality_metric",
    "get_metric_usage_summary",
    "reset_metric_usage",
]
