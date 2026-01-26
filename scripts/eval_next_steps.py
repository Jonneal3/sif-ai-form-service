#!/usr/bin/env python3
from __future__ import annotations

"""
Offline evaluation harness for next-step generation.

Design goals:
- Run a small set of "golden" contexts per use_case through the real orchestrator.
- Score outputs with hard validators (schema/type/option checks) and lightweight heuristics.
- Optionally compare base vs `.optimized.jsonl` demo packs by toggling their presence.

Run:
  PYTHONPATH=.:src python scripts/eval_next_steps.py --total-batches 3
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _load_golden_contexts(path: str) -> list[dict]:
    try:
        raw = json.loads(open(path, "r", encoding="utf-8").read())
    except Exception:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def _default_golden_contexts(use_case: str) -> list[dict]:
    # Minimal, vertical-agnostic contexts; RAG grounding can be injected later.
    base = {
        "platformGoal": "Pricing intake funnel for a design-first estimate.",
        "businessContext": "Ask like a pro designer/estimator. Keep it concrete and non-generic.",
        "instanceContext": {"industry": {"name": "Generic"}, "service": {"name": ""}},
    }
    if use_case == "tryon":
        return [
            {**base, "useCase": "tryon", "businessContext": base["businessContext"] + " This is visual-only."},
            {**base, "useCase": "tryon", "platformGoal": "Help a user pick a visual direction quickly."},
        ]
    return [
        {**base, "useCase": use_case},
        {**base, "useCase": use_case, "platformGoal": "Collect just enough to give a fast estimate."},
    ]


def _load_golden_for_use_case(*, use_case: str, golden_path: str, golden_dir: str) -> list[dict]:
    if golden_path:
        return _load_golden_contexts(golden_path)
    try:
        p = os.path.join(golden_dir, f"{use_case}.json")
        if os.path.exists(p):
            return _load_golden_contexts(p)
    except Exception:
        pass
    return []


def _best_effort_extract_json(text: str) -> dict:
    if not text:
        return {}
    t = str(text).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"(\{[\s\S]*\})", t)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _make_proxy_judge():
    """
    Proxy judge = a rubric-graded LLM check.
    This is optional because it requires a configured DSPy LM.
    """
    try:
        import dspy  # type: ignore
    except Exception:
        return None, None

    try:
        # Reuse the service's LM config logic so eval runs the same way as runtime.
        from programs.form_pipeline.orchestrator import _make_dspy_lm  # type: ignore

        cfg = _make_dspy_lm()
    except Exception:
        cfg = None
    if not cfg:
        return None, None

    lm = dspy.LM(
        model=cfg["model"],
        temperature=0.0,
        max_tokens=int(os.getenv("EVAL_PROXY_JUDGE_MAX_TOKENS") or "600"),
        timeout=float(os.getenv("EVAL_PROXY_JUDGE_TIMEOUT_SEC") or "20"),
        num_retries=0,
    )
    try:
        dspy.settings.configure(lm=lm, track_usage=False)
    except Exception:
        pass

    class NextStepsProxyJudge(dspy.Signature):  # type: ignore
        """
        You are grading next-step generation quality.

        Return STRICT JSON only in `result_json`:
          {
            "score": <0-100 integer>,
            "notes": "<short text>"
          }

        Scoring rubric:
        - Relevance to use_case + goal (0-40)
        - Non-genericness (no filler like "Option A") (0-30)
        - Uses grounding_summary when provided (0-20)
        - Clear, concrete questions (0-10)
        """

        prompt: str = dspy.InputField(desc="Rubric + context + steps")  # type: ignore
        result_json: str = dspy.OutputField(desc="JSON only")  # type: ignore

    judge = dspy.Predict(NextStepsProxyJudge)  # type: ignore
    return judge, cfg


def _proxy_judge_score(
    *,
    judge: Any,
    use_case: str,
    platform_goal: str,
    business_context: str,
    grounding_summary: str,
    steps: list[dict],
) -> tuple[int, str]:
    if not judge:
        return 0, "judge not configured"
    # Keep prompt short.
    steps_blob = _dump(steps)[:3500]
    grounding = str(grounding_summary or "").strip().replace("\n", " ")[:600]
    prompt = "\n".join(
        [
            "Grade the following next-step output. Return JSON only.",
            f"use_case: {use_case}",
            f"platform_goal: {str(platform_goal or '')[:300]}",
            f"business_context: {str(business_context or '')[:300]}",
            f"grounding_summary: {grounding}",
            "steps_json: " + steps_blob,
        ]
    )
    try:
        pred = judge(prompt=prompt)
        raw = getattr(pred, "result_json", None) or ""
    except Exception as e:
        return 0, f"judge error: {e}"
    obj = _best_effort_extract_json(str(raw))
    score_raw = obj.get("score")
    try:
        score = int(score_raw)
    except Exception:
        score = 0
    score = max(0, min(100, score))
    notes = str(obj.get("notes") or "").strip()[:240]
    return score, notes


def _validate_steps_schema(steps: list[dict]) -> tuple[bool, list[str]]:
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

    type_to_model = {
        "text": TextInputUI,
        "text_input": TextInputUI,
        "intro": IntroUI,
        "rating": RatingUI,
        "date_picker": DatePickerUI,
        "color_picker": ColorPickerUI,
        "lead_capture": LeadCaptureUI,
        "pricing": PricingUI,
        "confirmation": ConfirmationUI,
        "designer": DesignerUI,
        "file_upload": FileUploadUI,
        "upload": FileUploadUI,
        "file_picker": FileUploadUI,
        "budget_cards": BudgetCardsUI,
        "multiple_choice": MultipleChoiceUI,
        "choice": MultipleChoiceUI,
        "segmented_choice": MultipleChoiceUI,
        "chips_multi": MultipleChoiceUI,
        "yes_no": MultipleChoiceUI,
        "image_choice_grid": MultipleChoiceUI,
        "searchable_select": SearchableSelectUI,
        "composite": CompositeUI,
        "gallery": GalleryUI,
    }

    errors: list[str] = []
    for i, s in enumerate(steps or []):
        if not isinstance(s, dict):
            errors.append(f"step[{i}]: not an object")
            continue
        t = str(s.get("type") or "").strip().lower()
        m = type_to_model.get(t)
        if not m:
            errors.append(f"step[{i}]: unsupported type '{t}'")
            continue
        try:
            m.model_validate(s)
        except Exception as e:
            errors.append(f"step[{i}]: schema invalid: {e}")
    return (len(errors) == 0, errors)


def _validate_no_filler_patterns(steps: list[dict]) -> tuple[bool, list[str]]:
    bad = []
    pat = re.compile(r"\b(option\s*[a-d]|category\s*\d+)\b", re.IGNORECASE)
    for i, s in enumerate(steps or []):
        text = " ".join(
            [
                str(s.get("question") or ""),
                str(s.get("label") or ""),
                _dump(s.get("options") or []),
            ]
        )
        if pat.search(text):
            bad.append(f"step[{i}] contains filler pattern")
    return (len(bad) == 0, bad)


def _validate_choice_option_counts(steps: list[dict]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for i, s in enumerate(steps or []):
        t = str(s.get("type") or "").strip().lower()
        if t not in {"multiple_choice", "choice", "segmented_choice", "chips_multi", "image_choice_grid"}:
            continue
        opts = s.get("options") or []
        if not isinstance(opts, list):
            errors.append(f"step[{i}]: options not a list")
            continue
        # Allow 3-12 in eval (runtime constraints can be tighter); this is mainly to avoid "always 2" or "always 50".
        if len(opts) < 3 or len(opts) > 12:
            errors.append(f"step[{i}]: option count out of range ({len(opts)})")
    return (len(errors) == 0, errors)


def _validate_last_batch_finisher(steps: list[dict]) -> tuple[bool, list[str]]:
    if len(steps) < 2:
        return False, ["needs at least 2 steps for Upload → Gallery finisher"]
    a = str(steps[-2].get("type") or "").strip().lower()
    b = str(steps[-1].get("type") or "").strip().lower()
    ok = a in {"upload", "file_upload", "file_picker"} and b == "gallery"
    return ok, ([] if ok else [f"last two steps are {a!r} → {b!r}, expected upload → gallery"])


@dataclass
class CaseResult:
    use_case: str
    batch_number: int
    ok: bool
    errors: list[str]
    steps: int
    judge_score: int


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-batches", type=int, default=3)
    ap.add_argument("--use-cases", default="scene,scene_placement,tryon")
    ap.add_argument("--golden-path", default="", help="Optional JSON array of context dicts (overrides golden-dir)")
    ap.add_argument(
        "--golden-dir",
        default=str((__import__("pathlib").Path(__file__).resolve().parent / "golden_contexts")),
        help="Directory containing per-use_case golden context files (e.g. scene.json)",
    )
    ap.add_argument(
        "--demo-only",
        action="store_true",
        help="Only validate JSONL demo packs on disk (no LLM calls).",
    )
    ap.add_argument(
        "--proxy-judge",
        action="store_true",
        help="Enable an LLM rubric judge (requires DSPy LM configured).",
    )
    ap.add_argument(
        "--proxy-judge-min",
        type=int,
        default=60,
        help="Minimum proxy-judge score (0-100) to count as pass (only when --proxy-judge is set).",
    )
    args = ap.parse_args()

    total_batches = max(1, min(10, int(args.total_batches or 3)))
    use_cases = [u.strip() for u in str(args.use_cases or "").split(",") if u.strip()]

    results: list[CaseResult] = []
    if args.demo_only:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "src" / "programs" / "form_pipeline" / "demos"
        failures = 0
        for uc in use_cases:
            bdir = root / uc / f"b{total_batches}"
            for pack in sorted(bdir.glob("batch_*.jsonl")):
                if pack.name.endswith(".optimized.jsonl"):
                    continue
                for line in pack.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        failures += 1
                        print(f"- FAIL {pack}: invalid json line")
                        continue
                    outs = rec.get("outputs") if isinstance(rec, dict) else {}
                    ms = outs.get("mini_steps_jsonl") if isinstance(outs, dict) else ""
                    steps: list[dict] = []
                    for l in str(ms or "").splitlines():
                        try:
                            obj = json.loads(l)
                        except Exception:
                            continue
                        if isinstance(obj, dict):
                            steps.append(obj)
                    ok, errs = _validate_steps_schema(steps)
                    if not ok:
                        failures += 1
                        print(f"- FAIL {pack}: demo output invalid")
                        for e in errs[:5]:
                            print(f"  - {e}")
        print(f"Demo-pack validation: {'OK' if failures == 0 else f'{failures} failure(s)'}")
        return 0 if failures == 0 else 2

    from programs.form_pipeline.orchestrator import next_steps_jsonl
    judge = None
    judge_cfg = None
    if args.proxy_judge:
        judge, judge_cfg = _make_proxy_judge()
        if not judge:
            print("Proxy judge requested but LM is not configured. Continuing without proxy judge.")
        else:
            print(f"Proxy judge enabled (model={judge_cfg.get('modelName') or judge_cfg.get('model')})")
    for use_case in use_cases:
        contexts = _load_golden_for_use_case(use_case=use_case, golden_path=args.golden_path, golden_dir=args.golden_dir)
        if not contexts:
            contexts = _default_golden_contexts(use_case)

        for ctx in contexts:
            for batch_number in range(1, total_batches + 1):
                payload: Dict[str, Any] = {
                    "includeMeta": True,
                    "useCase": ctx.get("useCase") or use_case,
                    "platformGoal": ctx.get("platformGoal"),
                    "businessContext": ctx.get("businessContext"),
                    "instanceContext": ctx.get("instanceContext") or {},
                    "batchConstraints": {
                        "maxBatches": total_batches,
                        "minStepsPerBatch": 2,
                        "maxStepsPerBatch": 3,
                    },
                    "batchId": f"batch-{batch_number}",
                    "currentBatch": {"batchNumber": batch_number},
                    "session": {"sessionId": "eval", "instanceId": "eval"},
                    "stepDataSoFar": {},
                    "askedStepIds": [],
                    "answeredQA": [],
                }

                resp = next_steps_jsonl(payload)
                steps = resp.get("miniSteps") if isinstance(resp, dict) else None
                steps = steps if isinstance(steps, list) else []

                ok = True
                errors: list[str] = []
                judge_score = 0

                v_ok, v_err = _validate_steps_schema(steps)
                ok = ok and v_ok
                errors += v_err

                v_ok, v_err = _validate_no_filler_patterns(steps)
                ok = ok and v_ok
                errors += v_err

                v_ok, v_err = _validate_choice_option_counts(steps)
                ok = ok and v_ok
                errors += v_err

                if batch_number == total_batches:
                    v_ok, v_err = _validate_last_batch_finisher(steps)
                    ok = ok and v_ok
                    errors += v_err

                if args.proxy_judge and judge:
                    # Pull grounding from both common request shapes.
                    grounding = (
                        ctx.get("groundingSummary")
                        or ctx.get("grounding_summary")
                        or (ctx.get("instanceContext") or {}).get("groundingSummary")
                        or ""
                    )
                    judge_score, notes = _proxy_judge_score(
                        judge=judge,
                        use_case=use_case,
                        platform_goal=str(ctx.get("platformGoal") or ""),
                        business_context=str(ctx.get("businessContext") or ""),
                        grounding_summary=str(grounding or ""),
                        steps=steps,
                    )
                    if judge_score < int(args.proxy_judge_min):
                        ok = False
                        errors.append(f"proxy_judge score {judge_score} < {int(args.proxy_judge_min)} ({notes})")

                results.append(
                    CaseResult(
                        use_case=use_case,
                        batch_number=batch_number,
                        ok=ok,
                        errors=errors,
                        steps=len(steps),
                        judge_score=judge_score,
                    )
                )

    total = len(results)
    passed = sum(1 for r in results if r.ok)
    failed = total - passed
    print(f"Eval: {passed}/{total} passed ({failed} failed)")
    if args.proxy_judge:
        scores = [r.judge_score for r in results if isinstance(r.judge_score, int)]
        if scores:
            avg = sum(scores) / max(1, len(scores))
            print(f"Proxy judge avg score: {avg:.1f} (min={min(scores)}, max={max(scores)})")
    for r in results:
        if r.ok:
            continue
        extra = f" judge={r.judge_score}" if args.proxy_judge else ""
        print(f"- FAIL use_case={r.use_case} batch={r.batch_number} steps={r.steps}{extra}")
        for e in r.errors[:10]:
            print(f"  - {e}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

