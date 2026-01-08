from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from eval.datasets import load_eval_cases
from eval.metrics import compute_metrics


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _best_effort_parse_json(text: str) -> Any:
    """
    Minimal best-effort JSON parser (mirrors the behavior used at runtime).
    """
    if not text:
        return None
    t = str(text).strip()
    # Strip code fences if present.
    if t.startswith("```"):
        t = t.strip().lstrip("`")
        t = t.replace("json", "", 1).strip()
    parsed = _safe_json_loads(t)
    if parsed is not None:
        return parsed
    # Fallback: find first array/object block.
    import re

    m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", t)
    if not m:
        return None
    return _safe_json_loads(m.group(0))


def _normalize_step_id(step_id: str) -> str:
    t = str(step_id or "").strip()
    if not t:
        return t
    return t.replace("_", "-")


def _payload_from_example(ex: Any) -> Dict[str, Any]:
    """
    Reconstruct a payload-like dict from a DSPy Example for metric computation.
    """
    # Example behaves like an object with attributes in most DSPy versions.
    form_plan_json = getattr(ex, "form_plan_json", "") or ""
    already_asked_keys_json = getattr(ex, "already_asked_keys_json", "") or ""
    allowed_csv = getattr(ex, "allowed_mini_types", "") or ""
    max_steps = getattr(ex, "max_steps", "") or "4"

    form_plan = _best_effort_parse_json(form_plan_json)
    if not isinstance(form_plan, list):
        form_plan = []

    already = _best_effort_parse_json(already_asked_keys_json)
    if not isinstance(already, list):
        already = []

    allowed_list = [s.strip() for s in str(allowed_csv).split(",") if s.strip()]
    try:
        max_steps_int = int(str(max_steps))
    except Exception:
        max_steps_int = 4

    return {
        "formPlan": form_plan,
        "alreadyAskedKeys": [_normalize_step_id(str(x)) for x in already if str(x or "").strip()],
        "allowedMiniTypes": allowed_list,
        "maxSteps": max_steps_int,
    }


def _validated_steps_from_pred(pred: Any) -> List[Dict[str, Any]]:
    """
    Parse `pred.mini_steps_jsonl` and validate using the same Pydantic models as runtime.
    """
    from modules.signatures.flow_signatures import FileUploadMini, MultipleChoiceMini, RatingMini, TextInputMini

    raw_lines = getattr(pred, "mini_steps_jsonl", None) or ""
    out: List[Dict[str, Any]] = []
    for line in str(raw_lines).splitlines():
        t = (line or "").strip()
        if not t:
            continue
        obj = _best_effort_parse_json(t)
        if not isinstance(obj, dict):
            continue
        ttype = obj.get("type")
        try:
            if ttype == "text_input":
                m = TextInputMini.model_validate(obj).model_dump()
            elif ttype == "multiple_choice":
                if "options" not in obj or not obj.get("options"):
                    obj = dict(obj)
                    obj["options"] = ["Not sure"]
                m = MultipleChoiceMini.model_validate(obj).model_dump()
            elif ttype == "rating":
                m = RatingMini.model_validate(obj).model_dump()
            elif ttype == "file_upload":
                m = FileUploadMini.model_validate(obj).model_dump()
            else:
                continue
        except Exception:
            continue
        m["id"] = _normalize_step_id(str(m.get("id") or ""))
        out.append(m)
    return out


def _metric_fn_factory() -> Callable[..., float]:
    """
    Return a DSPy metric function compatible with common DSPy optimizer signatures.

    We score based on strict structural invariants (schema + constraints). 1.0 = pass, 0.0 = fail.
    """

    def metric(example: Any, pred: Any, *args: Any, **kwargs: Any) -> float:
        payload = _payload_from_example(example)
        steps = _validated_steps_from_pred(pred)
        result = {"ok": True, "miniSteps": steps}
        m = compute_metrics(payload, result)
        ok = (
            m.within_max_steps
            and m.ids_all_normalized
            and m.no_steps_in_already_asked
            and m.types_all_allowed
            and m.has_min_step_when_needed
        )
        return 1.0 if ok else 0.0

    return metric


def _pick_optimizer(dspy: Any) -> Tuple[str, Any]:
    """
    Best-effort selection of a DSPy v3 optimizer/teleprompter.
    We try common names across DSPy versions and return (name, class).
    """
    candidates = [
        ("MIPROv2", ("dspy.teleprompt", "MIPROv2")),
        ("MIPRO", ("dspy.teleprompt", "MIPRO")),
        ("BootstrapFewShotWithRandomSearch", ("dspy.teleprompt", "BootstrapFewShotWithRandomSearch")),
        ("BootstrapFewShot", ("dspy.teleprompt", "BootstrapFewShot")),
    ]
    for name, (mod, sym) in candidates:
        try:
            m = __import__(mod, fromlist=[sym])
            cls = getattr(m, sym, None)
            if cls is not None:
                return name, cls
        except Exception:
            continue
    raise RuntimeError(
        "No known DSPy optimizer found. Tried: "
        + ", ".join([n for n, _ in candidates])
        + ". Please check the installed dspy-ai version and update eval/optimize.py accordingly."
    )


def _examples_from_cases(dspy: Any, cases: List[Dict[str, Any]]) -> List[Any]:
    out: List[Any] = []
    for payload in cases:
        # Align with NextStepsJSONL signature input names (snake_case).
        ex = dspy.Example(
            platform_goal=str(payload.get("platformGoal") or payload.get("platform_goal") or "")[:600],
            batch_id=str(payload.get("batchId") or payload.get("batch_id") or "ContextCore")[:40],
            business_context=str(payload.get("businessContext") or payload.get("business_context") or "")[:200],
            industry=str(payload.get("industry") or payload.get("vertical") or "General")[:80],
            service=str(payload.get("service") or payload.get("subcategoryName") or "")[:80],
            grounding_preview=str(payload.get("groundingPreview") or payload.get("grounding_preview") or "")[:2000],
            required_uploads_json=json.dumps(payload.get("requiredUploads") or payload.get("required_uploads") or [])[:800],
            personalization_summary=str(payload.get("personalizationSummary") or payload.get("personalization_summary") or "")[:1200],
            known_answers_json=json.dumps(payload.get("stepDataSoFar") or payload.get("knownAnswers") or {})[:2400],
            already_asked_keys_json=json.dumps(payload.get("alreadyAskedKeys") or payload.get("alreadyAskedKeysJson") or [])[:2000],
            form_plan_json=json.dumps(payload.get("formPlan") or payload.get("form_plan") or [])[:3600],
            batch_state_json=json.dumps(payload.get("batchState") or payload.get("batch_state") or {})[:2000],
            max_steps=str(payload.get("maxSteps") or payload.get("max_steps") or "4"),
            allowed_mini_types=",".join(
                [str(x).strip() for x in (payload.get("allowedMiniTypes") or payload.get("allowed_mini_types") or []) if str(x).strip()]
            )[:200],
            # Outputs can be empty: our metric is invariant-based, not label-based.
            produced_form_plan_json="",
            must_have_copy_json="{}",
            ready_for_image_gen="false",
            mini_steps_jsonl="",
        )
        try:
            ex = ex.with_inputs(
                "platform_goal",
                "batch_id",
                "business_context",
                "industry",
                "service",
                "grounding_preview",
                "required_uploads_json",
                "personalization_summary",
                "known_answers_json",
                "already_asked_keys_json",
                "form_plan_json",
                "batch_state_json",
                "max_steps",
                "allowed_mini_types",
            )
        except Exception:
            pass
        out.append(ex)
    return out


def _write_examples_pack(path: str, demos: List[Any]) -> None:
    """
    Write demos (dspy.Example) to a JSONL pack compatible with examples/registry.py.
    """
    lines: List[str] = []
    for i, ex in enumerate(demos):
        inputs: Dict[str, Any] = {}
        outputs: Dict[str, Any] = {}
        # Best-effort: pull fields directly off the Example.
        for k in [
            "platform_goal",
            "batch_id",
            "business_context",
            "industry",
            "service",
            "grounding_preview",
            "required_uploads_json",
            "personalization_summary",
            "known_answers_json",
            "already_asked_keys_json",
            "form_plan_json",
            "batch_state_json",
            "max_steps",
            "allowed_mini_types",
        ]:
            inputs[k] = getattr(ex, k, None)
        for k in ["produced_form_plan_json", "must_have_copy_json", "ready_for_image_gen", "mini_steps_jsonl"]:
            outputs[k] = getattr(ex, k, None)
        lines.append(
            json.dumps(
                {"meta": {"name": f"optimized_demo_{i}"}, "inputs": inputs, "outputs": outputs},
                ensure_ascii=False,
            )
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="Optimize NextStepsJSONL demos using DSPy.")
    ap.add_argument("--cases", default="eval_cases.jsonl", help="JSONL filename under eval/ (default: eval_cases.jsonl)")
    ap.add_argument(
        "--out-pack",
        default="examples/next_steps_examples.optimized.jsonl",
        help="Where to write the optimized examples pack (repo-relative).",
    )
    ap.add_argument("--max-demos", type=int, default=8, help="Cap number of demos to extract (best-effort).")
    args = ap.parse_args()

    cases = load_eval_cases(args.cases)
    if not cases:
        sys.stderr.write(f"[optimize] No cases found for eval/{args.cases}\n")
        return 2

    try:
        import dspy  # type: ignore
    except Exception as e:
        sys.stderr.write(f"[optimize] Failed to import dspy: {e}\n")
        return 2

    # Create an LM the same way runtime does (so results are comparable).
    from flow_planner import _make_dspy_lm, _configure_dspy

    lm_cfg = _make_dspy_lm()
    if not lm_cfg:
        sys.stderr.write("[optimize] DSPy LM not configured. Set DSPY_PROVIDER + API key env vars.\n")
        return 2

    llm_timeout = float(os.getenv("DSPY_LLM_TIMEOUT_SEC") or "20")
    temperature = float(os.getenv("DSPY_TEMPERATURE") or "0.7")
    max_tokens = int(os.getenv("DSPY_NEXT_STEPS_MAX_TOKENS") or "2000")
    lm = dspy.LM(
        model=lm_cfg["model"],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=llm_timeout,
        num_retries=0,
    )
    _configure_dspy(lm)
    try:
        if hasattr(dspy, "settings") and hasattr(dspy.settings, "configure"):
            dspy.settings.configure(lm=lm, track_usage=True)
    except Exception:
        pass

    from modules.signatures.flow_signatures import NextStepsJSONL

    class NextStepsProgram(dspy.Module):  # type: ignore[misc]
        def __init__(self):
            super().__init__()
            self.prog = dspy.Predict(NextStepsJSONL)

        def forward(self, **kwargs: Any) -> Any:
            return self.prog(**kwargs)

    program = NextStepsProgram()
    metric = _metric_fn_factory()

    name, OptimizerCls = _pick_optimizer(dspy)
    sys.stderr.write(f"[optimize] Using optimizer={name}\n")

    train_payloads = [c.payload for c in cases]
    trainset = _examples_from_cases(dspy, train_payloads)

    # Instantiate optimizer with best-effort kwargs.
    opt = None
    try:
        opt = OptimizerCls(metric=metric)  # type: ignore[call-arg]
    except Exception:
        try:
            opt = OptimizerCls(metric)  # type: ignore[call-arg]
        except Exception as e:
            sys.stderr.write(f"[optimize] Failed to instantiate optimizer {name}: {e}\n")
            return 2

    compiled = None
    try:
        compile_fn = getattr(opt, "compile", None)
        if callable(compile_fn):
            compiled = compile_fn(program, trainset=trainset)  # type: ignore[misc]
        else:
            compiled = opt(program, trainset=trainset)  # type: ignore[misc]
    except TypeError:
        # Try common alt signature: compile(program, trainset, valset=None)
        try:
            compiled = opt.compile(program, trainset)  # type: ignore[misc]
        except Exception as e:
            sys.stderr.write(f"[optimize] Failed to compile with optimizer {name}: {e}\n")
            return 2
    except Exception as e:
        sys.stderr.write(f"[optimize] Failed to compile with optimizer {name}: {e}\n")
        return 2

    # Extract demos from the compiled program (best-effort).
    demos: List[Any] = []
    try:
        predictor = getattr(compiled, "prog", None) or getattr(compiled, "predictor", None)
        if predictor is None:
            predictor = getattr(compiled, "prog", None)
        maybe_demos = getattr(predictor, "demos", None) if predictor is not None else None
        if isinstance(maybe_demos, list):
            demos = maybe_demos[: max(0, int(args.max_demos))]
    except Exception:
        demos = []

    if not demos:
        sys.stderr.write("[optimize] No demos extracted from compiled program. Optimizer may not expose demos in this DSPy version.\n")
        return 2

    out_path = os.path.join(os.path.dirname(__file__), "..", args.out_pack)
    out_path = os.path.normpath(out_path)
    _write_examples_pack(out_path, demos)

    sys.stderr.write(
        f"[optimize] ✅ wrote {len(demos)} demos to {args.out_pack}\n"
        f"[optimize] To use at runtime: set DSPY_NEXT_STEPS_DEMO_PACK='{os.path.basename(args.out_pack)}' (and place it under examples/)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


