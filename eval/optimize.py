from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from eval.datasets import load_eval_cases
from eval.metrics import compute_metrics
from flow_planner import (
    _extract_grounding_summary,
    _extract_service_anchor_terms,
    _extract_use_case,
    _select_attribute_families,
)


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


def _build_context_json(payload: Dict[str, Any]) -> str:
    required_uploads_raw = payload.get("requiredUploads") or payload.get("required_uploads") or []
    required_uploads = required_uploads_raw if isinstance(required_uploads_raw, list) else []

    known_answers_raw = payload.get("stepDataSoFar") or payload.get("knownAnswers") or {}
    known_answers = known_answers_raw if isinstance(known_answers_raw, dict) else {}

    already_asked_raw = payload.get("alreadyAskedKeys") or payload.get("alreadyAskedKeysJson") or []
    already_asked: List[str] = []
    if isinstance(already_asked_raw, list):
        for x in already_asked_raw:
            t = str(x or "").strip()
            if not t:
                continue
            already_asked.append(_normalize_step_id(t))

    form_plan_raw = payload.get("formPlan") or payload.get("form_plan") or []
    form_plan = form_plan_raw if isinstance(form_plan_raw, list) else []

    batch_state_raw = payload.get("batchState") or payload.get("batch_state") or {}
    batch_state = batch_state_raw if isinstance(batch_state_raw, dict) else {}

    items_raw = payload.get("items") or []
    items = items_raw if isinstance(items_raw, list) else []

    instance_subcategories_raw = payload.get("instanceSubcategories") or payload.get("instance_subcategories") or []
    instance_subcategories = instance_subcategories_raw if isinstance(instance_subcategories_raw, list) else []

    industry = str(payload.get("industry") or payload.get("vertical") or "General")[:80]
    service = str(payload.get("service") or payload.get("subcategoryName") or "")[:80]
    use_case = _extract_use_case(payload)
    grounding_summary = _extract_grounding_summary(payload)
    service_anchor_terms = _extract_service_anchor_terms(industry, service, grounding_summary)
    attribute_families = _select_attribute_families(use_case)

    context = {
        "platform_goal": str(payload.get("platformGoal") or payload.get("platform_goal") or "")[:600],
        "business_context": str(payload.get("businessContext") or payload.get("business_context") or "")[:200],
        "industry": industry,
        "service": service,
        "use_case": use_case,
        "required_uploads": required_uploads,
        "personalization_summary": str(payload.get("personalizationSummary") or payload.get("personalization_summary") or "")[:1200],
        "known_answers": known_answers,
        "already_asked_keys": already_asked,
        "form_plan": form_plan,
        "batch_state": batch_state,
        "items": items,
        "instance_subcategories": instance_subcategories,
        "attribute_families": attribute_families,
        "service_anchor_terms": service_anchor_terms,
    }
    if grounding_summary:
        context["grounding_summary"] = grounding_summary
    return json.dumps(context, separators=(",", ":"), ensure_ascii=True, sort_keys=True)


def _payload_from_example(ex: Any) -> Dict[str, Any]:
    """
    Reconstruct a payload-like dict from a DSPy Example for metric computation.
    """
    # Example behaves like an object with attributes in most DSPy versions.
    context_json = getattr(ex, "context_json", "") or ""
    allowed_raw = getattr(ex, "allowed_mini_types", []) or []
    max_steps_raw = getattr(ex, "max_steps", "") or "4"

    context = _best_effort_parse_json(context_json)
    if not isinstance(context, dict):
        context = {}

    form_plan = context.get("form_plan")
    if not isinstance(form_plan, list):
        form_plan = []

    already = context.get("already_asked_keys")
    if not isinstance(already, list):
        already = []

    if isinstance(allowed_raw, list):
        allowed_list = [str(x).strip() for x in allowed_raw if str(x).strip()]
    else:
        allowed_list = [s.strip() for s in str(allowed_raw).split(",") if s.strip()]

    try:
        max_steps_int = int(str(max_steps_raw))
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
    from modules.signatures.json_signatures import FileUploadMini, MultipleChoiceMini, RatingMini, TextInputMini

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
        context_json = _build_context_json(payload)
        allowed_mini_types = payload.get("allowedMiniTypes") or payload.get("allowed_mini_types") or []
        ex = dspy.Example(
            context_json=context_json,
            batch_id=str(payload.get("batchId") or payload.get("batch_id") or "ContextCore")[:40],
            max_steps=int(str(payload.get("maxSteps") or payload.get("max_steps") or "4")),
            allowed_mini_types=[str(x).strip() for x in allowed_mini_types if str(x).strip()]
            if isinstance(allowed_mini_types, list)
            else [s.strip() for s in str(allowed_mini_types).split(",") if s.strip()],
            mini_steps_jsonl="",
        )
        try:
            ex = ex.with_inputs(
                "context_json",
                "batch_id",
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
            "context_json",
            "batch_id",
            "max_steps",
            "allowed_mini_types",
        ]:
            inputs[k] = getattr(ex, k, None)
        outputs["mini_steps_jsonl"] = getattr(ex, "mini_steps_jsonl", None)
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

    from modules.signatures.json_signatures import NextStepsJSONL

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
