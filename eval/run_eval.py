"""
Offline evaluation harness for NextStepsJSONL.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from examples.registry import load_jsonl_records
from flow_planner import _configure_dspy, _make_dspy_lm
from app.dspy.flow_planner_module import FlowPlannerModule

from eval.metrics import score_prediction


def _coerce_int(value: Any, default: int) -> int:
    try:
        n = int(value)
        return n if n > 0 else default
    except Exception:
        return default


def _ensure_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    return [s.strip() for s in str(value or "").split(",") if s.strip()]


def _build_inputs(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "context_json": raw.get("context_json") or "{}",
        "batch_id": raw.get("batch_id") or "EvalBatch",
        "max_steps": _coerce_int(raw.get("max_steps"), 4),
        "allowed_mini_types": _ensure_list(raw.get("allowed_mini_types")),
    }


def _wrap_throttled_lm(lm: Any, throttle_sec: float, max_calls: int) -> Any:
    if throttle_sec <= 0 and max_calls <= 0:
        return lm

    import dspy  # type: ignore

    class ThrottledLM(dspy.BaseLM):
        def __init__(self, inner: Any, delay: float, max_calls: int) -> None:
            model = getattr(inner, "model", "unknown")
            model_type = getattr(inner, "model_type", "chat")
            kwargs = getattr(inner, "kwargs", {}) or {}
            super().__init__(
                model=model,
                model_type=model_type,
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 1000),
                cache=getattr(inner, "cache", True),
            )
            self._inner = inner
            self._delay = delay
            self._max_calls = max_calls
            self._calls = 0

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def forward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any):
            import time

            if self._max_calls > 0:
                self._calls += 1
                if self._calls > self._max_calls:
                    raise RuntimeError("DSPy eval LLM call limit exceeded")
            if self._delay > 0:
                time.sleep(self._delay)
            return self._inner.forward(prompt=prompt, messages=messages, **kwargs)

        async def aforward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any):
            import asyncio

            if self._max_calls > 0:
                self._calls += 1
                if self._calls > self._max_calls:
                    raise RuntimeError("DSPy eval LLM call limit exceeded")
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            if hasattr(self._inner, "aforward"):
                return await self._inner.aforward(prompt=prompt, messages=messages, **kwargs)
            return self._inner.forward(prompt=prompt, messages=messages, **kwargs)

        def copy(self, **kwargs: Any):
            inner_copy = self._inner.copy(**kwargs) if hasattr(self._inner, "copy") else self._inner
            return ThrottledLM(inner_copy, self._delay, self._max_calls)

    return ThrottledLM(lm, throttle_sec, max_calls)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/datasets/next_steps_structural.jsonl")
    parser.add_argument("--max-examples", type=int, default=int(os.getenv("DSPY_EVAL_MAX_EXAMPLES") or "10"))
    parser.add_argument("--report", default="eval/reports/next_steps_eval.json")
    args = parser.parse_args()

    records = load_jsonl_records(args.dataset)
    if not records:
        print(f"No records found in {args.dataset}")
        return 1

    try:
        import dspy  # type: ignore
    except Exception as e:
        print(f"DSPy import failed: {e}")
        return 1

    lm_cfg = _make_dspy_lm()
    if not lm_cfg:
        print("DSPy LM not configured; set DSPY_PROVIDER + API key")
        return 1

    lm = dspy.LM(
        model=lm_cfg["model"],
        temperature=0.3,
        max_tokens=int(os.getenv("DSPY_EVAL_MAX_TOKENS") or os.getenv("DSPY_NEXT_STEPS_MAX_TOKENS") or "1000"),
        timeout=float(os.getenv("DSPY_LLM_TIMEOUT_SEC") or "20"),
        num_retries=0,
    )
    throttle_sec = float(os.getenv("DSPY_EVAL_THROTTLE_SEC") or "1.0")
    max_calls = int(os.getenv("DSPY_EVAL_MAX_CALLS") or "40")
    _configure_dspy(_wrap_throttled_lm(lm, throttle_sec, max_calls))

    module = FlowPlannerModule()

    results: List[Dict[str, Any]] = []
    totals: Dict[str, int] = {}
    score_sum = 0.0
    count = 0

    for rec in records:
        if args.max_examples and count >= args.max_examples:
            break
        inputs = _build_inputs(rec.inputs)
        try:
            pred = module(**inputs)
        except RuntimeError as e:
            print(f"Eval stopped: {e}")
            break
        prediction_jsonl = getattr(pred, "mini_steps_jsonl", "")
        score, details = score_prediction(inputs, prediction_jsonl)
        score_sum += score
        count += 1
        for key, val in details.items():
            totals[key] = totals.get(key, 0) + int(val)
        results.append(
            {
                "name": (rec.meta or {}).get("name"),
                "score": score,
                "details": details,
            }
        )

    avg = score_sum / count if count else 0.0
    report = {
        "avg_score": avg,
        "count": count,
        "totals": totals,
        "examples": results,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))

    print(f"Avg score: {avg:.3f} over {count} examples")
    print("Failure totals:")
    for key, val in sorted(totals.items()):
        if val:
            print(f"  {key}: {val}")

    worst = sorted(results, key=lambda r: r["score"])[:5]
    print("Worst examples:")
    for item in worst:
        print(f"  {item.get('name') or 'unknown'}: {item['score']:.3f}")

    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
