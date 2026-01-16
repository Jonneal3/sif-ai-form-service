"""
Offline compile script for NextStepsJSONL demos.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from examples.registry import load_jsonl_records
from app.pipeline.form_pipeline import _configure_dspy, _make_dspy_lm
from app.dspy.flow_planner_module import FlowPlannerModule

from eval.metrics import score_prediction


INPUT_KEYS = ["context_json", "batch_id", "max_steps", "allowed_mini_types"]
OUTPUT_KEYS = ["mini_steps_jsonl"]


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
        "batch_id": raw.get("batch_id") or "CompileBatch",
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
                    raise RuntimeError("DSPy compile LLM call limit exceeded")
            if self._delay > 0:
                time.sleep(self._delay)
            return self._inner.forward(prompt=prompt, messages=messages, **kwargs)

        async def aforward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any):
            import asyncio

            if self._max_calls > 0:
                self._calls += 1
                if self._calls > self._max_calls:
                    raise RuntimeError("DSPy compile LLM call limit exceeded")
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            if hasattr(self._inner, "aforward"):
                return await self._inner.aforward(prompt=prompt, messages=messages, **kwargs)
            return self._inner.forward(prompt=prompt, messages=messages, **kwargs)

        def copy(self, **kwargs: Any):
            inner_copy = self._inner.copy(**kwargs) if hasattr(self._inner, "copy") else self._inner
            return ThrottledLM(inner_copy, self._delay, self._max_calls)

    return ThrottledLM(lm, throttle_sec, max_calls)


def _get_example_value(ex: Any, key: str) -> Any:
    if hasattr(ex, "_store") and isinstance(ex._store, dict) and key in ex._store:
        return ex._store.get(key)
    if hasattr(ex, key):
        return getattr(ex, key)
    return None


def _example_to_record(ex: Any) -> Dict[str, Any]:
    inputs = {k: _get_example_value(ex, k) for k in INPUT_KEYS}
    outputs = {k: _get_example_value(ex, k) for k in OUTPUT_KEYS}
    return {"inputs": inputs, "outputs": outputs}


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def _utc_timestamp_slug() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _archive_file(src: Path, archive_dir: Path, *, label: str) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    dst = archive_dir / f"{label}.{_utc_timestamp_slug()}{src.suffix}"
    shutil.copyfile(src, dst)
    return dst


def _build_optimizer(metric_fn, rounds: int):
    try:
        from dspy.teleprompt import BootstrapFewShot  # type: ignore

        try:
            if rounds > 0:
                return BootstrapFewShot(metric=metric_fn, max_rounds=rounds)
            return BootstrapFewShot(metric=metric_fn)
        except Exception:
            return BootstrapFewShot()
    except Exception:
        return None


def _metric(example: Any, prediction: Any, trace: Any = None) -> float:
    try:
        inputs = {k: _get_example_value(example, k) for k in INPUT_KEYS}
        prediction_jsonl = _get_example_value(prediction, "mini_steps_jsonl")
        score, _details = score_prediction(inputs, prediction_jsonl)
        return score
    except Exception:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/datasets/next_steps_structural.jsonl")
    parser.add_argument("--output", default="compiled/next_steps_compiled.jsonl")
    parser.add_argument("--train-ratio", type=float, default=float(os.getenv("DSPY_COMPILE_TRAIN_RATIO") or "0.8"))
    parser.add_argument("--max-examples", type=int, default=int(os.getenv("DSPY_COMPILE_MAX_EXAMPLES") or "10"))
    parser.add_argument("--rounds", type=int, default=int(os.getenv("DSPY_COMPILE_ROUNDS") or "3"))
    parser.add_argument("--archive-dir", help="If set, copies the compiled output into this directory with a timestamp.")
    args = parser.parse_args()

    records = load_jsonl_records(args.dataset)
    if not records:
        print(f"No records found in {args.dataset}")
        return 1
    if args.max_examples > 0:
        records = records[: args.max_examples]

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
        max_tokens=int(os.getenv("DSPY_COMPILE_MAX_TOKENS") or os.getenv("DSPY_NEXT_STEPS_MAX_TOKENS") or "1000"),
        timeout=float(os.getenv("DSPY_LLM_TIMEOUT_SEC") or "20"),
        num_retries=0,
    )
    throttle_sec = float(os.getenv("DSPY_COMPILE_THROTTLE_SEC") or "1.0")
    max_calls = int(os.getenv("DSPY_COMPILE_MAX_CALLS") or "40")
    _configure_dspy(_wrap_throttled_lm(lm, throttle_sec, max_calls))

    examples: List[Any] = []
    for rec in records:
        ex = dspy.Example(**_build_inputs(rec.inputs), **(rec.outputs or {}))
        try:
            ex = ex.with_inputs(*INPUT_KEYS)
        except Exception:
            pass
        examples.append(ex)

    split = int(len(examples) * args.train_ratio)
    trainset = examples[:split]
    devset = examples[split:]

    module = FlowPlannerModule()
    optimizer = _build_optimizer(_metric, args.rounds)
    if optimizer is None:
        print("Optimizer unavailable; falling back to dataset demos")
        compiled_module = module
        compiled_demos = trainset
    else:
        try:
            compiled_module = optimizer.compile(module, trainset=trainset, valset=devset)
        except Exception as e:
            if "call limit" in str(e).lower():
                print(f"Compile stopped: {e}")
                compiled_module = module
            else:
                compiled_module = optimizer.compile(module, trainset=trainset)
        compiled_demos = getattr(compiled_module.prog, "demos", trainset)

    records_out = [_example_to_record(ex) for ex in compiled_demos]
    _write_jsonl(Path(args.output), records_out)
    print(f"Wrote compiled demos to {args.output}")
    if args.archive_dir:
        archived = _archive_file(Path(args.output), Path(args.archive_dir), label="next_steps_compiled")
        print(f"Archived compiled demos to {archived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
