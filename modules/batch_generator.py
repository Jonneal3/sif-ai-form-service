"""
Batch Generator Module - Mini-schema batch generation layer

Generates ALL MiniSteps for a given batch in ONE DSPy call.

This module intentionally outputs a compact JSON array of MiniSteps (abstract schema),
which the Node/TS layer maps deterministically to renderable StepDefinitions.

---

### DSPy beginner notes (what to look for here)

- `class BatchGenerator(dspy.Module)`: a DSPy “program” (like a tiny model-powered function).
- `self.prog = dspy.Predict(BatchGeneratorJSON)`: ties the program to a Signature contract.
- `forward(...)`: like PyTorch, the method DSPy calls when you run the module.

Why we validate:
- LLM output can be malformed. We parse JSON and then validate each step via Pydantic
  (`TextInputMini`, `MultipleChoiceMini`, etc.). Invalid items are dropped deterministically.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import dspy  # type: ignore

from modules.signatures import BatchGeneratorJSON
from modules.signatures.flow_signatures import (
    FileUploadMini,
    MultipleChoiceMini,
    RatingMini,
    TextInputMini,
)


def _is_debug_mode() -> bool:
    return os.getenv("AI_FORM_DEBUG") == "true" or os.getenv("NODE_ENV") == "development"


def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    t = s.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t, flags=re.IGNORECASE)
    return t.strip()


def _safe_json_loads_array(text: str) -> Optional[List[Any]]:
    """
    Best-effort: parse a JSON array from text.
    """
    if not text:
        return None
    t = _strip_code_fences(text)
    try:
        parsed = json.loads(t)
        return parsed if isinstance(parsed, list) else None
    except Exception:
        pass
    # Fallback: extract first [...] block
    m = re.search(r"\[[\s\S]*\]", t)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, list) else None
    except Exception:
        return None


def _validate_and_normalize_mini_step(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Validate and normalize a MiniStep using Pydantic models.
    Returns a dict suitable for JSON transport, or None if invalid.
    """
    if not isinstance(obj, dict):
        return None
    t = obj.get("type")
    try:
        if t == "text_input":
            m = TextInputMini.model_validate(obj)
        elif t == "multiple_choice":
            m = MultipleChoiceMini.model_validate(obj)
        elif t == "rating":
            m = RatingMini.model_validate(obj)
        elif t == "file_upload":
            m = FileUploadMini.model_validate(obj)
        else:
            return None
        return m.model_dump()
    except Exception:
        return None


class BatchGenerator(dspy.Module):
    """
    DSPy Module: Generate a list of MiniSteps for a batch in one request.
    """

    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(BatchGeneratorJSON)

    def forward(
        self,
        *,
        batch_id: str,
        industry: str,
        service: str,
        items: List[Dict[str, Any]],
        allowed_mini_types: List[str],
        max_steps: int,
        max_tokens_hint: str = "1500-3000",
        already_asked_keys: List[str] | None = None,
        personalization_summary: str = "",
        grounding_preview: str = "",
    ) -> dspy.Prediction:
        request_id = f"batch_generator_{int(time.time() * 1000)}_{id(self)}"
        t0 = time.time()

        already = already_asked_keys or []
        allowed_csv = ",".join([str(x).strip() for x in allowed_mini_types if str(x).strip()])

        # Deterministic step skeleton: ids are fixed by the caller.
        skeleton_items = items if isinstance(items, list) else []
        # Hard cap skeleton to max_steps deterministically.
        skeleton_items = skeleton_items[: int(max_steps)]
        expected_ids: List[str] = []
        for it in skeleton_items:
            if isinstance(it, dict) and isinstance(it.get("id"), str) and it["id"].strip():
                expected_ids.append(it["id"].strip())
        expected_id_set = set(expected_ids)

        if _is_debug_mode():
            sys.stderr.write(
                f"[BatchGenerator] [{request_id}] batch_id={batch_id} industry={industry} service={service} max_steps={max_steps} allowed={allowed_csv}\n"
            )

        out = self.prog(
            batch_id=str(batch_id)[:40],
            industry=str(industry or "General")[:80],
            service=str(service or "")[:80],
            items_json=json.dumps(skeleton_items)[:2600],
            allowed_mini_types=allowed_csv[:200],
            max_steps=str(int(max_steps)),
            max_tokens_hint=str(max_tokens_hint)[:40],
            already_asked_keys_json=json.dumps(already)[:1200],
            personalization_summary=str(personalization_summary or "")[:1200],
            grounding_preview=str(grounding_preview or "")[:2000],
        )

        latency_ms = int((time.time() - t0) * 1000)
        lm_usage = None
        try:
            get_usage = getattr(out, "get_lm_usage", None)
            if callable(get_usage):
                lm_usage = get_usage()
        except Exception:
            lm_usage = None

        raw = getattr(out, "mini_steps_json", None) or ""
        parsed = _safe_json_loads_array(str(raw))
        if not isinstance(parsed, list):
            return dspy.Prediction(
                error="BatchGenerator returned invalid JSON array",
                rawPreview=str(raw)[:1500],
                requestId=request_id,
                latencyMs=latency_ms,
                lmUsage=lm_usage,
            )

        # Enforce max_steps deterministically.
        parsed = parsed[: int(max_steps)]

        validated: List[Dict[str, Any]] = []
        for obj in parsed:
            v = _validate_and_normalize_mini_step(obj)
            if not v:
                continue
            # Enforce allowed types deterministically.
            if v.get("type") not in set(allowed_mini_types):
                continue
            # Enforce deterministic ids: must match skeleton set when provided.
            if expected_id_set and v.get("id") not in expected_id_set:
                continue
            # Tighten option limits deterministically.
            if v.get("type") == "multiple_choice":
                opts = v.get("options") if isinstance(v.get("options"), list) else []
                v["options"] = opts[:10]
            validated.append(v)

        if len(validated) == 0:
            return dspy.Prediction(
                error="BatchGenerator returned no valid mini steps",
                rawPreview=str(raw)[:1500],
                requestId=request_id,
                latencyMs=latency_ms,
                lmUsage=lm_usage,
            )

        # Reorder output to match expected id order (deterministic).
        if expected_ids:
            by_id = {s.get("id"): s for s in validated if isinstance(s, dict)}
            ordered = [by_id.get(i) for i in expected_ids]
            validated = [s for s in ordered if isinstance(s, dict)]

        return dspy.Prediction(
            miniSteps=validated,
            requestId=request_id,
            latencyMs=latency_ms,
            lmUsage=lm_usage,
        )



