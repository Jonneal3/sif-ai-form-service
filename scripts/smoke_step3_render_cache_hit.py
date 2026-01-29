#!/usr/bin/env python3
from __future__ import annotations

"""
Smoke checks for Step 3 renderer cache hit behavior.

This script avoids any real LLM calls by:
- seeding the planner cache with a plan JSON
- seeding the renderer cache with validated miniSteps[]

Run:
  PYTHONPATH=.:src python3 scripts/smoke_step3_render_cache_hit.py
"""

import json
import os


def _seed_env() -> None:
    # Satisfy provider config checks (no real network calls should happen).
    os.environ.setdefault("DSPY_PROVIDER", "groq")
    os.environ.setdefault("GROQ_API_KEY", "test_key")
    os.environ.setdefault("DSPY_PLANNER_MODEL_LOCK", "llama-3.3-70b-versatile")

    # Ensure meta is included so we can assert cache-hit flags.
    os.environ.setdefault("AI_FORM_INCLUDE_META", "true")
    os.environ.setdefault("AI_FORM_RENDER_CACHE", "true")
    os.environ.setdefault("AI_FORM_RENDER_CACHE_TTL_SEC", "600")
    os.environ.setdefault("AI_FORM_DEBUG", "true")


def _build_cached_steps() -> list[dict]:
    return [
        {
            "id": "step-kitchen-size",
            "type": "multiple_choice",
            "question": "How big is the kitchen?",
            "options": [
                {"label": "Small", "value": "small"},
                {"label": "Medium", "value": "medium"},
                {"label": "Large", "value": "large"},
                {"label": "Not sure yet", "value": "not_sure_yet"},
            ],
        },
        {
            "id": "step-layout-changes",
            "type": "multiple_choice",
            "question": "Any layout changes planned?",
            "options": [
                {"label": "No changes", "value": "no_changes"},
                {"label": "Minor changes", "value": "minor_changes"},
                {"label": "Major changes", "value": "major_changes"},
                {"label": "Not sure yet", "value": "not_sure_yet"},
            ],
        },
    ]


def main() -> int:
    _seed_env()

    from programs.form_pipeline.allowed_types import ensure_allowed_mini_types, extract_allowed_mini_types_from_payload
    from programs.form_pipeline.allowed_types import prefer_structured_allowed_mini_types
    from programs.form_pipeline.context_builder import build_context
    from programs.common.hashing import short_hash
    from programs.common.ttl_cache import ttl_cache_set
    from programs.question_planner.plan_parsing import derive_step_id_from_key, extract_plan_items, normalize_plan_key
    from programs.question_planner.renderer.validation import _validate_mini
    from programs.form_pipeline.orchestrator import (
        _RENDER_OUTPUT_CACHE,
        _PLANNER_PLAN_CACHE,
        _planner_cache_key,
        _render_cache_key,
        _select_ui_types,
        _resolve_max_plan_items,
        next_steps_jsonl,
        _compact_json,
    )

    # Clear caches for determinism.
    _PLANNER_PLAN_CACHE.clear()
    _RENDER_OUTPUT_CACHE.clear()

    payload: dict = {
        "schemaVersion": "dev",
        "sessionId": "sess_cache_hit",
        "useCase": "scene",
        "servicesSummary": "Industry: Interior Design. Service: Kitchen remodel. Client wants modern finishes.",
        # Make render_context deterministic (avoids random cache-key drift).
        "choiceOptionMin": 4,
        "choiceOptionMax": 6,
        "choiceOptionTarget": 5,
        "currentBatch": {"batchNumber": 1},
        # Keep everything else minimal.
        "answeredQA": [],
        "askedStepIds": [],
    }

    # Build the same context + defaults the orchestrator uses.
    ctx = build_context(payload)
    allowed_mini_types = ensure_allowed_mini_types(extract_allowed_mini_types_from_payload(payload))
    # Match orchestrator: prefer an explicit cap, otherwise use batch_constraints defaults.
    try:
        max_steps = int(payload.get("maxStepsThisCall") or 0)
    except Exception:
        max_steps = 0
    if max_steps <= 0:
        constraints = ctx.get("batch_constraints") if isinstance(ctx.get("batch_constraints"), dict) else {}
        try:
            default_steps = int(constraints.get("defaultStepsPerBatch") or 0)
        except Exception:
            default_steps = 0
        try:
            min_steps = int(constraints.get("minStepsPerBatch") or 0)
        except Exception:
            min_steps = 0
        try:
            max_steps_per_batch = int(constraints.get("maxStepsPerBatch") or 0)
        except Exception:
            max_steps_per_batch = 0
        if default_steps <= 0:
            default_steps = 5
        if min_steps <= 0:
            min_steps = 2
        if max_steps_per_batch <= 0:
            max_steps_per_batch = max(min_steps, 6)
        max_steps = max(min_steps, min(default_steps, max_steps_per_batch))
    if ctx.get("prefer_structured_inputs"):
        allowed_mini_types = prefer_structured_allowed_mini_types(allowed_mini_types)

    asked_ids = set([str(x).strip() for x in (ctx.get("asked_step_ids") or []) if str(x).strip()])
    services_hash = short_hash(str(ctx.get("services_summary") or ""), n=10)

    # Seed planner cache with a minimal plan.
    plan_items = [
        {"key": "kitchen_size", "question": "How big is the kitchen?"},
        {"key": "layout_changes", "question": "Any layout changes planned?"},
    ]
    raw_plan = json.dumps({"plan": plan_items}, separators=(",", ":"), sort_keys=True)
    pkey = _planner_cache_key(session_id=payload["sessionId"], services_fingerprint=services_hash)
    # Directly insert with long TTL (format matches orchestrator's cache).
    _PLANNER_PLAN_CACHE[pkey] = (10**12, raw_plan)

    # Reproduce the orchestrator's sliced plan to compute the render cache key.
    full_plan_items = extract_plan_items(raw_plan, max_items=int(_resolve_max_plan_items(ctx)), asked_step_ids=set())
    plan_sequence = list(full_plan_items)

    merged: list[dict] = []
    seen: set[str] = set()
    for item in plan_sequence:
        if not isinstance(item, dict):
            continue
        key = normalize_plan_key(item.get("key"))
        if not key or key in seen:
            continue
        sid = derive_step_id_from_key(key)
        if sid in asked_ids:
            continue
        obj = dict(item)
        obj["key"] = key
        merged.append(obj)
        seen.add(key)

    sliced: list[dict] = []
    for item in merged:
        key = normalize_plan_key(item.get("key"))
        if not key:
            continue
        sid = derive_step_id_from_key(key)
        if sid in asked_ids:
            continue
        sliced.append(item)
        if len(sliced) >= int(max_steps):
            break

    plan_json_for_render = _compact_json({"plan": sliced})
    render_context_json = _compact_json(
        {
            "services_summary": str(ctx.get("services_summary") or "").strip(),
            "choice_option_min": ctx.get("choice_option_min"),
            "choice_option_max": ctx.get("choice_option_max"),
            "choice_option_target": ctx.get("choice_option_target"),
            "required_uploads": ctx.get("required_uploads") if isinstance(ctx.get("required_uploads"), list) else [],
        }
    )
    rkey = _render_cache_key(
        session_id=payload["sessionId"],
        schema_version=str(payload.get("schemaVersion") or "0"),
        plan_json=plan_json_for_render,
        render_context_json=render_context_json,
        allowed_mini_types=allowed_mini_types,
    )

    # Validate cached steps against the service's schema gate before caching.
    ui_types = _select_ui_types()
    validated: list[dict] = []
    for step in _build_cached_steps():
        out = _validate_mini(step, ui_types)
        assert out is not None
        validated.append(out)

    ttl_cache_set(_RENDER_OUTPUT_CACHE, rkey, validated, ttl_sec=600)

    # Execute the pipeline: it should hit both caches and return our cached steps.
    resp = next_steps_jsonl(payload)
    assert isinstance(resp, dict) and resp.get("ok") is not False
    steps = resp.get("miniSteps")
    assert isinstance(steps, list) and len(steps) >= 2
    ids = [str(s.get("id") or "") for s in steps if isinstance(s, dict)]
    assert "step-kitchen-size" in ids
    assert "step-layout-changes" in ids

    dbg = (resp.get("debugContext") or {}) if isinstance(resp.get("debugContext"), dict) else {}
    assert dbg.get("plannerCacheHit") is True
    assert dbg.get("renderCacheHit") is True

    print("OK: render cache hit smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
