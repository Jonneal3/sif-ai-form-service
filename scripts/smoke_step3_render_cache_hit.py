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
    os.environ.setdefault("DSPY_RENDERER_MODEL", "llama-3.1-8b-instant")

    # Ensure meta is included so we can assert cache-hit flags.
    os.environ.setdefault("AI_FORM_INCLUDE_META", "true")
    os.environ.setdefault("AI_FORM_RENDER_CACHE", "true")
    os.environ.setdefault("AI_FORM_RENDER_CACHE_TTL_SEC", "600")
    os.environ.setdefault("AI_FORM_DEBUG", "true")


def _build_cached_steps() -> list[dict]:
    return [
        {"id": "step-upload-photo", "type": "file_upload", "question": "Upload a photo.", "required": True},
        {"id": "step-gallery", "type": "gallery", "question": "Review your images.", "required": False},
        {"id": "step-confirmation", "type": "confirmation", "question": "All set. Submit when ready.", "required": False},
    ]


def main() -> int:
    _seed_env()

    from programs.form_pipeline.allowed_types import ensure_allowed_mini_types, extract_allowed_mini_types_from_payload
    from programs.form_pipeline.allowed_types import prefer_structured_allowed_mini_types
    from programs.form_pipeline.context_builder import build_context
    from programs.form_pipeline.planning import apply_flow_guide, build_deterministic_suffix_plan_items
    from programs.form_pipeline.validation import _validate_mini
    from programs.form_pipeline.orchestrator import (
        _RENDER_OUTPUT_CACHE,
        _PLANNER_PLAN_CACHE,
        _planner_cache_key,
        _render_cache_key,
        _select_ui_types,
        _short_hash,
        _ttl_cache_set,
        _extract_plan_items,
        _resolve_max_plan_items,
        _normalize_plan_key,
        _derive_step_id_from_key,
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
        # Force single-batch behavior so suffix reservation is stable.
        "maxStepsThisCall": 6,
        "requiredUploads": [{"stepId": "step-upload-photo"}],
        # Keep everything else minimal.
        "answeredQA": [],
        "askedStepIds": [],
    }

    # Build the same context + flow guide the orchestrator uses.
    ctx = build_context(payload)
    allowed_mini_types = ensure_allowed_mini_types(extract_allowed_mini_types_from_payload(payload))
    ctx, allowed_mini_types, max_steps = apply_flow_guide(
        payload=payload,
        context=ctx,
        batch_number=1,
        extracted_allowed_mini_types=allowed_mini_types,
        extracted_max_steps=int(payload.get("maxStepsThisCall") or 0),
    )
    if ctx.get("prefer_structured_inputs"):
        allowed_mini_types = prefer_structured_allowed_mini_types(allowed_mini_types)

    asked_ids = set([str(x).strip() for x in (ctx.get("asked_step_ids") or []) if str(x).strip()])
    use_case_key = str(ctx.get("use_case") or "").strip().lower() or "none"
    services_hash = _short_hash(str(ctx.get("services_summary") or ""), n=10)

    # Seed planner cache with a minimal plan.
    plan_items = [
        {"key": "kitchen_size", "question": "How big is the kitchen?"},
        {"key": "layout_changes", "question": "Any layout changes planned?"},
    ]
    raw_plan = json.dumps({"plan": plan_items}, separators=(",", ":"), sort_keys=True)
    pkey = _planner_cache_key(session_id=payload["sessionId"], services_fingerprint=services_hash, use_case_key=use_case_key)
    # Directly insert with long TTL (format matches orchestrator's cache).
    _PLANNER_PLAN_CACHE[pkey] = (10**12, raw_plan)

    # Reproduce the orchestrator's sliced plan to compute the render cache key.
    full_plan_items = _extract_plan_items(raw_plan, max_items=int(_resolve_max_plan_items(ctx)), asked_step_ids=asked_ids)
    suffix_plan_items = build_deterministic_suffix_plan_items(context=ctx)

    # Single-batch → reserve suffix room.
    reserved = len([x for x in suffix_plan_items if isinstance(x, dict)])
    n_planner = max(0, int(max_steps) - int(reserved))
    plan_sequence = list(full_plan_items)[:n_planner] + list(suffix_plan_items)

    merged: list[dict] = []
    seen: set[str] = set()
    for item in plan_sequence:
        if not isinstance(item, dict):
            continue
        key = _normalize_plan_key(item.get("key"))
        if not key or key in seen:
            continue
        sid = _derive_step_id_from_key(key)
        if sid in asked_ids:
            continue
        obj = dict(item)
        obj["key"] = key
        merged.append(obj)
        seen.add(key)

    sliced: list[dict] = []
    for item in merged:
        key = _normalize_plan_key(item.get("key"))
        if not key:
            continue
        sid = _derive_step_id_from_key(key)
        if sid in asked_ids:
            continue
        sliced.append(item)
        if len(sliced) >= int(max_steps):
            break

    forced_types = set()
    for item in sliced:
        t = str(item.get("type_hint") or "").strip().lower()
        if t:
            forced_types.add(t)
    if forced_types:
        allowed_mini_types = sorted(set([str(x).strip().lower() for x in allowed_mini_types if str(x).strip()]) | forced_types)

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

    _ttl_cache_set(_RENDER_OUTPUT_CACHE, rkey, validated, ttl_sec=600)

    # Execute the pipeline: it should hit both caches and return our cached steps.
    resp = next_steps_jsonl(payload)
    assert isinstance(resp, dict) and resp.get("ok") is not False
    steps = resp.get("miniSteps")
    assert isinstance(steps, list) and len(steps) >= 3
    ids = [str(s.get("id") or "") for s in steps if isinstance(s, dict)]
    assert "step-upload-photo" in ids
    assert "step-gallery" in ids
    assert "step-confirmation" in ids

    dbg = (resp.get("debugContext") or {}) if isinstance(resp.get("debugContext"), dict) else {}
    assert dbg.get("plannerCacheHit") is True
    assert dbg.get("renderCacheHit") is True

    print("OK: renderer cache hit smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

