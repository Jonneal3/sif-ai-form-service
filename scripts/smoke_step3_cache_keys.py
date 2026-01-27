#!/usr/bin/env python3
from __future__ import annotations

"""
Smoke checks for Step 3 cache-key stability.

Run:
  PYTHONPATH=.:src python3 scripts/smoke_step3_cache_keys.py
"""


def main() -> int:
    from programs.form_pipeline.orchestrator import _planner_cache_key, _render_cache_key

    # Planner cache key: stable and empty without session id.
    assert _planner_cache_key(session_id="", services_fingerprint="abc", use_case_key="scene") == ""
    assert (
        _planner_cache_key(session_id="sess_1", services_fingerprint="abc", use_case_key="scene")
        == "question_plan:sess_1:abc:scene"
    )
    assert (
        _planner_cache_key(session_id="sess_1", services_fingerprint="abc", use_case_key="Scene")
        == "question_plan:sess_1:abc:scene"
    )

    # Renderer cache key: allowed types ordering is normalized.
    k1 = _render_cache_key(
        session_id="sess_1",
        schema_version="dev",
        plan_json='{"plan":[{"key":"a"}]}',
        render_context_json='{"services_summary":"x"}',
        allowed_mini_types=["multiple_choice", "slider"],
    )
    k2 = _render_cache_key(
        session_id="sess_1",
        schema_version="dev",
        plan_json='{"plan":[{"key":"a"}]}',
        render_context_json='{"services_summary":"x"}',
        allowed_mini_types=["slider", "multiple_choice"],
    )
    assert k1 == k2

    # Schema version should affect key (avoid cross-schema collisions).
    k3 = _render_cache_key(
        session_id="sess_1",
        schema_version="prod",
        plan_json='{"plan":[{"key":"a"}]}',
        render_context_json='{"services_summary":"x"}',
        allowed_mini_types=["multiple_choice", "slider"],
    )
    assert k1 != k3

    print("OK: cache-key smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

