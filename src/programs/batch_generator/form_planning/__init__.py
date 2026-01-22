"""
Deterministic (non-LLM) helpers for form planning and UI composition.

These modules are intentionally lightweight so the batch generator can:
- Build a deterministic per-batch `form_plan` prompt scaffold when available.
- Optionally emit deterministic UI placements (uploads, CTAs, etc).
- Optionally wrap uploads into a composite step without consuming LLM output.
"""

