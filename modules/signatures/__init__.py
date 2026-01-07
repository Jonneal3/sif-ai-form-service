"""
DSPy signatures for this service.

We keep the same structure as `sif-widget/dspy/modules/signatures` so we can reuse the planner code
without rewriting imports.
"""

from .flow_signatures import BatchGeneratorJSON

__all__ = ["BatchGeneratorJSON"]


