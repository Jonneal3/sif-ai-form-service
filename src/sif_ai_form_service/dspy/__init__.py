"""
DSPy module wrappers (program wiring only).

These classes are thin wrappers around DSPy Predict/Chain-of-thought programs and
should not contain request/response glue (that lives in `app.pipeline.*`).
"""

from sif_ai_form_service.dspy.batch_generator_module import BatchGeneratorModule
from sif_ai_form_service.dspy.flow_planner_module import FlowPlannerModule
from sif_ai_form_service.dspy.image_prompt_module import ImagePromptModule
from sif_ai_form_service.dspy.must_have_copy_module import MustHaveCopyModule

__all__ = [
    "BatchGeneratorModule",
    "FlowPlannerModule",
    "ImagePromptModule",
    "MustHaveCopyModule",
]
