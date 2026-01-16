"""
DSPy module wrappers (program wiring only).

These classes are thin wrappers around DSPy Predict/Chain-of-thought programs and
should not contain request/response glue (that lives in `app.pipeline.*`).
"""

from app.dspy.batch_generator_module import BatchGeneratorModule
from app.dspy.flow_plan_module import FlowPlanModule
from app.dspy.flow_planner_module import FlowPlannerModule
from app.dspy.image_prompt_module import ImagePromptModule
from app.dspy.must_have_copy_module import MustHaveCopyModule

__all__ = [
    "BatchGeneratorModule",
    "FlowPlanModule",
    "FlowPlannerModule",
    "ImagePromptModule",
    "MustHaveCopyModule",
]
