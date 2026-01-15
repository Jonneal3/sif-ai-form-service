"""
DSPy module wrappers (program wiring only).

These classes are thin wrappers around DSPy Predict/Chain-of-thought programs and
should not contain runtime request/response glue (that lives in flow_planner.py,
image_planner.py, etc.).
"""

from app.dspy.batch_generator_module import BatchGeneratorModule
from app.dspy.flow_planner_module import FlowPlannerModule
from app.dspy.form_planner_module import FormPlannerModule
from app.dspy.image_prompt_module import ImagePromptModule
from app.dspy.must_have_copy_module import MustHaveCopyModule

__all__ = [
    "BatchGeneratorModule",
    "FlowPlannerModule",
    "FormPlannerModule",
    "ImagePromptModule",
    "MustHaveCopyModule",
]
