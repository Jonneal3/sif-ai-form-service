from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class FlowNewBatchRequest(BaseModel):
    """
    Inbound request body for:
      - POST /flow/new-batch
      - POST /flow/new-batch/stream
    """

    model_config = ConfigDict(extra="allow")

    # Contract drift prevention
    schemaVersion: Optional[str] = Field(
        default=None,
        description="Optional client-expected contract/schema version. Service echoes schemaVersion in responses.",
    )

    # Core planner context
    mode: Optional[str] = Field(
        default=None, description="Optional legacy/debug mode flag (planner ignores it)."
    )
    batchId: str = Field(default="ContextCore", description="ContextCore | PersonalGuide")
    platformGoal: str = Field(default="", description="Overall platform goal/purpose")
    businessContext: str = Field(default="", description="Business context for tone/guidance")
    industry: str = Field(default="General", description="Industry label (informational)")
    service: str = Field(default="", description="Service label (informational)")
    groundingPreview: str = Field(default="", description="RAG/DB grounding snippet")

    # State / constraints
    requiredUploads: List[Dict[str, Any]] = Field(default_factory=list)
    personalizationSummary: str = Field(default="")
    stepDataSoFar: Dict[str, Any] = Field(default_factory=dict)
    alreadyAskedKeys: List[str] = Field(default_factory=list)
    formPlan: List[Dict[str, Any]] = Field(default_factory=list)
    batchState: Dict[str, Any] = Field(default_factory=dict)
    allowedMiniTypes: List[str] = Field(default_factory=list)
    maxSteps: int = Field(default=4, ge=1, le=20)


