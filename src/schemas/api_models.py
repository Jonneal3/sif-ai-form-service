from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class SessionInfo(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    session_id: str = Field(default="", alias="sessionId")
    instance_id: str = Field(default="", alias="instanceId")


class CurrentBatch(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    batch_id: str = Field(default="", alias="batchId")
    batch_number: Optional[int] = Field(default=None, alias="batchNumber")
    max_steps: Optional[int] = Field(default=None, alias="maxSteps")
    allowed_component_types: Optional[List[str]] = Field(default=None, alias="allowedComponentTypes")
    max_tokens: Optional[int] = Field(default=None, alias="maxTokens")


class WidgetState(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    answers: Dict[str, Any] = Field(default_factory=dict)
    asked_step_ids: List[str] = Field(default_factory=list, alias="askedStepIds")
    # Widget callers may send either:
    # - a full `formPlan` snapshot (object with batches/constraints), or
    # - a plan item list (array) used by the Next.js middleware to persist form plans.
    form_plan: Optional[Union[Dict[str, Any], List[Any]]] = Field(default=None, alias="formPlan")


class RequestFlags(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    no_cache: Optional[bool] = Field(default=None, alias="noCache")
    schema_version: Optional[str] = Field(default=None, alias="schemaVersion")
    max_tokens: Optional[int] = Field(default=None, alias="maxTokens")
    include_meta: Optional[bool] = Field(default=None, alias="includeMeta")


class FormRequest(BaseModel):
    """
    sif-widget `/api/ai-form/[instanceId]/new-batch` shape used by `POST /v1/api/form`.

    Expected shape:
      { session, currentBatch, state: { answers, askedStepIds, formPlan }, request }
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Widget shape fields
    session: Optional[SessionInfo] = None
    current_batch: CurrentBatch = Field(alias="currentBatch")
    state: WidgetState = Field(default_factory=WidgetState)
    request: Optional[RequestFlags] = None


class FormResponse(BaseModel):
    """
    Response for `POST /v1/api/form`.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    request_id: str = Field(alias="requestId")
    schema_version: str = Field(default="0", alias="schemaVersion")
    mini_steps: List[Dict[str, Any]] = Field(default_factory=list, alias="miniSteps")
