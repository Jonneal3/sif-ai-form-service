from __future__ import annotations

from typing import Any, Dict, List, Optional

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
    form_plan: Optional[Dict[str, Any]] = Field(default=None, alias="formPlan")


class RequestFlags(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    no_cache: Optional[bool] = Field(default=None, alias="noCache")
    schema_version: Optional[str] = Field(default=None, alias="schemaVersion")
    max_tokens: Optional[int] = Field(default=None, alias="maxTokens")
    include_meta: Optional[bool] = Field(default=None, alias="includeMeta")
    include_form_plan: Optional[bool] = Field(default=None, alias="includeFormPlan")


class FormRequest(BaseModel):
    """
    Accepts both the native service shape and the sif-widget new-batch shape.

    Native shape uses `batchId`, `stepDataSoFar`, `alreadyAskedKeys`, optional `batchPolicy`.
    Widget shape uses `{ session, currentBatch, state: { answers, askedStepIds, formPlan }, request }`.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Core planning inputs (native shape)
    batch_id: Optional[str] = Field(default=None, alias="batchId")
    batch_number: Optional[int] = Field(default=None, alias="batchNumber")
    max_steps: Optional[int] = Field(default=None, alias="maxSteps")
    step_data_so_far: Dict[str, Any] = Field(default_factory=dict, alias="stepDataSoFar")
    already_asked_keys: List[str] = Field(default_factory=list, alias="alreadyAskedKeys")
    batch_policy: Optional[Dict[str, Any]] = Field(default=None, alias="batchPolicy")

    # Widget shape fields
    session: Optional[SessionInfo] = None
    current_batch: Optional[CurrentBatch] = Field(default=None, alias="currentBatch")
    state: Optional[WidgetState] = None
    request: Optional[RequestFlags] = None


class FormResponse(BaseModel):
    """
    Response for `POST /v1/api/form`.

    Notes:
    - `formPlan` may be returned to bootstrap the session when the request is a "new batch"
      request and no plan was provided by the client.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    request_id: str = Field(alias="requestId")
    schema_version: str = Field(default="0", alias="schemaVersion")
    mini_steps: List[Dict[str, Any]] = Field(default_factory=list, alias="miniSteps")

    form_plan: Optional[Dict[str, Any]] = Field(default=None, alias="formPlan")

