from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class InstanceCategory(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: Optional[str] = None
    name: Optional[str] = None


class InstanceSubcategory(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: Optional[str] = None
    name: Optional[str] = None
    category_id: Optional[str] = Field(default=None, alias="categoryId")


class InstanceContext(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Back-compat single values (deprecated but supported)
    industry: Optional[InstanceCategory] = None
    service: Optional[InstanceSubcategory] = None

    # Preferred multi-value format
    categories: List[InstanceCategory] = Field(default_factory=list)
    subcategories: List[InstanceSubcategory] = Field(default_factory=list)


class NewBatchRequest(BaseModel):
    """
    Canonical request body for `POST /v1/api/form/{instanceId}` (matches the OpenAPI contract).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    session_id: str = Field(alias="sessionId")
    step_data_so_far: Dict[str, Any] = Field(default_factory=dict, alias="stepDataSoFar")
    asked_step_ids: List[str] = Field(default_factory=list, alias="askedStepIds")
    answered_qa: List[Dict[str, Any]] = Field(default_factory=list, alias="answeredQA")
    existing_step_ids: List[str] = Field(default_factory=list, alias="existingStepIds")
    question_step_ids: List[str] = Field(default_factory=list, alias="questionStepIds")
    form_state: Dict[str, Any] = Field(default_factory=dict, alias="formState")
    use_case: Optional[str] = Field(default=None, alias="useCase")
    instance_context: Optional[InstanceContext] = Field(default=None, alias="instanceContext")
    no_cache: Optional[bool] = Field(default=None, alias="noCache")


class FormResponse(BaseModel):
    """
    Response for `POST /v1/api/form/{instanceId}`.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    request_id: str = Field(alias="requestId")
    schema_version: str = Field(default="0", alias="schemaVersion")
    mini_steps: List[Dict[str, Any]] = Field(default_factory=list, alias="miniSteps")
