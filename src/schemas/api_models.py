from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_strings(cls, data: Any) -> Any:
        """
        Accept legacy/widget shapes where `industry` / `service` may be plain strings.

        Widget context often looks like:
          { industry: "Landscaping", industryId: "...", industryName: "Landscaping", ... }
        """
        if not isinstance(data, dict):
            return data
        out = dict(data)

        # industry: "Landscaping" -> { id?, name }
        raw_industry = out.get("industry")
        if isinstance(raw_industry, str):
            name = (out.get("industryName") or out.get("categoryName") or raw_industry).strip()
            iid = str(out.get("industryId") or "").strip() or None
            out["industry"] = {"id": iid, "name": name}
        elif raw_industry is None and (out.get("industryId") or out.get("industryName") or out.get("categoryName")):
            name = str(out.get("industryName") or out.get("categoryName") or "").strip() or None
            iid = str(out.get("industryId") or "").strip() or None
            if name or iid:
                out["industry"] = {"id": iid, "name": name}

        # service: "Pool Design" -> { id?, name, categoryId? }
        raw_service = out.get("service")
        if isinstance(raw_service, str):
            name = (out.get("serviceName") or out.get("subcategoryName") or raw_service).strip()
            sid = str(out.get("serviceId") or out.get("subcategoryId") or "").strip() or None
            cid = str(out.get("industryId") or out.get("categoryId") or "").strip() or None
            out["service"] = {"id": sid, "name": name, "categoryId": cid}
        elif raw_service is None and (
            out.get("serviceId") or out.get("serviceName") or out.get("subcategoryId") or out.get("subcategoryName")
        ):
            name = str(out.get("serviceName") or out.get("subcategoryName") or "").strip() or None
            sid = str(out.get("serviceId") or out.get("subcategoryId") or "").strip() or None
            cid = str(out.get("industryId") or out.get("categoryId") or "").strip() or None
            if name or sid:
                out["service"] = {"id": sid, "name": name, "categoryId": cid}

        return out


class SessionEnvelope(BaseModel):
    """
    Widget/back-compat shape: some clients send `{ session: { sessionId, instanceId } }`
    instead of a top-level `sessionId`.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    session_id: Optional[str] = Field(default=None, alias="sessionId")
    instance_id: Optional[str] = Field(default=None, alias="instanceId")


class NewBatchRequest(BaseModel):
    """
    Canonical request body for `POST /v1/api/form/{instanceId}` (matches the OpenAPI contract).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Accept either:
    # - top-level `sessionId` (OpenAPI canonical)
    # - nested `session.sessionId` (widget/back-compat)
    session_id: Optional[str] = Field(default=None, alias="sessionId")
    session: Optional[SessionEnvelope] = None
    step_data_so_far: Dict[str, Any] = Field(default_factory=dict, alias="stepDataSoFar")
    asked_step_ids: List[str] = Field(default_factory=list, alias="askedStepIds")
    answered_qa: List[Dict[str, Any]] = Field(default_factory=list, alias="answeredQA")
    existing_step_ids: List[str] = Field(default_factory=list, alias="existingStepIds")
    question_step_ids: List[str] = Field(default_factory=list, alias="questionStepIds")
    form_state: Dict[str, Any] = Field(default_factory=dict, alias="formState")
    use_case: Optional[str] = Field(default=None, alias="useCase")
    instance_context: Optional[InstanceContext] = Field(default=None, alias="instanceContext")
    # Frontend-provided summaries (sources of truth)
    service_summary: Optional[str] = Field(default=None, alias="serviceSummary")
    company_summary: Optional[str] = Field(default=None, alias="companySummary")
    no_cache: Optional[bool] = Field(default=None, alias="noCache")

    @model_validator(mode="before")
    @classmethod
    def _accept_widget_state_shape(cls, data: Any) -> Any:
        """
        Back-compat adapter so the API accepts the widget's shape:
          { session: {...}, state: { answers, askedStepIds, answeredQA, context } }
        """
        if not isinstance(data, dict):
            return data
        out = dict(data)

        # Some proxies/wrappers (e.g. Next.js API routes) forward the payload as:
        #   { body: { ...canonical fields... }, ... }
        # Accept this by shallow-merging `body` into the root and dropping the wrapper.
        body = out.get("body")
        if isinstance(body, dict):
            for k, v in body.items():
                # Prefer explicit top-level values when present, but fill in missing/empty ones.
                if k not in out or out.get(k) in (None, "", {}, []):
                    out[k] = v
            out.pop("body", None)

        # session.sessionId -> sessionId
        if "sessionId" not in out and "session_id" not in out:
            sess = out.get("session")
            if isinstance(sess, dict):
                sid = sess.get("sessionId") or sess.get("session_id")
                if sid:
                    out["sessionId"] = sid

        state = out.get("state")
        if isinstance(state, dict):
            # state.answers -> stepDataSoFar
            if "stepDataSoFar" not in out and "step_data_so_far" not in out:
                answers = state.get("answers")
                if isinstance(answers, dict):
                    out["stepDataSoFar"] = answers

            # state.askedStepIds -> askedStepIds
            if "askedStepIds" not in out and "asked_step_ids" not in out:
                asked = state.get("askedStepIds") or state.get("asked_step_ids")
                if isinstance(asked, list):
                    out["askedStepIds"] = asked

            # state.answeredQA -> answeredQA
            if "answeredQA" not in out and "answered_qa" not in out:
                aqa = state.get("answeredQA") or state.get("answered_qa")
                if isinstance(aqa, list):
                    out["answeredQA"] = aqa

            # state.context -> instanceContext
            if "instanceContext" not in out and "instance_context" not in out:
                ctx = state.get("context")
                if isinstance(ctx, dict):
                    out["instanceContext"] = ctx

        return out


class FormResponse(BaseModel):
    """
    Response for `POST /v1/api/form/{instanceId}`.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    request_id: str = Field(alias="requestId")
    schema_version: str = Field(default="0", alias="schemaVersion")
    mini_steps: List[Dict[str, Any]] = Field(default_factory=list, alias="miniSteps")


class ExecuteFunctionRequest(BaseModel):
    """
    Request body for function execution (e.g. generate initial image mid-flow).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    session_id: str = Field(default="", alias="sessionId")
    step_data: Dict[str, Any] = Field(default_factory=dict, alias="stepData")
    function_name: str = Field(default="", alias="functionName")
