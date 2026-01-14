from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FormContext(BaseModel):
    """Platform and business context for the form generation."""
    
    platform_goal: str = Field(
        default="",
        description="Overall platform purpose: what the form is trying to achieve (e.g., 'AI pre-design intake')"
    )
    business_context: str = Field(
        default="",
        description="Business-specific context for tone and guidance (e.g., 'We generate AI images for pre-design')"
    )
    industry: str = Field(
        default="General",
        description="Industry/vertical label (informational, e.g., 'Kitchen Remodeling', 'Interior Design')"
    )
    service: str = Field(
        default="",
        description="Service/subcategory label (informational, e.g., 'Cabinet Design', 'Color Consultation')"
    )


class FormState(BaseModel):
    """Current state of the form session."""
    
    batch_id: str = Field(
        default="ContextCore",
        description="Batch identifier: 'ContextCore' (initial broad questions) or 'PersonalGuide' (follow-up questions)"
    )
    answers: Dict[str, Any] = Field(
        default_factory=dict,
        description="User's answers collected so far (key-value pairs where keys are step IDs)"
    )
    asked_step_ids: List[str] = Field(
        default_factory=list,
        description="List of step IDs that have already been asked (to avoid duplicates)"
    )
    form_plan: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Optional pre-planned form structure (FormPlanItem[]). If empty, planner generates the plan."
    )
    personalization_summary: str = Field(
        default="",
        description="Summary of user responses so far (used for Batch2 personalization)"
    )


class FormConstraints(BaseModel):
    """Constraints and limits for form generation."""
    
    max_steps: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum number of steps to generate in this batch"
    )
    allowed_step_types: List[str] = Field(
        default_factory=list,
        description="Allowed step types (e.g., ['multiple_choice', 'text_input', 'rating']). Empty = all types allowed."
    )
    required_uploads: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Required file uploads with roles (e.g., [{'stepId': 'step-1', 'role': 'sceneImage'}])"
    )


class FormBatchState(BaseModel):
    """Internal batch state for tracking progress."""
    
    calls_used: int = Field(default=0, description="Number of API calls used so far")
    max_calls: int = Field(default=2, description="Maximum allowed API calls")
    calls_remaining: int = Field(default=2, description="Remaining API calls")
    tokens_total_budget: int = Field(default=0, description="Total token budget across the session (0 = unlimited)")
    tokens_used_so_far: int = Field(default=0, description="Tokens used so far across calls")
    satiety_so_far: float = Field(default=0.0, description="Information satiety accumulated so far")
    satiety_remaining: float = Field(default=1.0, description="Remaining information satiety needed")
    missing_high_impact_keys: List[str] = Field(
        default_factory=list,
        description="High-impact question keys that haven't been asked yet"
    )
    must_have_copy_needed: bool = Field(
        default=False,
        description="Whether must-have copy generation is needed"
    )


class SessionInfo(BaseModel):
    """Session and instance identifiers."""
    model_config = ConfigDict(populate_by_name=True)
    session_id: str = Field(..., alias="sessionId", description="Session identifier")
    instance_id: str = Field(..., alias="instanceId", description="Form instance ID")


class TelemetryEvent(BaseModel):
    """Client telemetry event payload."""
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(..., alias="sessionId")
    instance_id: str = Field(..., alias="instanceId")
    event_type: str = Field(..., alias="eventType")
    step_id: Optional[str] = Field(default=None, alias="stepId")
    batch_id: Optional[str] = Field(default=None, alias="batchId")
    model_request_id: Optional[str] = Field(default=None, alias="modelRequestId")
    timestamp: Optional[int] = Field(default=None, alias="timestamp")
    payload: Optional[Dict[str, Any]] = Field(default=None, alias="payload")


class FeedbackEvent(BaseModel):
    """Dev/user feedback annotation payload."""
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(..., alias="sessionId")
    instance_id: str = Field(..., alias="instanceId")
    step_id: Optional[str] = Field(default=None, alias="stepId")
    model_request_id: Optional[str] = Field(default=None, alias="modelRequestId")
    source: Optional[str] = Field(default=None, description="dev | user")
    rating: Optional[int] = Field(default=None, description="1-5 rating (user)")
    vote: Optional[str] = Field(default=None, description="up | down (dev)")
    tags: Optional[List[str]] = Field(default=None)
    comment: Optional[str] = Field(default=None)
    send_to_dataset: Optional[bool] = Field(default=None, alias="sendToDataset")
    timestamp: Optional[int] = Field(default=None, alias="timestamp")
    payload: Optional[Dict[str, Any]] = Field(default=None, alias="payload")


class PromptContextInfo(BaseModel):
    """Everything DSPy needs for prompt generation."""
    model_config = ConfigDict(populate_by_name=True)
    
    # Goal & Purpose
    goal: Optional[str] = Field(default=None, description="Platform goal: what the form is trying to achieve")
    business_context: Optional[str] = Field(default=None, alias="businessContext", description="Business-specific context/tone")
    
    # Vertical Information
    industry: Optional[str] = Field(default=None, description="Industry/vertical label")
    service: Optional[str] = Field(default=None, description="Service/subcategory label")


class FormPsychologyInfo(BaseModel):
    """Form psychology approach for question flow/ordering."""
    model_config = ConfigDict(populate_by_name=True)
    approach: Optional[str] = Field(default=None, description="Psychology approach (e.g., 'Escalation Ladder', 'Progressive Disclosure')")
    description: Optional[str] = Field(default=None, description="Description of the psychology approach")


class FormCopyInfo(BaseModel):
    """Form copy style and principles."""
    model_config = ConfigDict(populate_by_name=True)
    style: Optional[str] = Field(default=None, description="Copy style (e.g., 'Humanism (Ryan Levesque)')")
    principles: Optional[List[str]] = Field(default=None, description="Copy writing principles")
    tone: Optional[str] = Field(default=None, description="Tone guidance (e.g., 'Friendly, helpful, non-intimidating')")


class BatchHistoryItem(BaseModel):
    """
    History of a previous batch (for tracking and analytics).
    
    Note: The aggregated state is in `state.answers` and `state.askedStepIds`.
    This is just for tracking per-batch metrics and can be used for analytics/debugging.
    """
    model_config = ConfigDict(populate_by_name=True)
    batch_id: str = Field(..., alias="batchId", description="Batch identifier: 'ContextCore' or 'PersonalGuide'")
    batch_number: int = Field(..., alias="batchNumber", description="Batch number (1 or 2)")
    steps_generated: int = Field(..., alias="stepsGenerated", description="Number of steps generated in this batch")
    steps_asked: List[str] = Field(..., alias="stepsAsked", description="Step IDs that were asked in this batch (subset of state.askedStepIds)")
    satiety_achieved: float = Field(..., alias="satietyAchieved", description="Satiety score achieved in this batch (0-1)")
    satiety_at_start: float = Field(default=0.0, alias="satietyAtStart", description="Satiety score at the start of this batch (0-1)")
    satiety_gained: float = Field(default=0.0, alias="satietyGained", description="Satiety score gained during this batch (0-1)")
    answers: Dict[str, Any] = Field(
        default_factory=dict,
        description="Answers collected in this batch (subset of state.answers, for tracking only)"
    )


class CurrentBatchInfo(BaseModel):
    """Info about the current batch we're generating."""
    model_config = ConfigDict(populate_by_name=True)
    batch_id: str = Field(..., alias="batchId", description="Current batch: 'ContextCore' or 'PersonalGuide'")
    batch_number: int = Field(..., alias="batchNumber", description="Current batch number (1 or 2)")
    satiety_target: float = Field(default=1.0, alias="satietyTarget", description="Target satiety score for this batch (0-1)")
    satiety_remaining: float = Field(default=1.0, alias="satietyRemaining", description="Remaining satiety needed to reach target (calculated: target - current)")
    max_steps: Optional[int] = Field(default=None, ge=1, le=20, alias="maxSteps", description="Max steps to generate in this batch")
    max_tokens: Optional[int] = Field(default=None, alias="maxTokens", description="Max tokens for LLM response")
    allowed_component_types: Optional[List[str]] = Field(default=None, alias="allowedComponentTypes", description="Allowed component types (e.g., ['choice', 'text', 'slider'])")
    required_uploads: Optional[List[Dict[str, Any]]] = Field(default=None, alias="requiredUploads", description="Required file uploads")


class FormStateInfo(BaseModel):
    """
    Overall form state (aggregated across all batches).
    
    This is the single source of truth for the current state of the form.
    All answers and asked questions are aggregated here, regardless of which batch they came from.
    """
    model_config = ConfigDict(populate_by_name=True)
    
    # Overall Progress (aggregated across all batches)
    satiety_current: float = Field(default=0.0, alias="satietyCurrent", description="Current overall information satiety score (0-1), aggregated from all batches")
    
    # Previously Asked Q&As (aggregated - all batches combined)
    # This is where MCP would look for "what have we already asked?"
    answers: Dict[str, Any] = Field(
        ...,
        description="All user answers across all batches (key: stepId, value: answer). This is the aggregated state."
    )
    asked_step_ids: List[str] = Field(
        ...,
        alias="askedStepIds",
        description="All step IDs asked across all batches (to avoid duplicates). This is the aggregated list."
    )
    
    # Form Plan (AI-generated question plan from batch 1, used in batch 2)
    form_plan: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        alias="formPlan",
        description="AI-generated question plan from batch 1. NOT the list of component types. Contains FormPlanItem[] with planned questions (key, goal, why, component_hint, priority, expected_metric_gain). Used in batch 2 to know which questions to ask. Component types are in currentBatch.allowedComponentTypes."
    )
    personalization_summary: str = Field(
        default="",
        alias="personalizationSummary",
        description="Summary of user responses from previous batches (for batch 2+)"
    )

    # Planner-owned context (created once, then round-tripped)
    batch_policy: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="batchPolicy",
        description="Planner-owned batch policy (maxCalls, maxStepsPerCall, allowedMiniTypes, stop conditions).",
    )
    psychology_plan: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="psychologyPlan",
        description="Planner-owned psychology plan (e.g., escalation ladder stages and rules).",
    )


class RequestFlags(BaseModel):
    """Request flags for debugging and versioning."""
    model_config = ConfigDict(populate_by_name=True)
    no_cache: bool = Field(default=False, alias="noCache", description="Disable LLM cache")
    mode: str = Field(default="next_steps", description="Request mode")
    schema_version: Optional[str] = Field(default=None, alias="schemaVersion", description="Contract version")


class MinimalFormRequest(BaseModel):
    """
    Minimal request body - only requires 3 sections, everything else fetched from Supabase.
    
    **Required:**
    1. session: Identifiers (sessionId, instanceId)
    2. currentBatch: Info about the batch we're generating now (batchId, constraints, targets)
    3. state: Overall form state (aggregated answers, asked questions, satiety, form plan)
    
    **Optional (backend fetches from Supabase if not provided):**
    - prompt: Prompt context (goal, business context, industry)
    - psychology: Form psychology approach (backend uses defaults if not provided)
    - copy: Form copy style and principles (backend uses defaults if not provided)
    - batches: Previous batches history (for analytics only)
    - request: Request flags (debugging, versioning)
    
    Backend automatically fetches from Supabase:
    - Form config (goal, business context, industry, service)
    - Everything else needed for DSPy
    """
    
    model_config = ConfigDict(populate_by_name=True)
    
    # SECTION 1: Session & Instance
    session: SessionInfo = Field(..., description="Session and instance identifiers")
    
    # SECTION 2: Prompt Context (for DSPy)
    prompt: Optional[PromptContextInfo] = Field(
        default=None,
        description="Prompt context for DSPy (goal, business context, industry). Backend fetches from Supabase if not provided."
    )
    
    # SECTION 2b: Form Psychology (optional - backend uses defaults if not provided)
    psychology: Optional[FormPsychologyInfo] = Field(
        default=None,
        description="Form psychology approach (e.g., 'Escalation Ladder'). Backend uses defaults if not provided."
    )
    
    # SECTION 2c: Form Copy (optional - backend uses defaults if not provided)
    copy: Optional[FormCopyInfo] = Field(
        default=None,
        description="Form copy style and principles (e.g., 'Humanism (Ryan Levesque)'). Backend uses defaults if not provided."
    )
    
    # SECTION 3: Current Batch (what we're generating now)
    current_batch: CurrentBatchInfo = Field(
        ...,
        alias="currentBatch",
        description="Info about the current batch we're generating: batchId, batchNumber, constraints, targets"
    )
    
    # SECTION 4: Overall Form State (aggregated)
    state: FormStateInfo = Field(
        ...,
        description="Overall form state: aggregated answers, asked questions, satiety, form plan"
    )
    
    # SECTION 5: Previous Batches History (optional - only needed for analytics/debugging)
    batches: Optional[List[BatchHistoryItem]] = Field(
        default=None,
        description="History of previous batches (optional). Used for analytics/debugging only. State already has aggregated Q&As, so this is redundant but useful for per-batch metrics."
    )
    
    # SECTION 6: Request Flags
    request: Optional[RequestFlags] = Field(
        default=None,
        description="Request flags (debugging, versioning)"
    )
    
    # Helper properties to extract flat values for backward compatibility
    @property
    def session_id(self) -> str:
        return self.session.session_id
    
    @property
    def instance_id(self) -> str:
        return self.session.instance_id
    
    @property
    def batch_id(self) -> str:
        return self.current_batch.batch_id
    
    @property
    def batch_state(self) -> Dict[str, Any]:
        # Construct batch_state from state fields for backward compatibility
        return {
            "callsUsed": 0,  # Will be calculated by backend
            "maxCalls": 2,
            "callsRemaining": 2,
            "tokensTotalBudget": 0,
            "tokensUsedSoFar": 0,
            "satietySoFar": self.state.satiety_current,
            "satietyRemaining": self.current_batch.satiety_remaining,
            "batch1PredictedSatietyIfCompleted": self.current_batch.satiety_target,
            "plannedSatietyGainThisCall": 0,  # Will be calculated
            "plannedSatietyAfterThisCall": 0,  # Will be calculated
            "plannedStepIdsThisCall": [],  # Will be calculated
            "missingHighImpactKeys": [],  # Will be calculated
            "mustHaveCopyNeeded": {
                "budget": False,
                "uploads": [up.get("stepId") for up in (self.current_batch.required_uploads or [])]
            }
        }
    
    @property
    def answers(self) -> Dict[str, Any]:
        return self.state.answers
    
    @property
    def asked_step_ids(self) -> List[str]:
        return self.state.asked_step_ids
    
    @property
    def form_plan(self) -> Optional[List[Dict[str, Any]]]:
        return self.state.form_plan
    
    @property
    def personalization_summary(self) -> str:
        return self.state.personalization_summary
    
    @property
    def platform_goal(self) -> Optional[str]:
        return self.prompt.goal if self.prompt else None
    
    @property
    def business_context(self) -> Optional[str]:
        return self.prompt.business_context if self.prompt else None
    
    @property
    def industry(self) -> Optional[str]:
        return self.prompt.industry if self.prompt else None
    
    @property
    def service(self) -> Optional[str]:
        return self.prompt.service if self.prompt else None
    
    @property
    def psychology_approach(self) -> Optional[str]:
        return self.psychology.approach if self.psychology else None
    
    @property
    def copy_style(self) -> Optional[str]:
        return self.copy.style if self.copy else None
    
    @property
    def max_steps(self) -> Optional[int]:
        return self.current_batch.max_steps
    
    @property
    def allowed_step_types(self) -> Optional[List[str]]:
        return self.current_batch.allowed_component_types
    
    @property
    def required_uploads(self) -> Optional[List[Dict[str, Any]]]:
        return self.current_batch.required_uploads
    
    @property
    def items(self) -> Optional[List[Dict[str, Any]]]:
        return None  # Not in new structure, calculated from formPlan
    
    @property
    def no_cache(self) -> bool:
        return self.request.no_cache if self.request else False
    
    @property
    def schema_version(self) -> Optional[str]:
        return self.request.schema_version if self.request else None


class FormRequest(BaseModel):
    """
    Clean, organized request body for form batch generation.
    
    Accepts both the new structured format and the legacy flat format for backward compatibility.
    """
    
    model_config = ConfigDict(extra="allow")
    
    # Contract versioning
    schema_version: Optional[str] = Field(
        default=None,
        alias="schemaVersion",
        description="Optional client-expected contract/schema version. Service echoes this in responses."
    )
    
    # Organized sections
    context: Optional[FormContext] = Field(
        default=None,
        description="Platform and business context"
    )
    state: Optional[FormState] = Field(
        default=None,
        description="Current form session state"
    )
    constraints: Optional[FormConstraints] = Field(
        default=None,
        description="Generation constraints and limits"
    )
    batch_state: Optional[FormBatchState] = Field(
        default=None,
        alias="batchState",
        description="Internal batch tracking state"
    )
    
    # Legacy flat format support (for backward compatibility)
    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_format(cls, data: Any) -> Any:
        """Convert legacy flat format to structured format."""
        if not isinstance(data, dict):
            return data
        
        # If already structured, return as-is
        if "context" in data or "state" in data or "constraints" in data:
            return data
        
        # Convert from legacy flat format
        normalized = {}
        
        # Schema version
        if "schemaVersion" in data or "schema_version" in data:
            normalized["schema_version"] = data.get("schemaVersion") or data.get("schema_version")
        
        # Context section
        context_data = {}
        if "platformGoal" in data or "platform_goal" in data:
            context_data["platform_goal"] = data.get("platformGoal") or data.get("platform_goal") or ""
        if "businessContext" in data or "business_context" in data:
            context_data["business_context"] = data.get("businessContext") or data.get("business_context") or ""
        if "industry" in data:
            context_data["industry"] = data.get("industry", "General")
        if "service" in data:
            context_data["service"] = data.get("service", "")
        if context_data:
            normalized["context"] = context_data
        
        # State section
        state_data = {}
        if "batchId" in data or "batch_id" in data:
            state_data["batch_id"] = data.get("batchId") or data.get("batch_id") or "ContextCore"
        if "stepDataSoFar" in data or "step_data_so_far" in data or "answers" in data:
            state_data["answers"] = data.get("stepDataSoFar") or data.get("step_data_so_far") or data.get("answers") or {}
        if "alreadyAskedKeys" in data or "already_asked_keys" in data or "asked_step_ids" in data:
            state_data["asked_step_ids"] = data.get("alreadyAskedKeys") or data.get("already_asked_keys") or data.get("asked_step_ids") or []
        if "formPlan" in data or "form_plan" in data:
            state_data["form_plan"] = data.get("formPlan") or data.get("form_plan") or []
        if "personalizationSummary" in data or "personalization_summary" in data:
            state_data["personalization_summary"] = data.get("personalizationSummary") or data.get("personalization_summary") or ""
        if state_data:
            normalized["state"] = state_data
        
        # Constraints section
        constraints_data = {}
        if "maxSteps" in data or "max_steps" in data:
            constraints_data["max_steps"] = data.get("maxSteps") or data.get("max_steps") or 4
        if "allowedMiniTypes" in data or "allowed_mini_types" in data or "allowed_step_types" in data:
            constraints_data["allowed_step_types"] = data.get("allowedMiniTypes") or data.get("allowed_mini_types") or data.get("allowed_step_types") or []
        if "requiredUploads" in data or "required_uploads" in data:
            constraints_data["required_uploads"] = data.get("requiredUploads") or data.get("required_uploads") or []
        if constraints_data:
            normalized["constraints"] = constraints_data
        
        # Batch state
        if "batchState" in data or "batch_state" in data:
            batch_state_raw = data.get("batchState") or data.get("batch_state") or {}
            if isinstance(batch_state_raw, dict):
                normalized["batch_state"] = batch_state_raw
        
        # Preserve any extra fields
        for key, value in data.items():
            if key not in [
                "schemaVersion", "schema_version",
                "platformGoal", "platform_goal", "businessContext", "business_context",
                "industry", "service", "batchId", "batch_id",
                "stepDataSoFar", "step_data_so_far", "answers",
                "alreadyAskedKeys", "already_asked_keys", "asked_step_ids",
                "formPlan", "form_plan", "personalizationSummary", "personalization_summary",
                "maxSteps", "max_steps", "allowedMiniTypes", "allowed_mini_types", "allowed_step_types",
                "requiredUploads", "required_uploads",
                "batchState", "batch_state"
            ]:
                normalized[key] = value
        
        return normalized
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert structured format back to legacy flat format for flow_planner compatibility."""
        result = {}
        
        if self.schema_version:
            result["schemaVersion"] = self.schema_version
        
        # Context
        if self.context:
            result["platformGoal"] = self.context.platform_goal
            result["businessContext"] = self.context.business_context
            result["industry"] = self.context.industry
            result["service"] = self.context.service
        
        # State
        if self.state:
            result["batchId"] = self.state.batch_id
            result["stepDataSoFar"] = self.state.answers
            result["alreadyAskedKeys"] = self.state.asked_step_ids
            result["formPlan"] = self.state.form_plan
            result["personalizationSummary"] = self.state.personalization_summary
        
        # Constraints
        if self.constraints:
            result["maxSteps"] = self.constraints.max_steps
            result["allowedMiniTypes"] = self.constraints.allowed_step_types
            result["requiredUploads"] = self.constraints.required_uploads
        
        # Batch state
        if self.batch_state:
            result["batchState"] = self.batch_state.model_dump()
        
        return result


# Backward compatibility alias
FlowNewBatchRequest = FormRequest


class ImageGenConfig(BaseModel):
    """Image generation configuration for a session/batch."""

    model_config = ConfigDict(populate_by_name=True)

    prompt_template: Optional[str] = Field(
        default=None,
        alias="promptTemplate",
        description="Optional override prompt template. If provided, DSPy prompt-building is skipped.",
    )
    num_variants: int = Field(
        default=1,
        ge=1,
        le=8,
        alias="numVariants",
        description="Number of image variants to generate.",
    )
    provider: str = Field(
        default="mock",
        description="Image provider backend. Default 'mock' returns SVG data URLs for local/dev.",
    )
    size: Optional[str] = Field(
        default=None,
        description="Provider-specific size (e.g. '1024x1024').",
    )
    return_format: str = Field(
        default="url",
        alias="returnFormat",
        description="Return format: 'url' (recommended) or 'b64' (provider-dependent).",
    )


class MinimalImageRequest(BaseModel):
    """
    Minimal request body for image prompt + generation.

    Reuses the same session/currentBatch/state envelope as MinimalFormRequest so the frontend can
    trigger image generation after a batch completes.
    """

    model_config = ConfigDict(populate_by_name=True)

    session: SessionInfo = Field(..., description="Session and instance identifiers")
    current_batch: CurrentBatchInfo = Field(..., alias="currentBatch")
    state: FormStateInfo = Field(..., description="Overall form state: answers, asked ids, satiety, plan")

    prompt: Optional[PromptContextInfo] = Field(default=None)
    psychology: Optional[FormPsychologyInfo] = Field(default=None)
    copy: Optional[FormCopyInfo] = Field(default=None)
    request: Optional[RequestFlags] = Field(default=None)

    image: ImageGenConfig = Field(..., description="Image generation configuration")

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def instance_id(self) -> str:
        return self.session.instance_id

    @property
    def batch_id(self) -> str:
        return self.current_batch.batch_id

    @property
    def answers(self) -> Dict[str, Any]:
        return self.state.answers

    @property
    def asked_step_ids(self) -> List[str]:
        return self.state.asked_step_ids

    @property
    def form_plan(self) -> Optional[List[Dict[str, Any]]]:
        return self.state.form_plan

    @property
    def personalization_summary(self) -> str:
        return self.state.personalization_summary

    @property
    def platform_goal(self) -> Optional[str]:
        return self.prompt.goal if self.prompt else None

    @property
    def business_context(self) -> Optional[str]:
        return self.prompt.business_context if self.prompt else None

    @property
    def industry(self) -> Optional[str]:
        return self.prompt.industry if self.prompt else None

    @property
    def service(self) -> Optional[str]:
        return self.prompt.service if self.prompt else None


class ImagePromptBuildConfig(BaseModel):
    """Optional config blob used to build an image prompt when `prompt` is not provided."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    batch_id: Optional[str] = Field(default=None, alias="batchId")
    platform_goal: Optional[str] = Field(default=None, alias="platformGoal")
    business_context: Optional[str] = Field(default=None, alias="businessContext")
    industry: Optional[str] = Field(default=None)
    service: Optional[str] = Field(default=None)
    personalization_summary: Optional[str] = Field(default=None, alias="personalizationSummary")
    already_asked_keys: Optional[List[str]] = Field(default=None, alias="alreadyAskedKeys")
    form_plan: Optional[List[Dict[str, Any]]] = Field(default=None, alias="formPlan")
    batch_state: Optional[Dict[str, Any]] = Field(default=None, alias="batchState")


class ImageRequest(BaseModel):
    """
    `POST /api/image` request.

    Minimum:
    - instanceId
    - useCase: tryon | scene-placement | scene
    - either `prompt` OR `stepDataSoFar` + `config`
    """

    model_config = ConfigDict(populate_by_name=True)

    instance_id: str = Field(..., alias="instanceId")
    use_case: str = Field(..., alias="useCase")

    session_id: Optional[str] = Field(default=None, alias="sessionId")

    prompt: Optional[str] = Field(default=None)
    negative_prompt: Optional[str] = Field(default=None, alias="negativePrompt")

    user_image: Optional[str] = Field(default=None, alias="userImage")
    product_image: Optional[str] = Field(default=None, alias="productImage")
    scene_image: Optional[str] = Field(default=None, alias="sceneImage")
    reference_images: Optional[List[str]] = Field(default=None, alias="referenceImages")

    step_data_so_far: Optional[Dict[str, Any]] = Field(default=None, alias="stepDataSoFar")
    config: Optional[ImagePromptBuildConfig] = Field(default=None)

    regenerate: Optional[bool] = Field(default=None)
    num_outputs: int = Field(default=1, ge=1, le=8, alias="numOutputs")
    output_format: str = Field(default="url", alias="outputFormat")

    @field_validator("use_case")
    @classmethod
    def _validate_use_case(cls, v: str) -> str:
        t = str(v or "").strip().lower()
        if t in {"tryon", "try-on", "try on"}:
            return "tryon"
        if t in {"scene-placement", "scene placement", "placement", "scene_placement"}:
            return "scene-placement"
        if t in {"scene"}:
            return "scene"
        raise ValueError("useCase must be one of: tryon | scene-placement | scene")

    @field_validator("output_format")
    @classmethod
    def _validate_output_format(cls, v: str) -> str:
        t = str(v or "").strip().lower()
        if t in {"url", "b64"}:
            return t
        raise ValueError("outputFormat must be 'url' or 'b64'")
