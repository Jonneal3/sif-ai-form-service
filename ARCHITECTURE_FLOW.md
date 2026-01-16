# Architecture & Data Flow

## Overview

This service is a **DSPy-powered form generation microservice** that generates dynamic form questions based on user context, industry, and previous answers.

## High-Level Flow

```
Frontend (Next.js)
    ↓ POST /api/form (JSON or SSE)
Backend API (FastAPI)
    ↓ Validates & Normalizes Request
Supabase (Optional - if minimal request)
    ↓ Fetches Form Config
Backend Payload Builder
    ↓ Builds Full DSPy Payload
Pipeline (`src/app/pipeline/form_pipeline.py`)
    ↓ (Batch 1 only, if missing `state.formPlan`) Calls LLM to build the Flow Plan (a `nextBatchGuide` list)
    ↓ Calls LLM to generate Mini Steps for the current batch (questions)
LLM (Groq/OpenAI)
    ↓ Returns Flow Plan JSON (`nextBatchGuide` list) + JSONL Mini Steps
Pipeline (`src/app/pipeline/form_pipeline.py`)
    ↓ Parses & Validates JSONL
    ↓ Cleans Placeholders (NEW!)
    ↓ Validates with Pydantic Models (Mini → Full Schema)
    ↓ Builds `formPlan` snapshot (policy + nextBatchGuide) for the frontend
    ↓ Builds `deterministicPlacements` (uploads/CTAs) for the frontend
Backend API
    ↓ Returns JSON (or streams SSE if requested)
Frontend (Next.js)
    ↓ Renders Steps
```

## Image Generation Flow (New)

After the user answers a batch (i.e. once the frontend has updated `state.answers` and decides a batch is complete),
the frontend should call `POST /api/image` with the same session/currentBatch/state envelope plus an `image` config.

High level:

```
Frontend (Next.js)
    ↓ POST /api/image
Backend API (FastAPI)
    ↓ Builds the same normalized context payload
DSPy ImagePrompt Module (optional)
    ↓ Produces a validated prompt JSON (or deterministic fallback if no LLM configured)
Image Provider (mock by default)
    ↓ Returns image URLs (data URLs for mock) + generation metrics
Backend API
    ↓ Returns {prompt, images, metrics}
```

## Request Formats

### Minimal Format (Recommended)

**Endpoint:** `POST /api/form`

**Required Fields:**
```json
{
  "session": {
    "sessionId": "sess_xxx",
    "instanceId": "uuid-here"
  },
  "currentBatch": {
    "batchId": "ContextCore",  // phase id (e.g., "ContextCore", "Details", "Preferences")
    "batchNumber": 1,
    "satietyTarget": 0.77,
    "satietyRemaining": 0.77,
    "maxSteps": 5,
    "allowedComponentTypes": ["choice"]
  },
  "state": {
    "satietyCurrent": 0.0,
    "answers": {},
    "askedStepIds": [],
    "formPlan": null,
    "personalizationSummary": ""
  }
}
```

**Optional Fields (Backend fetches from Supabase if missing):**
- `prompt`: { goal, businessContext, industry, service }
- `psychology`: { approach, description }
- `copy`: { style, principles, tone }
- `request`: { noCache, schemaVersion }

### Full Format (Backward Compatible)

**Endpoint:** `POST /api/form`

If request doesn't have `sessionId` or `session`, backend uses full format:

```json
{
  "context": {
    "platform_goal": "...",
    "business_context": "...",
    "industry": "...",
    "service": "..."
  },
  "state": {
    "batch_id": "ContextCore",
    "answers": {...},
    "asked_step_ids": [...],
    "form_plan": [...],
    "personalization_summary": ""
  },
  "constraints": {
    "max_steps": 5,
    "allowed_step_types": ["choice"],
    "required_uploads": []
  }
}
```

## Data Transformation Pipeline

### 1. Request Validation (`api/routes/form.py`)

- Detects request format (minimal vs full)
- Validates with Pydantic models (`MinimalFormRequest` or `FormRequest`)
- If minimal: Fetches missing data from Supabase
- Converts to legacy format for DSPy compatibility

### 2. DSPy Payload Building (`api/supabase_client.py`)

- Combines frontend state with Supabase data
- Builds batch-specific constraints
- Calculates `allowedMiniTypes`, `maxSteps`, `items` from form plan
- Returns flat payload for DSPy

### 3. LLM Generation (`src/app/pipeline/form_pipeline.py`)

**Input to LLM:**
- Platform goal, business context, industry, service
- Current answers, asked step IDs
- Form plan (if exists)
- Batch constraints

**LLM Output (JSONL Mini Steps):**
```
{"id":"step-project-goal","type":"multiple_choice","question":"What is your goal?","options":[...]}
{"id":"step-space-type","type":"multiple_choice","question":"What type of space?","options":[...]}
```

**LLM Output (Flow Plan, batch 1 only when missing):**
- A JSON array of `FormPlanItem` objects (the “nextBatchGuide backlog”).
- This is an internal planner output; the pipeline converts it into the full `formPlan` snapshot returned to the frontend.

### 4. Validation & Schema Mapping (`src/app/pipeline/form_pipeline.py`)

**This is where "Mini Schema → Full Schema" happens:**

1. **Parse JSONL** - Each line becomes a dict
2. **Clean Placeholders** - Remove `<<max_depth>>` placeholders from options
3. **Validate with Pydantic** - Uses full schema models:
   - `MultipleChoiceUI` - Full option structure with label, value, description, etc.
   - `TextInputUI` - Full text input with placeholder, max_length, multiline
   - `RatingUI` - Full slider/rating with min, max, step, unit
   - etc.
4. **Normalize Step IDs** - Converts underscores to hyphens
5. **Return Full Schema** - Validated objects matching UI contract

**Key Point:** The "mini schema" is just the raw LLM JSONL output. The validation step (`_validate_mini()`) maps it to the full schema using Pydantic models that match the UI contract exactly.

### 5. Response (`api/routes/form.py`)

**JSON Response Format:**
```json
{
  "formPlan": {
    "v": 1,
    "constraints": {
      "maxBatches": 2,
      "maxStepsTotal": 12,
      "maxStepsPerBatch": 6,
      "tokenBudgetTotal": 3000
    },
    "flow": {
      "batchOrder": ["ContextCore", "Details", "Preferences"],
      "withinBatchStepOrder": "easy_to_deep",
      "priority": { "levels": ["critical", "optional"], "neverSkipCritical": true }
    },
    "batches": [
      {
        "batchId": "ContextCore",
        "purpose": "Collect minimum info needed to produce a usable output (pricing range).",
        "maxSteps": 5,
        "allowedComponentTypes": ["choice", "yes_no", "slider", "text"],
        "rigidity": 0.9
      },
      {
        "batchId": "Details",
        "purpose": "Quantify size/budget/timeline and key constraints.",
        "maxSteps": 6,
        "allowedComponentTypes": ["choice", "slider", "text"],
        "rigidity": 0.6
      },
      {
        "batchId": "Preferences",
        "purpose": "Capture preferences that change the output.",
        "maxSteps": 6,
        "allowedComponentTypes": ["choice", "slider", "text"],
        "rigidity": 0.4
      }
    ],
    "stop": { "requiredComplete": true, "satietyTarget": 1.0 },
    "keys": ["project_type", "area_location"],
    "nextBatchGuide": [
      {
        "key": "project_type",
        "goal": "Define the work type",
        "why": "Determines scope",
        "component_hint": "choice",
        "priority": "critical",
        "importance_weight": 0.2,
        "expected_metric_gain": 0.18
      }
    ]
  },
  "deterministicPlacements": {
    "v": 1,
    "placements": [
      {
        "id": "step-upload-scene",
        "type": "upload",
        "role": "sceneImage",
        "position": "after",
        "anchor_step_id": "step-project-goal",
        "deterministic": true
      }
    ]
  },
  "miniSteps": [
    {
      "id": "step-project-goal",
      "type": "multiple_choice",
      "question": "What is your project goal?",
      "metadata": { "family": "project_goal" },
      "options": [
        {"label": "Renovation", "value": "renovation"},
        {"label": "New Build", "value": "new_build"}
      ],
      "required": true,
      "metric_gain": 0.1
    }
  ]
}
```

**SSE Streaming (optional):**
- Enabled when `Accept: text/event-stream` or `?stream=1`
- Events: `open`, `mini_step`, `meta`, `error`

## Schema Mapping Details

### Mini Schema (LLM Output)
- Raw JSONL from LLM
- May have placeholders like `"<<max_depth>>"`
- May be missing optional fields
- Step IDs may use underscores

### Full Schema (Validated Output)
- Validated with Pydantic models
- All placeholders cleaned
- All required fields present
- Optional fields with defaults
- Step IDs normalized (underscores → hyphens)
- Matches UI contract exactly

### Validation Function (`_validate_mini()`)

```python
def _validate_mini(obj: Any) -> Optional[Dict[str, Any]]:
    # 1. Detect step type
    # 2. Clean placeholders (NEW!)
    # 3. Validate with Pydantic model (e.g., MultipleChoiceUI)
    # 4. Normalize step ID
    # 5. Return full schema dict
```

**Example:**
- Input: `{"id":"step_goal","type":"multiple_choice","options":[{"label":"<<max_depth>>","value":"<<max_depth>>"}]}`
- After cleaning: `{"id":"step-goal","type":"multiple_choice","options":[{"label":"Not sure","value":"not_sure"}]}`
- After validation: Full `MultipleChoiceUI` schema with all fields

## Placeholder Cleanup (NEW)

**Problem:** LLM sometimes generates `"<<max_depth>>"` placeholders instead of real option values.

**Solution:** `_clean_options()` function:
1. Detects placeholder patterns: `"<<max_depth>>"`, `"<<max_depth"`, `"max_depth>>"`, etc.
2. Removes placeholder options
3. If all options are placeholders, uses fallback: `[{"label": "Not sure", "value": "not_sure"}]`
4. Logs cleanup actions for debugging

**Location:** `src/app/pipeline/form_pipeline.py` → `_clean_options()` → called in `_validate_mini()`

## Batch Progression

### Batch 1: ContextCore
- **Goal:** Fast, broad context questions
- **Allowed Types:** `["choice"]` (simple)
- **Max Steps:** 5
- **Satiety Target:** 0.77
- **Generates (if missing):** Flow Plan (nextBatchGuide) → then returns `formPlan` snapshot + `miniSteps` + `deterministicPlacements`

### Batch 2+: Subsequent Phases (Policy-Driven)
- **Goal:** Progress through policy phases (default: `Details` → `Preferences`)
- **Allowed Types / Max Steps:** Derived from `formPlan.batches[*]` / policy defaults
- **Uses:** `state.formPlan.nextBatchGuide` (plan backlog) + `state.answers` + `state.askedStepIds`

### Deterministic Placements (Uploads, CTAs)
- `miniSteps` are the question steps the LLM generates for each batch.
- `deterministicPlacements` (formerly `uiPlan`) is the lightweight placement spec that tells the UI where to inject deterministic blocks (uploads, CTAs) without asking the LLM to produce them.
- `formPlan.nextBatchGuide` is the optional backlog of planned question keys so the frontend can round-trip the plan across calls even when state is rebuilt.

## Key Files

- **`api/routes/form.py`** - HTTP endpoints, request validation, JSON/SSE response
- **`api/models.py`** - Pydantic models for request/response validation
- **`api/supabase_client.py`** - Supabase integration, payload building
- **`src/app/pipeline/form_pipeline.py`** - DSPy integration, LLM calls, validation, placeholder cleanup
- **`src/app/dspy/flow_plan_module.py`** - DSPy wrapper for FlowPlanJSON (first-call plan builder)
- **`src/app/dspy/batch_generator_module.py`** - DSPy wrapper for NextStepsJSONL (batch question generator)
- **`src/app/signatures/json_signatures.py`** - DSPy signatures (LLM contracts)
- **`src/app/form_planning/`** - Form plan + policy + deterministic placements helpers
- **`shared/ai-form-contract/schema/`** - UI contract (JSON Schema, TypeScript types)

## Error Handling

1. **Request Validation Errors** → 400/500 with error message
2. **Supabase Fetch Errors** → JSON error response (or SSE error event when streaming)
3. **LLM Errors** → Caught, logged, returned in response
4. **Validation Errors** → Invalid steps skipped, valid steps returned
5. **Placeholder Options** → Cleaned automatically, fallback used if needed

## Debugging

**Logs to Watch:**
- `[FlowPlanner] Raw DSPy response fields: ...` - LLM output
- `[FlowPlanner] 🧹 Removed placeholder option: ...` - Placeholder cleanup
- `[FlowPlanner] ✅ Step 'xxx': Cleaned options (3 -> 2)` - Options cleaned
- `[FlowPlanner] ⚠️ Step 'xxx': All 3 option(s) were placeholders, using fallback` - Fallback used
- `[FlowPlanner] Validated steps count: 4` - Final step count

**Common Issues:**
- Placeholders in options → Fixed by cleanup function
- Missing options → Fallback to "Not sure"
- Invalid step types → Skipped during validation
- Step ID mismatches → Normalized automatically
