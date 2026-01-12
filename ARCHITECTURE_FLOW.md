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
DSPy Flow Planner
    ↓ Calls LLM (via LiteLLM)
LLM (Groq/OpenAI)
    ↓ Returns JSONL with Mini Steps
DSPy Flow Planner
    ↓ Parses & Validates JSONL
    ↓ Cleans Placeholders (NEW!)
    ↓ Validates with Pydantic Models (Mini → Full Schema)
Backend API
    ↓ Returns JSON (or streams SSE if requested)
Frontend (Next.js)
    ↓ Renders Steps
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
    "batchId": "ContextCore",  // or "PersonalGuide"
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

### 3. LLM Generation (`flow_planner.py`)

**Input to LLM:**
- Platform goal, business context, industry, service
- Current answers, asked step IDs
- Form plan (if exists)
- Batch constraints

**LLM Output (JSONL):**
```
{"id":"step-project-goal","type":"multiple_choice","question":"What is your goal?","options":[...]}
{"id":"step-space-type","type":"multiple_choice","question":"What type of space?","options":[...]}
```

### 4. Validation & Schema Mapping (`flow_planner.py`)

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
  "requestId": "next_steps_1234567890",
  "miniSteps": [
    {
      "id": "step-project-goal",
      "type": "multiple_choice",
      "question": "What is your project goal?",
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

**Location:** `flow_planner.py` → `_clean_options()` → called in `_validate_mini()`

## Batch Progression

### Batch 1: ContextCore
- **Goal:** Fast, broad context questions
- **Allowed Types:** `["choice"]` (simple)
- **Max Steps:** 5
- **Satiety Target:** 0.77
- **Generates:** Form plan for batch 2

### Batch 2: PersonalGuide
- **Goal:** Deeper, personalized follow-ups
- **Allowed Types:** `["choice", "slider", "text"]` (complex)
- **Max Steps:** 6-8
- **Satiety Target:** 1.0
- **Uses:** Form plan from batch 1 + personalization summary

## Key Files

- **`api/routes/form.py`** - HTTP endpoints, request validation, JSON/SSE response
- **`api/models.py`** - Pydantic models for request/response validation
- **`api/supabase_client.py`** - Supabase integration, payload building
- **`flow_planner.py`** - DSPy integration, LLM calls, validation, placeholder cleanup
- **`modules/signatures/json_signatures.py`** - DSPy signatures (LLM contracts)
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
