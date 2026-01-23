# Batch generation control flow (backend only)

This is the backend flow from an HTTP request to "questions we return", following the request path sequentially through each file.

## Sequential flow through files

### 1. Entry point: `api/index.py`

**File:** `api/index.py`

- Vercel routes all requests to `api/index.py` (configured in `vercel.json`)
- This file simply re-exports the FastAPI `app` from `api/main.py`
- Line 8: `from .main import app`

**Next:** Request flows to `api/main.py`

---

### 2. API layer: `api/main.py`

**File:** `api/main.py`

#### 2.1. Minimal inbound flow (the “boring” API layer)

**Route:** `POST /v1/api/form/{instanceId}`

- **Parse**: FastAPI validates the body as `NewBatchRequest` (canonical shape)
- **Adapt**: `api/request_adapter.py::to_next_steps_payload(instance_id, body_dict)` (adds `session.instanceId`, ensures a couple canonical keys)
- **Generate**: `next_steps_jsonl(payload_dict)`
- **Return**: `JSONResponse(resp)`

**Optional debug-only contract checks**: set `AI_FORM_VALIDATE_CONTRACT=true` to validate request/response using `api/openapi_contract.py`.

---

### 3. Batch generator orchestrator: `src/programs/batch_generator/orchestrator.py`

**File:** `src/programs/batch_generator/orchestrator.py`

#### 3.1. Entry function: `next_steps_jsonl`

**Function:** `next_steps_jsonl(payload: Dict[str, Any])` (line 1651)

This is the main entry point that orchestrates the entire batch generation process.

**Step 3.1.1:** Prepare predictor (line 1658)
- Calls `_prepare_predictor(payload)` to set up everything needed for the LLM call
- If preparation fails, returns error response (lines 1659-1666)

**Step 3.1.2:** Extract preparation results (lines 1668-1685)
- Extracts: `module`, `inputs`, `ui_types`, `request_id`, `start_time`, `lm_cfg`, `lm`
- Extracts: `track_usage`, `copy_needed`, `copy_context_json`, `max_steps`
- Extracts: `allowed_mini_types`, `already_asked_keys`, `required_upload_ids`
- Extracts: `service_anchor_terms`, `lint_config`, `context`

**Step 3.1.3:** Load copy linting functions (lines 1686-1701)
- Conditionally imports linting functions from `form_planning.copywriting.linter`:
  - `apply_reassurance`, `lint_steps`, `sanitize_steps`
- Wrapped in try/except, so missing imports don't crash

**Step 3.1.4:** Call LLM via DSPy (line 1705)
- Executes: `pred = module(**inputs)`
- This is where the actual LLM API call happens
- The `module` is a `BatchStepsModule` instance (from `batch_steps_module.py`)
- The `inputs` contain `context_json`, `max_steps`, `allowed_mini_types`

**Step 3.1.5:** Extract raw LLM output (lines 1710-1712)
- Reads `pred.mini_steps_jsonl` (a string containing JSONL)
- Logs the raw response for debugging

**Step 3.1.6:** Parse and validate LLM output (lines 1713-1794)
- Initializes `emitted` list and `seen_ids` set
- Calculates exploration budget based on rigidity and max steps
- Parses JSONL line-by-line (lines 1774-1786):
  - Splits `raw_mini_steps` by newlines
  - Parses each line as JSON via `_best_effort_parse_json(line)`
  - Extracts candidates via `_iter_candidates(parsed)`
  - Validates each candidate via `_maybe_accept(candidate)`
- Fallback: if no steps emitted, tries parsing entire string as single JSON (lines 1788-1794)

**Step 3.1.7:** Validation logic (`_maybe_accept` function, lines 1733-1772)
- Checks if `max_steps_limit` reached
- Validates step structure via `_validate_mini(candidate, ui_types)` (uses `ui_steps.py` schemas)
- Adds `batch_phase_id` from payload if present
- Filters out:
  - Steps already asked (checks `already_asked_keys`)
  - Duplicate step IDs (checks `seen_ids`)
  - Disallowed step types (checks `allowed_set`)
  - Steps with banned filler options (via `_apply_banned_option_policy`)
  - Upload steps with wrong type, or text steps with upload-like IDs
- Adds valid steps to `emitted` list

**Step 3.1.8:** Copy sanitizing and linting (lines 1805-1820)
- If `sanitize_steps` available: sanitizes emitted steps (line 1808)
- If `apply_reassurance` available: applies reassurance to steps (line 1810)
- If `lint_steps` available: runs linting and collects violations (lines 1811-1820)

**Step 3.1.9:** Optional must-have copy generation (lines 1850-1872)
- If `copy_needed` is True:
  - Calls `MustHaveCopyModule` from `must_have_copy_module.py`
  - Generates copy for steps that need it
  - Updates emitted steps with generated copy

**Step 3.1.10:** Optional composite step wrapping (lines 1879-1893)
- Tries to import `wrap_last_step_with_upload_composite` from `form_planning.composite`
- If available, wraps last AI step + uploads into one composite UI step
- Updates `meta["miniSteps"]` if wrapping occurred

**Step 3.1.11:** Build final response (lines 1822-1902)
- Calculates latency in milliseconds
- Builds response dict with:
  - `requestId`: unique request identifier
  - `schemaVersion`: UI step schema version
  - `miniSteps`: list of validated steps
- Optionally includes metadata (if `_include_response_meta` returns True):
  - `copyPackId`, `copyPackVersion`
  - `lintFailed`, `lintViolationCodes`
  - `debugContext` (industry, service, useCase, goalIntent, stage, allowedMiniTypes, maxSteps)
  - `lmUsage` (if tracking enabled)
- Logs final response summary
- Returns the `meta` dict (line 1903)

**Next:** Response flows back to `api/main.py`, which wraps it in JSONResponse

---

#### 3.2. Preparation function: `_prepare_predictor`

**Function:** `_prepare_predictor(payload: Dict[str, Any])` (line 1406)

This function prepares everything needed for the LLM call. It's called by `next_steps_jsonl` and returns a dict with all the prepared components.

**Step 3.2.1:** Initialize request tracking (lines 1408-1415)
- Generates `request_id` from timestamp
- Records `start_time`
- Extracts `schema_version` from payload or contract files

**Step 3.2.2:** Configure LLM (lines 1417-1450)
- Calls `_make_dspy_lm()` to get LLM configuration from environment variables
- If no LLM configured, returns error dict
- Imports `dspy` library
- Sets LLM parameters:
  - `llm_timeout`: from `DSPY_LLM_TIMEOUT_SEC` env var (default 20s)
  - `temperature`: from `DSPY_TEMPERATURE` env var (default 0.7)
  - `max_tokens`: from `DSPY_NEXT_STEPS_MAX_TOKENS` env var (default 2000), or override from payload

**Step 3.2.3:** Create DSPy LM object (lines 1451-1465)
- Creates `dspy.LM(...)` instance with the configured provider/model
- Stores in `lm` variable

**Step 3.2.4:** Load UI step schema (lines 1466-1480)
- Loads UI step schema from `src/schemas/ui_steps.py`
- Extracts allowed UI types for validation
- Stores in `ui_types` variable

**Step 3.2.5:** Load DSPy signature (lines 1481-1485)
- Imports `BatchNextStepsJSONL` signature from `src/programs/batch_generator/signatures/json_signatures.py`
- This signature defines:
  - Input: `context_json` (string)
  - Output: `mini_steps_jsonl` (string)
  - System instructions in the docstring

**Step 3.2.6:** Create DSPy module (lines 1486-1490)
- Creates `BatchStepsModule` instance from `src/programs/batch_generator/batch_steps_module.py`
- This module wraps the signature and will call the LLM

**Step 3.2.7:** Load demo examples (few-shot) (lines 1491-1505)
- Calls `load_demos()` from `src/programs/batch_generator/demos.py`
- Loads examples from `src/programs/batch_generator/examples/current/next_steps_examples.jsonl`
- If examples found, adds them to the module for few-shot learning

**Step 3.2.8:** Build context dict (lines 1506-1510)
- Calls `_build_context(payload)` (line 802)
- This function (lines 802-998) builds a comprehensive context dict containing:
  - Platform/goal: `platform_goal`, `business_context`, `goal_intent`
  - Vertical context: `industry`, `service` (multi-value support), `grounding_summary`, `vertical_context`
  - Instance context: `instance_categories`, `instance_subcategories` (arrays)
  - Use case: `use_case`, `flow_guide` (stage hints)
  - State/memory: `known_answers`, `asked_step_ids`, `answered_qa`
  - Constraints: `batch_constraints`, `max_steps`, `allowed_mini_types`
  - Uploads: `required_uploads`
  - Copy/style: `copy_style`, `copy_context`
  - Service anchor terms: extracted from grounding if available

**Step 3.2.9:** Add copy/style rules (lines 1511-1520)
- Calls `compile_copy_pack()` from `src/programs/batch_generator/form_planning/copywriting/compiler.py`
- Adds copy style rules to context if available
- Wrapped in try/except, so missing imports don't crash

**Step 3.2.10:** Add flow rules (lines 1521-1535)
- Calls various flow planning functions (all wrapped in try/except):
  - `get_allowed_components()` from `form_planning.components_allowed.py`
  - `get_flow_guide()` from `form_planning.flow.py`
  - `get_batch_ordering()` from `form_planning.batch_ordering.py`
  - `get_question_tonality()` from `form_planning.question_tonality.py`
  - `get_static_constraints()` from `form_planning.static_constraints.py`
- These functions add flow guidance, pacing rules, and constraints to context

**Step 3.2.11:** Serialize context to JSON string (lines 1536-1540)
- Calls `_compact_json(context)` to convert context dict to compact JSON string
- Stores in `inputs["context_json"]`

**Step 3.2.12:** Extract other inputs (lines 1541-1580)
- Extracts `max_steps` from payload or defaults to 4
- Extracts `allowed_mini_types` from context or uses all UI types
- Extracts `already_asked_keys` as a set
- Extracts `required_upload_ids` as a set
- Extracts `service_anchor_terms` from context
- Extracts `lint_config` from context
- Determines if copy generation is needed (`copy_needed`)
- Builds `copy_context_json` if needed

**Step 3.2.13:** Return preparation dict (lines 1581-1630)
- Returns dict containing:
  - `module`: DSPy module instance
  - `inputs`: dict with `context_json`, `max_steps`, `allowed_mini_types`
  - `ui_types`: UI step schema types
  - `request_id`, `start_time`, `schema_version`
  - `lm_cfg`, `lm`: LLM configuration and instance
  - `track_usage`: whether to track LLM usage
  - `copy_needed`, `copy_context_json`: copy generation flags
  - `max_steps`, `allowed_mini_types`: step constraints
  - `already_asked_keys`, `required_upload_ids`: filtering sets
  - `service_anchor_terms`, `lint_config`: quality controls
  - `context`: full context dict
  - `batch_constraints`: batch-level constraints

**Next:** Returns to `next_steps_jsonl` which uses these prepared components

---

### 4. Response flow back to API

**File:** `api/main.py`

- `next_steps_jsonl()` returns a dict (the `meta` object)
- Handler wraps it in `JSONResponse(resp)` (line 161)
- FastAPI serializes to JSON and returns HTTP response
- Response includes:
  - `requestId`: unique identifier
  - `schemaVersion`: UI step schema version
  - `miniSteps`: array of validated UI step objects
  - Optional metadata (if requested)

---

## Supporting files (referenced during flow)

### Schema files

**`src/schemas/api_models.py`**
- Defines `NewBatchRequest` and `FormResponse` Pydantic models
- Used by FastAPI for request/response validation

**`src/schemas/ui_steps.py`**
- Defines UI step schema models
- Used by `_validate_mini()` to validate LLM output

### DSPy components

**`src/programs/batch_generator/signatures/json_signatures.py`**
- Defines `BatchNextStepsJSONL` signature
- Contains system instructions for the LLM
- Defines input (`context_json`) and output (`mini_steps_jsonl`) fields

**`src/programs/batch_generator/batch_steps_module.py`**
- Defines `BatchStepsModule` class
- Wraps the signature in a DSPy module
- Handles the forward pass to call the LLM

**`src/programs/batch_generator/demos.py`**
- Loads few-shot examples from JSONL file
- Examples stored in `src/programs/batch_generator/examples/current/next_steps_examples.jsonl`

### Form planning modules

**`src/programs/batch_generator/form_planning/copywriting/compiler.py`**
- Compiles copy style rules into context

**`src/programs/batch_generator/form_planning/copywriting/linter.py`**
- Provides `sanitize_steps()`, `apply_reassurance()`, `lint_steps()` functions
- Used for post-processing LLM output

**`src/programs/batch_generator/form_planning/flow.py`**
- Provides flow guidance (early/middle/late pacing)

**`src/programs/batch_generator/form_planning/components_allowed.py`**
- Determines allowed component types

**`src/programs/batch_generator/form_planning/batch_ordering.py`**
- Provides batch ordering rules

**`src/programs/batch_generator/form_planning/question_tonality.py`**
- Provides question tonality/style rules

**`src/programs/batch_generator/form_planning/static_constraints.py`**
- Provides static constraints

**`src/programs/batch_generator/form_planning/composite.py`**
- Provides `wrap_last_step_with_upload_composite()` function
- Optionally wraps steps into composite UI steps

### Copy generation

**`src/programs/batch_generator/must_have_copy_module.py`**
- Defines `MustHaveCopyModule` for generating copy when needed
- Called conditionally if `copy_needed` is True

### Contract files

**`shared/ai-form-ui-contract/schema/ui_step.schema.json`**
- Canonical UI step schema JSON
- Used by `GET /v1/api/form/capabilities` endpoint

**`shared/ai-form-service-openapi/openapi.json`**
- OpenAPI contract for the service
- Used by `validate_new_batch_request()` and `validate_new_batch_response()`

---

## Optional / best-effort imports (can be missing without crashing)

- `grounding.keywords.extract_service_anchor_terms`
  - Used by: `orchestrator.py` to extract anchor terms
  - Behavior: wrapped in `try/except`; if missing, uses no anchor terms

- `programs.batch_generator.form_planning.plan` and `programs.batch_generator.form_planning.composite`
  - Used by: `orchestrator.py` for advanced planning features
  - Behavior: imported inside `try/except`; if missing, falls back to simpler behavior
