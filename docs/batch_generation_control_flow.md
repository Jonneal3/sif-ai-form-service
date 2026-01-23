# Batch generation control flow (backend)

This doc explains the **happy path** from an HTTP request to the `miniSteps[]` we return.

## The path (in order)

### 1) Vercel entrypoint: `api/index.py`

- Vercel routes requests to `api/index.py` (`vercel.json`).
- `api/index.py` just re-exports the FastAPI `app` from `api/main.py`.

### 2) HTTP route: `api/main.py`

**Route:** `POST /v1/api/form/{instanceId}`

What happens:

- **Parse**: FastAPI parses + validates the JSON body as `NewBatchRequest` (`src/schemas/api_models.py`).
- **Adapt**: `api/request_adapter.py::to_next_steps_payload(instance_id, body_dict)` adds `session.instanceId` and ensures a couple canonical keys.
- **Generate**: calls `programs.batch_generator.orchestrator.next_steps_jsonl(payload_dict)`.
- **Return**: `JSONResponse` with `{ requestId, schemaVersion, miniSteps, ... }`.

Optional:

- If `AI_FORM_VALIDATE_CONTRACT=true`, we also validate request/response against the OpenAPI JSON schema via `api/openapi_contract.py`.

### 3) Batch generation: `src/programs/batch_generator/orchestrator.py`

**Entry point:** `next_steps_jsonl(payload: dict)`

High-level flow:

- Build a **context dict** from the payload (`_build_context`).
  - Includes: `known_answers`, `asked_step_ids`, `required_uploads`, and `instanceContext` (categories/subcategories).
- Prepare the LLM call (`_prepare_predictor`) and run DSPy.
- Parse model output (JSONL) into candidate UI steps.
- Validate + filter steps (dedupe, avoid already-asked, enforce allowed types).
- Return the final response dict with `miniSteps[]`.

## What data matters most

- **Identity**: `instanceId` (path param) is injected into `payload["session"]["instanceId"]`.
- **State/memory**:
  - `stepDataSoFar` (answers so far)
  - `askedStepIds` (step ids already shown)
  - optional `answeredQA` (plain-English memory)
- **Instance context (multi-value)**: `instanceContext.categories[]` and `instanceContext.subcategories[]`
  - Used to ground question generation and avoid off-topic options.

## Key files (quick index)

- **HTTP**
  - `api/index.py`: Vercel entrypoint
  - `api/main.py`: FastAPI routes
  - `api/request_adapter.py`: the *only* inbound request shaping
- **Schemas**
  - `src/schemas/api_models.py`: `NewBatchRequest`, `FormResponse`
  - `src/schemas/ui_steps.py`: UI-step validation models
- **Batch generator**
  - `src/programs/batch_generator/orchestrator.py`: end-to-end generation
  - `src/programs/batch_generator/signatures/signature.py`: model instructions + I/O fields
  - `src/programs/batch_generator/engine/dspy_program.py`: DSPy program wrapper

