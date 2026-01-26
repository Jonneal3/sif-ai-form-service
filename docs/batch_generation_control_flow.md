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
- **Generate**: calls `programs.form_pipeline.orchestrator.next_steps_jsonl(payload_dict)`.
- **Return**: `JSONResponse` with `{ requestId, schemaVersion, miniSteps, ... }`.

Optional:

- If `AI_FORM_VALIDATE_CONTRACT=true`, we also validate request/response against the OpenAPI JSON schema via `api/openapi_contract.py`.

### 3) Form pipeline: `src/programs/form_pipeline/orchestrator.py`

**Entry point:** `next_steps_jsonl(payload: dict)`

High-level flow:

- Build a **context dict** from the payload.
  - Includes: answers so far, asked step ids, instance context (industry/service), and grounding text (if available).
- If grounding is missing, generate a **small grounding summary** first.
- Call the **planner** to get a question plan (a list of keys).
- Take the next slice of that plan (based on what was already asked).
- Call the **renderer** to turn that slice into UI steps (JSONL).
- Parse + validate the JSONL into `miniSteps[]`.
- Return the response dict with `miniSteps[]`.

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
- **Form pipeline**
  - `src/programs/form_pipeline/orchestrator.py`: end-to-end generation (Planner → Renderer)
  - `src/programs/form_pipeline/context/`: builds the context dict (split into small files)
  - `src/programs/form_pipeline/prompts/`: prompt text blocks used by signatures
  - `src/programs/form_pipeline/grounding_summary/`: small DSPy module that generates grounding when missing
- **Planner + renderer**
  - `src/programs/question_planner/`: creates a question plan (keys + intent)
  - `src/programs/renderer/`: turns a plan slice into UI steps (JSONL)

