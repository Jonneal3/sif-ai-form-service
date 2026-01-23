# Batch generation control flow (backend only)

This is the backend flow from an HTTP request to “questions we return”.

## Step-by-step flow (no gaps)

1. FastAPI receives the request.
   - File: `api/main.py`
   - Route: `POST /v1/api/form` (`api/main.py:180-191`)
   - Route: `POST /api/ai-form/{instanceId}/new-batch` (`api/main.py:193-211`)

2. FastAPI turns the body into a Python dict.
   - File: `src/schemas/api_models.py` (Pydantic models)
   - File: `api/main.py` calls `payload.model_dump(...)` (`api/main.py:188-191`)

3. FastAPI normalizes the dict into the shape the batch generator expects.
   - File: `api/main.py`
   - Function: `_normalize_form_payload` (`api/main.py:79-138`)
   - What it does:
     - Moves fields into names like `batchId`, `batchNumber`, `maxSteps`, `stepDataSoFar`, `askedStepIds`.
     - Merges multiple “asked step id” lists into one list.

4. FastAPI calls the batch generator entry function.
   - File: `src/programs/batch_generator/orchestrator.py`
   - Function: `next_steps_jsonl(payload_dict)` (`src/programs/batch_generator/orchestrator.py:1651`)

5. `next_steps_jsonl` prepares everything needed for the LLM call.
   - File: `src/programs/batch_generator/orchestrator.py`
   - Function: `_prepare_predictor(payload)` (`src/programs/batch_generator/orchestrator.py:1406`)
   - Inside `_prepare_predictor`, important sub-steps:
     - Pick an LLM provider/model from env vars (`_make_dspy_lm`) (same file).
     - Create the DSPy LM object (`dspy.LM(...)`) (same file).
     - Load the DSPy signature + UI step validators:
       - Signature file: `src/programs/batch_generator/signatures/json_signatures.py`
       - UI step schema models: `src/schemas/ui_steps.py`
     - Create the DSPy module that will call the LLM:
       - File: `src/programs/batch_generator/batch_steps_module.py`
     - Load demo examples (few-shot), if present:
       - Loader: `src/programs/batch_generator/demos.py`
       - Demo data: `src/programs/batch_generator/examples/current/next_steps_examples.jsonl`
     - Build the “context” dict that will be sent to the LLM:
       - Function: `_build_context(payload)` (`src/programs/batch_generator/orchestrator.py:802`)
       - Output: a big Python dict with service + answers + constraints.
     - Add copy/style rules (best-effort):
       - File: `src/programs/batch_generator/form_planning/copywriting/compiler.py`
       - File: `src/programs/batch_generator/form_planning/copywriting/linter.py`
     - Add flow rules (allowed types, early/middle/late pacing) (best-effort):
       - File: `src/programs/batch_generator/form_planning/flow.py`
       - File: `src/programs/batch_generator/form_planning/components_allowed.py`
       - File: `src/programs/batch_generator/form_planning/batch_ordering.py`
       - File: `src/programs/batch_generator/form_planning/question_tonality.py`
       - File: `src/programs/batch_generator/form_planning/static_constraints.py`
     - Turn the context dict into a string:
       - Function: `_compact_json(context)` (same file)
       - Field name: `inputs["context_json"]`

6. `next_steps_jsonl` runs DSPy (this is the LLM call).
   - File: `src/programs/batch_generator/orchestrator.py`
   - Line: `pred = module(**inputs)` (`src/programs/batch_generator/orchestrator.py:1705`)
   - What the LLM sees:
     - System instructions come from `src/programs/batch_generator/signatures/json_signatures.py` (signature docstring).
     - Real data comes from `inputs["context_json"]` (the serialized context dict).

7. The backend parses and validates the LLM output.
   - File: `src/programs/batch_generator/orchestrator.py`
   - Field read: `pred.mini_steps_jsonl` (`src/programs/batch_generator/orchestrator.py:1711`)
   - Parser: `_best_effort_parse_json` + JSONL loop (same file)
   - Validator: `_validate_mini(candidate, ui_types)` (same file)
   - Schema models used by the validator: `src/schemas/ui_steps.py`
   - Filters applied (same file):
     - Skip steps already asked
     - Skip duplicates
     - Skip disallowed step types
     - Skip invalid options / banned filler options

8. (Sometimes) the backend runs copy sanitizing and linting on the validated steps.
   - File: `src/programs/batch_generator/form_planning/copywriting/linter.py`
   - Called from: `src/programs/batch_generator/orchestrator.py` (inside `next_steps_jsonl`)

9. (Sometimes) the backend runs a second DSPy call to generate “must-have copy”.
   - File: `src/programs/batch_generator/must_have_copy_module.py`
   - Called from: `src/programs/batch_generator/orchestrator.py` (inside `next_steps_jsonl`)

10. The backend builds the final JSON response and returns it.
   - File: `src/programs/batch_generator/orchestrator.py`
   - Output keys: `requestId`, `schemaVersion`, `miniSteps` (and optional meta)

## Shared schema files used by the backend

- `shared/ai-form-ui-contract/schema/ui_step.schema.json` and `shared/ai-form-ui-contract/schema/schema_version.txt`
  - Used by: `GET /v1/api/form/capabilities` (`api/main.py`) to return the UI-step schema.
- `shared/ai-form-service-openapi/openapi.json`
  - Used by: `api/openapi_contract.py` to validate the OpenAPI contract route request/response.

## Optional / best-effort imports (can be missing without crashing)

- `grounding.keywords.extract_service_anchor_terms`
  - Used by: `src/programs/batch_generator/orchestrator.py`
  - Behavior: wrapped in `try/except`; if missing, we just use no anchor terms.
- `programs.batch_generator.form_planning.plan` and `programs.batch_generator.form_planning.composite`
  - Used by: `src/programs/batch_generator/orchestrator.py`
  - Behavior: imported inside `try/except`; if missing, we fall back to simpler behavior.
