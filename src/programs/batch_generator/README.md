# `programs.batch_generator` (simple overview)

This folder generates the next UI questions (“steps”) for a form.

Input: a Python dict payload (already shaped by the API layer)  
Output: a Python dict response that includes `miniSteps[]`

## Control flow (in order)

1. `orchestrator.py::next_steps_jsonl(payload)`
2. `orchestrator.py::_prepare_predictor(payload)`
   - Builds a `context` dict (by calling `_build_context(payload)`).
   - Turns that dict into a string: `context_json`.
   - Chooses `max_steps` and `allowed_mini_types`.
   - Creates a DSPy program (`BatchStepsProgram`) and attaches demos (examples) if present.
3. DSPy calls the model:
   - `BatchStepsProgram.forward(context_json, max_steps, allowed_mini_types)`
4. The model returns `mini_steps_jsonl` (a string).
5. `next_steps_jsonl` parses the string line-by-line as JSON.
6. Each step is validated and filtered (bad steps are dropped).
7. The final response is returned with `miniSteps[]`.

## What each file does

### `orchestrator.py`

Runs the whole pipeline.

- Builds `context_json` (the input to the model).
- Calls the model (through DSPy).
- Parses model output (JSONL string).
- Validates + filters steps.
- Returns `{ requestId, schemaVersion, miniSteps, ... }`.

### `signatures/signature.py`

Defines the DSPy “signature” (`BatchNextStepsJSONL`).

This declares:

- Inputs the model receives: `context_json`, `max_steps`, `allowed_mini_types`
- Output the model must return: `mini_steps_jsonl`

### `engine/dspy_program.py`

Defines the DSPy program wrapper (`BatchStepsProgram`).

- Wraps the signature using `dspy.Predict(...)`.
- Loads demos from `examples/next_steps_examples.jsonl` (optional).
- Makes the model callable from the orchestrator.

### `demos.py` and `examples/`

The demo file lives at:

- `examples/next_steps_examples.jsonl`

### `must_have_copy_module.py`

Optional second model call.

- Input: `context_json` + the batch steps (`mini_steps_jsonl`)
- Output: `must_have_copy_json`

### `planning/form_planning/`

Extra helpers that can add rules to the context (best-effort).

Examples:

- allowed UI step types
- pacing/stage rules
- copy style rules and linting

## Why this feels hard / messy

- The model output is text, so we must parse and validate it.
- We support optional features (copy packs, flow guide, demos).
- Some helpers are best-effort, so the flow has branches.

