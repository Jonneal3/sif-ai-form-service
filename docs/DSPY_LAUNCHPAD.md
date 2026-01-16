## DSPy Launchpad (how this repo is wired)

### Big picture

This repo is a **FastAPI microservice** that exposes a planner endpoint. The planner is powered by **DSPy**.

You can think of it as:
- **HTTP boundary** (`api/index.py`): receives JSON, returns JSON or SSE
- **Pipeline** (`src/app/pipeline/pipeline.py`): builds a DSPy predictor and calls it
- **Signature(s)** (`src/app/signatures/json_signatures.py`): define the input/output contracts
- **Validation** (Pydantic models in `src/app/schemas/ui_steps.py`): enforce schema correctness after the LLM responds
- **Examples / demos** (`examples/*.jsonl`): few-shot guidance for DSPy predictors
- **Eval + optimize** (`eval/*`): tooling to measure invariants and compile improved demo sets

### Where the LLM call happens

The only “real model call” is in `src/app/pipeline/form_pipeline.py` (invoked via `src/app/pipeline/pipeline.py`):
- `predictor = dspy.Predict(NextStepsJSONL)`
- then `predictor(...)`

DSPy v3 uses a LiteLLM-backed `dspy.LM(...)` internally, configured from env vars:
- `DSPY_PROVIDER=groq|openai`
- `GROQ_API_KEY=...` or `OPENAI_API_KEY=...`
- `DSPY_MODEL` / `DSPY_MODEL_LOCK`

### DSPy concepts used here (minimal set)

- **Signature**: a contract describing fields. In this repo:
  - `NextStepsJSONL` is the core signature.
  - Inputs are kept minimal via a compact `context_json` string; outputs are strings containing JSON/JSONL.
- **Predict**: `dspy.Predict(Signature)` turns the signature into an LLM-backed callable.
- **Demos**: examples attached via `predictor.demos = [...]`.
  - This repo stores demos in JSONL (`examples/next_steps/next_steps_examples.jsonl`) and loads them at runtime.
  - Override demo pack with `DSPY_NEXT_STEPS_DEMO_PACK`.

### Why outputs are strings (then parsed + validated)

LLMs output text. Asking for “structured objects” often still returns text under the hood.
So we make it explicit:
- The signature output is `mini_steps_jsonl: str`
- We parse each line as JSON
- We validate each object with Pydantic

This gives you:
- deterministic failures (invalid items get dropped)
- clear debugging points (raw model output vs validated output)

### Eval + optimize workflow

- Run eval (invariant checks):

```bash
python -m eval.run_eval
```

- Compile an optimized demo pack (best-effort across DSPy v3 APIs):

```bash
python -m eval.optimize --cases eval_cases.jsonl --out-pack examples/next_steps/next_steps_examples.optimized.jsonl
export DSPY_NEXT_STEPS_DEMO_PACK=examples/next_steps/next_steps_examples.optimized.jsonl
```

The eval metrics in this repo are intentionally “structural” (schema + determinism), not “semantic”.
That’s a good first step for productionizing: correctness first, quality next.
