# sif-ai-form-service

Python microservice for the SIF AI form flow. This service runs the DSPy planner and returns
form steps as JSON (or SSE streaming when requested).

## Endpoints

- `GET /health`
- `POST /api/form` (JSON by default, SSE when `Accept: text/event-stream` or `?stream=1`)
- `GET /api/form/capabilities` (JSON; contract schema + version)
- `POST /api/image` (image prompt + image generation)
- `POST /api/telemetry` (AI form telemetry events)
- `POST /api/feedback` (dev/user feedback labels)

## Shared contract workflow (service + UI)

The canonical "UIStep contract" lives under `shared/ai-form-contract/`:
- `shared/ai-form-contract/schema/schema_version.txt`
- `shared/ai-form-contract/schema/ui_step.schema.json`
- `shared/ai-form-contract/schema/ui_step.types.ts`
- `shared/ai-form-contract/demos/next_steps_examples.jsonl`

**Shared module setup (local dev):** this repo’s `shared/ai-form-contract` is a symlink to
`/Users/jon/Desktop/sif-ai-form-contract`, which is the shared source of truth between this service and the widget.

When you change UI components (e.g., add a field to a slider/rating component):
- bump `schema_version.txt`
- run:

```bash
python3 scripts/export_contract.py
```

- update the demo pack JSONL if the shape changed
- deploy UI + service together (or compare versions using `GET /api/form/capabilities`)

## Local dev

**Quick start:** Copy `.env.example` to `.env` and fill in your values.

**Required env vars:**

**Supabase (for minimal API requests):**
- `SUPABASE_URL` or `NEXT_PUBLIC_SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Service role key (for backend access)

**DSPy (for LLM calls):**
- `DSPY_PROVIDER=groq` (or `openai`)
- `GROQ_API_KEY=...` (or `OPENAI_API_KEY=...`)
- `DSPY_MODEL_LOCK=llama-3.3-70b-versatile` (optional)

**Optional:**
- `DSPY_NEXT_STEPS_DEMO_PACK=next_steps_examples.jsonl` (or an optimized pack written by `eval/optimize.py`)
  - You can also point this to a repo-relative path (useful if you keep the pack in a shared git submodule), e.g. `shared/ai-form/next_steps_examples.jsonl`
- Contract (recommended): keep schema + demos in `shared/ai-form-contract/`
  - Default demo pack is `shared/ai-form-contract/demos/next_steps_examples.jsonl` if present
  - Schema version is read from `shared/ai-form-contract/schema/schema_version.txt`
- Image generation:
  - `IMAGE_PROVIDER=mock` (default; returns SVG data URLs)
  - `DSPY_IMAGE_PROMPT_MAX_TOKENS=900` (prompt-builder token cap)

If you’re learning DSPy, start here:
- `docs/DSPY_LAUNCHPAD.md`

Install + run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8008
```

Test health:

```bash
curl -s http://localhost:8008/health | jq
```

Test form generation (JSON):

```bash
curl -X POST http://localhost:8008/api/form \
  -H 'content-type: application/json' \
  -d '{"mode":"next_steps","batchId":"ContextCore","platformGoal":"test","businessContext":"test","industry":"General","service":"","requiredUploads":[],"personalizationSummary":"","stepDataSoFar":{},"alreadyAskedKeys":[],"formPlan":[],"batchState":{},"allowedMiniTypes":["multiple_choice"],"maxSteps":3}'
```

Test streaming (SSE):

```bash
curl -N -X POST 'http://localhost:8008/api/form?stream=1' \
  -H 'accept: text/event-stream' \
  -H 'content-type: application/json' \
  -d '{"mode":"next_steps","batchId":"ContextCore","platformGoal":"test","businessContext":"test","industry":"General","service":"","requiredUploads":[],"personalizationSummary":"","stepDataSoFar":{},"alreadyAskedKeys":[],"formPlan":[],"batchState":{},"allowedMiniTypes":["multiple_choice"],"maxSteps":3}'
```

Test image generation (JSON):

```bash
curl -X POST http://localhost:8008/api/image \
  -H 'content-type: application/json' \
  -d '{"instanceId":"uuid-here","useCase":"scene","numOutputs":2,"outputFormat":"url","stepDataSoFar":{"step-space-type":"kitchen","step-budget":"5000"},"config":{"platformGoal":"AI pre-design intake","businessContext":"We generate AI images for early design concepts","industry":"Interior Design","service":"Kitchen Remodel","personalizationSummary":"Bright, warm, natural materials"}}'
```

## Eval (metrics)

Run an invariant-based eval on a small golden set (requires your DSPy provider env vars set so the planner can run):

```bash
python -m eval.run_eval
```

Cases live in `eval/eval_cases.jsonl`.

You can run the feedback export → eval → optimizer flow in one command (or via `make refresh-feedback` / `make refresh-feedback-insights`):

```bash
python scripts/refresh_feedback_pipeline.py \
  --cases-out eval/feedback_cases.jsonl \
  --failures-out eval/feedback_failures.jsonl \
  --optimize-out examples/next_steps_examples.optimized.jsonl \
  --include-negative
```

It writes fresh eval cases from Supabase, runs `eval.run_eval` (pointing at the generated eval cases by default; override with `--dataset` if you have another JSONL), and (unless you pass `--skip-optimize`) refreshes the optimized demo pack for DSPy. Pass `--collect-insights` to also summarize the latest telemetry rows with `scripts/telemetry_insights.py` (it keeps a `.telemetry_checkpoint.json` so future runs only process increments and writes summaries to `data/telemetry_summary.json`).

Pass `--optimizer-max-tokens <n>` (default is `$DSPY_NEXT_STEPS_MAX_TOKENS` or `1200`) to raise the optimizer’s LM budget and avoid the `max_tokens=900` truncation warnings.

### Daily optimizer refresh

Run the pipeline with `--collect-insights` every day to keep the telemetry summary and optimized demos fresh. Each run appends a timestamped entry to `data/optimizer_runs.jsonl`, so you can `make show-last-optimizer-run` (or `tail -n 1 data/optimizer_runs.jsonl`) to see when you last ran the optimizer, what eval/telemetry counts were processed, and which recent issues (dropoffs, low-scoring eval cases) were fed into the pack. The Makefile target also archives the generated pack/report/summary into `data/archives/optimizer_runs/`.

### Telemetry insights

Run `scripts/telemetry_insights.py` independently to keep a lightweight summary of the telemetry table (dropoffs, per-batch event counts, and feedback/rating aggregates). It only loads rows past the stored `.telemetry_checkpoint.json`, so you can call it on a schedule without reprocessing the entire history:

```bash
python scripts/telemetry_insights.py \
  --checkpoint .telemetry_checkpoint.json \
  --summary data/telemetry_summary.json \
  --limit 2000
```

The summary file accumulates batch-level totals, the most recent dropoff candidates, and feedback ratings while the checkpoint ensures the next run resumes where you left off. Use these insights to decide whether a batch needs new examples, copy tweaks, or UX changes (e.g., steps with repeated dropoffs).
The wrapper also automatically loads the repo’s `.env` file on invocation, so you don’t need to `source` it manually before running the command.


## Optimize (teleprompter/optimizer)

Generate an optimized demo pack (best-effort across DSPy v3 APIs; requires provider env vars):

```bash
python -m eval.optimize --cases eval_cases.jsonl --out-pack examples/next_steps_examples.optimized.jsonl
```

Then run the service using the optimized demos:

```bash
export DSPY_NEXT_STEPS_COMPILED=examples/next_steps_examples.optimized.jsonl
```

## Deploy to Vercel

This repo is set up as a Vercel **Python Serverless Function** with a catch-all route to `api/index.py`
(see `vercel.json`).

### Deployment Protection (401 Authentication Required)

If `GET /health` returns **401 Authentication Required**, your deployment is behind **Vercel Deployment Protection**
(sometimes shown as “Vercel Authentication”).

You have two options:

- **Make it public (recommended)**: Vercel Dashboard → Project → Settings → Deployment Protection → set **Production** to **Disabled/Public**.
- **Keep protection ON (automation bypass)**: enable **Protection Bypass for Automation** in the same settings page and use the
  generated bypass secret in your server-to-server requests:
  - Send header **`x-vercel-protection-bypass: <BYPASS_SECRET>`**
  - Vercel also exposes this secret to the deployment as **`VERCEL_AUTOMATION_BYPASS_SECRET`**

In Vercel Project Settings, set required env vars (same as local dev):
- `SUPABASE_URL` or `NEXT_PUBLIC_SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `DSPY_PROVIDER`
- `DSPY_MODEL_LOCK` (optional)
- `GROQ_API_KEY` (or `OPENAI_API_KEY` if using `DSPY_PROVIDER=openai`)

Deploy:
- **Git**: import the repo in Vercel and deploy from the dashboard
- **CLI**: from the repo root, run `vercel` (and `vercel --prod` when ready)

After deploy, verify:

```bash
curl -s https://YOUR_VERCEL_DOMAIN/health | jq
```

## Verify streaming on Vercel

```bash
curl -N -X POST 'https://YOUR_VERCEL_DOMAIN/api/form?stream=1' \
  -H 'accept: text/event-stream' \
  -H 'content-type: application/json' \
  -d '{"mode":"next_steps","batchId":"ContextCore","platformGoal":"test","businessContext":"test","industry":"General","service":"","requiredUploads":[],"personalizationSummary":"","stepDataSoFar":{},"alreadyAskedKeys":[],"formPlan":[],"batchState":{},"allowedMiniTypes":["multiple_choice"],"maxSteps":3}'
```
