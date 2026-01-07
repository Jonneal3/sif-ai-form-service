# sif-ai-form-service

Python microservice for the SIF AI form flow. This service runs the DSPy planner and exposes a
streaming SSE endpoint that emits `mini_step` events and one final `meta` event.

## Endpoints

- `GET /health`
- `POST /flow/new-batch/stream` (SSE)
- `POST /flow/new-batch` (non-streaming debug)

## Local dev

Set env vars:
- `DSPY_PROVIDER=groq`
- `GROQ_API_KEY=...`
- `DSPY_MODEL_LOCK=llama-3.3-70b-versatile`

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

Test streaming:

```bash
curl -N -X POST http://localhost:8008/flow/new-batch/stream \
  -H 'content-type: application/json' \
  -d '{"mode":"next_steps","batchId":"ContextCore","platformGoal":"test","businessContext":"test","industry":"General","service":"","groundingPreview":"{}","requiredUploads":[],"personalizationSummary":"","stepDataSoFar":{},"alreadyAskedKeys":[],"formPlan":[],"batchState":{},"allowedMiniTypes":["multiple_choice"],"maxSteps":3}'
```

