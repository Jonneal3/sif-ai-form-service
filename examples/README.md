# API Request Examples

**Note:** These are **examples** for documentation/testing. For deterministic defaults used by the backend, see `../templates/`.

## Structure Overview

The request body is organized into clear sections:

1. **`session`** - Identifiers (sessionId, instanceId)
2. **`currentBatch`** - Current batch info (batchId, batchNumber, constraints, targets)
3. **`state`** - Overall form state (aggregated answers, asked questions, satiety, form plan)
4. **`prompt`** - DSPy prompt context (optional - backend fetches from Supabase)
5. **`batches`** - Previous batches history (optional, for analytics only)
6. **`request`** - Request flags (optional, for debugging/versioning)

## No Duplication

- **`state.answers`** and **`state.askedStepIds`** = aggregated across ALL batches (source of truth)
- **`batches[]`** = per-batch breakdown (optional, for analytics/debugging only)
- **`currentBatch`** = info about the batch we're generating NOW

## What is `batches: []`?

The `batches` array is **optional** and contains history of previous batches. 

- **Batch 1**: `batches: []` (empty, no previous batches)
- **Batch 2**: `batches: [{batchNumber: 1, stepsAsked: [...], ...}]` (contains batch 1 history)

**Why have it?**
- Analytics: Track per-batch metrics (steps generated, satiety achieved)
- Debugging: See what happened in each batch
- **Note**: `state.answers` and `state.askedStepIds` already have the aggregated data, so `batches` is redundant but useful for per-batch analysis

**You can omit `batches` entirely** - it's optional. The backend only needs `state.answers` and `state.askedStepIds` to avoid duplicates.

## Files

### API request examples (docs/testing)

- `api_request_organized.json` - Batch 1 example (no previous batches)
- `api_request_batch2.json` - Batch 2 example (with batch 1 history)

### DSPy demo packs (few-shot)

- `flow_plan/flow_plan_examples.jsonl` - First-call flow plan demos (used only when no `state.formPlan` is provided)
- `next_steps/next_steps_examples.jsonl` - Main NextSteps demos
- `next_steps/next_steps_structural.jsonl` - Structural/edge-case demos
- `next_steps/next_steps_examples.optimized.jsonl` - Optimized/compiled-style demos
