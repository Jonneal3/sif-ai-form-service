# Batch Generator (Batch Plan) — Example Rules (Optimizer-Safe)

These rules are the **authoritative contract** for creating DSPy demo examples for the batch-plan layer (`BatchGeneratorJSON` → `batches_json`).

Goal: teach **constraints and mechanical structure only**. Examples must be safe to “bake in forever”.

## Allowed schema (strict)

`outputs.batches_json` must parse as JSON and contain only:

- `version`: integer
- `constraints`: object with `{ maxBatches, maxStepsPerBatch, tokenBudgetTotal }`
- `batches`: array of phase objects

Each phase object may contain only:

- `phaseIndex`: integer (0-based, unique, sequential)
- `maxSteps`: integer
- `allowedComponentTypes`: array of strings (UI type enum)
- `guidance`: object with only:
  - `verbosity`: `low|medium|high`
  - `specificity`: `low|medium|high`
  - `breadth`: `broad|focused`

No other keys are allowed in demo outputs for this layer.

## Invariants (must always hold)

- Must respect `context_json.batch_constraints.maxBatches` when present.
- Must respect `context_json.batch_constraints.maxStepsPerBatch` as a hard ceiling.
- Must not exceed `maxBatches` even if `context_json` includes a longer prior preview.
- If `context_json.policy.noUploads = true`, `allowedComponentTypes` must not include `file_upload`.

## Banned language (intent leakage)

Examples must not encode “why” or product strategy. Reject demos containing these words (case-insensitive) anywhere in `meta`, `inputs`, or `outputs`:

- baseline
- pricing
- estimate
- clarify
- quantify
- scope
- roi
- signal
- ambiguity
- funnel
- final
- initial
- early
- middle
- late

If a human PM would nod reading the text, it’s too semantic for this layer.
