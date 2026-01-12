# Templates / Defaults

This folder contains **deterministic static defaults** that the backend uses when the frontend doesn't provide them.

## Structure

- **`default_prompt_context.json`** - Default prompt context (goal, business context, industry, service)
  - Used when frontend doesn't send `prompt` section
  - Backend fetches from Supabase, but these are fallback defaults

- **`default_form_psychology.json`** - Default psychology approach (e.g., "Escalation Ladder")
  - Used for DSPy prompt generation
  - Defines the psychological flow/strategy for question ordering

- **`default_form_copy.json`** - Default copy style and principles (e.g., "Humanism (Ryan Levesque)")
  - Used for DSPy prompt generation
  - Defines tone, style, and writing principles

- **`default_batch_config.json`** - Batch constraints and capabilities (not flow-specific)
  - `maxBatches`: Maximum number of batches allowed (currently 2)
  - `satietyTarget`: Overall target satiety score (1.0 = 100%)
  - `stepsPerBatch`: Range of steps allowed per batch (min: 3, max: 10)
  - `availableComponentTypes`: All component types available in the system (the full catalog)
  - `maxTokensPerBatch`: Maximum tokens per batch
  - **This is about constraints/capabilities - what's POSSIBLE, not what to USE**

- **`default_form_psychology.json`** - Form psychology approach and batch progression
  - `approach`: Psychology strategy (e.g., "Escalation Ladder")
  - `batchProgression`: Defines batch IDs, progression, and per-batch strategy
    - `batch1` (ContextCore): First batch strategy
      - `allowedComponentTypes: ["choice"]` - Start simple (psychology: reduce friction)
      - `satietyTarget: 0.77` - First batch goal
      - `maxSteps: 5` - Constraint for this batch
    - `batch2` (PersonalGuide): Second batch strategy
      - `allowedComponentTypes: ["choice", "text", "slider", "file_upload"]` - Escalate to complex (psychology: user is engaged)
      - `satietyTarget: 1.0` - Final goal
      - `maxSteps: 8` - More steps allowed (user is committed)
  - **This is about flow/strategy - which batch ID to use, when, and HOW (including which component types per batch)**
  - **Component type escalation is a psychology decision - start simple, get complex as engagement builds**

## Usage

These are **not examples** - they're the actual defaults the backend falls back to.

**Examples vs Templates:**
- **`examples/`** - Sample requests/responses for documentation/testing
- **`templates/`** - Deterministic defaults used by the backend when frontend omits optional fields

## When Backend Uses These

1. **Prompt context**: If frontend doesn't send `prompt` section, backend:
   - First tries to fetch from Supabase (using `sessionId`)
   - Falls back to `default_prompt_context.json` if Supabase fetch fails

2. **Batch constraints**: If frontend doesn't send `currentBatch.maxSteps`, etc., backend:
   - Uses `default_batch_config.json` for overall constraints (maxBatches, stepsPerBatch range, availableComponentTypes catalog)
   - Uses `default_form_psychology.json[batchProgression.batch1/batch2]` for batch-specific strategy:
     - `batchId` - Which batch ID to use
     - `satietyTarget` - Target for this batch
     - `maxSteps` - Steps allowed for this batch
     - `allowedComponentTypes` - **Which component types to use THIS batch (psychology decision)**
   - Psychology determines **which batch ID**, **what strategy**, and **which component types** to use per batch

## Updating Defaults

These are **deterministic business logic defaults**. Update them when:

**Batch Config (`default_batch_config.json`)** - Constraints/capabilities:
- Change maximum number of batches allowed
- Change overall satiety target
- Change steps per batch range (min/max)
- Change available component types catalog (what's possible in the system)
- Change max tokens per batch

**Form Psychology (`default_form_psychology.json`)** - Flow/strategy:
- Change psychology approach (e.g., "Escalation Ladder" → "Progressive Disclosure")
- Change batch progression (which batch IDs, when to use them)
- Change per-batch strategy:
  - `satietyTarget` per batch
  - `maxSteps` per batch
  - **`allowedComponentTypes` per batch (psychology: which types to use when)**
- Add new batch types or remove batches
- Change component type escalation strategy (e.g., start with more types, or escalate faster)

**Prompt Context (`default_prompt_context.json`)** - Content:
- Change default prompt goal
- Change business context defaults

**Form Copy (`default_form_copy.json`)** - Writing style:
- Change copy style principles
- Update tone and writing guidelines
