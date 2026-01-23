# Current examples

Starter example payloads for `POST /v1/api/form` (batch generator / next-steps).

Run locally:
- `uvicorn api.main:app --reload --port 8000`
- `curl -sS -X POST http://localhost:8000/v1/api/form -H 'Content-Type: application/json' -d @<example.json> | jq`

## Vertical-agnostic examples (starting point)

- `src/programs/batch_generator/examples/current/widget_scene_batch1.json`
- `src/programs/batch_generator/examples/current/widget_scene_batch2.json`
- `src/programs/batch_generator/examples/current/widget_tryon_batch1.json`
- `src/programs/batch_generator/examples/current/widget_tryon_batch2.json`
- `src/programs/batch_generator/examples/current/widget_scene_placement_batch1.json`
- `src/programs/batch_generator/examples/current/widget_scene_placement_batch2.json`

## Structural examples for DSPy optimization

**Critical**: When training DSPy optimizers, examples must be **vertical-agnostic** to prevent the optimizer from baking industry-specific content into the compiled program.

Additional structural demo pack (batch-config layer):
- `src/programs/batch_generator/examples/current/batch_generator_structural_examples.jsonl` (targets `BatchGeneratorJSON` → `batches_json`)
- `src/programs/batch_generator/examples/current/batch_generator_structural_examples.pretty.json` (same demos, but human-readable)

Note: this pack is phase-id agnostic; phases are represented by list order (`phaseIndex`). These demos are optimizer-safe (constraints + mechanical guidance only; no intent language).
Authoritative rules for these demos live in `src/programs/batch_generator/examples/current/batch_generator_examples_rules.md`.

### Why vertical-agnostic examples?

DSPy "compile" can end up baking demos + prompt text that reflect whatever you trained on. If you optimize on "pool-company" examples, you can accidentally teach the system "pool-ish" phrasing forever.

**The fix**: Make your training signal **structural, not vertical**.

### What to teach (structure & behavior)

Your examples should primarily teach:

✅ **Valid JSONL formatting** - proper JSONL structure, one step per line  
✅ **Deterministic IDs** - `step-{key}` pattern, consistent naming  
✅ **Allowed type values** - `multiple_choice`, `text_input`, `rating`, etc.  
✅ **Option object shape** - `{label, value}` structure  
✅ **Question style constraints** - length limits, one-thing-at-a-time  
✅ **Don't repeat `already_asked_keys`** - skip steps already asked  
✅ **Don't ask for things in `known_answers`** - respect prior answers  
✅ **Respect `max_steps`** - hard limit on step count  
✅ **Respect `allowed_mini_types`** - only emit allowed component types  

### What NOT to teach (vertical content)

❌ **Industry nouns** - "patio", "kitchen remodel", "pool depth"  
❌ **Vertical-specific option sets** - domain-specific choices  
❌ **Domain facts** - industry knowledge, material names, measurements  
❌ **Business names or locations** - real company names, cities, etc.  

### Using generic placeholders

If you need semantics, use generic placeholders:

- `"Product A" / "Service B"` instead of `"Pool installation" / "Kitchen remodel"`
- `"Project Type 1/2/3"` instead of `"New install / Replace / Repair"`
- `"Material Option A/B/C"` instead of `"Tile / Concrete / Wood"`
- `"Style A/B/C"` instead of `"Modern / Traditional / Rustic"`
- `"Vertical_01"`, `"Service_A"` instead of `"Pool Company"`, `"Landscaping"`

### Structural examples files

**Two formats available:**

1. **`structural_examples.pretty.json`** - Human-readable format for editing/reading
   - `context_json` is a nested dict (not a string)
   - `mini_steps_jsonl` is a list of step objects (not a JSONL string)
   - Easy to read, edit, and understand

2. **`structural_examples.jsonl`** - Compact format for DSPy use
   - `context_json` is a JSON string (as required by DSPy signature)
   - `mini_steps_jsonl` is a JSONL string (one step per line)
   - Ready to use in DSPy optimizers

**Convert between formats:**
```bash
# Pretty -> Compact (before using in DSPy)
python src/programs/batch_generator/examples/convert_examples.py \
  structural_examples.pretty.json \
  structural_examples.jsonl

# Compact -> Pretty (for editing)
python src/programs/batch_generator/examples/convert_examples.py \
  structural_examples.jsonl \
  structural_examples.pretty.json

# Same converter also supports the batch-config demo pack:
python src/programs/batch_generator/examples/convert_examples.py \
  batch_generator_structural_examples.pretty.json \
  batch_generator_structural_examples.jsonl
```

**Workflow:**
1. Edit `structural_examples.pretty.json` (human-readable)
2. Convert to compact format when ready to use
3. Use `structural_examples.jsonl` in DSPy optimizers

### Example patterns

The examples teach these structural patterns:

- `structural_basic_choices` - Valid JSONL format, option shape
- `structural_skip_already_asked` - Respect `already_asked_keys`
- `structural_skip_known_answer` - Don't ask for things in `known_answers`
- `structural_max_steps_one` - Respect `max_steps` limit
- `structural_text_only` - Respect `allowed_mini_types`
- `structural_rating` - Rating type format
- `structural_choice_then_text` - Mixed types, proper JSONL
- `structural_skip_duplicates` - Skip duplicates even if in `form_plan`
- `structural_wrap_up` - Final batch behavior
- `structural_question_length` - Question style constraints

### Sanitization utility

Before using real examples for optimization, sanitize them:

```bash
python src/programs/batch_generator/examples/sanitize_examples.py input.jsonl output.jsonl
```

The sanitizer:
- Replaces vertical terms with generic placeholders
- Normalizes industry/service fields to `Vertical_XX` / `Service_X`
- Sanitizes question text, labels, and options
- Converts keys to `attribute_X` pattern

### Leak detection

Before committing examples, check for vertical leaks:

```python
from programs.batch_generator.examples.sanitize_examples import check_example_for_leaks

leaks = check_example_for_leaks(example)
if leaks:
    print(f"Leaks detected: {leaks}")
```

Forbidden terms include: pool, patio, kitchen, bathroom, landscaping, tile, concrete, wood, etc.

### Best practices workflow

1. **Generate structural examples** - Use `generate_structural_examples.py` to create 10-20 structural demos
2. **Sanitize any real examples** - Run sanitizer on production examples before optimization
3. **Run optimizer** - Compile DSPy program on sanitized/structural examples only
4. **CI leak test** - Add guardrail that scans compiled artifacts for forbidden vocabulary
5. **In production** - Inject real vertical context via `context_json` at runtime (RAG/template), but never store it as demos

### Key principle

**Only let the optimizer see what you're okay "baking in forever": structure + behavior. Everything vertical stays runtime-only.**
