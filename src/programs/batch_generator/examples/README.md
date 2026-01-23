# Batch generator examples

This folder is split into:

- `current/`: examples that match the current request/response shapes in this repo.
  - **`structural_examples.pretty.json`**: Human-readable format for editing (pretty JSON)
  - **`structural_examples.jsonl`**: Compact format for DSPy use (JSONL)
  - **`batch_generator_structural_examples.jsonl`**: Structural demos for the batch-config layer (`BatchGeneratorJSON`)
  - Use `convert_examples.py` to convert between formats (see `current/README.md`)
- `deprecated/`: older examples kept for reference.

## Vertical-agnostic examples for DSPy

**Critical**: When training DSPy optimizers, examples must be **vertical-agnostic** to prevent the optimizer from baking industry-specific content into the compiled program.

See `current/README.md` for:
- Why vertical-agnostic examples matter
- What to teach (structure) vs. what NOT to teach (vertical content)
- How to use the sanitization utility
- Leak detection guardrails

### Quick start

1. **Use structural examples**:
   ```bash
   # Examples are in current/structural_examples.jsonl
   ```

2. **Sanitize real examples before optimization**:
   ```bash
   python src/programs/batch_generator/examples/sanitize_examples.py input.jsonl output.jsonl
   ```

3. **Generate more structural examples**:
   ```bash
   python src/programs/batch_generator/examples/generate_structural_examples.py 20 output.jsonl
   ```

4. **Check for leaks (CI guardrail)**:
   ```bash
   make check-example-leaks
   # or
   python scripts/check_example_leaks.py path/to/examples.jsonl
   ```

### Key principle

**Only let the optimizer see what you're okay "baking in forever": structure + behavior. Everything vertical stays runtime-only.**
