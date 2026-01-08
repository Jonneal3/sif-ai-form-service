## AI Form Contract (shared)

This folder is the **UI Contract** (the "Real Schema") shared between:
- the **DSPy service** (this repo) which *generates* UI steps, and
- the **Next.js UI repo** which *renders* UI steps.

### Real UI Schema vs. Internal Mini-Schema
- **Real UI Schema** (this folder): The rich objects the UI needs to render.
- **Internal Mini-Schema** (service-only): A token-optimized format used ONLY inside the DSPy service to talk to the LLM. The UI never sees it.

### Folder layout
- `schema/schema_version.txt`: canonical schema version string
- `schema/ui_step.schema.json`: JSON Schema for UIStep (discriminated union by `type`)
- `schema/ui_step.types.ts`: TypeScript mirror of UIStep for UI consumption
- `demos/next_steps_examples.jsonl`: DSPy demo pack (JSONL)

### Update workflow (high level)
1. Change UI components (if needed)
2. Update the Pydantic models in the service and run `python scripts/export_contract.py`
3. Bump `schema/schema_version.txt`
4. Update `demos/next_steps_examples.jsonl` to reflect the new shape
5. Deploy UI + service together (or enforce schemaVersion compatibility)

### Next.js consumption
- **Types**: import from `shared/ai-form-contract/schema/ui_step.types.ts`
- **Runtime drift detection**: call `GET /api/form/capabilities` and compare `schemaVersion`


