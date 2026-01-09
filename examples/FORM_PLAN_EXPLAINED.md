# What is `formPlan`?

## TL;DR
**`formPlan` is NOT the list of component types.** It's an AI-generated question plan from batch 1.

## What it actually is:

`formPlan` is an array of `FormPlanItem` objects that represent **planned questions** that the AI wants to ask. It's generated in **batch 1** and used in **batch 2** to know which questions to ask.

## Structure:

```json
{
  "formPlan": [
    {
      "key": "budget_range",
      "goal": "What's your budget range?",
      "why": "Helps tailor recommendations",
      "component_hint": "slider",  // <-- This is just a HINT, not the component type list
      "priority": "high",
      "importance_weight": 0.15,
      "expected_metric_gain": 0.12
    }
  ]
}
```

## Where component types actually are:

Component types (the list of possible mini schemas) are in:
```json
{
  "currentBatch": {
    "allowedComponentTypes": ["choice", "text", "slider", "file_upload"]
  }
}
```

## Flow:

1. **Batch 1**: 
   - `formPlan: null` (AI generates it)
   - AI produces `producedFormPlan` in response
   - Frontend saves it

2. **Batch 2**:
   - `formPlan: [...]` (contains the plan from batch 1)
   - AI uses it to know which questions to ask
   - AI generates steps for items in `formPlan` that aren't in `askedStepIds`

## Key distinction:

- **`formPlan`** = Planned questions (what to ask)
- **`allowedComponentTypes`** = Available component types (how to ask)

