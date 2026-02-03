# Pricing examples

This folder contains **example inputs + outputs** for the pricing estimator.

## Input shape

These examples are shaped like the request body for:

- `POST /v1/api/pricing/{instanceId}`

The endpoint accepts the same shapes as step generation (see `NewBatchRequest`), including:

- top-level `sessionId`
- `stepDataSoFar` (answers)
- optional `answeredQA` (plain-English memory)
- optional `instanceContext` (industry/service)
- optional `serviceSummary` / `companySummary`

## Output shape (frontend contract)

The stable fields to consume from the pricing API response are:

```json
{
  "ok": true,
  "requestId": "pricing_1730000000000",
  "currency": "USD",
  "rangeLow": 12000,
  "rangeHigh": 28000,
  "confidence": "low"
}
```

Notes:

- `rangeLow` / `rangeHigh` are **integers** (minor units are not used; treat as whole dollars).
- The response may include extra keys (`basis`, `notes`, `inputs`) for debugging/telemetry; clients should ignore unknown fields.

