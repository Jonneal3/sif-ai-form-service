# API Contract (OpenAPI)

This folder contains the committed OpenAPI spec for this service.

**Source of truth:** `api-contract/openapi.json`

## Update the contract

```bash
python3 scripts/export_openapi_contract.py
```

## Verify the contract (CI)

```bash
python3 scripts/verify_openapi_contract.py
```

## Frontend type generation (recommended)

Use `api-contract/openapi.json` as the input to a generator like `orval` or `openapi-generator` in the frontend repo.

