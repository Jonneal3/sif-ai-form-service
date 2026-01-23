from __future__ import annotations

import sys


def main() -> int:
    try:
        from api.openapi_contract import load_openapi_spec

        spec = load_openapi_spec()
    except Exception as e:
        print(f"Error: failed to load OpenAPI spec: {e}", file=sys.stderr)
        return 2

    paths = (spec.get("paths") or {}) if isinstance(spec, dict) else {}
    if "/api/ai-form/{instanceId}/new-batch" not in paths:
        print("Error: OpenAPI spec missing /api/ai-form/{instanceId}/new-batch", file=sys.stderr)
        return 2

    try:
        from api.openapi_contract import validate_new_batch_request, validate_new_batch_response

        validate_new_batch_request({"sessionId": "sess_test"})
        validate_new_batch_response({"requestId": "req_test", "schemaVersion": None, "miniSteps": [], "lmUsage": None})
    except Exception as e:
        print(f"Error: OpenAPI validators failed: {e}", file=sys.stderr)
        return 2

    print("OK: service OpenAPI contract validators load and run", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

