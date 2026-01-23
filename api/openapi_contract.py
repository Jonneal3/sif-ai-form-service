from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import jsonschema


_SPEC_PATH = Path(__file__).resolve().parents[1] / "shared" / "ai-form-service-openapi" / "openapi.json"


def _deref_json_pointer(root: Dict[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        raise ValueError(f"Unsupported $ref (only '#/' supported): {ref}")
    cur: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Broken $ref pointer: {ref}")
        cur = cur[part]
    return cur


def _resolve_refs(obj: Any, root: Dict[str, Any], memo: Dict[int, Any]) -> Any:
    if obj is None:
        return None
    oid = id(obj)
    if oid in memo:
        return memo[oid]
    if isinstance(obj, dict):
        if "$ref" in obj and isinstance(obj["$ref"], str):
            resolved = _deref_json_pointer(root, obj["$ref"])
            out = _resolve_refs(resolved, root, memo)
            memo[oid] = out
            return out
        out: Dict[str, Any] = {}
        memo[oid] = out
        for k, v in obj.items():
            out[k] = _resolve_refs(v, root, memo)
        return out
    if isinstance(obj, list):
        out_list: list[Any] = []
        memo[oid] = out_list
        for item in obj:
            out_list.append(_resolve_refs(item, root, memo))
        return out_list
    return obj


@lru_cache(maxsize=1)
def load_openapi_spec() -> Dict[str, Any]:
    if not _SPEC_PATH.exists():
        raise FileNotFoundError(f"OpenAPI spec not found at {_SPEC_PATH}")
    return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _validators() -> Tuple[jsonschema.Validator, jsonschema.Validator]:
    spec = load_openapi_spec()

    path_item = (spec.get("paths") or {}).get("/v1/api/form/{instanceId}") or {}
    post = path_item.get("post") if isinstance(path_item, dict) else {}
    if not isinstance(post, dict):
        raise ValueError("OpenAPI spec missing POST /v1/api/form/{instanceId}")

    req_schema = (
        (((post.get("requestBody") or {}).get("content") or {}).get("application/json") or {}).get("schema") or {}
    )
    resp_schema = (
        ((((post.get("responses") or {}).get("200") or {}).get("content") or {}).get("application/json") or {}).get("schema")
        or {}
    )
    if not isinstance(req_schema, dict) or not req_schema:
        raise ValueError("OpenAPI spec missing requestBody schema for /v1/api/form/{instanceId}")
    if not isinstance(resp_schema, dict) or not resp_schema:
        raise ValueError("OpenAPI spec missing 200 response schema for /v1/api/form/{instanceId}")

    resolved_req = _resolve_refs(req_schema, spec, memo={})
    resolved_resp = _resolve_refs(resp_schema, spec, memo={})

    req_validator = jsonschema.Draft202012Validator(resolved_req)
    resp_validator = jsonschema.Draft202012Validator(resolved_resp)
    return req_validator, resp_validator


def validate_new_batch_request(body: Any) -> None:
    req_validator, _ = _validators()
    errors = sorted(req_validator.iter_errors(body), key=lambda e: list(e.path))
    if errors:
        e0 = errors[0]
        path = "/".join(str(p) for p in e0.path) or "<root>"
        raise ValueError(f"Request does not match OpenAPI schema at {path}: {e0.message}")


def validate_new_batch_response(body: Any) -> None:
    _, resp_validator = _validators()
    errors = sorted(resp_validator.iter_errors(body), key=lambda e: list(e.path))
    if errors:
        e0 = errors[0]
        path = "/".join(str(p) for p in e0.path) or "<root>"
        raise ValueError(f"Response does not match OpenAPI schema at {path}: {e0.message}")

