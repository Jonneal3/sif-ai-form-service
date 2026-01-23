from __future__ import annotations

from typing import Any, Dict


def to_next_steps_payload(*, instance_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the canonical NewBatchRequest body into the internal dict shape expected by
    `programs.batch_generator.orchestrator.next_steps_jsonl`.

    This is intentionally small: the API layer stays boring, and all request shaping lives here.
    """
    if not isinstance(body, dict):
        return {"session": {"instanceId": str(instance_id), "sessionId": ""}}

    out = dict(body)

    # Carry instance/session identifiers in a consistent place for downstream correlation.
    session_id = str(out.get("sessionId") or out.get("session_id") or "")
    out["session"] = {"instanceId": str(instance_id), "sessionId": session_id}

    # Canonical keys used by the generator.
    out.setdefault("stepDataSoFar", out.get("knownAnswers") or out.get("known_answers") or {})
    out.setdefault("askedStepIds", out.get("alreadyAskedKeys") or out.get("already_asked_keys") or [])

    # Preserve the OpenAPI field name as-is; orchestrator will read from `instanceContext`.
    if "instance_context" in out and "instanceContext" not in out:
        out["instanceContext"] = out.get("instance_context")

    return out

