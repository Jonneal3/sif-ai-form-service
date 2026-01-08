from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    # api/contract.py -> api/ -> repo root
    return Path(__file__).resolve().parents[1]


def contract_dir() -> Path:
    return _repo_root() / "shared" / "ai-form-contract"


def schema_version() -> str:
    # Env override is useful for emergency mismatches.
    import os

    v = (os.getenv("AI_FORM_SCHEMA_VERSION") or "").strip()
    if v:
        return v
    p = contract_dir() / "schema" / "schema_version.txt"
    try:
        return p.read_text(encoding="utf-8").strip() or "0"
    except Exception:
        return "0"


def ui_step_schema_json() -> Optional[Dict[str, Any]]:
    p = contract_dir() / "schema" / "ui_step.schema.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


