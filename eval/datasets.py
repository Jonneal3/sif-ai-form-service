from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EvalCase:
    name: str
    payload: Dict[str, Any]
    expected: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


def _eval_dir() -> str:
    return os.path.join(os.path.dirname(__file__))


def load_eval_cases(filename: str = "eval_cases.jsonl") -> List[EvalCase]:
    """
    Load evaluation cases from `eval/<filename>` as JSONL.

    Each line: {"name": str, "payload": {...}, "expected": {...?}, "tags": [...?]}
    """
    path = os.path.join(_eval_dir(), filename)
    if not os.path.exists(path):
        return []

    cases: List[EvalCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = (line or "").strip()
            if not t:
                continue
            obj = json.loads(t)
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("name") or "").strip()
            payload = obj.get("payload")
            if not name or not isinstance(payload, dict):
                continue
            expected = obj.get("expected") if isinstance(obj.get("expected"), dict) else None
            tags_raw = obj.get("tags")
            tags = [str(x) for x in tags_raw] if isinstance(tags_raw, list) else None
            cases.append(EvalCase(name=name, payload=payload, expected=expected, tags=tags))

    return cases


