#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def _read_last_json_line(path: Path) -> dict | None:
    if not path.exists():
        return None
    last: dict | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = (line or "").strip()
            if not t:
                continue
            try:
                obj = json.loads(t)
            except Exception:
                continue
            if isinstance(obj, dict):
                last = obj
    return last


def main() -> None:
    path = Path("data/optimizer_runs.jsonl")
    last = _read_last_json_line(path)
    if not last:
        print(f"No runs found in {path}")
        return

    print(json.dumps(last, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

