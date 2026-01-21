from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _ensure_import_paths() -> None:
    root = _repo_root()
    src = root / "src"
    for p in (root, src):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _canonical_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, indent=2) + "\n"


def verify_openapi_contract(*, contract_path: Path) -> int:
    _ensure_import_paths()
    from planner_api.api.main import create_app

    expected = _canonical_json(create_app().openapi())
    try:
        actual = contract_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"[openapi-contract] Missing: {contract_path}")
        print("[openapi-contract] Run: python3 scripts/export_openapi_contract.py")
        return 2

    if actual == expected:
        return 0

    diff = difflib.unified_diff(
        actual.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile=str(contract_path),
        tofile="generated:openapi.json",
    )
    print("[openapi-contract] MISMATCH: committed OpenAPI differs from generated OpenAPI.")
    print("[openapi-contract] Run: python3 scripts/export_openapi_contract.py")
    print("".join(list(diff)[:4000]))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the committed OpenAPI contract matches the code.")
    parser.add_argument(
        "--contract",
        default=str(_repo_root() / "api-contract" / "openapi.json"),
        help="Path to the committed OpenAPI JSON file.",
    )
    args = parser.parse_args()
    return verify_openapi_contract(contract_path=Path(args.contract))


if __name__ == "__main__":
    raise SystemExit(main())
