from __future__ import annotations

import argparse
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


def _canonical_json_bytes(obj: Dict[str, Any]) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode("utf-8")


def export_openapi_contract(*, out_path: Path) -> None:
    # Import inside function so importing this module doesn't eagerly load FastAPI app.
    _ensure_import_paths()
    from sif_ai_form_service.api.main import create_app

    spec = create_app().openapi()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(_canonical_json_bytes(spec))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the FastAPI OpenAPI contract to a committed file.")
    parser.add_argument(
        "--out",
        default=str(_repo_root() / "api-contract" / "openapi.json"),
        help="Output path for the OpenAPI JSON file.",
    )
    args = parser.parse_args()

    export_openapi_contract(out_path=Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
