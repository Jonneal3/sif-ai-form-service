"""
Lightweight leak checker for compiled artifacts and demo packs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple


DEFAULT_FORBIDDEN = {
    "real estate",
    "medical",
    "legal",
    "insurance",
    "mortgage",
    "bank",
    "clinic",
    "hospital",
    "law firm",
    "attorney",
    "dental",
    "restaurant",
    "hotel",
    "city",
    "state",
}

PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"\b\+?\d[\d\s().-]{7,}\b")
ADDRESS_RE = re.compile(r"\b\d{1,5}\s+\w+(?:\s+\w+){0,3}\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Dr|Drive)\b", re.IGNORECASE)

ALLOWED_PROPER_NOUNS = {
    "JSON",
    "JSONL",
    "DSPy",
    "Vercel",
    "Next",
    "FastAPI",
    "Pydantic",
}


def _iter_files(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend([x for x in path.rglob("*") if x.is_file()])
        elif path.is_file():
            files.append(path)
    return files


def _scan_text(text: str) -> List[str]:
    hits: List[str] = []
    lower = text.lower()
    for term in DEFAULT_FORBIDDEN:
        if term in lower:
            hits.append(f"forbidden:{term}")
    for match in PROPER_NOUN_RE.findall(text):
        if match not in ALLOWED_PROPER_NOUNS:
            hits.append(f"proper_noun:{match}")
    if EMAIL_RE.search(text):
        hits.append("email")
    if PHONE_RE.search(text):
        hits.append("phone")
    if ADDRESS_RE.search(text):
        hits.append("address")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["compiled", "eval/datasets"])
    args = parser.parse_args()

    files = _iter_files(args.paths)
    violations: List[Tuple[str, List[str]]] = []

    for path in files:
        if path.suffix.lower() not in {".json", ".jsonl", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        hits = _scan_text(text)
        if hits:
            violations.append((str(path), sorted(set(hits))))

    if violations:
        print("Leak check failed:")
        for path, hits in violations:
            print(f"- {path}: {', '.join(hits)}")
        return 1

    print("Leak check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
