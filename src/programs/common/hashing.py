from __future__ import annotations

import hashlib


def short_hash(text: str, *, n: int = 10) -> str:
    """
    Stable short SHA-256 prefix for cache keys and fingerprints.
    """
    t = str(text or "")
    if not t:
        return "none"
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[: max(6, min(24, int(n or 10)))]


__all__ = ["short_hash"]

