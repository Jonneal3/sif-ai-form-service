from __future__ import annotations

import time
from typing import Any, Optional, Tuple, TypeVar

T = TypeVar("T")


def ttl_cache_get(cache: dict[str, Tuple[float, T]], cache_key: str) -> Optional[T]:
    if not cache_key:
        return None
    rec = cache.get(cache_key)
    if not rec:
        return None
    expires_at, value = rec
    if time.time() >= float(expires_at):
        cache.pop(cache_key, None)
        return None
    return value


def ttl_cache_set(cache: dict[str, Tuple[float, T]], cache_key: str, value: T, *, ttl_sec: int) -> None:
    if not cache_key:
        return
    ttl = max(60, min(3600, int(ttl_sec or 0)))
    cache[cache_key] = (time.time() + ttl, value)


__all__ = ["ttl_cache_get", "ttl_cache_set"]

