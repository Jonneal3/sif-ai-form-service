from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "service": "sif-ai-form-service", "ts": int(time.time() * 1000)}


