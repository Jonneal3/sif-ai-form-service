"""
Vercel entrypoint (keep this file stable).

Vercel routes all requests to `api/index.py` (see `vercel.json`), so this file re-exports
the FastAPI `app`.
"""

from .main import app  # noqa: F401
