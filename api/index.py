"""
Vercel entrypoint (keep this file stable).

Vercel routes all requests to `api/index.py` (see `vercel.json`), so this file simply re-exports
the FastAPI `app` from the installable package.
"""

from planner_api.api.main import app  # noqa: F401
