"""
Vercel entrypoint (keep this file stable).

Vercel routes all requests to `api/index.py` (see `vercel.json`), so this file simply re-exports
the FastAPI `app` from our "proper" layout in `api/main.py`.
"""

from api.main import app  # noqa: F401


