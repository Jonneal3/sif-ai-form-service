"""
Local dev entrypoint.

Allows running `uvicorn api.main:app` from the repo root.
"""

from planner_api.api.main import app, create_app  # noqa: F401
