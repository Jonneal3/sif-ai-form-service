"""
Root entrypoint shim.

The implementation lives in `src/app/runtime/flow_planner.py`.
This file stays at repo root so existing tooling (and imports like `import flow_planner`)
keep working.
"""

from app.runtime import flow_planner as _impl

globals().update(_impl.__dict__)

