"""
Root entrypoint shim.

The implementation lives in `src/app/runtime/image_planner.py`.
"""

from app.runtime import image_planner as _impl

globals().update(_impl.__dict__)
