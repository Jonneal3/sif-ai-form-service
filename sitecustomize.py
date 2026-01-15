"""
Local path bootstrap for the `src/` layout.

Python auto-imports `sitecustomize` (if present on `sys.path`) during startup.
This keeps local dev, scripts, and serverless runtimes working without needing
an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_src = _root / "src"
if _src.exists():
    sys.path.insert(0, str(_src))

