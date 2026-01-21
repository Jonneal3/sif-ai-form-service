from __future__ import annotations

import base64
import os
from typing import Any, Dict, List


def _svg_data_url(svg: str) -> str:
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def generate_images(*, prompt: str, num_outputs: int = 1, output_format: str = "url") -> List[Dict[str, Any]]:
    provider = str(os.getenv("IMAGE_PROVIDER") or "mock").lower()
    n = max(1, min(8, int(num_outputs or 1)))
    out: List[Dict[str, Any]] = []

    if provider != "mock":
        raise NotImplementedError(f"IMAGE_PROVIDER={provider!r} not implemented in this repo")

    safe = (prompt or "").strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for i in range(n):
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' width='1024' height='1024'>"
            "<rect width='100%' height='100%' fill='#111827'/>"
            "<text x='48' y='96' font-size='28' fill='#F9FAFB' font-family='ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas'>"
            f"mock image {i+1}/{n}"
            "</text>"
            "<text x='48' y='148' font-size='18' fill='#D1D5DB' font-family='ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas'>"
            f"{safe[:140]}"
            "</text>"
            "</svg>"
        )
        url = _svg_data_url(svg)
        out.append({"index": i, "format": output_format, "url": url})

    return out
