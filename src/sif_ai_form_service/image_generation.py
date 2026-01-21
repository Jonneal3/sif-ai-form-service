from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ImageGenerationResult:
    images: List[str]
    metrics: Dict[str, Any]


def _mock_svg_data_url(text: str, seed: str) -> str:
    safe = (text or "").strip().replace("\n", " ")
    safe = safe[:120] + ("…" if len(safe) > 120 else "")
    safe_xml = (
        safe.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024">
  <rect width="100%" height="100%" fill="#111827"/>
  <text x="50%" y="46%" dominant-baseline="middle" text-anchor="middle"
        font-family="ui-sans-serif, system-ui, -apple-system" font-size="28" fill="#E5E7EB">
    mock image
  </text>
  <text x="50%" y="54%" dominant-baseline="middle" text-anchor="middle"
        font-family="ui-monospace, SFMono-Regular, Menlo" font-size="16" fill="#9CA3AF">
    {safe_xml}
  </text>
  <text x="50%" y="62%" dominant-baseline="middle" text-anchor="middle"
        font-family="ui-monospace, SFMono-Regular, Menlo" font-size="14" fill="#6B7280">
    seed:{seed}
  </text>
</svg>"""
    # URL-escape the SVG so the data URL is safe to embed.
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg, safe=",:;%/()[]@!$&'*+?=#-_~.")


def generate_images(
    *,
    prompt: str,
    num_variants: int = 1,
    provider: str = "mock",
    size: Optional[str] = None,
    return_format: str = "url",
    metadata: Optional[Dict[str, Any]] = None,
) -> ImageGenerationResult:
    """
    Provider-agnostic image generation entrypoint.

    Currently supports:
    - provider='mock': returns SVG data URLs (no network).
    """
    start = time.time()
    provider_norm = (provider or "mock").strip().lower()
    if provider_norm != "mock":
        raise NotImplementedError(
            f"IMAGE_PROVIDER '{provider}' not implemented in this repo yet. Use provider='mock' for local/dev."
        )
    if return_format not in {"url", "b64"}:
        raise ValueError("returnFormat must be 'url' or 'b64'")

    images: List[str] = []
    for i in range(max(1, int(num_variants))):
        seed = sha256(f"{prompt}|{size}|{i}".encode("utf-8")).hexdigest()[:12]
        images.append(_mock_svg_data_url(prompt, seed))

    latency_ms = int((time.time() - start) * 1000)
    metrics: Dict[str, Any] = {
        "qualityScore": 0.5,
        "generationTimeMs": latency_ms,
        "provider": provider_norm,
        "size": size,
    }
    if metadata:
        metrics["metadata"] = metadata
    return ImageGenerationResult(images=images, metrics=metrics)
