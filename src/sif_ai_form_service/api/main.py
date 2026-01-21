from __future__ import annotations

from fastapi import FastAPI
import warnings

# Load environment variables from .env and .env.local files if they exist
# .env.local takes precedence (loads after .env)
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env first
    load_dotenv(".env.local")  # Then load .env.local (overrides .env values)
except ImportError:
    pass  # python-dotenv not installed, skip

warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)

from .routes.form import router as form_router
from .routes.health import router as health_router
from .routes.image import router as image_router


def create_app() -> FastAPI:
    api_v1_prefix = "/v1"

    app = FastAPI(title="sif-ai-form-service", version="1.0.0")
    # Unversioned health is convenient for deployments and uptime checks.
    app.include_router(health_router)

    # v1 endpoints (committed OpenAPI contract).
    app.include_router(form_router, prefix=api_v1_prefix)
    app.include_router(image_router, prefix=api_v1_prefix)

    # Legacy (unversioned) endpoints for backward compatibility.
    app.include_router(form_router, include_in_schema=False)
    app.include_router(image_router, include_in_schema=False)
    return app


app = create_app()
