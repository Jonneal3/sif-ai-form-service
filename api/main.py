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

from api.routes.form import router as form_router
from api.routes.health import router as health_router
from api.routes.image import router as image_router
from api.routes.telemetry import router as telemetry_router


def create_app() -> FastAPI:
    app = FastAPI(title="sif-ai-form-service", version="0.1.0")
    app.include_router(health_router)
    app.include_router(form_router)
    app.include_router(image_router)
    app.include_router(telemetry_router)
    return app


app = create_app()
