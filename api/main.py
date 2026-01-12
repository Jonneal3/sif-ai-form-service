from __future__ import annotations

from fastapi import FastAPI

from api.routes.form import router as form_router
from api.routes.health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(title="sif-ai-form-service", version="0.1.0")
    app.include_router(health_router)
    app.include_router(form_router)
    return app


app = create_app()

