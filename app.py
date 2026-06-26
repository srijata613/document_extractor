from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from routes.upload import upload_router
from utils.logger import app_logger
from schemas.api import (
    HealthResponse,
    RootResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    app_logger.info(
        "Starting Document Extractor API..."
    )

    yield

    app_logger.info(
        "Shutting down Document Extractor API..."
    )


app = FastAPI(
    title="Document Extractor API",
    description=(
        "Production document extraction service "
        "supporting OCR, handwriting recognition, "
        "checkbox detection and Excel export."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(
    upload_router,
)


@app.get(
    "/",
    response_model=RootResponse,
    tags=["Health"],
)
async def root():

    return {
        "service": "Document Extractor API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
async def health():

    return JSONResponse(
        {
            "status": "healthy",
        }
    )