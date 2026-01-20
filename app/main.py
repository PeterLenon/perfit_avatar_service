"""
FastAPI application entry point.

Configures the application, middleware, and routes.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import avatar, health
from app.config import get_settings
from app.models.database import close_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events:
    - Startup: Initialize database, warm up ML models
    - Shutdown: Close connections, cleanup resources
    """
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Initialize database tables (use migrations in production)
    if settings.environment == "development":
        logger.info("Initializing database tables...")
        await init_db()

    yield

    # Cleanup on shutdown
    logger.info("Shutting down...")
    await close_db()


def create_app() -> FastAPI:
    """
    Application factory.

    Creates and configures the FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Digital twin avatar service for virtual clothing try-on and fit prediction. "
            "Upload a photo to create a 3D body model with accurate measurements."
        ),
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(
        health.router,
        prefix=settings.api_prefix,
        tags=["Health"],
    )
    app.include_router(
        avatar.router,
        prefix=settings.api_prefix,
        tags=["Avatar"],
    )

    return app


# Create the application instance
app = create_app()
