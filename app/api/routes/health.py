"""
Health check endpoints.

Used by load balancers, orchestrators, and monitoring systems
to verify service availability.
"""

from datetime import datetime

import redis
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import get_db

router = APIRouter()


class HealthStatus(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    version: str
    checks: dict[str, str]


@router.get("/health", response_model=HealthStatus)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthStatus:
    """
    Comprehensive health check.

    Verifies connectivity to:
    - Database (PostgreSQL)
    - Cache/Queue (Redis)

    Returns 200 if all checks pass, includes status of each dependency.
    """
    settings = get_settings()
    checks = {}

    # Check database
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"

    # Check Redis
    try:
        r = redis.from_url(settings.redis.url)
        r.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)}"

    # Determine overall status
    all_healthy = all(v == "healthy" for v in checks.values())
    status = "healthy" if all_healthy else "degraded"

    return HealthStatus(
        status=status,
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        checks=checks,
    )


@router.get("/health/live")
async def liveness():
    """
    Kubernetes liveness probe.

    Simple check that the service is running.
    Does not check dependencies.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness(db: AsyncSession = Depends(get_db)):
    """
    Kubernetes readiness probe.

    Verifies the service is ready to accept traffic.
    Checks database connectivity.
    """
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        return {"status": "not ready"}, 503
