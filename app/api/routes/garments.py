"""
Garment management endpoints.

Handles:
1. Upload clothing photos → queue processing job
2. Check garment processing status
3. Retrieve processed garments
4. List user's garments
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from redis import Redis
from rq import Queue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import Garment, get_db
from app.models.schemas import (
    ErrorResponse,
    GarmentCreateRequest,
    GarmentResponse,
    JobResponse,
    JobStatus,
    JobStatusResponse,
)
from app.services.image_validator import ImageValidator

router = APIRouter()


def get_redis() -> Redis:
    """Get Redis connection for job queue."""
    settings = get_settings()
    return Redis.from_url(settings.redis.url)


def get_queue(redis_conn: Redis = Depends(get_redis)) -> Queue:
    """Get RQ queue for job submission."""
    settings = get_settings()
    return Queue("garment_processing", connection=redis_conn)


@router.post(
    "/garments",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def create_garment(
    request: GarmentCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    """
    Upload a clothing photo for processing.

    Creates a background job to:
    1. Extract garment from photo (segmentation)
    2. Reconstruct 3D mesh
    3. Extract material properties
    4. Store processed garment

    Returns a job_id for status polling.
    """
    settings = get_settings()
    validator = ImageValidator(settings.image_validation)

    # Validate the image (no face detection required for garments)
    logger.info(f"Validating garment image for user {request.user_id}")
    validation_result = validator.validate_base64(request.image_base64, require_face=False)

    if not validation_result.is_valid:
        logger.warning(f"Image validation failed: {validation_result.errors}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "image_validation_failed",
                "message": "Image validation failed",
                "details": {"errors": validation_result.errors},
            },
        )

    # Create garment record
    garment = Garment(
        user_id=request.user_id,
        name=request.name,
        garment_type=request.garment_type.value if request.garment_type else "unknown",
        status="pending",
    )
    db.add(garment)
    await db.flush()

    garment_id = str(garment.id)
    logger.info(f"Created garment {garment_id} for user {request.user_id}")

    # Queue the job for background processing
    try:
        redis_conn = Redis.from_url(settings.redis.url)
        queue = Queue("garment_processing", connection=redis_conn)

        rq_job = queue.enqueue(
            "app.workers.garment_worker.process_garment",
            kwargs={
                "garment_id": garment_id,
                "user_id": request.user_id,
                "image_base64": request.image_base64,
                "garment_type": request.garment_type.value if request.garment_type else None,
            },
            job_id=garment_id,
            job_timeout=settings.job_timeout_seconds,
        )

        logger.info(f"Queued RQ job {rq_job.id} for garment {garment_id}")

    except Exception as e:
        logger.error(f"Failed to queue job: {e}")
        garment.status = "failed"
        garment.error_message = f"Failed to queue job: {str(e)}"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "queue_error",
                "message": "Failed to queue garment processing job",
            },
        )

    await db.commit()

    return JobResponse(
        job_id=garment_id,
        status=JobStatus.PENDING,
        message="Garment processing queued. Use GET /garments/job/{job_id} to check status.",
        created_at=garment.created_at,
    )


@router.get(
    "/garments/job/{job_id}",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_garment_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> JobStatusResponse:
    """Get the status of a garment processing job."""
    try:
        garment_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_job_id", "message": "Invalid job ID format"},
        )

    result = await db.execute(select(Garment).where(Garment.id == garment_uuid))
    garment = result.scalar_one_or_none()

    if garment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "garment_not_found", "message": f"Garment {job_id} not found"},
        )

    progress_map = {
        "pending": 0,
        "processing": 50,
        "completed": 100,
        "failed": None,
    }

    return JobStatusResponse(
        job_id=job_id,
        status=JobStatus(garment.status),
        message=_get_garment_status_message(garment.status),
        created_at=garment.created_at,
        started_at=garment.started_at,
        completed_at=garment.completed_at,
        error=garment.error_message,
        progress_percent=progress_map.get(garment.status),
    )


@router.get(
    "/garments/{garment_id}",
    response_model=GarmentResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_garment(
    garment_id: str,
    db: AsyncSession = Depends(get_db),
) -> GarmentResponse:
    """Get processed garment details."""
    try:
        garment_uuid = uuid.UUID(garment_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_garment_id", "message": "Invalid garment ID format"},
        )

    result = await db.execute(select(Garment).where(Garment.id == garment_uuid))
    garment = result.scalar_one_or_none()

    if garment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "garment_not_found", "message": f"Garment {garment_id} not found"},
        )

    return GarmentResponse(
        garment_id=str(garment.id),
        user_id=garment.user_id,
        name=garment.name,
        garment_type=garment.garment_type,
        status=garment.status,
        source_image_url=garment.source_image_url,
        segmented_image_url=garment.segmented_image_url,
        garment_mesh_url=garment.garment_mesh_url,
        texture_url=garment.texture_url,
        material_properties=garment.material_properties,
        key_points=garment.key_points,
        created_at=garment.created_at,
    )


@router.get(
    "/garments",
    response_model=list[GarmentResponse],
)
async def list_garments(
    user_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[GarmentResponse]:
    """List all garments for a user."""
    result = await db.execute(
        select(Garment)
        .where(Garment.user_id == user_id)
        .order_by(Garment.created_at.desc())
    )
    garments = result.scalars().all()

    return [
        GarmentResponse(
            garment_id=str(garment.id),
            user_id=garment.user_id,
            name=garment.name,
            garment_type=garment.garment_type,
            status=garment.status,
            source_image_url=garment.source_image_url,
            segmented_image_url=garment.segmented_image_url,
            garment_mesh_url=garment.garment_mesh_url,
            texture_url=garment.texture_url,
            material_properties=garment.material_properties,
            key_points=garment.key_points,
            created_at=garment.created_at,
        )
        for garment in garments
    ]


def _get_garment_status_message(status: str) -> str:
    """Get human-readable message for garment status."""
    messages = {
        "pending": "Garment processing is queued",
        "processing": "Extracting and reconstructing garment...",
        "completed": "Garment processing complete",
        "failed": "Garment processing failed - see error for details",
    }
    return messages.get(status, "Unknown status")
