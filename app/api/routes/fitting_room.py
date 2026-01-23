"""
Virtual fitting room endpoints.

Handles on-demand garment fitting:
1. Try on garment(s) on avatar
2. Check fitting job status
3. Retrieve fitted avatar results
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from redis import Redis
from rq import Queue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import Avatar, FittedGarment, Garment, get_db
from app.models.schemas import (
    ErrorResponse,
    FittedGarmentResponse,
    JobResponse,
    JobStatus,
    JobStatusResponse,
    TryOnRequest,
)
from app.models.schemas import JobStatus as JobStatusEnum

router = APIRouter()


def get_redis() -> Redis:
    """Get Redis connection for job queue."""
    settings = get_settings()
    return Redis.from_url(settings.redis.url)


def get_queue(redis_conn: Redis = Depends(get_redis)) -> Queue:
    """Get RQ queue for job submission."""
    settings = get_settings()
    return Queue("garment_fitting", connection=redis_conn)


@router.post(
    "/fitting-room/try-on",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: {"model": ErrorResponse, "description": "Avatar or garment not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def try_on_garments(
    request: TryOnRequest,
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    """
    Try on one or more garments on an avatar.

    Creates a background job to:
    1. Load avatar and garment meshes
    2. Fit garments with physics simulation
    3. Generate animated GLB with clothes
    4. Store fitted result (if caching enabled)

    Returns a job_id for status polling.
    """
    # Verify avatar exists
    try:
        avatar_uuid = uuid.UUID(request.avatar_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_avatar_id", "message": "Invalid avatar ID format"},
        )

    result = await db.execute(select(Avatar).where(Avatar.id == avatar_uuid))
    avatar = result.scalar_one_or_none()

    if avatar is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "avatar_not_found", "message": f"Avatar {request.avatar_id} not found"},
        )

    # Verify all garments exist and are completed
    garment_uuids = []
    for garment_id in request.garment_ids:
        try:
            garment_uuid = uuid.UUID(garment_id)
            garment_uuids.append(garment_uuid)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "invalid_garment_id", "message": f"Invalid garment ID: {garment_id}"},
            )

    result = await db.execute(
        select(Garment).where(Garment.id.in_(garment_uuids))
    )
    garments = result.scalars().all()

    if len(garments) != len(request.garment_ids):
        found_ids = {str(g.id) for g in garments}
        missing_ids = set(request.garment_ids) - found_ids
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "garments_not_found",
                "message": f"Garments not found: {', '.join(missing_ids)}",
            },
        )

    # Check all garments are completed
    incomplete = [g for g in garments if g.status != "completed"]
    if incomplete:
        incomplete_ids = [str(g.id) for g in incomplete]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "garments_not_ready",
                "message": f"Garments not ready (still processing): {', '.join(incomplete_ids)}",
            },
        )

    # Create fitting job ID (will be used as fitted_id when complete)
    fitting_id = uuid.uuid4()
    logger.info(f"Creating fitting job {fitting_id} for avatar {request.avatar_id} with {len(request.garment_ids)} garments")

    # Queue the job for background processing
    settings = get_settings()
    try:
        redis_conn = Redis.from_url(settings.redis.url)
        queue = Queue("garment_fitting", connection=redis_conn)

        rq_job = queue.enqueue(
            "app.workers.fitting_worker.fit_garments_to_avatar",
            kwargs={
                "fitting_id": str(fitting_id),
                "avatar_id": request.avatar_id,
                "garment_ids": request.garment_ids,
                "cache_result": request.cache_result,
            },
            job_id=str(fitting_id),
            job_timeout=settings.job_timeout_seconds,
        )

        logger.info(f"Queued RQ job {rq_job.id} for fitting {fitting_id}")

    except Exception as e:
        logger.error(f"Failed to queue fitting job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "queue_error",
                "message": "Failed to queue fitting job",
            },
        )

    # Store avatar reference for later use in job status
    # Note: In a real implementation, you might want to create a FittingJob table
    # For now, we use the fitting_id and check RQ/FittedGarment status
    
    return JobResponse(
        job_id=str(fitting_id),
        status=JobStatus.PENDING,
        message=f"Fitting job queued. Use GET /fitting-room/job/{fitting_id} to check status.",
        created_at=avatar.created_at,
    )


@router.get(
    "/fitting-room/job/{job_id}",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_fitting_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> JobStatusResponse:
    """
    Get the status of a garment fitting job.

    Poll this endpoint to check if the fitting is complete.
    Once status is 'completed', use GET /fitting-room/fitted/{job_id} to retrieve the result.
    """
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_job_id", "message": "Invalid job ID format"},
        )

    # Check if FittedGarment exists (job completed)
    result = await db.execute(select(FittedGarment).where(FittedGarment.id == job_uuid))
    fitted = result.scalar_one_or_none()

    if fitted is not None:
        # Job completed
        status = JobStatus.COMPLETED
        message = "Garment fitting complete"
        progress = 100
        error = None
        completed_at = fitted.created_at
        started_at = None  # Could add started_at to FittedGarment if needed
    else:
        # Check RQ job status
        try:
            from rq import Queue
            from redis import Redis
            
            settings = get_settings()
            redis_conn = Redis.from_url(settings.redis.url)
            queue = Queue("garment_fitting", connection=redis_conn)
            rq_job = queue.fetch_job(job_id)
            
            if rq_job is None:
                # Job not found
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "job_not_found", "message": f"Fitting job {job_id} not found"},
                )
            
            if rq_job.is_finished:
                status = JobStatus.COMPLETED
                message = "Garment fitting complete"
                progress = 100
            elif rq_job.is_failed:
                status = JobStatus.FAILED
                message = "Garment fitting failed"
                progress = None
                error = str(rq_job.exc_info) if rq_job.exc_info else "Unknown error"
            elif rq_job.is_started:
                status = JobStatus.PROCESSING
                message = "Fitting garments to avatar..."
                progress = 50
            else:
                status = JobStatus.PENDING
                message = "Fitting job queued"
                progress = 0
            
            error = error if rq_job.is_failed else None
            completed_at = rq_job.ended_at if rq_job.is_finished else None
            started_at = rq_job.started_at if rq_job.is_started else None
            created_at = rq_job.created_at if hasattr(rq_job, 'created_at') else None
            
        except Exception as e:
            logger.warning(f"Failed to check RQ job status: {e}")
            # Assume pending if we can't check
            status = JobStatus.PENDING
            message = "Fitting job status unknown"
            progress = 0
            error = None
            completed_at = None
            started_at = None
            created_at = None

    return JobStatusResponse(
        job_id=job_id,
        status=status,
        message=message,
        created_at=fitted.created_at if fitted else created_at,
        started_at=started_at,
        completed_at=completed_at,
        error=error,
        progress_percent=progress,
    )


@router.get(
    "/fitting-room/fitted/{fitted_id}",
    response_model=FittedGarmentResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_fitted_garment(
    fitted_id: str,
    db: AsyncSession = Depends(get_db),
) -> FittedGarmentResponse:
    """Get fitted garment/outfit result."""
    try:
        fitted_uuid = uuid.UUID(fitted_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_fitted_id", "message": "Invalid fitted ID format"},
        )

    result = await db.execute(select(FittedGarment).where(FittedGarment.id == fitted_uuid))
    fitted = result.scalar_one_or_none()

    if fitted is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "fitted_not_found", "message": f"Fitted garment {fitted_id} not found"},
        )

    # Extract layering order from fitting parameters
    layering_order = []
    if fitted.fitting_parameters:
        layering_order = fitted.fitting_parameters.get("layering_order", [])

    return FittedGarmentResponse(
        fitted_id=str(fitted.id),
        avatar_id=str(fitted.avatar_id),
        garment_ids=fitted.garment_ids,
        fitted_mesh_url=fitted.fitted_mesh_url,
        animated_glb_url=fitted.animated_glb_url,
        layering_order=layering_order,
        created_at=fitted.created_at,
    )
