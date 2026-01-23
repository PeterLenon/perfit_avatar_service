"""
Avatar creation and retrieval endpoints.

Handles the core workflow:
1. User submits photo → validate → queue extraction job
2. User polls for job status
3. User retrieves completed avatar with measurements
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from redis import Redis
from rq import Queue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import Avatar, ExtractionJob, get_db
from app.models.schemas import (
    AvatarCreateRequest,
    AvatarResponse,
    BodyMeasurements,
    ErrorResponse,
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
    return Queue(settings.rq_queue_name, connection=redis_conn)


# =============================================================================
# Avatar Creation
# =============================================================================


@router.post(
    "/avatar",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def create_avatar(
    request: AvatarCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    """
    Create a new avatar from a user photo.

    This endpoint:
    1. Validates the uploaded image (format, size, face detection)
    2. Creates an extraction job
    3. Queues the job for background processing
    4. Returns a job ID for status polling

    The actual body extraction happens asynchronously in a worker process.
    Use GET /avatar/job/{job_id} to check status.
    """
    settings = get_settings()
    validator = ImageValidator(settings.image_validation)

    # Validate the image
    logger.info(f"Validating image for user {request.user_id}")
    validation_result = validator.validate_base64(request.image_base64)

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

    # Log warnings but don't fail
    if validation_result.warnings:
        logger.warning(f"Image validation warnings: {validation_result.warnings}")

    # Create job record in database
    job = ExtractionJob(
        user_id=request.user_id,
        gender=request.gender,
        status=JobStatus.PENDING,
    )
    db.add(job)
    await db.flush()  # Get the job ID

    job_id = str(job.id)
    logger.info(f"Created extraction job {job_id} for user {request.user_id}")

    # Queue the job for background processing
    # Note: The actual worker function will be implemented in app/workers/extraction_worker.py
    try:
        redis_conn = Redis.from_url(settings.redis.url)
        queue = Queue(settings.rq_queue_name, connection=redis_conn)

        rq_job = queue.enqueue(
            "app.workers.extraction_worker.extract_body_shape",
            kwargs={
                "job_id":job_id,
                "user_id":request.user_id,
                "image_base64":request.image_base64,
                "gender":request.gender.value
            },
            job_id=job_id,
            job_timeout=settings.job_timeout_seconds,
        )

        # Update job with RQ job ID
        job.rq_job_id = rq_job.id
        logger.info(f"Queued RQ job {rq_job.id} for extraction job {job_id}")

    except Exception as e:
        logger.error(f"Failed to queue job: {e}")
        job.status = JobStatus.FAILED
        job.error_message = f"Failed to queue job: {str(e)}"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "queue_error",
                "message": "Failed to queue extraction job",
            },
        )

    await db.commit()

    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Job queued for processing. Use GET /avatar/job/{job_id} to check status.",
        created_at=job.created_at,
    )



@router.get(
    "/avatar/job/{job_id}",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> JobStatusResponse:
    """
    Get the status of an avatar extraction job.

    Poll this endpoint to check if the extraction is complete.
    Once status is 'completed', use GET /avatar/{user_id} to retrieve the avatar.
    """
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_job_id", "message": "Invalid job ID format"},
        )

    result = await db.execute(select(ExtractionJob).where(ExtractionJob.id == job_uuid))
    job = result.scalar_one_or_none()

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "job_not_found", "message": f"Job {job_id} not found"},
        )

    # Calculate progress percentage based on status
    progress_map = {
        JobStatus.PENDING: 0,
        JobStatus.PROCESSING: 50,
        JobStatus.COMPLETED: 100,
        JobStatus.FAILED: None,
    }

    return JobStatusResponse(
        job_id=job_id,
        status=job.status,
        message=_get_status_message(job.status),
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error_message,
        progress_percent=progress_map.get(job.status),
    )


def _get_status_message(status: JobStatus) -> str:
    """Get human-readable message for job status."""
    messages = {
        JobStatus.PENDING: "Job is queued and waiting to be processed",
        JobStatus.PROCESSING: "Extracting body shape from image...",
        JobStatus.COMPLETED: "Avatar creation complete",
        JobStatus.FAILED: "Job failed - see error for details",
    }
    return messages.get(status, "Unknown status")


# =============================================================================
# Avatar Retrieval
# =============================================================================


@router.get(
    "/avatar/{user_id}",
    response_model=AvatarResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_avatar(
    user_id: str,
    db: AsyncSession = Depends(get_db),
) -> AvatarResponse:
    """
    Get the latest avatar for a user.

    Returns the most recently created avatar with body measurements
    and URLs to download the SMPL-X parameters and mesh.
    """
    result = await db.execute(
        select(Avatar)
        .where(Avatar.user_id == user_id)
        .order_by(Avatar.created_at.desc())
        .limit(1)
    )
    avatar = result.scalar_one_or_none()

    if avatar is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "avatar_not_found",
                "message": f"No avatar found for user {user_id}",
            },
        )

    return AvatarResponse(
        user_id=avatar.user_id,
        avatar_id=str(avatar.id),
        measurements=BodyMeasurements(
            height_cm=avatar.height_cm or 0,
            chest_circumference_cm=avatar.chest_circumference_cm or 0,
            waist_circumference_cm=avatar.waist_circumference_cm or 0,
            hip_circumference_cm=avatar.hip_circumference_cm or 0,
            inseam_cm=avatar.inseam_cm or 0,
            shoulder_width_cm=avatar.shoulder_width_cm or 0,
            arm_length_cm=avatar.arm_length_cm or 0,
            thigh_circumference_cm=avatar.thigh_circumference_cm or 0,
            neck_circumference_cm=avatar.neck_circumference_cm or 0,
        ),
        smplx_params_url=avatar.smplx_params_url,
        mesh_url=avatar.mesh_url,
        animated_glb_url=avatar.animated_glb_url,
        created_at=avatar.created_at,
        gender=avatar.gender,
    )


@router.get(
    "/avatar/{user_id}/history",
    response_model=list[AvatarResponse],
)
async def get_avatar_history(
    user_id: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
) -> list[AvatarResponse]:
    """
    Get avatar history for a user.

    Returns all avatars for the user, ordered by creation date (newest first).
    Useful if the user wants to see or revert to previous body measurements.
    """
    result = await db.execute(
        select(Avatar)
        .where(Avatar.user_id == user_id)
        .order_by(Avatar.created_at.desc())
        .limit(limit)
    )
    avatars = result.scalars().all()

    return [
        AvatarResponse(
            user_id=avatar.user_id,
            avatar_id=str(avatar.id),
            measurements=BodyMeasurements(
                height_cm=avatar.height_cm or 0,
                chest_circumference_cm=avatar.chest_circumference_cm or 0,
                waist_circumference_cm=avatar.waist_circumference_cm or 0,
                hip_circumference_cm=avatar.hip_circumference_cm or 0,
                inseam_cm=avatar.inseam_cm or 0,
                shoulder_width_cm=avatar.shoulder_width_cm or 0,
                arm_length_cm=avatar.arm_length_cm or 0,
                thigh_circumference_cm=avatar.thigh_circumference_cm or 0,
                neck_circumference_cm=avatar.neck_circumference_cm or 0,
            ),
            smplx_params_url=avatar.smplx_params_url,
            mesh_url=avatar.mesh_url,
            animated_glb_url=avatar.animated_glb_url,
            created_at=avatar.created_at,
            gender=avatar.gender,
        )
        for avatar in avatars
    ]
