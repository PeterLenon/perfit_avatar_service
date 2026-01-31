"""
Body extraction worker.

This module contains the RQ job function that:
1. Loads the user's image
2. Runs HMR2 inference to get SMPL parameters
3. Extracts body measurements from SMPL mesh
4. Stores the results in the database and S3

This runs in a separate worker process from the API server.
"""

import base64
import io
import uuid
from datetime import datetime
import numpy as np
import json
from loguru import logger
from PIL import Image
from sqlalchemy import select


def extract_body_shape(
    job_id: str,
    user_id: str,
    image_base64: str,
    gender: str,
) -> dict:
    """
    Extract body shape from an image.

    This is the main RQ job function that processes avatar creation requests.

    Args:
        job_id: UUID of the ExtractionJob in the database
        user_id: User identifier
        image_base64: Base64-encoded image data
        gender: Gender hint ('male', 'female', 'neutral')

    Returns:
        dict with status and avatar_id if successful

    Raises:
        Exception: If extraction fails (job will be marked as failed)
    """
    logger.info(f"Starting body extraction for job {job_id}, user {user_id}")

    # Import here to avoid circular imports and allow lazy loading
    from app.config import get_settings
    from app.models.database import Avatar, ExtractionJob
    from app.models.schemas import JobStatus
    from app.services.storage import StorageService

    settings = get_settings()

    # Create sync session for worker (RQ doesn't support async)
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    sync_engine = create_engine(settings.database.sync_url)
    SessionLocal = sessionmaker(bind=sync_engine)

    with SessionLocal() as db:
        try:
            # Update job status to processing
            job = db.execute(
                select(ExtractionJob).where(ExtractionJob.id == uuid.UUID(job_id))
            ).scalar_one()

            job.status = JobStatus.PROCESSING
            job.started_at = datetime.utcnow()
            db.commit()

            logger.info(f"Job {job_id} status updated to PROCESSING")

            # Decode image
            image_data = _decode_base64(image_base64)
            image = Image.open(io.BytesIO(image_data))
            # Convert to RGB if necessary (e.g., RGBA or grayscale)
            if image.mode != "RGB":
                image = image.convert("RGB")
            logger.info(f"Image loaded: {image.size}")

            # =================================================================
            # HMR2 INFERENCE
            # =================================================================
            logger.info("Running HMR2 body reconstruction...")

            from ml.hmr2 import HMR2Inference

            hmr2 = HMR2Inference(
                device=settings.ml.hmr2_device,
                detection_threshold=settings.ml.detection_threshold,
            )
            smpl_output = hmr2.predict(image, gender=gender)

            logger.info("HMR2 inference complete")

            # Extract SMPL parameters
            smpl_betas = smpl_output["betas"].tolist()
            smpl_body_pose = smpl_output["body_pose"].tolist()
            smpl_global_orient = smpl_output["global_orient"].tolist()
            vertices = smpl_output["vertices"]

            # =================================================================
            # MEASUREMENT EXTRACTION
            # =================================================================
            logger.info("Extracting body measurements...")

            from ml.measurements import MeasurementExtractor

            extractor = MeasurementExtractor(
                smpl_model_path=settings.ml.smpl_model_path,
                gender=gender,
            )
            measurements = extractor.extract(
                betas=smpl_output["betas"],
                body_pose=smpl_output["body_pose"],
                global_orient=smpl_output["global_orient"],
            )

            logger.info(f"Measurements extracted: height={measurements['height_cm']:.1f}cm")

            # =================================================================
            # STORAGE UPLOAD
            # =================================================================
            logger.info("Uploading to storage...")

            # Create avatar ID first so we can use it for storage paths
            avatar_id = uuid.uuid4()

            storage = StorageService(settings.minio)
            storage.ensure_bucket_exists()

            # Upload SMPL parameters
            smplx_params_url = storage.upload_smplx_params(
                user_id=user_id,
                avatar_id=str(avatar_id),
                params={
                    "betas": smpl_betas,
                    "body_pose": smpl_body_pose,
                    "global_orient": smpl_global_orient,
                },
            )

            # Upload mesh
            mesh_url = storage.upload_mesh(
                user_id=user_id,
                avatar_id=str(avatar_id),
                vertices=vertices,
                faces=hmr2.smpl_faces,
            )

            # Generate and upload animated GLB
            logger.info("Generating animated GLB with multiple poses...")
            from app.services.animation import AnimationService

            animation_service = AnimationService(
                smpl_model_path=settings.ml.smpl_model_path,
                gender=gender,
            )
            poses = animation_service.generate_poses(
                betas=smpl_output["betas"],
                base_body_pose=smpl_output["body_pose"],
                base_global_orient=smpl_output["global_orient"],
            )
            glb_data = animation_service.create_animated_glb(
                poses, output_path=None, body_color="medium"
            )
            animated_glb_url = storage.upload_animated_glb(
                user_id=user_id,
                avatar_id=str(avatar_id),
                glb_data=glb_data,
            )
            logger.info(f"Animated GLB uploaded")

            # Upload source image
            # Determine content type from image format
            content_type = f"image/{image.format.lower()}" if image.format else "image/jpeg"
            source_image_url = storage.upload_source_image(
                user_id=user_id,
                avatar_id=str(avatar_id),
                image_data=image_data,
                content_type=content_type,
            )

            logger.info("Storage upload complete")

            # =================================================================
            # CREATE AVATAR RECORD
            # =================================================================
            avatar = Avatar(
                id=avatar_id,
                user_id=user_id,
                gender=gender,
                smplx_betas={"values": smpl_betas},
                smplx_body_pose={"values": smpl_body_pose},
                smplx_global_orient={"values": smpl_global_orient},
                height_cm=measurements["height_cm"],
                chest_circumference_cm=measurements["chest_circumference_cm"],
                waist_circumference_cm=measurements["waist_circumference_cm"],
                hip_circumference_cm=measurements["hip_circumference_cm"],
                inseam_cm=measurements["inseam_cm"],
                shoulder_width_cm=measurements["shoulder_width_cm"],
                arm_length_cm=measurements["arm_length_cm"],
                thigh_circumference_cm=measurements["thigh_circumference_cm"],
                neck_circumference_cm=measurements["neck_circumference_cm"],
                smplx_params_url=smplx_params_url,
                mesh_url=mesh_url,
                animated_glb_url=animated_glb_url,
                source_image_url=source_image_url,
            )
            db.add(avatar)
            db.flush()

            # Update job as completed
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.avatar_id = avatar.id
            db.commit()

            logger.info(f"Job {job_id} completed successfully. Avatar ID: {avatar.id}")

            return {
                "status": "success",
                "avatar_id": str(avatar.id),
                "measurements": measurements,
            }

        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")

            # Update job as failed
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()

            raise


def _decode_base64(image_base64: str) -> bytes:
    """Decode base64 image data, handling data URL prefix."""
    if "," in image_base64:
        _, image_base64 = image_base64.split(",", 1)
    return base64.b64decode(image_base64.strip())
