"""
Garment processing worker.

Background job that processes clothing photos:
1. Extracts garment from photo (segmentation)
2. Reconstructs 3D mesh
3. Estimates material properties
4. Stores results
"""

import base64
import io
import uuid
from datetime import datetime

import numpy as np
from loguru import logger
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.models.database import Garment
from app.services.cloth_physics import ClothPhysicsService
from app.services.garment_extraction import GarmentExtractionService
from app.services.garment_reconstruction import GarmentReconstructionService
from app.services.storage import StorageService


def process_garment(
    garment_id: str,
    user_id: str,
    image_base64: str,
    garment_type: str | None = None,
) -> dict:
    """
    Process a garment from a photo.

    Args:
        garment_id: UUID of the Garment in the database
        user_id: User identifier
        image_base64: Base64-encoded image data
        garment_type: Optional garment type hint

    Returns:
        dict with status and garment_id if successful
    """
    logger.info(f"Starting garment processing for {garment_id}, user {user_id}")

    settings = get_settings()

    # Create sync session for worker
    sync_engine = create_engine(settings.database.sync_url)
    SessionLocal = sessionmaker(bind=sync_engine)

    with SessionLocal() as db:
        try:
            # Update garment status to processing
            garment = db.execute(
                select(Garment).where(Garment.id == uuid.UUID(garment_id))
            ).scalar_one()

            garment.status = "processing"
            garment.started_at = datetime.utcnow()
            db.commit()

            logger.info(f"Garment {garment_id} status updated to processing")

            # Decode image
            image_data = _decode_base64(image_base64)
            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")
            logger.info(f"Image loaded: {image.size}")

            # =================================================================
            # GARMENT EXTRACTION
            # =================================================================
            logger.info("Extracting garment from image...")

            extraction_service = GarmentExtractionService()
            extraction_result = extraction_service.extract_garment(
                image, garment_type_hint=garment_type
            )

            # =================================================================
            # 3D RECONSTRUCTION
            # =================================================================
            logger.info("Reconstructing 3D garment mesh...")

            reconstruction_service = GarmentReconstructionService()
            reconstruction_result = reconstruction_service.reconstruct_3d(
                image=image,
                segmented_mask=extraction_result["segmented_mask"],
                key_points=extraction_result["key_points"],
                garment_type=extraction_result["garment_type"],
            )

            # =================================================================
            # MATERIAL PROPERTIES & COLOR
            # =================================================================
            logger.info("Estimating material properties...")

            physics_service = ClothPhysicsService()
            material_properties = physics_service.estimate_material_properties(
                extraction_result["garment_type"], image_data
            )

            # Extract dominant color from garment texture
            logger.info("Extracting dominant garment color...")
            from app.services.skin_tone import extract_dominant_color
            dominant_color = extract_dominant_color(extraction_result["texture"])
            logger.info(f"Dominant color: RGB({dominant_color[0]:.2f}, {dominant_color[1]:.2f}, {dominant_color[2]:.2f})")

            # =================================================================
            # STORAGE UPLOAD
            # =================================================================
            logger.info("Uploading to storage...")

            storage = StorageService(settings.minio)
            storage.ensure_bucket_exists()

            # Upload source image
            content_type = f"image/{image.format.lower()}" if image.format else "image/jpeg"
            source_image_url = storage.upload_image(
                user_id=user_id,
                item_id=garment_id,
                image_data=image_data,
                content_type=content_type,
                path_suffix="source",
                storage_type="garments",
            )

            # Upload segmented image
            segmented_image = Image.fromarray(extraction_result["segmented_mask"])
            segmented_buffer = io.BytesIO()
            segmented_image.save(segmented_buffer, format="PNG")
            segmented_image_url = storage.upload_image(
                user_id=user_id,
                item_id=garment_id,
                image_data=segmented_buffer.getvalue(),
                content_type="image/png",
                path_suffix="segmented",
                storage_type="garments",
            )

            # Upload garment mesh
            garment_mesh_url = storage.upload_mesh(
                user_id=user_id,
                avatar_id=garment_id,
                vertices=reconstruction_result["vertices"],
                faces=reconstruction_result["faces"],
                storage_type="garments",
            )

            # Upload texture
            texture_buffer = io.BytesIO()
            extraction_result["texture"].save(texture_buffer, format="PNG")
            texture_url = storage.upload_image(
                user_id=user_id,
                item_id=garment_id,
                image_data=texture_buffer.getvalue(),
                content_type="image/png",
                path_suffix="texture",
                storage_type="garments",
            )

            logger.info("Storage upload complete")

            # =================================================================
            # UPDATE GARMENT RECORD
            # =================================================================
            garment.status = "completed"
            garment.completed_at = datetime.utcnow()
            garment.garment_type = extraction_result["garment_type"]
            garment.source_image_url = source_image_url
            garment.segmented_image_url = segmented_image_url
            garment.garment_mesh_url = garment_mesh_url
            garment.texture_url = texture_url
            garment.material_properties = material_properties
            garment.key_points = extraction_result["key_points"]
            garment.dominant_color = {"rgb": dominant_color}
            db.commit()

            logger.info(f"Garment {garment_id} processing completed successfully")

            return {
                "status": "success",
                "garment_id": garment_id,
                "garment_type": extraction_result["garment_type"],
            }

        except Exception as e:
            logger.error(f"Garment {garment_id} processing failed: {str(e)}")

            # Update garment as failed
            garment.status = "failed"
            garment.error_message = str(e)
            db.commit()

            raise


def _decode_base64(image_base64: str) -> bytes:
    """Decode base64 image data, handling data URL prefix."""
    if "," in image_base64:
        _, image_base64 = image_base64.split(",", 1)
    return base64.b64decode(image_base64.strip())
