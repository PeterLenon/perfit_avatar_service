"""
Garment fitting worker.

Background job that fits garments onto avatars:
1. Loads avatar and garment meshes
2. Fits garments with physics simulation
3. Generates animated GLB with clothes
4. Stores fitted result (optional caching)
"""

import json
import uuid
from datetime import datetime

import numpy as np
from loguru import logger
from sqlalchemy import String, cast, create_engine, select
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.models.database import Avatar, FittedGarment, Garment
from app.services.animation import AnimationService
from app.services.garment_fitting import GarmentFittingService
from app.services.mesh_loader import MeshLoaderService
from app.services.storage import StorageService


def fit_garments_to_avatar(
    fitting_id: str,
    avatar_id: str,
    garment_ids: list[str],
    cache_result: bool | None = None,  # None = use config default
) -> dict:
    """
    Fit garments onto an avatar.

    Args:
        fitting_id: UUID for the fitting job/result
        avatar_id: Avatar UUID
        garment_ids: List of garment UUIDs
        cache_result: Whether to cache the result

    Returns:
        dict with status and fitted_id if successful
    """
    logger.info(f"Starting garment fitting {fitting_id} for avatar {avatar_id} with {len(garment_ids)} garments")

    settings = get_settings()
    
    # Use config default if cache_result not specified
    if cache_result is None:
        cache_result = settings.fitting_room.cache_enabled
        logger.info(f"Using default cache setting: {cache_result}")

    # Create sync session for worker
    sync_engine = create_engine(settings.database.sync_url)
    SessionLocal = sessionmaker(bind=sync_engine)

    with SessionLocal() as db:
        try:
            # Load avatar
            avatar = db.execute(
                select(Avatar).where(Avatar.id == uuid.UUID(avatar_id))
            ).scalar_one()

            # Load garments
            garment_uuids = [uuid.UUID(gid) for gid in garment_ids]
            garments = db.execute(
                select(Garment).where(Garment.id.in_(garment_uuids))
            ).scalars().all()

            if len(garments) != len(garment_ids):
                raise ValueError(f"Expected {len(garment_ids)} garments, found {len(garments)}")

            # =================================================================
            # LOAD MESHES
            # =================================================================
            logger.info("Loading avatar and garment meshes...")

            storage = StorageService(settings.minio)
            storage.ensure_bucket_exists()
            mesh_loader = MeshLoaderService(storage)

            # Load avatar mesh (from OBJ or reconstruct from SMPL params)
            if avatar.mesh_url:
                logger.info(f"Loading avatar mesh from: {avatar.mesh_url}")
                avatar_vertices, avatar_faces = mesh_loader.load_mesh_from_url(avatar.mesh_url)
            elif avatar.smplx_params_url:
                logger.info(f"Reconstructing avatar mesh from SMPL params: {avatar.smplx_params_url}")
                # Reconstruct from SMPL parameters
                avatar_vertices, avatar_faces = mesh_loader.reconstruct_mesh_from_smpl(
                    smpl_params_url=avatar.smplx_params_url,
                    gender=avatar.gender,
                )
            else:
                raise ValueError(f"Avatar {avatar_id} has neither mesh_url nor smplx_params_url")

            # Load garment meshes
            garment_data = []
            for garment in garments:
                if not garment.garment_mesh_url:
                    raise ValueError(f"Garment {garment.id} does not have a mesh_url (status: {garment.status})")
                
                logger.info(f"Loading garment mesh from: {garment.garment_mesh_url}")
                garment_vertices, garment_faces = mesh_loader.load_mesh_from_url(garment.garment_mesh_url)

                garment_data.append({
                    "vertices": garment_vertices,
                    "faces": garment_faces,
                    "garment_type": garment.garment_type,
                    "material_properties": garment.material_properties or {},
                    "key_points": garment.key_points or {},
                })

            # =================================================================
            # FIT GARMENTS
            # =================================================================
            logger.info("Fitting garments to avatar...")

            fitting_service = GarmentFittingService()
            fitting_result = fitting_service.fit_garments_to_avatar(
                avatar_vertices=avatar_vertices,
                avatar_faces=avatar_faces,
                garments=garment_data,
            )

            # =================================================================
            # GENERATE ANIMATED GLB
            # =================================================================
            logger.info("Generating animated GLB with clothes...")

            # Upload fitted mesh first
            fitted_mesh_url = storage.upload_mesh(
                user_id=avatar.user_id,
                avatar_id=fitting_id,
                vertices=fitting_result["combined_vertices"],
                faces=fitting_result["combined_faces"],
                storage_type="fittings",
            )

            # Generate animated GLB with fitted mesh
            # Create multiple views/poses of the fitted avatar with clothes
            animated_glb_url = None
            try:
                # Create poses by rotating the fitted mesh (simpler than re-fitting for each pose)
                # This shows the avatar with clothes from different angles
                combined_vertices = fitting_result["combined_vertices"]
                combined_faces = fitting_result["combined_faces"]
                
                # Generate poses with rotations (showing fitted avatar from different angles)
                poses = []
                
                # Center the mesh for rotation
                center = combined_vertices.mean(axis=0)
                centered_vertices = combined_vertices - center
                
                # Create poses with different rotations
                rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 views around Y-axis
                
                for angle in rotation_angles:
                    # Rotate around Y-axis
                    angle_rad = np.radians(angle)
                    rotation_matrix = np.array([
                        [np.cos(angle_rad), 0, np.sin(angle_rad)],
                        [0, 1, 0],
                        [-np.sin(angle_rad), 0, np.cos(angle_rad)],
                    ])
                    
                    rotated_vertices = (centered_vertices @ rotation_matrix.T) + center
                    
                    poses.append({
                        "name": f"View {angle}°",
                        "vertices": rotated_vertices,
                        "faces": combined_faces,
                    })
                
                # Use animation service to create GLB
                animation_service = AnimationService(
                    smpl_model_path=settings.ml.smpl_model_path,
                    gender=avatar.gender,
                )
                
                glb_data = animation_service.create_animated_glb(poses, output_path=None)
                
                animated_glb_url = storage.upload_animated_glb(
                    user_id=avatar.user_id,
                    avatar_id=fitting_id,
                    glb_data=glb_data,
                    storage_type="fittings",
                    filename="fitted.glb",
                )
                logger.info(f"Animated GLB generated and uploaded: {animated_glb_url}")
                
            except Exception as e:
                logger.warning(f"Failed to generate animated GLB: {e}. Continuing without animation.")
                # Continue without animated GLB - fitted mesh is still available

            # =================================================================
            # CREATE FITTED GARMENT RECORD (if caching enabled)
            # =================================================================
            # Check both per-request setting and global config
            should_cache = cache_result and settings.fitting_room.cache_enabled

            if should_cache:
                # Check if already exists (same avatar + garments)
                # Cast JSON to text for comparison since PostgreSQL JSON doesn't support =
                existing = db.execute(
                    select(FittedGarment)
                    .where(FittedGarment.avatar_id == avatar.id)
                    .where(cast(FittedGarment.garment_ids, String) == json.dumps(garment_ids))
                ).scalar_one_or_none()

                if existing:
                    # Update existing record
                    existing.fitted_mesh_url = fitted_mesh_url
                    existing.animated_glb_url = animated_glb_url
                    existing.fitting_parameters = {
                        "layering_order": fitting_result["layering_order"],
                        **fitting_result["fitting_parameters"],
                    }
                    logger.info(f"Updated existing fitted garment {existing.id}")
                else:
                    # Create new record
                    fitted_garment = FittedGarment(
                        id=uuid.UUID(fitting_id),
                        avatar_id=avatar.id,
                        garment_ids=garment_ids,
                        fitted_mesh_url=fitted_mesh_url,
                        animated_glb_url=animated_glb_url,
                        fitting_parameters={
                            "layering_order": fitting_result["layering_order"],
                            **fitting_result["fitting_parameters"],
                        },
                    )
                    db.add(fitted_garment)
                    logger.info(f"Created new fitted garment {fitting_id}")

                db.commit()
                logger.info(f"Fitted garment {fitting_id} cached")
            else:
                logger.info(f"Caching disabled - not storing fitted result")

            logger.info(f"Garment fitting {fitting_id} completed successfully")

            return {
                "status": "success",
                "fitted_id": fitting_id,
                "fitted_mesh_url": fitted_mesh_url,
                "animated_glb_url": animated_glb_url,
            }

        except Exception as e:
            logger.error(f"Garment fitting {fitting_id} failed: {str(e)}")
            raise
