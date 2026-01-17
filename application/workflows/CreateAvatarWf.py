from __future__ import annotations
from temporalio import workflow
from temporalio.common import RetryPolicy
from avatar_service.application.activities.image_processing_activity import save_user_image
from avatar_service.application.activities.econ_activity import run_econ_inference
from avatar_service.application.activities.texture_generation_activity import generate_realistic_textures
from avatar_service.application.activities.storage_activity import upload_avatar_to_s3
import os
from loguru import logger


@workflow.defn
class CreateAvatarWf:
    def __init__(self):
        pass

    """
    Workflow to create a full-body digital twin using ECON + optional texture generation.
    
    Expected payload schema:
    {
        "user_photo": <bytes or base64 encoded image>,
        "user_id": <string>,
        "measurements": {
            "height": <float>,
            "weight": <float>,
            "chest": <float>,
            "waist": <float>,
            "hips": <float>
        },
        "preferences": {
            "skin_tone": <string>,
            "hair_style": <string>,
            "pose": <string>,
            "texture_tool": <string>  # Optional: "pshuman" (default, photorealistic), "hunyuan3d", "texdreamer", "enhanced", "simple", "opentryon", "skip", "phorhum", or "morphx"
        }
    }
    """
    
    @workflow.run
    async def run(self, payload: dict) -> dict:
        user_id = payload.get("user_id")
        user_img_data = payload.get("user_photo")
        body_measurements = payload.get("measurements", {})
        optional_preferences = payload.get("preferences", {})

        if isinstance(user_img_data, list):
            user_img_data = bytes(user_img_data)
        elif not isinstance(user_img_data, bytes):
            raise ValueError("user_photo must be bytes or list of bytes")
        
        image_path = await workflow.execute_activity(
            save_user_image,
            args=[user_img_data, user_id],
            start_to_close_timeout=60,
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        
        # Step 2: Run ECON inference to generate mesh and SMPL-X parameters
        output_dir = os.path.join(
            os.getenv("AVATAR_OUTPUT_DIR", "/tmp/avatar_output"),
            user_id
        )
        
        econ_result = await workflow.execute_activity(
            run_econ_inference,
            args=[image_path, output_dir],
            start_to_close_timeout=600,  # ECON can take several minutes
            retry_policy=RetryPolicy(
                maximum_attempts=2,
                initial_interval=30,
            ),
        )
        
        # Step 3: Generate photorealistic textures (optional)
        # Default to "pshuman" for best photorealistic quality (falls back to enhanced if not available)
        texture_tool = optional_preferences.get("texture_tool", "pshuman")
        texture_result = {}
        
        if texture_tool != "skip":
            texture_result = await workflow.execute_activity(
                generate_realistic_textures,
                args=[
                    econ_result["mesh_path"],
                    image_path,
                    output_dir,
                ],
                kwargs={
                    "texture_tool": texture_tool,
                    "smplx_params": econ_result.get("smplx_params"),  # Pass SMPL-X params for better texture mapping
                },
                start_to_close_timeout=600,  # Texture generation can take several minutes
                retry_policy=RetryPolicy(
                    maximum_attempts=2,
                    initial_interval=30,
                ),
            )
        else:
            logger.info("Skipping texture generation step")
            texture_result = {
                "textured_mesh_path": None,
                "texture_map_path": None,
                "diffuse_map_path": None,
                "normal_map_path": None,
            }

        final_mesh_path = texture_result.get("textured_mesh_path") or econ_result["mesh_path"]
        
        s3_urls = await workflow.execute_activity(
            upload_avatar_to_s3,
            args=[
                final_mesh_path,
                econ_result["smplx_params"],
                user_id,
            ],
            kwargs={
                "normal_map_path": texture_result.get("normal_map_path") or econ_result.get("normal_map_path"),
                "texture_path": texture_result.get("texture_map_path") or econ_result.get("texture_path"),
                "diffuse_map_path": texture_result.get("diffuse_map_path"),
            },
            start_to_close_timeout=300,
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        return {
            "user_id": user_id,
            "status": "completed",
            "mesh_path": final_mesh_path,
            "original_mesh_path": econ_result["mesh_path"],
            "smplx_params": econ_result["smplx_params"],
            "texture_generation": {
                "tool": texture_tool,
                "textured_mesh_path": texture_result.get("textured_mesh_path"),
                "texture_map_path": texture_result.get("texture_map_path"),
                "diffuse_map_path": texture_result.get("diffuse_map_path"),
                "normal_map_path": texture_result.get("normal_map_path"),
            },
            "s3_urls": s3_urls,
            "measurements": body_measurements,
            "preferences": optional_preferences,
        }