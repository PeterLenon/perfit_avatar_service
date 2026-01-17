import os
import json
import tempfile
from pathlib import Path
from temporalio import activity
from loguru import logger
from typing import Dict, Any, Optional


@activity.defn
async def upload_avatar_to_s3(
    mesh_path: str,
    smplx_params: Dict[str, Any],
    user_id: str,
    bucket_name: Optional[str] = None,
    normal_map_path: Optional[str] = None,
    texture_path: Optional[str] = None,
    diffuse_map_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Upload avatar assets (mesh, SMPL-X params, textures) to S3.
    
    Args:
        mesh_path: Path to mesh file
        smplx_params: SMPL-X parameters dictionary
        user_id: User identifier
        bucket_name: S3 bucket name (defaults to env var or config)
        normal_map_path: Optional path to normal map
        texture_path: Optional path to texture map
        diffuse_map_path: Optional path to diffuse texture map
    
    Returns:
        Dictionary with S3 URLs for uploaded assets
    """
    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not installed, skipping S3 upload. Install with: pip install boto3")
        return {
            "mesh_url": f"file://{mesh_path}",
            "smplx_params_url": "not_uploaded",
            "status": "boto3_not_installed",
        }
    
    if bucket_name is None:
        bucket_name = os.getenv("S3_BUCKET_NAME", "avatar-service-assets")
    
    s3_client = boto3.client("s3")
    
    urls = {}
    
    # Upload mesh
    if mesh_path and os.path.exists(mesh_path):
        mesh_key = f"avatars/{user_id}/mesh.obj"
        s3_client.upload_file(mesh_path, bucket_name, mesh_key)
        mesh_url = f"s3://{bucket_name}/{mesh_key}"
        urls["mesh_url"] = mesh_url
        logger.info(f"Uploaded mesh to {mesh_url}")
    else:
        logger.warning(f"Mesh file not found at {mesh_path}")
        urls["mesh_url"] = None
    
    # Upload SMPL-X parameters as JSON
    if smplx_params:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(smplx_params, f, indent=2)
            smplx_temp_path = f.name
        
        try:
            smplx_key = f"avatars/{user_id}/smplx_params.json"
            s3_client.upload_file(smplx_temp_path, bucket_name, smplx_key)
            smplx_url = f"s3://{bucket_name}/{smplx_key}"
            urls["smplx_params_url"] = smplx_url
            logger.info(f"Uploaded SMPL-X params to {smplx_url}")
        finally:
            os.unlink(smplx_temp_path)
    else:
        urls["smplx_params_url"] = None
    
    # Upload optional assets
    if normal_map_path and os.path.exists(normal_map_path):
        normal_key = f"avatars/{user_id}/normal_map.png"
        s3_client.upload_file(normal_map_path, bucket_name, normal_key)
        urls["normal_map_url"] = f"s3://{bucket_name}/{normal_key}"
        logger.info(f"Uploaded normal map to {urls['normal_map_url']}")
    
    if texture_path and os.path.exists(texture_path):
        texture_key = f"avatars/{user_id}/texture.png"
        s3_client.upload_file(texture_path, bucket_name, texture_key)
        urls["texture_url"] = f"s3://{bucket_name}/{texture_key}"
        logger.info(f"Uploaded texture to {urls['texture_url']}")
    
    if diffuse_map_path and os.path.exists(diffuse_map_path):
        diffuse_key = f"avatars/{user_id}/diffuse_map.png"
        s3_client.upload_file(diffuse_map_path, bucket_name, diffuse_key)
        urls["diffuse_map_url"] = f"s3://{bucket_name}/{diffuse_key}"
        logger.info(f"Uploaded diffuse map to {urls['diffuse_map_url']}")
    
    logger.info(f"Uploaded avatar assets for user {user_id} to S3")
    return urls

