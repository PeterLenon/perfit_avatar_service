"""
Storage service for S3/MinIO operations.

Handles upload and retrieval of:
- SMPL-X parameters (numpy arrays serialized to .npz)
- Body meshes (OBJ format)
- Source images
"""

import io
import json
from typing import Any

import boto3
import numpy as np
from botocore.client import Config
from botocore.exceptions import ClientError
from loguru import logger

from app.config import MinioSettings


class StorageService:
    """
    S3-compatible storage service.

    Works with both MinIO (local development) and AWS S3 (production).
    """

    def __init__(self, settings: MinioSettings):
        self.settings = settings
        self._client = None

    @property
    def client(self):
        """Lazy initialization of S3 client."""
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=f"{'https' if self.settings.secure else 'http'}://{self.settings.endpoint}",
                aws_access_key_id=self.settings.access_key,
                aws_secret_access_key=self.settings.secret_key,
                config=Config(signature_version="s3v4"),
                region_name="us-east-1",  # Required for MinIO
            )
        return self._client

    def upload_smplx_params(
        self,
        user_id: str,
        avatar_id: str,
        params: dict[str, Any],
    ) -> str:
        """
        Upload SMPL-X parameters as a compressed numpy archive.

        Args:
            user_id: User identifier
            avatar_id: Avatar identifier
            params: Dictionary containing SMPL-X parameters
                   (betas, body_pose, global_orient, etc.)

        Returns:
            URL to the uploaded file
        """
        key = f"avatars/{user_id}/{avatar_id}/smplx_params.npz"

        # Convert lists to numpy arrays and save to buffer
        buffer = io.BytesIO()
        np_params = {k: np.array(v) for k, v in params.items()}
        np.savez_compressed(buffer, **np_params)
        buffer.seek(0)

        try:
            self.client.upload_fileobj(
                buffer,
                self.settings.bucket,
                key,
                ExtraArgs={"ContentType": "application/octet-stream"},
            )
            logger.info(f"Uploaded SMPL-X params to {key}")
            return self._get_url(key)
        except ClientError as e:
            logger.error(f"Failed to upload SMPL-X params: {e}")
            raise

    def upload_mesh(
        self,
        user_id: str,
        avatar_id: str,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> str:
        """
        Upload body mesh as an OBJ file.

        Args:
            user_id: User identifier
            avatar_id: Avatar identifier
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of face indices

        Returns:
            URL to the uploaded file
        """
        key = f"avatars/{user_id}/{avatar_id}/mesh.obj"

        # Generate OBJ content
        obj_content = self._generate_obj(vertices, faces)

        try:
            self.client.put_object(
                Bucket=self.settings.bucket,
                Key=key,
                Body=obj_content.encode("utf-8"),
                ContentType="model/obj",
            )
            logger.info(f"Uploaded mesh to {key}")
            return self._get_url(key)
        except ClientError as e:
            logger.error(f"Failed to upload mesh: {e}")
            raise

    def upload_source_image(
        self,
        user_id: str,
        avatar_id: str,
        image_data: bytes,
        content_type: str = "image/jpeg",
    ) -> str:
        """
        Upload the source image used for avatar creation.

        Args:
            user_id: User identifier
            avatar_id: Avatar identifier
            image_data: Raw image bytes
            content_type: MIME type of the image

        Returns:
            URL to the uploaded file
        """
        extension = content_type.split("/")[-1]
        key = f"avatars/{user_id}/{avatar_id}/source.{extension}"

        try:
            self.client.put_object(
                Bucket=self.settings.bucket,
                Key=key,
                Body=image_data,
                ContentType=content_type,
            )
            logger.info(f"Uploaded source image to {key}")
            return self._get_url(key)
        except ClientError as e:
            logger.error(f"Failed to upload source image: {e}")
            raise

    def download_smplx_params(self, url: str) -> dict[str, np.ndarray]:
        """
        Download and load SMPL-X parameters.

        Args:
            url: URL to the .npz file

        Returns:
            Dictionary of numpy arrays
        """
        key = self._url_to_key(url)

        try:
            response = self.client.get_object(Bucket=self.settings.bucket, Key=key)
            buffer = io.BytesIO(response["Body"].read())
            data = np.load(buffer, allow_pickle=False)
            return dict(data)
        except ClientError as e:
            logger.error(f"Failed to download SMPL-X params: {e}")
            raise

    def generate_presigned_url(self, key: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for temporary access.

        Args:
            key: Object key in the bucket
            expiration: URL expiration time in seconds

        Returns:
            Presigned URL
        """
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.settings.bucket, "Key": key},
                ExpiresIn=expiration,
            )
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise

    def _get_url(self, key: str) -> str:
        """Get the public URL for an object."""
        protocol = "https" if self.settings.secure else "http"
        return f"{protocol}://{self.settings.endpoint}/{self.settings.bucket}/{key}"

    def _url_to_key(self, url: str) -> str:
        """Extract object key from URL."""
        # URL format: http://endpoint/bucket/key
        parts = url.split(f"/{self.settings.bucket}/")
        if len(parts) != 2:
            raise ValueError(f"Invalid storage URL: {url}")
        return parts[1]

    @staticmethod
    def _generate_obj(vertices: np.ndarray, faces: np.ndarray) -> str:
        """
        Generate OBJ file content from vertices and faces.

        Args:
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of face indices (1-indexed for OBJ)

        Returns:
            OBJ file content as string
        """
        lines = ["# Generated by Perfit Avatar Service"]

        # Write vertices
        for v in vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

        # Write faces (OBJ uses 1-indexed vertices)
        for f in faces:
            lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")

        return "\n".join(lines)

    def ensure_bucket_exists(self) -> None:
        """Create the bucket if it doesn't exist."""
        try:
            self.client.head_bucket(Bucket=self.settings.bucket)
            logger.info(f"Bucket {self.settings.bucket} exists")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchBucket"):
                logger.info(f"Creating bucket {self.settings.bucket}")
                self.client.create_bucket(Bucket=self.settings.bucket)
            else:
                raise
