"""
Mesh loading service.

Handles loading meshes from various sources:
- OBJ files from storage
- SMPL parameter reconstruction
- Mesh format conversion
"""

import uuid
from typing import Any

import numpy as np
import smplx
import torch
from loguru import logger

from app.config import get_settings
from app.services.storage import StorageService


class MeshLoaderService:
    """Service for loading and reconstructing meshes."""

    def __init__(self, storage_service: StorageService):
        """
        Initialize the mesh loader service.

        Args:
            storage_service: Storage service for downloading files
        """
        self.storage = storage_service
        self.settings = get_settings()
        self._smpl_models = {}  # Cache SMPL models by gender

    def load_mesh_from_url(self, mesh_url: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load mesh from storage URL (OBJ format).

        Args:
            mesh_url: URL to the OBJ file

        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        logger.info(f"Loading mesh from URL: {mesh_url}")
        vertices, faces = self.storage.download_mesh(mesh_url)
        logger.info(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
        return vertices, faces

    def reconstruct_mesh_from_smpl(
        self,
        smpl_params_url: str,
        gender: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct mesh from SMPL parameters.

        Args:
            smpl_params_url: URL to the SMPL parameters file (.npz)
            gender: Gender for SMPL model ('male', 'female', 'neutral')

        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        logger.info(f"Reconstructing mesh from SMPL params: {smpl_params_url}")

        # Load SMPL parameters
        params = self.storage.download_smplx_params(smpl_params_url)

        # Get or create SMPL model
        smpl_model = self._get_smpl_model(gender)

        # Extract parameters
        betas = torch.tensor(params["betas"], dtype=torch.float32).unsqueeze(0)
        body_pose = torch.tensor(
            params["body_pose"], dtype=torch.float32
        ).reshape(1, 23, 3)
        global_orient = torch.tensor(
            params["global_orient"], dtype=torch.float32
        ).reshape(1, 1, 3)

        # Generate mesh
        with torch.no_grad():
            output = smpl_model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                return_verts=True,
            )

        vertices = output.vertices[0].numpy()
        faces = smpl_model.faces

        logger.info(f"Reconstructed mesh: {len(vertices)} vertices, {len(faces)} faces")
        return vertices, faces

    def _get_smpl_model(self, gender: str):
        """
        Get or create SMPL model for given gender.

        Caches models to avoid reloading.
        """
        if gender not in self._smpl_models:
            logger.info(f"Loading SMPL model ({gender}) from {self.settings.ml.smpl_model_path}")
            self._smpl_models[gender] = smplx.create(
                str(self.settings.ml.smpl_model_path),
                model_type="smpl",
                gender=gender,
                use_pca=False,
                batch_size=1,
            )
        return self._smpl_models[gender]
