"""
3D garment reconstruction service.

Reconstructs 3D garment meshes from 2D images:
- Garment parameterization
- 3D mesh generation
- UV mapping
"""

import numpy as np
from loguru import logger
from PIL import Image

from app.config import get_settings


class GarmentReconstructionService:
    """Service for reconstructing 3D garments from 2D images."""

    def __init__(self):
        """
        Initialize the garment reconstruction service.
        
        Uses geometric mesh generation based on garment type and mask analysis.
        Can be extended with ML-based 3D reconstruction models.
        """
        self.settings = get_settings()

    def reconstruct_3d(
        self,
        image: Image.Image,
        segmented_mask: np.ndarray,
        key_points: dict[str, list[float]],
        garment_type: str,
    ) -> dict[str, Any]:
        """
        Reconstruct 3D garment mesh from 2D image.

        Args:
            image: Original garment image
            segmented_mask: Binary mask of garment
            key_points: Key points dictionary
            garment_type: Type of garment (shirt, pants, etc.)

        Returns:
            Dictionary with:
                - vertices: Nx3 array of vertex positions
                - faces: Mx3 array of face indices
                - uvs: Nx2 array of UV coordinates
                - texture: Texture image
        """
        logger.info(f"Reconstructing 3D garment: type={garment_type}")

        # Generate 3D mesh from 2D image and mask
        vertices, faces = self._generate_mesh_from_mask(
            garment_type, segmented_mask
        )
        uvs = self._generate_uvs(vertices, faces)
        texture = image  # Use original image as texture

        logger.info(f"3D garment reconstructed: {len(vertices)} vertices, {len(faces)} faces")

        return {
            "vertices": vertices,
            "faces": faces,
            "uvs": uvs,
            "texture": texture,
        }

    def _generate_mesh_from_mask(
        self, garment_type: str, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D mesh from garment type and segmentation mask.

        Creates garment-appropriate 3D geometry:
        - Shirts/dresses: Cylindrical body shape
        - Pants: Two cylindrical legs
        - Shoes: Foot-shaped mesh
        - Other: Curved surface following mask shape

        Args:
            garment_type: Type of garment
            mask: Binary segmentation mask

        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        # Get mask dimensions
        mask_height, mask_width = mask.shape
        
        # Create a more realistic mesh based on garment type and mask shape
        if garment_type in ["shirt", "t-shirt", "dress"]:
            # Create a cylindrical/body-like shape for upper body garments
            # Use mask to determine width and height
            mask_area = np.sum(mask > 0)
            if mask_area > 0:
                # Estimate garment dimensions from mask
                y_coords, x_coords = np.where(mask > 0)
                width_estimate = (x_coords.max() - x_coords.min()) / mask_width
                height_estimate = (y_coords.max() - y_coords.min()) / mask_height
            else:
                width_estimate = 0.6
                height_estimate = 0.8

            # Create a cylindrical mesh (simplified shirt shape)
            # Generate vertices in a cylindrical pattern
            num_rings = 8  # Vertical rings
            num_points_per_ring = 16  # Points around the cylinder
            
            vertices = []
            for ring in range(num_rings):
                y = -height_estimate * (ring / (num_rings - 1))  # From top to bottom
                radius = width_estimate * 0.3 * (1 - 0.1 * (ring / num_rings))  # Taper slightly
                for point in range(num_points_per_ring):
                    angle = 2 * np.pi * point / num_points_per_ring
                    x = radius * np.cos(angle)
                    z = radius * np.sin(angle) * 0.3  # Flatten front-to-back
                    vertices.append([x, y, z])
            
            vertices = np.array(vertices, dtype=np.float32)
            
            # Create faces (quadrilaterals converted to triangles)
            faces = []
            for ring in range(num_rings - 1):
                for point in range(num_points_per_ring):
                    # Current ring indices
                    curr = ring * num_points_per_ring + point
                    next_curr = ring * num_points_per_ring + ((point + 1) % num_points_per_ring)
                    # Next ring indices
                    next_ring = (ring + 1) * num_points_per_ring + point
                    next_next_ring = (ring + 1) * num_points_per_ring + ((point + 1) % num_points_per_ring)
                    
                    # Two triangles per quad
                    faces.append([curr, next_curr, next_ring])
                    faces.append([next_curr, next_next_ring, next_ring])
            
            faces = np.array(faces, dtype=np.uint32)

        elif garment_type in ["pants", "shorts"]:
            # Create two cylindrical legs
            num_rings = 10
            num_points_per_ring = 12
            
            vertices = []
            # Left leg
            for ring in range(num_rings):
                y = -0.8 * (ring / (num_rings - 1))
                radius = 0.15 * (1 - 0.2 * (ring / num_rings))
                for point in range(num_points_per_ring):
                    angle = 2 * np.pi * point / num_points_per_ring
                    x = -0.2 + radius * np.cos(angle)
                    z = radius * np.sin(angle) * 0.5
                    vertices.append([x, y, z])
            
            # Right leg
            for ring in range(num_rings):
                y = -0.8 * (ring / (num_rings - 1))
                radius = 0.15 * (1 - 0.2 * (ring / num_rings))
                for point in range(num_points_per_ring):
                    angle = 2 * np.pi * point / num_points_per_ring
                    x = 0.2 + radius * np.cos(angle)
                    z = radius * np.sin(angle) * 0.5
                    vertices.append([x, y, z])
            
            vertices = np.array(vertices, dtype=np.float32)
            
            # Create faces for both legs
            faces = []
            for leg_offset in [0, num_rings * num_points_per_ring]:
                for ring in range(num_rings - 1):
                    for point in range(num_points_per_ring):
                        curr = leg_offset + ring * num_points_per_ring + point
                        next_curr = leg_offset + ring * num_points_per_ring + ((point + 1) % num_points_per_ring)
                        next_ring = leg_offset + (ring + 1) * num_points_per_ring + point
                        next_next_ring = leg_offset + (ring + 1) * num_points_per_ring + ((point + 1) % num_points_per_ring)
                        
                        faces.append([curr, next_curr, next_ring])
                        faces.append([next_curr, next_next_ring, next_ring])
            
            faces = np.array(faces, dtype=np.uint32)

        elif garment_type in ["shoes", "boots", "socks"]:
            # Simple foot-shaped mesh
            vertices = np.array(
                [
                    [-0.1, -0.05, 0.0],  # Heel left
                    [0.1, -0.05, 0.0],   # Heel right
                    [-0.1, 0.15, 0.0],   # Toe left
                    [0.1, 0.15, 0.0],    # Toe right
                    [0.0, 0.0, 0.05],    # Top center
                ],
                dtype=np.float32,
            )
            faces = np.array(
                [[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4], [0, 1, 3], [0, 3, 2]],
                dtype=np.uint32,
            )

        else:
            # Default: simple curved surface based on mask
            # Create a basic mesh that follows the mask shape
            mask_area = np.sum(mask > 0)
            if mask_area > 0:
                y_coords, x_coords = np.where(mask > 0)
                width_estimate = (x_coords.max() - x_coords.min()) / mask_width if len(x_coords) > 0 else 0.5
                height_estimate = (y_coords.max() - y_coords.min()) / mask_height if len(y_coords) > 0 else 0.8
            else:
                width_estimate = 0.5
                height_estimate = 0.8

            # Simple plane with slight curve
            num_x = 8
            num_y = 10
            vertices = []
            for y_idx in range(num_y):
                y = -height_estimate * (y_idx / (num_y - 1))
                for x_idx in range(num_x):
                    x = width_estimate * (x_idx / (num_x - 1) - 0.5)
                    z = 0.05 * np.sin(x_idx * np.pi / num_x)  # Slight curve
                    vertices.append([x, y, z])
            
            vertices = np.array(vertices, dtype=np.float32)
            
            # Create faces
            faces = []
            for y_idx in range(num_y - 1):
                for x_idx in range(num_x - 1):
                    curr = y_idx * num_x + x_idx
                    next_x = y_idx * num_x + (x_idx + 1)
                    next_y = (y_idx + 1) * num_x + x_idx
                    next_xy = (y_idx + 1) * num_x + (x_idx + 1)
                    
                    faces.append([curr, next_x, next_y])
                    faces.append([next_x, next_xy, next_y])
            
            faces = np.array(faces, dtype=np.uint32)

        return vertices, faces

    def _generate_uvs(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> np.ndarray:
        """
        Generate UV coordinates for texture mapping.

        Automatically selects projection method:
        - Cylindrical mapping for body-like shapes (shirts, pants)
        - Planar mapping for flat garments

        Args:
            vertices: Mesh vertices (Nx3)
            faces: Mesh faces (Mx3)

        Returns:
            UV coordinates (Nx2) in [0, 1] range
        """
        # Determine if mesh is cylindrical (like shirt) or planar
        # Check if vertices form a roughly cylindrical shape
        center = vertices.mean(axis=0)
        distances_from_center = np.linalg.norm(vertices - center, axis=1)
        max_distance = distances_from_center.max()
        
        # If vertices are spread in a circle, use cylindrical mapping
        # Otherwise use planar mapping
        if max_distance > 0.1:  # Has some spread
            # Cylindrical UV mapping (good for shirts, pants)
            # U = angle around center, V = height
            centered = vertices - center
            angles = np.arctan2(centered[:, 0], centered[:, 2])  # Angle in XZ plane
            u_coords = (angles + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
            
            # V coordinate based on Y (height)
            min_y = vertices[:, 1].min()
            max_y = vertices[:, 1].max()
            y_range = max_y - min_y
            if y_range > 1e-6:
                v_coords = (vertices[:, 1] - min_y) / y_range
            else:
                v_coords = np.zeros(len(vertices))
            
            uvs = np.column_stack([u_coords, v_coords])
        else:
            # Planar UV mapping (good for flat garments)
            # Project to XZ plane
            min_vals = vertices.min(axis=0)
            max_vals = vertices.max(axis=0)
            ranges = max_vals - min_vals
            ranges[ranges < 1e-6] = 1  # Avoid division by zero

            # Use X and Z coordinates for UV (Y is depth)
            uvs = (vertices[:, [0, 2]] - min_vals[[0, 2]]) / ranges[[0, 2]]
        
        # Ensure UVs are in [0, 1] range
        uvs = np.clip(uvs, 0.0, 1.0)
        
        return uvs.astype(np.float32)
