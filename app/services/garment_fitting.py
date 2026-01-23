"""
Garment fitting service.

Fits one or more garments onto an avatar with physics simulation:
- Handles multiple garments with layering
- Determines layering order (configurable with defaults)
- Fits garments with physics simulation
- Combines meshes for final output
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from app.config import get_settings
from app.services.cloth_physics import ClothPhysicsService
from app.services.garment_reconstruction import GarmentReconstructionService


class GarmentFittingService:
    """Service for fitting garments onto avatars."""

    # Default layering order (lower number = closer to body)
    # This is used if no custom config is provided
    DEFAULT_LAYERING_ORDER = {
        "underwear": 0,
        "undershirt": 1,
        "socks": 2,
        "shirt": 3,
        "t-shirt": 4,
        "dress": 5,
        "pants": 6,
        "shorts": 7,
        "skirt": 8,
        "jacket": 9,
        "coat": 10,
        "sweater": 11,
        "shoes": 12,
        "boots": 13,
        "hat": 14,
        "belt": 15,
    }

    def __init__(self):
        """
        Initialize the garment fitting service.
        
        Loads layering order from config file if provided, otherwise uses defaults.
        """
        self.settings = get_settings()
        self.physics_service = ClothPhysicsService()
        self.reconstruction_service = GarmentReconstructionService()
        
        # Load layering order (configurable)
        self.layering_order = self._load_layering_order()

    def _load_layering_order(self) -> dict[str, int]:
        """
        Load layering order from config file or use defaults.
        
        Returns:
            Dictionary mapping garment types to layer numbers
        """
        config_path = self.settings.fitting_room.layering_config_path
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    custom_order = json.load(f)
                logger.info(f"Loaded custom layering order from {config_path}")
                # Merge with defaults (custom overrides defaults)
                layering_order = self.DEFAULT_LAYERING_ORDER.copy()
                layering_order.update(custom_order)
                return layering_order
            except Exception as e:
                logger.warning(f"Failed to load layering config from {config_path}: {e}. Using defaults.")
                return self.DEFAULT_LAYERING_ORDER.copy()
        else:
            logger.debug("Using default layering order")
            return self.DEFAULT_LAYERING_ORDER.copy()

    def determine_layering_order(
        self, garment_types: list[str]
    ) -> list[tuple[int, str]]:
        """
        Determine the layering order for multiple garments.

        Args:
            garment_types: List of garment types

        Returns:
            List of (layer_order, garment_type) tuples, sorted by layer order
        """
        # Get layering order for each garment
        layered = [
            (self.layering_order.get(gt.lower(), 100), gt) for gt in garment_types
        ]
        # Sort by layer order (innermost first)
        layered.sort(key=lambda x: x[0])
        return layered

    def fit_garments_to_avatar(
        self,
        avatar_vertices: np.ndarray,
        avatar_faces: np.ndarray,
        garments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Fit multiple garments onto an avatar.

        Args:
            avatar_vertices: Avatar mesh vertices
            avatar_faces: Avatar mesh faces
            garments: List of garment dicts, each with:
                - vertices: Garment mesh vertices
                - faces: Garment mesh faces
                - garment_type: Type of garment
                - material_properties: Material properties dict
                - key_points: Key points dict

        Returns:
            Dictionary with:
                - combined_vertices: Combined mesh vertices
                - combined_faces: Combined mesh faces
                - layering_order: Order garments were fitted
                - fitting_parameters: Fitting parameters for each garment
        """
        logger.info(f"Fitting {len(garments)} garments to avatar")

        # Determine layering order
        garment_types = [g["garment_type"] for g in garments]
        layering_order = self.determine_layering_order(garment_types)
        logger.info(f"Layering order: {layering_order}")

        # Sort garments by layering order
        sorted_garments = sorted(
            garments,
            key=lambda g: self.layering_order.get(g["garment_type"].lower(), 100),
        )

        # Fit garments in order (innermost first)
        fitted_garments = []
        current_body_vertices = avatar_vertices.copy()

        for garment in sorted_garments:
            logger.info(f"Fitting {garment['garment_type']}...")

            # Find attachment points (garment to body)
            attachment_points = self._find_attachment_points(
                garment["vertices"],
                garment["key_points"],
                current_body_vertices,
                garment["garment_type"],
            )

            # Simulate draping
            draped_vertices = self.physics_service.simulate_draping(
                garment_vertices=garment["vertices"],
                garment_faces=garment["faces"],
                body_vertices=current_body_vertices,
                attachment_points=attachment_points,
                material_properties=garment["material_properties"],
            )

            fitted_garments.append({
                "vertices": draped_vertices,
                "faces": garment["faces"],
                "garment_type": garment["garment_type"],
                "attachment_points": attachment_points,
            })

            # Update body vertices for next garment (include current garment in collision)
            # This allows outer garments to collide with inner garments
            current_body_vertices = np.vstack([current_body_vertices, draped_vertices])

        # Combine all meshes
        combined_vertices, combined_faces = self._combine_meshes(
            avatar_vertices, avatar_faces, fitted_garments
        )

        logger.info(
            f"Fitted {len(garments)} garments: {len(combined_vertices)} vertices, {len(combined_faces)} faces"
        )

        return {
            "combined_vertices": combined_vertices,
            "combined_faces": combined_faces,
            "layering_order": [g["garment_type"] for g in sorted_garments],
            "fitting_parameters": {
                g["garment_type"]: {
                    "attachment_points": g["attachment_points"],
                }
                for g in fitted_garments
            },
        }

    def _find_attachment_points(
        self,
        garment_vertices: np.ndarray,
        key_points: dict[str, list[float]],
        body_vertices: np.ndarray,
        garment_type: str,
    ) -> list[tuple[int, int]]:
        """
        Find attachment points between garment and body.

        Uses key points from garment and finds nearest vertices on both
        garment and body meshes for attachment.

        Args:
            garment_vertices: Garment vertices (Nx3)
            key_points: Key points dict from garment (2D or 3D coordinates)
            body_vertices: Body vertices (Mx3)
            garment_type: Type of garment

        Returns:
            List of (garment_vertex_idx, body_vertex_idx) tuples
        """
        attachment_points = []

        # Helper function to find nearest vertex to a point
        def find_nearest_vertex(vertices: np.ndarray, target_point: np.ndarray) -> int:
            """Find index of vertex closest to target point."""
            if len(target_point) == 2:
                # 2D key point - use only X and Z coordinates (ignore Y/height)
                distances = np.linalg.norm(vertices[:, [0, 2]] - target_point, axis=1)
            else:
                # 3D point - use all coordinates
                distances = np.linalg.norm(vertices - target_point, axis=1)
            return int(np.argmin(distances))

        # Define attachment regions based on garment type
        if garment_type in ["shirt", "t-shirt", "dress"]:
            # Attach at collar/neck area
            if "collar" in key_points:
                collar_point = np.array(key_points["collar"])
                # Find body vertex at neck/collar level (typically around y=0.1-0.15 in SMPL)
                # Filter body vertices near collar height
                collar_height = collar_point[1] if len(collar_point) > 1 else 0.12
                height_tolerance = 0.05
                body_candidates = body_vertices[
                    np.abs(body_vertices[:, 1] - collar_height) < height_tolerance
                ]
                if len(body_candidates) > 0:
                    # Find closest to collar X position
                    collar_xz = collar_point[[0, 2]] if len(collar_point) > 2 else collar_point
                    body_distances = np.linalg.norm(
                        body_candidates[:, [0, 2]] - collar_xz, axis=1
                    )
                    body_idx = np.where(
                        np.abs(body_vertices[:, 1] - collar_height) < height_tolerance
                    )[0][np.argmin(body_distances)]
                else:
                    body_idx = find_nearest_vertex(body_vertices, collar_point)
                
                garment_idx = find_nearest_vertex(garment_vertices, collar_point)
                attachment_points.append((garment_idx, body_idx))

        elif garment_type in ["pants", "shorts"]:
            # Attach at waist
            if "waist" in key_points:
                waist_point = np.array(key_points["waist"])
                # Waist is typically around y=-0.1 to 0.0 in SMPL
                waist_height = waist_point[1] if len(waist_point) > 1 else -0.05
                height_tolerance = 0.03
                body_candidates = body_vertices[
                    np.abs(body_vertices[:, 1] - waist_height) < height_tolerance
                ]
                if len(body_candidates) > 0:
                    waist_xz = waist_point[[0, 2]] if len(waist_point) > 2 else waist_point
                    body_distances = np.linalg.norm(
                        body_candidates[:, [0, 2]] - waist_xz, axis=1
                    )
                    body_idx = np.where(
                        np.abs(body_vertices[:, 1] - waist_height) < height_tolerance
                    )[0][np.argmin(body_distances)]
                else:
                    body_idx = find_nearest_vertex(body_vertices, waist_point)
                
                garment_idx = find_nearest_vertex(garment_vertices, waist_point)
                attachment_points.append((garment_idx, body_idx))

        elif garment_type in ["shoes", "boots", "socks"]:
            # Attach at ankle/feet
            # Find lowest body vertices (feet)
            lowest_body_y = body_vertices[:, 1].min()
            foot_tolerance = 0.02
            foot_vertices = body_vertices[
                body_vertices[:, 1] < lowest_body_y + foot_tolerance
            ]
            if len(foot_vertices) > 0:
                # Use center of foot area
                foot_center = foot_vertices.mean(axis=0)
                body_idx = find_nearest_vertex(body_vertices, foot_center)
                # Find corresponding garment vertex (lowest point)
                lowest_garment_y = garment_vertices[:, 1].min()
                garment_foot = garment_vertices[
                    garment_vertices[:, 1] < lowest_garment_y + foot_tolerance
                ]
                if len(garment_foot) > 0:
                    garment_center = garment_foot.mean(axis=0)
                    garment_idx = find_nearest_vertex(garment_vertices, garment_center)
                    attachment_points.append((garment_idx, body_idx))

        # If no specific attachment points found, use a default (center-top for upper body, center-bottom for lower)
        if not attachment_points:
            logger.warning(f"No attachment points found for {garment_type}, using default")
            if garment_type in ["shirt", "t-shirt", "jacket", "coat", "sweater"]:
                # Attach at top center
                top_garment = garment_vertices[garment_vertices[:, 1].argmax()]
                top_body = body_vertices[body_vertices[:, 1].argmax()]
                garment_idx = find_nearest_vertex(garment_vertices, top_garment)
                body_idx = find_nearest_vertex(body_vertices, top_body)
            else:
                # Attach at center
                garment_center = garment_vertices.mean(axis=0)
                body_center = body_vertices.mean(axis=0)
                garment_idx = find_nearest_vertex(garment_vertices, garment_center)
                body_idx = find_nearest_vertex(body_vertices, body_center)
            attachment_points.append((garment_idx, body_idx))

        logger.debug(f"Found {len(attachment_points)} attachment points for {garment_type}")
        return attachment_points

    def _combine_meshes(
        self,
        avatar_vertices: np.ndarray,
        avatar_faces: np.ndarray,
        fitted_garments: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Combine avatar mesh with all fitted garment meshes.

        Args:
            avatar_vertices: Avatar vertices
            avatar_faces: Avatar faces
            fitted_garments: List of fitted garment dicts

        Returns:
            Tuple of (combined_vertices, combined_faces)
        """
        all_vertices = [avatar_vertices]
        all_faces = [avatar_faces]

        vertex_offset = len(avatar_vertices)

        for garment in fitted_garments:
            all_vertices.append(garment["vertices"])
            # Offset face indices
            offset_faces = garment["faces"] + vertex_offset
            all_faces.append(offset_faces)
            vertex_offset += len(garment["vertices"])

        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)

        return combined_vertices, combined_faces
