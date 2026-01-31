"""
Garment fitting service.

Fits one or more garments onto an avatar with physics simulation:
- Handles multiple garments with layering
- Determines layering order (configurable with defaults)
- Fits garments with proper 3D body-aware alignment
- Combines meshes for final output
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from app.config import get_settings
from app.services.cloth_physics import ClothPhysicsService


# SMPL joint indices for body landmarks
SMPL_JOINTS = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_hand": 22,
    "right_hand": 23,
}

# Key SMPL vertex indices for body regions
SMPL_VERTICES = {
    "head_top": 411,
    "chin": 3052,
    "neck_front": 3068,
    "neck_back": 828,
    "left_shoulder_top": 3011,
    "right_shoulder_top": 6470,
    "chest_center": 3076,
    "waist_front": 3500,
    "waist_back": 3022,
    "crotch": 3149,
    "left_ankle_inner": 3327,
    "right_ankle_inner": 6728,
}


class GarmentFittingService:
    """Service for fitting garments onto avatars."""

    # Default layering order (lower number = closer to body)
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

    # Garment type to body region mapping
    GARMENT_BODY_REGIONS = {
        "shirt": "torso_upper",
        "t-shirt": "torso_upper",
        "dress": "torso_full",
        "jacket": "torso_upper",
        "coat": "torso_full",
        "sweater": "torso_upper",
        "pants": "legs_full",
        "shorts": "legs_upper",
        "skirt": "legs_upper",
        "underwear": "torso_lower",
        "undershirt": "torso_upper",
        "socks": "feet",
        "shoes": "feet",
        "boots": "feet",
        "hat": "head",
        "belt": "waist",
    }

    def __init__(self):
        """Initialize the garment fitting service."""
        self.settings = get_settings()
        self.physics_service = ClothPhysicsService()
        self.layering_order = self._load_layering_order()

    def _load_layering_order(self) -> dict[str, int]:
        """Load layering order from config file or use defaults."""
        config_path = self.settings.fitting_room.layering_config_path

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    custom_order = json.load(f)
                logger.info(f"Loaded custom layering order from {config_path}")
                layering_order = self.DEFAULT_LAYERING_ORDER.copy()
                layering_order.update(custom_order)
                return layering_order
            except Exception as e:
                logger.warning(f"Failed to load layering config: {e}. Using defaults.")
                return self.DEFAULT_LAYERING_ORDER.copy()
        else:
            logger.debug("Using default layering order")
            return self.DEFAULT_LAYERING_ORDER.copy()

    def determine_layering_order(
        self, garment_types: list[str]
    ) -> list[tuple[int, str]]:
        """Determine the layering order for multiple garments."""
        layered = [
            (self.layering_order.get(gt.lower(), 100), gt) for gt in garment_types
        ]
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
            avatar_vertices: Avatar mesh vertices (SMPL format, 6890 vertices)
            avatar_faces: Avatar mesh faces
            garments: List of garment dicts with vertices, faces, garment_type,
                      material_properties, key_points

        Returns:
            Dictionary with combined_vertices, combined_faces, layering_order,
            fitting_parameters
        """
        logger.info(f"Fitting {len(garments)} garments to avatar")

        # Extract body landmarks from avatar mesh
        body_landmarks = self._extract_body_landmarks(avatar_vertices)
        logger.debug(f"Extracted body landmarks: {list(body_landmarks.keys())}")

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
            garment_type = garment["garment_type"]
            logger.info(f"Fitting {garment_type}...")

            # Transform garment to fit body
            transformed_vertices = self._transform_garment_to_body(
                garment_vertices=garment["vertices"],
                garment_type=garment_type,
                body_vertices=avatar_vertices,
                body_landmarks=body_landmarks,
            )

            # Find attachment points based on body landmarks
            attachment_points = self._find_body_based_attachments(
                garment_vertices=transformed_vertices,
                body_vertices=current_body_vertices,
                garment_type=garment_type,
                body_landmarks=body_landmarks,
            )

            # Simulate draping with physics
            draped_vertices = self.physics_service.simulate_draping(
                garment_vertices=transformed_vertices,
                garment_faces=garment["faces"],
                body_vertices=current_body_vertices,
                attachment_points=attachment_points,
                material_properties=garment["material_properties"],
            )

            fitted_garments.append({
                "vertices": draped_vertices,
                "faces": garment["faces"],
                "garment_type": garment_type,
                "attachment_points": attachment_points,
            })

            # Update collision body for next garment
            current_body_vertices = np.vstack([current_body_vertices, draped_vertices])

        # Combine all meshes
        combined_vertices, combined_faces = self._combine_meshes(
            avatar_vertices, avatar_faces, fitted_garments
        )

        logger.info(
            f"Fitted {len(garments)} garments: {len(combined_vertices)} vertices"
        )

        return {
            "combined_vertices": combined_vertices,
            "combined_faces": combined_faces,
            "layering_order": [g["garment_type"] for g in sorted_garments],
            "fitting_parameters": {
                g["garment_type"]: {"attachment_points": g["attachment_points"]}
                for g in fitted_garments
            },
        }

    def _extract_body_landmarks(
        self, body_vertices: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Extract key body landmarks from SMPL mesh vertices.

        Args:
            body_vertices: SMPL mesh vertices (6890, 3)

        Returns:
            Dictionary mapping landmark names to 3D positions
        """
        landmarks = {}

        # Use known SMPL vertex indices for landmarks
        for name, idx in SMPL_VERTICES.items():
            if idx < len(body_vertices):
                landmarks[name] = body_vertices[idx].copy()

        # Calculate derived landmarks
        if "left_shoulder_top" in landmarks and "right_shoulder_top" in landmarks:
            landmarks["shoulder_center"] = (
                landmarks["left_shoulder_top"] + landmarks["right_shoulder_top"]
            ) / 2
            landmarks["shoulder_width"] = np.linalg.norm(
                landmarks["left_shoulder_top"] - landmarks["right_shoulder_top"]
            )

        if "waist_front" in landmarks and "waist_back" in landmarks:
            landmarks["waist_center"] = (
                landmarks["waist_front"] + landmarks["waist_back"]
            ) / 2

        # Calculate body bounding box for different regions
        landmarks["body_min"] = body_vertices.min(axis=0)
        landmarks["body_max"] = body_vertices.max(axis=0)
        landmarks["body_center"] = (landmarks["body_min"] + landmarks["body_max"]) / 2
        landmarks["body_height"] = landmarks["body_max"][1] - landmarks["body_min"][1]

        # Upper body region (above waist)
        if "waist_center" in landmarks:
            waist_y = landmarks["waist_center"][1]
            upper_mask = body_vertices[:, 1] > waist_y
            if upper_mask.any():
                upper_verts = body_vertices[upper_mask]
                landmarks["upper_body_min"] = upper_verts.min(axis=0)
                landmarks["upper_body_max"] = upper_verts.max(axis=0)
                landmarks["upper_body_center"] = (
                    landmarks["upper_body_min"] + landmarks["upper_body_max"]
                ) / 2

        # Lower body region (below waist)
        if "waist_center" in landmarks:
            waist_y = landmarks["waist_center"][1]
            lower_mask = body_vertices[:, 1] < waist_y
            if lower_mask.any():
                lower_verts = body_vertices[lower_mask]
                landmarks["lower_body_min"] = lower_verts.min(axis=0)
                landmarks["lower_body_max"] = lower_verts.max(axis=0)
                landmarks["lower_body_center"] = (
                    landmarks["lower_body_min"] + landmarks["lower_body_max"]
                ) / 2

        return landmarks

    def _transform_garment_to_body(
        self,
        garment_vertices: np.ndarray,
        garment_type: str,
        body_vertices: np.ndarray,
        body_landmarks: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Transform garment mesh to align with body.

        Garment meshes are now generated in SMPL-compatible coordinates,
        so we primarily need to position them correctly on the body.

        Args:
            garment_vertices: Garment mesh vertices (SMPL-compatible coords)
            garment_type: Type of garment
            body_vertices: Avatar body vertices
            body_landmarks: Extracted body landmarks

        Returns:
            Transformed garment vertices
        """
        garment_type_lower = garment_type.lower()
        region = self.GARMENT_BODY_REGIONS.get(garment_type_lower, "torso_upper")

        # Calculate garment bounding box
        garment_min = garment_vertices.min(axis=0)
        garment_max = garment_vertices.max(axis=0)
        garment_center = (garment_min + garment_max) / 2

        # Get body reference points
        body_center = body_landmarks["body_center"]

        # Position garment based on type
        if region == "torso_upper":
            # Shirt: align center X/Z with body, position Y at shoulder level
            if "shoulder_center" in body_landmarks:
                shoulder = body_landmarks["shoulder_center"]
                # Move garment so its top aligns with shoulders
                garment_top_y = garment_max[1]
                y_offset = shoulder[1] - garment_top_y + 0.02  # Slight below shoulders
                offset = np.array([
                    body_center[0] - garment_center[0],
                    y_offset,
                    body_center[2] - garment_center[2]
                ])
            else:
                offset = body_center - garment_center
                offset[1] = 0.1  # Position in upper body

        elif region in ["legs_full", "legs_upper"]:
            # Pants: align top with waist
            if "waist_center" in body_landmarks:
                waist = body_landmarks["waist_center"]
                garment_top_y = garment_max[1]
                y_offset = waist[1] - garment_top_y
                offset = np.array([
                    body_center[0] - garment_center[0],
                    y_offset,
                    body_center[2] - garment_center[2]
                ])
            else:
                offset = body_center - garment_center
                offset[1] = -0.3  # Position at lower body

        elif region == "feet":
            # Shoes: position at feet
            body_min_y = body_landmarks["body_min"][1]
            offset = np.array([0, body_min_y - garment_min[1], 0])

        else:
            # Default: center on body
            offset = body_center - garment_center

        transformed = garment_vertices + offset

        logger.info(
            f"Transformed {garment_type}: offset={offset}, "
            f"garment_range=[{garment_min[1]:.2f}, {garment_max[1]:.2f}]"
        )

        return transformed

    def _find_body_based_attachments(
        self,
        garment_vertices: np.ndarray,
        body_vertices: np.ndarray,
        garment_type: str,
        body_landmarks: dict[str, np.ndarray],
    ) -> list[tuple[int, int]]:
        """
        Find attachment points between garment and body using body landmarks.

        Args:
            garment_vertices: Transformed garment vertices
            body_vertices: Body vertices (may include previously fitted garments)
            garment_type: Type of garment
            body_landmarks: Extracted body landmarks

        Returns:
            List of (garment_vertex_idx, body_vertex_idx) tuples
        """
        attachment_points = []
        garment_type_lower = garment_type.lower()

        def find_nearest_vertex(vertices: np.ndarray, point: np.ndarray) -> int:
            """Find index of vertex closest to point."""
            distances = np.linalg.norm(vertices - point, axis=1)
            return int(np.argmin(distances))

        def find_vertices_near_height(
            vertices: np.ndarray, height: float, tolerance: float = 0.03
        ) -> np.ndarray:
            """Find vertex indices near a specific height."""
            mask = np.abs(vertices[:, 1] - height) < tolerance
            return np.where(mask)[0]

        # Upper body garments: attach at shoulders/collar
        if garment_type_lower in ["shirt", "t-shirt", "dress", "jacket", "coat", "sweater"]:
            # Find shoulder attachment points
            if "left_shoulder_top" in body_landmarks:
                left_shoulder = body_landmarks["left_shoulder_top"]
                body_idx = find_nearest_vertex(body_vertices, left_shoulder)

                # Find garment vertex at similar position (top-left area)
                garment_top_y = garment_vertices[:, 1].max()
                top_vertices = find_vertices_near_height(
                    garment_vertices, garment_top_y, tolerance=0.05
                )
                if len(top_vertices) > 0:
                    # Find leftmost among top vertices
                    top_verts = garment_vertices[top_vertices]
                    left_idx = top_vertices[np.argmin(top_verts[:, 0])]
                    attachment_points.append((int(left_idx), body_idx))

            if "right_shoulder_top" in body_landmarks:
                right_shoulder = body_landmarks["right_shoulder_top"]
                body_idx = find_nearest_vertex(body_vertices, right_shoulder)

                garment_top_y = garment_vertices[:, 1].max()
                top_vertices = find_vertices_near_height(
                    garment_vertices, garment_top_y, tolerance=0.05
                )
                if len(top_vertices) > 0:
                    top_verts = garment_vertices[top_vertices]
                    right_idx = top_vertices[np.argmax(top_verts[:, 0])]
                    attachment_points.append((int(right_idx), body_idx))

            # Neck/collar attachment
            if "neck_front" in body_landmarks:
                neck = body_landmarks["neck_front"]
                body_idx = find_nearest_vertex(body_vertices, neck)
                garment_idx = find_nearest_vertex(garment_vertices, neck)
                attachment_points.append((garment_idx, body_idx))

        # Lower body garments: attach at waist
        elif garment_type_lower in ["pants", "shorts", "skirt"]:
            if "waist_center" in body_landmarks:
                waist = body_landmarks["waist_center"]
                waist_y = waist[1]

                # Find body vertices at waist height
                waist_body_indices = find_vertices_near_height(
                    body_vertices, waist_y, tolerance=0.05
                )
                if len(waist_body_indices) > 0:
                    # Find garment vertices at top (waistband)
                    garment_top_y = garment_vertices[:, 1].max()
                    waist_garment_indices = find_vertices_near_height(
                        garment_vertices, garment_top_y, tolerance=0.05
                    )

                    if len(waist_garment_indices) > 0:
                        # Create multiple attachment points around waist
                        # Front
                        front_body = body_vertices[waist_body_indices]
                        front_idx = waist_body_indices[np.argmax(front_body[:, 2])]
                        front_garment = garment_vertices[waist_garment_indices]
                        front_g_idx = waist_garment_indices[
                            np.argmax(front_garment[:, 2])
                        ]
                        attachment_points.append((int(front_g_idx), int(front_idx)))

                        # Back
                        back_idx = waist_body_indices[np.argmin(front_body[:, 2])]
                        back_g_idx = waist_garment_indices[
                            np.argmin(front_garment[:, 2])
                        ]
                        attachment_points.append((int(back_g_idx), int(back_idx)))

                        # Sides
                        left_idx = waist_body_indices[np.argmin(front_body[:, 0])]
                        left_g_idx = waist_garment_indices[
                            np.argmin(front_garment[:, 0])
                        ]
                        attachment_points.append((int(left_g_idx), int(left_idx)))

                        right_idx = waist_body_indices[np.argmax(front_body[:, 0])]
                        right_g_idx = waist_garment_indices[
                            np.argmax(front_garment[:, 0])
                        ]
                        attachment_points.append((int(right_g_idx), int(right_idx)))

        # Footwear: attach at ankles
        elif garment_type_lower in ["shoes", "boots", "socks"]:
            if "left_ankle_inner" in body_landmarks:
                ankle = body_landmarks["left_ankle_inner"]
                body_idx = find_nearest_vertex(body_vertices, ankle)
                garment_idx = find_nearest_vertex(garment_vertices, ankle)
                attachment_points.append((garment_idx, body_idx))

            if "right_ankle_inner" in body_landmarks:
                ankle = body_landmarks["right_ankle_inner"]
                body_idx = find_nearest_vertex(body_vertices, ankle)
                garment_idx = find_nearest_vertex(garment_vertices, ankle)
                attachment_points.append((garment_idx, body_idx))

        # Fallback: use garment center to body center
        if not attachment_points:
            logger.warning(
                f"No specific attachments for {garment_type}, using center fallback"
            )
            garment_center = garment_vertices.mean(axis=0)
            body_center = body_landmarks.get(
                "body_center", body_vertices.mean(axis=0)
            )
            garment_idx = find_nearest_vertex(garment_vertices, garment_center)
            body_idx = find_nearest_vertex(body_vertices, body_center)
            attachment_points.append((garment_idx, body_idx))

        logger.debug(
            f"Found {len(attachment_points)} attachment points for {garment_type}"
        )
        return attachment_points

    def _combine_meshes(
        self,
        avatar_vertices: np.ndarray,
        avatar_faces: np.ndarray,
        fitted_garments: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine avatar mesh with all fitted garment meshes."""
        all_vertices = [avatar_vertices]
        all_faces = [avatar_faces]

        vertex_offset = len(avatar_vertices)

        for garment in fitted_garments:
            all_vertices.append(garment["vertices"])
            offset_faces = garment["faces"] + vertex_offset
            all_faces.append(offset_faces)
            vertex_offset += len(garment["vertices"])

        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)

        return combined_vertices, combined_faces
