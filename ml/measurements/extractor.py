"""
Body measurement extractor from SMPL parameters.

Extracts anthropometric measurements from SMPL body model parameters,
including heights, lengths, and circumferences.

Measurements are calculated using the SMPL mesh in T-pose (zeroed pose
parameters) for consistency, using only the shape (beta) parameters.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import smplx
import torch
import trimesh
from loguru import logger
from scipy.spatial import ConvexHull

# SMPL joint indices (from SMPL topology)
# Reference: https://meshcapade.wiki/SMPL
JOINT_INDICES = {
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

# Key vertex indices for measurements (approximate, based on SMPL topology)
# These vertices are used when joint positions aren't sufficient
VERTEX_INDICES = {
    "head_top": 411,  # Top of head
    "chin": 3052,  # Chin
    "left_armpit": 3011,  # Left armpit
    "right_armpit": 6470,  # Right armpit
    "crotch": 3149,  # Crotch center
    "left_ankle_inner": 3327,  # Inner left ankle
    "right_ankle_inner": 6728,  # Inner right ankle
}

# Module-level cache for SMPL model
_smpl_model: Optional[object] = None
_smpl_model_path: Optional[str] = None
_smpl_gender: Optional[str] = None


class MeasurementExtractor:
    """
    Extract body measurements from SMPL parameters.

    Uses the SMPL body model to generate a mesh from shape parameters,
    then calculates anthropometric measurements from the mesh geometry.

    All measurements are returned in centimeters.
    """

    # Height fractions for circumference measurements (relative to total height)
    # These are approximate anatomical positions
    CHEST_HEIGHT_FRACTION = 0.72  # Chest at ~72% of height from floor
    WAIST_HEIGHT_FRACTION = 0.58  # Waist at ~58% of height
    HIP_HEIGHT_FRACTION = 0.52  # Hip at ~52% of height
    NECK_HEIGHT_FRACTION = 0.85  # Neck at ~85% of height
    THIGH_HEIGHT_FRACTION = 0.45  # Upper thigh at ~45% of height

    def __init__(self, smpl_model_path: str, gender: str = "neutral"):
        """
        Initialize the measurement extractor.

        Args:
            smpl_model_path: Path to directory containing SMPL model files
                            (SMPL_MALE.pkl, SMPL_FEMALE.pkl, SMPL_NEUTRAL.pkl)
            gender: Gender of body model ('male', 'female', 'neutral')
        """
        self.smpl_model_path = Path(smpl_model_path)
        self.gender = gender.lower()
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        """Load SMPL model if not cached or if parameters changed."""
        global _smpl_model, _smpl_model_path, _smpl_gender

        if (
            _smpl_model is not None
            and _smpl_model_path == str(self.smpl_model_path)
            and _smpl_gender == self.gender
        ):
            return

        logger.info(f"Loading SMPL model ({self.gender}) from {self.smpl_model_path}")

        _smpl_model = smplx.create(
            str(self.smpl_model_path),
            model_type="smpl",
            gender=self.gender,
            use_pca=False,
            batch_size=1,
        )
        _smpl_model_path = str(self.smpl_model_path)
        _smpl_gender = self.gender

        logger.info("SMPL model loaded successfully")

    def extract(
        self,
        betas: np.ndarray,
        body_pose: Optional[np.ndarray] = None,
        global_orient: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """
        Extract body measurements from SMPL parameters.

        Measurements are extracted from the T-pose mesh (zeroed pose) to ensure
        consistency regardless of the input pose. Only shape (beta) parameters
        affect the measurements.

        Args:
            betas: Shape parameters (10,)
            body_pose: Joint rotations (69,) - ignored, T-pose used
            global_orient: Global rotation (3,) - ignored, T-pose used

        Returns:
            Dictionary with measurements in centimeters:
                - height_cm
                - chest_circumference_cm
                - waist_circumference_cm
                - hip_circumference_cm
                - inseam_cm
                - shoulder_width_cm
                - arm_length_cm
                - thigh_circumference_cm
                - neck_circumference_cm
        """
        # Generate T-pose mesh from shape parameters only
        vertices, joints = self._generate_tpose_mesh(betas)

        # SMPL units are meters, convert to cm
        vertices_cm = vertices * 100
        joints_cm = joints * 100

        # Extract measurements
        measurements = {
            "height_cm": self._measure_height(vertices_cm),
            "chest_circumference_cm": self._measure_chest_circumference(vertices_cm),
            "waist_circumference_cm": self._measure_waist_circumference(vertices_cm),
            "hip_circumference_cm": self._measure_hip_circumference(vertices_cm),
            "inseam_cm": self._measure_inseam(vertices_cm, joints_cm),
            "shoulder_width_cm": self._measure_shoulder_width(joints_cm),
            "arm_length_cm": self._measure_arm_length(joints_cm),
            "thigh_circumference_cm": self._measure_thigh_circumference(vertices_cm),
            "neck_circumference_cm": self._measure_neck_circumference(vertices_cm),
        }

        logger.debug(f"Extracted measurements: {measurements}")
        return measurements

    def _generate_tpose_mesh(self, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate mesh vertices and joints in T-pose.

        Args:
            betas: Shape parameters (10,)

        Returns:
            Tuple of (vertices (6890, 3), joints (24, 3))
        """
        betas_tensor = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)

        # Zero pose for T-pose
        body_pose = torch.zeros(1, 69)
        global_orient = torch.zeros(1, 3)

        with torch.no_grad():
            output = _smpl_model(
                betas=betas_tensor,
                body_pose=body_pose,
                global_orient=global_orient,
                return_verts=True,
            )

        vertices = output.vertices[0].numpy()
        joints = output.joints[0].numpy()

        return vertices, joints

    def _measure_height(self, vertices: np.ndarray) -> float:
        """Measure total height from floor to top of head."""
        return float(vertices[:, 1].max() - vertices[:, 1].min())

    def _measure_shoulder_width(self, joints: np.ndarray) -> float:
        """Measure shoulder width (left shoulder to right shoulder)."""
        left_shoulder = joints[JOINT_INDICES["left_shoulder"]]
        right_shoulder = joints[JOINT_INDICES["right_shoulder"]]
        return float(np.linalg.norm(left_shoulder - right_shoulder))

    def _measure_arm_length(self, joints: np.ndarray) -> float:
        """Measure arm length (shoulder to wrist)."""
        # Use average of left and right arm
        left_length = (
            np.linalg.norm(
                joints[JOINT_INDICES["left_shoulder"]] - joints[JOINT_INDICES["left_elbow"]]
            )
            + np.linalg.norm(
                joints[JOINT_INDICES["left_elbow"]] - joints[JOINT_INDICES["left_wrist"]]
            )
        )
        right_length = (
            np.linalg.norm(
                joints[JOINT_INDICES["right_shoulder"]] - joints[JOINT_INDICES["right_elbow"]]
            )
            + np.linalg.norm(
                joints[JOINT_INDICES["right_elbow"]] - joints[JOINT_INDICES["right_wrist"]]
            )
        )
        return float((left_length + right_length) / 2)

    def _measure_inseam(self, vertices: np.ndarray, joints: np.ndarray) -> float:
        """Measure inseam (crotch to ankle)."""
        crotch = vertices[VERTEX_INDICES["crotch"]]
        # Use average of left and right ankle
        left_ankle = joints[JOINT_INDICES["left_ankle"]]
        right_ankle = joints[JOINT_INDICES["right_ankle"]]
        avg_ankle_height = (left_ankle[1] + right_ankle[1]) / 2

        # Inseam is vertical distance from crotch to floor (ankle level)
        return float(crotch[1] - avg_ankle_height)

    def _measure_chest_circumference(self, vertices: np.ndarray) -> float:
        """Measure chest circumference at armpit level."""
        height = vertices[:, 1].max() - vertices[:, 1].min()
        floor = vertices[:, 1].min()
        slice_height = floor + height * self.CHEST_HEIGHT_FRACTION
        return self._measure_circumference_at_height(vertices, slice_height)

    def _measure_waist_circumference(self, vertices: np.ndarray) -> float:
        """Measure waist circumference at narrowest point."""
        height = vertices[:, 1].max() - vertices[:, 1].min()
        floor = vertices[:, 1].min()
        slice_height = floor + height * self.WAIST_HEIGHT_FRACTION
        return self._measure_circumference_at_height(vertices, slice_height)

    def _measure_hip_circumference(self, vertices: np.ndarray) -> float:
        """Measure hip circumference at widest point."""
        height = vertices[:, 1].max() - vertices[:, 1].min()
        floor = vertices[:, 1].min()
        slice_height = floor + height * self.HIP_HEIGHT_FRACTION
        return self._measure_circumference_at_height(vertices, slice_height)

    def _measure_neck_circumference(self, vertices: np.ndarray) -> float:
        """Measure neck circumference."""
        height = vertices[:, 1].max() - vertices[:, 1].min()
        floor = vertices[:, 1].min()
        slice_height = floor + height * self.NECK_HEIGHT_FRACTION
        return self._measure_circumference_at_height(vertices, slice_height)

    def _measure_thigh_circumference(self, vertices: np.ndarray) -> float:
        """Measure thigh circumference (upper leg)."""
        height = vertices[:, 1].max() - vertices[:, 1].min()
        floor = vertices[:, 1].min()
        slice_height = floor + height * self.THIGH_HEIGHT_FRACTION

        # For thigh, we need to measure just one leg (take the larger one)
        # Filter vertices to right half of body (positive X)
        right_leg_mask = vertices[:, 0] > 0
        right_leg_verts = vertices[right_leg_mask]

        if len(right_leg_verts) > 10:
            right_circ = self._measure_circumference_at_height(right_leg_verts, slice_height)
        else:
            right_circ = 0

        # Filter vertices to left half of body (negative X)
        left_leg_mask = vertices[:, 0] < 0
        left_leg_verts = vertices[left_leg_mask]

        if len(left_leg_verts) > 10:
            left_circ = self._measure_circumference_at_height(left_leg_verts, slice_height)
        else:
            left_circ = 0

        # Return average of both legs
        if right_circ > 0 and left_circ > 0:
            return float((right_circ + left_circ) / 2)
        return float(max(right_circ, left_circ))

    def _measure_circumference_at_height(
        self, vertices: np.ndarray, height: float, tolerance: float = 2.0
    ) -> float:
        """
        Measure circumference by slicing mesh at given height.

        Uses a convex hull approximation of the cross-section to calculate
        the perimeter.

        Args:
            vertices: Mesh vertices (N, 3)
            height: Y-coordinate to slice at
            tolerance: Thickness of slice in cm

        Returns:
            Circumference in cm
        """
        # Find vertices near the slice plane
        mask = np.abs(vertices[:, 1] - height) < tolerance
        slice_vertices = vertices[mask]

        if len(slice_vertices) < 3:
            logger.warning(f"Insufficient vertices for circumference at height {height}")
            return 0.0

        # Project to XZ plane and compute convex hull
        points_2d = slice_vertices[:, [0, 2]]  # X and Z coordinates

        try:
            hull = ConvexHull(points_2d)
            # Calculate perimeter of convex hull
            perimeter = 0.0
            for i in range(len(hull.vertices)):
                p1 = points_2d[hull.vertices[i]]
                p2 = points_2d[hull.vertices[(i + 1) % len(hull.vertices)]]
                perimeter += np.linalg.norm(p2 - p1)
            return float(perimeter)
        except Exception as e:
            logger.warning(f"Could not compute convex hull: {e}")
            return 0.0

    @property
    def faces(self) -> np.ndarray:
        """Get SMPL mesh faces."""
        return _smpl_model.faces
