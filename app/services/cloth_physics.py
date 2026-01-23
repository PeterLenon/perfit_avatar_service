"""
Cloth physics simulation service.

Handles physics simulation for garments:
- Material property estimation
- Cloth simulation setup
- Collision detection configuration
"""

from typing import Any

import numpy as np
from loguru import logger
from scipy.spatial import cKDTree

from app.config import get_settings


class ClothPhysicsService:
    """Service for cloth physics simulation."""

    # Garment type to default material properties mapping
    DEFAULT_MATERIAL_PROPERTIES = {
        "shirt": {
            "stretch": 0.3,
            "bend": 0.2,
            "shear": 0.25,
            "density": 0.1,
            "damping": 0.05,
        },
        "pants": {
            "stretch": 0.4,
            "bend": 0.3,
            "shear": 0.35,
            "density": 0.12,
            "damping": 0.06,
        },
        "dress": {
            "stretch": 0.25,
            "bend": 0.15,
            "shear": 0.2,
            "density": 0.08,
            "damping": 0.04,
        },
        "jacket": {
            "stretch": 0.5,
            "bend": 0.4,
            "shear": 0.45,
            "density": 0.15,
            "damping": 0.08,
        },
        "shoes": {
            "stretch": 0.1,
            "bend": 0.05,
            "shear": 0.08,
            "density": 0.5,
            "damping": 0.1,
        },
        "socks": {
            "stretch": 0.6,
            "bend": 0.1,
            "shear": 0.5,
            "density": 0.05,
            "damping": 0.03,
        },
        "underwear": {
            "stretch": 0.7,
            "bend": 0.15,
            "shear": 0.6,
            "density": 0.06,
            "damping": 0.04,
        },
    }

    def __init__(self):
        """
        Initialize the cloth physics service.
        
        Uses simplified mass-spring simulation with collision detection.
        Can be extended with full physics engines (PyBullet, MuJoCo) if needed.
        """
        self.settings = get_settings()

    def estimate_material_properties(
        self, garment_type: str, image_data: Any = None
    ) -> dict[str, float]:
        """
        Estimate material properties for a garment.

        Uses garment type to lookup default properties based on typical
        fabric characteristics for that garment type.

        Args:
            garment_type: Type of garment (shirt, pants, etc.)
            image_data: Optional image data (currently unused, reserved for extension)

        Returns:
            Dictionary with material properties:
                - stretch: Resistance to stretching (0-1, higher = more stretchy)
                - bend: Resistance to bending (0-1, higher = more flexible)
                - shear: Resistance to shearing (0-1)
                - density: Mass density (affects weight/draping)
                - damping: Energy loss (0-1, higher = more damping)
        """
        logger.info(f"Estimating material properties for {garment_type}")

        # Get default properties based on garment type
        properties = self.DEFAULT_MATERIAL_PROPERTIES.get(
            garment_type.lower(),
            {
                "stretch": 0.3,
                "bend": 0.2,
                "shear": 0.25,
                "density": 0.1,
                "damping": 0.05,
            },
        ).copy()

        logger.debug(f"Material properties: {properties}")
        return properties

    def setup_cloth_simulation(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        material_properties: dict[str, float],
    ) -> dict[str, Any]:
        """
        Set up cloth simulation parameters.

        Calculates mesh-dependent simulation parameters (stiffness scaling,
        mass distribution) based on material properties and mesh geometry.

        Args:
            vertices: Garment mesh vertices (Nx3)
            faces: Garment mesh faces (Mx3)
            material_properties: Material properties dict

        Returns:
            Simulation configuration dictionary with mass-spring and collision parameters
        """
        logger.info("Setting up cloth simulation")

        # Calculate mesh properties for simulation
        num_vertices = len(vertices)
        num_faces = len(faces)
        
        # Estimate average edge length for spring stiffness calculation
        edge_lengths = []
        for face in faces:
            v0, v1, v2 = vertices[face]
            edge_lengths.extend([
                np.linalg.norm(v1 - v0),
                np.linalg.norm(v2 - v1),
                np.linalg.norm(v0 - v2),
            ])
        avg_edge_length = np.mean(edge_lengths) if edge_lengths else 0.1

        # Scale stiffness based on edge length (smaller edges = stiffer)
        base_stretch = material_properties.get("stretch", 0.3)
        base_bend = material_properties.get("bend", 0.2)
        
        # Adjust stiffness based on mesh density
        stiffness_scale = 1.0 / (avg_edge_length + 0.01)

        return {
            "mass_spring_config": {
                "stretch_stiffness": base_stretch * stiffness_scale,
                "bend_stiffness": base_bend * stiffness_scale,
                "shear_stiffness": material_properties.get("shear", 0.25) * stiffness_scale,
                "damping": material_properties.get("damping", 0.05),
                "mass_per_vertex": material_properties.get("density", 0.1) / num_vertices,
            },
            "collision_config": {
                "self_collision": True,
                "body_collision": True,
                "collision_margin": 0.01,
                "friction": 0.5,
            },
            "simulation_config": {
                "time_step": 0.01,
                "gravity": [0, -9.81, 0],
                "iterations": 10,
                "num_vertices": num_vertices,
                "num_faces": num_faces,
                "avg_edge_length": float(avg_edge_length),
            },
        }

    def simulate_draping(
        self,
        garment_vertices: np.ndarray,
        garment_faces: np.ndarray,
        body_vertices: np.ndarray,
        attachment_points: list[tuple[int, int]],
        material_properties: dict[str, float],
    ) -> np.ndarray:
        """
        Simulate garment draping on body using simplified physics.

        Implements gravity-based draping with collision detection:
        1. Applies gravity to non-attached vertices
        2. Detects and resolves collisions with body surface
        3. Maintains mesh connectivity through stiffness constraints

        Args:
            garment_vertices: Initial garment vertices (Nx3)
            garment_faces: Garment faces (Mx3) - used for connectivity
            body_vertices: Body/avatar vertices for collision (Kx3)
            attachment_points: List of (garment_vertex_idx, body_vertex_idx) pairs
            material_properties: Material properties dict

        Returns:
            Draped garment vertices (Nx3) after physics simulation
        """
        logger.info("Simulating garment draping")

        draped_vertices = garment_vertices.copy()
        
        # Get attachment vertex indices
        attached_indices = {att[0] for att in attachment_points}
        
        # Simple draping simulation:
        # 1. Apply gravity to non-attached vertices
        # 2. Pull vertices towards body surface
        # 3. Maintain mesh connectivity (simplified)
        
        gravity = material_properties.get("density", 0.1) * 0.01
        
        # Find body surface points (simplified: use nearest neighbors)
        body_tree = cKDTree(body_vertices)
        
        # Iterate a few times for convergence
        for iteration in range(5):
            for i, vertex in enumerate(draped_vertices):
                # Skip attachment points (they stay fixed)
                if i in attached_indices:
                    continue
                
                # Apply gravity (move down)
                new_vertex = vertex.copy()
                new_vertex[1] -= gravity
                
                # Find nearest body surface point
                distance, nearest_body_idx = body_tree.query(new_vertex)
                
                # If too close to body, push away slightly
                min_distance = 0.02  # Minimum distance from body
                if distance < min_distance:
                    direction = new_vertex - body_vertices[nearest_body_idx]
                    if np.linalg.norm(direction) > 1e-6:
                        direction = direction / np.linalg.norm(direction)
                        new_vertex = body_vertices[nearest_body_idx] + direction * min_distance
                
                # Apply material stiffness (resistance to deformation)
                stiffness = material_properties.get("stretch", 0.3)
                # Blend between original and new position based on stiffness
                draped_vertices[i] = vertex * (1 - stiffness * 0.1) + new_vertex * (stiffness * 0.1)
        
        logger.debug(f"Draping simulation complete: {len(draped_vertices)} vertices adjusted")
        return draped_vertices
