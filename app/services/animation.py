"""
Animation service for generating animated GLB files from SMPL meshes.

Creates multiple poses that simulate a person trying on clothes:
- T-pose (arms out)
- Arms up (putting on shirt)
- One arm up
- Hands on hips
- Turning slightly
"""

import base64
import numpy as np
import torch
from loguru import logger
from pygltflib import (
    GLTF2,
    Accessor,
    Animation,
    AnimationChannel,
    AnimationSampler,
    Asset,
    Buffer,
    BufferView,
    Material,
    Mesh,
    Node,
    Primitive,
    Scene,
)

from app.config import get_settings


class AnimationService:
    """Service for generating animated GLB files from SMPL parameters."""

    def __init__(self, smpl_model_path: str, gender: str):
        """
        Initialize the animation service.

        Args:
            smpl_model_path: Path to SMPL model files
            gender: Gender ('male', 'female', 'neutral')
        """
        self.smpl_model_path = smpl_model_path
        self.gender = gender
        self._smpl_model = None

    def _get_smpl_model(self):
        """Lazy load SMPL model."""
        if self._smpl_model is None:
            import smplx

            logger.info(f"Loading SMPL model ({self.gender}) from {self.smpl_model_path}")
            self._smpl_model = smplx.create(
                str(self.smpl_model_path),
                model_type="smpl",
                gender=self.gender,
                use_pca=False,
                batch_size=1,
            )
        return self._smpl_model

    def generate_poses(self, betas: np.ndarray, base_body_pose: np.ndarray, base_global_orient: np.ndarray) -> list[dict]:
        """
        Generate multiple poses for animation.

        Args:
            betas: Shape parameters (10,)
            base_body_pose: Base body pose (69,) - from original image
            base_global_orient: Base global orientation (3,)

        Returns:
            List of pose dictionaries with 'name', 'body_pose', 'global_orient', 'vertices', 'faces'
        """
        smpl_model = self._get_smpl_model()
        betas_tensor = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
        base_body_pose_tensor = torch.tensor(base_body_pose, dtype=torch.float32).reshape(1, 23, 3)
        base_global_orient_tensor = torch.tensor(base_global_orient, dtype=torch.float32).reshape(1, 1, 3)

        poses = []

        # 1. T-pose (arms out horizontally)
        t_pose_body = torch.zeros(1, 23, 3)
        t_pose_global = torch.zeros(1, 1, 3)
        with torch.no_grad():
            output = smpl_model(
                betas=betas_tensor,
                body_pose=t_pose_body,
                global_orient=t_pose_global,
                return_verts=True,
            )
        poses.append({
            "name": "T-pose",
            "body_pose": t_pose_body[0].numpy().flatten(),
            "global_orient": t_pose_global[0].numpy().flatten(),
            "vertices": output.vertices[0].numpy(),
            "faces": smpl_model.faces,
        })

        # 2. Arms up (like putting on a shirt)
        arms_up_body = torch.zeros(1, 23, 3)
        # Left shoulder: raise arm up (rotation around x-axis, ~90 degrees)
        arms_up_body[0, 16] = torch.tensor([np.radians(90), 0, 0])  # Left shoulder
        # Right shoulder: raise arm up
        arms_up_body[0, 17] = torch.tensor([np.radians(90), 0, 0])  # Right shoulder
        # Elbows: slight bend
        arms_up_body[0, 18] = torch.tensor([np.radians(-30), 0, 0])  # Left elbow
        arms_up_body[0, 19] = torch.tensor([np.radians(-30), 0, 0])  # Right elbow

        with torch.no_grad():
            output = smpl_model(
                betas=betas_tensor,
                body_pose=arms_up_body,
                global_orient=t_pose_global,
                return_verts=True,
            )
        poses.append({
            "name": "Arms Up",
            "body_pose": arms_up_body[0].numpy().flatten(),
            "global_orient": t_pose_global[0].numpy().flatten(),
            "vertices": output.vertices[0].numpy(),
            "faces": smpl_model.faces,
        })

        # 3. One arm up (left arm)
        one_arm_up_body = torch.zeros(1, 23, 3)
        one_arm_up_body[0, 16] = torch.tensor([np.radians(90), 0, 0])  # Left shoulder
        one_arm_up_body[0, 18] = torch.tensor([np.radians(-30), 0, 0])  # Left elbow

        with torch.no_grad():
            output = smpl_model(
                betas=betas_tensor,
                body_pose=one_arm_up_body,
                global_orient=t_pose_global,
                return_verts=True,
            )
        poses.append({
            "name": "Left Arm Up",
            "body_pose": one_arm_up_body[0].numpy().flatten(),
            "global_orient": t_pose_global[0].numpy().flatten(),
            "vertices": output.vertices[0].numpy(),
            "faces": smpl_model.faces,
        })

        # 4. Hands on hips
        hands_hips_body = torch.zeros(1, 23, 3)
        # Shoulders: arms out to sides, then down
        hands_hips_body[0, 16] = torch.tensor([np.radians(30), np.radians(45), 0])  # Left shoulder
        hands_hips_body[0, 17] = torch.tensor([np.radians(30), np.radians(-45), 0])  # Right shoulder
        # Elbows: bend back
        hands_hips_body[0, 18] = torch.tensor([np.radians(-90), 0, 0])  # Left elbow
        hands_hips_body[0, 19] = torch.tensor([np.radians(-90), 0, 0])  # Right elbow

        with torch.no_grad():
            output = smpl_model(
                betas=betas_tensor,
                body_pose=hands_hips_body,
                global_orient=t_pose_global,
                return_verts=True,
            )
        poses.append({
            "name": "Hands on Hips",
            "body_pose": hands_hips_body[0].numpy().flatten(),
            "global_orient": t_pose_global[0].numpy().flatten(),
            "vertices": output.vertices[0].numpy(),
            "faces": smpl_model.faces,
        })

        # 5. Original pose from image
        with torch.no_grad():
            output = smpl_model(
                betas=betas_tensor,
                body_pose=base_body_pose_tensor,
                global_orient=base_global_orient_tensor,
                return_verts=True,
            )
        poses.append({
            "name": "Original",
            "body_pose": base_body_pose,
            "global_orient": base_global_orient,
            "vertices": output.vertices[0].numpy(),
            "faces": smpl_model.faces,
        })

        # 6. Turned slightly (rotate body)
        turn_global = torch.tensor([[[0, np.radians(30), 0]]], dtype=torch.float32)  # Rotate 30 degrees around Y
        with torch.no_grad():
            output = smpl_model(
                betas=betas_tensor,
                body_pose=base_body_pose_tensor,
                global_orient=turn_global,
                return_verts=True,
            )
        poses.append({
            "name": "Turned",
            "body_pose": base_body_pose,
            "global_orient": turn_global[0].numpy().flatten(),
            "vertices": output.vertices[0].numpy(),
            "faces": smpl_model.faces,
        })

        logger.info(f"Generated {len(poses)} poses for animation")
        return poses

    def create_animated_glb(self, poses: list[dict], output_path: str) -> bytes:
        """
        Create an animated GLB file from multiple poses.

        Uses morph targets (shape keys) to animate between poses.

        Args:
            poses: List of pose dictionaries with 'vertices' and 'faces'
            output_path: Path to save the GLB file (or None to return bytes)

        Returns:
            GLB file data as bytes
        """
        import tempfile
        import os

        if not poses:
            raise ValueError("At least one pose is required")

        # Use first pose as base mesh
        base_vertices = poses[0]["vertices"]
        base_faces = poses[0]["faces"]

        # Convert to float32 and ensure correct shape
        base_vertices = base_vertices.astype(np.float32)
        base_faces = base_faces.astype(np.uint32)

        # Flatten vertices and faces for binary data
        vertices_flat = base_vertices.flatten()
        faces_flat = base_faces.flatten()

        # Calculate sizes
        vertex_buffer_size = len(vertices_flat) * 4  # float32 = 4 bytes
        index_buffer_size = len(faces_flat) * 4  # uint32 = 4 bytes

        # Create morph target data (differences from base)
        morph_targets_data = []
        for pose in poses[1:]:
            pose_vertices = pose["vertices"].astype(np.float32)
            # Calculate difference from base (morph target delta)
            diff = pose_vertices - base_vertices
            morph_targets_data.append(diff.flatten().astype(np.float32))

        # Total buffer size
        total_buffer_size = vertex_buffer_size + index_buffer_size
        if morph_targets_data:
            total_buffer_size += sum(len(morph) * 4 for morph in morph_targets_data)

        # Create binary data
        binary_data = bytearray()
        
        # Base vertices
        binary_data.extend(vertices_flat.tobytes())
        vertex_buffer_view_start = 0
        vertex_buffer_view_length = vertex_buffer_size

        # Base indices
        index_buffer_view_start = len(binary_data)
        binary_data.extend(faces_flat.tobytes())
        index_buffer_view_length = index_buffer_size

        # Morph targets
        morph_buffer_views = []
        for morph_data in morph_targets_data:
            morph_start = len(binary_data)
            binary_data.extend(morph_data.tobytes())
            morph_buffer_views.append({
                "start": morph_start,
                "length": len(morph_data) * 4,
            })

        # Create GLTF structure
        gltf = GLTF2(
            asset=Asset(version="2.0", generator="Perfit Avatar Service"),
            scene=0,
            scenes=[Scene(nodes=[0])],
            nodes=[Node(mesh=0)],
            meshes=[
                Mesh(
                    primitives=[
                        Primitive(
                            attributes={"POSITION": 0},
                            indices=1,
                            material=0,
                            targets=[
                                {
                                    "POSITION": i + 2
                                }
                                for i in range(len(morph_targets_data))
                            ],
                        )
                    ],
                    weights=[0.0] * len(morph_targets_data),  # Start with all morphs at 0
                )
            ],
            accessors=[
                # Vertex positions
                Accessor(
                    bufferView=0,
                    componentType=5126,  # FLOAT
                    count=len(base_vertices),
                    type="VEC3",
                    min=base_vertices.min(axis=0).tolist(),
                    max=base_vertices.max(axis=0).tolist(),
                ),
                # Indices
                Accessor(
                    bufferView=1,
                    componentType=5125,  # UNSIGNED_INT
                    count=len(faces_flat),
                    type="SCALAR",
                ),
                # Morph targets (deltas)
                *[
                    Accessor(
                        bufferView=i + 2,
                        componentType=5126,  # FLOAT
                        count=len(base_vertices),
                        type="VEC3",
                        min=(morph_targets_data[i].reshape(-1, 3).min(axis=0)).tolist(),
                        max=(morph_targets_data[i].reshape(-1, 3).max(axis=0)).tolist(),
                    )
                    for i in range(len(morph_targets_data))
                ],
            ],
            bufferViews=[
                # Base vertices
                BufferView(
                    buffer=0,
                    byteOffset=vertex_buffer_view_start,
                    byteLength=vertex_buffer_view_length,
                ),
                # Indices
                BufferView(
                    buffer=0,
                    byteOffset=index_buffer_view_start,
                    byteLength=index_buffer_view_length,
                ),
                # Morph targets
                *[
                    BufferView(
                        buffer=0,
                        byteOffset=morph["start"],
                        byteLength=morph["length"],
                    )
                    for morph in morph_buffer_views
                ],
            ],
            buffers=[
                Buffer(
                    byteLength=total_buffer_size,
                    uri=None,  # Embedded in GLB
                )
            ],
            materials=[
                Material(
                    pbrMetallicRoughness={
                        "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.5,
                    }
                )
            ],
        )

        # Add animation (morph target weights)
        if len(poses) > 1:
            # Create keyframe animation that cycles through poses
            duration = len(poses) * 2.0  # 2 seconds per pose
            num_frames = len(poses) * 20  # 20 frames per pose for smooth animation
            times = np.linspace(0, duration, num_frames).astype(np.float32)
            
            # Create weight array: [num_frames, num_morph_targets]
            weights = np.zeros((num_frames, len(morph_targets_data)), dtype=np.float32)

            for i, t in enumerate(times):
                pose_time = t % (duration / len(poses) * len(poses))
                pose_idx = int(pose_time / (duration / len(poses)))
                
                if pose_idx == 0:
                    # Base pose - all weights at 0
                    weights[i, :] = 0.0
                elif pose_idx <= len(morph_targets_data):
                    # Activate morph target for this pose
                    morph_idx = pose_idx - 1
                    # Fade in/out for smooth transitions
                    cycle_pos = (pose_time % (duration / len(poses))) / (duration / len(poses))
                    if cycle_pos < 0.2:
                        # Fade in
                        weights[i, morph_idx] = cycle_pos / 0.2
                    elif cycle_pos < 0.8:
                        # Hold
                        weights[i, morph_idx] = 1.0
                    else:
                        # Fade out
                        weights[i, morph_idx] = (1.0 - cycle_pos) / 0.2

            # Animation data
            time_buffer_size = len(times) * 4
            weight_buffer_size = len(weights.flatten()) * 4
            animation_buffer_size = time_buffer_size + weight_buffer_size

            animation_binary = bytearray()
            time_start = len(binary_data)
            animation_binary.extend(times.tobytes())
            weight_start = len(animation_binary)
            animation_binary.extend(weights.flatten().tobytes())

            # Update total buffer size
            total_buffer_size += animation_buffer_size
            binary_data.extend(animation_binary)

            # Add animation accessors and buffer views
            time_accessor_idx = len(gltf.accessors)
            weight_accessor_idx = time_accessor_idx + 1

            gltf.accessors.extend([
                Accessor(
                    bufferView=len(gltf.bufferViews),
                    componentType=5126,  # FLOAT
                    count=len(times),
                    type="SCALAR",
                    min=[float(times.min())],
                    max=[float(times.max())],
                ),
                Accessor(
                    bufferView=len(gltf.bufferViews) + 1,
                    componentType=5126,  # FLOAT
                    count=len(weights),
                    type="SCALAR",
                ),
            ])

            gltf.bufferViews.extend([
                BufferView(
                    buffer=0,
                    byteOffset=time_start,
                    byteLength=time_buffer_size,
                ),
                BufferView(
                    buffer=0,
                    byteOffset=time_start + weight_start,
                    byteLength=weight_buffer_size,
                ),
            ])

            # Create animation
            gltf.animations = [
                Animation(
                    samplers=[
                        AnimationSampler(
                            input=time_accessor_idx,
                            output=weight_accessor_idx,
                            interpolation="LINEAR",
                        )
                    ],
                    channels=[
                        AnimationChannel(
                            sampler=0,
                            target={
                                "node": 0,
                                "path": "weights",
                            },
                        )
                    ],
                )
            ]

        # Update buffer size
        buffer_data = bytes(binary_data)
        gltf.buffers[0].byteLength = len(buffer_data)
        
        # Set buffer data as data URI, then convert to GLB binary
        # pygltflib needs buffer data to be accessible when converting
        gltf.buffers[0].uri = f"data:application/octet-stream;base64,{base64.b64encode(buffer_data).decode()}"
        
        # Save to temporary GLTF file first, then convert to GLB
        if output_path:
            tmp_gltf = output_path.replace('.glb', '.gltf')
            tmp_glb = output_path
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gltf") as tmp:
                tmp_gltf = tmp.name
            tmp_glb = tmp_gltf.replace('.gltf', '.glb')
        
        try:
            # Save as GLTF with data URI buffer
            gltf.save(tmp_gltf)
            
            # Load and convert to GLB with embedded binary
            gltf_loaded = GLTF2.load(tmp_gltf)
            gltf_loaded.convert_buffers(glb=True)
            gltf_loaded.save_binary(tmp_glb)
            
            # Clean up temporary GLTF file
            if os.path.exists(tmp_gltf):
                os.unlink(tmp_gltf)
            
            # Read and return GLB data
            with open(tmp_glb, "rb") as f:
                glb_data = f.read()
            
            if not output_path:
                # Clean up temporary GLB file if we created it
                os.unlink(tmp_glb)
            
            logger.info(f"Created animated GLB with {len(poses)} poses")
            return glb_data
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(tmp_gltf):
                os.unlink(tmp_gltf)
            if not output_path and os.path.exists(tmp_glb):
                os.unlink(tmp_glb)
            raise
