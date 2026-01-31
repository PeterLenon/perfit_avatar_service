"""
Animation service for generating animated GLB files from SMPL meshes.

Creates multiple poses that simulate a person trying on clothes:
- T-pose (arms out)
- Arms up (putting on shirt)
- One arm up
- Hands on hips
- Turning slightly

Supports textures for realistic rendering.
"""

import base64
import io
import os
import tempfile
from typing import Any

import numpy as np
import torch
from loguru import logger
from PIL import Image
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
    BufferFormat,
    Image as GLTFImage,
    Texture,
    Sampler,
)

from app.config import get_settings


class AnimationService:
    """Service for generating animated GLB files from SMPL parameters."""

    # Default skin tones (RGB, 0-1 range)
    SKIN_TONES = {
        "light": [0.96, 0.87, 0.80],
        "medium": [0.87, 0.72, 0.58],
        "tan": [0.78, 0.60, 0.45],
        "brown": [0.60, 0.42, 0.30],
        "dark": [0.40, 0.28, 0.20],
    }

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

    def generate_poses(
        self, betas: np.ndarray, base_body_pose: np.ndarray, base_global_orient: np.ndarray
    ) -> list[dict]:
        """
        Generate multiple poses for animation.

        Args:
            betas: Shape parameters (10,)
            base_body_pose: Base body pose (69,) - from original image
            base_global_orient: Base global orientation (3,)

        Returns:
            List of pose dictionaries with 'name', 'body_pose', 'global_orient',
            'vertices', 'faces'
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
        arms_up_body[0, 16] = torch.tensor([np.radians(90), 0, 0])
        arms_up_body[0, 17] = torch.tensor([np.radians(90), 0, 0])
        arms_up_body[0, 18] = torch.tensor([np.radians(-30), 0, 0])
        arms_up_body[0, 19] = torch.tensor([np.radians(-30), 0, 0])

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
        one_arm_up_body[0, 16] = torch.tensor([np.radians(90), 0, 0])
        one_arm_up_body[0, 18] = torch.tensor([np.radians(-30), 0, 0])

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
        hands_hips_body[0, 16] = torch.tensor([np.radians(30), np.radians(45), 0])
        hands_hips_body[0, 17] = torch.tensor([np.radians(30), np.radians(-45), 0])
        hands_hips_body[0, 18] = torch.tensor([np.radians(-90), 0, 0])
        hands_hips_body[0, 19] = torch.tensor([np.radians(-90), 0, 0])

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
        turn_global = torch.tensor([[[0, np.radians(30), 0]]], dtype=torch.float32)
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

    def create_animated_glb(
        self,
        poses: list[dict],
        output_path: str | None = None,
        body_color: list[float] | str | None = None,
        textures: list[dict[str, Any]] | None = None,
    ) -> bytes:
        """
        Create an animated GLB file from multiple poses.

        Uses morph targets (shape keys) to animate between poses.
        Supports textures for realistic rendering.

        Args:
            poses: List of pose dictionaries with 'vertices' and 'faces'
            output_path: Path to save the GLB file (or None to return bytes)
            body_color: Body color as [R, G, B] (0-1) or skin tone name
                       ('light', 'medium', 'tan', 'brown', 'dark')
            textures: List of texture dicts with 'image' (PIL Image or bytes),
                     'uvs' (Nx2 array), optional 'vertex_offset' for multi-mesh

        Returns:
            GLB file data as bytes
        """
        if not poses:
            raise ValueError("At least one pose is required")

        # Resolve body color
        if body_color is None:
            base_color = [0.87, 0.72, 0.58, 1.0]  # Default medium skin tone
        elif isinstance(body_color, str):
            rgb = self.SKIN_TONES.get(body_color, self.SKIN_TONES["medium"])
            base_color = [*rgb, 1.0]
        else:
            base_color = [*body_color[:3], 1.0]

        # Use first pose as base mesh
        base_vertices = poses[0]["vertices"].astype(np.float32)
        base_faces = poses[0]["faces"].astype(np.uint32)

        # Flatten for binary data
        vertices_flat = base_vertices.flatten()
        faces_flat = base_faces.flatten()

        # Buffer sizes
        vertex_buffer_size = len(vertices_flat) * 4
        index_buffer_size = len(faces_flat) * 4

        # Create morph target data
        morph_targets_data = []
        for pose in poses[1:]:
            pose_vertices = pose["vertices"].astype(np.float32)
            diff = pose_vertices - base_vertices
            morph_targets_data.append(diff.flatten().astype(np.float32))

        # Build binary buffer
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

        # Process textures if provided
        texture_data = []
        uv_buffer_views = []
        image_buffer_views = []
        has_textures = textures is not None and len(textures) > 0

        if has_textures:
            for tex_info in textures:
                # Get UV coordinates
                uvs = tex_info.get("uvs")
                if uvs is not None:
                    uvs = np.array(uvs, dtype=np.float32)
                    uv_start = len(binary_data)
                    binary_data.extend(uvs.flatten().tobytes())
                    uv_buffer_views.append({
                        "start": uv_start,
                        "length": len(uvs) * 2 * 4,  # Nx2 floats
                        "count": len(uvs),
                    })

                # Get texture image
                img = tex_info.get("image")
                if img is not None:
                    if isinstance(img, Image.Image):
                        # Convert PIL Image to PNG bytes
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format="PNG")
                        img_bytes = img_buffer.getvalue()
                    elif isinstance(img, bytes):
                        img_bytes = img
                    else:
                        continue

                    img_start = len(binary_data)
                    binary_data.extend(img_bytes)
                    image_buffer_views.append({
                        "start": img_start,
                        "length": len(img_bytes),
                    })
                    texture_data.append(tex_info)

        # Build GLTF structure
        buffer_views = [
            # Base vertices
            BufferView(
                buffer=0,
                byteOffset=vertex_buffer_view_start,
                byteLength=vertex_buffer_view_length,
                target=34962,  # ARRAY_BUFFER
            ),
            # Indices
            BufferView(
                buffer=0,
                byteOffset=index_buffer_view_start,
                byteLength=index_buffer_view_length,
                target=34963,  # ELEMENT_ARRAY_BUFFER
            ),
        ]

        # Add morph target buffer views
        for morph in morph_buffer_views:
            buffer_views.append(
                BufferView(
                    buffer=0,
                    byteOffset=morph["start"],
                    byteLength=morph["length"],
                    target=34962,
                )
            )

        # Add UV buffer views
        uv_accessor_start = 2 + len(morph_buffer_views)
        for uv_bv in uv_buffer_views:
            buffer_views.append(
                BufferView(
                    buffer=0,
                    byteOffset=uv_bv["start"],
                    byteLength=uv_bv["length"],
                    target=34962,
                )
            )

        # Add image buffer views (no target for images)
        img_bv_start = len(buffer_views)
        for img_bv in image_buffer_views:
            buffer_views.append(
                BufferView(
                    buffer=0,
                    byteOffset=img_bv["start"],
                    byteLength=img_bv["length"],
                )
            )

        # Build accessors
        accessors = [
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
        ]

        # Add morph target accessors
        for i, morph_data in enumerate(morph_targets_data):
            accessors.append(
                Accessor(
                    bufferView=i + 2,
                    componentType=5126,
                    count=len(base_vertices),
                    type="VEC3",
                    min=morph_data.reshape(-1, 3).min(axis=0).tolist(),
                    max=morph_data.reshape(-1, 3).max(axis=0).tolist(),
                )
            )

        # Add UV accessors
        for i, uv_bv in enumerate(uv_buffer_views):
            accessors.append(
                Accessor(
                    bufferView=uv_accessor_start + i,
                    componentType=5126,
                    count=uv_bv["count"],
                    type="VEC2",
                )
            )

        # Build images and textures for GLTF
        gltf_images = []
        gltf_textures = []
        gltf_samplers = []

        if has_textures and image_buffer_views:
            # Add a sampler (linear filtering with repeat)
            gltf_samplers.append(
                Sampler(
                    magFilter=9729,  # LINEAR
                    minFilter=9987,  # LINEAR_MIPMAP_LINEAR
                    wrapS=10497,  # REPEAT
                    wrapT=10497,  # REPEAT
                )
            )

            for i, img_bv in enumerate(image_buffer_views):
                gltf_images.append(
                    GLTFImage(
                        bufferView=img_bv_start + i,
                        mimeType="image/png",
                    )
                )
                gltf_textures.append(
                    Texture(
                        sampler=0,
                        source=i,
                    )
                )

        # Build materials
        materials = []
        if has_textures and gltf_textures:
            # Material with texture
            materials.append(
                Material(
                    pbrMetallicRoughness={
                        "baseColorTexture": {"index": 0},
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.7,
                    },
                    doubleSided=True,
                )
            )
        else:
            # Material with solid color
            materials.append(
                Material(
                    pbrMetallicRoughness={
                        "baseColorFactor": base_color,
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.5,
                    },
                    doubleSided=True,
                )
            )

        # Build primitive attributes
        primitive_attributes = {"POSITION": 0}
        if uv_buffer_views:
            # Add UV coordinates to primitive
            primitive_attributes["TEXCOORD_0"] = uv_accessor_start

        # Build mesh primitive
        primitive = Primitive(
            attributes=primitive_attributes,
            indices=1,
            material=0,
            targets=[{"POSITION": i + 2} for i in range(len(morph_targets_data))],
        )

        # Create GLTF object
        gltf = GLTF2(
            asset=Asset(version="2.0", generator="Perfit Avatar Service"),
            scene=0,
            scenes=[Scene(nodes=[0])],
            nodes=[Node(mesh=0)],
            meshes=[
                Mesh(
                    primitives=[primitive],
                    weights=[0.0] * len(morph_targets_data),
                )
            ],
            accessors=accessors,
            bufferViews=buffer_views,
            buffers=[Buffer(byteLength=len(binary_data))],
            materials=materials,
            images=gltf_images if gltf_images else None,
            textures=gltf_textures if gltf_textures else None,
            samplers=gltf_samplers if gltf_samplers else None,
        )

        # Add animation
        if len(poses) > 1:
            self._add_morph_animation(gltf, binary_data, poses, morph_targets_data)

        # Convert to GLB
        return self._save_as_glb(gltf, binary_data, output_path)

    def _add_morph_animation(
        self,
        gltf: GLTF2,
        binary_data: bytearray,
        poses: list[dict],
        morph_targets_data: list[np.ndarray],
    ) -> None:
        """Add morph target animation to GLTF."""
        duration = len(poses) * 2.0
        num_frames = len(poses) * 20
        times = np.linspace(0, duration, num_frames).astype(np.float32)

        weights = np.zeros((num_frames, len(morph_targets_data)), dtype=np.float32)

        for i, t in enumerate(times):
            pose_time = t % (duration / len(poses) * len(poses))
            pose_idx = int(pose_time / (duration / len(poses)))

            if pose_idx == 0:
                weights[i, :] = 0.0
            elif pose_idx <= len(morph_targets_data):
                morph_idx = pose_idx - 1
                cycle_pos = (pose_time % (duration / len(poses))) / (duration / len(poses))
                if cycle_pos < 0.2:
                    weights[i, morph_idx] = cycle_pos / 0.2
                elif cycle_pos < 0.8:
                    weights[i, morph_idx] = 1.0
                else:
                    weights[i, morph_idx] = (1.0 - cycle_pos) / 0.2

        # Add animation data to buffer
        time_start = len(binary_data)
        binary_data.extend(times.tobytes())
        weight_start = len(binary_data)
        binary_data.extend(weights.flatten().tobytes())

        time_buffer_size = len(times) * 4
        weight_buffer_size = len(weights.flatten()) * 4

        # Add buffer views
        time_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(
            BufferView(buffer=0, byteOffset=time_start, byteLength=time_buffer_size)
        )
        weight_bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(
            BufferView(buffer=0, byteOffset=weight_start, byteLength=weight_buffer_size)
        )

        # Add accessors
        time_accessor_idx = len(gltf.accessors)
        gltf.accessors.append(
            Accessor(
                bufferView=time_bv_idx,
                componentType=5126,
                count=len(times),
                type="SCALAR",
                min=[float(times.min())],
                max=[float(times.max())],
            )
        )
        weight_accessor_idx = len(gltf.accessors)
        gltf.accessors.append(
            Accessor(
                bufferView=weight_bv_idx,
                componentType=5126,
                count=len(weights.flatten()),
                type="SCALAR",
            )
        )

        # Add animation
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
                        target={"node": 0, "path": "weights"},
                    )
                ],
            )
        ]

    def _save_as_glb(
        self, gltf: GLTF2, binary_data: bytearray, output_path: str | None
    ) -> bytes:
        """Save GLTF as GLB binary format."""
        buffer_data = bytes(binary_data)
        gltf.buffers[0].byteLength = len(buffer_data)
        gltf.buffers[0].uri = f"data:application/octet-stream;base64,{base64.b64encode(buffer_data).decode()}"

        if output_path:
            tmp_gltf = output_path.replace(".glb", ".gltf")
            tmp_glb = output_path
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gltf") as tmp:
                tmp_gltf = tmp.name
            tmp_glb = tmp_gltf.replace(".gltf", ".glb")

        try:
            gltf.save(tmp_gltf)
            gltf_loaded = GLTF2.load(tmp_gltf)
            gltf_loaded.convert_buffers(buffer_format=BufferFormat.DATAURI)
            gltf_loaded.save_binary(tmp_glb)

            if os.path.exists(tmp_gltf):
                os.unlink(tmp_gltf)

            with open(tmp_glb, "rb") as f:
                glb_data = f.read()

            if not output_path:
                os.unlink(tmp_glb)

            logger.info(f"Created GLB file: {len(glb_data)} bytes")
            return glb_data

        except Exception as e:
            if os.path.exists(tmp_gltf):
                os.unlink(tmp_gltf)
            if not output_path and os.path.exists(tmp_glb):
                os.unlink(tmp_glb)
            raise
