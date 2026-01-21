"""
HMR2 inference wrapper.

This module provides a high-level interface to the 4D-Humans HMR2 model
for body reconstruction from single images.

The module uses a singleton pattern for model caching to ensure efficient
memory usage across multiple inference calls in a worker process.
"""

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from scipy.spatial.transform import Rotation

# Add vendor directory to path for 4D-Humans imports
_VENDOR_PATH = Path(__file__).parent.parent / "vendor" / "4D-Humans"
if str(_VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(_VENDOR_PATH))

# Module-level singletons for model caching
_hmr2_model: Optional[object] = None
_hmr2_cfg: Optional[object] = None
_detector: Optional[object] = None
_device: Optional[torch.device] = None


class HMR2Inference:
    """
    HMR 2.0 body reconstruction from single images.

    This class wraps the 4D-Humans HMR2 model and provides a simple interface
    for extracting SMPL body parameters from images.

    The model is loaded once per process and cached for subsequent calls.
    """

    def __init__(
        self,
        device: str = "cuda",
        detection_threshold: float = 0.5,
    ):
        """
        Initialize the HMR2 inference wrapper.

        Args:
            device: Device to run inference on ('cuda' or 'cpu')
            detection_threshold: Confidence threshold for person detection
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.detection_threshold = detection_threshold
        self._ensure_models_loaded()

    def _ensure_models_loaded(self) -> None:
        """Load models if not already cached."""
        global _hmr2_model, _hmr2_cfg, _detector, _device

        if _hmr2_model is not None and _device == self.device:
            return

        logger.info(f"Loading HMR2 model on {self.device}...")

        # Import 4D-Humans components
        from hmr2.configs import CACHE_DIR_4DHUMANS
        from hmr2.models import download_models, load_hmr2

        # Download models if needed (auto-downloads to ~/.cache/4DHumans/)
        download_models(CACHE_DIR_4DHUMANS)

        # Patch MeshRenderer to skip pyrender initialization (not needed for inference)
        # This avoids OSMesa/OpenGL issues in headless Docker containers
        import hmr2.utils.mesh_renderer as mesh_renderer_module

        class DummyMeshRenderer:
            """No-op renderer for inference-only mode."""
            def __init__(self, *args, **kwargs):
                pass

        mesh_renderer_module.MeshRenderer = DummyMeshRenderer

        # Load HMR2 model
        _hmr2_model, _hmr2_cfg = load_hmr2()
        _hmr2_model = _hmr2_model.to(self.device)
        _hmr2_model.eval()

        # Load person detector (ViTDet)
        _detector = self._load_detector()
        _device = self.device

        logger.info("HMR2 model loaded successfully")

    def _load_detector(self) -> object:
        """Load the ViTDet person detector."""
        from detectron2.config import LazyConfig

        import hmr2
        from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy

        cfg_path = Path(hmr2.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )

        # Set detection threshold
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25

        return DefaultPredictor_Lazy(detectron2_cfg)

    def predict(self, image: Image.Image, gender: str = "neutral") -> dict:
        """
        Run body reconstruction on an image.

        Args:
            image: PIL Image to process
            gender: Gender hint ('male', 'female', 'neutral'). Note: HMR2
                   uses a neutral SMPL model, so this is informational only.

        Returns:
            Dictionary containing:
                - betas: Shape parameters (10,)
                - body_pose: Joint rotations as axis-angle (23*3=69,)
                - global_orient: Global rotation as axis-angle (3,)
                - vertices: Mesh vertices (6890, 3)
                - joints: Joint positions (44, 3)

        Raises:
            ValueError: If no person is detected in the image
        """
        from hmr2.datasets.vitdet_dataset import ViTDetDataset
        from hmr2.utils import recursive_to

        # Convert PIL to OpenCV format (BGR)
        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect people in image
        det_out = _detector(img_cv2)
        det_instances = det_out["instances"]

        # Filter to person class (class 0) with sufficient confidence
        valid_idx = (det_instances.pred_classes == 0) & (
            det_instances.scores > self.detection_threshold
        )
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        if len(boxes) == 0:
            raise ValueError("No person detected in image")

        # If multiple people detected, use the one with largest bounding box
        if len(boxes) > 1:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_idx = np.argmax(areas)
            boxes = boxes[largest_idx : largest_idx + 1]
            logger.info(f"Multiple people detected, using largest (area={areas[largest_idx]:.0f})")

        # Prepare input for HMR2
        dataset = ViTDetDataset(_hmr2_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )

        # Run inference
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                output = _hmr2_model(batch)

        # Extract SMPL parameters
        pred_smpl_params = output["pred_smpl_params"]

        # Convert rotation matrices to axis-angle representation
        global_orient_rotmat = pred_smpl_params["global_orient"][0].cpu().numpy()  # (1, 3, 3)
        body_pose_rotmat = pred_smpl_params["body_pose"][0].cpu().numpy()  # (23, 3, 3)

        global_orient_aa = self._rotmat_to_axis_angle(global_orient_rotmat.reshape(-1, 3, 3))
        body_pose_aa = self._rotmat_to_axis_angle(body_pose_rotmat)

        return {
            "betas": pred_smpl_params["betas"][0].cpu().numpy(),  # (10,)
            "body_pose": body_pose_aa.flatten(),  # (69,) = 23 joints * 3
            "global_orient": global_orient_aa.flatten(),  # (3,)
            "vertices": output["pred_vertices"][0].cpu().numpy(),  # (6890, 3)
            "joints": output["pred_keypoints_3d"][0].cpu().numpy(),  # (44, 3)
        }

    @staticmethod
    def _rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrices to axis-angle representation.

        Args:
            rotmat: Rotation matrices of shape (N, 3, 3)

        Returns:
            Axis-angle vectors of shape (N, 3)
        """
        n = rotmat.shape[0]
        axis_angle = np.zeros((n, 3))
        for i in range(n):
            r = Rotation.from_matrix(rotmat[i])
            axis_angle[i] = r.as_rotvec()
        return axis_angle

    @property
    def smpl_faces(self) -> np.ndarray:
        """Get SMPL mesh faces for OBJ export."""
        return _hmr2_model.smpl.faces
