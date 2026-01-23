"""
Garment extraction service.

Extracts clothing items from photos using computer vision:
- Semantic segmentation to isolate garment
- Garment type classification
- Key point detection (collar, sleeves, etc.)
- Texture extraction
"""

import io
from typing import Any

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from app.config import get_settings


class GarmentExtractionService:
    """Service for extracting garments from photos."""

    def __init__(self):
        """
        Initialize the garment extraction service.
        
        Uses computer vision techniques for segmentation and key point detection.
        Can be extended with ML models by replacing method implementations.
        """
        self.settings = get_settings()

    def extract_garment(
        self,
        image: Image.Image,
        garment_type_hint: str | None = None,
    ) -> dict[str, Any]:
        """
        Extract garment from photo.

        Args:
            image: PIL Image of clothing item
            garment_type_hint: Optional hint for garment type (shirt, pants, etc.)

        Returns:
            Dictionary with:
                - segmented_mask: Binary mask of garment
                - garment_type: Detected/classified garment type
                - key_points: Dictionary of key points (collar, sleeves, etc.)
                - texture: Extracted texture image
        """
        logger.info("Extracting garment from image...")

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Extract garment using computer vision techniques
        segmented_mask = self._segment_garment(image)
        garment_type = self._classify_garment_type(image, garment_type_hint)
        key_points = self._detect_key_points(image, garment_type)
        texture = self._extract_texture(image, segmented_mask)

        logger.info(f"Garment extracted: type={garment_type}")

        return {
            "segmented_mask": segmented_mask,
            "garment_type": garment_type,
            "key_points": key_points,
            "texture": texture,
        }

    def _segment_garment(self, image: Image.Image) -> np.ndarray:
        """
        Segment garment from background using GrabCut algorithm.

        Uses OpenCV's GrabCut for foreground/background separation.
        This provides good results for most garment photos.

        Args:
            image: PIL Image

        Returns:
            Binary mask as numpy array (0 = background, 255 = garment)
        """

        # Convert PIL to numpy for OpenCV processing
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        # Use GrabCut algorithm for foreground/background segmentation
        height, width = img_bgr.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        
        # Define rectangle for GrabCut (assume garment is in center 60% of image)
        rect = (
            int(width * 0.2),
            int(height * 0.1),
            int(width * 0.6),
            int(height * 0.8),
        )
        
        # Initialize background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Run GrabCut (iterations=5 for speed, increase for better quality)
            cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create binary mask (0 and 2 = background, 1 and 3 = foreground)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            
            return mask2
        except Exception as e:
            logger.warning(f"GrabCut segmentation failed: {e}. Using fallback method.")
            # Fallback: simple center region
            mask = np.zeros((height, width), dtype=np.uint8)
            center_x, center_y = width // 2, height // 2
            mask[
                center_y - height // 4 : center_y + height // 4,
                center_x - width // 4 : center_x + width // 4,
            ] = 255
            return mask

    def _classify_garment_type(
        self, image: Image.Image, hint: str | None
    ) -> str:
        """
        Classify garment type.

        Uses user-provided hint if available, otherwise analyzes image
        characteristics (aspect ratio, shape) to infer garment type.

        Args:
            image: PIL Image
            hint: Optional user-provided garment type hint

        Returns:
            Garment type string (shirt, pants, etc.)
        """
        if hint:
            return hint.lower()
        
        # Analyze image characteristics to infer garment type
        width, height = image.size
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Heuristics based on aspect ratio
        if aspect_ratio > 1.2:
            # Wide image - likely pants or horizontal garment
            return "pants"
        elif aspect_ratio < 0.8:
            # Tall image - likely dress or long garment
            return "dress"
        else:
            # Square-ish - likely shirt or top
            return "shirt"

    def _detect_key_points(
        self, image: Image.Image, garment_type: str
    ) -> dict[str, list[float]]:
        """
        Detect key points on garment using computer vision.

        Uses edge detection and contour analysis to identify
        key anatomical points (collar, sleeves, hem, etc.).

        Args:
            image: PIL Image
            garment_type: Type of garment

        Returns:
            Dictionary mapping key point names to normalized [x, y] coordinates (0-1 range)
        """

        width, height = image.size
        img_array = np.array(image)
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        key_points = {}

        if garment_type in ["shirt", "t-shirt", "dress"]:
            # Detect collar (top center, typically has horizontal edge)
            # Find horizontal edges in top 20% of image
            top_region = gray[: int(height * 0.2), :]
            edges = cv2.Canny(top_region, 50, 150)
            horizontal_edges = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=50, minLineLength=width // 4, maxLineGap=10
            )
            
            if horizontal_edges is not None and len(horizontal_edges) > 0:
                # Find center of horizontal line near top
                line = horizontal_edges[0][0]
                collar_x = (line[0] + line[2]) / 2
                collar_y = line[1]
            else:
                collar_x = width * 0.5
                collar_y = height * 0.1
            
            key_points["collar"] = [float(collar_x), float(collar_y)]
            
            # Detect sleeves (left and right edges in upper region)
            left_region = gray[:, : int(width * 0.3)]
            right_region = gray[:, int(width * 0.7) :]
            
            # Find vertical edges (sleeve boundaries)
            left_edges = cv2.Canny(left_region, 50, 150)
            right_edges = cv2.Canny(right_region, 50, 150)
            
            # Find strongest vertical line in left region
            left_y = int(height * 0.3)
            if np.sum(left_edges[left_y - 10 : left_y + 10, :]) > 0:
                left_x = np.argmax(left_edges[left_y, :])
            else:
                left_x = width * 0.2
            
            # Find strongest vertical line in right region
            if np.sum(right_edges[left_y - 10 : left_y + 10, :]) > 0:
                right_x = np.argmax(right_edges[left_y, :]) + int(width * 0.7)
            else:
                right_x = width * 0.8
            
            key_points["left_sleeve"] = [float(left_x), float(left_y)]
            key_points["right_sleeve"] = [float(right_x), float(left_y)]
            
            # Hem (bottom center)
            key_points["hem"] = [float(width * 0.5), float(height * 0.9)]

        elif garment_type in ["pants", "shorts"]:
            # Waist (top center)
            key_points["waist"] = [float(width * 0.5), float(height * 0.1)]
            
            # Leg openings (bottom left and right)
            bottom_region = gray[int(height * 0.8) :, :]
            if np.sum(bottom_region) > 0:
                # Find two separate regions (legs)
                _, binary = cv2.threshold(bottom_region, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) >= 2:
                    # Sort by x position
                    centers = [cv2.moments(c) for c in contours]
                    centers = [(int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])) for m in centers if m["m00"] > 0]
                    centers.sort(key=lambda x: x[0])
                    
                    if len(centers) >= 2:
                        key_points["left_leg"] = [float(centers[0][0]), float(height * 0.9)]
                        key_points["right_leg"] = [float(centers[1][0]), float(height * 0.9)]
                    else:
                        key_points["left_leg"] = [float(width * 0.3), float(height * 0.9)]
                        key_points["right_leg"] = [float(width * 0.7), float(height * 0.9)]
                else:
                    key_points["left_leg"] = [float(width * 0.3), float(height * 0.9)]
                    key_points["right_leg"] = [float(width * 0.7), float(height * 0.9)]
            else:
                key_points["left_leg"] = [float(width * 0.3), float(height * 0.9)]
                key_points["right_leg"] = [float(width * 0.7), float(height * 0.9)]

        elif garment_type in ["shoes", "boots", "socks"]:
            # Toe (front center)
            key_points["toe"] = [float(width * 0.5), float(height * 0.8)]
            # Heel (back center)
            key_points["heel"] = [float(width * 0.5), float(height * 0.2)]

        # Convert to relative coordinates (0-1 range) for consistency
        normalized_key_points = {}
        for name, point in key_points.items():
            normalized_key_points[name] = [
                point[0] / width,
                point[1] / height,
            ]

        return normalized_key_points

    def _extract_texture(
        self, image: Image.Image, mask: np.ndarray
    ) -> Image.Image:
        """
        Extract texture from segmented garment.

        Args:
            image: Original image
            mask: Binary mask of garment

        Returns:
            Texture image (cropped and masked)
        """
        # Apply mask to image
        mask_pil = Image.fromarray(mask).convert("L")
        texture = image.copy()
        texture.putalpha(mask_pil)
        return texture
