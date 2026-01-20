"""
Image validation service.

Validates uploaded images before processing:
- Format and size checks
- Quality assessment (brightness, blur)
- Face detection (required for body extraction)
"""

import base64
import io
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger
from PIL import Image

from app.config import ImageValidationSettings
from app.models.schemas import ImageValidationResult


class ImageValidator:
    """
    Validates images for body extraction pipeline.

    Performs the following checks:
    1. Format validation (JPEG, PNG, WebP)
    2. Size validation (min/max dimensions, file size)
    3. Quality checks (brightness, contrast)
    4. Face detection (required for PIXIE body extraction)
    """

    def __init__(self, settings: ImageValidationSettings):
        self.settings = settings
        self._face_detector: Optional[mp.solutions.face_detection.FaceDetection] = None

    @property
    def face_detector(self):
        """Lazy initialization of MediaPipe face detector."""
        if self._face_detector is None:
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # 1 = full range model (better for various distances)
                min_detection_confidence=0.5,
            )
        return self._face_detector

    def validate_base64(self, image_base64: str) -> ImageValidationResult:
        """
        Validate a base64-encoded image.

        Args:
            image_base64: Base64-encoded image data (may include data URL prefix)

        Returns:
            ImageValidationResult with validation status and any errors/warnings
        """
        result = ImageValidationResult(is_valid=True)

        # Decode base64
        try:
            image_data = self._decode_base64(image_base64)
        except ValueError as e:
            result.is_valid = False
            result.errors.append(str(e))
            return result

        # Check file size
        file_size_mb = len(image_data) / (1024 * 1024)
        if file_size_mb > self.settings.max_file_size_mb:
            result.is_valid = False
            result.errors.append(
                f"File size ({file_size_mb:.1f}MB) exceeds maximum ({self.settings.max_file_size_mb}MB)"
            )
            return result

        # Load image
        try:
            pil_image = Image.open(io.BytesIO(image_data))
            cv_image = self._pil_to_cv2(pil_image)
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Failed to load image: {str(e)}")
            return result

        # Check format
        image_format = pil_image.format.lower() if pil_image.format else "unknown"
        if image_format not in self.settings.allowed_formats:
            result.is_valid = False
            result.errors.append(
                f"Image format '{image_format}' not allowed. "
                f"Allowed formats: {self.settings.allowed_formats}"
            )
            return result

        # Check dimensions
        result.image_width = pil_image.width
        result.image_height = pil_image.height

        if pil_image.width < self.settings.min_width:
            result.is_valid = False
            result.errors.append(
                f"Image width ({pil_image.width}px) below minimum ({self.settings.min_width}px)"
            )

        if pil_image.height < self.settings.min_height:
            result.is_valid = False
            result.errors.append(
                f"Image height ({pil_image.height}px) below minimum ({self.settings.min_height}px)"
            )

        if pil_image.width > self.settings.max_width:
            result.is_valid = False
            result.errors.append(
                f"Image width ({pil_image.width}px) exceeds maximum ({self.settings.max_width}px)"
            )

        if pil_image.height > self.settings.max_height:
            result.is_valid = False
            result.errors.append(
                f"Image height ({pil_image.height}px) exceeds maximum ({self.settings.max_height}px)"
            )

        if not result.is_valid:
            return result

        # Check brightness
        brightness = self._calculate_brightness(cv_image)
        if brightness < self.settings.min_brightness:
            result.warnings.append(
                f"Image may be too dark (brightness: {brightness:.0f}). "
                "Consider using a better-lit photo."
            )
        elif brightness > self.settings.max_brightness:
            result.warnings.append(
                f"Image may be overexposed (brightness: {brightness:.0f}). "
                "Consider using a less bright photo."
            )

        # Detect face
        face_detection_result = self._detect_face(cv_image)
        result.face_detected = face_detection_result["detected"]
        result.face_confidence = face_detection_result.get("confidence")

        if not result.face_detected:
            result.is_valid = False
            result.errors.append(
                "No face detected in image. "
                "Please upload a photo showing your full body with face visible."
            )

        return result

    def _decode_base64(self, image_base64: str) -> bytes:
        """
        Decode base64 image data.

        Handles both raw base64 and data URL format (data:image/jpeg;base64,...).
        """
        # Strip data URL prefix if present
        if "," in image_base64:
            # Format: data:image/jpeg;base64,/9j/4AAQ...
            header, image_base64 = image_base64.split(",", 1)
            logger.debug(f"Stripped data URL header: {header}")

        # Remove whitespace
        image_base64 = image_base64.strip()

        try:
            return base64.b64decode(image_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL image to OpenCV format (BGR)."""
        # Convert to RGB if necessary
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert to numpy array
        rgb_array = np.array(pil_image)

        # Convert RGB to BGR for OpenCV
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        return bgr_array

    def _calculate_brightness(self, cv_image: np.ndarray) -> float:
        """
        Calculate average brightness of an image.

        Returns value between 0 (black) and 255 (white).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Calculate mean brightness
        return float(np.mean(gray))

    def _detect_face(self, cv_image: np.ndarray) -> dict:
        """
        Detect face in image using MediaPipe.

        Returns dict with:
        - detected: bool
        - confidence: float (0-1) if detected
        - bbox: dict with x, y, width, height if detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Run face detection
        results = self.face_detector.process(rgb_image)

        if not results.detections:
            return {"detected": False}

        # Get the detection with highest confidence
        best_detection = max(results.detections, key=lambda d: d.score[0])
        confidence = best_detection.score[0]

        # Get bounding box
        bbox = best_detection.location_data.relative_bounding_box
        h, w = cv_image.shape[:2]

        return {
            "detected": True,
            "confidence": float(confidence),
            "bbox": {
                "x": int(bbox.xmin * w),
                "y": int(bbox.ymin * h),
                "width": int(bbox.width * w),
                "height": int(bbox.height * h),
            },
        }

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if self._face_detector is not None:
            self._face_detector.close()
