import os
import tempfile
import cv2
import numpy as np
from temporalio import activity
from loguru import logger
from typing import Tuple


@activity.defn
async def save_user_image(image_data: bytes, user_id: str) -> str:
    """
    Save user image to temporary file for ECON processing.
    
    Args:
        image_data: Raw image bytes
        user_id: User identifier for unique file naming
    
    Returns:
        Path to saved image file
    """
    # Create temporary directory for this user
    temp_dir = os.path.join(tempfile.gettempdir(), "avatar_service", user_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Decode image from bytes
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise activity.ApplicationError("Failed to decode image", non_retryable=True)
    
    # Save image
    image_path = os.path.join(temp_dir, f"{user_id}_input.jpg")
    cv2.imwrite(image_path, img)
    
    logger.info(f"Saved user image to {image_path}")
    return image_path

