"""
Skin tone extraction service.

Extracts skin color from source images by sampling pixels in face/body regions.
Uses SMPL mesh projection to identify skin areas.
"""

import numpy as np
from loguru import logger
from PIL import Image


# SMPL vertex indices for face region (approximate)
# These vertices correspond to the face/cheek area which is reliable for skin sampling
SMPL_FACE_VERTICES = [
    # Forehead
    336, 337, 338, 339, 340,
    # Cheeks
    1850, 1851, 1852, 1653, 1654,
    # Chin/jaw
    3052, 3053, 3054, 3055,
    # Nose bridge
    1733, 1734, 1735,
]

# SMPL vertex indices for neck/upper chest (backup sampling area)
SMPL_NECK_VERTICES = [
    3068, 3069, 3070,  # Front neck
    828, 829, 830,  # Back neck
]


def extract_skin_tone_from_mesh_projection(
    image: Image.Image,
    vertices: np.ndarray,
    camera_intrinsics: dict | None = None,
) -> list[float]:
    """
    Extract skin tone by projecting SMPL mesh onto image and sampling face region.

    Args:
        image: Source PIL image
        vertices: SMPL mesh vertices (6890, 3)
        camera_intrinsics: Optional camera parameters for projection

    Returns:
        RGB skin color as [R, G, B] in 0-1 range
    """
    logger.info("Extracting skin tone from mesh projection")

    # Convert image to numpy array
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    # Simple orthographic projection if no camera params
    # Assuming SMPL mesh is roughly centered and facing camera
    if camera_intrinsics is None:
        # Project vertices to 2D using simple orthographic projection
        # SMPL uses Y-up, so we project X and -Y to image coordinates
        x_coords = vertices[:, 0]
        y_coords = -vertices[:, 1]  # Flip Y for image coordinates

        # Normalize to 0-1 range
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        x_normalized = (x_coords - x_min) / (x_max - x_min + 1e-6)
        y_normalized = (y_coords - y_min) / (y_max - y_min + 1e-6)

        # Scale to image dimensions with padding
        padding = 0.1  # 10% padding
        x_img = (x_normalized * (1 - 2 * padding) + padding) * width
        y_img = (y_normalized * (1 - 2 * padding) + padding) * height
    else:
        # Use provided camera intrinsics for perspective projection
        fx = camera_intrinsics.get("fx", width)
        fy = camera_intrinsics.get("fy", height)
        cx = camera_intrinsics.get("cx", width / 2)
        cy = camera_intrinsics.get("cy", height / 2)

        # Assuming mesh is at some distance from camera
        z = vertices[:, 2] + 2.0  # Add offset to ensure positive z
        z = np.maximum(z, 0.1)  # Avoid division by zero

        x_img = (vertices[:, 0] / z) * fx + cx
        y_img = (-vertices[:, 1] / z) * fy + cy  # Flip Y

    # Sample colors from face vertices
    sampled_colors = []

    for vertex_idx in SMPL_FACE_VERTICES:
        if vertex_idx >= len(vertices):
            continue

        x = int(np.clip(x_img[vertex_idx], 0, width - 1))
        y = int(np.clip(y_img[vertex_idx], 0, height - 1))

        # Sample pixel color
        pixel = img_array[y, x]
        if len(pixel) >= 3:
            sampled_colors.append(pixel[:3])

    # If no face samples, try neck region
    if len(sampled_colors) < 5:
        logger.debug("Few face samples, trying neck region")
        for vertex_idx in SMPL_NECK_VERTICES:
            if vertex_idx >= len(vertices):
                continue
            x = int(np.clip(x_img[vertex_idx], 0, width - 1))
            y = int(np.clip(y_img[vertex_idx], 0, height - 1))
            pixel = img_array[y, x]
            if len(pixel) >= 3:
                sampled_colors.append(pixel[:3])

    if not sampled_colors:
        logger.warning("No skin samples found, using default medium tone")
        return [0.87, 0.72, 0.58]  # Default medium skin tone

    # Convert to numpy array and compute median color (more robust than mean)
    colors_array = np.array(sampled_colors, dtype=np.float32)
    median_color = np.median(colors_array, axis=0)

    # Normalize to 0-1 range
    skin_color = (median_color / 255.0).tolist()

    logger.info(f"Extracted skin tone: RGB({skin_color[0]:.2f}, {skin_color[1]:.2f}, {skin_color[2]:.2f})")

    return skin_color


def extract_skin_tone_simple(image: Image.Image) -> list[float]:
    """
    Extract skin tone using simple face detection heuristics.

    Fallback method when mesh projection is not available.
    Samples from upper-center region of image (typical face location).

    Args:
        image: Source PIL image

    Returns:
        RGB skin color as [R, G, B] in 0-1 range
    """
    logger.info("Extracting skin tone using simple method")

    img_array = np.array(image)
    height, width = img_array.shape[:2]

    # Sample from face region (upper center of image)
    # Assuming typical portrait orientation
    face_top = int(height * 0.15)
    face_bottom = int(height * 0.45)
    face_left = int(width * 0.35)
    face_right = int(width * 0.65)

    face_region = img_array[face_top:face_bottom, face_left:face_right]

    if face_region.size == 0:
        logger.warning("Empty face region, using default")
        return [0.87, 0.72, 0.58]

    # Filter for skin-like colors
    # Skin typically has R > G > B pattern and specific luminance
    skin_samples = []

    for y in range(0, face_region.shape[0], 5):
        for x in range(0, face_region.shape[1], 5):
            pixel = face_region[y, x]
            if len(pixel) >= 3:
                r, g, b = pixel[0], pixel[1], pixel[2]

                # Simple skin color detection heuristics
                # - R > G > B (typical for skin)
                # - Not too dark or too bright
                # - R and G are close (not too red)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b

                if (r > g >= b and
                    50 < luminance < 230 and
                    abs(r - g) < 80):
                    skin_samples.append([r, g, b])

    if len(skin_samples) < 10:
        # Not enough skin-like pixels, use center region average
        logger.debug("Few skin-like pixels, using region average")
        center_color = face_region.mean(axis=(0, 1))[:3]
        return (center_color / 255.0).tolist()

    # Compute median of skin samples
    colors_array = np.array(skin_samples, dtype=np.float32)
    median_color = np.median(colors_array, axis=0)
    skin_color = (median_color / 255.0).tolist()

    logger.info(f"Extracted skin tone (simple): RGB({skin_color[0]:.2f}, {skin_color[1]:.2f}, {skin_color[2]:.2f})")

    return skin_color


def extract_dominant_color(image: Image.Image) -> list[float]:
    """
    Extract dominant color from an image (for garment coloring).

    Args:
        image: Source PIL image

    Returns:
        RGB color as [R, G, B] in 0-1 range
    """
    # Resize for faster processing
    small_img = image.copy()
    small_img.thumbnail((100, 100))
    img_array = np.array(small_img)

    if len(img_array.shape) < 3:
        # Grayscale
        avg = img_array.mean()
        return [avg / 255.0] * 3

    # Get pixels, ignoring alpha if present
    pixels = img_array.reshape(-1, img_array.shape[-1])[:, :3]

    # Filter out very dark and very light pixels (likely background)
    luminance = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    mask = (luminance > 30) & (luminance < 240)
    filtered_pixels = pixels[mask]

    if len(filtered_pixels) == 0:
        filtered_pixels = pixels

    # Compute mean color
    mean_color = filtered_pixels.mean(axis=0)
    return (mean_color / 255.0).tolist()
