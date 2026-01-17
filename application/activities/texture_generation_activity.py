import os
import subprocess
from pathlib import Path
from temporalio import activity
from loguru import logger
from typing import Dict, Any, Optional
import cv2
import numpy as np
import shutil
import json


@activity.defn
async def generate_realistic_textures(
    mesh_path: str,
    user_image_path: str,
    output_dir: str,
    texture_tool: str = "pshuman",  # "pshuman" (recommended), "hunyuan3d", "texdreamer", "enhanced", "simple", "opentryon", "skip", "phorhum", "morphx"
    docker_compose_path: Optional[str] = None,
    smplx_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate realistic textures for the avatar mesh suitable for virtual try-on.
    
    Args:
        mesh_path: Path to the ECON-generated mesh file (.obj or .ply)
        user_image_path: Path to the original user image
        output_dir: Directory where textured mesh will be saved
        texture_tool: Tool to use - "pshuman" (recommended for photorealistic), "hunyuan3d", "texdreamer", "enhanced", "simple", "opentryon", "skip", "phorhum", or "morphx"
        docker_compose_path: Path to docker-compose.yaml file for the texture tool
        smplx_params: Optional SMPL-X parameters from ECON for better texture mapping
    
    Returns:
        Dictionary containing:
        - textured_mesh_path: Path to mesh with realistic textures
        - texture_map_path: Path to texture map/image
        - normal_map_path: Path to enhanced normal map (if generated)
        - diffuse_map_path: Path to diffuse texture map (if generated)
    """
    if texture_tool not in ["phorhum", "morphx", "simple", "skip", "opentryon", "enhanced", "pshuman", "hunyuan3d", "texdreamer"]:
        raise activity.ApplicationError(
            f"Invalid texture tool: {texture_tool}. Must be 'pshuman' (recommended), 'hunyuan3d', 'texdreamer', 'enhanced', 'simple', 'opentryon', 'skip', 'phorhum', or 'morphx'",
            non_retryable=True,
        )
    
    # Skip texture generation if requested
    if texture_tool == "skip":
        logger.info("Skipping texture generation as requested")
        return {
            "textured_mesh_path": None,
            "texture_map_path": None,
            "diffuse_map_path": None,
            "normal_map_path": None,
            "output_dir": output_dir,
        }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get absolute paths
    abs_mesh_path = os.path.abspath(mesh_path)
    abs_image_path = os.path.abspath(user_image_path)
    abs_output_dir = os.path.abspath(output_dir)
    
    # Verify inputs exist
    if not os.path.exists(abs_mesh_path):
        raise activity.ApplicationError(
            f"Mesh file not found: {abs_mesh_path}",
            non_retryable=True,
        )
    if not os.path.exists(abs_image_path):
        raise activity.ApplicationError(
            f"User image not found: {abs_image_path}",
            non_retryable=True,
        )
    
    try:
        if texture_tool == "phorhum":
            abs_compose_path = os.path.abspath(docker_compose_path or os.getenv("PHORHUM_COMPOSE_PATH", "./docker-compose-phorhum.yaml"))
            compose_dir = os.path.dirname(abs_compose_path)
            return await _run_phorhum(
                abs_mesh_path,
                abs_image_path,
                abs_output_dir,
                abs_compose_path,
                compose_dir,
            )
        elif texture_tool == "morphx":
            abs_compose_path = os.path.abspath(docker_compose_path or os.getenv("MORPHX_COMPOSE_PATH", "./docker-compose-morphx.yaml"))
            compose_dir = os.path.dirname(abs_compose_path)
            return await _run_morphx(
                abs_mesh_path,
                abs_image_path,
                abs_output_dir,
                abs_compose_path,
                compose_dir,
            )
        elif texture_tool == "opentryon":
            return await _run_opentryon(
                abs_mesh_path,
                abs_image_path,
                abs_output_dir,
                smplx_params,
            )
        elif texture_tool == "pshuman":
            return await _run_pshuman(
                abs_mesh_path,
                abs_image_path,
                abs_output_dir,
                smplx_params,
            )
        elif texture_tool == "hunyuan3d":
            return await _run_hunyuan3d(
                abs_mesh_path,
                abs_image_path,
                abs_output_dir,
                smplx_params,
            )
        elif texture_tool == "texdreamer":
            return await _run_texdreamer(
                abs_mesh_path,
                abs_image_path,
                abs_output_dir,
                smplx_params,
            )
        elif texture_tool == "enhanced":
            return await _run_enhanced_texture_projection(
                abs_mesh_path,
                abs_image_path,
                abs_output_dir,
                smplx_params,
            )
        elif texture_tool == "simple":
            return await _run_simple_texture_projection(
                abs_mesh_path,
                abs_image_path,
                abs_output_dir,
            )
        else:
            raise ValueError(f"Unknown texture tool: {texture_tool}")
    except Exception as e:
        logger.error(f"Error running {texture_tool}: {str(e)}")
        raise activity.ApplicationError(
            f"Error running {texture_tool}: {str(e)}",
            non_retryable=False,
        )


async def _run_enhanced_texture_projection(
    mesh_path: str,
    image_path: str,
    output_dir: str,
    smplx_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Enhanced texture projection that creates better textures for virtual try-on.
    
    This method:
    1. Uses the input image more intelligently
    2. Creates a higher resolution texture map (2048x2048)
    3. Generates a basic normal map for better lighting
    4. Optimizes the texture for clothing visualization
    
    Args:
        mesh_path: Path to the ECON-generated mesh file
        image_path: Path to the original user image
        output_dir: Directory where outputs will be saved
        smplx_params: Optional SMPL-X parameters for better mapping
    
    Returns:
        Dictionary with texture generation results
    """
    logger.info("Running enhanced texture projection for virtual try-on")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess the input image
    img = cv2.imread(image_path)
    if img is None:
        raise activity.ApplicationError(f"Could not load image from {image_path}", non_retryable=True)
    
    # Create high-resolution texture map (2048x2048 for better quality)
    # Use LANCZOS4 interpolation for better quality
    texture_map = cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_LANCZOS4)
    
    # Enhance the texture for better virtual try-on visualization
    # Apply slight sharpening to preserve details
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    texture_map = cv2.filter2D(texture_map, -1, kernel * 0.1) + texture_map * 0.9
    
    # Normalize brightness/contrast for better clothing visualization
    lab = cv2.cvtColor(texture_map, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    texture_map = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    # Save enhanced texture map
    texture_map_path = os.path.join(output_dir, "texture_map_enhanced.png")
    cv2.imwrite(texture_map_path, texture_map)
    
    # Generate a basic normal map from the texture (for better lighting in 3D viewers)
    # Convert to grayscale for normal map generation
    gray = cv2.cvtColor(texture_map, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradients for normal map
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Normalize gradients
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2 + 1.0)
    normal_x = (sobel_x / magnitude + 1.0) * 127.5
    normal_y = (sobel_y / magnitude + 1.0) * 127.5
    normal_z = (1.0 / magnitude) * 255.0
    
    # Create normal map (BGR format: X=Blue, Y=Green, Z=Red)
    normal_map = np.zeros((2048, 2048, 3), dtype=np.uint8)
    normal_map[:, :, 0] = normal_z.astype(np.uint8)  # B
    normal_map[:, :, 1] = normal_y.astype(np.uint8)  # G
    normal_map[:, :, 2] = normal_x.astype(np.uint8)  # R
    
    normal_map_path = os.path.join(output_dir, "normal_map.png")
    cv2.imwrite(normal_map_path, normal_map)
    
    # Copy the mesh as the "textured" mesh
    mesh_filename = os.path.basename(mesh_path)
    textured_mesh_path = os.path.join(output_dir, f"textured_{mesh_filename}")
    shutil.copy2(mesh_path, textured_mesh_path)
    
    # Create a diffuse map (same as texture for now, but can be enhanced)
    diffuse_map_path = texture_map_path
    
    logger.info(f"Enhanced texture projection completed. Texture: {texture_map_path}, Normal: {normal_map_path}")
    
    return {
        "textured_mesh_path": textured_mesh_path,
        "texture_map_path": texture_map_path,
        "diffuse_map_path": diffuse_map_path,
        "normal_map_path": normal_map_path,
        "output_dir": output_dir,
    }


async def _run_opentryon(
    mesh_path: str,
    image_path: str,
    output_dir: str,
    smplx_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run OpenTryOn for photorealistic texture generation.
    
    OpenTryOn is an open-source toolkit for virtual try-on that can generate
    photorealistic textures suitable for clothing visualization.
    
    GitHub: https://github.com/tryonlabs/opentryon
    
    Args:
        mesh_path: Path to the ECON-generated mesh file
        image_path: Path to the original user image
        output_dir: Directory where outputs will be saved
        smplx_params: Optional SMPL-X parameters
    
    Returns:
        Dictionary with texture generation results
    """
    logger.info("Running OpenTryOn texture generation")
    
    # Check if OpenTryOn is available
    opentryon_path = os.getenv("OPENTRYON_PATH", None)
    if not opentryon_path or not os.path.exists(opentryon_path):
        logger.warning(
            "OpenTryOn not found. Install from https://github.com/tryonlabs/opentryon. "
            "Falling back to enhanced texture projection."
        )
        return await _run_enhanced_texture_projection(mesh_path, image_path, output_dir, smplx_params)
    
    # OpenTryOn integration would go here
    # This is a placeholder - actual implementation depends on OpenTryOn's API
    logger.info("OpenTryOn integration not yet implemented. Using enhanced projection.")
    return await _run_enhanced_texture_projection(mesh_path, image_path, output_dir, smplx_params)


async def _run_simple_texture_projection(
    mesh_path: str,
    image_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Generate a simple texture by projecting the input image onto the mesh.
    
    This is a basic fallback method that creates a simple texture map from the input image.
    
    Args:
        mesh_path: Path to the ECON-generated mesh file
        image_path: Path to the original user image
        output_dir: Directory where outputs will be saved
    
    Returns:
        Dictionary with texture generation results
    """
    logger.info("Running simple texture projection (fallback method)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy the original mesh as the "textured" mesh
    mesh_filename = os.path.basename(mesh_path)
    textured_mesh_path = os.path.join(output_dir, f"textured_{mesh_filename}")
    shutil.copy2(mesh_path, textured_mesh_path)
    
    # Create a simple texture map from the input image
    # Load the image and resize it to a standard texture size (e.g., 1024x1024)
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"Could not load image from {image_path}, creating placeholder texture")
        # Create a simple colored texture as fallback
        texture_map = np.ones((1024, 1024, 3), dtype=np.uint8) * 128  # Gray texture
    else:
        # Resize to standard texture size
        texture_map = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
    
    # Save texture map
    texture_map_path = os.path.join(output_dir, "texture_map.png")
    cv2.imwrite(texture_map_path, texture_map)
    
    logger.info(f"Simple texture projection completed. Texture saved to {texture_map_path}")
    
    return {
        "textured_mesh_path": textured_mesh_path,
        "texture_map_path": texture_map_path,
        "diffuse_map_path": texture_map_path,  # Use same texture as diffuse
        "normal_map_path": None,  # No normal map generated
        "output_dir": output_dir,
    }


async def _run_phorhum(
    mesh_path: str,
    image_path: str,
    output_dir: str,
    compose_path: str,
    compose_dir: str,
) -> Dict[str, Any]:
    """
    Run PHORHUM to generate realistic textures.
    
    PHORHUM typically takes:
    - Input image
    - Mesh (optional, can generate from image)
    - Output directory
    
    Command format: python -m phorhum.infer --input_image <path> --mesh <path> --output <path>
    """
    logger.info(f"Running PHORHUM texture generation")
    
    # PHORHUM docker-compose command
    # Adjust service name and command based on PHORHUM's actual API
    docker_compose_cmd = [
        "docker-compose",
        "-f", compose_path,
        "run", "--rm",
        "phorhum",  # service name - adjust if different
        "python", "-m", "phorhum.infer",
        "--input_image", "/app/input_image.jpg",
        "--mesh", "/app/input_mesh.obj",
        "--output", "/app/output",
    ]
    
    # Set up input directory for PHORHUM
    phorhum_input_dir = os.path.join(output_dir, "phorhum_input")
    os.makedirs(phorhum_input_dir, exist_ok=True)
    
    import shutil
    # Copy files to input directory
    input_image_path = os.path.join(phorhum_input_dir, "input_image.jpg")
    input_mesh_path = os.path.join(phorhum_input_dir, "input_mesh.obj")
    shutil.copy2(image_path, input_image_path)
    shutil.copy2(mesh_path, input_mesh_path)
    
    abs_input_dir = os.path.abspath(phorhum_input_dir)
    
    # Set environment variables
    env = os.environ.copy()
    env["INPUT_DIR"] = abs_input_dir
    env["OUTPUT_DIR"] = output_dir
    
    try:
        result = subprocess.run(
            docker_compose_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=1800,  # 30 minute timeout
            env=env,
            cwd=compose_dir,
        )
        
        logger.info(f"PHORHUM execution completed. Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"PHORHUM stderr: {result.stderr}")
        
        # PHORHUM typically outputs textured mesh and texture maps
        phorhum_output_dir = Path(output_dir) / "phorhum_output"
        if not phorhum_output_dir.exists():
            phorhum_output_dir = Path(output_dir)
        
        # Find textured mesh
        textured_mesh_path = None
        for mesh_file in phorhum_output_dir.glob("*.obj"):
            textured_mesh_path = str(mesh_file)
            break
        
        # Find texture maps
        texture_map_path = None
        diffuse_map_path = None
        normal_map_path = None
        
        for texture_file in phorhum_output_dir.glob("*texture*.png"):
            texture_map_path = str(texture_file)
            break
        
        for diffuse_file in phorhum_output_dir.glob("*diffuse*.png"):
            diffuse_map_path = str(diffuse_file)
            break
        
        for normal_file in phorhum_output_dir.glob("*normal*.png"):
            normal_map_path = str(normal_file)
            break
        
        return {
            "textured_mesh_path": textured_mesh_path,
            "texture_map_path": texture_map_path,
            "diffuse_map_path": diffuse_map_path,
            "normal_map_path": normal_map_path,
            "output_dir": str(phorhum_output_dir),
        }
        
    except subprocess.TimeoutExpired:
        logger.error("PHORHUM execution timed out")
        raise activity.ApplicationError(
            "PHORHUM texture generation timed out after 30 minutes",
            non_retryable=False,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"PHORHUM execution failed: {e.stderr}")
        raise activity.ApplicationError(
            f"PHORHUM texture generation failed: {e.stderr}",
            non_retryable=False,
        )


async def _run_morphx(
    mesh_path: str,
    image_path: str,
    output_dir: str,
    compose_path: str,
    compose_dir: str,
) -> Dict[str, Any]:
    """
    Run MORPHX to generate realistic textures.
    
    MORPHX typically takes:
    - Input image
    - Mesh
    - Output directory
    
    Command format: python -m morphx.infer --image <path> --mesh <path> --output <path>
    """
    logger.info(f"Running MORPHX texture generation")
    
    # MORPHX docker-compose command
    # Adjust service name and command based on MORPHX's actual API
    docker_compose_cmd = [
        "docker-compose",
        "-f", compose_path,
        "run", "--rm",
        "morphx",  # service name - adjust if different
        "python", "-m", "morphx.infer",
        "--image", "/app/input_image.jpg",
        "--mesh", "/app/input_mesh.obj",
        "--output", "/app/output",
    ]
    
    # Set up input directory for MORPHX
    morphx_input_dir = os.path.join(output_dir, "morphx_input")
    os.makedirs(morphx_input_dir, exist_ok=True)
    
    import shutil
    # Copy files to input directory
    input_image_path = os.path.join(morphx_input_dir, "input_image.jpg")
    input_mesh_path = os.path.join(morphx_input_dir, "input_mesh.obj")
    shutil.copy2(image_path, input_image_path)
    shutil.copy2(mesh_path, input_mesh_path)
    
    abs_input_dir = os.path.abspath(morphx_input_dir)
    
    # Set environment variables
    env = os.environ.copy()
    env["INPUT_DIR"] = abs_input_dir
    env["OUTPUT_DIR"] = output_dir
    
    try:
        result = subprocess.run(
            docker_compose_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=1800,  # 30 minute timeout
            env=env,
            cwd=compose_dir,
        )
        
        logger.info(f"MORPHX execution completed. Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"MORPHX stderr: {result.stderr}")
        
        # MORPHX typically outputs textured mesh and texture maps
        morphx_output_dir = Path(output_dir) / "morphx_output"
        if not morphx_output_dir.exists():
            morphx_output_dir = Path(output_dir)
        
        # Find textured mesh
        textured_mesh_path = None
        for mesh_file in morphx_output_dir.glob("*.obj"):
            textured_mesh_path = str(mesh_file)
            break
        
        # Find texture maps
        texture_map_path = None
        diffuse_map_path = None
        normal_map_path = None
        
        for texture_file in morphx_output_dir.glob("*texture*.png"):
            texture_map_path = str(texture_file)
            break
        
        for diffuse_file in morphx_output_dir.glob("*diffuse*.png"):
            diffuse_map_path = str(diffuse_file)
            break
        
        for normal_file in morphx_output_dir.glob("*normal*.png"):
            normal_map_path = str(normal_file)
            break
        
        return {
            "textured_mesh_path": textured_mesh_path,
            "texture_map_path": texture_map_path,
            "diffuse_map_path": diffuse_map_path,
            "normal_map_path": normal_map_path,
            "output_dir": str(morphx_output_dir),
        }
        
    except subprocess.TimeoutExpired:
        logger.error("MORPHX execution timed out")
        raise activity.ApplicationError(
            "MORPHX texture generation timed out after 30 minutes",
            non_retryable=False,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"MORPHX execution failed: {e.stderr}")
        raise activity.ApplicationError(
            f"MORPHX texture generation failed: {e.stderr}",
            non_retryable=False,
        )


async def _run_pshuman(
    mesh_path: str,
    image_path: str,
    output_dir: str,
    smplx_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run PSHuman for photorealistic texture generation.
    
    PSHuman generates photorealistic textures from single images using cross-scale
    multiview diffusion. It produces high-quality UV-mapped textures suitable for
    virtual try-on applications.
    
    Paper: https://arxiv.org/abs/2505.12635
    Project: https://penghtyx.github.io/PSHuman/
    
    Args:
        mesh_path: Path to the ECON-generated mesh file
        image_path: Path to the original user image
        output_dir: Directory where outputs will be saved
        smplx_params: Optional SMPL-X parameters
    
    Returns:
        Dictionary with texture generation results
    """
    logger.info("Running PSHuman for photorealistic texture generation")
    
    # Check if PSHuman is available
    pshuman_path = os.getenv("PSHUMAN_PATH", None)
    if not pshuman_path or not os.path.exists(pshuman_path):
        logger.warning(
            "PSHuman not found. Install from https://github.com/penghtyx/PSHuman. "
            "Falling back to enhanced texture projection."
        )
        return await _run_enhanced_texture_projection(mesh_path, image_path, output_dir, smplx_params)
    
    # Set up input directory
    pshuman_input_dir = os.path.join(output_dir, "pshuman_input")
    os.makedirs(pshuman_input_dir, exist_ok=True)
    
    # Copy input files
    input_image_path = os.path.join(pshuman_input_dir, "input_image.jpg")
    shutil.copy2(image_path, input_image_path)
    
    # If SMPL-X params available, save them
    if smplx_params:
        smplx_path = os.path.join(pshuman_input_dir, "smplx_params.json")
        with open(smplx_path, "w") as f:
            json.dump(smplx_params, f)
    
    # PSHuman command (adjust based on actual API)
    # Typical command: python inference.py --input_image <path> --output <path>
    pshuman_cmd = [
        "python",
        os.path.join(pshuman_path, "inference.py"),  # Adjust path as needed
        "--input_image", input_image_path,
        "--output", output_dir,
    ]
    
    # Add mesh path if PSHuman supports it
    if mesh_path:
        pshuman_cmd.extend(["--mesh", mesh_path])
    
    try:
        result = subprocess.run(
            pshuman_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=1800,  # 30 minute timeout
            cwd=pshuman_path,
        )
        
        logger.info(f"PSHuman execution completed. Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"PSHuman stderr: {result.stderr}")
        
        # PSHuman typically outputs:
        # - Textured mesh (.obj)
        # - Texture map (.png)
        # - Normal map (.png)
        pshuman_output_dir = Path(output_dir) / "pshuman_output"
        if not pshuman_output_dir.exists():
            pshuman_output_dir = Path(output_dir)
        
        # Find outputs
        textured_mesh_path = None
        for mesh_file in pshuman_output_dir.glob("*.obj"):
            textured_mesh_path = str(mesh_file)
            break
        
        texture_map_path = None
        for texture_file in pshuman_output_dir.glob("*texture*.png"):
            texture_map_path = str(texture_file)
            break
        
        normal_map_path = None
        for normal_file in pshuman_output_dir.glob("*normal*.png"):
            normal_map_path = str(normal_file)
            break
        
        diffuse_map_path = texture_map_path  # PSHuman may generate separate diffuse
        
        return {
            "textured_mesh_path": textured_mesh_path or mesh_path,
            "texture_map_path": texture_map_path,
            "diffuse_map_path": diffuse_map_path,
            "normal_map_path": normal_map_path,
            "output_dir": str(pshuman_output_dir),
        }
        
    except subprocess.TimeoutExpired:
        logger.error("PSHuman execution timed out")
        raise activity.ApplicationError(
            "PSHuman texture generation timed out after 30 minutes",
            non_retryable=False,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"PSHuman execution failed: {e.stderr}")
        logger.warning("Falling back to enhanced texture projection")
        return await _run_enhanced_texture_projection(mesh_path, image_path, output_dir, smplx_params)
    except FileNotFoundError:
        logger.warning("PSHuman not found, falling back to enhanced texture projection")
        return await _run_enhanced_texture_projection(mesh_path, image_path, output_dir, smplx_params)


async def _run_hunyuan3d(
    mesh_path: str,
    image_path: str,
    output_dir: str,
    smplx_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run Hunyuan3D-2.1 for photorealistic texture generation with PBR materials.
    
    Hunyuan3D-2.1 is Tencent's open-source pipeline for generating production-ready
    3D assets with photorealistic PBR textures.
    
    GitHub: https://github.com/tencent-hunyuan/hunyuan3d-2.1
    
    Args:
        mesh_path: Path to the ECON-generated mesh file
        image_path: Path to the original user image
        output_dir: Directory where outputs will be saved
        smplx_params: Optional SMPL-X parameters
    
    Returns:
        Dictionary with texture generation results
    """
    logger.info("Running Hunyuan3D-2.1 for photorealistic PBR texture generation")
    
    # Check if Hunyuan3D is available
    hunyuan3d_path = os.getenv("HUNYUAN3D_PATH", None)
    if not hunyuan3d_path or not os.path.exists(hunyuan3d_path):
        logger.warning(
            "Hunyuan3D-2.1 not found. Install from https://github.com/tencent-hunyuan/hunyuan3d-2.1. "
            "Falling back to enhanced texture projection."
        )
        return await _run_enhanced_texture_projection(mesh_path, image_path, output_dir, smplx_params)
    
    # Set up input directory
    hunyuan3d_input_dir = os.path.join(output_dir, "hunyuan3d_input")
    os.makedirs(hunyuan3d_input_dir, exist_ok=True)
    
    # Copy input files
    input_image_path = os.path.join(hunyuan3d_input_dir, "input_image.jpg")
    shutil.copy2(image_path, input_image_path)
    
    # Hunyuan3D command (adjust based on actual API)
    # Typical command: python inference.py --image <path> --output <path>
    hunyuan3d_cmd = [
        "python",
        os.path.join(hunyuan3d_path, "inference.py"),  # Adjust path as needed
        "--image", input_image_path,
        "--output", output_dir,
    ]
    
    try:
        result = subprocess.run(
            hunyuan3d_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=1800,  # 30 minute timeout
            cwd=hunyuan3d_path,
        )
        
        logger.info(f"Hunyuan3D execution completed. Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Hunyuan3D stderr: {result.stderr}")
        
        # Hunyuan3D outputs PBR materials:
        # - Albedo/diffuse map
        # - Normal map
        # - Roughness map
        # - Specular map
        hunyuan3d_output_dir = Path(output_dir) / "hunyuan3d_output"
        if not hunyuan3d_output_dir.exists():
            hunyuan3d_output_dir = Path(output_dir)
        
        # Find outputs
        textured_mesh_path = None
        for mesh_file in hunyuan3d_output_dir.glob("*.obj"):
            textured_mesh_path = str(mesh_file)
            break
        
        texture_map_path = None
        for texture_file in hunyuan3d_output_dir.glob("*albedo*.png"):
            texture_map_path = str(texture_file)
            break
        if not texture_map_path:
            for texture_file in hunyuan3d_output_dir.glob("*diffuse*.png"):
                texture_map_path = str(texture_file)
                break
        
        normal_map_path = None
        for normal_file in hunyuan3d_output_dir.glob("*normal*.png"):
            normal_map_path = str(normal_file)
            break
        
        diffuse_map_path = texture_map_path
        
        return {
            "textured_mesh_path": textured_mesh_path or mesh_path,
            "texture_map_path": texture_map_path,
            "diffuse_map_path": diffuse_map_path,
            "normal_map_path": normal_map_path,
            "output_dir": str(hunyuan3d_output_dir),
        }
        
    except subprocess.TimeoutExpired:
        logger.error("Hunyuan3D execution timed out")
        raise activity.ApplicationError(
            "Hunyuan3D texture generation timed out after 30 minutes",
            non_retryable=False,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Hunyuan3D execution failed: {e.stderr}")
        logger.warning("Falling back to enhanced texture projection")
        return await _run_enhanced_texture_projection(mesh_path, image_path, output_dir, smplx_params)
    except FileNotFoundError:
        logger.warning("Hunyuan3D not found, falling back to enhanced texture projection")
        return await _run_enhanced_texture_projection(mesh_path, image_path, output_dir, smplx_params)


async def _run_texdreamer(
    mesh_path: str,
    image_path: str,
    output_dir: str,
    smplx_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run TexDreamer for zero-shot high-fidelity human texture generation.
    
    TexDreamer generates photorealistic textures from text or image input with
    semantic UV map support.
    
    Project: https://ggxxii.github.io/texdreamer/
    
    Args:
        mesh_path: Path to the ECON-generated mesh file
        image_path: Path to the original user image
        output_dir: Directory where outputs will be saved
        smplx_params: Optional SMPL-X parameters
    
    Returns:
        Dictionary with texture generation results
    """
    logger.info("Running TexDreamer for photorealistic texture generation")
    
    # Check if TexDreamer is available
    texdreamer_path = os.getenv("TEXDREAMER_PATH", None)
    if not texdreamer_path or not os.path.exists(texdreamer_path):
        logger.warning(
            "TexDreamer not found. Install from https://github.com/ggxxii/TexDreamer. "
            "Falling back to enhanced texture projection."
        )
        return await _run_enhanced_texture_projection(mesh_path, image_path, output_dir, smplx_params)
    
    # Set up input directory
    texdreamer_input_dir = os.path.join(output_dir, "texdreamer_input")
    os.makedirs(texdreamer_input_dir, exist_ok=True)
    
    # Copy input files
    input_image_path = os.path.join(texdreamer_input_dir, "input_image.jpg")
    shutil.copy2(image_path, input_image_path)
    
    # TexDreamer command (adjust based on actual API)
    # Typical command: python inference.py --image <path> --output <path>
    texdreamer_cmd = [
        "python",
        os.path.join(texdreamer_path, "inference.py"),  # Adjust path as needed
        "--image", input_image_path,
        "--output", output_dir,
    ]
    
    try:
        result = subprocess.run(
            texdreamer_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=1800,  # 30 minute timeout
            cwd=texdreamer_path,
        )
        
        logger.info(f"TexDreamer execution completed. Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"TexDreamer stderr: {result.stderr}")
        
        # TexDreamer outputs high-resolution textures
        texdreamer_output_dir = Path(output_dir) / "texdreamer_output"
        if not texdreamer_output_dir.exists():
            texdreamer_output_dir = Path(output_dir)
        
        # Find outputs
        textured_mesh_path = None
        for mesh_file in texdreamer_output_dir.glob("*.obj"):
            textured_mesh_path = str(mesh_file)
            break
        
        texture_map_path = None
        for texture_file in texdreamer_output_dir.glob("*texture*.png"):
            texture_map_path = str(texture_file)
            break
        
        normal_map_path = None
        for normal_file in texdreamer_output_dir.glob("*normal*.png"):
            normal_map_path = str(normal_file)
            break
        
        diffuse_map_path = texture_map_path
        
        return {
            "textured_mesh_path": textured_mesh_path or mesh_path,
            "texture_map_path": texture_map_path,
            "diffuse_map_path": diffuse_map_path,
            "normal_map_path": normal_map_path,
            "output_dir": str(texdreamer_output_dir),
        }
        
    except subprocess.TimeoutExpired:
        logger.error("TexDreamer execution timed out")
        raise activity.ApplicationError(
            "TexDreamer texture generation timed out after 30 minutes",
            non_retryable=False,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"TexDreamer execution failed: {e.stderr}")
        logger.warning("Falling back to enhanced texture projection")
        return await _run_enhanced_texture_projection(mesh_path, image_path, output_dir, smplx_params)
    except FileNotFoundError:
        logger.warning("TexDreamer not found, falling back to enhanced texture projection")
        return await _run_enhanced_texture_projection(mesh_path, image_path, output_dir, smplx_params)
