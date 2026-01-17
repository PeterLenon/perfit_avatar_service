import os
import subprocess
import json
import numpy as np
from pathlib import Path
from temporalio import activity
from loguru import logger
from typing import Dict, Any, Optional


@activity.defn
async def run_econ_inference(
    user_image_path: str,
    output_dir: str,
    econ_compose_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run ECON inference using docker-compose to generate 3D mesh and SMPL-X parameters.
    
    Args:
        user_image_path: Path to the input user image (on host)
        output_dir: Directory where ECON will write outputs (on host, will be mounted)
        econ_compose_path: Path to docker-compose.yaml file (defaults to env var or ./docker-compose.yaml)
    
    Returns:
        Dictionary containing:
        - mesh_path: Path to generated mesh file (.obj or .ply)
        - smplx_params: SMPL-X parameters (betas, body_pose, global_orient, etc.)
        - normal_map_path: Path to normal map (if generated)
        - texture_path: Path to texture (if generated)
    """
    # Get docker-compose path from env or use default
    if econ_compose_path is None:
        econ_compose_path = os.getenv("ECON_COMPOSE_PATH", "./docker-compose.yaml")
    
    # Ensure output directory exists on host
    os.makedirs(output_dir, exist_ok=True)
    
    # Create input directory structure that ECON expects
    input_dir = os.path.join(output_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    
    # Copy user image to input directory (ECON processes all images in input dir)
    import shutil
    image_filename = os.path.basename(user_image_path)
    temp_image_path = os.path.join(input_dir, image_filename)
    shutil.copy2(user_image_path, temp_image_path)
    
    # Get absolute paths for Docker volume mounts
    abs_output_dir = os.path.abspath(output_dir)
    abs_input_dir = os.path.abspath(input_dir)
    abs_compose_path = os.path.abspath(econ_compose_path)
    
    # When running in a container, docker-compose (running on host via bind-mounted socket)
    # needs host paths, not container paths. If AVATAR_WORKDIR_HOST is set, use it to map paths.
    if os.path.exists("/.dockerenv"):  # Running in Docker
        host_workdir = os.getenv("AVATAR_WORKDIR_HOST")
        if host_workdir:
            # Map container paths to host paths
            # Container: /app/workdir/user_id -> Host: ${AVATAR_WORKDIR_HOST}/user_id
            if abs_output_dir.startswith("/app/workdir"):
                rel_path = abs_output_dir.replace("/app/workdir", "").lstrip("/")
                abs_output_dir = os.path.join(host_workdir, rel_path) if rel_path else host_workdir
            if abs_input_dir.startswith("/app/workdir"):
                rel_path = abs_input_dir.replace("/app/workdir", "").lstrip("/")
                abs_input_dir = os.path.join(host_workdir, rel_path) if rel_path else host_workdir
            logger.info(f"Mapped container paths to host: output={abs_output_dir}, input={abs_input_dir}")
    
    # Get the directory containing docker-compose.yaml (needed for docker-compose context)
    compose_dir = os.path.dirname(abs_compose_path)
    
    try:
        # Run ECON using docker-compose
        # Based on ECON docs: python -m apps.infer -cfg ./configs/econ.yaml -in_dir ./examples -out_dir ./results
        # Note: Volume mounts should be configured in docker-compose.yaml
        # The compose file should mount abs_input_dir to /app/input and abs_output_dir to /app/output
        # You may need to update your docker-compose.yaml to include:
        # volumes:
        #   - ${INPUT_DIR}:/app/input:ro
        #   - ${OUTPUT_DIR}:/app/output
        docker_compose_cmd = [
            "docker-compose",
            "-f", abs_compose_path,
            "run", "--rm",
            "econ",  # service name in docker-compose (adjust if different)
            "python", "-m", "apps.infer",
            "-cfg", "/app/configs/econ.yaml",  # Path inside container
            "-in_dir", "/app/input",
            "-out_dir", "/app/output",
        ]
        
        # Set environment variables for docker-compose
        env = os.environ.copy()
        env["INPUT_DIR"] = abs_input_dir
        env["OUTPUT_DIR"] = abs_output_dir
        
        logger.info(f"Running ECON via docker-compose: {' '.join(docker_compose_cmd)}")
        
        result = subprocess.run(
            docker_compose_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=1800,  # 30 minute timeout
            env=env,
            cwd=compose_dir,  # Run from the directory containing docker-compose.yaml
        )
        
        logger.info(f"ECON Docker Compose execution completed. Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"ECON stderr: {result.stderr}")
        
        # ECON typically outputs to: output_dir/econ/cache/<image_name>/
        # Look for mesh files and SMPL-X parameters
        image_name = Path(image_filename).stem
        econ_output_dir = Path(output_dir) / "econ" / "cache" / image_name
        
        if not econ_output_dir.exists():
            # Try alternative output locations
            econ_output_dir = Path(output_dir) / "econ" / image_name
            if not econ_output_dir.exists():
                econ_output_dir = Path(output_dir) / image_name
        
        if not econ_output_dir.exists():
            raise activity.ApplicationError(
                f"ECON output directory not found: {econ_output_dir}",
                non_retryable=False,
            )
        
        # Find mesh file (.obj or .ply)
        mesh_path = None
        for ext in [".obj", ".ply"]:
            potential_mesh = econ_output_dir / f"{image_name}{ext}"
            if potential_mesh.exists():
                mesh_path = str(potential_mesh)
                break
        
        if mesh_path is None:
            # Try alternative locations/names
            for mesh_file in econ_output_dir.glob("*.obj"):
                mesh_path = str(mesh_file)
                break
            if mesh_path is None:
                for mesh_file in econ_output_dir.glob("*.ply"):
                    mesh_path = str(mesh_file)
                    break
        
        if mesh_path is None:
            logger.warning(f"No mesh file found in {econ_output_dir}")
        
        # Load SMPL-X parameters
        # ECON typically saves SMPL-X params as .npz or .pkl files
        smplx_params = {}
        smplx_file = None
        
        for ext in [".npz", ".pkl"]:
            potential_smplx = econ_output_dir / f"smplx_params{ext}"
            if potential_smplx.exists():
                smplx_file = potential_smplx
                break
        
        if smplx_file is None:
            # Try alternative naming conventions
            for smplx_file_candidate in econ_output_dir.glob("*smplx*"):
                if smplx_file_candidate.suffix in [".npz", ".pkl"]:
                    smplx_file = smplx_file_candidate
                    break
        
        # Also check for common ECON output files
        if smplx_file is None:
            for pattern in ["*param*.npz", "*param*.pkl", "*body*.npz", "*body*.pkl"]:
                for candidate in econ_output_dir.glob(pattern):
                    smplx_file = candidate
                    break
                if smplx_file:
                    break
        
        if smplx_file:
            if smplx_file.suffix == ".npz":
                smplx_data = np.load(smplx_file, allow_pickle=True)
                # Convert numpy arrays to lists for JSON serialization
                smplx_params = {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in smplx_data.items()
                }
            else:  # .pkl
                import pickle
                with open(smplx_file, "rb") as f:
                    smplx_params = pickle.load(f)
                    # Convert numpy arrays to lists
                    if isinstance(smplx_params, dict):
                        smplx_params = {
                            key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in smplx_params.items()
                        }
        else:
            logger.warning(f"No SMPL-X parameters file found in {econ_output_dir}")
        
        # Find normal map and texture if available
        normal_map_path = None
        texture_path = None
        
        for normal_file in econ_output_dir.glob("*normal*.png"):
            normal_map_path = str(normal_file)
            break
        
        for texture_file in econ_output_dir.glob("*texture*.png"):
            texture_path = str(texture_file)
            break
        
        return {
            "mesh_path": mesh_path,
            "smplx_params": smplx_params,
            "normal_map_path": normal_map_path,
            "texture_path": texture_path,
            "output_dir": str(econ_output_dir),
        }
        
    except subprocess.TimeoutExpired:
        logger.error("ECON Docker Compose execution timed out")
        raise activity.ApplicationError(
            "ECON inference timed out after 30 minutes",
            non_retryable=False,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"ECON Docker Compose execution failed: {e.stderr}")
        raise activity.ApplicationError(
            f"ECON inference failed: {e.stderr}",
            non_retryable=False,
        )
    except Exception as e:
        logger.error(f"Error running ECON inference: {str(e)}")
        raise activity.ApplicationError(
            f"Error running ECON inference: {str(e)}",
            non_retryable=False,
        )
    finally:
        # Clean up temporary input directory (optional - you might want to keep it for debugging)
        # if os.path.exists(temp_image_path):
        #     os.remove(temp_image_path)
        pass

