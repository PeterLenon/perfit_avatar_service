# ECON Docker Compose Setup Guide

This guide explains how to configure ECON with docker-compose for the avatar service.

## Running Context

**Important:** This guide assumes the avatar service code runs **directly on the host machine**. 

If you want to run the entire avatar service via docker-compose, see [DOCKER_SETUP.md](./DOCKER_SETUP.md) for instructions on containerizing the avatar service itself.

## Prerequisites

1. Clone the ECON repository:
   ```bash
   git clone https://github.com/YuliangXiu/ECON.git
   cd ECON
   ```

2. Follow ECON's installation instructions to set up the docker-compose.yaml file.

## Docker Compose Configuration

Your `docker-compose.yaml` file should have a service named `econ` (or update the service name in `config.yaml` and `econ_activity.py`).

### Current Implementation: Volume Mounts (Recommended)

The current implementation uses **volume mounts** to share files between the host and container. This is the recommended approach because:
- ✅ No need to copy files into/out of the container
- ✅ Files are accessible on the host for debugging
- ✅ More efficient (no file copying overhead)
- ✅ Automatic cleanup when host directories are removed

The docker-compose.yaml should mount volumes for input and output. Example configuration:

```yaml
version: '3.8'

services:
  econ:
    build: .
    # or use: image: econ:latest
    volumes:
      - ${INPUT_DIR:-./input}:/app/input:ro  # Host input dir → container /app/input
      - ${OUTPUT_DIR:-./output}:/app/output   # Host output dir → container /app/output
      - ./configs:/app/configs:ro
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**How it works:**

Here's the complete flow showing how the hardcoded container paths in the command relate to the volume mounts:

1. **Activity creates host directories** (e.g., `/tmp/avatar_output/input` and `/tmp/avatar_output`)
2. **Activity sets environment variables** (`econ_activity.py` lines 79-80):
   ```python
   env["INPUT_DIR"] = abs_input_dir  # e.g., "/tmp/avatar_output/input"
   env["OUTPUT_DIR"] = abs_output_dir  # e.g., "/tmp/avatar_output"
   ```

3. **Activity builds docker-compose command** (`econ_activity.py` lines 73-74):
   ```python
   "-in_dir", "/app/input",   # Container path (hardcoded)
   "-out_dir", "/app/output", # Container path (hardcoded)
   ```
   These are **container paths** - they're what ECON will see inside the container.

4. **Docker-compose reads `docker-compose.yaml`** and sees:
   ```yaml
   volumes:
     - ${INPUT_DIR}:/app/input:ro  # Maps host dir to container /app/input
     - ${OUTPUT_DIR}:/app/output    # Maps host dir to container /app/output
   ```

5. **Docker-compose mounts volumes** before running the command:
   - Host `/tmp/avatar_output/input` → Container `/app/input`
   - Host `/tmp/avatar_output` → Container `/app/output`
   
   **The container paths `/app/input` and `/app/output` now point to the mounted host directories.**

6. **Command executes** with `-in_dir /app/input -out_dir /app/output`:
   - When ECON reads from `/app/input` (container path), it's actually reading from the mounted host directory
   - When ECON writes to `/app/output` (container path), it's actually writing to the mounted host directory

7. **Results are automatically available on the host** (no copying needed)

**Key Point:** The paths in the command (`/app/input`, `/app/output`) are **not switched out**. They're the **mount points** inside the container. Docker-compose mounts your host directories **to** those container paths, so when the command uses those paths, it's accessing the mounted host directories.

### Alternative: Service-Based File Management

If you prefer a service-based approach, you would need a helper service that:
1. Copies files into the container's filesystem (e.g., using `docker cp` or a shared volume)
2. Triggers ECON processing
3. Copies results back out
4. Cleans up files inside the container

This approach is more complex and less efficient, so volume mounts are recommended.

## Configuration

Update `config.yaml` with the path to your ECON docker-compose.yaml:

```yaml
ECON_PARAMETERS:
  COMPOSE_PATH: /path/to/ECON/docker-compose.yaml
```

Or set the environment variable:
```bash
export ECON_COMPOSE_PATH=/path/to/ECON/docker-compose.yaml
```

## Environment Variables

The activity will set `INPUT_DIR` and `OUTPUT_DIR` environment variables that your docker-compose.yaml can use for volume mounts.

## Testing

To test ECON manually:
```bash
cd /path/to/ECON
docker-compose run --rm econ python -m apps.infer -cfg /app/configs/econ.yaml -in_dir /app/input -out_dir /app/output
```

## File Paths

**Important:** ECON expects files at these paths **inside the container**:
- Input images: `/app/input/` (mounted from host `INPUT_DIR`)
- Output results: `/app/output/` (mounted from host `OUTPUT_DIR`)
- Config files: `/app/configs/econ.yaml` (mounted from host)

The activity passes these paths to ECON:
```bash
python -m apps.infer -cfg /app/configs/econ.yaml -in_dir /app/input -out_dir /app/output
```

## Cleanup

The current implementation keeps input files on the host for debugging. To enable automatic cleanup after processing, uncomment the cleanup code in `econ_activity.py` (lines 223-225):

```python
finally:
    # Clean up temporary input directory
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
```

Or clean up the entire input directory:
```python
finally:
    import shutil
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
```

## Notes

- ECON requires GPU access. Ensure your docker-compose.yaml includes GPU configuration.
- The service name in docker-compose.yaml should match what's used in `econ_activity.py` (default: `econ`).
- Input directory is mounted as read-only (`:ro`) to prevent accidental modification.
- Output directory needs write access for ECON to save results.
- The container path is `/app/input` (not `apps/input`).

