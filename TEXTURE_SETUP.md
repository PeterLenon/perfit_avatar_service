# Texture Generation Setup Guide

This guide explains how to configure texture generation for the avatar service.

**For Virtual Try-On Applications:** See [VIRTUAL_TRYON_SETUP.md](./VIRTUAL_TRYON_SETUP.md) for specific guidance on creating photorealistic textures for digital twin avatars that customers can use to try on clothes.

## Overview

After ECON generates the initial 3D mesh and SMPL-X parameters, the workflow can optionally generate textures for the avatar. Since PHORHUM is not publicly available, this guide covers alternative options.

## Available Texture Generation Options

### 1. **Simple Texture Projection (Default, Recommended)**

**Status:** ✅ **Available Now** - No external tools required

The `simple` mode creates a basic texture map by resizing the input image to a standard texture size (1024x1024). This is a fallback method that works immediately without any external dependencies.

**Pros:**
- ✅ Works out of the box - no setup required
- ✅ Fast and lightweight
- ✅ No external tools needed
- ✅ Good for basic texture mapping

**Cons:**
- ⚠️ Less sophisticated than specialized tools
- ⚠️ Doesn't handle lighting or complex surface details

**Usage:**
```json
{
  "preferences": {
    "texture_tool": "simple"
  }
}
```

### 2. **Skip Texture Generation**

**Status:** ✅ **Available Now**

Skip texture generation entirely and use the raw ECON mesh output.

**Usage:**
```json
{
  "preferences": {
    "texture_tool": "skip"
  }
}
```

### 3. **PHORHUM** (Not Publicly Available)

**Status:** ❌ **Not Available** - PHORHUM is not publicly available on GitHub

If you have access to PHORHUM through other means, you can configure it as described below.

### 4. **MORPHX** (If Available)

**Status:** ⚠️ **May Not Be Available** - Check if MORPHX is publicly available

If MORPHX is available, you can configure it as described below.

### 5. **Alternative Open-Source Tools**

Consider these publicly available alternatives:

- **TexGen** (ECCV 2024): Text-guided texture generation
  - GitHub: Check for `dong-huo/TexGen` or similar
  - Requires: Text prompts, multi-view sampling

- **LumiTex**: High-fidelity PBR texture generation
  - Check: `lumitex-pbr` repositories

- **HRDreamer**: Diffusion-based high-res textures
  - Check: Research paper implementations

- **Simple UV Mapping**: Use Blender or similar tools for manual texture mapping

## Quick Start (Recommended)

**For immediate use without external tools**, use the `simple` texture mode:

```yaml
# config.yaml
TEXTURE_GENERATION_PARAMETERS:
  TOOL: simple  # Uses built-in simple texture projection
```

Or in your API request:
```json
{
  "preferences": {
    "texture_tool": "simple"
  }
}
```

This will automatically create a texture map from your input image - no setup required!

## Advanced Setup (Optional External Tools)

If you want to use external texture generation tools (PHORHUM, MORPHX, or alternatives):

1. Clone and set up your chosen tool (if publicly available)
2. Create a docker-compose.yaml for the tool
3. Configure paths in `config.yaml`

## Docker Compose Configuration

### PHORHUM docker-compose.yaml Example

```yaml
version: '3.8'

services:
  phorhum:
    build: .
    # or use: image: phorhum:latest
    volumes:
      - ${INPUT_DIR:-./input}:/app/input:ro
      - ${OUTPUT_DIR:-./output}:/app/output
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

### MORPHX docker-compose.yaml Example

```yaml
version: '3.8'

services:
  morphx:
    build: .
    # or use: image: morphx:latest
    volumes:
      - ${INPUT_DIR:-./input}:/app/input:ro
      - ${OUTPUT_DIR:-./output}:/app/output
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

## Configuration

### Default Configuration (Simple Mode)

```yaml
TEXTURE_GENERATION_PARAMETERS:
  TOOL: simple  # Built-in simple texture projection
```

### External Tool Configuration

If you have access to PHORHUM, MORPHX, or other tools:

```yaml
TEXTURE_GENERATION_PARAMETERS:
  TOOL: phorhum  # "phorhum", "morphx", "simple", or "skip"
  PHORHUM_COMPOSE_PATH: /path/to/phorhum/docker-compose.yaml
  MORPHX_COMPOSE_PATH: /path/to/morphx/docker-compose.yaml
```

Or set environment variables:
```bash
export PHORHUM_COMPOSE_PATH=/path/to/phorhum/docker-compose.yaml
export MORPHX_COMPOSE_PATH=/path/to/morphx/docker-compose.yaml
```

### Runtime Selection

You can also specify the texture tool per request:

```json
{
  "user_photo": "...",
  "user_id": "user123",
  "preferences": {
    "texture_tool": "simple"  // or "skip", "phorhum", "morphx"
  }
}
```

## Workflow Integration

The texture generation step runs automatically after ECON in the workflow:

1. **ECON** generates mesh + SMPL-X parameters
2. **Texture Generation** (optional) - generates textures using the selected method
3. **S3 Upload** stores all assets

### Texture Tool Options

- **`"simple"`** (default): Built-in simple texture projection - works immediately
- **`"skip"`**: Skip texture generation, use raw ECON mesh
- **`"phorhum"`**: Use PHORHUM (if available)
- **`"morphx"`**: Use MORPHX (if available)

Example request:
```json
{
  "user_photo": "...",
  "user_id": "user123",
  "preferences": {
    "texture_tool": "simple"  // Recommended: works without external tools
  }
}
```

## Command Format

The activity expects the texture tool to accept:
- Input image: `/app/input_image.jpg`
- Input mesh: `/app/input_mesh.obj`
- Output directory: `/app/output`

Adjust the commands in `texture_generation_activity.py` if your tool uses different parameters.

### PHORHUM Command
```bash
python -m phorhum.infer --input_image /app/input_image.jpg --mesh /app/input_mesh.obj --output /app/output
```

### MORPHX Command
```bash
python -m morphx.infer --image /app/input_image.jpg --mesh /app/input_mesh.obj --output /app/output
```

## Expected Outputs

The texture generation tool should produce:
- **Textured mesh** (`.obj` file with texture coordinates)
- **Texture map** (`*texture*.png`)
- **Diffuse map** (`*diffuse*.png`, optional)
- **Normal map** (`*normal*.png`, optional)

## Testing

To test PHORHUM manually:
```bash
cd /path/to/phorhum
docker-compose run --rm phorhum python -m phorhum.infer --input_image /app/input_image.jpg --mesh /app/input_mesh.obj --output /app/output
```

To test MORPHX manually:
```bash
cd /path/to/morphx
docker-compose run --rm morphx python -m morphx.infer --image /app/input_image.jpg --mesh /app/input_mesh.obj --output /app/output
```

## Notes

- Both tools require GPU access. Ensure your docker-compose.yaml includes GPU configuration.
- The service name in docker-compose.yaml should match what's used in `texture_generation_activity.py`:
  - PHORHUM: `phorhum`
  - MORPHX: `morphx`
- Input directory is mounted as read-only (`:ro`).
- Output directory needs write access.
- Processing can take 10-30 minutes depending on image resolution and hardware.

## Troubleshooting

1. **Service name mismatch**: Update the service name in `texture_generation_activity.py` to match your docker-compose.yaml.

2. **Command format**: Adjust the Python module path and arguments in `_run_phorhum()` or `_run_morphx()` functions to match your tool's actual API.

3. **Output location**: The activity searches for outputs in `{output_dir}/phorhum_output` or `{output_dir}/morphx_output`. Adjust the search paths if your tool outputs to a different location.

4. **File naming**: The activity looks for files matching patterns like `*texture*.png`, `*diffuse*.png`, `*normal*.png`. Adjust the glob patterns if your tool uses different naming conventions.

