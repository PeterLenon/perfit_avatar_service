# Photorealistic Texture Generation Setup

This guide explains how to set up **photorealistic texture generation** for digital twin avatars that customers can use to try on clothes virtually.

## Overview

For virtual try-on applications, you need **truly photorealistic textures** that accurately represent the customer's appearance. This guide covers publicly available tools that can generate photorealistic textures suitable for production use.

## Recommended Tools (Publicly Available)

### 1. **PSHuman** ⭐ (Recommended)

**Status:** ✅ **Publicly Available** - Best for photorealistic textures

PSHuman generates photorealistic textures from single images using cross-scale multiview diffusion. It produces high-quality UV-mapped textures with excellent identity preservation.

**Features:**
- ✅ Photorealistic texture generation from single image
- ✅ High-resolution UV-mapped textures
- ✅ Normal maps for realistic lighting
- ✅ Strong identity preservation
- ✅ Continuous texture quality
- ✅ Realistic clothing wrinkles and natural poses

**Installation:**
```bash
# Clone PSHuman repository
git clone https://github.com/penghtyx/PSHuman.git
cd PSHuman
# Follow installation instructions in the repository

# Set environment variable
export PSHUMAN_PATH=/path/to/PSHuman
```

**Paper:** https://arxiv.org/abs/2505.12635  
**Project Page:** https://penghtyx.github.io/PSHuman/

**Usage:**
```json
{
  "preferences": {
    "texture_tool": "pshuman"
  }
}
```

**Note:** If PSHuman is not installed, the system automatically falls back to `enhanced` mode.

### 2. **Hunyuan3D-2.1**

**Status:** ✅ **Publicly Available** - Tencent's open-source tool

Hunyuan3D-2.1 generates production-ready 3D assets with full PBR (Physically Based Rendering) materials, including photorealistic textures.

**Features:**
- ✅ Full PBR material generation (albedo, normal, roughness, specular)
- ✅ Production-ready quality
- ✅ Open-source (Tencent)
- ✅ High-resolution textures

**Installation:**
```bash
# Clone Hunyuan3D-2.1 repository
git clone https://github.com/tencent-hunyuan/hunyuan3d-2.1.git
cd hunyuan3d-2.1
# Follow installation instructions

# Set environment variable
export HUNYUAN3D_PATH=/path/to/hunyuan3d-2.1
```

**GitHub:** https://github.com/tencent-hunyuan/hunyuan3d-2.1

**Usage:**
```json
{
  "preferences": {
    "texture_tool": "hunyuan3d"
  }
}
```

### 3. **TexDreamer**

**Status:** ✅ **Publicly Available** - Zero-shot high-fidelity textures

TexDreamer generates high-fidelity human textures from text or image input with semantic UV map support.

**Features:**
- ✅ Zero-shot texture generation
- ✅ Text or image input
- ✅ Semantic UV map support
- ✅ High-resolution textures (1024×1024+)
- ✅ Style control via prompts

**Installation:**
```bash
# Clone TexDreamer repository
git clone https://github.com/ggxxii/TexDreamer.git
cd TexDreamer
# Follow installation instructions

# Set environment variable
export TEXDREAMER_PATH=/path/to/TexDreamer
```

**Project Page:** https://ggxxii.github.io/texdreamer/

**Usage:**
```json
{
  "preferences": {
    "texture_tool": "texdreamer"
  }
}
```

## Configuration

### Default Configuration (PSHuman)

```yaml
# config.yaml
TEXTURE_GENERATION_PARAMETERS:
  TOOL: pshuman  # Recommended for photorealistic textures
  PSHUMAN_PATH: /path/to/PSHuman  # Optional, falls back to enhanced if not set
```

### Environment Variables

```bash
# Set paths to installed tools
export PSHUMAN_PATH=/path/to/PSHuman
export HUNYUAN3D_PATH=/path/to/hunyuan3d-2.1
export TEXDREAMER_PATH=/path/to/TexDreamer
```

### API Request

```json
{
  "user_photo": "...",
  "user_id": "customer123",
  "preferences": {
    "texture_tool": "pshuman"  // Recommended for photorealistic textures
  }
}
```

## Automatic Fallback

The system includes intelligent fallback:

1. **First choice:** PSHuman (if installed)
2. **Fallback:** Enhanced texture projection (if PSHuman not found)
3. **Final fallback:** Simple texture projection

This ensures the service always works, even if photorealistic tools aren't installed.

## Comparison

| Tool | Quality | Setup | Speed | Best For |
|------|---------|-------|-------|----------|
| **PSHuman** | ⭐⭐⭐⭐⭐ | Medium | Medium | Photorealistic textures |
| **Hunyuan3D-2.1** | ⭐⭐⭐⭐⭐ | Medium | Medium | PBR materials |
| **TexDreamer** | ⭐⭐⭐⭐ | Medium | Medium | Style control |
| **Enhanced** | ⭐⭐⭐ | Easy | Fast | Good quality without setup |
| **Simple** | ⭐⭐ | Easy | Fast | Basic textures |

## Requirements

### Hardware

- **GPU:** Required for PSHuman, Hunyuan3D, and TexDreamer
- **Memory:** 8GB+ VRAM recommended
- **Storage:** 10GB+ for models and dependencies

### Software

- Python 3.8+
- CUDA (for GPU acceleration)
- PyTorch or TensorFlow (depending on tool)

## Integration Workflow

1. **Install chosen tool** (PSHuman recommended)
2. **Set environment variable** (`PSHUMAN_PATH`, etc.)
3. **Test with sample image** to verify installation
4. **Use in production** - system will automatically use the tool

## Troubleshooting

### Issue: Tool not found

**Solution:** 
- Verify installation path
- Check environment variable is set correctly
- System will automatically fall back to `enhanced` mode

### Issue: GPU out of memory

**Solution:**
- Reduce input image resolution
- Use smaller model variants if available
- Fall back to `enhanced` mode for lower memory usage

### Issue: Slow generation

**Solution:**
- This is normal for photorealistic generation (30-60 seconds)
- Consider caching results
- Use `enhanced` mode for faster generation (lower quality)

## Production Recommendations

For production virtual try-on:

1. **Start with PSHuman** - Best balance of quality and availability
2. **Cache generated textures** - Avoid regenerating for same user
3. **Use CDN** - Serve textures from CDN for fast delivery
4. **Monitor quality** - Collect user feedback on texture quality
5. **Fallback strategy** - Always have `enhanced` mode as backup

## Next Steps

1. **Install PSHuman** (recommended):
   ```bash
   git clone https://github.com/penghtyx/PSHuman.git
   cd PSHuman
   # Follow README for installation
   export PSHUMAN_PATH=$(pwd)
   ```

2. **Test installation:**
   ```bash
   # The system will automatically detect and use PSHuman
   # Make a test API call with texture_tool: "pshuman"
   ```

3. **Verify output quality** - Check generated textures are photorealistic

4. **Deploy to production** - System will use PSHuman when available

## Summary

- ✅ **PSHuman** is recommended for photorealistic textures
- ✅ Automatic fallback ensures service always works
- ✅ Multiple options available (PSHuman, Hunyuan3D, TexDreamer)
- ✅ Production-ready with proper setup

The service now supports truly photorealistic texture generation suitable for digital twin avatars in virtual try-on applications.
