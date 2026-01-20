# Avatar Creation Pipeline Implementation Plan

## Overview

Implement the ML pipeline for avatar creation using HMR 2.0 (4D-Humans) for body reconstruction and custom measurement extraction from SMPL meshes.

**Key Decision**: HMR 2.0 outputs **SMPL** parameters (not SMPL-X). Since SMPL-X conversion is complex and SMPL is sufficient for body measurements, we'll use SMPL directly but keep the existing `smplx_*` field names for API compatibility.

---

## Architecture

```
POST /avatar (image_base64)
     │
     ▼
┌─────────────────────────────────────┐
│  Image Validation (existing)        │
│  - Face detection via MediaPipe     │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  RQ Worker                          │
│  ┌───────────────────────────────┐  │
│  │ 1. HMR2Inference              │  │
│  │    - ViTDet person detection  │  │
│  │    - HMR2 body estimation     │  │
│  │    → betas, body_pose, verts  │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ 2. MeasurementExtractor       │  │
│  │    - SMPL mesh from params    │  │
│  │    - 9 body measurements      │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ 3. Storage Upload             │  │
│  │    - SMPL params (.npz)       │  │
│  │    - Mesh (.obj)              │  │
│  │    - Source image             │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
     │
     ▼
Avatar record in DB with measurements
```

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `ml/hmr2/__init__.py` | Module exports |
| `ml/hmr2/inference.py` | HMR2 model loading and inference |
| `ml/hmr2/detector.py` | ViTDet person detection wrapper |
| `ml/measurements/extractor.py` | Measurement extraction from SMPL mesh |

### Modified Files

| File | Changes |
|------|---------|
| `app/workers/extraction_worker.py` | Replace placeholders with real ML pipeline |
| `app/config.py` | Update `MLSettings` for HMR2 (remove PIXIE) |
| `requirements.txt` | Add HMR2/detectron2 dependencies |

---

## Implementation Details

### 1. HMR2 Inference Module (`ml/hmr2/`)

**`ml/hmr2/inference.py`**:
```python
# Singleton pattern for model caching (one model per worker process)
_model = None
_detector = None

class HMR2Inference:
    """HMR 2.0 body reconstruction from single image."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self._ensure_models_loaded()

    def predict(self, image: PIL.Image, gender: str = "neutral") -> dict:
        """
        Run body reconstruction on image.

        Returns:
            {
                "betas": np.ndarray (10,),
                "body_pose": np.ndarray (69,),  # 23 joints * 3
                "global_orient": np.ndarray (3,),
                "vertices": np.ndarray (6890, 3),
                "joints": np.ndarray (24, 3),
            }
        """
```

**Detection Strategy**:
- Use ViTDet to detect people in image
- Select largest/most confident person bounding box
- Crop and run HMR2 on the detected region
- Return SMPL parameters

**Model Caching**:
- Module-level singletons `_model` and `_detector`
- Loaded once per worker process on first call
- Auto-download checkpoints to `~/.cache/4DHumans/`

### 2. Measurement Extraction (`ml/measurements/`)

**`ml/measurements/extractor.py`**:
```python
class MeasurementExtractor:
    """Extract body measurements from SMPL parameters."""

    def __init__(self, smpl_model_path: str, gender: str = "neutral"):
        self.body_model = smplx.create(smpl_model_path, model_type='smpl', gender=gender)

    def extract(self, betas: np.ndarray, body_pose: np.ndarray,
                global_orient: np.ndarray) -> dict[str, float]:
        """
        Extract 9 body measurements in centimeters.

        Uses T-pose (zeroed pose params) for consistent measurements.
        """
```

**Measurement Algorithms**:

| Measurement | Method |
|-------------|--------|
| Height | Distance from head vertex to floor plane |
| Chest circumference | Horizontal slice at armpit level → polygon perimeter |
| Waist circumference | Horizontal slice at narrowest torso → polygon perimeter |
| Hip circumference | Horizontal slice at widest hip → polygon perimeter |
| Inseam | Distance from crotch vertex to ankle |
| Shoulder width | Distance between left/right shoulder joints |
| Arm length | Distance from shoulder to wrist joints |
| Thigh circumference | Horizontal slice at upper thigh → polygon perimeter |
| Neck circumference | Horizontal slice at neck base → polygon perimeter |

**Key Vertices/Joints** (SMPL topology):
- Head top: vertex ~411
- Shoulders: joints 16 (left), 17 (right)
- Wrists: joints 20 (left), 21 (right)
- Ankles: joints 7 (left), 8 (right)
- Crotch region: vertices ~3500

### 3. Worker Integration

**`app/workers/extraction_worker.py`** changes:

```python
def extract_body_shape(job_id, user_id, image_base64, gender):
    # ... existing setup code ...

    # NEW: Run HMR2 inference
    from ml.hmr2 import HMR2Inference
    hmr2 = HMR2Inference(device=settings.ml.hmr2_device)
    smpl_output = hmr2.predict(image, gender=gender)

    # NEW: Extract measurements
    from ml.measurements import MeasurementExtractor
    extractor = MeasurementExtractor(settings.ml.smpl_model_path, gender=gender)
    measurements = extractor.extract(
        betas=smpl_output["betas"],
        body_pose=smpl_output["body_pose"],
        global_orient=smpl_output["global_orient"],
    )

    # NEW: Upload to storage
    storage = StorageService(settings.minio)
    smplx_params_url = storage.upload_smplx_params(user_id, str(avatar.id), {
        "betas": smpl_output["betas"],
        "body_pose": smpl_output["body_pose"],
        "global_orient": smpl_output["global_orient"],
    })
    mesh_url = storage.upload_mesh(
        user_id, str(avatar.id),
        smpl_output["vertices"],
        body_model.faces,
    )
    source_image_url = storage.upload_source_image(user_id, str(avatar.id), image_data)

    # ... existing avatar creation code ...
```

### 4. Configuration Updates

**`app/config.py`** - Replace PIXIE settings with HMR2:

```python
class MLSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ML_")

    # HMR2 configuration
    hmr2_checkpoint_path: str = "~/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
    hmr2_device: str = "cuda"  # or "cpu"

    # SMPL model path
    smpl_model_path: str = ""

    # Inference settings
    batch_size: int = 1
    detection_threshold: float = 0.5  # ViTDet confidence threshold
```

### 5. Dependencies

**Add to `requirements.txt`**:
```
# HMR2 / 4D-Humans
detectron2 @ git+https://github.com/facebookresearch/detectron2.git
pytorch-lightning>=2.0.0
timm>=0.9.0
einops>=0.7.0

# SMPL model
smplx>=0.1.28  # (already present)
```

**4D-Humans Integration**: Vendor as git submodule at `ml/vendor/4D-Humans/`

```bash
# Add submodule
git submodule add https://github.com/shubham-goel/4D-Humans.git ml/vendor/4D-Humans
git submodule update --init --recursive
```

Then import from `ml.vendor.4D-Humans.hmr2`.

---

## Model Files Required

1. **HMR2 checkpoint**: Auto-downloads to `~/.cache/4DHumans/` on first run
2. **ViTDet weights**: Auto-downloads via detectron2 model zoo
3. **SMPL model**: Manual download required (user will register and download)
   - Register at [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/)
   - Download "SMPL for Python" package
   - Extract to `./ml/smpl/models/`:
     ```
     ml/smpl/models/
     ├── SMPL_MALE.pkl
     ├── SMPL_FEMALE.pkl
     └── SMPL_NEUTRAL.pkl
     ```

---

## Error Handling

- **No person detected**: Return error "No person detected in image"
- **Multiple people**: Use largest bounding box (most prominent person)
- **Model loading failure**: Fail fast with clear error message
- **CUDA OOM**: Fall back to CPU or return error

---

## Verification Plan

1. **Unit tests**:
   - `test_hmr2_inference.py`: Test model loading, inference output shapes
   - `test_measurement_extractor.py`: Test measurement algorithms with known inputs

2. **Integration test**:
   - Upload test image via API
   - Poll job until complete
   - Verify avatar has measurements within expected ranges
   - Verify storage URLs are accessible

3. **Manual verification**:
   ```bash
   # Start services
   docker-compose -f docker/docker-compose.yaml up -d

   # Create avatar
   curl -X POST http://localhost:8000/api/v1/avatar \
     -H "Content-Type: application/json" \
     -d '{"user_id": "test-user", "image_base64": "<base64_image>", "gender": "male"}'

   # Poll status
   curl http://localhost:8000/api/v1/avatar/job/{job_id}

   # Get avatar
   curl http://localhost:8000/api/v1/avatar/test-user
   ```

---

## Implementation Order

1. **Setup** (prerequisites):
   - [ ] Add 4D-Humans as git submodule: `git submodule add https://github.com/shubham-goel/4D-Humans.git ml/vendor/4D-Humans`
   - [ ] Update `requirements.txt` with detectron2 and other dependencies
   - [ ] User downloads SMPL models to `ml/smpl/models/`

2. **ML modules** (can test independently):
   - [ ] `ml/hmr2/__init__.py` and `ml/hmr2/inference.py` - HMR2 wrapper
   - [ ] `ml/measurements/__init__.py` and `ml/measurements/extractor.py` - Measurement extraction

3. **Configuration**:
   - [ ] Update `app/config.py` with HMR2 settings (replace PIXIE)

4. **Worker integration**:
   - [ ] Update `extraction_worker.py` to use real ML pipeline
   - [ ] Wire up storage uploads

5. **Testing**:
   - [ ] Unit tests for ML modules
   - [ ] Integration test for full pipeline

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Large model download (>2GB) | Auto-download on first run, cache persistently |
| GPU memory constraints | Support CPU fallback, configurable batch size |
| SMPL license restrictions | Document manual download requirement |
| Measurement accuracy | Use T-pose for consistency, validate against known measurements |
