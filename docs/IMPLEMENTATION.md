# Perfit Avatar Service - Implementation Guide

## Project Structure

```
perfit_avatar_service/
├── app/                          # Main application
│   ├── main.py                   # FastAPI app factory & lifespan
│   ├── config.py                 # Settings (pydantic-settings)
│   ├── api/
│   │   └── routes/
│   │       ├── health.py         # Health check endpoints
│   │       └── avatar.py         # Avatar CRUD endpoints
│   ├── models/
│   │   ├── database.py           # SQLAlchemy ORM models
│   │   └── schemas.py            # Pydantic request/response schemas
│   ├── services/
│   │   ├── image_validator.py    # Image validation with face detection
│   │   └── storage.py            # S3/MinIO storage operations
│   └── workers/
│       └── extraction_worker.py  # RQ background job
├── ml/                           # Machine learning modules
│   ├── hmr2/
│   │   ├── __init__.py
│   │   └── inference.py          # HMR2 body reconstruction
│   ├── measurements/
│   │   ├── __init__.py
│   │   └── extractor.py          # Measurement extraction
│   ├── smpl/
│   │   └── models/               # SMPL model files (manual download)
│   └── vendor/
│       └── 4D-Humans/            # Git submodule
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   └── docker-compose.yaml
├── alembic/                      # Database migrations
├── tests/                        # Test suite
├── docs/                         # Documentation
├── requirements.txt
└── pyproject.toml
```

---

## Core Components

### 1. Configuration (`app/config.py`)

Uses `pydantic-settings` for type-safe configuration from environment variables.

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Nested settings
    database: DatabaseSettings
    redis: RedisSettings
    minio: MinioSettings
    ml: MLSettings
```

**Key Settings**:

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `ml.hmr2_device` | `ML_HMR2_DEVICE` | `cuda` | GPU or CPU inference |
| `ml.detection_threshold` | `ML_DETECTION_THRESHOLD` | `0.5` | Person detection confidence |
| `ml.smpl_model_path` | `ML_SMPL_MODEL_PATH` | `./ml/smpl/models` | Path to SMPL .pkl files |

**Usage**:
```python
from app.config import get_settings
settings = get_settings()  # Cached singleton
```

---

### 2. Database Models (`app/models/database.py`)

SQLAlchemy 2.0 async models with PostgreSQL.

**Engine Setup**:
```python
engine = create_async_engine(settings.database.async_url)
async_session = async_sessionmaker(engine, class_=AsyncSession)
```

**Avatar Model** (key fields):
```python
class Avatar(Base):
    __tablename__ = "avatars"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    gender: Mapped[str] = mapped_column(Enum(Gender))

    # SMPL parameters stored as JSON
    smplx_betas: Mapped[dict] = mapped_column(JSON)
    smplx_body_pose: Mapped[dict] = mapped_column(JSON)
    smplx_global_orient: Mapped[dict] = mapped_column(JSON)

    # Measurements in centimeters
    height_cm: Mapped[float]
    chest_circumference_cm: Mapped[float]
    # ... other measurements
```

**Session Management**:
```python
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session
```

---

### 3. API Routes (`app/api/routes/avatar.py`)

**Create Avatar** (`POST /api/v1/avatar`):
```python
@router.post("/avatar", status_code=202)
async def create_avatar(
    request: AvatarCreateRequest,
    session: AsyncSession = Depends(get_session),
) -> JobResponse:
    # 1. Validate image
    validation = await image_validator.validate(request.image_base64)
    if not validation.is_valid:
        raise HTTPException(400, validation.errors)

    # 2. Create job record
    job = ExtractionJob(user_id=request.user_id, gender=request.gender)
    session.add(job)
    await session.commit()

    # 3. Enqueue to Redis
    rq_job = queue.enqueue(
        extract_body_shape,
        job_id=str(job.id),
        user_id=request.user_id,
        image_base64=request.image_base64,
        gender=request.gender,
    )

    return JobResponse(job_id=job.id, status="pending")
```

**Poll Job Status** (`GET /api/v1/avatar/job/{job_id}`):
```python
@router.get("/avatar/job/{job_id}")
async def get_job_status(job_id: UUID) -> JobStatusResponse:
    job = await session.get(ExtractionJob, job_id)
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        progress_percent=_calculate_progress(job.status),
        avatar_id=job.avatar_id,
    )
```

---

### 4. Image Validation (`app/services/image_validator.py`)

Validates images before processing:

1. **Base64 Decoding**: Handles data URL prefix
2. **Format Check**: JPEG, PNG, WebP only
3. **Size Check**: 512x512 min, 4096x4096 max, 10MB max
4. **Face Detection**: MediaPipe face detection

```python
class ImageValidator:
    def __init__(self):
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5
        )

    async def validate(self, image_base64: str) -> ImageValidationResult:
        # Decode
        image_data = self._decode_base64(image_base64)
        image = Image.open(io.BytesIO(image_data))

        # Check format
        if image.format.lower() not in self.allowed_formats:
            return ImageValidationResult(is_valid=False, errors=["Invalid format"])

        # Check dimensions
        if image.width < 512 or image.height < 512:
            return ImageValidationResult(is_valid=False, errors=["Image too small"])

        # Face detection
        results = self.face_detector.process(np.array(image))
        if not results.detections:
            return ImageValidationResult(is_valid=False, errors=["No face detected"])

        return ImageValidationResult(is_valid=True)
```

---

### 5. Storage Service (`app/services/storage.py`)

S3-compatible storage operations using boto3.

**Upload SMPL Parameters**:
```python
def upload_smplx_params(self, user_id: str, avatar_id: str, params: dict) -> str:
    key = f"avatars/{user_id}/{avatar_id}/smplx_params.npz"

    buffer = io.BytesIO()
    np_params = {k: np.array(v) for k, v in params.items()}
    np.savez_compressed(buffer, **np_params)
    buffer.seek(0)

    self.client.upload_fileobj(buffer, self.bucket, key)
    return self._get_url(key)
```

**Upload Mesh**:
```python
def upload_mesh(self, user_id: str, avatar_id: str, vertices: np.ndarray, faces: np.ndarray) -> str:
    key = f"avatars/{user_id}/{avatar_id}/mesh.obj"
    obj_content = self._generate_obj(vertices, faces)
    self.client.put_object(Bucket=self.bucket, Key=key, Body=obj_content)
    return self._get_url(key)
```

---

### 6. HMR2 Inference (`ml/hmr2/inference.py`)

Wraps 4D-Humans for body reconstruction.

**Model Loading** (singleton pattern):
```python
_hmr2_model = None
_detector = None

class HMR2Inference:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self._ensure_models_loaded()

    def _ensure_models_loaded(self):
        global _hmr2_model, _detector
        if _hmr2_model is not None:
            return

        # Download and load HMR2
        from hmr2.models import download_models, load_hmr2
        download_models(CACHE_DIR_4DHUMANS)
        _hmr2_model, _ = load_hmr2()
        _hmr2_model = _hmr2_model.to(self.device).eval()

        # Load ViTDet detector
        _detector = self._load_detector()
```

**Inference**:
```python
def predict(self, image: Image.Image, gender: str = "neutral") -> dict:
    # Convert to OpenCV format
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect people
    det_out = _detector(img_cv2)
    boxes = det_out["instances"].pred_boxes.tensor[valid_idx].cpu().numpy()

    if len(boxes) == 0:
        raise ValueError("No person detected")

    # Use largest bounding box
    if len(boxes) > 1:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxes = boxes[np.argmax(areas):np.argmax(areas)+1]

    # Run HMR2
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
    with torch.no_grad():
        output = _hmr2_model(batch)

    # Convert rotation matrices to axis-angle
    return {
        "betas": output["pred_smpl_params"]["betas"][0].cpu().numpy(),
        "body_pose": self._rotmat_to_axis_angle(body_pose_rotmat).flatten(),
        "global_orient": self._rotmat_to_axis_angle(global_orient_rotmat).flatten(),
        "vertices": output["pred_vertices"][0].cpu().numpy(),
    }
```

---

### 7. Measurement Extraction (`ml/measurements/extractor.py`)

Extracts body measurements from SMPL parameters.

**T-Pose Mesh Generation**:
```python
def _generate_tpose_mesh(self, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    betas_tensor = torch.tensor(betas).unsqueeze(0)
    body_pose = torch.zeros(1, 69)  # Zero pose = T-pose
    global_orient = torch.zeros(1, 3)

    output = _smpl_model(
        betas=betas_tensor,
        body_pose=body_pose,
        global_orient=global_orient,
    )

    return output.vertices[0].numpy(), output.joints[0].numpy()
```

**Height Measurement**:
```python
def _measure_height(self, vertices: np.ndarray) -> float:
    return float(vertices[:, 1].max() - vertices[:, 1].min())
```

**Circumference Measurement**:
```python
def _measure_circumference_at_height(self, vertices: np.ndarray, height: float) -> float:
    # Find vertices near slice plane
    tolerance = 2.0  # cm
    mask = np.abs(vertices[:, 1] - height) < tolerance
    slice_vertices = vertices[mask]

    # Project to XZ plane
    points_2d = slice_vertices[:, [0, 2]]

    # Compute convex hull perimeter
    hull = ConvexHull(points_2d)
    perimeter = 0.0
    for i in range(len(hull.vertices)):
        p1 = points_2d[hull.vertices[i]]
        p2 = points_2d[hull.vertices[(i + 1) % len(hull.vertices)]]
        perimeter += np.linalg.norm(p2 - p1)

    return perimeter
```

**Measurement Heights** (as fraction of total height):
```python
CHEST_HEIGHT_FRACTION = 0.72   # 72% from floor
WAIST_HEIGHT_FRACTION = 0.58   # 58% from floor
HIP_HEIGHT_FRACTION = 0.52     # 52% from floor
NECK_HEIGHT_FRACTION = 0.85    # 85% from floor
THIGH_HEIGHT_FRACTION = 0.45   # 45% from floor
```

---

### 8. Extraction Worker (`app/workers/extraction_worker.py`)

RQ job function that orchestrates the ML pipeline.

```python
def extract_body_shape(job_id: str, user_id: str, image_base64: str, gender: str) -> dict:
    # Update job status
    job.status = JobStatus.PROCESSING
    db.commit()

    # Decode image
    image_data = _decode_base64(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # HMR2 inference
    hmr2 = HMR2Inference(device=settings.ml.hmr2_device)
    smpl_output = hmr2.predict(image, gender=gender)

    # Extract measurements
    extractor = MeasurementExtractor(settings.ml.smpl_model_path, gender=gender)
    measurements = extractor.extract(
        betas=smpl_output["betas"],
        body_pose=smpl_output["body_pose"],
        global_orient=smpl_output["global_orient"],
    )

    # Upload to storage
    storage = StorageService(settings.minio)
    smplx_params_url = storage.upload_smplx_params(user_id, avatar_id, {...})
    mesh_url = storage.upload_mesh(user_id, avatar_id, vertices, faces)
    source_image_url = storage.upload_source_image(user_id, avatar_id, image_data)

    # Create avatar record
    avatar = Avatar(
        user_id=user_id,
        gender=gender,
        smplx_betas={"values": smpl_output["betas"].tolist()},
        height_cm=measurements["height_cm"],
        # ... other measurements
        smplx_params_url=smplx_params_url,
        mesh_url=mesh_url,
        source_image_url=source_image_url,
    )
    db.add(avatar)

    # Mark job complete
    job.status = JobStatus.COMPLETED
    job.avatar_id = avatar.id
    db.commit()

    return {"status": "success", "avatar_id": str(avatar.id)}
```

---

## Docker Setup

### Dockerfile.api

Key points:
- Multi-stage build (builder + runtime)
- Install detectron2 after torch with `--no-build-isolation`
- Non-root user for security
- Health check endpoint

```dockerfile
# Builder stage
FROM python:3.11-slim as builder
RUN pip install -r requirements.txt && \
    pip install "detectron2 @ git+..." --no-build-isolation

# Runtime stage
FROM python:3.11-slim as runtime
COPY --from=builder /opt/venv /opt/venv
USER appuser
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yaml

Services:
- `postgres`: Database
- `redis`: Job queue
- `minio`: Object storage
- `minio-init`: Bucket initialization
- `api`: FastAPI server
- `worker`: RQ worker

Environment variables are set per-service to use Docker service names (e.g., `DB_HOST: postgres`).

---

## Running Locally

### Without Docker

```bash
# 1. Install dependencies
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install "detectron2 @ git+https://github.com/facebookresearch/detectron2.git" --no-build-isolation

# 2. Download SMPL models
# Register at https://smpl.is.tue.mpg.de/
# Extract to ml/smpl/models/

# 3. Start infrastructure
docker-compose -f docker/docker-compose.yaml up -d postgres redis minio minio-init

# 4. Run API
uvicorn app.main:app --reload

# 5. Run worker (separate terminal)
rq worker avatar_extraction --url redis://localhost:6379/0
```

### With Docker

```bash
# Build and run everything
cd docker
docker-compose up --build

# Or just infrastructure + local dev
docker-compose up -d postgres redis minio minio-init
```

---

## Testing

### Manual API Test

```bash
# Create avatar
curl -X POST http://localhost:8000/api/v1/avatar \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-user",
    "image_base64": "<base64_encoded_image>",
    "gender": "male"
  }'

# Response: {"job_id": "abc-123", "status": "pending", ...}

# Poll status
curl http://localhost:8000/api/v1/avatar/job/abc-123

# Get avatar
curl http://localhost:8000/api/v1/avatar/test-user
```

### Unit Tests

```bash
pytest tests/ -v
```

---

## Common Issues

### 1. "No module named 'torch'" during detectron2 install

**Solution**: Install torch first, then detectron2 with `--no-build-isolation`

### 2. "SMPL model not found"

**Solution**: Download SMPL models from https://smpl.is.tue.mpg.de/ and place in `ml/smpl/models/`

### 3. "Connection refused" to PostgreSQL in Docker

**Solution**: Ensure `DB_HOST` is set to `postgres` (Docker service name), not `localhost`

### 4. CUDA out of memory

**Solution**: Set `ML_HMR2_DEVICE=cpu` or reduce batch size

### 5. Model download stalls

**Solution**: HMR2 models auto-download to `~/.cache/4DHumans/`. Ensure network access and disk space (~2GB).

---

## Performance Notes

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| Image validation | ~50ms | Face detection is main cost |
| HMR2 inference | ~100ms (GPU) / ~2s (CPU) | First run slower (model load) |
| Measurement extraction | ~50ms | T-pose generation + calculations |
| Storage upload | ~200ms | Depends on network/file size |
| **Total job time** | ~500ms (GPU) / ~3s (CPU) | Excludes queue wait |

---

## Key Files Quick Reference

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app entry point |
| `app/config.py` | All configuration settings |
| `app/api/routes/avatar.py` | Avatar API endpoints |
| `app/workers/extraction_worker.py` | Main ML job function |
| `ml/hmr2/inference.py` | HMR2 body reconstruction |
| `ml/measurements/extractor.py` | Body measurement algorithms |
| `app/services/storage.py` | S3/MinIO operations |
| `docker/docker-compose.yaml` | Full stack deployment |
