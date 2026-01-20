# Perfit Avatar Service - Design Document

## Overview

The Perfit Avatar Service is a digital twin creation system that generates 3D body models from user photos. It extracts accurate body measurements for virtual clothing try-on and fit prediction applications.

## System Goals

1. **Accuracy**: Extract body measurements within acceptable tolerances for clothing fit
2. **Scalability**: Handle concurrent requests via async processing and job queues
3. **Reliability**: Graceful failure handling with job status tracking
4. **Maintainability**: Clean separation of concerns between API, ML, and storage

---

## Architecture

### High-Level Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Client App    │────▶│   FastAPI API   │────▶│   Redis Queue   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   PostgreSQL    │     │   RQ Worker     │
                        │   (Jobs/Avatars)│     │   (ML Pipeline) │
                        └─────────────────┘     └─────────────────┘
                                                       │
                               ┌───────────────────────┼───────────────────────┐
                               ▼                       ▼                       ▼
                        ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
                        │   HMR2      │         │ Measurement │         │   MinIO     │
                        │  Inference  │         │  Extractor  │         │  (Storage)  │
                        └─────────────┘         └─────────────┘         └─────────────┘
```

### Request Flow

1. **Client** uploads base64-encoded photo via `POST /api/v1/avatar`
2. **API Server** validates image (format, size, face detection)
3. **API Server** creates `ExtractionJob` record (status=pending)
4. **API Server** enqueues job to Redis and returns `job_id` immediately (202 Accepted)
5. **RQ Worker** picks up job, updates status to `processing`
6. **Worker** runs HMR2 inference → SMPL body parameters
7. **Worker** extracts measurements from SMPL mesh
8. **Worker** uploads artifacts to MinIO (params, mesh, source image)
9. **Worker** creates `Avatar` record with measurements
10. **Worker** marks job as `completed`
11. **Client** polls `GET /api/v1/avatar/job/{job_id}` until complete
12. **Client** retrieves avatar via `GET /api/v1/avatar/{user_id}`

---

## Design Decisions

### 1. Async Job Processing (RQ over Celery)

**Decision**: Use Redis Queue (RQ) instead of Celery

**Rationale**:
- Simpler architecture for single-task workloads
- Lower memory footprint
- Sufficient for our use case (no complex task chaining needed)
- Easier debugging with straightforward job model

**Trade-offs**:
- Less feature-rich than Celery
- No built-in rate limiting (can add if needed)

### 2. HMR2 over PIXIE

**Decision**: Use HMR 2.0 (4D-Humans) for body reconstruction instead of PIXIE

**Rationale**:
- Better body shape accuracy (state-of-the-art as of 2023)
- Publicly available weights (no registration required)
- Cleaner dependency chain
- Active maintenance

**Trade-offs**:
- Outputs SMPL (not SMPL-X), so no face/hand articulation
- For our use case (clothing fit), body shape is primary concern

### 3. SMPL vs SMPL-X

**Decision**: Use SMPL body model (6,890 vertices) instead of SMPL-X (10,475 vertices)

**Rationale**:
- HMR2 outputs SMPL parameters directly
- SMPL→SMPL-X conversion is complex and lossy
- SMPL is sufficient for body measurements
- Smaller model = faster inference

**Trade-offs**:
- No facial expressions or hand poses
- Acceptable since we only need body shape for fit prediction

### 4. T-Pose for Measurements

**Decision**: Extract measurements from T-pose mesh, ignoring input pose

**Rationale**:
- Consistent measurements regardless of photo pose
- Industry standard for body scanning
- Circumference calculations require standardized pose

**Implementation**:
- Use only `betas` (shape) parameters
- Zero out `body_pose` and `global_orient`
- Generate mesh in canonical T-pose

### 5. Convex Hull for Circumferences

**Decision**: Use convex hull approximation for circumference measurements

**Rationale**:
- Robust to mesh noise and self-intersections
- Fast computation via scipy
- Acceptable accuracy for clothing fit

**Trade-offs**:
- Slight overestimation vs true surface path
- Could improve with geodesic distance if needed

### 6. Storage Architecture (S3-Compatible)

**Decision**: Use MinIO locally, S3 in production

**Rationale**:
- Same API for development and production
- Easy local testing without AWS credentials
- Presigned URLs for temporary access

**Storage Structure**:
```
s3://perfit-avatars/
└── avatars/
    └── {user_id}/
        └── {avatar_id}/
            ├── smplx_params.npz
            ├── mesh.obj
            └── source.{ext}
```

### 7. Model Caching Strategy

**Decision**: Module-level singletons for ML models

**Rationale**:
- Models load once per worker process
- Subsequent requests reuse loaded model
- No cross-process sharing needed (each worker independent)

**Implementation**:
```python
_hmr2_model = None  # Module-level singleton

class HMR2Inference:
    def __init__(self):
        self._ensure_models_loaded()  # Loads only if _hmr2_model is None
```

---

## Data Models

### ExtractionJob

Tracks the lifecycle of an avatar creation request.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| user_id | String | Customer reference |
| status | Enum | pending → processing → completed/failed |
| gender | Enum | male/female/neutral |
| input_image_path | String | Storage reference |
| avatar_id | UUID | Link to completed Avatar (nullable) |
| error_message | Text | Failure details |
| rq_job_id | String | Redis job ID |
| created_at | Timestamp | Job creation time |
| started_at | Timestamp | Processing start time |
| completed_at | Timestamp | Completion time |

### Avatar

Stores the processed body model and measurements.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| user_id | String | Customer reference (indexed) |
| gender | Enum | Body model gender |
| smplx_betas | JSON | Shape coefficients (10 values) |
| smplx_body_pose | JSON | Joint rotations (69 values) |
| smplx_global_orient | JSON | Global rotation (3 values) |
| height_cm | Float | Total body height |
| chest_circumference_cm | Float | Chest at armpit level |
| waist_circumference_cm | Float | Narrowest torso |
| hip_circumference_cm | Float | Widest hip |
| inseam_cm | Float | Crotch to ankle |
| shoulder_width_cm | Float | Shoulder to shoulder |
| arm_length_cm | Float | Shoulder to wrist |
| thigh_circumference_cm | Float | Upper leg |
| neck_circumference_cm | Float | Neck base |
| smplx_params_url | String | S3 URL to .npz file |
| mesh_url | String | S3 URL to .obj file |
| source_image_url | String | S3 URL to source photo |
| created_at | Timestamp | Creation time |

---

## API Design

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/v1/avatar | Create avatar from photo |
| GET | /api/v1/avatar/job/{job_id} | Poll job status |
| GET | /api/v1/avatar/{user_id} | Get latest avatar |
| GET | /api/v1/avatar/{user_id}/history | Get all avatars |
| GET | /api/v1/health | Health check |
| GET | /api/v1/health/live | Liveness probe |
| GET | /api/v1/health/ready | Readiness probe |

### Response Codes

- `202 Accepted`: Job created successfully (async processing)
- `200 OK`: Data retrieved successfully
- `400 Bad Request`: Invalid input (e.g., no face detected)
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

---

## ML Pipeline

### Stage 1: Person Detection (ViTDet)

- Input: RGB image
- Output: Bounding boxes with confidence scores
- Threshold: 0.5 (configurable)
- Selection: Largest bounding box if multiple people

### Stage 2: Body Reconstruction (HMR2)

- Input: Cropped person image (256x192)
- Output: SMPL parameters
  - `betas`: Shape coefficients (10,)
  - `body_pose`: Joint rotations as rotation matrices (23, 3, 3)
  - `global_orient`: Global rotation (1, 3, 3)
  - `vertices`: Mesh vertices (6890, 3)

### Stage 3: Measurement Extraction

- Input: SMPL betas (shape only)
- Process: Generate T-pose mesh, calculate measurements
- Output: 9 body measurements in centimeters

### Measurement Algorithms

| Measurement | Method |
|-------------|--------|
| Height | max(y) - min(y) of all vertices |
| Shoulder Width | Euclidean distance between shoulder joints |
| Arm Length | Sum of upper arm + forearm segment lengths |
| Inseam | Crotch vertex Y - ankle Y |
| Circumferences | Horizontal slice → convex hull perimeter |

---

## Error Handling

### Validation Errors (Synchronous)

- Invalid image format → 400 with error details
- Image too small/large → 400 with size requirements
- No face detected → 400 "No face detected in image"

### Processing Errors (Asynchronous)

- No person detected → Job failed, error in `error_message`
- Model loading failure → Job failed, logged
- Storage upload failure → Job failed, logged

### Recovery

- Failed jobs can be retried by creating a new request
- Job status persists for debugging
- Logs capture full stack traces

---

## Security Considerations

1. **Input Validation**: Image size limits, format whitelist
2. **Non-root Containers**: Docker runs as `appuser`
3. **No Secrets in Code**: All via environment variables
4. **Presigned URLs**: Temporary access to storage objects
5. **Face Detection**: Basic privacy check (ensures human photo)

---

## Scalability

### Horizontal Scaling

- **API Servers**: Stateless, scale via load balancer
- **Workers**: Scale by adding RQ worker instances
- **Storage**: S3/MinIO handles scale automatically

### Bottlenecks

- **GPU Memory**: One inference per worker at a time
- **Model Loading**: ~5s cold start per worker
- **Database**: Connection pooling via SQLAlchemy

### Recommendations

- Pre-warm workers with model loading
- Use GPU instances for production workers
- Consider batch processing for high volume

---

## Monitoring

### Health Checks

- `/health/live`: Process is running
- `/health/ready`: Database + Redis connected

### Metrics to Track

- Job queue depth
- Processing time per job
- Success/failure rates
- Storage usage

### Logging

- Structured logging via Loguru
- Request IDs for tracing
- ML inference timing

---

## Future Considerations

1. **Batch Processing**: Process multiple people in one image
2. **Video Input**: Track body across frames for stability
3. **SMPL-X Upgrade**: If face/hands needed, convert or use different model
4. **Accuracy Improvements**: Fine-tune on domain-specific data
5. **Caching**: Cache measurements for same user/similar photos
