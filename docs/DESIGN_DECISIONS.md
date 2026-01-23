# Design Decisions & Architecture

This document explains the design philosophy, architectural decisions, and implementation approach for the virtual fitting room system.

## Table of Contents

1. [Core Design Principles](#core-design-principles)
2. [Service Architecture](#service-architecture)
3. [Design Patterns Used](#design-patterns-used)
4. [Data Flow Design](#data-flow-design)
5. [Error Handling Strategy](#error-handling-strategy)
6. [Extensibility Considerations](#extensibility-considerations)
7. [Trade-offs & Decisions](#trade-offs--decisions)

---

## Core Design Principles

### 1. Separation of Concerns

**Principle**: Each service has a single, well-defined responsibility.

**Why**:
- **Maintainability**: Easy to understand what each service does
- **Testability**: Can test each service independently
- **Reusability**: Services can be used in different contexts
- **Debugging**: When something breaks, you know where to look

**Example**:
```
GarmentExtractionService    → Only handles image → garment extraction
GarmentReconstructionService → Only handles 2D → 3D conversion
ClothPhysicsService         → Only handles physics simulation
GarmentFittingService       → Orchestrates fitting process
```

**Not**:
```
❌ One giant service that does everything
❌ Services that depend on each other's internals
```

### 2. Dependency Injection

**Principle**: Services receive their dependencies rather than creating them.

**Why**:
- **Testability**: Can inject mocks for testing
- **Flexibility**: Easy to swap implementations
- **Configuration**: Dependencies come from config, not hardcoded

**Example**:
```python
class GarmentFittingService:
    def __init__(self):
        # ✅ Good: Creates dependencies it needs
        self.physics_service = ClothPhysicsService()
        
    # But better would be:
    def __init__(self, physics_service: ClothPhysicsService):
        # ✅ Better: Receives dependency
        self.physics_service = physics_service
```

**Current Implementation**: Services create their own dependencies for simplicity, but structure allows easy refactoring to dependency injection.

### 3. Placeholder Pattern

**Principle**: Implement structure first, then fill in ML models.

**Why**:
- **Incremental Development**: Can test pipeline without ML models
- **Clear Integration Points**: Obvious where ML models go
- **No Blocking**: Development can continue while ML models are being trained/integrated

**Example**:
```python
def _placeholder_segmentation(self, image: Image.Image) -> np.ndarray:
    """
    Placeholder for garment segmentation.
    
    TODO: Implement using DeepLabV3, SAM, or similar model.
    """
    # Returns simple mask for now
    # When ready, replace with actual model inference
```

**When to Replace**: 
- When ML models are ready
- When you have test data to validate
- When performance becomes important

### 4. Explicit Over Implicit

**Principle**: Code should be self-documenting and explicit about what it does.

**Why**:
- **Readability**: Future you (or teammates) can understand code quickly
- **Debugging**: Easier to trace what's happening
- **Maintenance**: Changes are safer when intent is clear

**Example**:
```python
# ✅ Good: Explicit
layering_order = self.determine_layering_order(garment_types)
sorted_garments = sorted(garments, key=lambda g: self.LAYERING_ORDER.get(g["garment_type"].lower(), 100))

# ❌ Bad: Implicit magic numbers
sorted_garments = sorted(garments, key=lambda g: [0,1,2,3,6,9,12][garment_type_map[g.type]])
```

---

## Service Architecture

### Service Responsibilities

#### 1. GarmentExtractionService

**Purpose**: Extract garment information from 2D photos

**Responsibilities**:
- Image segmentation (isolate garment from background)
- Garment type classification
- Key point detection
- Texture extraction

**Input**: PIL Image
**Output**: Dictionary with segmented_mask, garment_type, key_points, texture

**Design Decisions**:
- **Why PIL Image?**: Standard format, easy to work with, compatible with ML libraries
- **Why dictionary output?**: Flexible, can add fields without breaking API
- **Why separate methods?**: Each step can be tested/improved independently

**Future ML Integration**:
```python
def _placeholder_segmentation(self, image: Image.Image) -> np.ndarray:
    # Replace with:
    # return self.sam_model.predict(image)
```

#### 2. GarmentReconstructionService

**Purpose**: Convert 2D garment image to 3D mesh

**Responsibilities**:
- 3D mesh generation from 2D image
- UV coordinate generation
- Texture mapping

**Input**: Image, segmented_mask, key_points, garment_type
**Output**: Dictionary with vertices, faces, uvs, texture

**Design Decisions**:
- **Why separate from extraction?**: Different concerns (2D processing vs 3D generation)
- **Why return vertices/faces?**: Standard mesh format, compatible with all 3D tools
- **Why include UVs?**: Needed for texture mapping in final render

**Future ML Integration**:
```python
def _placeholder_mesh_generation(self, garment_type: str, mask: np.ndarray):
    # Replace with:
    # return self.pifu_model.reconstruct(image, mask)
```

#### 3. ClothPhysicsService

**Purpose**: Handle physics simulation for garments

**Responsibilities**:
- Material property estimation
- Physics simulation setup
- Draping simulation

**Input**: Garment mesh, material properties, body mesh
**Output**: Draped garment vertices

**Design Decisions**:
- **Why default properties?**: Provides reasonable defaults, can be overridden
- **Why separate from fitting?**: Physics is a distinct concern from fitting logic
- **Why dictionary for properties?**: Easy to extend with new properties

**Future Physics Engine Integration**:
```python
def simulate_draping(self, ...):
    # Replace with:
    # return self.physics_engine.simulate(garment, body, properties)
```

#### 4. GarmentFittingService

**Purpose**: Orchestrate fitting multiple garments to avatar

**Responsibilities**:
- Determine layering order
- Fit garments in sequence
- Combine meshes
- Handle inter-garment collisions

**Input**: Avatar mesh, list of garment meshes
**Output**: Combined mesh with all garments fitted

**Design Decisions**:
- **Why orchestration service?**: Coordinates multiple services, doesn't do the work itself
- **Why layering order?**: Critical for realistic appearance (underwear before shirt)
- **Why sequential fitting?**: Outer garments need to account for inner garments
- **Why combine meshes?**: Single mesh is easier to render/animate

**Key Algorithm**:
```
1. Sort garments by layering order (innermost first)
2. For each garment:
   a. Find attachment points to current body
   b. Simulate draping with physics
   c. Add garment to "current body" for next iteration
3. Combine all meshes
```

---

## Design Patterns Used

### 1. Service Layer Pattern

**What**: Business logic separated into service classes

**Why**:
- Keeps API routes thin (just validation + orchestration)
- Business logic is reusable
- Easy to test without HTTP layer

**Structure**:
```
API Route → Service → Storage/Database
   ↓         ↓            ↓
Thin      Business    Infrastructure
```

### 2. Worker Pattern

**What**: Background jobs for long-running tasks

**Why**:
- Don't block API responses
- Can retry on failure
- Can scale workers independently

**Flow**:
```
API → Create Job → Queue → Worker → Update Job Status
```

### 3. Repository Pattern (Implicit)

**What**: Database access abstracted through models

**Why**:
- Database details hidden from business logic
- Easy to swap database implementations
- Consistent data access patterns

### 4. Strategy Pattern (For ML Models)

**What**: Different ML models can be swapped

**Why**:
- Easy to try different models
- Can use simpler models for development
- Can upgrade models without changing code structure

**Example**:
```python
# Current: Placeholder strategy
def _placeholder_segmentation(...)

# Future: SAM strategy
def _sam_segmentation(...)

# Future: DeepLabV3 strategy  
def _deeplab_segmentation(...)
```

---

## Data Flow Design

### Garment Upload Flow

```
User Uploads Photo
    ↓
API Validates Image
    ↓
Create Garment Record (status=pending)
    ↓
Queue Background Job
    ↓
Return job_id (202 Accepted)
    ↓
[Background] Worker Processes:
    ↓
1. Extract Garment (segmentation, classification)
    ↓
2. Reconstruct 3D Mesh
    ↓
3. Estimate Material Properties
    ↓
4. Upload to Storage
    ↓
5. Update Garment Record (status=completed)
```

**Design Decisions**:
- **Why async?**: Processing takes time, don't block user
- **Why job status?**: User can poll for completion
- **Why separate steps?**: Can debug/optimize each step independently

### Try-On Flow

```
User Requests Try-On (avatar_id + [garment_ids])
    ↓
API Validates (avatar exists, garments exist & ready)
    ↓
Create Fitting Job
    ↓
Queue Background Job
    ↓
Return job_id
    ↓
[Background] Worker Processes:
    ↓
1. Load Avatar Mesh
    ↓
2. Load All Garment Meshes
    ↓
3. Determine Layering Order
    ↓
4. Fit Garments (with physics simulation)
    ↓
5. Generate Animated GLB
    ↓
6. Upload to Storage
    ↓
7. Create FittedGarment Record (if caching)
```

**Design Decisions**:
- **Why validate first?**: Fail fast, better error messages
- **Why load all meshes?**: Need all data before fitting
- **Why layering order?**: Critical for realistic appearance
- **Why optional caching?**: Trade-off between storage cost and speed

---

## Error Handling Strategy

### Principle: Fail Explicitly

**Approach**: Errors should be clear and actionable

**Levels of Error Handling**:

1. **Validation Errors** (400 Bad Request)
   - Invalid input format
   - Missing required fields
   - Invalid IDs
   - **Action**: Return clear error message to user

2. **Business Logic Errors** (400/404)
   - Avatar not found
   - Garment not ready
   - Invalid garment combination
   - **Action**: Return specific error with context

3. **Processing Errors** (500)
   - ML model failure
   - Storage failure
   - Unexpected exceptions
   - **Action**: Log error, update job status, return generic error to user

**Example**:
```python
# ✅ Good: Specific error
if avatar is None:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "error": "avatar_not_found",
            "message": f"Avatar {avatar_id} not found"
        }
    )

# ❌ Bad: Generic error
if avatar is None:
    raise Exception("Error")
```

### Error Recovery

**Jobs**: 
- Mark as `failed` with error message
- User can retry or see error details
- Error message stored in database

**Services**:
- Let exceptions propagate to worker
- Worker catches and updates job status
- Logs full stack trace for debugging

---

## Extensibility Considerations

### 1. Adding New Garment Types

**Current**: Hardcoded layering order dictionary

**Extension Point**:
```python
LAYERING_ORDER = {
    "shirt": 3,
    "pants": 6,
    # Add new types here
    "new_type": 5,
}
```

**Future Improvement**: Could be database-driven or config file

### 2. Adding New ML Models

**Current**: Placeholder methods ready for replacement

**Extension Point**:
```python
def _placeholder_segmentation(self, image):
    # Replace this method
    # Or add new method and switch in __init__
    if self.use_sam:
        return self._sam_segmentation(image)
    else:
        return self._deeplab_segmentation(image)
```

### 3. Adding New Physics Engines

**Current**: Simple placeholder simulation

**Extension Point**:
```python
def simulate_draping(self, ...):
    # Could support multiple engines
    if self.engine == "pybullet":
        return self._pybullet_simulate(...)
    elif self.engine == "mujoco":
        return self._mujoco_simulate(...)
```

### 4. Adding New Storage Backends

**Current**: S3/MinIO via boto3

**Extension Point**: StorageService uses boto3 client, but interface is abstracted. Could add:
- Local file storage
- Azure Blob Storage
- Google Cloud Storage

---

## Trade-offs & Decisions

### 1. Placeholder vs Real Implementation

**Decision**: Use placeholders initially

**Trade-off**:
- ✅ Can develop/test pipeline structure
- ✅ Clear integration points
- ❌ Doesn't produce real results yet
- ❌ Need to replace later

**Rationale**: Better to have working structure than perfect ML models that can't be integrated

### 2. Single vs Multiple Garments

**Decision**: Support multiple garments from start

**Trade-off**:
- ✅ More flexible
- ✅ Handles real use cases (outfits)
- ❌ More complex implementation
- ❌ Harder to test initially

**Rationale**: Real users need complete outfits, not just single items

### 3. Caching vs On-Demand

**Decision**: Optional caching (configurable)

**Trade-off**:
- ✅ Fast for repeated requests
- ✅ Reduces computation cost
- ❌ Increases storage cost
- ❌ May serve stale results

**Rationale**: Let users decide based on their use case

### 4. Synchronous vs Asynchronous Processing

**Decision**: Asynchronous (background jobs)

**Trade-off**:
- ✅ Fast API responses
- ✅ Can retry on failure
- ✅ Can scale workers
- ❌ More complex (job status, polling)
- ❌ User must poll for results

**Rationale**: Processing takes time (seconds to minutes), can't block HTTP request

### 5. Combined vs Layered Meshes

**Decision**: Combine meshes into single mesh

**Trade-off**:
- ✅ Simpler rendering (one mesh)
- ✅ Smaller file size
- ❌ Can't adjust individual garments
- ❌ Harder to swap garments

**Rationale**: For viewing/display, combined is simpler. Could add layered GLB later if needed.

---

## Code Organization Principles

### 1. Service Files Structure

```
app/services/
├── garment_extraction.py      # 2D image processing
├── garment_reconstruction.py  # 2D → 3D conversion
├── cloth_physics.py           # Physics simulation
├── garment_fitting.py         # Orchestration
├── storage.py                 # Infrastructure
└── animation.py               # GLB generation
```

**Why This Structure**:
- One file per concern
- Easy to find relevant code
- Clear dependencies

### 2. Worker Files Structure

```
app/workers/
├── extraction_worker.py   # Avatar extraction
├── garment_worker.py      # Garment processing
└── fitting_worker.py      # Garment fitting
```

**Why This Structure**:
- One worker per job type
- Each worker is independent
- Easy to scale specific workers

### 3. API Routes Structure

```
app/api/routes/
├── avatar.py          # Avatar endpoints
├── garments.py       # Garment endpoints
├── fitting_room.py   # Try-on endpoints
└── health.py         # Health checks
```

**Why This Structure**:
- One file per domain
- Clear API organization
- Easy to add new endpoints

---

## Maintenance Guide

### Adding a New Garment Type

1. **Update Layering Order**:
   ```python
   # app/services/garment_fitting.py
   LAYERING_ORDER = {
       # ... existing ...
       "new_type": 7,  # Add here
   }
   ```

2. **Update Material Properties** (if needed):
   ```python
   # app/services/cloth_physics.py
   DEFAULT_MATERIAL_PROPERTIES = {
       # ... existing ...
       "new_type": {...},  # Add defaults
   }
   ```

3. **Update Key Point Detection** (if needed):
   ```python
   # app/services/garment_extraction.py
   def _detect_key_points(self, image, garment_type):
       if garment_type == "new_type":
           return {...}  # Add key points
   ```

### Replacing Placeholder with Real ML Model

1. **Install Model Dependencies**:
   ```bash
   pip install model-library
   ```

2. **Download Model Checkpoint**:
   ```bash
   # Download to ml/models/garment_extraction/
   ```

3. **Update Service**:
   ```python
   # app/services/garment_extraction.py
   def __init__(self):
       # Load model
       self.model = load_model("path/to/checkpoint")
   
   def _placeholder_segmentation(self, image):
       # Replace with:
       return self.model.segment(image)
   ```

4. **Update Config** (if needed):
   ```python
   # app/config.py
   garment_segmentation_model: str = "./ml/models/..."
   ```

### Debugging a Fitting Issue

1. **Check Logs**: Look for errors in worker logs
2. **Verify Inputs**: Check that meshes are loaded correctly
3. **Test Services Individually**: Test each service in isolation
4. **Check Layering Order**: Verify garments are in correct order
5. **Inspect Meshes**: Save intermediate meshes to debug

---

## Testing Strategy

### Unit Tests

**What**: Test individual services

**Example**:
```python
def test_layering_order():
    service = GarmentFittingService()
    order = service.determine_layering_order(["shirt", "pants", "shoes"])
    assert order == [("shirt", 3), ("pants", 6), ("shoes", 12)]
```

### Integration Tests

**What**: Test service interactions

**Example**:
```python
def test_garment_fitting_flow():
    # Test full flow: extraction → reconstruction → fitting
    ...
```

### End-to-End Tests

**What**: Test API → Worker → Storage flow

**Example**:
```python
def test_garment_upload_flow():
    # Upload image → check job status → verify garment created
    ...
```

---

## Performance Considerations

### Current Limitations

1. **Placeholder Methods**: Not optimized, just structure
2. **No Caching**: Every request processes from scratch
3. **Sequential Processing**: Garments fitted one at a time

### Future Optimizations

1. **Model Caching**: Load ML models once, reuse
2. **Result Caching**: Cache fitted results
3. **Parallel Processing**: Fit multiple garments in parallel
4. **Model Quantization**: Use smaller/faster models
5. **GPU Acceleration**: Use GPU for ML inference

---

## Summary

### Key Takeaways

1. **Services are focused**: Each does one thing well
2. **Structure is ready**: ML models can be dropped in
3. **Error handling is explicit**: Failures are clear
4. **Code is maintainable**: Easy to understand and modify
5. **Extensible by design**: Easy to add features

### Next Steps for ML Integration

1. Choose ML models (SAM, PIFu recommended for quick start)
2. Download checkpoints
3. Replace placeholder methods
4. Test with sample data
5. Optimize for production

### Questions to Consider

- Do you want to support multiple ML models (switchable)?
- Should material properties be ML-estimated or user-provided?
- Do you need real-time fitting or is async acceptable?
- How important is caching vs storage cost?

This architecture gives you a solid foundation that's ready for ML integration while remaining maintainable and extensible.
