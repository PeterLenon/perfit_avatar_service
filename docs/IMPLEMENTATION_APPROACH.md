# Implementation Approach & Differences Explained

## 1. Placeholder vs Real Implementation - Detailed Explanation

### What is a Placeholder?

A **placeholder** is a simplified implementation that:
- ✅ Maintains the same interface/API as the real implementation
- ✅ Returns data in the correct format
- ✅ Allows the pipeline to run end-to-end
- ❌ Doesn't use actual ML models
- ❌ Produces simplified/approximate results

### Example: Segmentation

#### Placeholder Implementation (Current)
```python
def _placeholder_segmentation(self, image: Image.Image) -> np.ndarray:
    """
    Uses GrabCut algorithm (computer vision, not ML).
    - Works on any image
    - No training needed
    - Fast but less accurate
    - Good enough for testing pipeline
    """
    # Uses OpenCV's GrabCut - a traditional CV algorithm
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    return mask
```

**Characteristics**:
- Uses traditional computer vision (GrabCut algorithm)
- Works immediately, no model training
- ~70-80% accuracy on good images
- Fast (~100ms per image)

#### Real ML Implementation (Future)
```python
def _sam_segmentation(self, image: Image.Image) -> np.ndarray:
    """
    Uses Segment Anything Model (ML-based).
    - Requires model checkpoint (~2.4GB)
    - Needs GPU for good performance
    - Very accurate (~95%+ accuracy)
    - Can segment any object, not just garments
    """
    # Load SAM model (done once in __init__)
    self.sam_predictor.set_image(np.array(image))
    masks, scores, logits = self.sam_predictor.predict(...)
    return masks[0]  # Best mask
```

**Characteristics**:
- Uses deep learning model (SAM)
- Requires model file download
- ~95%+ accuracy
- Slower (~500ms per image on GPU)

### Why Use Placeholders First?

**Development Benefits**:
1. **Test Pipeline Structure**: Can test entire flow without waiting for ML models
2. **Clear Integration Points**: Obvious where ML models go
3. **Incremental Development**: Add ML models one at a time
4. **No Blocking**: Development continues while models are being trained/set up

**Example Flow**:
```
Week 1: Build pipeline with placeholders → Test structure works
Week 2: Add SAM for segmentation → Test segmentation works
Week 3: Add PIFu for 3D → Test 3D works
Week 4: Optimize and fine-tune
```

**Without Placeholders**:
```
Week 1-3: Wait for ML models to be set up
Week 4: Try to integrate → Find structural issues
Week 5: Fix structure → Test again
```

### When to Replace Placeholders?

**Replace when**:
- ✅ ML models are ready and tested
- ✅ You have validation data to compare results
- ✅ Performance/accuracy becomes important
- ✅ You're ready for production

**Keep placeholders if**:
- Still in development/testing phase
- ML models aren't ready yet
- Placeholder accuracy is acceptable for your use case

---

## 2. Dependency Injection vs Direct Creation

### Current Approach: Direct Creation

```python
class GarmentFittingService:
    def __init__(self):
        # Creates dependencies directly
        self.physics_service = ClothPhysicsService()
        self.reconstruction_service = GarmentReconstructionService()
```

**Pros**:
- ✅ Simple - no need to pass dependencies
- ✅ Easy to use - just instantiate the service
- ✅ Works well for MVP

**Cons**:
- ❌ Hard to test (can't inject mocks)
- ❌ Hard to swap implementations
- ❌ Services are tightly coupled

### Dependency Injection Approach

```python
class GarmentFittingService:
    def __init__(
        self,
        physics_service: ClothPhysicsService,
        reconstruction_service: GarmentReconstructionService,
    ):
        # Receives dependencies
        self.physics_service = physics_service
        self.reconstruction_service = reconstruction_service

# Usage:
physics = ClothPhysicsService()
reconstruction = GarmentReconstructionService()
fitting = GarmentFittingService(physics, reconstruction)
```

**Pros**:
- ✅ Easy to test (inject mocks)
- ✅ Easy to swap implementations
- ✅ Services are loosely coupled
- ✅ Better for production

**Cons**:
- ❌ More verbose
- ❌ Need to manage dependencies
- ❌ Slightly more complex

### Decision for MVP

**We'll use Direct Creation for MVP** because:
- Simpler to get started
- Less boilerplate
- Can refactor to DI later if needed

**We can add DI later** when:
- We need better testability
- We want to swap implementations
- Codebase grows larger

---

## 3. MVP-First Approach

### MVP Definition

**Minimum Viable Product** = The simplest version that works end-to-end

**MVP Features**:
1. ✅ Upload garment photo → Get processed garment
2. ✅ Try on garment(s) on avatar → Get fitted result
3. ✅ View animated GLB with clothes
4. ✅ Basic layering (underwear → shirt → pants → shoes)
5. ✅ Default material properties
6. ✅ Caching enabled by default

**NOT in MVP**:
- ❌ ML-based segmentation (use GrabCut placeholder)
- ❌ ML-based 3D reconstruction (use geometric mesh)
- ❌ Advanced physics simulation (use simple draping)
- ❌ Custom material properties (use defaults)
- ❌ Multiple physics engines (use one approach)

### MVP Implementation Strategy

**Phase 1: Core Structure** ✅ (Done)
- Services, workers, API endpoints
- Database models
- Basic placeholders

**Phase 2: Make It Work** (Current)
- Improve placeholders to be more realistic
- Add mesh loading from storage
- Generate animated GLB with clothes
- Test end-to-end flow

**Phase 3: Polish** (Next)
- Better error handling
- Performance optimization
- Documentation

**Phase 4: ML Integration** (Later)
- Replace placeholders with ML models
- Fine-tune for accuracy
- Optimize for production

### Customization Later

**What can be customized later**:
- ML models (swap SAM for DeepLabV3, etc.)
- Physics engines (swap simple for PyBullet, etc.)
- Layering rules (add new garment types, custom orders)
- Caching strategy (always, never, smart caching)
- Material properties (ML-based estimation)

**How to enable customization**:
- Configuration files
- Environment variables
- Database settings
- Feature flags

---

## 4. Caching Strategy

### Current Design: Optional Caching

```python
# In fitting_room.py
cache_result: bool = Field(True, description="Whether to cache...")

# In worker
if cache_result:
    fitted_garment = FittedGarment(...)
    db.add(fitted_garment)
```

### Updated Design: Always Cache by Default, Configurable

**Approach**:
1. **Default**: Always cache (good for most use cases)
2. **Configurable**: Can disable per-request or globally
3. **Smart**: Check cache before processing

**Implementation**:

```python
# Config
class Settings:
    fitting_cache_enabled: bool = True  # Global default

# API Request
class TryOnRequest:
    cache_result: bool = Field(
        default=True,  # Default to caching
        description="Whether to cache the fitted result"
    )

# Worker Logic
def fit_garments_to_avatar(..., cache_result: bool = True):
    # 1. Check cache first (if enabled)
    if cache_result and settings.fitting_cache_enabled:
        cached = check_cache(avatar_id, garment_ids)
        if cached:
            return cached
    
    # 2. Process fitting
    result = do_fitting(...)
    
    # 3. Cache result (if enabled)
    if cache_result and settings.fitting_cache_enabled:
        save_to_cache(result)
    
    return result
```

**Benefits**:
- ✅ Fast for repeated requests
- ✅ Reduces computation cost
- ✅ Can still disable if needed
- ✅ Configurable per use case

---

## Summary of Decisions

### 1. Placeholders vs Real Implementation
- **MVP**: Use improved placeholders (GrabCut, geometric meshes)
- **Later**: Replace with ML models when ready
- **Why**: Get working system first, improve incrementally

### 2. Dependency Injection
- **MVP**: Direct creation (simpler)
- **Later**: Can refactor to DI if needed
- **Why**: Less complexity for MVP

### 3. MVP-First
- **Focus**: Get end-to-end working
- **Customization**: Add later via config
- **Why**: Validate approach before optimizing

### 4. Caching
- **Default**: Always cache
- **Configurable**: Can disable per-request or globally
- **Why**: Best performance by default, flexibility when needed

---

## Next Steps for MVP

1. ✅ **Structure is done** - Services, workers, APIs
2. 🔄 **Improve placeholders** - More realistic results
3. 🔄 **Add mesh loading** - Load from storage
4. 🔄 **Generate GLB** - Animated avatar with clothes
5. 🔄 **Test end-to-end** - Verify full flow works
6. ⏳ **ML integration** - Replace placeholders (later)

The code is structured to make these improvements incrementally without major refactoring.
