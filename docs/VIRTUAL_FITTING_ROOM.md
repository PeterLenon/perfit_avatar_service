# Virtual Fitting Room - Design Document

## Overview

The Virtual Fitting Room service enables users to:
1. Upload photos of clothing items
2. Extract and reconstruct 3D garment models from 2D images
3. Simulate cloth physics and material properties
4. Fit garments onto avatars with proper physics simulation
5. Animate avatars wearing clothes in various poses

## Architecture

### Workflow Overview

The system supports two main workflows:

1. **Garment Upload & Processing** (One-time per clothing item):
   ```
   User uploads clothing photo
        ↓
   Garment Extraction (segmentation, classification)
        ↓
   3D Garment Reconstruction
        ↓
   Store Garment in database (garment_id)
   ```

2. **On-Demand Virtual Try-On** (When user wants to try on):
   ```
   User selects: avatar_id + garment_id
        ↓
   Load Avatar mesh + Garment mesh
        ↓
   Garment Fitting Service (physics simulation)
        ↓
   Generate Animated GLB with clothes
        ↓
   Return fitted_avatar_id (or direct GLB URL)
   ```

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GARMENT UPLOAD PIPELINE                      │
│  (One-time processing when clothing item is uploaded)           │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Clothing     │────▶│  Garment Extract  │────▶│  3D Garment     │
│ Photo Upload │     │     Service       │     │  Reconstruction │
└──────────────┘     └──────────────────┘     └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  Store Garment  │
                                              │  (garment_id)   │
                                              └─────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  ON-DEMAND TRY-ON PIPELINE                      │
│  (Executed when user requests to try on clothing)                │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ User Request │────▶│  Load Avatar +    │────▶│  Garment Fitting │
│ avatar_id +  │     │  Load Garments    │     │  Service         │
│ [garment_ids]│     │  (multiple) from  │     │  (Physics Sim   │
│              │     │  Storage          │     │   with Layering)│
└──────────────┘     └──────────────────┘     └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  Generate       │
                                              │  Animated GLB   │
                                              │  with Clothes   │
                                              └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  Return GLB URL │
                                              │  (or cache it)  │
                                              └─────────────────┘
```

## Components

### 1. Garment Extraction Service
- **Input**: Clothing photo (shirt, pants, dress, etc.)
- **Process**:
  - Semantic segmentation (extract clothing from background)
  - Garment type classification (shirt, pants, dress, etc.)
  - Key point detection (collar, sleeves, hem, etc.)
  - Texture extraction
- **Output**: Segmented garment mask, key points, texture

### 2. 3D Garment Reconstruction Service
- **Input**: Segmented garment image, key points
- **Process**:
  - Garment parameterization (pattern-based or neural)
  - 3D mesh generation from 2D image
  - UV mapping for textures
- **Output**: 3D garment mesh (vertices, faces, UVs)

### 3. Cloth Physics Simulation Service
- **Input**: 3D garment mesh, material properties
- **Process**:
  - Cloth simulation setup (mass-spring or FEM)
  - Material property estimation (stretch, bend, shear)
  - Collision detection setup
- **Output**: Physics-ready garment mesh with constraints

### 4. Garment Fitting Service
- **Input**: Avatar mesh, garment mesh(es), physics parameters
- **Process**:
  - Handle multiple garments with layering order
  - Garment-to-body attachment points (per garment)
  - Initial draping simulation (with inter-garment collision)
  - Collision detection and response (body + garment + garment)
  - Skinning/weighting for animation
  - Combine all garments into single mesh or layered GLB
- **Output**: Fitted mesh with all garments attached to avatar

### 5. Virtual Fitting Room API
- Endpoints for uploading clothing photos
- Endpoints for trying on garments
- Endpoints for retrieving fitted avatars with clothes

## Technology Stack

### Computer Vision
- **Segmentation**: DeepLabV3, Mask R-CNN, or Segment Anything Model (SAM)
- **3D Reconstruction**: PIFu, PIFuHD, or Garment3D
- **Key Point Detection**: OpenPose, MediaPipe, or custom models

### Physics Simulation
- **Cloth Simulation**: 
  - PyBullet (bullet physics)
  - MuJoCo (advanced physics)
  - Taichi (GPU-accelerated)
  - Custom mass-spring system
- **Collision Detection**: Bullet, FCL, or custom

### 3D Processing
- **Mesh Processing**: Trimesh, Open3D, PyMeshLab
- **Animation**: SMPL-X with garment skinning

## Data Flow

### Phase 1: Garment Upload (One-time)

1. **User uploads clothing photo**:
   - `POST /api/v1/garments` with image_base64, garment_type (optional)
   - Service validates image
   - Creates GarmentProcessingJob (status=pending)

2. **Background processing**:
   - Garment extraction (segmentation, key points)
   - 3D reconstruction (mesh generation)
   - Material property estimation
   - Stores results in database and storage
   - Updates job status to completed

3. **User retrieves processed garment**:
   - `GET /api/v1/garments/{garment_id}`
   - Returns garment metadata and URLs

### Phase 2: On-Demand Try-On

1. **User requests try-on**:
   - `POST /api/v1/fitting-room/try-on` with `avatar_id` and `garment_ids` (array)
   - Example: `{"avatar_id": "...", "garment_ids": ["shirt_id", "pants_id", "shoes_id"]}`
   - Service creates FittingJob (status=pending)
   - Service determines layering order based on garment types

2. **Background fitting**:
   - Loads avatar mesh from storage
   - Loads all garment meshes from storage (in parallel)
   - Determines layering order (e.g., underwear → shirt → jacket → pants → shoes)
   - Runs physics simulation with layering:
     - Fit innermost garments first
     - Fit outer garments with collision against inner garments
     - Handle inter-garment collisions
   - Generates combined fitted mesh (or layered GLB)
   - Creates animated GLB with avatar + all garments
   - Stores fitted result (optional caching)

3. **User retrieves fitted avatar**:
   - `GET /api/v1/fitting-room/fitted/{fitted_id}` or
   - Direct GLB URL returned from try-on endpoint
   - Returns animated GLB with avatar wearing clothes

### Key Design Decisions

- **Separation of concerns**: Garment processing is separate from fitting
- **On-demand fitting**: Fitting happens when requested, not pre-computed
- **Caching**: Fitted results can be cached (same avatar + garments = same result)
- **Multiple garments**: User can try multiple garments simultaneously (complete outfits)
- **Layering**: Automatic layering order based on garment types:
  - **Base layer**: underwear, undershirt
  - **Mid layer**: shirt, t-shirt, dress
  - **Outer layer**: jacket, coat, sweater
  - **Bottom**: pants, shorts, skirt
  - **Footwear**: shoes, socks, boots
  - **Accessories**: hat, belt (optional)
- **Inter-garment collision**: Physics simulation handles collisions between garments
- **Outfit composition**: Support for saving complete outfits (optional)

## Database Schema

### Garment
Represents a clothing item that has been processed and is ready for try-on.

- `id`: UUID primary key
- `user_id`: Owner of the garment
- `name`: Optional user-friendly name
- `garment_type`: shirt, pants, dress, jacket, etc.
- `source_image_url`: Original photo
- `segmented_image_url`: Extracted garment mask
- `garment_mesh_url`: 3D mesh (OBJ/GLB)
- `texture_url`: Texture image
- `material_properties`: JSON with stretch, bend, shear, density
- `key_points`: JSON with key points (collar, sleeves, etc.)
- `status`: processing, completed, failed
- `created_at`, `updated_at`

### FittedGarment (Optional - for caching)
Cached result of fitting one or more garments to an avatar.

- `id`: UUID primary key
- `avatar_id`: Reference to Avatar
- `garment_ids`: JSON array of garment UUIDs (supports multiple garments)
- `fitted_mesh_url`: Combined mesh (avatar + all garments)
- `animated_glb_url`: Animated GLB with all clothes
- `fitting_parameters`: JSON with:
  - `layering_order`: Order of garments (innermost to outermost)
  - `attachment_points`: Per-garment attachment points
  - `draping_parameters`: Physics simulation parameters
- `created_at`: When it was fitted

**Note**: This can be optional if we want to always compute on-demand. 
Caching is useful for performance but increases storage costs.

### Outfit (Optional - for saving complete outfits)
Saved collection of garments that form a complete outfit.

- `id`: UUID primary key
- `user_id`: Owner of the outfit
- `name`: User-friendly name (e.g., "Casual Friday", "Business Suit")
- `description`: Optional description
- `garment_ids`: JSON array of garment UUIDs in this outfit
- `created_at`, `updated_at`

**Note**: This is optional but useful for UX - users can save favorite outfits
and try them on quickly without selecting individual garments each time.

## API Endpoints

### Garment Management

- `POST /api/v1/garments` - Upload clothing photo (creates processing job)
- `GET /api/v1/garments/job/{job_id}` - Check garment processing status
- `GET /api/v1/garments/{garment_id}` - Get processed garment details
- `GET /api/v1/garments` - List user's garments
- `DELETE /api/v1/garments/{garment_id}` - Delete garment

### Virtual Fitting Room

- `POST /api/v1/fitting-room/try-on` - Try on garment(s) on avatar (on-demand)
  - Supports single garment or multiple garments (outfit)
  - Body: `{"avatar_id": "...", "garment_ids": ["...", "..."]}`
- `GET /api/v1/fitting-room/job/{job_id}` - Check fitting job status
- `GET /api/v1/fitting-room/fitted/{fitted_id}` - Get fitted avatar result
- `GET /api/v1/fitting-room/avatar/{avatar_id}/outfits` - List all fitted outfits for avatar

### Outfit Management (Optional - for saving complete outfits)

- `POST /api/v1/outfits` - Save an outfit (collection of garments)
- `GET /api/v1/outfits/{outfit_id}` - Get outfit details
- `POST /api/v1/fitting-room/try-on-outfit` - Try on saved outfit

## Implementation Phases

### Phase 1: Basic Structure ✅ (Current)
- Database models (Garment, FittedGarment)
- Service interfaces (extraction, reconstruction, fitting)
- API endpoint structure
- Basic placeholders for ML components

### Phase 2: Garment Upload & Processing
- Implement garment extraction service
- Implement 3D reconstruction service
- Background job processing for garments
- Storage integration

### Phase 3: On-Demand Fitting
- Implement garment fitting service
- Physics simulation integration
- Avatar + garment mesh combination
- Animated GLB generation with clothes

### Phase 4: Advanced Features
- ✅ Multiple garment layering (implemented in architecture)
- Inter-garment collision detection optimization
- Material property optimization
- Outfit saving and management
- Real-time preview (optional)
- Caching strategy for fitted results
- Garment size adjustment (if garment doesn't fit avatar perfectly)
