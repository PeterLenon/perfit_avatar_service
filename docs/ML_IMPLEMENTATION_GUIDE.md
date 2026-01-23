# ML Models Implementation Guide

This document outlines what's needed to implement the actual ML models for the virtual fitting room.

## Overview

The virtual fitting room requires several ML models across different stages:

1. **Garment Extraction** - Computer vision models
2. **3D Garment Reconstruction** - 3D reconstruction models
3. **Material Property Estimation** - Optional ML-based estimation
4. **Physics Simulation** - Physics engines (not ML, but included for completeness)

---

## 1. Garment Extraction Models

### 1.1 Semantic Segmentation

**Purpose**: Extract garment from background in photos

**Options**:

#### Option A: Segment Anything Model (SAM) - Recommended
- **Model**: Meta's Segment Anything Model
- **Repository**: https://github.com/facebookresearch/segment-anything
- **Pros**: 
  - Very accurate segmentation
  - Works on any object type
  - Can be fine-tuned for garments
- **Cons**: 
  - Large model size (~2.4GB)
  - Requires GPU for real-time
- **Implementation**:
  ```python
  # Install: pip install git+https://github.com/facebookresearch/segment-anything.git
  from segment_anything import sam_model_registry, SamPredictor
  
  sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
  predictor = SamPredictor(sam)
  ```

#### Option B: DeepLabV3
- **Model**: DeepLabV3 with ResNet backbone
- **Library**: torchvision
- **Pros**: 
  - Smaller model
  - Good for clothing segmentation
  - Can fine-tune on fashion datasets
- **Cons**: 
  - Less general than SAM
  - Requires training data
- **Implementation**:
  ```python
  import torchvision.models as models
  deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)
  # Fine-tune on fashion/clothing dataset
  ```

#### Option C: Mask R-CNN
- **Model**: Mask R-CNN for instance segmentation
- **Library**: detectron2 (already in your dependencies)
- **Pros**: 
  - Already have detectron2 installed
  - Good for multiple garments in one image
- **Cons**: 
  - Requires training on clothing dataset
- **Implementation**:
  ```python
  from detectron2.engine import DefaultPredictor
  from detectron2.config import get_cfg
  # Configure and load trained model
  ```

**Recommended**: Start with SAM for best results, or DeepLabV3 if you want smaller model.

**Training Data Needed**:
- Fashion datasets: DeepFashion2, ModaNet, or custom dataset
- Annotations: Segmentation masks for clothing items

### 1.2 Garment Type Classification

**Purpose**: Classify garment type (shirt, pants, dress, etc.)

**Options**:

#### Option A: Fine-tuned ResNet/EfficientNet
- **Model**: ResNet50 or EfficientNet-B3
- **Library**: torchvision or timm
- **Pros**: 
  - Simple and effective
  - Fast inference
- **Implementation**:
  ```python
  import torchvision.models as models
  classifier = models.resnet50(pretrained=True)
  # Replace final layer for garment classification
  # Fine-tune on fashion dataset
  ```

#### Option B: Fashion-Specific Models
- **Model**: FashionNet, DeepFashion models
- **Pros**: 
  - Pre-trained on fashion data
  - Better accuracy for clothing
- **Cons**: 
  - May need to find/adapt existing models

**Training Data Needed**:
- Fashion classification datasets
- Labels: garment types (shirt, pants, dress, etc.)

### 1.3 Key Point Detection

**Purpose**: Detect key points on garments (collar, sleeves, hem, etc.)

**Options**:

#### Option A: MediaPipe (Already Installed)
- **Library**: mediapipe (already in requirements.txt)
- **Pros**: 
  - Already available
  - Fast inference
  - Good for body key points
- **Cons**: 
  - Not specifically for garment key points
  - May need adaptation

#### Option B: OpenPose
- **Model**: OpenPose for pose estimation
- **Pros**: 
  - Accurate key point detection
  - Can be adapted for garments
- **Cons**: 
  - Complex setup
  - Slower inference

#### Option C: Custom Key Point Model
- **Model**: Train custom model on garment key points
- **Pros**: 
  - Most accurate for garment-specific points
- **Cons**: 
  - Requires training data and annotation

**Training Data Needed**:
- Garment images with key point annotations
- Key points: collar, sleeves, hem, waist, etc.

---

## 2. 3D Garment Reconstruction

**Purpose**: Generate 3D mesh from 2D garment image

**Options**:

### Option A: PIFu / PIFuHD - Recommended
- **Model**: Pixel-Aligned Implicit Function
- **Repository**: https://github.com/shunsukesaito/PIFu
- **Pros**: 
  - High quality 3D reconstruction
  - Works from single image
  - Good for clothing
- **Cons**: 
  - Requires GPU
  - Model size ~200MB
- **Implementation**:
  ```python
  # Install: pip install git+https://github.com/shunsukesaito/PIFu.git
  from lib.model import PIFu
  # Load pre-trained model
  # Run inference on garment image
  ```

### Option B: Garment3D
- **Model**: Garment3D for clothing reconstruction
- **Repository**: https://github.com/Garment3D/Garment3D
- **Pros**: 
  - Specifically designed for garments
  - Handles different garment types
- **Cons**: 
  - May be less maintained
  - Requires setup

### Option C: TailorNet / Cloth3D
- **Model**: Neural garment reconstruction
- **Pros**: 
  - Good for parametric garments
  - Can handle different poses
- **Cons**: 
  - More complex setup
  - May require garment patterns

### Option D: Pattern-Based Approach
- **Approach**: Use garment patterns + 2D image to reconstruct 3D
- **Pros**: 
  - More accurate for specific garment types
  - Can use existing pattern libraries
- **Cons**: 
  - Requires pattern database
  - More complex implementation

**Recommended**: Start with PIFu for general reconstruction, or Garment3D if specifically for clothing.

**Training Data Needed**:
- 3D garment meshes with corresponding 2D images
- Datasets: DeepFashion3D, Cloth3D, or custom

---

## 3. Material Property Estimation (Optional)

**Purpose**: Estimate cloth material properties from images

**Options**:

### Option A: Image-Based Property Estimation
- **Approach**: Train CNN to predict material properties from images
- **Model**: ResNet or Vision Transformer
- **Input**: Garment image
- **Output**: stretch, bend, shear, density values
- **Training Data**: 
  - Images of different fabric types
  - Known material properties (from physics measurements)

### Option B: Fabric Classification + Lookup
- **Approach**: Classify fabric type, then lookup properties
- **Simpler**: Just classify fabric (cotton, silk, denim, etc.)
- **Lookup**: Use database of known properties per fabric type

**Recommended**: Start with Option B (simpler), upgrade to Option A if needed.

---

## 4. Implementation Steps

### Phase 1: Garment Extraction (Start Here)

1. **Set up SAM or DeepLabV3**:
   ```bash
   # For SAM
   pip install git+https://github.com/facebookresearch/segment-anything.git
   # Download checkpoint: sam_vit_h_4b8939.pth
   
   # For DeepLabV3 (already in torchvision)
   # No additional install needed
   ```

2. **Update `GarmentExtractionService`**:
   - Replace `_placeholder_segmentation()` with actual model
   - Load model in `__init__()`
   - Implement segmentation inference

3. **Add model checkpoints to storage**:
   - Store model files in `/app/models/garment_extraction/`
   - Update config to point to model paths

### Phase 2: 3D Reconstruction

1. **Set up PIFu or Garment3D**:
   ```bash
   # For PIFu
   git clone https://github.com/shunsukesaito/PIFu.git
   cd PIFu
   pip install -r requirements.txt
   # Download pre-trained checkpoints
   ```

2. **Update `GarmentReconstructionService`**:
   - Replace `_placeholder_mesh_generation()` with actual model
   - Load model in `__init__()`
   - Implement 3D reconstruction inference

3. **Handle model dependencies**:
   - May need additional dependencies (OpenGL, etc.)
   - Update Dockerfile if needed

### Phase 3: Material Properties

1. **Create fabric classification model** (if using Option B):
   - Fine-tune ResNet on fabric dataset
   - Create lookup table for properties

2. **Update `ClothPhysicsService`**:
   - Replace default properties with ML-based estimation
   - Add fabric classification

---

## 5. Required Dependencies

Add to `requirements.txt`:

```txt
# For SAM
segment-anything>=1.0

# For PIFu (if using)
# May need additional dependencies - check PIFu requirements

# For custom models
torch>=2.0.0  # Already have
torchvision>=0.15.0  # Already have
timm>=0.9.0  # Already have

# For 3D processing
open3d>=0.17.0  # For mesh processing
pymeshlab>=2022.2  # Advanced mesh processing
```

---

## 6. Model Storage & Configuration

### Update `app/config.py`:

```python
class MLSettings(BaseSettings):
    # ... existing settings ...
    
    # Garment extraction models
    garment_segmentation_model: str = "./ml/models/garment_segmentation/sam_vit_h.pth"
    garment_classification_model: str = "./ml/models/garment_classification/resnet50.pth"
    garment_keypoint_model: str = "./ml/models/garment_keypoints/keypoint_model.pth"
    
    # 3D reconstruction models
    garment_3d_model: str = "./ml/models/garment_3d/pifu.pth"
    
    # Model device
    garment_ml_device: str = "cuda"  # or "cpu"
```

### Model Directory Structure:

```
ml/
├── models/
│   ├── garment_extraction/
│   │   ├── sam_vit_h_4b8939.pth
│   │   └── garment_classifier.pth
│   ├── garment_3d/
│   │   └── pifu_checkpoint.pth
│   └── material_properties/
│       └── fabric_classifier.pth
```

---

## 7. Training Data Requirements

### For Garment Segmentation:
- **Dataset**: DeepFashion2, ModaNet, or custom
- **Format**: Images + segmentation masks
- **Size**: ~10k-100k images recommended

### For Garment Classification:
- **Dataset**: Fashion-MNIST (simpler) or DeepFashion (complex)
- **Format**: Images + class labels
- **Size**: ~1k-10k images per class

### For 3D Reconstruction:
- **Dataset**: DeepFashion3D, Cloth3D, or custom
- **Format**: 2D images + 3D meshes
- **Size**: ~1k-10k garment meshes

### For Key Point Detection:
- **Dataset**: Custom annotations
- **Format**: Images + key point coordinates
- **Size**: ~500-5k images

---

## 8. Quick Start: Minimal Implementation

To get started quickly with pre-trained models:

1. **Use SAM for segmentation** (no training needed):
   - Download SAM checkpoint
   - Integrate into `GarmentExtractionService`
   - Works out of the box

2. **Use PIFu for 3D reconstruction** (no training needed):
   - Download PIFu checkpoint
   - Integrate into `GarmentReconstructionService`
   - Works for general objects

3. **Use default material properties** (no ML needed):
   - Keep current lookup table approach
   - Can upgrade later with ML

This gives you a working system that can be improved incrementally.

---

## 9. Performance Considerations

- **GPU**: Required for real-time inference (SAM, PIFu)
- **Model Size**: SAM ~2.4GB, PIFu ~200MB
- **Inference Time**: 
  - Segmentation: ~100-500ms (GPU)
  - 3D Reconstruction: ~1-5s (GPU)
  - Classification: ~10-50ms (GPU)

- **Optimization**:
  - Use smaller models for CPU (SAM-ViT-B instead of ViT-H)
  - Quantize models for faster inference
  - Cache model outputs when possible

---

## 10. Next Steps

1. **Choose models** based on your requirements (accuracy vs speed)
2. **Download pre-trained checkpoints** for quick start
3. **Update service implementations** to use actual models
4. **Test with sample images** to verify pipeline
5. **Fine-tune models** on your specific data if needed
6. **Optimize for production** (model quantization, caching, etc.)
