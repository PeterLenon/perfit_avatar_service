"""
Pydantic schemas for API request/response validation.

These schemas define the contract between the API and clients.
Internal data structures (like database models) are separate.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    """Status of an avatar extraction job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Gender(str, Enum):
    """Gender for body model selection."""

    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


# =============================================================================
# Request Schemas
# =============================================================================


class AvatarCreateRequest(BaseModel):
    """Request to create a new avatar from a user photo."""

    user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the user",
    )
    image_base64: str = Field(
        ...,
        min_length=100,
        description="Base64-encoded image data (JPEG, PNG, or WebP)",
    )
    gender: Gender = Field(
        default=Gender.NEUTRAL,
        description="Gender hint for body model (improves accuracy)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "image_base64": "/9j/4AAQSkZJRg...",
                "gender": "female",
            }
        }
    )


# =============================================================================
# Response Schemas
# =============================================================================


class BodyMeasurements(BaseModel):
    """
    Body measurements extracted from SMPL-X parameters.

    All measurements are in centimeters unless otherwise noted.
    """

    height_cm: float = Field(..., description="Total body height")
    chest_circumference_cm: float = Field(..., description="Chest circumference at widest point")
    waist_circumference_cm: float = Field(..., description="Waist circumference at narrowest point")
    hip_circumference_cm: float = Field(..., description="Hip circumference at widest point")
    inseam_cm: float = Field(..., description="Inner leg length from crotch to ankle")
    shoulder_width_cm: float = Field(..., description="Shoulder to shoulder width")
    arm_length_cm: float = Field(..., description="Shoulder to wrist length")
    thigh_circumference_cm: float = Field(..., description="Thigh circumference at widest point")
    neck_circumference_cm: float = Field(..., description="Neck circumference")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "height_cm": 170.5,
                "chest_circumference_cm": 96.2,
                "waist_circumference_cm": 78.5,
                "hip_circumference_cm": 101.3,
                "inseam_cm": 76.2,
                "shoulder_width_cm": 45.0,
                "arm_length_cm": 58.5,
                "thigh_circumference_cm": 55.8,
                "neck_circumference_cm": 38.0,
            }
        }
    )


class JobResponse(BaseModel):
    """Response after submitting an avatar creation job."""

    job_id: str = Field(..., description="Unique identifier for tracking the job")
    status: JobStatus = Field(..., description="Current job status")
    message: str = Field(..., description="Human-readable status message")
    created_at: datetime = Field(..., description="When the job was created")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "job_abc123",
                "status": "pending",
                "message": "Job queued for processing",
                "created_at": "2024-01-15T10:30:00Z",
            }
        }
    )


class JobStatusResponse(BaseModel):
    """Response when polling for job status."""

    job_id: str
    status: JobStatus
    message: str
    created_at: datetime
    started_at: datetime | None = Field(None, description="When processing started")
    completed_at: datetime | None = Field(None, description="When processing completed")
    error: str | None = Field(None, description="Error message if job failed")
    progress_percent: int | None = Field(None, ge=0, le=100, description="Progress percentage")


class AvatarResponse(BaseModel):
    """Complete avatar data for a user."""

    user_id: str
    avatar_id: str = Field(..., description="Unique identifier for this avatar version")
    measurements: BodyMeasurements
    smplx_params_url: str | None = Field(
        None,
        description="URL to download SMPL-X parameters (for 3D rendering)",
    )
    mesh_url: str | None = Field(
        None,
        description="URL to download body mesh (OBJ format)",
    )
    created_at: datetime
    gender: Gender

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "avatar_id": "avatar_xyz789",
                "measurements": {
                    "height_cm": 170.5,
                    "chest_circumference_cm": 96.2,
                    "waist_circumference_cm": 78.5,
                    "hip_circumference_cm": 101.3,
                    "inseam_cm": 76.2,
                    "shoulder_width_cm": 45.0,
                    "arm_length_cm": 58.5,
                    "thigh_circumference_cm": 55.8,
                    "neck_circumference_cm": 38.0,
                },
                "smplx_params_url": "https://storage.example.com/avatars/user_12345/params.npz",
                "mesh_url": "https://storage.example.com/avatars/user_12345/mesh.obj",
                "created_at": "2024-01-15T10:35:00Z",
                "gender": "female",
            }
        }
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "validation_error",
                "message": "Image validation failed",
                "details": {"reason": "No face detected in image"},
            }
        }
    )


class ImageValidationResult(BaseModel):
    """Result of image validation checks."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    image_width: int | None = None
    image_height: int | None = None
    face_detected: bool = False
    face_confidence: float | None = None
