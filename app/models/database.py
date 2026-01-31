"""
SQLAlchemy database models and connection management.

Uses async SQLAlchemy 2.0 style for non-blocking database operations.
"""

import uuid
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import JSON, DateTime, Enum, Float, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings
from app.models.schemas import Gender, JobStatus


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class Avatar(Base):
    """
    Stored avatar with body shape parameters and measurements.

    Each user can have multiple avatars (e.g., if they upload a new photo),
    but typically we use the most recent one.
    """

    __tablename__ = "avatars"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(
        String(100),
        index=True,
        nullable=False,
    )
    gender: Mapped[str] = mapped_column(
        Enum(Gender, name="gender_enum"),
        nullable=False,
        default=Gender.NEUTRAL,
    )

    # SMPL-X parameters stored as JSON
    # Shape params (betas): controls body shape (10-300 coefficients)
    # Pose params (body_pose): joint rotations
    # Expression params (expression): facial expression
    smplx_betas: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    smplx_body_pose: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    smplx_global_orient: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Pre-computed measurements (centimeters)
    height_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    chest_circumference_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    waist_circumference_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    hip_circumference_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    inseam_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    shoulder_width_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    arm_length_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    thigh_circumference_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    neck_circumference_cm: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Appearance
    # Skin color extracted from source image as [R, G, B] in 0-1 range
    skin_color: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Storage URLs
    smplx_params_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    mesh_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    animated_glb_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    source_image_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Avatar(id={self.id}, user_id={self.user_id})>"


class ExtractionJob(Base):
    """
    Tracks the status of avatar extraction jobs.

    Jobs are created when a user submits a photo and updated
    as the extraction pipeline progresses.
    """

    __tablename__ = "extraction_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(
        String(100),
        index=True,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        Enum(JobStatus, name="job_status_enum"),
        nullable=False,
        default=JobStatus.PENDING,
    )
    gender: Mapped[str] = mapped_column(
        Enum(Gender, name="gender_enum"),
        nullable=False,
        default=Gender.NEUTRAL,
    )

    # Storage references
    input_image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Result reference (set when completed)
    avatar_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # RQ job tracking
    rq_job_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def __repr__(self) -> str:
        return f"<ExtractionJob(id={self.id}, status={self.status})>"


class Garment(Base):
    """
    Represents a clothing item extracted from a photo.

    Stores garment information including type, mesh, texture, and material properties.
    """

    __tablename__ = "garments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(
        String(100),
        index=True,
        nullable=False,
    )
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    garment_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )  # shirt, pants, dress, jacket, etc.

    # Storage URLs
    source_image_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    segmented_image_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    garment_mesh_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    texture_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Material properties (JSON)
    material_properties: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Example: {"stretch": 0.5, "bend": 0.3, "shear": 0.4, "density": 0.1}

    # Dominant color extracted from garment as [R, G, B] in 0-1 range
    dominant_color: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Key points (JSON) - for garment fitting
    key_points: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Example: {"collar": [x, y], "sleeves": [[x1, y1], [x2, y2]], ...}

    # Processing status
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
    )  # pending, processing, completed, failed

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Job tracking timestamps
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Garment(id={self.id}, user_id={self.user_id}, type={self.garment_type})>"


class FittedGarment(Base):
    """
    Represents a fitted outfit (avatar with one or more garments).

    Supports multiple garments per fitting (complete outfits).
    """

    __tablename__ = "fitted_garments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    avatar_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    
    # Multiple garments stored as JSON array
    garment_ids: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
    )  # Array of garment UUIDs: ["garment_id1", "garment_id2", ...]

    # Storage URLs
    fitted_mesh_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    animated_glb_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Fitting parameters (JSON)
    fitting_parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Example: {
    #   "layering_order": ["underwear", "shirt", "pants", "shoes"],
    #   "attachment_points": {...},
    #   "draping_parameters": {...}
    # }

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<FittedGarment(id={self.id}, avatar_id={self.avatar_id}, garments={len(self.garment_ids)})>"


class Outfit(Base):
    """
    Represents a saved outfit (collection of garments).

    Users can save complete outfits for quick try-on later.
    """

    __tablename__ = "outfits"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(
        String(100),
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Garment IDs in this outfit
    garment_ids: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
    )  # Array of garment UUIDs

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Outfit(id={self.id}, user_id={self.user_id}, name={self.name})>"


# =============================================================================
# Database Connection Management
# =============================================================================

_engine = None
_session_factory = None


def get_engine():
    """Get or create the async database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database.async_url,
            echo=settings.debug,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session.

    Usage in FastAPI:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            ...
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """
    Initialize database tables.

    In production, use Alembic migrations instead.
    This is for development convenience.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections on shutdown."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
