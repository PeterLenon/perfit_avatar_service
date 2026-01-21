"""
Application configuration using pydantic-settings.

Settings are loaded from environment variables with sensible defaults for local development.
All secrets should be provided via environment variables in production.
"""

from functools import lru_cache

from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL database configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5432
    user: str = "perfit"
    password: str = "perfit_dev"
    name: str = "perfit_avatars"

    @property
    def async_url(self) -> str:
        """Async database URL for SQLAlchemy."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        """Sync database URL for Alembic migrations."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis configuration for task queue and caching."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None

    @property
    def url(self) -> str:
        """Redis URL for RQ and caching."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class MinioSettings(BaseSettings):
    """MinIO (S3-compatible) storage configuration."""

    model_config = SettingsConfigDict(env_prefix="MINIO_")

    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "perfit-avatars"
    secure: bool = False  # Use HTTPS in production


class ImageValidationSettings(BaseSettings):
    """Image validation thresholds."""

    model_config = SettingsConfigDict(env_prefix="IMAGE_")

    min_width: int = 512
    min_height: int = 512
    max_width: int = 4096
    max_height: int = 4096
    max_file_size_mb: int = 10
    min_brightness: int = 50  # 0-255 scale
    max_brightness: int = 240
    allowed_formats: list[str] = Field(default_factory=lambda: ["jpeg", "jpg", "png", "webp"])


class MLSettings(BaseSettings):
    """Machine learning model configuration."""

    model_config = SettingsConfigDict(env_prefix="ML_")

    # HMR2 configuration
    hmr2_device: str = "cuda"  # or "cpu"
    detection_threshold: float = 0.5  # ViTDet person detection confidence

    # SMPL model path (for measurement extraction)
    smpl_model_path: str = "./ml/smpl/models"

    # Inference settings
    batch_size: int = 1


class Settings(BaseSettings):
    """Root application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "Perfit Avatar Service"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"  # development, staging, production

    # API
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # Task queue
    rq_queue_name: str = "avatar_extraction"
    job_timeout_seconds: int = 900  # 5 minutes max per job

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    minio: MinioSettings = Field(default_factory=MinioSettings)
    image_validation: ImageValidationSettings = Field(default_factory=ImageValidationSettings)
    ml: MLSettings = Field(default_factory=MLSettings)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Uses lru_cache to ensure settings are only loaded once per process.
    """
    return Settings()
