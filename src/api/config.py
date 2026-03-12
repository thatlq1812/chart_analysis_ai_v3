"""
API Configuration

Pydantic Settings model for FastAPI server configuration.
Values are loaded from environment variables or .env file.

Usage:
    from src.api.config import get_settings, Settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    API Server configuration.

    All settings are read from environment variables (case-insensitive).
    Prefix: API_ (e.g., API_HOST, API_PORT).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="API_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Server ---
    host: str = Field(default="0.0.0.0", description="Bind address.")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port.")
    workers: int = Field(default=1, ge=1, description="Number of uvicorn workers.")
    reload: bool = Field(default=False, description="Enable hot-reload (dev only).")

    # --- Application ---
    app_name: str = Field(default="Geo-SLM Chart Analysis API")
    app_version: str = Field(default="3.0.0")
    debug: bool = Field(default=False)

    # --- CORS ---
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins. Use ['*'] for development.",
    )

    # --- Upload limits ---
    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum accepted file upload size in megabytes.",
    )

    # --- Storage ---
    data_dir: Path = Field(
        default=Path("data"),
        description="Root data directory for input/output files.",
    )
    upload_dir: Path = Field(
        default=Path("data/uploads"),
        description="Temporary directory for uploaded files.",
    )
    output_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory for processed pipeline outputs.",
    )

    # --- Pipeline defaults ---
    default_pdf_dpi: int = Field(default=150, ge=72, le=300)
    default_extract_context: bool = Field(default=True)

    # --- Task queue (future Celery integration) ---
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis broker URL for Celery task queue.",
    )
    celery_enabled: bool = Field(
        default=False,
        description=(
            "Enable Celery async task queue. "
            "When False, jobs run in-process via FastAPI BackgroundTasks."
        ),
    )

    # --- Job storage ---
    job_ttl_seconds: int = Field(
        default=86400,
        description="Time-to-live for job records in seconds (default: 24h).",
    )

    def setup_directories(self) -> None:
        """Create required directories if they do not exist."""
        for directory in [self.data_dir, self.upload_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Use as a FastAPI dependency:
        settings: Settings = Depends(get_settings)
    """
    settings = Settings()
    settings.setup_directories()
    return settings
