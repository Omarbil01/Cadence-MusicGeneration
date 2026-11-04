"""Environment settings shared by the web services and inference worker."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MUSICGEN_", env_file=".env", extra="ignore")

    run_mode: Literal["local", "distributed"] = "local"
    checkpoint_path: Path = Path("models/best.pt")
    output_dir: Path = Path("storage/generated")
    device: Literal["auto", "cpu", "cuda"] = "auto"
    torch_threads: int = Field(default=2, ge=1, le=64)
    palette: Literal["original", "band"] = "original"
    ending_style: Literal["release", "none"] = "release"
    repetition_penalty: float = Field(default=0.3, ge=0, le=2)
    max_duration_seconds: int = Field(default=240, ge=5, le=240)
    default_duration_seconds: int = Field(default=45, ge=5, le=240)
    generation_timeout_seconds: int = Field(default=1800, ge=5, le=7200)
    max_tokens: int = Field(default=30000, ge=100, le=100000)
    max_pending_jobs: int = Field(default=32, ge=1, le=1000)
    job_retention_seconds: int = Field(default=604800, ge=60)
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/1"
    job_store_url: str = "redis://localhost:6379/2"
    redis_prefix: str = "musicgen"
    worker_metrics_port: int = Field(default=9101, ge=1, le=65535)
    worker_heartbeat_seconds: int = Field(default=5, ge=1, le=30)
    frontend_origin: str = "http://localhost:5000"
    api_base_url: str = ""
    log_level: str = "INFO"
    admin_db_path: Path = Path("storage/admin/admin.sqlite3")
    admin_session_seconds: int = Field(default=28800, ge=300, le=86400)
    admin_cookie_secure: bool = False
