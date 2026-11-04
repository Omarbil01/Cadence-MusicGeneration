"""Public generation contract."""

from pydantic import BaseModel, ConfigDict, Field

from musicgen.shared.jobs import Genre


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    genre: Genre
    duration_seconds: int = Field(
        default=45,
        ge=5,
        le=240,
        strict=True,
        description="When omitted, uses the configured default capped by the configured maximum.",
    )
