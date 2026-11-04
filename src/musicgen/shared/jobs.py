"""Serializable job records; storage and HTTP representations stay separate."""

import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

GENRES = ("classical", "romantic", "rock_pop")
Genre = Literal["classical", "romantic", "rock_pop"]
Status = Literal["queued", "generating", "completed", "failed"]
TERMINAL = {"completed", "failed"}
TRANSITIONS = {
    "queued": {"generating", "failed"},
    "generating": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
}


class Job(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    genre: Genre
    duration_seconds: int
    seed: int
    status: Status = "queued"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    worker_id: str | None = None
    actual_duration_seconds: float | None = None
    error: str | None = None


class QueueFullError(RuntimeError):
    pass


class StoreUnavailable(RuntimeError):
    pass


def advance(job: Job, status: Status, **changes) -> Job | None:
    if status not in TRANSITIONS[job.status]:
        return None
    now = time.time()
    fields = {"status": status, "updated_at": now, **changes}
    if status == "generating":
        fields["started_at"] = now
    if status in TERMINAL:
        fields["completed_at"] = now
    return job.model_copy(update=fields)
