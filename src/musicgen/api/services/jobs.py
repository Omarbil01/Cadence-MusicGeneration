"""Job submission, public representations, and interrupted-worker detection."""

import logging
import secrets
import time

from musicgen.shared.jobs import Job

logger = logging.getLogger(__name__)


def public_job(job: Job):
    base = f"/api/generations/{job.id}"
    completed = job.status == "completed"
    return {
        "id": job.id,
        "status": job.status,
        "genre": job.genre,
        "duration_seconds": job.duration_seconds,
        "actual_duration_seconds": job.actual_duration_seconds,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error": job.error,
        "status_url": base,
        "download_url": base + "/download" if completed else None,
        "metadata_url": base + "/metadata" if completed else None,
    }


def submit(runtime, request):
    if not any(w.get("ready") for w in runtime.store.workers().values()):
        raise RuntimeError("Music generation is currently unavailable.")
    job = Job(
        genre=request.genre,
        duration_seconds=request.duration_seconds,
        seed=secrets.randbelow(2**31),
    )
    runtime.store.create(job)
    try:
        runtime.dispatcher.submit(job.id)
    except Exception as exc:
        runtime.store.transition(job.id, "failed", error="The job could not be queued.")
        logger.exception("Queue submission failed job_id=%s", job.id)
        raise RuntimeError("The generation queue is unavailable. Please try again.") from exc
    return job


def refresh_interrupted(runtime, job):
    worker = runtime.store.workers().get(job.worker_id) if job.status == "generating" else None
    inactive = worker is None or ("active_job_id" in worker and worker["active_job_id"] != job.id)
    if (
        job.status == "generating"
        and job.worker_id
        and job.started_at is not None
        and time.time() - job.started_at > 30
        and inactive
    ):
        updated = runtime.store.transition(
            job.id,
            "failed",
            error="Generation was interrupted. Please submit a new request.",
        )
        return updated or runtime.store.get(job.id) or job
    return job
