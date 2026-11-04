"""Create, poll, and download generated compositions."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from musicgen.api.schemas.generation import GenerationRequest
from musicgen.api.services.jobs import public_job, refresh_interrupted, submit
from musicgen.shared.jobs import QueueFullError
from musicgen.shared.storage import JOB_ID

router = APIRouter(prefix="/api/generations", tags=["generation"])


def known_job(request, job_id):
    if not JOB_ID.fullmatch(job_id):
        raise HTTPException(404, "Generation not found.")
    runtime = request.app.state.runtime
    job = runtime.store.get(job_id)
    if job is None:
        raise HTTPException(404, "Generation not found.")
    return refresh_interrupted(runtime, job)


@router.post("", status_code=202)
def create_generation(body: GenerationRequest, request: Request):
    runtime = request.app.state.runtime
    if "duration_seconds" not in body.model_fields_set:
        body = body.model_copy(
            update={
                "duration_seconds": min(
                    runtime.settings.default_duration_seconds,
                    runtime.settings.max_duration_seconds,
                )
            }
        )
    if body.duration_seconds > runtime.settings.max_duration_seconds:
        raise HTTPException(422, "The duration exceeds the configured maximum.")
    try:
        return public_job(submit(runtime, body))
    except QueueFullError as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "5"}) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/{job_id}")
def generation_status(job_id: str, request: Request):
    return public_job(known_job(request, job_id))


def completed_file(request, job_id, name):
    job = known_job(request, job_id)
    if job.status != "completed":
        raise HTTPException(409, "This generation has no completed download.")
    try:
        path = request.app.state.runtime.outputs.path(job_id, name)
    except ValueError as exc:
        raise HTTPException(404, "Output file is unavailable.") from exc
    if not path.is_file():
        raise HTTPException(404, "Output file is unavailable.")
    return job, path


@router.get("/{job_id}/download")
def download_midi(job_id: str, request: Request):
    job, path = completed_file(request, job_id, "output.mid")
    return FileResponse(
        path,
        media_type="audio/midi",
        filename=f"{job.genre}_{job.duration_seconds}s_{job.id[:8]}.mid",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{job_id}/metadata")
def download_metadata(job_id: str, request: Request):
    _, path = completed_file(request, job_id, "metadata.json")
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"{job_id}_metadata.json",
        headers={"Cache-Control": "no-store"},
    )
