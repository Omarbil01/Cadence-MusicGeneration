"""Liveness, readiness and the settings the frontend needs."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from musicgen.shared.jobs import StoreUnavailable

router = APIRouter(prefix="/api", tags=["service"])


@router.get("/health/live")
def live():
    return {"status": "ok"}


@router.get("/health/ready")
def ready(request: Request):
    runtime = request.app.state.runtime
    try:
        runtime.store.ping()
        workers = runtime.store.workers()
        available = any(w.get("ready") for w in workers.values())
        reason = "ready" if available else "generator_unavailable"
    except StoreUnavailable:
        available, reason = False, "job_store_unavailable"
        workers = {}
    return JSONResponse(
        {
            "ready": available,
            "reason": reason,
            "workers": len(workers),
            "worker_reasons": sorted({w.get("reason", "unknown") for w in workers.values()}),
        },
        status_code=200 if available else 503,
    )


@router.get("/config")
def configuration(request: Request):
    settings = request.app.state.runtime.settings
    return {
        "genres": [
            {"id": "classical", "label": "Classical"},
            {"id": "romantic", "label": "Romantic"},
            {"id": "rock_pop", "label": "Rock & Pop"},
        ],
        "min_duration_seconds": 5,
        "max_duration_seconds": settings.max_duration_seconds,
        "default_duration_seconds": min(
            settings.default_duration_seconds, settings.max_duration_seconds
        ),
        "output_format": "MIDI",
    }
