"""Celery messages contain only a job ID; settings live in the shared job record."""

from musicgen.shared.jobs import StoreUnavailable

from .celery_app import app


@app.task(
    name="musicgen.generate",
    ignore_result=True,
    autoretry_for=(StoreUnavailable,),
    retry_backoff=True,
    retry_backoff_max=60,
    max_retries=None,
)
def generate(job_id):
    # Keep the delivery until storage recovers. Expired or already-claimed jobs are no-ops.
    from .celery_app import service

    if service is None:
        raise RuntimeError("Generation worker has not initialized")
    service.execute(job_id)
