"""Distributed worker entry point. Run a single solo process per model copy."""

from celery import Celery, signals
from prometheus_client import CollectorRegistry, ProcessCollector, start_http_server

from musicgen.shared.config import Settings
from musicgen.shared.job_store import RedisStore
from musicgen.shared.logging import configure_logging

from .service import WorkerService

settings = Settings()
app = Celery(
    "musicgen",
    broker=settings.broker_url,
    backend=settings.result_backend,
    include=["musicgen.worker.tasks"],
)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_default_queue="generation",
    worker_prefetch_multiplier=1,
    worker_concurrency=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": settings.generation_timeout_seconds + 300},
    result_backend_transport_options={
        "visibility_timeout": settings.generation_timeout_seconds + 300
    },
    visibility_timeout=settings.generation_timeout_seconds + 300,
    result_expires=settings.job_retention_seconds,
)
service = None


@signals.worker_init.connect
def initialize_worker(**kwargs):
    global service
    configure_logging(settings.log_level)
    registry = CollectorRegistry()
    ProcessCollector(registry=registry)
    service = WorkerService(settings, RedisStore(settings), registry)
    start_http_server(settings.worker_metrics_port, registry=registry)
    service.start()


@signals.worker_shutting_down.connect
def stop_worker(**kwargs):
    if service is not None:
        service.stop()
