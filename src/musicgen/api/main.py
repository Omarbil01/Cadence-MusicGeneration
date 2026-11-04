"""FastAPI entry point. The distributed API never loads PyTorch."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Gauge,
    ProcessCollector,
    generate_latest,
)

from musicgen.api.routes import generations, health
from musicgen.shared.config import Settings
from musicgen.shared.job_store import MemoryStore, RedisStore
from musicgen.shared.jobs import StoreUnavailable
from musicgen.shared.logging import configure_logging
from musicgen.shared.metrics import HTTPMetrics
from musicgen.shared.storage import OutputStore
from musicgen.worker.dispatch import CeleryDispatcher, LocalDispatcher

logger = logging.getLogger(__name__)


def create_app(settings=None, *, store=None, dispatcher=None, generator=None):
    settings = settings or Settings()
    configure_logging(settings.log_level)
    registry = CollectorRegistry()
    ProcessCollector(registry=registry)
    http_metrics = HTTPMetrics(registry, "api")
    if store is None:
        store = MemoryStore(settings) if settings.run_mode == "local" else RedisStore(settings)
    if dispatcher is None:
        dispatcher = (
            LocalDispatcher(settings, store, registry, generator)
            if settings.run_mode == "local"
            else CeleryDispatcher(settings)
        )
    runtime = SimpleNamespace(
        settings=settings,
        store=store,
        dispatcher=dispatcher,
        outputs=OutputStore(settings.output_dir),
    )
    queues = Gauge(
        "musicgen_queue_jobs", "Jobs waiting or executing.", ("state",), registry=registry
    )

    def queue_count(state):
        try:
            return store.counts()[state]
        except StoreUnavailable:
            return float("nan")

    for state in ("queued", "generating"):
        queues.labels(state).set_function(lambda state=state: queue_count(state))

    @asynccontextmanager
    async def lifespan(app):
        await asyncio.to_thread(dispatcher.start)
        try:
            yield
        finally:
            await asyncio.to_thread(dispatcher.close)
            store.close()

    app = FastAPI(
        title="Music Generation API",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    app.state.runtime = runtime
    allowed = [settings.frontend_origin.rstrip("/")]
    if settings.run_mode == "local":
        allowed += ["http://localhost:5000", "http://127.0.0.1:5000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(set(allowed)),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started, status = time.monotonic(), 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response
        finally:
            if request.url.path != "/metrics":
                route = getattr(request.scope.get("route"), "path", "unmatched")
                http_metrics.record(request.method, route, status, time.monotonic() - started)

    @app.exception_handler(StoreUnavailable)
    async def unavailable(request, exc):
        logger.error("Job storage unavailable: %s", exc)
        return JSONResponse({"detail": "The service is temporarily unavailable."}, status_code=503)

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    app.include_router(health.router)
    app.include_router(generations.router)
    return app


app = create_app()
