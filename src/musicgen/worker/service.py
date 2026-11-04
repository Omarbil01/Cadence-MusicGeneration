"""A serial worker service usable from Celery or the local development runner."""

import logging
import threading
import time
from uuid import uuid4

from musicgen.generation.generator import MidiGenerator, ModelUnavailable
from musicgen.shared.storage import OutputStore

from .metrics import WorkerMetrics

logger = logging.getLogger(__name__)


class WorkerService:
    def __init__(self, settings, store, registry, generator=None):
        self.settings = settings
        self.store = store
        self.engine = generator if generator is not None else MidiGenerator(settings)
        self.outputs = OutputStore(settings.output_dir)
        self.metrics = WorkerMetrics(registry)
        self.worker_id = uuid4().hex
        self.ready = False
        self.reason = "loading"
        self._quit = threading.Event()
        self._heartbeat_thread = None
        self._serial = threading.Lock()
        self._active_job = None

    def _beat(self):
        self.store.heartbeat(
            self.worker_id,
            {"ready": self.ready, "reason": self.reason, "active_job_id": self._active_job},
            ttl=max(15, self.settings.worker_heartbeat_seconds * 3),
        )

    def _heartbeat(self):
        while not self._quit.wait(self.settings.worker_heartbeat_seconds):
            try:
                self._beat()
            except Exception:
                logger.exception("Worker heartbeat failed")

    def start(self):
        self._beat()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._heartbeat_thread.start()
        started = time.monotonic()
        try:
            self.engine.load()
            self.ready = True
            self.reason = "ready"
            self.metrics.ready.set(1)
            logger.info("Generation worker ready worker_id=%s", self.worker_id)
        except ModelUnavailable as exc:
            self.reason = exc.code
            logger.warning("Generation unavailable code=%s detail=%s", exc.code, exc)
        except Exception:
            self.reason = "model_load_failed"
            logger.exception("Model initialization failed")
        self.metrics.load_seconds.set(time.monotonic() - started)
        self._beat()

    def execute(self, job_id):
        with self._serial:
            job = self.store.get(job_id)
            if job is None or job.status != "queued":
                return
            job = self.store.transition(job_id, "generating", worker_id=self.worker_id)
            if job is None:
                return
            self._active_job = job_id
            self.metrics.wait.observe(max(0, time.time() - job.created_at))
            self.metrics.active.set(1)
            started = time.monotonic()
            outcome = "failed"
            try:
                if not self.ready:
                    raise RuntimeError("The music generator is unavailable.")
                self.outputs.prepare(job_id)
                metadata = self.engine.generate(
                    genre=job.genre,
                    duration_seconds=job.duration_seconds,
                    output_path=self.outputs.path(job_id),
                    seed=job.seed,
                    cancelled=self._quit.is_set,
                )
                self.outputs.write_metadata(job_id, {"job_id": job_id, **metadata})
                completed = self.store.transition(
                    job_id,
                    "completed",
                    actual_duration_seconds=metadata["actual_duration_seconds"],
                )
                outcome = "completed" if completed else "failed"
                logger.info("Generation finished job_id=%s status=%s", job_id, outcome)
            except Exception as exc:
                logger.exception("Generation failed job_id=%s", job_id)
                message = (
                    str(exc)
                    if isinstance(exc, TimeoutError)
                    else "Music generation failed. Please try another request."
                )
                self.store.transition(job_id, "failed", error=message)
            finally:
                self._active_job = None
                self.metrics.active.set(0)
                self.metrics.finished.labels(job.genre, outcome).inc()
                self.metrics.duration.labels(job.genre).observe(time.monotonic() - started)

    def stop(self):
        self.ready = False
        self.metrics.ready.set(0)
        self._quit.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=3)
        try:
            self.store.remove_worker(self.worker_id)
        except Exception:
            logger.warning("Worker heartbeat removal failed during shutdown")
