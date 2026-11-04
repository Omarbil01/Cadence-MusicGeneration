"""Local serial execution or distributed Celery submission."""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .service import WorkerService


class LocalDispatcher:
    def __init__(self, settings, store, registry, generator=None):
        self.service = WorkerService(settings, store, registry, generator)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="musicgen")
        self.pending = {}
        self._lock = Lock()

    def start(self):
        self.executor.submit(self.service.start).result()

    def submit(self, job_id):
        with self._lock:
            self.pending = {k: v for k, v in self.pending.items() if not v.done()}
            self.pending[job_id] = self.executor.submit(self.service.execute, job_id)

    def close(self):
        self.service.stop()
        self.executor.shutdown(wait=True, cancel_futures=True)


class CeleryDispatcher:
    def __init__(self, settings):
        from celery import Celery

        self.app = Celery("musicgen-client", broker=settings.broker_url)
        self.app.conf.update(
            task_serializer="json",
            accept_content=["json"],
            broker_connection_timeout=2,
            broker_transport_options={"socket_connect_timeout": 2, "socket_timeout": 2},
            task_publish_retry=False,
        )

    def start(self):
        pass

    def submit(self, job_id):
        self.app.send_task("musicgen.generate", args=[job_id], task_id=job_id, queue="generation")

    def close(self):
        self.app.close()
