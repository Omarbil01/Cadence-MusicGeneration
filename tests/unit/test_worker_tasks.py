import pytest
from prometheus_client import CollectorRegistry

from musicgen.shared.job_store import MemoryStore
from musicgen.shared.jobs import Job, StoreUnavailable
from musicgen.worker import celery_app
from musicgen.worker.service import WorkerService
from musicgen.worker.tasks import generate


@pytest.mark.parametrize("failure_at", ["read", "claim"])
def test_task_recovers_from_store_outage(settings, fixture_generator, monkeypatch, failure_at):
    class RecoveringStore(MemoryStore):
        failures_remaining = 5

        def fail_if_unavailable(self, operation):
            if operation == failure_at and self.failures_remaining:
                self.failures_remaining -= 1
                raise StoreUnavailable("Temporary job store outage")

        def get(self, job_id):
            self.fail_if_unavailable("read")
            return super().get(job_id)

        def transition(self, job_id, status, **changes):
            if status == "generating":
                self.fail_if_unavailable("claim")
            return super().transition(job_id, status, **changes)

    store = RecoveringStore(settings)
    job = Job(genre="classical", duration_seconds=5, seed=1)
    store.create(job)
    worker = WorkerService(settings, store, CollectorRegistry(), fixture_generator)
    monkeypatch.setattr(celery_app, "service", worker)
    worker.start()
    try:
        # Eager execution runs Celery's retry lifecycle without sleeping or using a broker.
        result = generate.apply(args=[job.id])
        assert result.successful(), result.result
        assert store.failures_remaining == 0
        assert store.get(job.id).status == "completed"
        assert store.counts() == {"queued": 0, "generating": 0}
        assert worker.outputs.path(job.id).read_bytes().startswith(b"MThd")
        assert generate.apply(args=[job.id]).successful()
        assert fixture_generator.calls == 1
    finally:
        worker.stop()
