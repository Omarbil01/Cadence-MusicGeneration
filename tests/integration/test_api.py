import time
from uuid import uuid4

import fakeredis
import pytest
from conftest import FixtureGenerator, wait_for_job
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from musicgen.api.main import create_app
from musicgen.shared.job_store import MemoryStore, RedisStore
from musicgen.shared.jobs import Job, StoreUnavailable
from musicgen.worker.service import WorkerService


def test_missing_model_keeps_web_api_alive_but_blocks_generation(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health/live").status_code == 200
        readiness = client.get("/api/health/ready")
        assert readiness.status_code == 503
        assert "checkpoint_missing" in readiness.json()["worker_reasons"]
        result = client.post("/api/generations", json={"genre": "rock_pop", "duration_seconds": 45})
        assert result.status_code == 503
        assert client.app.state.runtime.store.counts()["queued"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"genre": "jazz", "duration_seconds": 45},
        {"genre": "classical", "duration_seconds": 4},
        {"genre": "classical", "duration_seconds": 241},
        {"genre": "classical", "duration_seconds": 45.5},
        {"genre": "classical", "duration_seconds": True},
        {"genre": "classical", "duration_seconds": "45"},
        {"genre": "classical", "duration_seconds": None},
        {"genre": "classical", "duration_seconds": 45, "checkpoint": "/tmp/other.pt"},
    ],
)
def test_input_contract(settings, fixture_generator, payload):
    with TestClient(create_app(settings, generator=fixture_generator)) as client:
        assert client.post("/api/generations", json=payload).status_code == 422
        assert fixture_generator.calls == 0


def test_generate_poll_and_download(settings, fixture_generator):
    with TestClient(create_app(settings, generator=fixture_generator)) as client:
        submitted = client.post(
            "/api/generations", json={"genre": "rock_pop", "duration_seconds": 45}
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["id"]
        job = wait_for_job(client, job_id)
        assert job["status"] == "completed"
        midi = client.get(job["download_url"])
        assert midi.status_code == 200 and midi.content.startswith(b"MThd")
        assert "rock_pop_45s_" in midi.headers["content-disposition"]
        metadata = client.get(job["metadata_url"]).json()
        assert metadata["job_id"] == job_id
        assert metadata["actual_duration_seconds"] == 45
        assert fixture_generator.loads == 1 and fixture_generator.calls == 1
        assert client.get("/api/generations/" + uuid4().hex).status_code == 404
        assert client.get("/api/generations/not-a-job/download").status_code == 404
        metrics = client.get("/metrics").text
        assert job_id not in metrics
        assert "musicgen_generation_jobs_total" in metrics
        assert 'route="/api/generations/{job_id}"' in metrics


def test_failure_is_terminal_and_has_no_download(settings):
    with TestClient(create_app(settings, generator=FixtureGenerator(fail=True))) as client:
        response = client.post(
            "/api/generations", json={"genre": "romantic", "duration_seconds": 5}
        )
        job_id = response.json()["id"]
        job = wait_for_job(client, job_id)
        assert job["status"] == "failed"
        assert "/private/" not in job["error"]
        assert job["download_url"] is None
        assert client.get("/api/generations/" + job_id + "/download").status_code == 409


class NoopDispatcher:
    def start(self):
        pass

    def submit(self, job_id):
        pass

    def close(self):
        pass


def test_docs_config_and_configured_duration_limit(settings, fixture_generator):
    settings.max_duration_seconds = 30
    with TestClient(create_app(settings, generator=fixture_generator)) as client:
        assert client.get("/api/docs").status_code == 200
        assert client.get("/api/openapi.json").status_code == 200
        config = client.get("/api/config").json()
        assert config["max_duration_seconds"] == 30
        assert config["default_duration_seconds"] == 30
        response = client.post(
            "/api/generations", json={"genre": "classical", "duration_seconds": 31}
        )
        assert response.status_code == 422
        assert fixture_generator.calls == 0


@pytest.mark.parametrize(
    ("configured_default", "configured_maximum", "expected_duration"),
    [(15, 30, 15), (45, 30, 30), (60, 240, 60)],
)
def test_omitted_duration_uses_configured_default(
    settings, fixture_generator, configured_default, configured_maximum, expected_duration
):
    settings.default_duration_seconds = configured_default
    settings.max_duration_seconds = configured_maximum
    with TestClient(create_app(settings, generator=fixture_generator)) as client:
        config = client.get("/api/config").json()
        assert config["default_duration_seconds"] == expected_duration
        response = client.post("/api/generations", json={"genre": "classical"})
        assert response.status_code == 202
        job = wait_for_job(client, response.json()["id"])
        assert job["status"] == "completed"
        assert job["duration_seconds"] == expected_duration
        assert job["actual_duration_seconds"] == expected_duration


def test_explicit_duration_overrides_configured_default(settings, fixture_generator):
    settings.default_duration_seconds = 15
    settings.max_duration_seconds = 60
    with TestClient(create_app(settings, generator=fixture_generator)) as client:
        response = client.post(
            "/api/generations", json={"genre": "classical", "duration_seconds": 45}
        )
        assert response.status_code == 202
        job = wait_for_job(client, response.json()["id"])
        assert job["status"] == "completed"
        assert job["duration_seconds"] == 45
        assert job["actual_duration_seconds"] == 45


def test_api_replicas_share_job_and_download_state(settings, fixture_generator):
    settings.run_mode = "distributed"
    server = fakeredis.FakeServer()
    stores = [
        RedisStore(settings, fakeredis.FakeRedis(server=server, decode_responses=True))
        for _ in range(2)
    ]
    worker = WorkerService(settings, stores[0], CollectorRegistry(), fixture_generator)
    worker.start()
    try:
        with (
            TestClient(create_app(settings, store=stores[0], dispatcher=NoopDispatcher())) as one,
            TestClient(create_app(settings, store=stores[1], dispatcher=NoopDispatcher())) as two,
        ):
            response = one.post(
                "/api/generations", json={"genre": "classical", "duration_seconds": 5}
            )
            job_id = response.json()["id"]
            assert two.get("/api/generations/" + job_id).json()["status"] == "queued"
            assert two.get("/api/generations/" + job_id + "/download").status_code == 409
            worker.execute(job_id)
            worker.execute(job_id)
            job = two.get("/api/generations/" + job_id).json()
            assert job["status"] == "completed"
            assert two.get(job["download_url"]).content.startswith(b"MThd")
            assert fixture_generator.calls == 1
    finally:
        worker.stop()


def test_full_queue_and_dispatch_failure(settings):
    settings.max_pending_jobs = 1
    store = MemoryStore(settings)
    store.heartbeat("test", {"ready": True}, 60)
    app = create_app(settings, store=store, dispatcher=NoopDispatcher())
    with TestClient(app) as client:
        first = client.post("/api/generations", json={"genre": "classical", "duration_seconds": 5})
        assert first.status_code == 202
        second = client.post("/api/generations", json={"genre": "romantic", "duration_seconds": 5})
        assert second.status_code == 429
        assert second.headers["retry-after"] == "5"


def test_interrupted_worker_is_reported_as_failed(settings):
    store = MemoryStore(settings)
    job = Job(genre="romantic", duration_seconds=5, seed=1)
    store.create(job)
    store.transition(job.id, "generating", worker_id="gone")
    store._jobs[job.id].started_at = time.time() - 60
    with TestClient(create_app(settings, store=store, dispatcher=NoopDispatcher())) as client:
        response = client.get("/api/generations/" + job.id)
        assert response.json()["status"] == "failed"
        assert "interrupted" in response.json()["error"]


def test_worker_returning_idle_cannot_leave_job_generating_forever(settings):
    store = MemoryStore(settings)
    job = Job(genre="romantic", duration_seconds=5, seed=1)
    store.create(job)
    store.transition(job.id, "generating", worker_id="idle-worker")
    store._jobs[job.id].started_at = time.time() - 60
    store.heartbeat("idle-worker", {"ready": True, "active_job_id": None}, 60)
    with TestClient(create_app(settings, store=store, dispatcher=NoopDispatcher())) as client:
        assert client.get("/api/generations/" + job.id).json()["status"] == "failed"


def test_store_outage_returns_503(settings):
    class UnavailableStore(MemoryStore):
        def workers(self):
            raise StoreUnavailable("redis details must not be exposed")

    store = UnavailableStore(settings)
    with TestClient(create_app(settings, store=store, dispatcher=NoopDispatcher())) as client:
        response = client.post(
            "/api/generations", json={"genre": "classical", "duration_seconds": 5}
        )
        assert response.status_code == 503


def test_queue_publish_failure_marks_record_failed(settings):
    class BrokenDispatcher(NoopDispatcher):
        def submit(self, job_id):
            raise ConnectionError("broker unavailable")

    store = MemoryStore(settings)
    store.heartbeat("test", {"ready": True}, 60)
    with TestClient(create_app(settings, store=store, dispatcher=BrokenDispatcher())) as client:
        response = client.post(
            "/api/generations", json={"genre": "classical", "duration_seconds": 5}
        )
        assert response.status_code == 503
        assert store.counts() == {"queued": 0, "generating": 0}
        assert next(iter(store._jobs.values())).status == "failed"
