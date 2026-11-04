"""Real Redis + Celery + FastAPI integration; uses a tiny untrained test checkpoint."""

import os
import socket
import subprocess
import sys
import time
from uuid import uuid4

import pytest
from conftest import wait_for_job
from fastapi.testclient import TestClient

from musicgen.api.main import create_app
from musicgen.shared.job_store import RedisStore

pytestmark = [pytest.mark.distributed, pytest.mark.checkpoint]


def test_real_celery_worker_request_to_download(settings, checkpoint, tmp_path):
    url = os.environ.get("MUSICGEN_TEST_REDIS_URL")
    if not url:
        pytest.skip("Set MUSICGEN_TEST_REDIS_URL to a dedicated Redis test database.")
    pytest.importorskip("torch")
    pytest.importorskip("redis")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    prefix = "musicgen-test-" + uuid4().hex
    settings = settings.model_copy(
        update={
            "run_mode": "distributed",
            "checkpoint_path": checkpoint,
            "job_store_url": url,
            "broker_url": url.rsplit("/", 1)[0] + "/14",
            "result_backend": url.rsplit("/", 1)[0] + "/13",
            "redis_prefix": prefix,
            "worker_metrics_port": port,
        }
    )
    store = RedisStore(settings)
    store.ping()
    env = dict(os.environ)
    env.update(
        {
            "MUSICGEN_RUN_MODE": "distributed",
            "MUSICGEN_CHECKPOINT_PATH": str(checkpoint),
            "MUSICGEN_OUTPUT_DIR": str(settings.output_dir),
            "MUSICGEN_JOB_STORE_URL": settings.job_store_url,
            "MUSICGEN_BROKER_URL": settings.broker_url,
            "MUSICGEN_RESULT_BACKEND": settings.result_backend,
            "MUSICGEN_REDIS_PREFIX": prefix,
            "MUSICGEN_WORKER_METRICS_PORT": str(port),
            "MUSICGEN_WORKER_HEARTBEAT_SECONDS": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    log_path = tmp_path / "celery.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "celery",
                "-A",
                "musicgen.worker.celery_app:app",
                "worker",
                "--pool=solo",
                "--concurrency=1",
                "--queues=generation",
                "--loglevel=WARNING",
                "--without-gossip",
                "--without-mingle",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while not any(w.get("ready") for w in store.workers().values()):
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("Worker did not become ready:\n" + log_path.read_text())
                time.sleep(0.1)
            with TestClient(create_app(settings)) as client:
                response = client.post(
                    "/api/generations", json={"genre": "rock_pop", "duration_seconds": 5}
                )
                assert response.status_code == 202, response.text
                job = wait_for_job(client, response.json()["id"], timeout=20)
                assert job["status"] == "completed", log_path.read_text()
                assert client.get(job["download_url"]).content.startswith(b"MThd")
                assert client.get(job["metadata_url"]).json()["context_mode"] == "framed"
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            for key in store.client.scan_iter(prefix + ":*"):
                store.client.delete(key)
            store.close()
