import threading
import time
from concurrent.futures import ThreadPoolExecutor

import fakeredis
import pytest

from musicgen.shared.job_store import MemoryStore, RedisStore
from musicgen.shared.jobs import Job, QueueFullError


@pytest.fixture(params=["memory", "redis"])
def store(request, settings):
    settings.max_pending_jobs = 3
    if request.param == "memory":
        return MemoryStore(settings)
    return RedisStore(settings, fakeredis.FakeRedis(decode_responses=True))


def test_queue_capacity_and_terminal_transition(store):
    jobs = [Job(genre="classical", duration_seconds=5, seed=n) for n in range(3)]
    for job in jobs:
        store.create(job)
    with pytest.raises(QueueFullError):
        store.create(Job(genre="classical", duration_seconds=5, seed=9))
    first = store.transition(jobs[0].id, "generating", worker_id="worker")
    assert first.started_at is not None
    assert store.transition(first.id, "generating") is None
    assert store.transition(first.id, "completed", actual_duration_seconds=5)
    assert store.transition(first.id, "failed", error="late duplicate") is None
    assert store.get(first.id).status == "completed"
    assert store.counts() == {"queued": 2, "generating": 0}
    store.create(Job(genre="romantic", duration_seconds=5, seed=10))
    assert store.get("unknown") is None


def test_only_one_worker_claims_a_job(store):
    job = Job(genre="rock_pop", duration_seconds=5, seed=1)
    store.create(job)
    barrier = threading.Barrier(8)

    def claim():
        barrier.wait()
        return store.transition(job.id, "generating", worker_id="worker")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: claim(), range(8)))
    assert sum(result is not None for result in results) == 1


def test_concurrent_queue_limit_is_atomic(store):
    def create(seed):
        try:
            store.create(Job(genre="classical", duration_seconds=5, seed=seed))
            return True
        except QueueFullError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(create, range(12)))
    assert sum(results) == 3
    assert store.counts()["queued"] == 3


def test_expired_job_is_unknown(store):
    job = Job(
        genre="classical",
        duration_seconds=5,
        seed=1,
        updated_at=time.time() - store.settings.job_retention_seconds - 10,
    )
    store.create(job)
    if isinstance(store, RedisStore):
        store.client.delete(store._key("job", job.id))
    assert store.get(job.id) is None


def test_worker_presence_is_shared(settings):
    server = fakeredis.FakeServer()
    one = RedisStore(settings, fakeredis.FakeRedis(server=server, decode_responses=True))
    two = RedisStore(settings, fakeredis.FakeRedis(server=server, decode_responses=True))
    one.heartbeat("worker", {"ready": True}, 15)
    assert two.workers()["worker"]["ready"]
    one.remove_worker("worker")
    assert two.workers() == {}
