"""Thread-safe local state and atomic Redis state for multiple API replicas."""

import json
import threading
import time
from typing import Protocol

from .config import Settings
from .jobs import Job, QueueFullError, StoreUnavailable, advance


class JobStore(Protocol):
    def create(self, job: Job) -> None: ...
    def get(self, job_id: str) -> Job | None: ...
    def transition(self, job_id: str, status: str, **changes) -> Job | None: ...
    def counts(self) -> dict[str, int]: ...
    def heartbeat(self, worker_id: str, payload: dict, ttl: int) -> None: ...
    def workers(self) -> dict[str, dict]: ...
    def remove_worker(self, worker_id: str) -> None: ...
    def ping(self) -> bool: ...
    def close(self) -> None: ...


class MemoryStore:
    """Single-process development backend; use Redis for API replicas."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._jobs: dict[str, Job] = {}
        self._workers: dict[str, tuple[float, dict]] = {}
        self._lock = threading.RLock()

    def _prune(self):
        cutoff = time.time() - self.settings.job_retention_seconds
        self._jobs = {k: v for k, v in self._jobs.items() if v.updated_at > cutoff}
        terminal = sorted(
            (v for v in self._jobs.values() if v.status in {"completed", "failed"}),
            key=lambda j: j.updated_at,
        )
        for job in terminal[:-1000]:
            self._jobs.pop(job.id, None)

    def create(self, job: Job):
        with self._lock:
            self._prune()
            if job.id in self._jobs:
                raise ValueError("Job ID already exists")
            if sum(j.status in {"queued", "generating"} for j in self._jobs.values()) >= (
                self.settings.max_pending_jobs
            ):
                raise QueueFullError("The generation queue is full.")
            self._jobs[job.id] = job.model_copy(deep=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._prune()
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def transition(self, job_id: str, status: str, **changes):
        with self._lock:
            job = self._jobs.get(job_id)
            updated = advance(job, status, **changes) if job else None
            if updated:
                self._jobs[job_id] = updated
            return updated

    def counts(self):
        with self._lock:
            self._prune()
            return {
                state: sum(j.status == state for j in self._jobs.values())
                for state in ("queued", "generating")
            }

    def heartbeat(self, worker_id: str, payload: dict, ttl: int):
        with self._lock:
            self._workers[worker_id] = (time.time() + ttl, dict(payload))

    def workers(self):
        with self._lock:
            now = time.time()
            self._workers = {k: v for k, v in self._workers.items() if v[0] > now}
            return {k: dict(v[1]) for k, v in self._workers.items()}

    def remove_worker(self, worker_id: str):
        with self._lock:
            self._workers.pop(worker_id, None)

    def ping(self):
        return True

    def close(self):
        pass


class RedisStore:
    def __init__(self, settings: Settings, client=None):
        import redis

        self.settings = settings
        self.client = client or redis.Redis.from_url(
            settings.job_store_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        self.prefix = settings.redis_prefix

    def _key(self, kind: str, identifier: str):
        return f"{self.prefix}:{kind}:{identifier}"

    def _active(self):
        return [self._key("active", s) for s in ("queued", "generating")]

    def create(self, job: Job):
        from redis.exceptions import RedisError, WatchError

        key = self._key("job", job.id)
        active = self._active()
        try:
            for _ in range(20):
                with self.client.pipeline() as pipe:
                    try:
                        pipe.watch(key, *active)
                        if pipe.exists(key):
                            raise ValueError("Job ID already exists")
                        cutoff = time.time() - self.settings.job_retention_seconds
                        count = sum(pipe.zcount(k, cutoff, "+inf") for k in active)
                        if count >= self.settings.max_pending_jobs:
                            raise QueueFullError("The generation queue is full.")
                        pipe.multi()
                        for active_key in active:
                            pipe.zremrangebyscore(active_key, "-inf", cutoff)
                        pipe.set(key, job.model_dump_json(), ex=self.settings.job_retention_seconds)
                        pipe.zadd(active[0], {job.id: job.updated_at})
                        pipe.execute()
                        return
                    except WatchError:
                        continue
            raise StoreUnavailable("Job state is busy. Please retry.")
        except RedisError as exc:
            raise StoreUnavailable("Job storage is unavailable.") from exc

    def get(self, job_id: str):
        from redis.exceptions import RedisError

        try:
            raw = self.client.get(self._key("job", job_id))
            return Job.model_validate_json(raw) if raw else None
        except RedisError as exc:
            raise StoreUnavailable("Job storage is unavailable.") from exc

    def transition(self, job_id: str, status: str, **changes):
        from redis.exceptions import RedisError, WatchError

        key = self._key("job", job_id)
        try:
            for _ in range(20):
                with self.client.pipeline() as pipe:
                    try:
                        pipe.watch(key)
                        raw = pipe.get(key)
                        if raw is None:
                            return None
                        updated = advance(Job.model_validate_json(raw), status, **changes)
                        if updated is None:
                            return None
                        pipe.multi()
                        pipe.set(
                            key, updated.model_dump_json(), ex=self.settings.job_retention_seconds
                        )
                        for active in self._active():
                            pipe.zrem(active, job_id)
                        if status in {"queued", "generating"}:
                            pipe.zadd(self._key("active", status), {job_id: updated.updated_at})
                        pipe.execute()
                        return updated
                    except WatchError:
                        continue
            raise StoreUnavailable("Job state is busy. Please retry.")
        except RedisError as exc:
            raise StoreUnavailable("Job storage is unavailable.") from exc

    def counts(self):
        from redis.exceptions import RedisError

        try:
            cutoff = time.time() - self.settings.job_retention_seconds
            return {
                s: self.client.zcount(self._key("active", s), cutoff, "+inf")
                for s in ("queued", "generating")
            }
        except RedisError as exc:
            raise StoreUnavailable("Job storage is unavailable.") from exc

    def heartbeat(self, worker_id: str, payload: dict, ttl: int):
        from redis.exceptions import RedisError

        try:
            self.client.set(self._key("worker", worker_id), json.dumps(payload), ex=ttl)
        except RedisError as exc:
            raise StoreUnavailable("Job storage is unavailable.") from exc

    def workers(self):
        from redis.exceptions import RedisError

        try:
            keys = list(self.client.scan_iter(match=self._key("worker", "*"), count=100))
            values = self.client.mget(keys) if keys else []
            return {
                key.rsplit(":", 1)[-1]: json.loads(value)
                for key, value in zip(keys, values, strict=True)
                if value
            }
        except RedisError as exc:
            raise StoreUnavailable("Job storage is unavailable.") from exc

    def remove_worker(self, worker_id: str):
        from redis.exceptions import RedisError

        try:
            self.client.delete(self._key("worker", worker_id))
        except RedisError as exc:
            raise StoreUnavailable("Job storage is unavailable.") from exc

    def ping(self):
        from redis.exceptions import RedisError

        try:
            return bool(self.client.ping())
        except RedisError as exc:
            raise StoreUnavailable("Job storage is unavailable.") from exc

    def close(self):
        self.client.close()
