"""Worker metrics come from actual inference attempts."""

from prometheus_client import Counter, Gauge, Histogram


class WorkerMetrics:
    def __init__(self, registry):
        self.ready = Gauge(
            "musicgen_worker_ready", "Model has loaded and accepts jobs.", registry=registry
        )
        self.load_seconds = Gauge(
            "musicgen_model_load_seconds", "Time to load the current model.", registry=registry
        )
        self.active = Gauge(
            "musicgen_worker_active_jobs", "Jobs executing in this worker.", registry=registry
        )
        self.finished = Counter(
            "musicgen_generation_jobs_total",
            "Terminal inference attempts.",
            ("genre", "outcome"),
            registry=registry,
        )
        self.duration = Histogram(
            "musicgen_generation_duration_seconds",
            "Inference attempt elapsed time.",
            ("genre",),
            buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1200, 1800),
            registry=registry,
        )
        self.wait = Histogram(
            "musicgen_queue_wait_seconds",
            "Queue wait before inference begins.",
            buckets=(0.1, 1, 5, 15, 30, 60, 120, 300, 600, 1800),
            registry=registry,
        )
