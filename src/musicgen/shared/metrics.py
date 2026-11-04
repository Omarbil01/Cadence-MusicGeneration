"""Bounded-label HTTP metrics shared by Flask and FastAPI."""

from prometheus_client import Counter, Histogram


class HTTPMetrics:
    def __init__(self, registry, service):
        self.service = service
        self.requests = Counter(
            "musicgen_http_requests_total",
            "HTTP requests by service and route template.",
            ("service", "method", "route", "status"),
            registry=registry,
        )
        self.duration = Histogram(
            "musicgen_http_request_duration_seconds",
            "HTTP response latency.",
            ("service", "method", "route"),
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
            registry=registry,
        )

    def record(self, method, route, status, duration):
        self.requests.labels(self.service, method, route, str(status)).inc()
        self.duration.labels(self.service, method, route).observe(duration)
