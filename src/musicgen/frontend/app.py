"""Flask template frontend; no inference or queue dependencies are needed here."""

import time

from flask import Flask, Response, g, jsonify, render_template, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    ProcessCollector,
    generate_latest,
)

from musicgen.shared.config import Settings
from musicgen.shared.metrics import HTTPMetrics

from .admin import create_admin_blueprint


def create_app(settings=None):
    settings = settings or Settings()
    app = Flask(__name__)
    registry = CollectorRegistry()
    ProcessCollector(registry=registry)
    metrics = HTTPMetrics(registry, "frontend")
    api_base_url = settings.api_base_url.rstrip("/")
    if not api_base_url and settings.run_mode == "local":
        api_base_url = "http://localhost:8000"
    app.register_blueprint(create_admin_blueprint(settings))

    @app.before_request
    def begin_request():
        g.started_at = time.monotonic()

    @app.after_request
    def end_request(response):
        if request.path != "/metrics":
            route = request.url_rule.rule if request.url_rule else "unmatched"
            metrics.record(
                request.method, route, response.status_code, time.monotonic() - g.started_at
            )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers.setdefault(
            "Content-Security-Policy",
            (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                f"connect-src 'self' {api_base_url}; img-src 'self' data:; "
                "base-uri 'self'; frame-ancestors 'none'"
            ),
        )
        return response

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            app_config={
                "apiBaseUrl": api_base_url,
                "defaultDuration": min(
                    settings.default_duration_seconds, settings.max_duration_seconds
                ),
                "maxDuration": settings.max_duration_seconds,
            },
        )

    @app.get("/health/live")
    def health():
        return jsonify(status="ok")

    @app.get("/metrics")
    def prometheus():
        return Response(generate_latest(registry), content_type=CONTENT_TYPE_LATEST)

    return app
