"""Check the real shared login at an explicitly selected isolated Compose stack."""

import os
import re
from urllib.parse import urlsplit

import httpx
import pytest

pytestmark = pytest.mark.admin_stack


def test_shared_admin_login_gates_real_monitoring_services():
    base = os.environ.get("MUSICGEN_TEST_ADMIN_URL")
    if not base:
        pytest.skip("Set MUSICGEN_TEST_ADMIN_URL to a dedicated admin test stack.")
    username = os.environ.get("MUSICGEN_TEST_ADMIN_USERNAME")
    password = os.environ.get("MUSICGEN_TEST_ADMIN_PASSWORD")
    if not username or not password:
        pytest.fail("Configure the dedicated test administrator username and password.")
    paths = (
        "/admin/redis",
        "/admin/prometheus/",
        "/admin/prometheus/api/v1/status/runtimeinfo",
        "/admin/grafana/",
        "/admin/grafana/api/user",
    )
    with httpx.Client(base_url=base, timeout=15, follow_redirects=False) as client:
        for path in paths:
            denied = client.get(
                path,
                headers={
                    "X-WEBAUTH-USER": username,
                    "X-Forwarded-Prefix": "/admin/login",
                },
            )
            assert denied.status_code == 302, (path, denied.status_code)
            destination = urlsplit(denied.headers["location"])
            assert destination.path == "/admin/login"
            assert destination.netloc in {"", urlsplit(base).netloc}
        page = client.get("/admin/login")
        assert page.status_code == 200
        csrf = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
        response = client.post(
            "/admin/login",
            data={"username": username, "password": password, "csrf_token": csrf},
            headers={"Origin": base.rstrip("/")},
        )
        assert response.status_code == 303
        prometheus = client.get("/admin/prometheus/api/v1/status/runtimeinfo")
        assert prometheus.status_code == 200 and prometheus.json()["status"] == "success"
        grafana = client.get("/admin/grafana/api/user")
        assert grafana.status_code == 200
        assert grafana.json()["login"] == username.lower()
        datasource = client.get("/admin/grafana/api/datasources/uid/cadence-prometheus/health")
        assert datasource.status_code == 200 and datasource.json()["status"] == "OK"
        redis = client.get("/admin/redis")
        assert redis.status_code == 200 and "Redis is currently unavailable" not in redis.text
        assert 'class="healthy">Available' in redis.text
        overview = client.get("/admin/")
        csrf = re.search(r'name="csrf_token" value="([^"]+)"', overview.text).group(1)
        assert client.post("/admin/logout", data={"csrf_token": csrf}).status_code == 303
        for path in paths:
            response = client.get(path)
            assert response.status_code == 302, path
            assert urlsplit(response.headers["location"]).netloc in {"", urlsplit(base).netloc}
        signed_out = client.get("/admin/grafana/api/user", follow_redirects=True)
        assert signed_out.status_code == 200 and 'name="password"' in signed_out.text
