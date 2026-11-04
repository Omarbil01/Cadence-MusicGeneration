"""Shared admin access is enforced before exposing monitoring tools."""

import re
from urllib.parse import parse_qs, urlsplit

import pytest

from musicgen.admin.store import AdminStore
from musicgen.frontend.admin import LOGIN_CSRF_COOKIE, SESSION_COOKIE, safe_next
from musicgen.frontend.app import create_app

PASSWORD = "a-long-test-password-42"


@pytest.fixture
def admin_app(settings, tmp_path):
    settings.admin_db_path = tmp_path / "admin" / "accounts.sqlite3"
    store = AdminStore(settings.admin_db_path, session_seconds=settings.admin_session_seconds)
    store.create_user("admin", PASSWORD)
    app = create_app(settings)
    app.config["TESTING"] = True
    return app, store, settings


def hidden(response, name):
    return re.search(rf'name="{name}" value="([^"]*)"', response.get_data(as_text=True)).group(1)


def sign_in(client, *, username="admin", password=PASSWORD, next_path="/admin/"):
    page = client.get("/admin/login", query_string={"next": next_path})
    return client.post(
        "/admin/login",
        data={
            "username": username,
            "password": password,
            "csrf_token": hidden(page, "csrf_token"),
            "next": next_path,
        },
    )


def test_music_studio_remains_available_without_admin_setup(settings, tmp_path):
    settings.admin_db_path = tmp_path / "admin" / "empty.sqlite3"
    client = create_app(settings).test_client()
    page = client.get("/")
    assert page.status_code == 200 and b"Administration" in page.data
    assert not settings.admin_db_path.exists()  # Ordinary generation does not initialize auth.
    setup = client.get("/admin/login")
    assert setup.status_code == 503
    assert b"scripts/manage_admin.py create admin" in setup.data
    assert client.get("/health/live").status_code == 200


def test_docker_setup_instructions_target_the_running_frontend_database(settings):
    settings.run_mode = "distributed"
    client = create_app(settings).test_client()
    response = client.get("/admin/login")
    assert response.status_code == 503
    assert (
        b"docker compose exec frontend python scripts/manage_admin.py create admin" in response.data
    )
    assert b".venv/bin/python" not in response.data


@pytest.mark.parametrize(
    "path",
    ["/admin/", "/admin/redis", "/admin/prometheus/", "/admin/grafana/api/user"],
)
def test_monitoring_pages_require_an_actual_admin_session(admin_app, path):
    client = admin_app[0].test_client()
    response = client.get(path, headers={"X-WEBAUTH-USER": "admin"})
    assert response.status_code == 302
    assert urlsplit(response.location).path == "/admin/login"


def test_shared_login_cookie_and_logout_revoke_proxy_access(admin_app):
    client = admin_app[0].test_client()
    login = sign_in(client)
    assert login.status_code == 303 and login.location == "/admin/"
    cookie = next(x for x in login.headers.getlist("Set-Cookie") if x.startswith(SESSION_COOKIE))
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie and "Path=/admin" in cookie
    assert "Max-Age=" in cookie
    page = client.get("/admin/")
    assert page.status_code == 200
    assert b"/admin/prometheus/" in page.data and b"/admin/grafana/" in page.data
    assert "no-store" in page.headers["Cache-Control"]
    assert "form-action 'self'" in page.headers["Content-Security-Policy"]
    auth = client.get("/admin/auth", headers={"X-WEBAUTH-USER": "spoofed"})
    assert auth.status_code == 204 and auth.headers["X-WEBAUTH-USER"] == "admin"
    logout = client.post("/admin/logout", data={"csrf_token": hidden(page, "csrf_token")})
    assert logout.status_code == 303
    assert client.get("/admin/auth").status_code == 302


def test_login_and_logout_reject_missing_or_foreign_csrf(admin_app):
    client = admin_app[0].test_client()
    assert (
        client.post("/admin/login", data={"username": "admin", "password": PASSWORD}).status_code
        == 400
    )
    page = client.get("/admin/login")
    foreign = client.post(
        "/admin/login",
        data={"username": "admin", "password": PASSWORD, "csrf_token": hidden(page, "csrf_token")},
        headers={"Origin": "https://other.example"},
    )
    assert foreign.status_code == 400
    assert sign_in(client).status_code == 303
    assert client.post("/admin/logout").status_code == 400
    assert client.get("/admin/auth").status_code == 204


def test_login_failures_are_generic_and_throttled(admin_app):
    client = admin_app[0].test_client()
    known = sign_in(client, password="incorrect-password")
    unknown = sign_in(client, username="not-an-admin", password="incorrect-password")
    assert known.status_code == unknown.status_code == 401
    assert b"The username or password is incorrect." in known.data
    assert b"The username or password is incorrect." in unknown.data
    for _ in range(5):
        response = sign_in(client, password="incorrect-password")
    assert response.status_code == 429 and int(response.headers["Retry-After"]) > 0
    assert client.get("/admin/auth").status_code == 302


@pytest.mark.parametrize(
    "target",
    [
        "https://other.example/admin/",
        "https://[invalid/admin/",
        "//other.example/admin/",
        "/admin/../../outside",
        "/admin/%2e%2e/outside",
        "/admin/\\other.example",
        "/admin/login",
        "/outside",
        "/admin/\r\nLocation:other",
    ],
)
def test_login_redirects_stay_inside_admin_pages(admin_app, target):
    assert safe_next(target) == "/admin/"
    client = admin_app[0].test_client()
    response = sign_in(client, next_path=target)
    assert response.status_code == 303 and response.location == "/admin/"


def test_forwarded_path_never_authorizes_anonymous_requests(admin_app):
    client = admin_app[0].test_client()
    response = client.get(
        "/admin/auth",
        headers={
            "X-Forwarded-Uri": "https://other.example/",
            "X-Forwarded-Prefix": "/admin/login",
            "X-WEBAUTH-USER": "admin",
        },
    )
    assert response.status_code == 302
    assert "X-WEBAUTH-USER" not in response.headers
    assert parse_qs(urlsplit(response.location).query)["next"] == ["/admin/"]


def test_malformed_csrf_and_oversized_forms_fail_without_server_error(admin_app):
    client = admin_app[0].test_client()
    client.get("/admin/login")
    assert client.post("/admin/login", data={"csrf_token": "é" * 64}).status_code == 400
    assert client.post("/admin/login", data={"password": "x" * 17000}).status_code == 413


def test_redis_status_is_read_only_and_does_not_expose_credentials(admin_app, monkeypatch):
    app, _, settings = admin_app
    settings.run_mode = "distributed"
    settings.job_store_url = "redis://:secret-password@redis:6379/2"
    calls = []

    class FakeRedis:
        def ping(self):
            calls.append("ping")

        def info(self, section):
            calls.append("info:" + section)
            return {
                "redis_version": "7.4.0",
                "uptime_in_seconds": 1234,
                "used_memory_human": "1M",
                "connected_clients": 4,
                "requirepass": "secret-password",
                "private_value": "must-not-render",
            }

        def dbsize(self):
            calls.append("dbsize")
            return 7

        def zcount(self, key, start, stop):
            calls.append("zcount:" + key)
            return 2 if key.endswith("queued") else 1

        def close(self):
            calls.append("close")

    monkeypatch.setattr("redis.Redis.from_url", lambda *a, **k: FakeRedis())
    client = app.test_client()
    sign_in(client)
    response = client.get("/admin/redis")
    assert response.status_code == 200 and b"7.4.0" in response.data
    assert b"secret-password" not in response.data and b"must-not-render" not in response.data
    assert "zcount:musicgen:active:queued" in calls
    assert set(calls) <= {
        "ping",
        "info:server",
        "info:memory",
        "info:clients",
        "dbsize",
        "close",
        "zcount:musicgen:active:queued",
        "zcount:musicgen:active:generating",
    }


def test_redis_failure_has_no_connection_or_password_details(admin_app, monkeypatch):
    app, _, settings = admin_app
    settings.run_mode = "distributed"

    def unavailable(*args, **kwargs):
        raise ConnectionError("redis://:private-password@internal-host/2")

    monkeypatch.setattr("redis.Redis.from_url", unavailable)
    client = app.test_client()
    sign_in(client)
    response = client.get("/admin/redis")
    assert response.status_code == 200 and b"Redis is currently unavailable" in response.data
    assert b"private-password" not in response.data and b"internal-host" not in response.data


def test_direct_frontend_tool_routes_fail_closed(admin_app):
    client = admin_app[0].test_client()
    sign_in(client)
    assert client.get("/admin/prometheus/").status_code == 503
    assert client.get("/admin/grafana/").status_code == 503
    assert client.post("/admin/grafana/api/admin/users").status_code == 405


def test_secure_cookie_configuration_and_password_reset_revoke_sessions(admin_app):
    app, store, settings = admin_app
    settings.admin_cookie_secure = True
    client = app.test_client()
    response = sign_in(client)
    cookie = next(x for x in response.headers.getlist("Set-Cookie") if x.startswith(SESSION_COOKIE))
    assert "Secure" in cookie
    assert client.get_cookie(LOGIN_CSRF_COOKIE, path="/admin/login") is None
    store.set_password("admin", "another-long-password-42")
    assert client.get("/admin/auth").status_code == 302
