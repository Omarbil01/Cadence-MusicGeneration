"""Shared administrator login and a read-only monitoring landing page."""

import hashlib
import secrets
import sqlite3
import time
from functools import wraps
from urllib.parse import unquote, urlsplit

from flask import Blueprint, abort, g, make_response, redirect, render_template, request, url_for

from musicgen.admin.store import AdminStore, RateLimited

SESSION_COOKIE = "cadence_admin_session"
LOGIN_CSRF_COOKIE = "cadence_admin_login_csrf"


def safe_next(value):
    """Keep login redirects within admin pages on this origin."""
    if not isinstance(value, str) or len(value) > 2048:
        return "/admin/"
    decoded = unquote(unquote(value))
    if any(ord(char) < 32 or ord(char) == 127 for char in decoded) or "\\" in decoded:
        return "/admin/"
    try:
        parsed = urlsplit(decoded)
    except ValueError:
        return "/admin/"
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/admin/"):
        return "/admin/"
    if any(part in {".", ".."} for part in parsed.path.split("/")):
        return "/admin/"
    if parsed.path.rstrip("/") in {"/admin/login", "/admin/logout", "/admin/auth"}:
        return "/admin/"
    return value


def csrf_value(token, purpose):
    return hashlib.sha256(f"{purpose}:{token}".encode()).hexdigest()


def _csrf_matches(value, expected):
    return len(value) == 64 and value.isascii() and secrets.compare_digest(value, expected)


def _valid_origin():
    origin = request.headers.get("Origin")
    if not origin:
        return True  # Non-browser clients must still supply the bound CSRF token.
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc.lower() == request.host.lower()
        and not parsed.username
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )


def create_admin_blueprint(settings, store=None):
    blueprint = Blueprint("admin", __name__, url_prefix="/admin")
    store = store or AdminStore(
        settings.admin_db_path, session_seconds=settings.admin_session_seconds
    )

    def current_user():
        token = request.cookies.get(SESSION_COOKIE, "")
        try:
            user = store.session_user(token) if token else None
        except (OSError, sqlite3.Error):
            abort(503, "Administrator access is temporarily unavailable.")
        g.admin_token, g.admin_user = token, user
        return user

    def require_admin(view):
        @wraps(view)
        def guarded(*args, **kwargs):
            if not current_user():
                return redirect(url_for("admin.login", next=safe_next(request.full_path)))
            return view(*args, **kwargs)

        return guarded

    def render(name, **context):
        token = getattr(g, "admin_token", "")
        return render_template(
            name,
            user=getattr(g, "admin_user", None),
            logout_csrf=csrf_value(token, "logout") if token else "",
            setup_command=(
                "docker compose exec frontend python scripts/manage_admin.py create admin"
                if settings.run_mode == "distributed"
                else ".venv/bin/python scripts/manage_admin.py create admin"
            ),
            **context,
        )

    @blueprint.before_request
    def limit_form_size():
        if request.method == "POST" and request.content_length and request.content_length > 16384:
            abort(413)

    @blueprint.after_request
    def private_response(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'self'; img-src 'self'; "
            "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @blueprint.route("/login", methods=["GET", "POST"])
    def login():
        target = safe_next(request.values.get("next", "/admin/"))
        try:
            has_users = store.has_users()
        except (OSError, sqlite3.Error):
            return render("admin_setup.html", unavailable=True), 503
        if not has_users:
            return render("admin_setup.html", unavailable=False), 503
        if current_user():
            return redirect(target, code=303)
        nonce = request.cookies.get(LOGIN_CSRF_COOKIE, "")
        error, status, retry_after = None, 200, None
        if request.method == "POST":
            if (
                not _valid_origin()
                or not 32 <= len(nonce) <= 128
                or not _csrf_matches(request.form.get("csrf_token", ""), csrf_value(nonce, "login"))
            ):
                abort(400, "The sign-in form expired. Reload this page and try again.")
            try:
                token = store.authenticate(
                    request.form.get("username", ""),
                    request.form.get("password", ""),
                    request.remote_addr or "unknown",
                )
            except RateLimited as exc:
                token, status, retry_after = None, 429, exc.retry_after
                error = "Too many sign-in attempts. Please wait before trying again."
            except (OSError, sqlite3.Error):
                return render("admin_setup.html", unavailable=True), 503
            if token:
                response = redirect(target, code=303)
                response.set_cookie(
                    SESSION_COOKIE,
                    token,
                    max_age=settings.admin_session_seconds,
                    secure=settings.admin_cookie_secure,
                    httponly=True,
                    samesite="Lax",
                    path="/admin",
                )
                response.delete_cookie(LOGIN_CSRF_COOKIE, path="/admin/login")
                return response
            error = error or "The username or password is incorrect."
            status = 401 if status == 200 else status
        nonce = secrets.token_urlsafe(32)
        response = make_response(
            render(
                "admin_login.html",
                csrf_token=csrf_value(nonce, "login"),
                next_path=target,
                error=error,
            ),
            status,
        )
        response.set_cookie(
            LOGIN_CSRF_COOKIE,
            nonce,
            max_age=900,
            secure=settings.admin_cookie_secure,
            httponly=True,
            samesite="Strict",
            path="/admin/login",
        )
        if retry_after is not None:
            response.headers["Retry-After"] = str(retry_after)
        return response

    @blueprint.get("/")
    @require_admin
    def dashboard():
        return render("admin_dashboard.html", local_mode=settings.run_mode == "local")

    @blueprint.post("/logout")
    @require_admin
    def logout():
        if not _valid_origin() or not _csrf_matches(
            request.form.get("csrf_token", ""), csrf_value(g.admin_token, "logout")
        ):
            abort(400, "The sign-out form expired. Reload this page and try again.")
        store.revoke_session(g.admin_token)
        response = redirect(url_for("admin.login"), code=303)
        response.delete_cookie(SESSION_COOKIE, path="/admin")
        response.delete_cookie(LOGIN_CSRF_COOKIE, path="/admin/login")
        return response

    @blueprint.get("/auth")
    def authorize_proxy():
        # Identity comes exclusively from the server-side session. Forwarded
        # paths only select a safe post-login destination, never authorization.
        if not current_user():
            destination = safe_next(request.headers.get("X-Forwarded-Uri", "/admin/"))
            return redirect(url_for("admin.login", next=destination))
        response = make_response("", 204)
        response.headers["X-WEBAUTH-USER"] = g.admin_user["username"]
        return response

    @blueprint.get("/redis")
    @require_admin
    def redis_status():
        status = {"available": False, "local_mode": settings.run_mode == "local"}
        if not status["local_mode"]:
            client = None
            try:
                from redis import Redis

                client = Redis.from_url(
                    settings.job_store_url,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                    decode_responses=True,
                )
                client.ping()
                server, memory, clients = (
                    client.info("server"),
                    client.info("memory"),
                    client.info("clients"),
                )
                cutoff = time.time() - settings.job_retention_seconds
                status.update(
                    available=True,
                    version=server.get("redis_version", "unknown"),
                    uptime_seconds=server.get("uptime_in_seconds", 0),
                    memory=memory.get("used_memory_human", "unknown"),
                    connected_clients=clients.get("connected_clients", 0),
                    database_keys=client.dbsize(),
                    queued=client.zcount(f"{settings.redis_prefix}:active:queued", cutoff, "+inf"),
                    generating=client.zcount(
                        f"{settings.redis_prefix}:active:generating", cutoff, "+inf"
                    ),
                )
            except Exception:
                # Redis errors can contain credential-bearing URLs. Render no
                # exception messages, keys, values, configuration or commands.
                status["available"] = False
            finally:
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass
        return render("admin_redis.html", status=status)

    @blueprint.get("/prometheus")
    @blueprint.get("/prometheus/")
    @blueprint.get("/prometheus/<path:tool_path>")
    @blueprint.get("/grafana")
    @blueprint.get("/grafana/")
    @blueprint.get("/grafana/<path:tool_path>")
    @require_admin
    def tool_requires_proxy(tool_path=""):
        return render("admin_tool_unavailable.html"), 503

    return blueprint
