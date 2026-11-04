"""Small SQLite identity store for a single-host administrator portal.

Passwords use Werkzeug's salted scrypt hashes. Browser session tokens are
opaque random values; only their SHA-256 digests are stored. Credentials and
tokens must not be logged. Each operation opens its own SQLite connection.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

USERNAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", re.ASCII)
TOKEN = re.compile(r"[A-Za-z0-9_-]{32,128}", re.ASCII)
WINDOW_SECONDS = 900
USERNAME_FAILURE_LIMIT = 5
IP_FAILURE_LIMIT = 20
MAX_FAILURE_RECORDS = 10000
MAX_SESSIONS_PER_USER = 20

# This public dummy hash never authorizes a user. Unknown accounts still perform
# the same password derivation as an existing account to avoid a fast lookup path.
DUMMY_PASSWORD_HASH = (
    "scrypt:32768:8:1$XQONmAmBqQkX96Gc$"
    "a80ce672503f8321504a435ef6e03172dcb191b9bf52be61c484c6f8b2af3bc57"
    "8f16a85e95491782f8bdd909839b0b36d49ec00316e39a4927cd957d4e95363"
)


class RateLimited(RuntimeError):
    """A persisted sign-in attempt limit has been reached."""

    def __init__(self, retry_after: int):
        super().__init__("Too many sign-in attempts. Try again later.")
        self.retry_after = max(1, int(retry_after))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _username(value: str) -> str:
    if not isinstance(value, str) or not USERNAME.fullmatch(value):
        raise ValueError(
            "Username must be 1–64 ASCII letters, digits, dots, underscores or hyphens, "
            "starting with a letter or digit."
        )
    return value.lower()


def _password(value: str) -> str:
    if not isinstance(value, str) or not 12 <= len(value) <= 1024 or value.isspace():
        raise ValueError("Password must contain 12–1024 characters and cannot be only spaces.")
    return value


class AdminStore:
    """SQLite users, fixed-expiry sessions, and bounded login throttling.

    The database is initialized lazily. A newly created parent directory has
    mode 0700 and the database mode is enforced as 0600. Existing parent folders
    are left unchanged because an explicit database override may use a shared
    application directory. Use a dedicated private directory for deployment.
    """

    def __init__(
        self,
        path: Path | str,
        session_seconds: int = 28800,
        *,
        clock: Callable[[], float] | None = None,
    ):
        if (
            isinstance(session_seconds, bool)
            or not isinstance(session_seconds, int)
            or session_seconds <= 0
        ):
            raise ValueError("Session duration must be positive.")
        self.path = Path(path)
        self.session_seconds = int(session_seconds)
        self._clock = clock or time.time
        self._initialized = False
        self._initialization_lock = threading.Lock()

    def _open(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self):
        if self._initialized:
            return
        with self._initialization_lock:
            if self._initialized:
                return
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.path, flags, 0o600)
            try:
                os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)
            connection = self._open()
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS admin_users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                        password_hash TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        password_changed_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS admin_sessions (
                        token_hash TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES admin_users(id) ON DELETE CASCADE,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS admin_session_user
                        ON admin_sessions(user_id, created_at);
                    CREATE INDEX IF NOT EXISTS admin_session_expiry
                        ON admin_sessions(expires_at);
                    CREATE TABLE IF NOT EXISTS admin_login_failures (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username_key TEXT NOT NULL,
                        ip_key TEXT NOT NULL,
                        attempted_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS admin_failures_username
                        ON admin_login_failures(username_key, attempted_at);
                    CREATE INDEX IF NOT EXISTS admin_failures_ip
                        ON admin_login_failures(ip_key, attempted_at);
                    CREATE INDEX IF NOT EXISTS admin_failures_time
                        ON admin_login_failures(attempted_at);
                    """
                )
                connection.commit()
            finally:
                connection.close()
            self._initialized = True

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self._initialize()
        connection = self._open()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _cleanup(connection: sqlite3.Connection, now: float):
        connection.execute("DELETE FROM admin_sessions WHERE expires_at <= ?", (now,))
        connection.execute(
            "DELETE FROM admin_login_failures WHERE attempted_at <= ?",
            (now - WINDOW_SECONDS,),
        )

    def create_user(self, username: str, password: str) -> dict:
        username = _username(username)
        password_hash = generate_password_hash(_password(password), method="scrypt")
        now = self._clock()
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    "INSERT INTO admin_users "
                    "(username, password_hash, created_at, password_changed_at) "
                    "VALUES (?, ?, ?, ?)",
                    (username, password_hash, now, now),
                )
                return {"id": cursor.lastrowid, "username": username}
        except sqlite3.IntegrityError as exc:
            raise ValueError("That administrator username already exists.") from exc

    def has_users(self) -> bool:
        with self._connection() as connection:
            return connection.execute("SELECT 1 FROM admin_users LIMIT 1").fetchone() is not None

    def list_users(self) -> list[dict]:
        with self._connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT id, username, created_at, password_changed_at "
                    "FROM admin_users ORDER BY username"
                )
            ]

    @staticmethod
    def _retry_after(
        connection: sqlite3.Connection, username_key: str, ip_key: str, now: float
    ) -> int:
        waits = []
        for column, value, limit in (
            ("username_key", username_key, USERNAME_FAILURE_LIMIT),
            ("ip_key", ip_key, IP_FAILURE_LIMIT),
        ):
            # Column identifiers are internal constants, never user input.
            row = connection.execute(
                f"SELECT attempted_at FROM admin_login_failures WHERE {column} = ? "
                "ORDER BY attempted_at DESC LIMIT 1 OFFSET ?",
                (value, limit - 1),
            ).fetchone()
            if row is not None:
                waits.append(row["attempted_at"] + WINDOW_SECONDS - now)
        total = connection.execute("SELECT COUNT(*) FROM admin_login_failures").fetchone()[0]
        if total >= MAX_FAILURE_RECORDS:
            oldest = connection.execute(
                "SELECT MIN(attempted_at) FROM admin_login_failures"
            ).fetchone()[0]
            waits.append(oldest + WINDOW_SECONDS - now)
        return max(1, math.ceil(max(waits))) if waits else 0

    def authenticate(self, username: str, password: str, client_ip: str) -> str | None:
        """Return a new opaque session, None for invalid credentials, or RateLimited.

        An attempt is reserved transactionally before expensive password checking.
        Successful attempts remove only their own reservation, so concurrent
        failures cannot bypass thresholds or be erased by another user's success.
        """
        raw_username = username if isinstance(username, str) else ""
        normalized = raw_username[:256].lower()
        username_key = _digest(normalized)
        ip_key = _digest(str(client_ip or "unknown")[:128])
        now = self._clock()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._cleanup(connection, now)
            wait = self._retry_after(connection, username_key, ip_key, now)
            if wait:
                raise RateLimited(wait)
            reservation = connection.execute(
                "INSERT INTO admin_login_failures (username_key, ip_key, attempted_at) "
                "VALUES (?, ?, ?)",
                (username_key, ip_key, now),
            ).lastrowid
            user = (
                connection.execute(
                    "SELECT id, username, password_hash FROM admin_users WHERE username = ?",
                    (normalized,),
                ).fetchone()
                if USERNAME.fullmatch(raw_username)
                else None
            )
        candidate = password if isinstance(password, str) and len(password) <= 1024 else ""
        password_hash = user["password_hash"] if user is not None else DUMMY_PASSWORD_HASH
        try:
            verified = check_password_hash(password_hash, candidate)
        except (ValueError, TypeError):
            verified = False
        if not verified or user is None:
            return None
        token = secrets.token_urlsafe(32)
        issued_at = self._clock()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT password_hash FROM admin_users WHERE id = ?", (user["id"],)
            ).fetchone()
            if current is None or not hmac.compare_digest(current["password_hash"], password_hash):
                return None
            self._cleanup(connection, issued_at)
            connection.execute("DELETE FROM admin_login_failures WHERE id = ?", (reservation,))
            connection.execute(
                "DELETE FROM admin_sessions WHERE user_id = ? AND token_hash IN ("
                "SELECT token_hash FROM admin_sessions WHERE user_id = ? "
                "ORDER BY created_at DESC, rowid DESC LIMIT -1 OFFSET ?)",
                (user["id"], user["id"], MAX_SESSIONS_PER_USER - 1),
            )
            connection.execute(
                "INSERT INTO admin_sessions (token_hash, user_id, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (_digest(token), user["id"], issued_at, issued_at + self.session_seconds),
            )
        return token

    def session_user(self, token: str) -> dict | None:
        if not isinstance(token, str) or not TOKEN.fullmatch(token):
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT u.id, u.username FROM admin_sessions AS s "
                "JOIN admin_users AS u ON u.id = s.user_id "
                "WHERE s.token_hash = ? AND s.expires_at > ?",
                (_digest(token), self._clock()),
            ).fetchone()
            return dict(row) if row else None

    def revoke_session(self, token: str):
        if not isinstance(token, str) or not TOKEN.fullmatch(token):
            return
        with self._connection() as connection:
            connection.execute("DELETE FROM admin_sessions WHERE token_hash = ?", (_digest(token),))

    def set_password(self, username: str, password: str):
        username = _username(username)
        password_hash = generate_password_hash(_password(password), method="scrypt")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id FROM admin_users WHERE username = ?", (username,)
            ).fetchone()
            if row is None:
                raise ValueError("That administrator username does not exist.")
            connection.execute(
                "UPDATE admin_users SET password_hash = ?, password_changed_at = ? WHERE id = ?",
                (password_hash, self._clock(), row["id"]),
            )
            connection.execute("DELETE FROM admin_sessions WHERE user_id = ?", (row["id"],))
