"""Security behavior of persisted administrator accounts and opaque sessions."""

from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from musicgen.admin import AdminStore, RateLimited
from musicgen.admin.store import (
    DUMMY_PASSWORD_HASH,
    MAX_FAILURE_RECORDS,
    MAX_SESSIONS_PER_USER,
    WINDOW_SECONDS,
)

PASSWORD = "temporary-test-password-123"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def clock():
    return [1000.0]


@pytest.fixture
def store(tmp_path, clock):
    return AdminStore(tmp_path / "private" / "admin.sqlite3", clock=lambda: clock[0])


def test_private_lazy_database_and_salted_passwords(store):
    assert not store.path.exists()
    assert not store.has_users()
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert store.create_user("Owner", PASSWORD) == {"id": 1, "username": "owner"}
    store.create_user("second", PASSWORD)
    assert store.has_users()
    with sqlite3.connect(store.path) as connection:
        hashes = [row[0] for row in connection.execute("SELECT password_hash FROM admin_users")]
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert all(value.startswith("scrypt:") and PASSWORD not in value for value in hashes)
    assert hashes[0] != hashes[1]
    assert [u["username"] for u in store.list_users()] == ["owner", "second"]
    assert all("password_hash" not in user for user in store.list_users())
    with pytest.raises(ValueError, match="already exists"):
        store.create_user("OWNER", PASSWORD)


@pytest.mark.parametrize("username", ["", "bad name", "../owner", "éclair", "x" * 65])
def test_username_validation_prevents_ambiguous_identities(store, username):
    with pytest.raises(ValueError, match="Username"):
        store.create_user(username, PASSWORD)
    assert not store.path.exists()


@pytest.mark.parametrize("password", ["short", " " * 12, "x" * 1025])
def test_password_creation_policy(store, password):
    with pytest.raises(ValueError, match="Password"):
        store.create_user("owner", password)


def test_sessions_are_random_hashed_fixed_expiry_and_revocable(store, clock):
    user = store.create_user("owner", PASSWORD)
    token = store.authenticate("OWNER", PASSWORD, "127.0.0.1")
    second = store.authenticate("owner", PASSWORD, "127.0.0.1")
    assert token and second and token != second
    assert store.session_user(token) == user
    with sqlite3.connect(store.path) as connection:
        rows = connection.execute("SELECT token_hash, expires_at FROM admin_sessions").fetchall()
    assert hashlib.sha256(token.encode()).hexdigest() in {row[0] for row in rows}
    assert token not in {row[0] for row in rows}
    assert all(expiry == clock[0] + 28800 for _, expiry in rows)
    store.revoke_session(second)
    assert store.session_user(second) is None
    clock[0] += 28799
    assert store.session_user(token) == user
    clock[0] += 1
    assert store.session_user(token) is None
    assert store.session_user("untrusted-short-token") is None
    assert store.session_user(None) is None


def test_password_reset_revokes_all_sessions_and_changes_credentials(store):
    store.create_user("owner", PASSWORD)
    tokens = [store.authenticate("owner", PASSWORD, "127.0.0.1") for _ in range(2)]
    store.set_password("OWNER", "new-temporary-password-456")
    assert all(store.session_user(token) is None for token in tokens)
    assert store.authenticate("owner", PASSWORD, "127.0.0.1") is None
    assert store.authenticate("owner", "new-temporary-password-456", "127.0.0.1")
    with pytest.raises(ValueError, match="does not exist"):
        store.set_password("missing", PASSWORD)


def test_unknown_accounts_still_verify_a_hash_and_cannot_authorize(store, monkeypatch):
    checked = []

    def verify(password_hash, candidate):
        checked.append((password_hash, candidate))
        return True

    monkeypatch.setattr("musicgen.admin.store.check_password_hash", verify)
    assert store.authenticate("unknown", PASSWORD, "127.0.0.1") is None
    assert checked == [(DUMMY_PASSWORD_HASH, PASSWORD)]


def test_username_limit_persists_across_store_instances_and_expires(store, clock):
    store.create_user("owner", PASSWORD)
    for index in range(5):
        assert store.authenticate("owner", "wrong", f"192.0.2.{index}") is None
    reopened = AdminStore(store.path, clock=lambda: clock[0])
    with pytest.raises(RateLimited) as error:
        reopened.authenticate("OWNER", PASSWORD, "198.51.100.1")
    assert error.value.retry_after == WINDOW_SECONDS
    clock[0] += WINDOW_SECONDS
    assert reopened.authenticate("owner", PASSWORD, "198.51.100.1")


def test_ip_limit_spans_unknown_usernames_and_success_does_not_clear_failures(store):
    store.create_user("owner", PASSWORD)
    for index in range(19):
        assert store.authenticate(f"unknown{index}", "wrong", "192.0.2.1") is None
    assert store.authenticate("owner", PASSWORD, "192.0.2.1")
    assert store.authenticate("unknown19", "wrong", "192.0.2.1") is None
    with pytest.raises(RateLimited):
        store.authenticate("owner", PASSWORD, "192.0.2.1")
    assert store.authenticate("owner", PASSWORD, "192.0.2.2")


def test_parallel_failures_cannot_race_past_the_username_limit(store, monkeypatch):
    store.has_users()
    monkeypatch.setattr("musicgen.admin.store.check_password_hash", lambda *_: False)

    def fail(index):
        try:
            return store.authenticate("owner", "wrong", f"192.0.2.{index}")
        except RateLimited:
            return "limited"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fail, range(12)))
    assert results.count(None) == 5
    assert results.count("limited") == 7


def test_expired_failures_are_purged_and_global_history_is_bounded(store, clock):
    store.has_users()
    with sqlite3.connect(store.path) as connection:
        connection.executemany(
            "INSERT INTO admin_login_failures (username_key, ip_key, attempted_at) "
            "VALUES (?, ?, ?)",
            [(f"user{i}", f"ip{i}", clock[0]) for i in range(MAX_FAILURE_RECORDS)],
        )
    with pytest.raises(RateLimited):
        store.authenticate("unknown", "wrong", "192.0.2.1")
    clock[0] += WINDOW_SECONDS
    assert store.authenticate("unknown", "wrong", "192.0.2.1") is None
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM admin_login_failures").fetchone()[0] == 1


def test_session_history_has_a_per_user_bound(store, monkeypatch):
    store.create_user("owner", PASSWORD)
    monkeypatch.setattr("musicgen.admin.store.check_password_hash", lambda *_: True)
    tokens = [
        store.authenticate("owner", PASSWORD, "127.0.0.1") for _ in range(MAX_SESSIONS_PER_USER + 2)
    ]
    assert store.session_user(tokens[0]) is None
    assert store.session_user(tokens[-1]) == {"id": 1, "username": "owner"}
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM admin_sessions").fetchone()[0] == 20


def _cli():
    spec = importlib.util.spec_from_file_location(
        "manage_admin_cli", ROOT / "scripts/manage_admin.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_environment_initialization_is_private_idempotent_and_silent(tmp_path, capsys):
    from dotenv import dotenv_values

    cli = _cli()
    environment = tmp_path / "credentials.env"
    environment.write_text(
        "KEEP_THIS=value\nMUSICGEN_REDIS_PASSWORD=\n"
        "GF_SECURITY_ADMIN_PASSWORD=replace_with_your_local_password\n"
    )
    assert cli.main(["init-env", "--env-file", str(environment)]) == 0
    first = dotenv_values(environment)
    assert cli.REDIS_PASSWORD.fullmatch(first["MUSICGEN_REDIS_PASSWORD"])
    assert first["MUSICGEN_REDIS_PASSWORD"] != first["GF_SECURITY_ADMIN_PASSWORD"]
    assert first["KEEP_THIS"] == "value"
    assert stat.S_IMODE(environment.stat().st_mode) == 0o600
    assert cli.main(["init-env", "--env-file", str(environment)]) == 0
    assert dotenv_values(environment) == first
    output = capsys.readouterr().out
    assert first["MUSICGEN_REDIS_PASSWORD"] not in output
    assert first["GF_SECURITY_ADMIN_PASSWORD"] not in output


def test_cli_environment_copies_template_and_rejects_invalid_existing_secret(tmp_path, capsys):
    cli = _cli()
    destination = tmp_path / "new.env"
    assert cli.main(["init-env", "--env-file", str(destination)]) == 0
    assert "MUSICGEN_RUN_MODE=" in destination.read_text()
    invalid = tmp_path / "invalid.env"
    original = "MUSICGEN_REDIS_PASSWORD=short-private-value\nGF_SECURITY_ADMIN_PASSWORD=realpass\n"
    invalid.write_text(original)
    with pytest.raises(SystemExit) as error:
        cli.main(["init-env", "--env-file", str(invalid)])
    assert error.value.code == 2
    assert invalid.read_text() == original
    assert "short-private-value" not in capsys.readouterr().err


def test_cli_requires_private_password_entry_and_does_not_accept_password_argument(
    tmp_path, capsys
):
    cli = _cli()
    database = tmp_path / "private" / "admins.sqlite3"
    with pytest.raises(SystemExit):
        cli.main(["--database", str(database), "create", "owner", "--password", PASSWORD])
    assert not database.exists()
    with pytest.raises(SystemExit):
        cli.main(["--database", str(database), "create", "owner"])
    assert "interactive terminal" in capsys.readouterr().err
    assert not database.exists()
