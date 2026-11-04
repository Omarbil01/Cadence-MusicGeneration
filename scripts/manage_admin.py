"""Create, reset or list local administrator accounts using interactive passwords."""

from __future__ import annotations

import argparse
import getpass
import io
import os
import re
import secrets
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musicgen.admin import AdminStore  # noqa: E402
from musicgen.shared.config import Settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REDIS_PASSWORD = re.compile(r"[A-Za-z0-9_-]{32,}", re.ASCII)
GRAFANA_PLACEHOLDERS = {
    "replace_with_your_local_password",
    "replace_with_a_strong_password",
    "replace-me",
    "change-me",
    "change_me",
    "changeme",
}


def configure_environment(env_file: Path) -> list[str]:
    """Populate missing machine credentials without exposing or replacing real ones."""
    from dotenv import dotenv_values

    env_file = Path(env_file)
    if env_file.is_symlink():
        raise ValueError("Use a regular environment file, not a symbolic link.")
    source = env_file if env_file.exists() else ROOT / ".env.example"
    content = source.read_text()
    values = dotenv_values(stream=io.StringIO(content), interpolate=False)
    redis_password = values.get("MUSICGEN_REDIS_PASSWORD") or ""
    updates = {}
    if not redis_password:
        updates["MUSICGEN_REDIS_PASSWORD"] = secrets.token_urlsafe(36)
    elif not REDIS_PASSWORD.fullmatch(redis_password):
        raise ValueError(
            "Existing MUSICGEN_REDIS_PASSWORD must contain at least 32 URL-safe "
            "letters, digits, underscores or hyphens. Its value was not changed."
        )
    grafana_password = values.get("GF_SECURITY_ADMIN_PASSWORD") or ""
    if not grafana_password or grafana_password.lower() in GRAFANA_PLACEHOLDERS:
        updates["GF_SECURITY_ADMIN_PASSWORD"] = secrets.token_urlsafe(36)
    for name, value in updates.items():
        expression = re.compile(rf"^[ \t]*(?:export[ \t]+)?{name}[ \t]*=.*$", re.MULTILINE)
        content, replacements = expression.subn(f"{name}={value}", content)
        if not replacements:
            content = content.rstrip("\n") + f"\n{name}={value}\n"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".env.admin-", dir=env_file.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(content)
        os.replace(temporary, env_file)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return list(updates)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, help="Override the configured administrator database"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser(
        "init-env", help="Initialize local machine credentials without printing their values"
    )
    initialize.add_argument("--env-file", type=Path, default=ROOT / ".env")
    for name, help_text in (
        ("create", "Create an administrator; password is entered privately"),
        ("reset", "Reset a password and revoke that administrator's existing sessions"),
        ("list", "List administrator usernames without password hashes or session tokens"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--database", type=Path, default=argparse.SUPPRESS)
        if name != "list":
            command.add_argument("username")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(value == "--password" or value.startswith("--password=") for value in arguments):
        parser.error("Password arguments are not supported; enter passwords at the private prompt.")
    args = parser.parse_args(arguments)
    if args.command == "init-env":
        try:
            configured = configure_environment(args.env_file)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if configured:
            print("Configured " + ", ".join(configured) + "; environment file mode is 0600.")
        else:
            print("Machine credentials already configured; environment file mode is 0600.")
        return 0
    path = args.database if args.database is not None else Settings().admin_db_path
    store = AdminStore(path)
    if args.command == "list":
        users = store.list_users()
        if not users:
            print("No administrator accounts. Use the create command to add one.")
        for user in users:
            created = datetime.fromtimestamp(user["created_at"], UTC).isoformat()
            print(f"{user['id']}\t{user['username']}\t{created}")
        return 0
    if not sys.stdin.isatty():
        parser.error("Password entry requires an interactive terminal.")
    try:
        password = getpass.getpass("New password (at least 12 characters): ")
        confirmation = getpass.getpass("Confirm password: ")
    except (EOFError, KeyboardInterrupt):
        print("\nPassword entry cancelled.", file=sys.stderr)
        return 1
    if password != confirmation:
        parser.error("Passwords do not match.")
    try:
        if args.command == "create":
            user = store.create_user(args.username, password)
            print(f"Created administrator '{user['username']}'.")
        else:
            store.set_password(args.username, password)
            print(f"Reset password for '{args.username.lower()}' and revoked existing sessions.")
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
