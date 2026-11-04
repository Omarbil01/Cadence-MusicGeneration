# Admin login and monitoring

The studio's **Admin** link opens one administrator login for Prometheus, Grafana
and a read-only Redis status page. SQLite stores the human accounts and sessions;
Redis remains the Celery broker, result backend and distributed job store.
RabbitMQ is not required for this architecture. Celery supports Redis and
RabbitMQ as alternative brokers; this application already uses Redis for its
small job-ID messages. See [Celery's broker overview](https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/index.html).

## First local login

Install the frontend dependencies using the [README](../README.md), then run
these commands from the repository root:

```bash
.venv/bin/python scripts/manage_admin.py init-env
.venv/bin/python scripts/manage_admin.py create admin
```

The first command creates or fills missing private `.env` settings and generates
random machine secrets without displaying them. Existing configured values are
preserved. The second prompts privately for your chosen password and confirmation.
Passwords must contain at least 12 characters. No default browser password or
administrator account is supplied.

Start Flask and FastAPI using the README commands. Open
[localhost:5000/admin/](http://localhost:5000/admin/) or choose **Admin** in the
studio. Local accounts are stored at `storage/admin/admin.sqlite3` by default.

Local mode runs the generator inside FastAPI with in-memory jobs. It does not
start Prometheus, Grafana or Redis. The local admin portal explains that dashboard
access needs the full Compose stack; its Redis page reports local mode.

## First Docker login

Local and Compose accounts use separate databases. After initializing `.env` as
above, create the Compose account in its persistent volume:

```bash
docker compose build frontend
docker compose run --rm --no-deps frontend python scripts/manage_admin.py create admin
docker compose up --build -d
```

The account command prompts for your password in the terminal. It does not need
Redis or a loaded model. Install the intended compatible checkpoint at
`models/best.pt` before testing model generation. An override used by a local API
process does not change Docker's model.

| Page | Compose address |
|---|---|
| Sign in / admin portal | [localhost:8080/admin/](http://localhost:8080/admin/) |
| Prometheus | [localhost:8080/admin/prometheus/](http://localhost:8080/admin/prometheus/) |
| Grafana | [localhost:8080/admin/grafana/](http://localhost:8080/admin/grafana/) |
| Redis status | [localhost:8080/admin/redis](http://localhost:8080/admin/redis) |

Prometheus/Grafana ports 9090/3000 and Redis port 6379 remain internal. Use the
portal addresses. Grafana receives your verified portal username; its native
password form, basic authentication and anonymous access are disabled.
Authenticated portal users receive Grafana's Admin organisation role.

If you change the host port or domain, set `MUSICGEN_PUBLIC_ORIGIN` to the browser
origin, for example `http://localhost:8081`, alongside `MUSICGEN_HTTP_PORT=8081`.
The default dedicated tool network is `172.30.250.0/24`, with Traefik pinned at
`172.30.250.2`. Dynamic container addresses come from `172.30.250.128/25`, leaving
that proxy address outside the allocation range. If this conflicts with another
local network, change `MUSICGEN_ADMIN_NETWORK_SUBNET`,
`MUSICGEN_ADMIN_NETWORK_RANGE` and `MUSICGEN_ADMIN_PROXY_IP` together. Keep the
range inside the subnet and the proxy address outside that dynamic range.
Grafana trusts the configured proxy IP, not arbitrary incoming identity headers.

## Accounts, passwords and sessions

List local accounts without revealing hashes or tokens:

```bash
.venv/bin/python scripts/manage_admin.py list
```

Reset a local account's password and revoke its existing sessions:

```bash
.venv/bin/python scripts/manage_admin.py reset admin
```

For the Compose database, run the same operations in the frontend container:

```bash
docker compose run --rm --no-deps frontend python scripts/manage_admin.py list
docker compose run --rm --no-deps frontend python scripts/manage_admin.py reset admin
```

Use `create another-name` to add another administrator. An explicit local database
can be selected with `--database /path/to/admin.sqlite3` before the subcommand.
The CLI requires an interactive terminal for passwords; it does not accept them
as command arguments.

Sessions expire after eight hours by default (`MUSICGEN_ADMIN_SESSION_SECONDS`).
Signing out revokes the current session. Password reset revokes all sessions for
that account. Sign-in and sign-out forms use CSRF protection. Repeated failures
are throttled by account and client address; wait for the displayed retry interval.

## What is stored where

| Data | Location | Git? |
|---|---|---|
| Local administrator database | `storage/admin/admin.sqlite3` | No |
| Compose administrator database | `admin_data` volume, `/admin-data/admin.sqlite3` | No |
| Browser passwords | Salted scrypt hashes in the administrator database | No |
| Session records | SHA-256 token digests and expiry in that database | No |
| Redis machine secret | Private `.env`; ACL file generated inside the Redis volume | No |
| Grafana bootstrap secret | Private `.env`; Grafana retains its own persistent state | No |
| Auth implementation and example settings | `src/`, `infra/`, `scripts/`, `.env.example` | Yes |

Redis uses the named ACL user `cadence`; the default account is disabled. Its
machine secret is separate from every human password and must be a random
URL-safe value of at least 32 characters. `init-env` generates it. Application
connections use that account internally. The portal exposes server version,
uptime, memory, clients, key count and queued/generating counts; it does not expose
a Redis command console, key contents or credentials.

Normal `docker compose down` preserves the named volumes. Preserve the admin
volume with other deployment data when moving the stack; removing it removes
accounts and sessions. A new database starts without accounts and requires the
CLI setup again. SQLite is appropriate for the current single-host deployment;
it is separate from the window-cache SQLite files used by training.

## How the tool gate works

Every published Prometheus/Grafana request passes through Traefik's session check,
including tool APIs and assets. Incoming identity headers are cleared. The
frontend validates the server-side session and returns the authenticated username;
Grafana accepts that identity only from the configured proxy address. This follows
[Traefik ForwardAuth](https://doc.traefik.io/traefik/reference/routing-configuration/http/middlewares/forwardauth/)
and [Grafana auth-proxy configuration](https://grafana.com/docs/grafana/latest/setup-grafana/configure-access/configure-authentication/auth-proxy/).
Redis's machine connection uses its separate [ACL mechanism](https://redis.io/docs/latest/operate/oss_and_stack/management/security/acl/).

The current defaults bind the stack to loopback HTTP. Set
`MUSICGEN_ADMIN_COOKIE_SECURE=true` when using an HTTPS public origin. The
administrator portal protects operational pages; the existing music-generation
API remains unchanged and does not require an administrator login.

## Troubleshooting

- **Setup required:** create an account against the database used by that frontend.
  A local account does not automatically appear in the Compose volume.
- **Prometheus/Grafana unavailable locally:** start the full Compose stack and use
  its port-8080 admin portal.
- **Dashboard redirects to the wrong port:** align `MUSICGEN_PUBLIC_ORIGIN` with
  the browser address, then recreate the affected services.
- **Cannot sign in after password reset:** use the new password; prior sessions
  have been revoked.
- **Redis unavailable:** inspect Redis/API/worker logs and confirm the shared
  machine-secret configuration. Keep credential-bearing configuration out of
  shared logs or Git.

See [operations](operations.md), [architecture](architecture.md) and the
[README project guide](../README.md#project-guide) for implementation locations and normal commands.
