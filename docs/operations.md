# Development operations

## Configuration

Copy `.env.example` to `.env` at the repository root, or use `python scripts/manage_admin.py init-env` to initialize missing settings and generate machine secrets. Explicit environment variables override that file. Compose supplies container paths and authenticated Redis URLs directly, overriding local defaults. See [admin setup](admin.md) before the first Compose startup.

| Setting | Use |
|---|---|
| `MUSICGEN_RUN_MODE` | `local`; Compose uses `distributed` |
| `MUSICGEN_CHECKPOINT_PATH` | `models/best.pt` locally; `/models/best.pt` in worker |
| `MUSICGEN_OUTPUT_DIR` | `storage/generated` locally; `/outputs` in containers |
| `MUSICGEN_DEVICE` | Auto locally; CPU in Compose; CUDA in GPU override |
| `MUSICGEN_TORCH_THREADS` | Two CPU intra-op threads per worker |
| `MUSICGEN_MAX_DURATION_SECONDS` | 240, configurable within 5–240 |
| `MUSICGEN_GENERATION_TIMEOUT_SECONDS` | 1800, checked between model token steps |
| `MUSICGEN_MAX_PENDING_JOBS` | 32 queued/running records |
| `MUSICGEN_JOB_RETENTION_SECONDS` | 604800, seven days |
| `MUSICGEN_PALETTE` | `original`; optional `band` applies to Rock/Pop |
| `MUSICGEN_ENDING_STYLE` | `release` tapers the closing section and reserves a short final silence; `none` disables it |
| `MUSICGEN_REPETITION_PENALTY` | Soft melodic event-loop penalty, default `0.3`; `0` disables it, maximum `2` |
| `MUSICGEN_API_BASE_URL` | Local API URL; empty for same-origin Compose routing |
| `MUSICGEN_ADMIN_DB_PATH` | Local SQLite account/session database; default `storage/admin/admin.sqlite3`; Compose uses `/admin-data/admin.sqlite3` |
| `MUSICGEN_ADMIN_SESSION_SECONDS` | Fixed login lifetime, default 28800 seconds (eight hours) |
| `MUSICGEN_ADMIN_COOKIE_SECURE` | `false` for loopback HTTP; `true` when the public origin uses HTTPS |
| `MUSICGEN_REDIS_PASSWORD` | Separate random URL-safe machine secret, at least 32 characters; required in Compose |
| `MUSICGEN_PUBLIC_ORIGIN` | Compose browser origin, default `http://localhost:8080`; used for Grafana subpath URLs |
| `MUSICGEN_ADMIN_NETWORK_SUBNET` | Dedicated proxy/tool subnet, default `172.30.250.0/24` |
| `MUSICGEN_ADMIN_NETWORK_RANGE` | Dynamic container allocation range, default `172.30.250.128/25`; excludes the pinned proxy address |
| `MUSICGEN_ADMIN_PROXY_IP` | Traefik address trusted by Grafana auth proxy, default `172.30.250.2` |

Generation timeout is cooperative between forward passes, not a hard process-kill deadline. Redis task visibility is set beyond that duration.

The release ending is a rendering policy, not a learned harmonic cadence. It reduces late onset density, softens velocity and holds a recent model-selected sonority before a silent tail inside the requested duration. Metadata records adjustments and descriptive repetition/ending metrics. Musical quality still requires listening. For a baseline comparison, use `MUSICGEN_ENDING_STYLE=none` and `MUSICGEN_REPETITION_PENALTY=0`. Restart the API or worker after changing inference settings or source code.

Celery retries job-store outages with exponential backoff and jitter, capped at 60 seconds between attempts. A queued job resumes when storage recovers; deliveries for expired or already-claimed jobs are ignored. Store outages do not consume a fixed retry limit and strand queued jobs.

## Checkpoint readiness

Use a compatible full-run checkpoint. The loader accepts the baseline and structured PDMX architectures, checks vocabulary, dataset kind/signatures, model state and instrument/meter metadata, and rejects smoke-training models. It loads with `weights_only=True`.

Restart FastAPI locally or the worker in Compose after placing/replacing weights:

```bash
docker compose logs --tail=100 worker api
docker compose restart worker
```

The browser displays generator availability. `/api/health/live` is liveness; `/api/health/ready` additionally requires a ready model worker. Container liveness can be healthy while generation is unavailable.

## Outputs and persistence

Local jobs write `storage/generated/<job_id>/output.mid` and `metadata.json`.

Compose uses the `midi_outputs` volume, writable by workers and read-only in APIs. Retrieve completed files through the download/metadata endpoints. Redis, Prometheus, Grafana and the admin account database have separate persistent volumes. The `admin_data` volume contains `/admin-data/admin.sqlite3`; it is independent of a local `storage/admin/` database. Create/reset accounts against the database for the mode you use.

```bash
docker compose ps
docker compose logs -f worker
docker compose down
```

Normal `down` preserves named volumes. Output cleanup is manual; expiring a job record does not delete its files.

The tracked [`outputs/`](../outputs/README.md) folder contains three selected one-minute MIDI examples and their provenance. It is separate from application output storage; bulk generations, audio previews and private evaluation runs remain under ignored `storage/`.

## Admin and monitoring

Use the studio's **Admin** link and sign in with an account created by the [admin CLI](admin.md). Full Compose entry points are:

| Service | URL |
|---|---|
| Studio | [localhost:8080](http://localhost:8080) |
| FastAPI documentation | [localhost:8080/api/docs](http://localhost:8080/api/docs) |
| Admin portal | [localhost:8080/admin/](http://localhost:8080/admin/) |
| Prometheus | [localhost:8080/admin/prometheus/](http://localhost:8080/admin/prometheus/) |
| Grafana | [localhost:8080/admin/grafana/](http://localhost:8080/admin/grafana/) |
| Redis status | [localhost:8080/admin/redis](http://localhost:8080/admin/redis) |

Initialize `.env`, build the frontend, create a private admin account in its persistent volume, then start the stack as shown in [admin setup](admin.md). If changing `MUSICGEN_HTTP_PORT`, also set `MUSICGEN_PUBLIC_ORIGIN` to the matching browser origin. If the dedicated admin subnet conflicts with an existing network, change the subnet, dynamic allocation range and proxy IP together. Keep the proxy IP inside the subnet but outside the dynamic range; Grafana's trusted proxy address must match.

Traefik checks the shared server-side session before every Prometheus/Grafana page, asset or API request. Tool ports 9090/3000 are not published directly. Grafana obtains the verified username from the proxy and uses that identity for an Admin organisation role. Its native basic/anonymous login and sign-in form are disabled. Use the portal's sign-out action or the CLI password-reset command to revoke access.

The broker is **Redis**. RabbitMQ is not part of this deployment. Redis port 6379 and worker metrics port 9101 remain internal. The default Redis account is disabled; application connections use ACL user `cadence` with the generated machine secret. The portal exposes limited read-only Redis server/job statistics, not a command console or key browser.

Grafana provisions HTTP services and MIDI generation dashboards under Cadence. Counter-rate panels need traffic and a short observation window. Queue gauges describe shared state, so dashboards use max across API replicas. Completion and timing metrics come from actual worker attempts. Prometheus discovers individual replicas directly; job IDs appear in logs and metadata rather than metric labels.

The two-terminal Flask/Uvicorn setup provides [local admin](http://localhost:5000/admin/) and raw [frontend metrics](http://localhost:5000/metrics) / [API metrics](http://localhost:8000/metrics). It does not start monitoring containers or use Redis in local mode. Its dashboard links explain that the full Compose stack is required. The checked-in Prometheus configuration targets Docker service names; monitoring host-local processes needs separate scrape targets.

Compose always loads the local `models/best.pt` through its read-only model mount. An environment override used for a local test does not select a different Compose checkpoint. Install the intended compatible checkpoint there before comparing local and container results; see [checkpoint setup](../models/README.md).

## Runtime scope

Local mode is one API process and resets job records on restart. Compose is a one-host deployment; additional API replicas share state and files. Additional inference workers need additional model memory. Public multi-host hosting is separate deployment work.

If a worker disappears, polling reports active generation failed after heartbeat expiry and a 30-second grace period. Submit a new job; incomplete files are not served as completed output.

## Development checks

The [README](../README.md#development) lists the standard lint, formatting, test and package-build commands. Checkpoint-compatible runtime modules are excluded from automatic formatting. Vocabulary, architecture or parameter-name changes need matching compatibility checks and may require checkpoint migration.

For frontend interaction tests, install Playwright Chromium:

```bash
.venv/bin/python -m pip install -e ".[ui]"
.venv/bin/python -m playwright install chromium
MUSICGEN_BROWSER_TESTS=1 .venv/bin/python -m pytest -q -m browser
```

The Redis/Celery integration test uses a dedicated Redis server and databases 13–15. Set its URL to that test server, including credentials when required:

```bash
MUSICGEN_TEST_REDIS_URL=redis://localhost:6379/15 \
  .venv/bin/python -m pytest -q -m distributed
```

The real monitoring-login integration check requires a dedicated running stack and test account, selected by `MUSICGEN_TEST_ADMIN_URL`, `MUSICGEN_TEST_ADMIN_USERNAME` and `MUSICGEN_TEST_ADMIN_PASSWORD`. Run it with `pytest -q -m admin_stack` from the project environment. It verifies shared tool access, datasource health, denied anonymous requests and logout.

Optional checks skip when their hardware, dependencies or explicit service settings are absent. GitHub Actions runs the configured Python, browser, queue, package and container checks. Commit the retained notebook with outputs cleared; keep weights, account databases, active settings and bulk generated music local.
