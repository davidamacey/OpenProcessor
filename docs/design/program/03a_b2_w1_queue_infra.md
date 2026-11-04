# B2 stage 1: foundation (Postgres, Valkey, the Celery app, the schema, the storage seam, the test database)

| Field | Value |
|---|---|
| Status | plan, not started |
| Depends on | [03_b2_overview.md](03_b2_overview.md) section 2; `int/b2-control-plane` cut from `main`; `13_package_pins.md` merged |
| Design | [02a](02a_postgres_schema.md) (the whole schema), [02b](02b_jobs_locking_events.md) sections 7-8 (Celery, Valkey, queues, pools), [02c](02c_migration_ops_tests.md) sections 1, 4-7 (Alembic, services, storage, backup, tests) |
| Lanes | 1-0 (coordinator), then pins part A ([13](13_package_pins.md) section 7), 1A, 1B, 1C, 1E in parallel, then 1D |
| Merge order | 1-0, pins part A, 1A, 1E, 1B, 1C, 1D ([00_master_plan.md](00_master_plan.md) W2.1) |
| Compose projects | `-p op-b2-s0` (spike S0, lane 1C), `-p op-b2-s1` (spike S1, lane 1B); CPU only; `down -v` after each |

No job family moves in stage 1. At the end of it the stack starts with Postgres at head, Valkey, idle
workers and beat, and the API checks the schema head at startup.

---

## Lane 1-0: dependency pins (coordinator, one commit, first)

- **Files:** `requirements.txt`, `requirements-test.txt`, `pyproject.toml` (the new packages, and the `db`
  marker in `[tool.pytest.ini_options] markers`, moved here from lane 1A so that pins part A and 1A
  never edit `pyproject.toml` at the same time).
- **Adds** the packages named in the overview header at the exact versions of file 13 (runtime:
  SQLAlchemy with the `asyncio` extra, Alembic, `psycopg[binary]`, Celery, kombu, billiard, redis-py;
  test: the same plus `testcontainers[postgres,redis]` (the extras of 13 section 4); Flower only where
  its service runs). Image tags and digests for Postgres and Valkey are the ones of 13 section 4 (the
  Postgres base is owner question MQ-2: `18.6-trixie` recommended); every `postgres:18-alpine` or
  `valkey/valkey:9-alpine` below means "the image of 13 section 4".
- **Red test:** `tests/test_control_plane_pins.py` asserts each package is `==`-pinned and identical
  in the three manifests where it appears, `billiard >= 4.3.1`, and that the redis-py pin satisfies
  the kombu pin's declared range (read from `importlib.metadata.requires('kombu')`). Red before the
  edit. **Control:** `test_ultralytics_is_one_exact_version_everywhere` passes.
- **Done when:** both green; `.venv/bin/pip install -r requirements-test.txt` then `.venv/bin/pip check` clean.

## Lane 1A `b2/db-core`: the schema and the database layer

**Owns:** `src/db/**` (except `src/db/repos/*.py` other than `__init__.py`), `alembic.ini`,
`scripts/db/__init__.py`, `scripts/db/migrate.py`, `src/config/database.py`,
`src/domain/job_states.py`, `src/core/db_session.py`, `src/app/lifespan.py` (schema check and engine
disposal only), `scripts/codegen/check_layering.py`, `tests/conftest.py` (plugin registration),
`tests/fixtures/postgres.py`, `tests/db/**` (the `db` marker itself is added by lane 1-0).

| ID | Task | Docs | Red test (fails on the base) | Control |
|---|---|---|---|---|
| 1A-1 | `src/config/database.py`: `DatabaseSettings` from `OP_DB_HOST`, `OP_DB_PORT`, `OP_DB_NAME`, `OP_DB_APP_USER`, `OP_DB_APP_PASSWORD_FILE`, `OP_DB_OWNER_USER`, `OP_DB_OWNER_PASSWORD_FILE`, `OP_DB_DIRECT_HOST`, `OP_DB_SSLMODE`, `OP_DB_POOLER`, `OP_DB_API_CONNECTIONS_TOTAL`, read through `src/config/env.py`; a missing or empty password file, a URL-shaped host, or an unknown `OP_DB_POOLER` fails closed with a named error | pydantic settings; env rules of CONTRIBUTING | `tests/config/test_database_settings.py`: each bad input raises with the variable's name | valid files and values load |
| 1A-2 | `src/db/base.py` (naming convention, `DeclarativeBase`), `src/db/types.py`, `src/db/models/{projects,classes,config,jobs,datasets,training,ops}.py`: every table of [02a](02a_postgres_schema.md), `lazy="raise"` on relationships | [naming](https://alembic.sqlalchemy.org/en/latest/naming.html), [ORM declarative](https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html) | `tests/db/test_models_import_light.py`: importing `src.db.models` in a fresh interpreter does not import `fastapi`, `opensearchpy`, `torch`, `celery` | `src.db.models` imports |
| 1A-3 | `src/db/migrations/` (async template), `0001_control_plane` (autogenerate draft, then by hand: RLS function and policies, triggers `config_revisions_immutable` and `jobs_status_transition_guard` generated from `src/domain/job_states.py`, partial unique indexes, `NULLS NOT DISTINCT`), `alembic.ini` | [autogenerate limits](https://alembic.sqlalchemy.org/en/latest/autogenerate.html), [cookbook](https://alembic.sqlalchemy.org/en/latest/cookbook.html) | `tests/db/test_migrations.py`: upgrade head on an empty database; `alembic check` clean; downgrade base and upgrade head; exactly one head | - |
| 1A-4 | `tests/db/test_schema_matches_plan.py`: introspects every table, unique constraint, partial index, policy and trigger named in 02a | - | red before 1A-3 | - |
| 1A-5 | `src/domain/job_states.py` (`TRANSITIONS`, `TERMINAL`, `ACTIVE`, pure) and `tests/db/test_job_transitions.py` (every listed transition passes; every other pair raises in the trigger) | [02b](02b_jobs_locking_events.md) section 1 | the trigger test | a listed transition succeeds |
| 1A-6 | `src/db/engine.py`: lazy engines keyed by `os.getpid()` (`get_async_engine()`, `get_sync_engine()`, `new_task_async_engine()` with `NullPool`, `direct_connect()`), `pool_pre_ping`, `pool_recycle=1800`, `pool_use_lifo`, sizes from [02b](02b_jobs_locking_events.md) section 8, `prepare_threshold=None` when `OP_DB_POOLER=pgbouncer`; `src/db/session.py`: `async_sessionmaker(expire_on_commit=False)`, `project_tx`, `deployment_tx` with the allowlist | [asyncio extension](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html), [pooling and fork](https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork), [psycopg prepare](https://www.psycopg.org/psycopg3/docs/advanced/prepare.html) | `tests/db/test_engine_per_process.py`: an engine is not shared after `os.fork()` (child gets a new one); importing `src.db` opens no socket; `tests/db/test_scope_helpers.py`: `deployment_tx` from a module outside the allowlist raises | `project_tx` on a migrated database runs a query |
| 1A-7 | RLS behaviour: `tests/db/test_row_level_security.py` ([02a](02a_postgres_schema.md) section 1.4): a project-scoped transaction cannot see, update or insert another project's rows in every RLS table; an unscoped session sees zero rows; the app role cannot bypass; `set_config` is transaction-local on a reused pooled connection (trap T-8) | [row security](https://www.postgresql.org/docs/current/ddl-rowsecurity.html) | red before 1A-3's policies | the owning project sees its rows |
| 1A-8 | `scripts/db/migrate.py` (`upgrade`: owner connection on the direct host, `pg_advisory_lock(2, 0)`, `command.upgrade`, ensure role `op_app` and grants idempotently, unlock; `current`; `check`) | OpenTranscribe `backend/app/db/migrations.py` (advisory-locked runner) | `tests/db/test_migrate_runner.py`: two runners at once both exit 0 and the schema is at head once; the app role exists with no DDL right (`CREATE TABLE` as `op_app` fails) | a re-run is a no-op |
| 1A-9 | schema-head check: `src/db/schema_check.py::assert_schema_current()` called from the API lifespan (and later from worker and beat init); mismatch exits with `schema_not_current` naming both revisions | - | `tests/db/test_schema_check.py`: a database at an older revision refuses | head passes |
| 1A-10 | `src/core/db_session.py`: `DbSessionDep` (one `AsyncSession` per request, a `yield` dependency; rollback on exception, re-raise) | [dependencies with yield](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/) | `tests/core/test_db_session_dep.py`: an exception in a handler rolls back and re-raises; two concurrent requests get two sessions | a normal request commits |
| 1A-11 | layering contracts ([02](02_control_plane_database.md) section 6) in `check_layering.py`, plus the two clauses of the original W1-12: nothing under `scripts/` imports `src.tasks`, and `src/tasks` imports no `src.routers`, `fastapi` or `starlette` | CONTRIBUTING (ratchets) | a planted `src/db` module importing `src.services` fails; a planted `src/domain` module importing `src.db` fails; a planted `scripts/` module importing `src.tasks` fails; a planted `src/tasks` module importing `fastapi` fails | the real tree passes |
| 1A-12 | test fixtures: `tests/fixtures/postgres.py` (session container per xdist worker via testcontainers, or `OP_TEST_DATABASE_URL`; template database migrated once; a fresh database per module; `two_sessions`; fails, never skips, when `-m db` is selected and no database is reachable), registered in `tests/conftest.py`; the `db` marker | [testcontainers Postgres](https://testcontainers-python.readthedocs.io/en/latest/modules/postgres/README.html) | `tests/db/test_fixture_policy.py`: with `OP_TEST_DATABASE_URL` pointing at a closed port the fixture raises `DatabaseUnavailable` (not skip) | the fixture yields a migrated database |

**Done when:** every 1A test green with `-m db`; the Docker-free command still collects and passes
without Docker (`-m 'not live and not families and not db'`).

## Lane 1B `b2/queue-core`: the Celery app, queues, the producer

**Owns:** `src/config/queue.py`, `src/config/job_kinds/**`, `src/config/task_routes/**`,
`src/config/advisory_locks.py`, `src/tasks/__init__.py`, `src/tasks/app.py`, `src/tasks/infra.py`,
the empty modules `src/tasks/{reaper,jobs_a,jobs_b,jobs_c,trigger,promote,periodic,b4,beat_schedule}.py`
(every later task module already in `include=`, so no later lane edits `app.py`; `b4` is for the B4
batch kinds of [05](05_b4_stateless_api.md) 4A, which also get empty `src/config/job_kinds/b4.py` and
`src/config/task_routes/b4.py` here), `tests/live/test_queue_infra.py` (the live check the B2 sweep step 2 runs),
`src/tasks/runtime/__init__.py`,
`src/tasks/runtime/gpu_app.py`, `src/clients/task_queue.py`,
`scripts/healthcheck/celery_pool_healthcheck.py`, `tests/tasks/**`, `tests/clients/test_task_queue.py`,
`tests/clients/test_queued_marker.py`, `tests/config/test_queue_settings.py`,
`tests/healthcheck/test_celery_pool_healthcheck.py`, `tests/test_no_revoke_terminate.py`.

| ID | Task | Docs | Red test | Control |
|---|---|---|---|---|
| 1B-0 | OPS-2 evidence (no code): time `/metrics` with 0, 100 and 1,000 dead-process files in a temporary `PROMETHEUS_MULTIPROC_DIR` using the project's metric definitions; 5 runs each; verdict "act" if 1,000 costs more than 2x the spread over 0, else "no action"; numbers and `uptime` in the PR notes | [prometheus multiprocess](https://prometheus.github.io/client_python/multiprocess/) | - | - |
| 1B-1 | **Spike S1** (throwaway, not committed; `-p op-b2-s1`): `valkey/valkey:9-alpine` by digest with `requirepass` from a mounted file, AOF, `noeviction`; one worker from the api image on the file-13 pins. Prove and record: (1) the worker connects with `redis://valkey:6379/0` and the password from a file; (2) an `acks_late` task killed with SIGKILL of the container is redelivered only after a lowered test visibility timeout; (3) a cold shutdown (`REMAP_SIGTERM=SIGQUIT`, soft shutdown 10 s) requeues a running `acks_late` task, and with billiard 4.3.0 it is acked instead; (4) a plain SIGTERM without the remap ignores the soft timeout; (5) `revoke(task_id)` of a queued task skips it; (6) `valkey-cli` honours `REDISCLI_AUTH` (else use `redis-cli`) and a uid-1000 client and the Valkey process both read the secret file: record the ownership rule | [Redis broker](https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html), [workers](https://docs.celeryq.dev/en/stable/userguide/workers.html), [kombu Redis transport](https://docs.celeryq.dev/projects/kombu/en/stable/reference/kombu.transport.redis.html) | - | - |
| 1B-2 | `src/config/queue.py`: `TaskQueues` (`jobs`, `periodic`, `open_vocab`, `autolabel`, `assist_loop`, `gpu_train`, `gpu_export`, `gpu_eval`; `ALL`), `QUEUE_PROFILE`, `QueueSettings` (`OP_QUEUE_URL`, `OP_QUEUE_PASSWORD_FILE`; only `redis` and `rediss` schemes; no password in the URL); `src/config/job_kinds/` (`JobKind` dataclass and the aggregating `__init__`, `infra.py` with `noop`, the empty modules `families_a`, `families_b`, `families_c`, `trigger`, `gpu_train`, `gpu_export`, `gpu_eval`, `promote`, `b4`); `src/config/task_routes/` the same plus an empty `periodic.py`; `src/config/advisory_locks.py` | [routing](https://docs.celeryq.dev/en/stable/userguide/routing.html) | `tests/config/test_queue_settings.py`: a `valkey://` URL, a URL with userinfo, a missing password file each raise; `ALL` equals the routed queues plus the declared ones | valid settings load |
| 1B-3 | `src/tasks/app.py`: the configuration of [02b](02b_jobs_locking_events.md) section 7.1 exactly; `include=` every `src/tasks/*` module; `worker_ready` refuses an unrouted or kind-less task; `worker_process_init` disposes inherited engines (`close=False`) and nothing else; `worker_init` calls `reject_retired_env()` (ENV-6) and the schema check; `src/tasks/runtime/gpu_app.py::make_app(include)` builds the same configuration for the GPU images | [configuration](https://docs.celeryq.dev/en/stable/userguide/configuration.html), [signals](https://docs.celeryq.dev/en/stable/userguide/signals.html) | `tests/tasks/test_celery_config.py` (the ordering `visibility_timeout > task_time_limit > task_soft_time_limit`, prefetch 1, JSON only, no missing queues, acks late, reject on lost, ignore result, **no `result_backend`**, soft shutdown 30 s on idle too, no cancel on connection loss, unlimited broker retries); `tests/tasks/test_no_cuda_import.py` (importing `src.tasks.app` leaves `torch`, `cuml`, `tritonclient` out of `sys.modules` and creates no engine) | `src.tasks.app` imports |
| 1B-4 | `src/clients/task_queue.py`: `enqueue(task_name, *, job_id)` (queue from the routes only, `task_id = job_id`, `send_task` by name), `revoke(job_id)` (no `terminate`), `ping()` (bounded), `mark_queued`, `clear_queued`, `queued_state -> Present / Absent / Unknown` (connection errors give `Unknown`); a minimal producer app from `src/config/queue.py` (never an import of `src.tasks.app`) | [Redis `SET` with `EX`](https://valkey.io/commands/set/) | `tests/clients/test_task_queue.py` (one message, `task_id == job_id`, routed queue; unknown name raises before sending; broker error raises `QueueUnavailable`); `tests/clients/test_queued_marker.py` | two job ids are independent |
| 1B-5 | `src/tasks/infra.py`: `op.infra.noop` (a plain task, for the live check); routes in `task_routes/infra.py`; `tests/live/test_queue_infra.py` (starts `op.infra.noop` and waits for `completed`; `live` marker, run only in the B2 sweep) | - | `tests/tasks/test_task_routes_complete.py`: every registered task has a route and a kind; no decorator passes `queue=` (OpenTranscribe `tests/unit/test_task_routes_complete.py`) | - |
| 1B-6 | `scripts/healthcheck/celery_pool_healthcheck.py`, ported from OpenTranscribe `backend/scripts/celery_pool_healthcheck.py` with an attribution header (AGPL to AGPL, public project): dead pool; respawn storm (pid set replaced while the task counter is frozen); pool type from `pool.implementation` | - | `tests/healthcheck/test_celery_pool_healthcheck.py` (fake `inspect stats` replies) | a steady pool is healthy |
| 1B-7 | `tests/test_no_revoke_terminate.py`: AST scan of `src/` and `scripts/` for a `terminate=` keyword in any `revoke` call | [revoking](https://docs.celeryq.dev/en/stable/userguide/workers.html#revoking-tasks) | a planted fixture fails | the real tree passes |

**Done when:** all 1B tests green; S1 items 1-6 recorded in the PR notes (a failure of 2, 3 or 6 stops
the lane and goes to the coordinator).

## Lane 1C `b2/compose-installer`: services, secrets, installer, CLI

**Owns:** `docker-compose.yml`, `docker-compose.dev.yml`, `docker/test/compose.yml`, `env.template`,
`setup-openprocessor.sh`, `scripts/setup.sh`, `openprocessor`, `scripts/lib/config.sh`,
`scripts/lib/bind_dirs.sh`, `scripts/lib/cli_core.sh`, new `scripts/lib/secrets_control_plane.sh`,
new `scripts/lib/cli_db.sh`, `tests/compose/test_control_plane_services.py`,
`tests/compose/test_beat_single_instance.py`, `tests/installer/test_control_plane_secrets.py`,
`tests/installer/test_env_generator.py`, `tests/installer/test_cli_db.py`,
`tests/installer/test_cli_status_lists_new_services.py`, `tests/compose/test_beat_healthcheck.py`,
`tests/config/test_db_connection_budget.py`.

| ID | Task | Docs | Red test | Control |
|---|---|---|---|---|
| 1C-0 | **ENV-4 first:** one `.env` generator in `scripts/lib/` used by the installer and `scripts/setup.sh` (copy the template with `umask 077`, `install -m 600`, upsert computed keys, never overwrite a user value) | the installer's `env_create_or_merge` | `tests/installer/test_env_generator.py`: the checkout generator writes mode 600 and every active template key | an existing value is kept |
| 1C-1 | **Spike S0** (throwaway; `-p op-b2-s0`): `postgres:18-alpine` by digest; record: the volume path (`/var/lib/postgresql`, `PGDATA` under `18/docker`); that `POSTGRES_PASSWORD_FILE` works when the file is owned per the rule you choose (the entrypoint drops to `postgres`); whether `pg_isready -h 127.0.0.1` stays false during first init; that the container runs with `no-new-privileges` and which capabilities it needs; the `pg_dump -Fc` and `pg_restore` round trip through `docker compose exec -T` | [postgres image](https://hub.docker.com/_/postgres), [pg_dump](https://www.postgresql.org/docs/current/app-pgdump.html) | - | - |
| 1C-2 | compose services of [02c](02c_migration_ops_tests.md) section 4 (`postgres`, `valkey`, `db-migrate`, `jobs-worker`, `periodic-worker`, `open-vocab-worker`, `jobs-beat`, `flower` under `monitoring`); `api` and the existing curation services gain `depends_on` on `db-migrate` (completed) and `valkey` (healthy); new services have no `container_name` (the existing ones lose theirs in stage 6); named volumes `openprocessor-pg`, `openprocessor-queue`; the Docker secrets directories mounted read-only | [compose services](https://docs.docker.com/reference/compose-file/services/), [secrets](https://docs.docker.com/compose/how-tos/use-secrets/) | `tests/compose/test_control_plane_services.py` (offline render): Postgres mounts `/var/lib/postgresql`, has `max_connections` and a TCP `pg_isready`; Valkey has `appendonly yes`, `noeviction`, no `ports`; no new service publishes a port except Flower on loopback; every Celery worker has `REMAP_SIGTERM=SIGQUIT` and `stop_grace_period` at least 30 s above the soft timeout; no worker shares the API's `PROMETHEUS_MULTIPROC_DIR`; every service except `db-migrate` waits on `db-migrate`; `tests/compose/test_beat_single_instance.py` (`jobs-beat` has `deploy.replicas: 1` and no fixed `container_name`, D-54; the old container-name assertion is gone); `tests/compose/test_beat_healthcheck.py` (the beat healthcheck globs `celerybeat-schedule*` and handles an empty glob, original W1-9). The compose files carry the comment "nothing acts on `unhealthy`; `restart:` fires only when the main process exits" (original W1-9) | `tests/compose/test_overlay_render.py` |
| 1C-3 | dev overlay (source mounts for the new Python services; the "Celery does not hot-reload" comment) and the live harness services ([02c](02c_migration_ops_tests.md) section 4, last paragraph) | - | render tests for both files | existing harness services unchanged |
| 1C-4 | secrets: `scripts/lib/secrets_control_plane.sh` creates `secrets/postgres/{owner_password,app_password}`, `secrets/valkey/{password,valkey.conf}`, `secrets/flower/auth` with `umask 077` before compose runs, on fresh install, `upgrade` and `repair`, never rotating; the S0 and S1 ownership rule applied; `config show` redacts them | the existing `secrets/vlm` and Cropwright patterns | `tests/installer/test_control_plane_secrets.py`: created with mode 600 when missing on upgrade; a re-run keeps the values; `repair` recreates a deleted one; no value appears in argv or the log | the Cropwright secret test passes |
| 1C-5 | env keys in `env.template` (advanced tier): the `OP_DB_*` set of 1A-1, `OP_PG_MAX_CONNECTIONS`, `OP_PG_SHARED_BUFFERS`, `OP_QUEUE_URL`, `OP_QUEUE_PASSWORD_FILE`, `OP_JOBS_CONCURRENCY`, `OP_STOP_GRACE_JOBS` (compose-only), `OP_DB_BACKUP_KEEP`, `OP_DB_BACKUP_INTERVAL_H` | `tests/test_env_surface.py` | the env-surface test fails until the template lines exist | - |
| 1C-6 | CLI: `openprocessor status/health` list the new services; `openprocessor db backup [--out]`, `db restore <file> [--yes]` (`scripts/lib/cli_db.sh`, [02c](02c_migration_ops_tests.md) section 6); `upgrade` takes a backup first when Postgres exists; `uninstall` removes the new volumes with the others (existing confirmation rules) | - | `tests/installer/test_cli_db.py` (a fake `docker compose exec` records the `pg_dump -Fc` and `pg_restore --clean --if-exists` argv and the keep-7 pruning); `tests/installer/test_cli_status_lists_new_services.py` | the existing CLI tests |
| 1C-7 | `tests/config/test_db_connection_budget.py` ([02b](02b_jobs_locking_events.md) section 8): computes the connection table from the compose file and the settings; fails when the sum exceeds `OP_PG_MAX_CONNECTIONS - 13` (named in 02b, 10 and 11 M36; no lane created it) | - | red with a planted compose whose sum exceeds the limit | the real compose fits |

**Done when:** all 1C tests green; S0 recorded; `docker compose config` renders for the default, dev
and arbiter overlays (offline render only in the lane).

## Lane 1E `b2/storage-seam`

**Owns:** `src/clients/storage/{__init__,base,local,keys,roots}.py`, `src/config/storage.py`,
`tests/clients/storage/**`, `tests/test_no_absolute_paths_in_db.py`.

| ID | Task | Red test | Control |
|---|---|---|---|
| 1E-1 | `BlobStore` protocol and `LocalFsStore` over `src/utils/atomic_write.py`; roots of [02c](02c_migration_ops_tests.md) section 5 mapped from the existing settings (`OP_SOURCE_ROOT`, `OP_PROJECTS_DATA_ROOT`, `OP_STATE_DIR`, the models path, the crop cache) | `tests/clients/storage/test_local_store.py`: atomic write (a crash mid-write leaves the old file), `list_prefix`, `delete_prefix`, `local_path` | read after write |
| 1E-2 | `keys.py`: `make_key(root, *parts)` and `parse_key`; refuses `..`, a leading `/`, a backslash, an empty part | `tests/clients/storage/test_keys.py` | round trip |
| 1E-3 | `tests/test_no_absolute_paths_in_db.py`: AST scan of `src/db/repos/` for an assignment to a column whose name ends in `_storage_key`, `_prefix` or `_key` that is not the result of `make_key` | a planted fixture fails | the real tree passes |

## Lane 1D `b2/health-ci` (starts after 1A and 1B are merged)

**Owns:** `src/routers/health.py`, `.github/workflows/ci.yml`, `CONTRIBUTING.md`,
`tests/test_health_database_queue.py`, `tests/test_db_marker_policy.py`.

| ID | Task | Red test | Control |
|---|---|---|---|
| 1D-1 | `/health` reports `database: ok \| down` and `queue: ok \| down` (degraded, HTTP 200, inference routes unaffected); each probe sequential with a short timeout, no new `gather` (B4-4A owns the existing fan-out) | `tests/test_health_database_queue.py` with fake engine and broker | both up gives `ok` |
| 1D-2 | CI: a `db` job with a Postgres service container (the image and digest of compose, 13 section 4), `OP_TEST_DATABASE_URL` set, running `pytest -m db`, gated by the existing `changes` path filter (code paths only, so docs-only PRs skip it); the existing `pytest` job keeps the Docker-free command | `act`-free check: `tests/test_ci_db_job.py` parses the workflow and asserts the service image equals the compose image and the job runs `-m db` | the existing workflow tests |
| 1D-3 | marker policy: a test under `tests/db/` or using the `pg` fixture without the `db` marker fails collection | `tests/test_db_marker_policy.py` (a planted unmarked test) | the real tree passes |
| 1D-4 | `CONTRIBUTING.md`: the test tiers (Docker-free, `db`, live), how to run `-m db` locally (Docker needed, no GPU), the one-head Alembic rule, the lane rules for generated `contracts/` | `scripts/docs/check_docs_vs_code.py --only CONTRIBUTING.md` | - |

## Stage 1 exit criteria

- every lane's tests green on the merged integration branch; `-m db` green; `make contracts-check`
  green (1D may change the health model: `make contracts` in that commit);
- `docker compose -p op-b2-s1check config` renders; a short CPU bring-up on the merged branch
  (coordinator, `-p op-b2-s1check`, 15 min cap): `db-migrate` exits 0, the API starts and reports
  `database: ok`, `queue: ok`; `op.infra.noop` runs; `down -v`;
- S0 and S1 results and the OPS-2 verdict recorded in the stage-1 merge notes (private ledger, and
  the facts for lane 6D).

## Lane brief (common block; add the lane's section)

```
You are a B2 stage-1 lane agent (Sonnet). Worktree <repo>-wt-b2-<lane>, branch b2/<lane>, cut from
int/b2-control-plane after lane 1-0. Read: docs/design/program/03_b2_overview.md (sections 3-4),
docs/design/program/03a_b2_w1_queue_infra.md (your lane), the 02/02a/02b/02c sections it cites,
CLAUDE.md, CONTRIBUTING.md, docs/design/engineering_standards.md. Edit only the files your lane owns.
Read each cited doc before the task and cite it in the commit body. One commit per task; run the red
test on the unchanged base first and paste the failure in the commit body; then the fix and the
control. Lane checks only: `.venv/bin/python -m pytest <your test files> -q --no-cov` (add `-m db`
for database tests; testcontainers needs Docker, no GPU) and `.venv/bin/pre-commit run --files
<touched files>`. No full suite, no --all-files, no live harness. Never --no-verify, never push,
merge, check out another branch or use git stash. Docker only for your spike, with your -p name,
CPU only, `down -v` after; never touch another compose project. Do NOT use pgrep -f, pkill -f or
sleep/until loops; wait with healthchecks or the Monitor tool. Never read .env or secret file
contents. Stop and report if a precondition is missing or a spike item fails.
GPU: none. Deadline: <set by the coordinator>.
Final report: commits, red outputs, spike results, anything not done.
```
