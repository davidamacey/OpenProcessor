# 02c: migrations, the 0.6.0 upgrade, services, backup, storage, tests and traps

| Field | Value |
|---|---|
| Status | plan, not started |
| Part of | [02_control_plane_database.md](02_control_plane_database.md) |
| Built by | B2 stage 1 (Alembic, services, backup, storage seam, test fixtures), stage 2 (the upgrade step), stage 6 (scale-out compose) |

---

## 1. Alembic: layout, who runs it, when

### 1.1 Layout

| Path | What |
|---|---|
| `alembic.ini` (repo root) | `script_location = %(here)s/src/db/migrations`; no URL in the file (the URL is built at runtime from the owner secret) |
| `src/db/migrations/env.py` | the async template (`alembic init --template async`, [tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)), reading `target_metadata = src.db.base.Base.metadata` and running on the connection passed in `config.attributes["connection"]` ([sharing a connection](https://alembic.sqlalchemy.org/en/latest/cookbook.html#sharing-a-connection-across-one-or-more-programmatic-migration-commands)) |
| `src/db/migrations/versions/0001_control_plane.py` | the whole schema of [02a](02a_postgres_schema.md); later changes add revisions, never edit a released one |
| `scripts/db/migrate.py` | the runner (`upgrade`, `current`, `check`); used by `db-migrate` and the CLI |

Autogenerate drafts a revision; a person reviews it against the plan, because it misses renames,
anonymous constraints and some server defaults
([what autogenerate detects](https://alembic.sqlalchemy.org/en/latest/autogenerate.html)). Triggers,
RLS policies, the `op_scope_allows` function and the generated `CHECK` lists are written as explicit
`op.execute()` steps.

### 1.2 Who runs `alembic upgrade head`, and when

- **Only the one-shot `db-migrate` service.** No API or worker process runs DDL (FastAPI: "you would
  probably use a migration script that runs before you start your app",
  [SQL databases](https://fastapi.tiangolo.com/tutorial/sql-databases/)).
- `db-migrate` (api image, `restart: "no"`) connects as `op_owner` on the direct host, takes
  `pg_advisory_lock(MIGRATE = 2, 0)` on that session for the whole run (the OpenTranscribe runner's
  pattern), runs `command.upgrade(cfg, "head")` on the same connection, ensures the `op_app` role and
  grants, unlocks, exits 0. Two migrators started at once serialize; the second finds the schema at
  head.
- Every other service has `depends_on: db-migrate: condition: service_completed_successfully`
  ([depends_on](https://docs.docker.com/reference/compose-file/services/#depends_on)).
- **Startup check, fail closed:** every process reads `SELECT version_num FROM alembic_version` and
  compares it with the head of the bundled `src/db/migrations` (`ScriptDirectory.get_current_head()`);
  a mismatch exits with `schema_not_current` naming both revisions. A new image never runs on an old
  schema, and an old image never runs on a newer one.
- `openprocessor upgrade` order: back up Postgres if it exists (section 6) -> pull and verify images
  -> `up -d postgres valkey opensearch` -> `run --rm db-migrate` -> (from 0.6.0 only) the control-data
  step of section 2 -> `up -d --remove-orphans` for the rest.
- Downgrades: each revision has a `downgrade()` for development; production rollback restores the
  pre-upgrade dump (section 6), never `alembic downgrade`.

## 2. 0.6.0 -> 0.7.0: moving the control data out of OpenSearch

### 2.1 What a 0.6.0 install has

B1 is merged on `main` but was never released, so a real 0.6.0 install has **no** lease documents
and **no** class-registry document; it has `class_registry.json` files. A development install of
`main` may have both. The step reads whichever exists and prefers the registry document when both do
(the document is newer by construction: B1 renamed the file after migrating).

| Source (0.6.0 or `main`) | Read from | Loaded into |
|---|---|---|
| project records, `meta:projects_revision` | `op_projects` | `project_slugs`, `projects`, `project_indexes` (from each record's `resources.indexes`, only `images` and `items`), `deployment_counters` |
| retired slugs | `op_projects` records with status `deleted` | `project_slugs` with `retired_at` |
| class registry | the `class_registry` document if present, else `OP_PROJECTS_DATA_ROOT/<slug>/class_registry.json` | `class_registries` (`next_class_id`, `migrated_from`), `project_classes` (ids, names, groups, deprecated, `merged_into`, hotkeys, notes; counts dropped) |
| config documents, `@rev` copies, activations, activation events | each project's `configs` index (or the legacy separate settings index, by the record's role map) | `config_documents`, `config_revisions`, `config_activations`, `config_activation_events` |
| VLM endpoints, probes, `local_vlm:desired` | `op_global_configs` | `config_documents` (`project_id` NULL), `vlm_endpoint_probes`, `deployment_settings` |
| settings doc `default`, keymap, cluster-refresh state, UMAP view state | the settings role | `project_settings`, `project_keymaps`, `project_state`, `projection_states` |
| UMAP reducer | `umap_state` index | storage key `projects/<slug>/umap/<state_id>.bin` |
| pause flag | `pipeline_paused.flag` | `projects.paused_at` |
| holdout freeze | `<project_state_dir>/test_holdout/current.json` and history | `holdout_freezes` |
| label operations | `<project_state_dir>/label_ops/<op_id>/` | `label_operations`, `label_operation_members` |
| bulk-ingest sources | `source.json`, `state.json`, `queue.sqlite` | `ingest_sources`, `ingest_source_files` (pending rows only) |
| dataset imports and uploads | `ImportStore` directories | `dataset_imports`, `dataset_import_images`, `dataset_uploads` (non-expired) + terminal `jobs` rows (D-52) |
| training runs | `<id>.{job,status,manifest,registry_snapshot}.json` | terminal `jobs` + `training_runs` (artifacts stay where they are, recorded as storage keys) |
| promotions, `class_remap.json`, sharing | `promote.json`, remap files | `models`, `promotions`, `model_shares` |
| exports | export directories with `manifest.json` | `dataset_exports` |
| combine provenance | `/jobs/combine/*` | terminal `jobs` + `combine_runs`, `combine_sources` |
| non-terminal job files of every family | `state.json` with an active status | `jobs` rows with status `interrupted`, `auto_resume = false`, `error_code = 'upgraded_from_0_6_0'` (D-51) |
| transient job histories (reprocess, probe, scores, selection, viz, cluster, assist runs) | job directories | not loaded; kept in the bundle (D-52) |
| leases | lease documents (`main` only) | not loaded (transient) |

### 2.2 The steps (`scripts/db/upgrade_from_060.py`, called by the installer; code in `src/services/upgrade/from_060/`)

Preconditions: the 0.6.0 API and workers are stopped; OpenSearch is up; Postgres is at head; no
`upgrade_runs` row for `(0.6.0, delete_sources)` is `completed`.

1. **export_bundle**: read every source above (OpenSearch through an unguarded admin client that
   only issues `GET`, `search` with `search_after`, and `_mapping` reads; files read only) and write
   `.install/upgrade-0.6.0-<utc>/control_bundle.jsonl.zst` with one record per source document or
   file, the index mappings and settings of every control index, and a `MANIFEST.json` with counts
   and the bundle's sha256. Record `upgrade_runs(export_bundle)`.
2. **load**: one Postgres transaction under deployment scope that inserts everything from the bundle
   (natural keys, so a re-run after a failure starts clean: the transaction either committed or
   not). Record `upgrade_runs(load)` in the same transaction.
3. **verify**: counts per table equal the manifest; per project, class ids and names equal the
   source; every config revision body's sha256 equals the source; every `project_indexes.index_name`
   exists in OpenSearch; a class reconciliation over items reports items whose `class_id` is not in
   the registry or whose `class_name` disagrees with it (report only; the D-47 verifier is the same
   code). Any count or checksum mismatch stops here with the report path; nothing is deleted.
4. **delete_sources**: delete the per-project `configs`, `classes`, `umap_state`,
   `labels_confirmed` (and legacy settings) indexes, `op_projects` and `op_global_configs`; rename
   each `class_registry.json` to `class_registry.json.migrated-0.7.0`; delete the migrated state
   files and job directories; the `openprocessor-jobs` volume is removed by
   `up -d --remove-orphans` plus an explicit `docker volume rm` after the bundle is written. Each
   delete tolerates "already gone", so the step is re-runnable. Record `upgrade_runs(delete_sources)`.
5. The installer prints the report: projects, classes, configs moved; jobs recorded as interrupted;
   items with class disagreements; the bundle path.

The 0.7.0 code has no reader for any deleted source (no dual read). A project whose OpenSearch
indexes are missing at verify is moved with status `failed` and reported, as a 0.6.0 `failed`
project is.

### 2.3 Rollback to 0.6.0

`openprocessor upgrade --rollback` (the existing `backup_install` restore) gains one step:
`scripts/db/rollback_to_060.py --bundle <path>` recreates each deleted control index from the
mappings in the bundle, re-indexes the source documents, restores the job files and state files, and
renames `class_registry.json.migrated-0.7.0` back. The Postgres and Valkey volumes are left unused
(removable with `openprocessor uninstall --purge-volumes`). Anything created after the upgrade
(classes, configs, jobs) exists only in Postgres and is not carried back; INSTALLATION's downgrade
note says to export first. R-final rehearses upgrade, rollback, re-upgrade on a real 0.6.0 install
(RF-40, RF-62).

### 2.4 Tests

- `tests/upgrade/fixtures/v060/`: a small synthetic 0.6.0 control set (two projects, one deleted
  slug, classes with a merge and hotkeys, two prompt-pack revisions with an activation history, a VLM
  endpoint with a probe, a settings doc, a keymap, a holdout freeze, a label op, an ingest source with
  a `queue.sqlite`, an import with a ledger, a training run, a promotion, an active reprocess job),
  generated from the 0.6.0 index bodies (`git show v0.6.0:src/clients/curation_opensearch/bodies_other.py`)
  by `scripts/db/make_v060_fixture.py`, committed as JSON.
- `tests/upgrade/test_load_from_060.py` (`db` marker): bundle -> load -> verify gives equal counts,
  ids and checksums; a re-run after a forced failure in load gives the same result; the active job
  arrives `interrupted` without auto-resume.
- `tests/upgrade/test_delete_sources_idempotent.py`: delete twice; the second is a no-op.
- `tests/upgrade/test_rollback_bundle_roundtrip.py`: bundle -> recreate -> the recreated documents
  equal the fixture.
- Live: R-final `installer_upgrade` from a real 0.6.0 install with real class registries and an
  in-flight job, then rollback, then re-upgrade.

## 3. Where B1's upgrade code goes

Deleted in stage 2: `scripts/curation/migrate_class_registry.py`, `scripts/lib/registry_migration.sh`,
the CLI `curation migrate-registry` (`scripts/lib/cli_core.sh:236-258`), the upgrade and repair hooks
that call it, `registry_transfer.py`, the 409 `registry_not_migrated` path and its tests, and the
`.migrated` rename. `projects/legacy_repair.py` (VLM pin, active region class, `embedding_state`
backfill) becomes a step of the 0.6.0 upgrade (after load, per project, using the new repositories)
and the runtime module is deleted.

## 4. Compose services (core unless a profile is named)

Every new service: an image pinned by digest (tags in file 13), a healthcheck, secrets from files,
no published port, `security_opt: [no-new-privileges:true]`, `cap_drop: [ALL]` where it starts
without capabilities (stage 6 verifies each), `restart: unless-stopped` (except `db-migrate`), no
`container_name` (D-54), logging as the existing services.

| Service | Image | Command and settings | Healthcheck | Volumes | Notes |
|---|---|---|---|---|---|
| `postgres` | the Postgres image and digest of [13](13_package_pins.md) section 4 (owner question MQ-2: `18.6-trixie` recommended over `18-alpine`) ([image](https://hub.docker.com/_/postgres)) | `postgres -c max_connections=${OP_PG_MAX_CONNECTIONS:-300} -c shared_buffers=${OP_PG_SHARED_BUFFERS:-512MB} -c password_encryption=scram-sha-256`; env `POSTGRES_USER=op_owner`, `POSTGRES_DB=openprocessor`, `POSTGRES_PASSWORD_FILE=/run/secrets/op_pg/owner_password`; `shm_size: 256m`; `stop_grace_period: 60s` | `pg_isready -h 127.0.0.1 -U op_owner -d openprocessor` (TCP, so the socket-only server the entrypoint runs during first init does not count; spike S0 verifies) | `openprocessor-pg:/var/lib/postgresql` (**18 moved `PGDATA` to `/var/lib/postgresql/18/docker` and the volume to `/var/lib/postgresql`**; 17 and older mount `/var/lib/postgresql/data`) | the entrypoint starts as root and drops to `postgres`; the secret must be readable by that user (S0 records the ownership rule) |
| `valkey` | the Valkey image and digest of [13](13_package_pins.md) section 4 (`9.1.3-alpine3.24`) ([image](https://hub.docker.com/r/valkey/valkey)) | `valkey-server /run/secrets/op_valkey/valkey.conf --appendonly yes --appendfsync everysec --maxmemory-policy noeviction` (the conf holds only `requirepass`) | `sh -c 'REDISCLI_AUTH="$(cat /run/secrets/op_valkey/password)" valkey-cli ping \| grep -q PONG'` (password not in argv; S1 verifies `valkey-cli` honours the variable, else `redis-cli`, which the image also ships, as Immich uses) | `openprocessor-queue:/data` | no backup (D-24) |
| `db-migrate` | api image | `python -m scripts.db.migrate upgrade`; `restart: "no"`; mounts the owner and app password files | none (one-shot) | none | every service below `depends_on` it with `service_completed_successfully` |
| `jobs-worker` | api image | `celery -A src.tasks.app worker -Q jobs --concurrency=${OP_JOBS_CONCURRENCY:-4} --max-tasks-per-child=50 --hostname jobs@%h -E`; `REMAP_SIGTERM=SIGQUIT`; `stop_grace_period: ${OP_STOP_GRACE_JOBS:-60}s`; own `PROMETHEUS_MULTIPROC_DIR` tmpfs | the ported pool healthcheck (`scripts/healthcheck/celery_pool_healthcheck.py`) | storage roots as `api`, no `/jobs` | `API_GPU_ID` reservation |
| `periodic-worker` | api image | `-Q periodic -c 2 --max-tasks-per-child=200 --hostname periodic@%h -E` | pool healthcheck | as `api` | the GPU-arbiter overlay mounts the Docker socket here, not in `api` |
| `open-vocab-worker` | api image | `-Q open_vocab -c 1 --max-tasks-per-child=50` | pool healthcheck | as `api` | |
| `jobs-beat` | api image | `celery -A src.tasks.app beat -s /run/beat/celerybeat-schedule` (tmpfs) | newest mtime of `/run/beat/celerybeat-schedule*` under 300 s, failing on an empty glob | tmpfs only | single-beat lock (D-60); `deploy.replicas: 1` |
| `curation-auto-label-worker`, `curation-assist-loop-worker` (`curation`) | api image | `-Q autolabel -c 1 --max-tasks-per-child=20`, `-Q assist_loop -c 1 --max-tasks-per-child=20` | pool healthcheck | as today minus `/jobs` | |
| `curation-trainer` (`training`) | trainer image | `celery -A src.tasks.runtime.gpu_app worker -Q gpu_train -c 1 --max-tasks-per-child=1` | pool healthcheck | storage roots, `/jobs` removed | |
| `model-export-worker` (`export`) | export image | `celery -A src.tasks.runtime.gpu_app worker -Q gpu_export -c 1 --max-tasks-per-child=1` | pool healthcheck | models, storage roots | |
| `curation-evaluator` (`evaluator`) | evaluator image | `-Q gpu_eval -c 1 --max-tasks-per-child=5` | pool healthcheck | storage roots | |
| `flower` (`monitoring`) | api image | `celery -A src.tasks.app flower --basic-auth-file=/run/secrets/op_flower/auth` (or the env form Flower documents), URL prefix for the gateway | HTTP `/healthcheck` | none | published only on `${OP_BIND_ADDRESS:-127.0.0.1}:4613` (ENV-18) |

Secrets (generated by the installer with `umask 077` before compose runs, so Docker never creates a
root-owned directory; never rotated on upgrade; created on `upgrade` and `repair` when missing;
redacted by `config show`): `secrets/postgres/owner_password`, `secrets/postgres/app_password`,
`secrets/valkey/password`, `secrets/valkey/valkey.conf`, `secrets/flower/auth`. They are mounted
read-only as directories (`/run/secrets/op_pg`, `/run/secrets/op_valkey`, `/run/secrets/op_flower`),
the pattern `secrets/vlm` already uses. Services get only the files they need: the API and workers
get the app password and the Valkey password; only `db-migrate` and the backup command get the owner
password.

Named volumes: new `openprocessor-pg`, `openprocessor-queue`; deleted `openprocessor-jobs`.

The dev overlay adds source mounts for the new Python services and a comment that Celery does not
hot-reload (`docker compose -p <name> restart jobs-worker`). The live harness
(`docker/test/compose.yml`) gains `postgres`, `valkey`, `db-migrate`, the core workers and beat,
under its own project name; because no service has a `container_name`, the harness scales
`jobs-worker` with `--scale jobs-worker=2` instead of declaring two services.

## 5. The storage seam (D-46)

`src/clients/storage/` defines `BlobStore` (`open_read`, `write_atomic`, `exists`, `stat`,
`list_prefix`, `delete_prefix`, `local_path`) and named roots:

| Root | Today | Key prefix |
|---|---|---|
| `source` (read only) | `OP_SOURCE_ROOT` bind mount | as given by the ingest source |
| `projects` | `OP_PROJECTS_DATA_ROOT/<slug>/`, `OP_STATE_DIR/projects/<slug>/` | `projects/<slug>/{uploads,exports,bakeoff_eval,umap,holdout,training,label_ops}/...` |
| `models` | `./models` (Triton repository), `pytorch_models` | `models/...` |
| `cache` | `openprocessor-crop-cache` (tmpfs) | `cache/...` (regenerable) |
| `upgrade` | `.install/` | `upgrade/...` |

- 0.7.0 ships only `LocalFsStore(root_path)`, built on the existing `src/utils/atomic_write.py`
  (stdlib `tempfile` + `os.replace`). Postgres rows store keys, never absolute paths. Code that needs
  a real path (Ultralytics training, Triton loading, `faiss.write_index`) asks `local_path(key)`, which
  the local backend returns directly.
- Multi-host in 0.7.0: mount the same POSIX volume (NFS or similar) at the same roots on every host;
  INSTALLATION documents it and the locking caveat no longer applies because no state file is
  read-modify-written any more.
- v0.8.0: an S3-compatible backend behind the same interface (library choice then, with its
  licence; MinIO's community distribution changed in 2025, so the candidates are re-checked at that
  time). Triton can read a model repository from S3 natively
  ([cloud storage](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html#cloud-storage-with-environment-variables)),
  and Immich keeps originals and regenerable derivatives apart, which this root list mirrors.
- A ratchet (`tests/test_no_absolute_paths_in_db.py`) asserts every `*_storage_key`, `*_prefix` and
  `*_key` column is written through `src/clients/storage/keys.py` (relative, no `..`, no leading `/`).

## 6. Backup and restore (D-61)

| Command | Does |
|---|---|
| `openprocessor db backup [--out <dir>]` | `docker compose -p <p> exec -T postgres pg_dump -U op_owner -d openprocessor -Fc` to `<install>/backups/postgres/openprocessor-<utc>.dump` (custom format: compressed, restorable selectively; consistent without blocking writers: [pg_dump](https://www.postgresql.org/docs/current/app-pgdump.html)); keeps the newest `OP_DB_BACKUP_KEEP` (default 7) |
| `openprocessor db restore <file> [--yes]` | stops the API and workers, `pg_restore --clean --if-exists --no-owner --role=op_owner -d openprocessor`, runs `db-migrate` (a dump from an older schema is upgraded), runs `openprocessor db verify`, starts the stack |
| `openprocessor db verify` | the D-47 class reconciliation, `project_indexes` vs OpenSearch, `storage_key` existence for promotions and training runs; report only |
| `openprocessor upgrade` | always takes a backup first when Postgres exists |
| optional beat task | `op.periodic.db_backup` when `OP_DB_BACKUP_INTERVAL_H` is set (off by default) |

Order rule (Immich: "back up the database first, then the files"): a database restored to an older
point than the OpenSearch data can name a class id the registry does not have; `db verify` reports
such items, and the docs say to restore both from the same time. OpenSearch snapshots stay out of
scope (unchanged).

## 7. Tests (D-62)

| Tier | What | How |
|---|---|---|
| Offline, Docker-free (`-m 'not live and not families and not db'`, the command CLAUDE.md names) | everything that does not need SQL semantics: services against in-memory repository fakes, wire models, config, the kind table, compose renders, Celery config | unchanged command; nothing new needs Docker here |
| `db` (new marker) | repositories, the claim, heartbeat and fencing statements, unique and partial indexes, RLS, the transition trigger, the reaper rules, the events cursor, migrations, the 0.6.0 upgrade | a real Postgres 18 from **testcontainers-python** (`PostgresContainer`, image pinned as in compose) on a developer machine, or from `OP_TEST_DATABASE_URL` (a CI service container: [PostgreSQL service containers](https://docs.github.com/en/actions/use-cases-and-examples/using-containerized-services/creating-postgresql-service-containers)). Selected with `-m db`; when selected and no database can be reached, the fixture **fails** (never skips) |
| Live harness | the job and event behaviour with real workers | `docker/test/compose.yml` with the new services |

Fixture design (`tests/db/conftest.py`, registered for the whole suite):
- one container per test session (one per xdist worker), migrated once to head into a template
  database `op_template`;
- each test module gets `CREATE DATABASE op_t_<n> TEMPLATE op_template` (fast copy) so concurrency
  tests can commit from several connections; dropped at module end;
- helpers `project_tx`/`deployment_tx` as in production; a `two_sessions` fixture for races.

Required `db` tests (each red first where it guards a behaviour):
- `tests/db/test_migrations.py`: upgrade head on an empty database; `alembic check` reports no
  pending operations ([alembic check](https://alembic.sqlalchemy.org/en/latest/autogenerate.html));
  downgrade to base and upgrade again.
- `tests/db/test_schema_matches_plan.py` (section 1 of [02a](02a_postgres_schema.md)).
- `tests/db/test_row_level_security.py`, `tests/db/test_job_transitions.py`.
- `tests/db/test_job_start_race.py`: 20 threads start the same exclusive kind: exactly one row, 19
  answers of `job_busy`; with an idempotency key: one row, 20 identical answers.
- `tests/db/test_claim_and_fencing.py`: two claims of one queued job: one wins; a write with an old
  fencing token changes nothing; an expired lease is taken over by the reaper only.
- `tests/db/test_reaper_rules.py`: rules a-g, including two reapers at once (no double requeue).
- `tests/db/test_events_cursor.py`: two transactions commit their events out of id order; a reader
  that polls between the commits emits every event exactly once.
- `tests/db/test_class_registry.py`: 10 concurrent `ensure_class` for 50 names: 50 classes, unique
  ids, `next_class_id = 50`; rename writes the outbox row in the same transaction.
- `tests/db/test_connection_budget.py` (pure, no database): section 8 of [02b](02b_jobs_locking_events.md).

The 1-vs-N replica equivalence test runs in B4's live run (the third budgeted run,
[05_b4_stateless_api.md](05_b4_stateless_api.md) section 7): one scripted scenario with
`api=1 replica x API_WORKERS=1, jobs-worker=1` and with `api=2 replicas x API_WORKERS=4,
jobs-worker=2`; the normalized outcomes (job rows, class ids by name, config revisions, event
sequences per project) must be identical.

## 8. Trap register (ways it looks done but is not)

| # | Trap | Guard |
|---|---|---|
| T-1 | An async engine created in one `asyncio.run` loop is reused in the next task's loop ("attached to a different loop") | per-task async engines with `NullPool`, disposed in `finally`; `tests/tasks/test_engine_per_task.py` runs two tasks in sequence |
| T-2 | An engine created at import is inherited by Celery prefork children and two processes share a socket | lazy engines keyed by `os.getpid()`; `worker_process_init` disposes with `close=False`; a fresh-interpreter test: importing `src.tasks.app` creates no engine and opens no socket |
| T-3 | `expire_on_commit=True` makes an attribute read after commit do implicit I/O (`MissingGreenlet`) | `async_sessionmaker(expire_on_commit=False)`; relationships `lazy="raise"` |
| T-4 | Select-then-insert under READ COMMITTED races | unique constraints and `INSERT ... ON CONFLICT`; ratchet: no `exists()` check before an insert in `src/db/repos` |
| T-5 | A status added in Python but not to the `CHECK`, the trigger or the active-set index | all three generated from `src/domain/job_states.py`; `tests/db/test_job_transitions.py` |
| T-6 | A query on a project table without the scope helper | RLS returns zero rows (fail closed) and `tests/db/test_row_level_security.py`; a ratchet on `src/db/repos` for direct `session.execute` outside `project_tx`/`deployment_tx` |
| T-7 | The table owner bypasses RLS | the app connects as `op_app`; `FORCE ROW LEVEL SECURITY`; a test connects as `op_app` |
| T-8 | `set_config(..., false)` (session level) leaks a project binding to the next request on a pooled connection | only `is_local = true`; a test reuses one pooled connection for two projects |
| T-9 | The message is sent before the commit, and the worker cannot see the row | `start_job` commits first; a test with a fake producer asserts the order |
| T-10 | A duplicate delivery runs the job twice | the claim `UPDATE ... WHERE status = 'queued'`; Celery does not de-duplicate by `task_id` |
| T-11 | A visibility-timeout redelivery during a 24 h job | the claim finds `running` and acks; documented for every kind above 6 h |
| T-12 | Lease expiry compared on a worker clock | expiry only in SQL with `now()` |
| T-13 | A long transaction around an OpenSearch or VLM call holds row locks and the events `xmin` | rule 4; a ratchet over `src/services` for network clients used inside `project_tx` blocks (AST: calls on `OpenSearch*`, `httpx`, Triton clients inside `async with project_tx`) |
| T-14 | A bulk insert of a million `ingest_source_files` rows in one transaction stalls the event feed | batches of 5,000 per transaction |
| T-15 | `NOTIFY` assumed durable | the consumer always re-reads from its cursor; the 2 s safety tick |
| T-16 | `LISTEN`, the beat lock or the migration lock sent through a transaction pooler | `OP_DB_DIRECT_HOST`; a test that the pooled engine never issues `LISTEN` or session locks |
| T-17 | kombu 5.7 pulled under Celery 5.6, or a redis-py the transport does not accept | exact pins (file 13); `pip check` in each image; spike S1 on the pins |
| T-18 | `valkey://` in a URL | `QueueSettings` refuses any scheme but `redis` and `rediss`; test |
| T-19 | Valkey started with `appendonly no` (its default) or an eviction policy | compose render test asserts the flags |
| T-20 | Postgres 18 volume mounted at `/var/lib/postgresql/data` (the 17 path) | compose render test; S0 |
| T-21 | A secret file unreadable by the image's user (Postgres drops to `postgres`, workers run as `appuser`) | S0 and S1 record the ownership rule; installer test |
| T-22 | `pg_isready` passes during first-time init | TCP check to 127.0.0.1; S0 |
| T-23 | A second beat on another host | the session advisory lock; live test starts two beats and sees one schedule |
| T-24 | A Celery result backend configured by habit | config test asserts `result_backend` is unset |
| T-25 | 0.6.0 sources deleted before verify passed | step order is fixed in code; test forces a verify failure and asserts nothing was deleted |
| T-26 | A 0.7.0 reader still looks for `op_projects` or `class_registry.json` | `rg` done-when lines in stage 2; the naming of the deleted indexes appears only in `src/services/upgrade/` |
| T-27 | A project deleted while one of its jobs runs: the cascade removes the row under the worker | delete drains first (cancels and waits); a lost row is `Lost` to the worker, which stops |
| T-28 | Absolute host paths stored in rows block a move to another host or to object storage | storage keys only; ratchet |
| T-29 | A GPU-queue worker imports CUDA in the Celery parent | the handler imports torch inside the task; fresh-interpreter test per image entry |
| T-30 | `--max-tasks-per-child=1` on GPU queues hides a leak elsewhere | documented as intentional (GPU memory release); measured once in the B2 live run |
| T-31 | Progress written on every item floods `events` | 1/s throttle in `ctx.progress`; a test with 10,000 progress calls writes at most the elapsed seconds plus stage changes |
| T-32 | Alembic autogenerate drops the trigger or RLS policy it cannot see | `alembic check` plus `test_schema_matches_plan.py` |
