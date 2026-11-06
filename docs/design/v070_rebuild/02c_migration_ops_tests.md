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
| `src/db/migrations/env.py` | the **sync** template (review fix R-29: the runner is a one-shot sync script that holds a session advisory lock on one psycopg connection, so the async template's `run_sync` plumbing buys nothing), reading `target_metadata = src.db.base.Base.metadata` and running on the connection passed in `config.attributes["connection"]` ([sharing a connection](https://alembic.sqlalchemy.org/en/latest/cookbook.html#sharing-a-connection-across-one-or-more-programmatic-migration-commands), [tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)) |
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
- `db-migrate` (api image, `restart: "no"`) bootstraps the roles as `op_super` ([02a](02a_postgres_schema.md)
  section 1.5), then connects as `op_owner` on the direct host, takes `pg_advisory_lock(MIGRATE = 2, 0)`
  on that session for the whole run (the OpenTranscribe runner's pattern), runs
  `command.upgrade(cfg, "head")` on the same connection, re-applies the `op_app` grants, unlocks.
  It exits 0. Two migrators started at once serialize; the second finds the schema at head.
- **The OpenSearch step runs in `db-migrate` after Alembic** (review fix R-7; re-check RC-1 as
  re-evaluated under D-69, which replaced D-64 on 2026-11-04: nothing runs without Postgres, so the
  one migrate step serves both stores). `scripts/db/migrate.py` calls `scripts/opensearch/migrate.py`
  after the Alembic run, idempotently: it creates the global visual-search indexes and the `images`
  and `items` indexes of every project in the Postgres `projects` table that are missing, applies the
  mapping migrations, stamps each index's mapping `_meta` with its schema fingerprint (the existing
  `SCHEMA_META_KEY` stamp of `src/services/curation/index_bootstrap.py`), and runs the kNN warm-up
  once (server-side, so once per deployment is enough). While the 0.6.0 data is not loaded yet
  (`upgrade_state` is `in_progress`, or the old `op_projects` index still exists) it does the global
  indexes only and logs that the project step waits for the upgrade, because the project rows come
  from the control-data step of section 2. No API process creates indexes, migrates mappings or warms
  kNN any more. `project_create` (B4 4D) creates and stamps its two indexes in its job. Review RC-7:
  the warm state is lost when OpenSearch restarts or evicts its graph cache; the first query after
  that pays the load (accepted; INSTALLATION says so, and 5a measures the cold-query latency).
- **Who waits on whom (owner decision D-69, 2026-11-04):** the API and every worker require
  PostgreSQL, Valkey and OpenSearch, and Triton where they call it (the API always). Every Python
  service except `db-migrate` itself has `depends_on: db-migrate: condition:
  service_completed_successfully` and `condition: service_healthy` on `postgres`, `valkey`,
  `opensearch` and (the API and the Triton-calling workers) `triton-server`
  ([depends_on](https://docs.docker.com/reference/compose-file/services/#depends_on)); `db-migrate`
  waits on `postgres` and `opensearch` healthy. There is no degraded mode and no flag for one.
- **Startup check, fail closed (D-69):** every process waits for its dependencies with bounded
  retries (a 60 s budget with a fixed backoff, a constant in `src/db/schema_check.py`, not a
  setting), then refuses to start
  with one error naming the missing dependency (`dependency_unavailable: database`, `queue`,
  `opensearch` or `triton`). Then it reads `SELECT version_num FROM alembic_version` and compares it
  with the head of the bundled `src/db/migrations` (`ScriptDirectory.get_current_head()`): another
  revision exits with `schema_not_current` naming both revisions, and an `upgrade_state` row of
  `in_progress` exits with `upgrade_incomplete` (section 2.2). It reads every index's `_meta` stamp
  in one `GET <global and project indexes>/_mapping?filter_path=*.mappings._meta` call and exits with
  `opensearch_mapping_not_current` naming the index when one is wrong or missing. A new image never
  runs on an old schema or mapping, and an old image never runs on a newer one.
- **Liveness and readiness (D-69, the existing split of `src/routers/health.py`):** `/live` says the
  process is up and never probes a dependency; `/ready` (and `/health`, its existing alias) answers
  503 while any required dependency is down and names each one (`database`, `queue`, `opensearch`,
  `triton` and its required models); the compose healthcheck of `api` probes `/ready`. At run time a
  route that reaches a dependency that is down answers 503 naming it (`database_unavailable`,
  `queue_unavailable`); no route is designed or tested to keep serving through an outage.
- **Upgrades are stop-the-world in 0.7.0** (review R-26): the strict head equality means old and new
  processes never run together. From v0.8.0 migrations follow expand-then-contract, so the check can
  become "the database head is at or ahead of the code head and marked compatible".
- `openprocessor upgrade` order: back up Postgres if it exists (section 6) -> pull and verify images
  -> `up -d postgres valkey opensearch` -> `run --rm db-migrate` (Alembic, then the global indexes)
  -> (from 0.6.0 only) the control-data step of section 2, which loads the projects, then `run --rm
  db-migrate` again, which now finds them and migrates and stamps their indexes (RC-1) -> `up -d
  --remove-orphans` for the rest.
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
`upgrade_runs` row for `(0.6.0, delete_sources)` is `completed`. The load is the one sanctioned large
transaction (review fix R-25): nothing else runs during it (the precondition), and its expected size
and time are measured on a 1,000,000-row ledger fixture.

**The gate (review fix R-9a).** Step 1 first writes `deployment_settings('upgrade_state') =
{"from": "0.6.0", "state": "in_progress"}`; step 4 sets it to `completed` in the transaction that
records `delete_sources`. While it is `in_progress` (or while `op_projects` still exists in
OpenSearch), every 0.7.0 process refuses to start with `upgrade_incomplete`, so nothing can write to
Postgres while the operator fixes a failed verify. `openprocessor status` prints the state and the
next command.

1. **export_bundle**: read every source above (OpenSearch through an unguarded admin client that
   only issues `GET`, `search` with `search_after`, and `_mapping` reads; files read only) and write
   `.install/upgrade-0.6.0-<utc>/control_bundle.jsonl.gz` (gzip from the standard library; review
   fix R-31: Python 3.13 has no standard-library zstd and no zstd package is pinned) with one record
   per source document or file, the index mappings and settings of every control index, and a
   `MANIFEST.json` with counts and the bundle's sha256. The training-run logs (`<id>.run.log`) and
   any other `/jobs` file a kept row points to are copied to storage keys
   (`projects/<slug>/training/<run>/run.log`) and listed in the manifest (review fix R-9e), before
   the `openprocessor-jobs` volume can be removed. Record `upgrade_runs(export_bundle)`.
2. **load**: one Postgres transaction under deployment scope that inserts everything from the bundle
   (natural keys); the ledger tables (`dataset_import_images`, `label_operation_members`,
   `ingest_source_files`) are loaded with `COPY` through psycopg's `cursor.copy()`
   ([COPY](https://www.psycopg.org/psycopg3/docs/basic/copy.html)), not row inserts (R-25). Record
   `upgrade_runs(load)` in the same transaction. A load that failed rolled back, so a re-run starts
   clean. **After a committed load whose verify failed**, the re-run path is
   `upgrade_from_060 --reset` (review fix R-9b): one transaction that truncates the control tables and
   deletes the `upgrade_runs` rows, allowed only while `delete_sources` has not completed; then the
   step runs again from the bundle.
3. **verify**: counts per table equal the manifest; per project, class ids and names equal the
   source; every config revision body's sha256 equals the source; every `project_indexes.index_name`
   exists in OpenSearch; a class reconciliation over items reports items whose `class_id` is not in
   the registry or whose `class_name` disagrees with it (report only; the D-47 verifier is the same
   code). Any count or checksum mismatch stops here with the report path; nothing is deleted.
3a. **analysis_settings** (lane 2F, task 2F-6; review RN-3): the bounded close, settings and reopen of
   the plate analysis settings on every upgraded `items` index (the table below), run here so that a
   failure stops the upgrade with the sources still in place. Record
   `upgrade_runs(analysis_settings)`; step 4 refuses unless it is `completed`.
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

**Research-seam steps (approved by the owner 2026-11-04, RD-1; [02d](02d_research_seam_schema.md)).** The load
is a registry of step modules, `src/services/upgrade/from_060/steps/<name>.py`, each with an
`ORDER` integer and `run(ctx)`, discovered with `pkgutil.iter_modules` (lane 2E, task 2E-8), so the
W2.3a lanes add their steps as new files without editing 2E's code:

| Step module | Lane | Writes | Verify adds |
|---|---|---|---|
| `serving_profile.py` | 2E | the deployment `serving_profile` document, revision 1, generated from the installed `config_templates/profiles/*.json` and the `config.pbtxt` instance counts, and `deployment_settings('active_serving_profile')` (research GPU S-5) | the generated body equals today's values (one server, load lists, instance counts) |
| `embedding_spaces.py` | 2E | the catalog rows (if `db-migrate` has not) and three `pe_core_l14_336_v1` spaces per loaded project, `created_by = 'upgrade'`, `knn_method` read from each live index mapping (faiss for old projects, Lucene for new); the 0.6.0 physical names kept (RD-30 answered as recommended (D-78)) | a space per (project, target); the recorded engine equals the live mapping |
| `clustering.py` | 4F (W2.3a) | per project, one `ready` partition (`method = 'ivf'`, activation reason `upgrade`) built from the items' existing assignment so the served view does not change at the upgrade; `last_clustering.json` and the region-cluster marker files deleted with the other state files | the partition's member count equals the items with a cluster id |
| `near_duplicates.py` | ND-A (W2.3a) | one `update_by_query` per 0.6.0 index clearing the `dup_*` and `duplicate_*` values (mappings cannot be dropped without a reindex); the `near_dup_profile` revision 1 per project from the code defaults; the levels marked `rebuild_required`, so a full build runs after the upgrade; the upgrade report offers the pausable `source_backfill` job, whose content step writes `content_sha256` for every carried-over image (review RR-10; the W4 capture step joins the same pass) | no document keeps a non-empty `dup_*` value; after the backfill a re-ingest of an upgraded image is refused as an exact duplicate |

Step 3a adds the registration-plate analysis settings to the `items` index of every **upgraded**
project (kept by the owner, Q-A of 2026-11-04). It runs inside `upgrade_from_060`, after verify and
before `delete_sources` (review RN-3), not in the later `db-migrate` OpenSearch migrate (2F, R-7),
which only skips indexes that already carry the settings. Analysis
settings are static, so they need `_close`, `PUT _settings`, `_open`
([update settings](https://docs.opensearch.org/latest/api-reference/index-apis/update-settings/)).
New projects get them in the creation body. The step is safe and bounded (review RR-8, task 2F-6):

| Property | Rule |
|---|---|
| per index | one items index at a time, in slug order; each is closed only while its own settings call runs |
| dry run | `scripts/opensearch/migrate.py --step analysis-settings --dry-run` lists every index, its doc count and shard size, and the measured downtime estimate, and changes nothing; the upgrade report prints it before the real run |
| progress | one line and one `upgrade_runs.report` entry per index: `n/N <index> closed <seconds> s, reopened, health <colour>` |
| always reopen | close, settings, open in `try/finally`; the `finally` reopens and waits for at least yellow health (bounded wait, 120 s per index) |
| verify | `_analyze` with `op_text_compact` on the reopened index; a mismatch stops the upgrade |
| rollback | on any failure the index is reopened unchanged (the settings call is atomic per index), the upgrade stops before `delete_sources` with the report, and the 0.6.0 rollback (section 2.3) still recreates every index from the bundle; an added but unused normalizer is inert (no field uses it until W5e), so a partly applied run is safe to re-run (idempotent: an index that already has it is skipped) |
| stated downtime | inside the stop-the-world window only (no API serves during the upgrade); per index from close to yellow, measured on the 0.6.0 fixture and printed by the dry run (seconds per index on one node; a faiss-engine index also reloads its native graph on its first search after reopen) |
| failure-path test | `tests/opensearch/test_analysis_settings_failure.py`: a refused settings call leaves the index open and the upgrade stopped with the index named |

### 2.3 Rollback to 0.6.0

`openprocessor upgrade --rollback` (the existing `backup_install` restore) gains these steps, in
this order (review fix R-9d), with every 0.6.0 and 0.7.0 API and worker stopped:
1. start OpenSearch only;
2. `scripts/db/rollback_to_060.py --bundle <path>` recreates each deleted control index from the
   mappings in the bundle, re-indexes the source documents, restores the job files and state files
   (the run logs from their storage keys), and renames `class_registry.json.migrated-0.7.0` back;
3. verify the recreated documents against the bundle manifest;
4. mark the Postgres data abandoned (review fix R-9c): after a confirmation, the installer drops and
   recreates the `openprocessor` database (the Valkey volume is left unused), so a later re-upgrade
   is a fresh upgrade and cannot collide with the first one's rows;
5. only then start the 0.6.0 services (a 0.6.0 API that started earlier would create the missing
   control indexes empty).

Anything created after the upgrade (classes, configs, jobs) exists only in Postgres and is not
carried back; INSTALLATION's downgrade note says to export first. R-final rehearses upgrade,
rollback and re-upgrade on a real 0.6.0 install (RF-40, RF-62, RF-63: the re-upgrade gives the same
counts and ids as the first upgrade).

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
- `tests/upgrade/test_upgrade_gate.py` (`db`): with `upgrade_state = in_progress` a process start
  refuses with `upgrade_incomplete`; `--reset` after a committed load and a forced verify failure
  gives the same rows on the re-run; `--reset` after `delete_sources` refuses.
- `tests/upgrade/test_load_large_ledger.py` (`db`): a 1,000,000-row ledger loads through `COPY` in
  one transaction; the time is recorded in the PR.
- The fixture holds a deprecated class and an active class with the same name, and names that differ
  only by case, spaces and underscores (review fix R-3).
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
| `postgres` | `postgres:18.6-trixie` by the digest of [01](01_package_pins.md) section 4 (owner decision 2026-11-04) ([image](https://hub.docker.com/_/postgres)) | `postgres -c max_connections=${OP_PG_MAX_CONNECTIONS:-300} -c shared_buffers=${OP_PG_SHARED_BUFFERS:-512MB} -c password_encryption=scram-sha-256`; env `POSTGRES_USER=op_super`, `POSTGRES_DB=openprocessor`, `POSTGRES_PASSWORD_FILE=/run/secrets/op_pg/super_password`; `shm_size: 256m`; `stop_grace_period: 60s`; `deploy.resources.limits.memory: ${OP_PG_MEM_LIMIT:-4g}` (R-19) | `pg_isready -h 127.0.0.1 -U op_super -d openprocessor` (TCP, so the socket-only server the entrypoint runs during first init does not count; spike S0 verifies) | `openprocessor-pg:/var/lib/postgresql` (**18 moved `PGDATA` to `/var/lib/postgresql/18/docker` and the volume to `/var/lib/postgresql`**; 17 and older mount `/var/lib/postgresql/data`) | the entrypoint starts as root and drops to `postgres`; the secret must be readable by that user (S0 records the ownership rule) |
| `valkey` | the Valkey image and digest of [01](01_package_pins.md) section 4 (`9.1.3-alpine3.24`) ([image](https://hub.docker.com/r/valkey/valkey)) | `valkey-server /run/secrets/op_valkey/valkey.conf --appendonly yes --appendfsync everysec --maxmemory ${OP_VALKEY_MAXMEMORY:-1gb} --maxmemory-policy noeviction` (the conf holds only `requirepass`); `deploy.resources.limits.memory` above `maxmemory` (R-19) | `sh -c 'REDISCLI_AUTH="$(cat /run/secrets/op_valkey/password)" valkey-cli ping \| grep -q PONG'` (password not in argv; S1 verifies `valkey-cli` honours the variable, else `redis-cli`, which the image also ships, as Immich uses) | `openprocessor-queue:/data` | no backup (D-24) |
| `db-migrate` | api image | `python -m scripts.db.migrate upgrade` (Alembic, then the OpenSearch step, R-7, RC-1); `restart: "no"`; mounts the superuser, owner and app password files; waits on `postgres` and `opensearch` healthy | none (one-shot) | none | every Python service below `depends_on` it with `service_completed_successfully` (D-69) |
| `jobs-worker` | api image | `celery -A src.tasks.app worker -Q jobs --concurrency=${OP_JOBS_CONCURRENCY:-4} --max-tasks-per-child=50 --max-memory-per-child=${OP_JOBS_MAX_MEMORY_PER_CHILD_KB:-8000000} --hostname jobs@%h -E`; `REMAP_SIGTERM=SIGQUIT`; `stop_grace_period: ${OP_STOP_GRACE_JOBS:-60}s`; own `PROMETHEUS_MULTIPROC_DIR` tmpfs; `deploy.resources.limits.memory: ${OP_JOBS_MEM_LIMIT:-24g}` (R-19) | the ported pool healthcheck (`scripts/healthcheck/celery_pool_healthcheck.py`) | storage roots as `api`, no `/jobs` | `API_GPU_ID` reservation |
| `periodic-worker` | api image | `-Q periodic -c 2 --max-tasks-per-child=200 --hostname periodic@%h -E`; memory limit `${OP_PERIODIC_MEM_LIMIT:-2g}` (R-19) | pool healthcheck | as `api` | never mounts the Docker socket (R-20) |
| `api` (changed) | api image | as today, plus `deploy.replicas: ${OP_API_REPLICAS:-1}` (RC-4: the one source of the replica count, read by the budget test; scale through `.env`, not `--scale`) | `/ready` | as today | waits on `db-migrate` (completed) and on `postgres`, `valkey`, `opensearch`, `triton-server` healthy (D-69) |
| `gpu-arbiter-worker` (GPU-arbiter overlay only) | api image | `-Q arbiter -c 1 --hostname arbiter@%h` | pool healthcheck | as `api` | the only service that mounts the Docker socket; consumes only `arbiter` (review fix R-20) |
| `open-vocab-worker` | api image | `-Q open_vocab -c 1 --max-tasks-per-child=50` | pool healthcheck | as `api` | |
| `jobs-beat` | api image | `celery -A src.tasks.app beat -s /run/beat/celerybeat-schedule` (tmpfs); memory limit `${OP_BEAT_MEM_LIMIT:-512m}` (R-19) | newest mtime of `/run/beat/celerybeat-schedule*` under 300 s, failing on an empty glob | tmpfs only | single-beat lock (D-60); `deploy.replicas: 1` |
| `curation-auto-label-worker`, `curation-assist-loop-worker` (`curation`) | api image | `-Q autolabel -c 1 --max-tasks-per-child=20`, `-Q assist_loop -c 1 --max-tasks-per-child=20` | pool healthcheck | as today minus `/jobs` | |
| `cluster-worker` (`curation`; research clustering plan 5.2, decided, RD-25) | api image | `-Q cluster -c 1 --max-tasks-per-child=1` (a fresh child per run releases memory); CPU only in 0.7.0 (`CLUSTER_GPU_ID` only after a passing `cluster_backend_validations` row and the owner's yes on a RAPIDS layer) | pool healthcheck | storage roots as `api` | added by lane 1C; its first task arrives with lane 4F (W2.3a) |
| `text-read-worker` (`curation`; W5e, research plate plan 5.3) | api image | `-Q text_read -c 1 --prefetch-multiplier=20`; no GPU device (the reader is a Triton model) | pool healthcheck | as `api`; the `identity_text` secret read-only | added by `w5e/region-text-jobs`, not in B2 |
| `curation-trainer` (`training`) | trainer image | `celery -A src.tasks.runtime.gpu_app worker -Q gpu_train -c 1 --max-tasks-per-child=1` | pool healthcheck | storage roots, `/jobs` removed | |
| `model-export-worker` (`export`) | export image | `celery -A src.tasks.runtime.gpu_app worker -Q gpu_export -c 1 --max-tasks-per-child=1` | pool healthcheck | models, storage roots | |
| `curation-evaluator` (`evaluator`) | evaluator image | `-Q gpu_eval -c 1 --max-tasks-per-child=5` | pool healthcheck | storage roots | |
| `flower` (`monitoring`) | api image | `celery -A src.tasks.app flower --conf=/app/scripts/flower/flowerconfig.py` (review fix R-30: Flower documents `--basic-auth` and `FLOWER_BASIC_AUTH`, not a file option; its `--conf` Python file reads `/run/secrets/op_flower/auth` at start, so the secret is never in argv or the environment; spike S1 verifies), URL prefix for the gateway | HTTP `/healthcheck` | none | published only on `${OP_BIND_ADDRESS:-127.0.0.1}:4613` (ENV-18) |

Secrets (generated by the installer with `umask 077` before compose runs, so Docker never creates a
root-owned directory; never rotated on upgrade; created on `upgrade` and `repair` when missing;
redacted by `config show`): `secrets/postgres/owner_password`, `secrets/postgres/app_password`,
`secrets/postgres/super_password` (R-13), `secrets/valkey/password`, `secrets/valkey/valkey.conf`,
`secrets/flower/auth`. They are mounted
read-only as directories (`/run/secrets/op_pg`, `/run/secrets/op_valkey`, `/run/secrets/op_flower`),
the pattern `secrets/vlm` already uses. Services get only the files they need: the API and workers
get the app password and the Valkey password; only `db-migrate` and the backup command get the owner
password; only `postgres` and `db-migrate` get the superuser password.

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
- Research seams (lane 1E): crop cache keys carry the size variant,
  `projects/<slug>/crops/<item>/<box>/<max_side>.jpg` with `0` for the source size, so a cached
  source-size crop is never served as a capped one (GPU S-10, used by #219's `crop_sizing` in W5c);
  `delete_prefix` removes every area under `projects/<slug>/`, including an area no code lists yet
  (the spatial plan's later `spatial/` area: door test
  `tests/clients/storage/test_delete_prefix_unknown_area.py`). Loop node member snapshots
  (`projects/<slug>/assist_loop/<run>/<path>.json`) and cluster audit blobs use the same seam.
- For the v0.8.0 S3-compatible backend (feature F10 of
  [10_v080_platform_breadth.md](10_v080_platform_breadth.md)): `local_path(key)` must
  materialise the object to job-local scratch and write back with `write_atomic`; research scratch
  databases (COLMAP's SQLite in the spatial plan) are always job-local and never shared.

## 6. Backup and restore (D-61)

| Command | Does |
|---|---|
| `openprocessor db backup [--out <dir>]` | `docker compose -p <p> exec -T postgres pg_dump -U op_owner -d openprocessor -Fc` to `${OP_DB_BACKUP_DIR:-<install>/backups/postgres}/openprocessor-<utc>.dump` (`OP_DB_BACKUP_DIR` may name another mount or disk, so a backup does not share the failure of the data volume; the directory must be writable by the backup user and `db verify` reports a backup directory on the same filesystem as the Postgres volume; custom format: compressed, restorable selectively; consistent without blocking writers: [pg_dump](https://www.postgresql.org/docs/current/app-pgdump.html)); keeps the newest `OP_DB_BACKUP_KEEP` (default 7) |
| `openprocessor db restore <file> [--yes]` | stops the API and workers, runs `db-migrate`'s role bootstrap (so `op_owner` and `op_app` exist on a fresh host), `pg_restore --clean --if-exists --no-owner --no-privileges --role=op_owner -d openprocessor` (review fix R-13: no grant is restored before its role exists), runs `db-migrate` (it re-applies the grants and upgrades a dump from an older schema), **`TRUNCATE events`** (D-73: `events.xid` defaults to `pg_current_xact_id()`, which belongs to one cluster's transaction counter and does not carry across a restore into a new cluster, so old cursors would compare wrongly; the rows are only the 24-hour UI feed and the tables of each topic are the record), runs `openprocessor db verify`, starts the stack |
| `openprocessor db verify` | the D-47 class reconciliation, `project_indexes` vs OpenSearch, `storage_key` existence for promotions and training runs; report only |
| `openprocessor upgrade` | always takes a backup first when Postgres exists |
| optional beat task | `op.periodic.db_backup` when `OP_DB_BACKUP_INTERVAL_H` is set (off by default) |

**Recovery point, stated plainly.** After a failure you get back the state of the **last `pg_dump`**: with the default (a backup at each `openprocessor upgrade`, plus `op.periodic.db_backup` when `OP_DB_BACKUP_INTERVAL_H` is set, off by default) that can be hours or days old, and everything written since is lost. **Point-in-time recovery (continuous WAL archiving) is a conscious non-goal for v0.7.0**: it needs an archive target, a base-backup schedule and a restore drill that this release does not budget; it is a candidate for v0.8.0 multi-host. The 0.6.0 upgrade also `TRUNCATE`s `events` after the load (the table is new in 0.7.0; the rule is the same as for restore, so no code path depends on a cursor from another cluster).

Order rule (Immich: "back up the database first, then the files"): a database restored to an older
point than the OpenSearch data can name a class id the registry does not have; `db verify` reports
such items, and the docs say to restore both from the same time. OpenSearch snapshots stay out of
scope (unchanged).

## 7. Tests (D-62)

| Tier | What | How |
|---|---|---|
| Offline, Docker-free (`-m 'not live and not families and not db'`) | everything whose outcome depends on no SQL behaviour: wire models, config, the kind table, compose renders, Celery config, and services whose test checks no constraint, lock or RLS (in-memory fakes only there; every service test whose outcome depends on a constraint, lock or RLS is `db`, review fix R-14) | review fix R-4: today every entry point says `-m 'not live and not families'` (`pyproject.toml` addopts, `.github/workflows/ci.yml` twice, `CONTRIBUTING.md`, `CLAUDE.md`), and an explicit `-m` replaces the addopts one. Lane 1-0 changes the addopts expression to `not live and not families and not db`; lane 1D changes both `ci.yml` commands (and PR #477's shard command), `CONTRIBUTING.md` and the `Makefile` `test` target's help text; lane 6D changes `CLAUDE.md`; `tests/test_db_marker_policy.py` asserts the default selection holds no `db` test |
| `db` (new marker) | repositories, the claim, heartbeat and fencing statements, unique and partial indexes, RLS, the transition trigger, the reaper rules, the events cursor, migrations, the 0.6.0 upgrade | a real Postgres 18 from **testcontainers-python** (`PostgresContainer`, image pinned as in compose) on a developer machine, or from `OP_TEST_DATABASE_URL` (a CI service container: [PostgreSQL service containers](https://docs.github.com/en/actions/use-cases-and-examples/using-containerized-services/creating-postgresql-service-containers)). Selected with `-m db`; when selected and no database can be reached, the fixture **fails** (never skips) |
| Live harness | the job and event behaviour with real workers | `docker/test/compose.yml` with the new services |

Fixture design (`tests/db/conftest.py`, registered for the whole suite):
- one container per test session **shared by all xdist workers** (review fix R-14; 24 containers at
  `-n 24` would load the shared host): the controller starts it in `pytest_configure` and passes the
  URL to the workers through the environment; in CI the service container plays that role through
  `OP_TEST_DATABASE_URL`;
- each xdist worker migrates its own template database `op_template_<worker>` (keyed by
  `PYTEST_XDIST_WORKER`) once to head, so workers never race on one template;
- each test module gets `CREATE DATABASE op_t_<n> TEMPLATE op_template_<worker>` (fast copy) so concurrency
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
  that polls between the commits emits every event exactly once; the global stream (`project_id IS NULL`) has its own case and its own `EXPLAIN` index check.
- `tests/db/test_events_restore_resume.py` (D-73): write events, `pg_dump`, `pg_restore` into a **second cluster** (a second test container), run `db restore`'s truncate, write new events there, and assert a reader resumed with a cursor from the first cluster gets the `reset` event once and then every new event exactly once.
- `tests/db/test_role_settings.py` and `tests/db/test_jobs_hot_updates.py` of [02a](02a_postgres_schema.md) sections 1.5.1 and 5; `tests/tasks/test_poison_redelivery.py` of [02b](02b_jobs_locking_events.md) section 7.1.1.
- `tests/db/test_class_registry.py`: 10 concurrent `ensure_class` for 50 names: 50 classes, unique
  ids, `next_class_id = 50`; rename writes the outbox row in the same transaction.
- `tests/config/test_db_connection_budget.py` (pure, no database; the name 02b, 10 and 11 use): section 8 of
  [02b](02b_jobs_locking_events.md), with the scale factors (R-6).
- Research seams ([02d](02d_research_seam_schema.md) section 6): the schema test of 02d, the
  partition immutability and NULL-project foreign-key tests, the registry and loop schema tests,
  `tests/db/test_profile_fk_cascade.py`; the job-control tests of 02b section 3.4 (three pause and
  resume cycles then a fourth claim; a yielded slice keeps its retry budget; the reaper's
  system-pause resume); `tests/db/test_near_dup_current_flip.py` (two `current` runs impossible) with
  ND-A; `tests/upgrade/test_research_steps.py`: each step module of section 2.2 is discovered, runs
  in `ORDER`, and is idempotent on the 0.6.0 fixture.

Each stage file lists the fake-based test modules it deletes or rewrites (review fix R-14):
`tests/_fake_project_registry.py` (2A), `tests/_registry_doc_fake.py` (2B),
`tests/_fake_lease_opensearch.py` (6A deletes it with the leases), and the 13 `FileJob` test modules
(rewritten by the stage-4 and stage-5 lane that owns each family).

The 1-vs-N replica equivalence test runs in B4's live run (the third budgeted run,
[05_b4_stateless_api.md](05_b4_stateless_api.md) section 7), in the three configurations of 05
(`API_WORKERS=1`, one `api`, one `jobs-worker`; `API_WORKERS=32`, one replica; `API_WORKERS=4`, two
`api` and two `jobs-worker` replicas), as two scenarios (review fix R-15): (a) a **sequential**
scenario whose normalized records (class ids by name, config revisions, terminal job states) must be
identical across the configurations; (b) a **concurrent** scenario judged by invariants (exactly one
winner per singleton start, no lost or duplicated event per cursor, `next_class_id` equals the number
of distinct names), because which racer wins legitimately differs between runs.

## 8. Trap register (ways it looks done but is not)

| # | Trap | Guard |
|---|---|---|
| T-1 | An async engine created in one `asyncio.run` loop is reused in the next task's loop ("attached to a different loop") | per-task async engines with `NullPool`, disposed in `finally`; `tests/tasks/test_engine_per_task.py` runs two tasks in sequence |
| T-2 | An engine created at import is inherited by Celery prefork children and two processes share a socket | lazy engines keyed by `os.getpid()`; `worker_process_init` disposes with `close=False`; a fresh-interpreter test: importing `src.tasks.app` creates no engine and opens no socket |
| T-3 | `expire_on_commit=True` makes an attribute read after commit do implicit I/O (`MissingGreenlet`) | `async_sessionmaker(expire_on_commit=False)`; relationships `lazy="raise"` |
| T-4 | Select-then-insert under READ COMMITTED races | unique constraints and `INSERT ... ON CONFLICT`; ratchet: no `exists()` check before an insert in `src/db/repos` |
| T-5 | A status added in Python but not to the `CHECK`, the trigger or the active-set index | all three generated from `src/domain/job_states.py`; `tests/db/test_job_transitions.py` |
| T-6 | A query on a project table without the scope helper | RLS raises `scope_not_set` (fail closed and loud, R-13; review RR-23) and `tests/db/test_row_level_security.py`; a ratchet on `src/db/repos` for direct `session.execute` outside `project_tx`/`deployment_tx` |
| T-7 | The table owner bypasses RLS | the app connects as `op_app`; `FORCE ROW LEVEL SECURITY`; a test connects as `op_app` |
| T-8 | `set_config(..., false)` (session level) leaks a project binding to the next request on a pooled connection | only `is_local = true`; a test reuses one pooled connection for two projects |
| T-9 | The message is sent before the commit, and the worker cannot see the row | `start_job` commits first; a test with a fake producer asserts the order |
| T-10 | A duplicate delivery runs the job twice | the claim `UPDATE ... WHERE status = 'queued'`; Celery does not de-duplicate by `task_id` |
| T-11 | A visibility-timeout redelivery during a 24 h job | the claim finds `running` and acks; expected every 6 h for any task longer than that and **not to be tuned** ([02b](02b_jobs_locking_events.md) section 7.1.2) |
| T-12 | Lease expiry compared on a worker clock | expiry only in SQL with `now()` |
| T-13 | A long transaction around an OpenSearch or VLM call holds row locks and the events `xmin` | rule 4; a ratchet over `src/services` for network clients used inside `project_tx` blocks (AST: calls on `OpenSearch*`, `httpx`, Triton clients inside `async with project_tx`) |
| T-14 | A bulk insert of a million `ingest_source_files` rows in one transaction stalls the event feed | batches of 5,000 per transaction |
| T-15 | the Valkey wake-up assumed durable | the consumer always re-reads from its cursor; the 2 s safety tick; a test drops every wake-up and still delivers every event |
| T-16 | the beat lock or the migration lock sent through a transaction pooler | `OP_DB_DIRECT_HOST`; a test that the pooled engine never issues session locks |
| T-17 | a kombu or redis-py the Celery release does not accept, or redis-py 8's RESP3 default misbehaving against Valkey | exact pins `celery==5.7.0`, `kombu==5.7.0`, `redis==8.1.0` ([01](01_package_pins.md) section 4, D-70); `pip check` in each image; spike S1 on the pins includes a Valkey round trip with `protocol=2` as the fallback |
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
