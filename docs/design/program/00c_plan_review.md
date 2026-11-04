# 00c: independent review of the v0.7.0 plan set (before any code)

| Field | Value |
|---|---|
| Status | review, written 2026-11-04 on `plan/review` (from `plan/master`) |
| Reviewer | an independent reviewer (operations, Python back-end architecture, PostgreSQL, Celery, Docker) who did not write the plans |
| Scope | [00_master_plan.md](00_master_plan.md), [00a_decision_register.md](00a_decision_register.md), [00b_traceability.md](00b_traceability.md), [02](02_control_plane_database.md), [02a](02a_postgres_schema.md), [02b](02b_jobs_locking_events.md), [02c](02c_migration_ops_tests.md), [03](03_b2_overview.md) to [03e](03e_stage5_gpu_queues.md), [04](04_b3_ultralytics_one_version.md), [05](05_b4_stateless_api.md), [07](07_release_acceptance.md), the 07 files, [08](08_environment_and_setup.md), [10](10_standards_and_libraries.md), [11](11_risks_cleanup_host.md), [01](01_package_pins.md) |
| Ground truth | the code on `plan/master` (`src/`, `scripts/`, compose files, Dockerfiles, `tests/`, `.github/workflows/`, `scripts/codegen/`, `monitoring/`), the `v0.6.0` tag, PyPI metadata and the official docs cited in each finding |
| Method | every claim checked against the code or a doc, not against the plans' own description; line numbers are from `plan/master` on 2026-11-04 and move, so re-find by symbol |

---

## 1. Summary in plain words

**Verdict: yes, with fixes.** Do not start W2.1 until the three blockers below are fixed in the
plan text. Each fix is small and stays inside the plan's own design. None of them needs a new
component.

The plan's foundation is sound. It uses PostgreSQL as the system of record. A job is a database row
that Celery only carries. Claims are conditional `UPDATE`s with a fencing token, a partial unique
index makes singletons, the reaper rebuilds lost messages, there is no result backend and Alembic
runs from a one-shot service. This is the documented pattern, and it matches the public projects it
cites. The duplicate-delivery analysis is right: redelivery, visibility-timeout restore and
`task_reject_on_worker_lost` loops are all absorbed by the claim and the attempt counter. The
traceability work is honest and almost complete: every one of the 44 open issues is placed, and the
15+ rows I spot-checked against `origin/main` hold.

What is wrong falls into five groups:

1. **The plan is wrong about three library and framework rules** (the blockers):
   - FastAPI's default `yield` dependency commits after the response has gone;
   - the GPU-image runtime package is forbidden to import the heartbeat, event and OpenSearch
     code it must call;
   - the class-name uniqueness constraint does not match the registry's real name rule.
2. **Scale-out has unbudgeted costs.** These are:
   - one `LISTEN` connection per API process, which no pooler can carry;
   - a connection "total" knob that is not a total;
   - OpenSearch index bootstrap and mapping migrations that still run in every API process;
   - stream-worker claims that never rebalance;
   - Prometheus and Loki configurations that see one replica or none of the new services.
3. **There are failure paths without a defined end state:**
   - a unique violation inside the reaper's batch transaction wedges rule b;
   - an upgrade whose verify step fails leaves a half-migrated install that can still be started;
   - re-upgrade after rollback (RF-63) contradicts the upgrade's own preconditions.
4. **Tests and selection.** The new `db` marker is never deselected by default, so the offline
   command in CLAUDE.md, CONTRIBUTING and CI collects tests that fail without Docker from the day
   lane 1A merges.
5. **Pins.** kombu 5.7.0 came out yesterday and Celery 5.6.3 predates it by seven months. 02b and 13
   disagree about which redis-py version it accepts.

Counts: **3 blockers, 9 high, 15 medium, 11 low.**

## 2. Findings by severity

Each finding gives: where (file and section), what is wrong, the evidence, and the fix to the plan
text.

### 2.1 Blockers

**R-1 The per-request database session commits after the response is sent (02b section 8; 03a 1A-10).**
- *What.* 1A-10 specifies `DbSessionDep` as a `yield` dependency that "commits on a normal request"
  and rolls back on an exception. FastAPI's default dependency scope is `"request"`, and its exit
  code runs *after* the response is sent. The effects:
  - A commit that fails (a serialization error, a constraint violation, Postgres going away) is
    reported to nobody. The client already has a 200 or 202. This fails open, against rule 7 of
    02 section 2.
  - Any streaming route that declares the dependency holds a pooled connection for the whole
    stream. With `pool_size=2, max_overflow=1` per API process, three SSE clients exhaust that
    process's pool.
- *Evidence.* FastAPI, "Dependencies with yield", section "Early exit and `scope`": "Normally the
  exit code of dependencies with `yield` is executed after the response is sent to the client" and
  "If not specified and the dependency has `yield`, it will have a `scope` of `"request"` by
  default" (https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/).
  `fastapi==0.162.0` (`requirements.txt:2`) has the `scope` parameter.
- *Fix (1A-10 and 02b section 8):*
  - `DbSessionDep = Annotated[AsyncSession, Depends(get_db, scope="function")]`.
  - The dependency only opens and closes the session. Every write commits explicitly inside
    `project_tx` / `deployment_tx` (`async with session.begin()`) in the service, before the
    handler returns.
  - SSE and streaming routes never declare `DbSessionDep`; they use the listener's own connection
    (3B).
  - The red test is "a commit failure turns into a 5xx, not a 2xx" and "an SSE route holds no
    pooled connection".

**R-2 The GPU-image runtime package may not import the code it must call (02 section 6; 02b section 3.1; 03e 3A-4, 3A-5; 03f).**
- *What.* `src/tasks/runtime/` is limited to `src.db`, `src.config`, `src.domain` and `src.utils`.
  A fresh-interpreter test checks the limit, and the trainer, export and evaluator images copy only
  those packages. But the wrapper the same files specify calls:
  - the heartbeat thread, placed in `src/services/jobs/heartbeat.py` (02 section 6; 3A-4);
  - the event publish (`src/services/events/publish.py`, 3B-1), for `job.state` / `job.progress`
    rows and `pg_notify`;
  - `make_script_opensearch` through the `JobContext` client factory (3A-5).
  The last of these lives in `src/services/projects/guard.py:142`. `check_layering.py` checks
  function-level (lazy) imports too (its module docstring), so "the factory imports them lazily"
  does not pass the check.
- *Evidence.*
  - `src/services/projects/guard.py:142` (`def make_script_opensearch`).
  - `scripts/codegen/check_layering.py` docstring: "at module level and inside functions".
  - 02 section 6, the `src/tasks/runtime/` row.
- *Fix.*
  - Move the heartbeat thread, the fenced job-row writes and the event-row insert plus `pg_notify`
    into `src/tasks/runtime/` (or `src/db/repos/`). Keep `src/services/jobs/` for start, cancel,
    resume and status, which are API-side.
  - Inject the OpenSearch client factory into `JobContext` from the image-specific handler module
    (`src/tasks/jobs_*.py` for the api image; GPU handlers that need OpenSearch get it from their
    own module), so `src/tasks/runtime/` never names `src.services`.
  - Update 3A's and 3B's owned-file lists to match.
  - Add a planted-import red test to 1A-11.

**R-3 The class-name uniqueness in the schema does not match the registry's name rule (02a section 3; 03d 2B-1; 02c section 2).**
- *What.* `project_classes` has `UNIQUE (project_id, name)` on the raw name. The current rule
  differs in two ways:
  1. Names compare by `normalize_class_name` (lowercased, runs of other characters collapsed to
     `_`), so "Car", "car" and "car " are one class and "traffic light" equals "traffic_light".
     The raw unique key lets two active classes with equal normalized names exist, which breaks
     class identity by name at the database layer.
  2. "a name may be reused after deprecation, so both can exist". The raw unique key refuses a
     legitimate 0.6.0 registry that has a deprecated class and an active class with the same name,
     so the one-transaction upgrade `load` aborts on real installs. It also removes the
     reuse-after-deprecation behaviour without an owner decision.
  The 02a "Name matching" rule ("encodes the same normalization before the unique comparison") is
  application-only. 1A-4's `test_schema_matches_plan.py` would lock the wrong constraint in.
- *Evidence.*
  - `src/utils/class_names.py:157-166` (`normalize_class_name`).
  - `src/utils/class_names.py:194-214` (`resolve_class_by_name`: "a name may be reused after
    deprecation, so both can exist"; an active class wins over a deprecated one).
  - `src/clients/curation_opensearch/registry.py:238-248` (`_add`).
- *Fix (02a section 3):*
  - Add `name_key text NOT NULL CHECK (name_key ~ '^[a-z0-9_]+$')`, written only by
    `src/services/classes/resolve.py` from `normalize_class_name`.
  - Replace the raw `UNIQUE (project_id, name)` with
    `CREATE UNIQUE INDEX uq_project_classes_active_name ON project_classes (project_id, name_key) WHERE NOT deprecated`.
  - `ensure_class` looks up by `name_key` with the same "active wins, then exact spelling, then
    lowest id" rule.
  - Add to the 2E fixture a deprecated class and an active class with the same name.
  - Add to 2B-1's test "Car" / "car" / "traffic light" vs "traffic_light", plus a non-slug name
    (which refuses).

### 2.2 High

**R-4 The `db` marker is never deselected, so the offline suite breaks (02c section 7; 03a lane 1-0, 1D-2; CLAUDE.md).**
- *What.* 02c says the Docker-free command is "`-m 'not live and not families and not db'`, the
  command CLAUDE.md names". That is not true. Every entry point names
  `-m 'not live and not families'`:
  - `pyproject.toml` addopts;
  - `.github/workflows/ci.yml` (two places);
  - `CONTRIBUTING.md`;
  - `CLAUDE.md`.
  An explicit `-m` on the command line replaces the addopts one. The new `db` fixture "fails, never
  skips". So from the merge of lane 1A, the CI shards, CLAUDE.md's offline command and every
  developer's default `pytest` collect the `db` tests and fail without Docker. No lane owns that
  change: lane 1-0 adds only the marker, and 1D says the pytest job "keeps the Docker-free
  command".
- *Evidence.*
  - `pyproject.toml:332-341` (addopts `-m "not live and not families"`).
  - `.github/workflows/ci.yml:69` and `:102`.
  - `CONTRIBUTING.md:39`, `:42`.
  - `CLAUDE.md:32`, `:226`.
  - `Makefile` `test` target.
- *Fix.*
  - Lane 1-0 changes the addopts expression to `not live and not families and not db`.
  - Lane 1D updates both `ci.yml` commands (and PR #477's shard command if it lands first) and
    `CONTRIBUTING.md`.
  - 6D updates CLAUDE.md. The Makefile `test` target goes to 1D.
  - Add `tests/test_db_marker_policy.py` (1D-3): the default selection contains no `db` test.

**R-5 Unique violations abort the surrounding transaction, which wedges reaper rule b and breaks the start path (02b sections 1, 2.2, 4).**
- *What.* Three related defects:
  1. Reaper rule b moves up to 200 `interrupted` rows to `queued` in one transaction. If another
     job of the same group became active meanwhile (allowed: `interrupted` is not active), that
     `UPDATE` raises `unique_violation` on `uq_jobs_one_active`. The whole batch rolls back, and
     the same row is picked again on every tick, so rule b stops for every row. Resume (section
     3.3) has the same race.
  2. `start_job` step 1 "answers 409 `job_busy` with the active job's id (read in the same
     transaction)". After a unique violation, PostgreSQL refuses every further statement in that
     transaction ("current transaction is aborted"). The same applies to the idempotency-key read.
  3. The transition table has no `interrupted -> failed`, but rule b writes `failed
     (attempts_exhausted)` from `interrupted`. The generated trigger
     `jobs_status_transition_guard` would raise on it.
- *Evidence.*
  - 02b section 1 table (no `interrupted -> failed` row), section 4 rule b, section 2.2 step 1.
  - PostgreSQL, "Transactions"/`SAVEPOINT`
    (https://www.postgresql.org/docs/current/sql-savepoint.html) and the error-handling
    behaviour of an aborted transaction
    (https://www.postgresql.org/docs/current/tutorial-transactions.html).
  - SQLAlchemy `Session.begin_nested()`
    (https://docs.sqlalchemy.org/en/21/orm/session_transaction.html#using-savepoint).
- *Fix.*
  - Wrap each per-row action in a savepoint (`begin_nested`), or exclude conflicting rows in the
    selecting statement
    (`AND NOT EXISTS (SELECT 1 FROM jobs a WHERE a.project_id IS NOT DISTINCT FROM j.project_id AND a.exclusive_group = j.exclusive_group AND a.scope = j.scope AND a.status IN (<ACTIVE>))`),
    and still use the savepoint for the race.
  - A row that loses stays `interrupted` and is logged once.
  - In `start_job`, insert inside a savepoint and read the active id after rolling back to it.
  - Add `interrupted -> failed` to `src/domain/job_states.py`.
  - Add to `tests/db/test_reaper_rules.py` "an interrupted job whose group is busy does not block
    the batch".

**R-6 Connection math: the "total" is per replica and per-process minimums dominate it; `LISTEN` is unpoolable and grows with workers times replicas (02b sections 6.3, 8; 05 4B-2).**
- *What.*
  - `pool_size = max(2, OP_DB_API_CONNECTIONS_TOTAL // API_WORKERS)` plus `max_overflow=1` plus
    one `LISTEN` gives at least 4 connections per process. At 32 workers that is 128 per replica
    against a "total" of 64, so the knob does not bound anything.
  - Each replica divides its own total, so `--scale api=2` at the installer default (`API_WORKERS =
    min(nproc, 32)`) needs 256 API connections. Adding 48 Celery, 2 beat, 4 stream and 13 reserved
    gives about 323 against `max_connections=300`.
  - `tests/config/test_db_connection_budget.py` reads the compose file without scale factors, so it
    stays green.
  - PgBouncer (D-50) does not help the largest term: `LISTEN` needs a direct session connection,
    and the plan opens one per API process. That is workers x replicas direct connections; 4
    replicas x 32 workers = 128 before any query traffic.
  - Every commit that issued `NOTIFY` also takes a database-wide lock while it queues the
    notification. That serializes those commits (`PreCommit_Notify` in PostgreSQL's
    `src/backend/commands/async.c`), and the plan puts `pg_notify` in the caller's transaction for
    every progress write.
- *Evidence.*
  - 02b section 8 table (128 for 32 workers) and section 8.1.
  - PgBouncer feature map ("LISTEN: never" in transaction mode, https://www.pgbouncer.org/features.html).
  - PostgreSQL NOTIFY (https://www.postgresql.org/docs/current/sql-notify.html).
  - 05 4B-1 installer default.
- *Fix.*
  - Make the API pool budget a real deployment total: per process =
    `floor(OP_DB_API_CONNECTIONS_TOTAL / (API_WORKERS * OP_API_REPLICAS))`, with a minimum of 1
    and no overflow. Or rename it "per replica" and state the formula.
  - The budget test takes `--scale` factors and fails above `max_connections - reserved`.
  - Stop opening one `LISTEN` per process. Either option works:
    - (a) recommended: the wake-up goes over Valkey pub/sub. After commit, the writer publishes
      the project id to `op:events:<project>`, and each API process subscribes on its existing
      Valkey client. Postgres stays the durable feed and cursor, and both `LISTEN` and the
      `NOTIFY` commit lock disappear. The 02b section 6.1 objection to pub/sub ("not resumable")
      applies equally to `NOTIFY`, which is also a wake-up only.
    - (b) keep `NOTIFY`, but document a hard cap (for example replicas x workers <= 64) and state
      that PgBouncer does not cover it.
  - Document when PgBouncer becomes mandatory with the formula, not "for several replicas".
  - Owner question Q-2.

**R-7 OpenSearch index bootstrap and mapping migrations still run in every API process (02 section 4.4; 02c section 1.2).**
- *What.* The plan moves all Postgres DDL into the one-shot `db-migrate`, because no app process
  should run migrations. But the API lifespan still runs, in every uvicorn worker of every replica,
  on start and after every `--limit-max-requests` recycle:
  - `create_all_indexes` for the global indexes;
  - `ensure_indexes` per project ("index creation, mapping migrations, classes self-heal");
  - a kNN warm-up task per project.
  With N replicas x 32 workers this is a thundering herd of index-existence checks and
  mapping-migration attempts against OpenSearch, racing each other on a schema change. It is the
  same defect class the plan fixes for Postgres. 02 section 4.4 keeps "kNN warm-up stays" and never
  names the bootstrap.
- *Evidence.* `src/core/dependencies.py:140-158`; `docker-compose.yml:264-265`
  (`--limit-max-requests=10000` with jitter).
- *Fix.*
  - Move index creation and mapping migrations for the global indexes and every project into the
    one-shot migration step. Either extend `db-migrate` into a `migrate` service that also runs an
    idempotent `scripts/opensearch/migrate.py`, or add a second one-shot service before `api`.
  - The API lifespan only verifies the mapping fingerprint (read-only) and refuses to start on a
    mismatch, like the Alembic head check.
  - Project creation (4D) creates its indexes in its job.
  - Bound the kNN warm-up to one process per replica, or drop it after 5a measures it.
  - Name an owner lane: 2F, which already owns `index_bootstrap.py`.

**R-8 Stream-worker project claims never rebalance, so adding detection or VLM replicas does nothing (02b section 5; 03e 3A-10).**
- *What.*
  - "Live replicas = distinct `worker_id` with a live claim, plus itself", and a replica only
    *adds* up to `ceil(active / live)` new projects per pass. Nothing makes a replica release
    projects it holds above its fair share.
  - Start with one replica: it claims every project and renews them forever. A second replica sees
    no expired claim and stays idle until the first one dies.
  - A replica with zero claims is also invisible to the "live replicas" count.
  - Separately, the claim statement bumps `fencing_token` on every 15 s renewal (the `DO UPDATE`
    runs for the holder too), so the token is not a stable fence.
- *Evidence.* 02b section 5 SQL and text; sweep step 13 starts both replicas at once, which hides
  the problem.
- *Fix.*
  - Count live replicas from a liveness row (`worker_capabilities.seen_at` within the TTL, written
    by every stream worker each pass).
  - A replica holding more than `ceil(active / live)` releases its excess (oldest first) at a
    project boundary.
  - Bump `fencing_token` only on takeover (`CASE WHEN stream_worker_claims.worker_id <> excluded.worker_id`).
  - Change sweep step 13 to "start one replica, wait for its claims, then start the second; both
    hold projects within two TTLs".

**R-9 A failed or interrupted upgrade has no defined end state, and re-upgrade after rollback contradicts the preconditions (02c sections 2.2, 2.3; 06 RF-62, RF-63).**
- *What.*
  - (a) If `verify` fails after `load` committed, Postgres holds the loaded rows and the 0.6.0
    sources still exist. Nothing stops `openprocessor start` or `docker compose up` from running
    0.7.0 on that state: the startup check compares only the Alembic head. Users would then write
    to Postgres while the sources the operator is still fixing diverge.
  - (b) A re-run has no specified path. `load` uses natural keys "so a re-run after a failure
    starts clean", which holds only if `load` itself rolled back. After a committed `load` the
    re-run collides or must be skipped, and there is no reset step.
  - (c) RF-63 (re-upgrade after RF-62 rollback: "migration is a no-op on equal docs; ids
    unchanged") cannot pass as designed. Rollback leaves the Postgres volume in place with
    `upgrade_runs(0.6.0, delete_sources) = completed`, and the step's precondition then refuses to
    run. If the rows were kept, any data created after the first upgrade conflicts.
  - (d) Rollback ordering is not specified. The 0.6.0 API lifespan creates missing control indexes
    empty (`ensure_indexes`), so if a 0.6.0 API starts before `rollback_to_060.py` re-indexes from
    the bundle, the recreate collides or 0.6.0 serves empty registries.
  - (e) 0.6.0 training run logs (`<id>.run.log`) live in the `openprocessor-jobs` volume, which
    the upgrade removes. The bundle sources (02c section 2.1) do not list them, so the history the
    upgrade preserves loses its logs.
- *Evidence.*
  - 02c section 2.2 steps 2-4 and preconditions; section 2.3.
  - `docs/design/program/07_release_acceptance.md` RF-62, RF-63.
  - `src/core/dependencies.py:140-152`.
  - `docker-compose.yml:876-891` (`OP_TRAIN_JOBS_DIR=/jobs`).
  - `docker/trainer/trainer.py:152` (`tee_to_run_log(spec.run_log_path)`).
- *Fix.*
  - Add a `deployment_settings('upgrade_state')` gate. Every 0.7.0 process refuses to start with
    `upgrade_incomplete` while an upgrade from 0.6.0 has started and `delete_sources` is not
    completed (or while `op_projects` still exists).
  - Add `upgrade_from_060 --reset` (one transaction: truncate the control tables, delete the
    `upgrade_runs` rows), allowed only before `delete_sources`.
  - Make rollback mark the Postgres data as abandoned (drop and recreate the database, or delete
    the volume after a confirmation), so RF-63 is a fresh upgrade. Reword RF-63 to "re-upgrade
    after rollback gives the same counts and ids as the first upgrade".
  - Order rollback as: OpenSearch up -> `rollback_to_060.py` -> verify -> 0.6.0 services up.
  - Add the run logs (and any other `/jobs` file a kept row points to) to the bundle and copy them
    to storage keys before the volume is removed.

**R-10 kombu 5.7.0 under Celery 5.6.3, and two contradicting redis-py ranges (13 section 4; 02b section 7.1).**
- *What.*
  - Celery 5.6.3 (2026-03-26, still the newest Celery) requires `kombu>=5.6.0` with no cap.
    kombu 5.7.0 was uploaded on 2026-11-03, the day before the survey. No Celery release has been
    tested against it.
  - kombu 5.7.0's `redis` extra also hard-pins `packaging==26.3`, which every other tool in the
    images must then accept.
  - 02b says "kombu 5.7.x accepts redis-py <8.0" and that redis-py is pinned to what kombu accepts.
    13 says kombu 5.7.0 "accepts `redis>=6.4,<9`" and pins redis-py 8.1.0. PyPI shows 13 is right
    (`redis<9.0.0,>=6.4.0`).
  - kombu 5.6.2 accepts `redis>=4.5.2,<6.5` (excluding two versions).
- *Evidence.* PyPI JSON for `celery/5.6.3` (`kombu>=5.6.0`, `billiard<5.0,>=4.2.1`), `kombu/5.7.0`
  (redis extra `redis<9.0.0,>=6.4.0`, `packaging==26.3`), `kombu/5.6.2` (redis extra
  `redis!=4.5.5,!=5.0.2,<6.5,>=4.5.2`, released 2025-12-29), read 2026-11-04.
- *Fix.*
  - Pin `kombu==5.6.2` and the newest redis-py 6.4.x until a Celery release declares support for
    kombu 5.7, and record this as a held-back row in 13 section 1. Owner question Q-3.
  - Correct the 02b sentence.
  - Spike S1 runs on the chosen set and adds a check that `pip check` passes in the api image with
    the `redis` extra.

**R-11 The inference API now cannot start without PostgreSQL and Valkey (02c sections 1.2, 4; 03a 1C-2; VISION).**
- *What.*
  - Every service waits on `db-migrate` with `service_completed_successfully`.
  - `api` also waits on `valkey` being healthy.
  - The API lifespan exits on `schema_not_current`.
  So `/detect`, `/embed`, `/faces` and `/ocr`, which never touch control data, are down whenever
  Postgres, Valkey or a migration is down. 1D-1 says the opposite at runtime ("degraded, HTTP 200,
  inference routes unaffected"). The vision says "the backend stands on its own ... as an inference
  engine and API by itself", and the owner's goal is "OpenProcessor first: full inference app
  first". The plan never asks whether the core inference tier should take a hard dependency on the
  curation control plane.
- *Evidence.* 02c section 1.2 ("Every other service has `depends_on: db-migrate`") and the
  schema-check bullet; 03a 1C-2; `docs/VISION_AND_GOALS.md` "The standards this codebase is held
  to".
- *Fix (recommended).*
  - The `api` service does not `depends_on` `db-migrate` or `valkey`.
  - The lifespan treats an unreachable database as "control plane down": curation and job routes
    answer 503, inference routes serve, and `/health` reports it.
  - The process refuses to start only on a *reachable* database at the wrong head.
  - Workers and beat keep the hard dependency.
  - 1C-2's "every service except `db-migrate` waits on `db-migrate`" also cannot hold for
    `postgres` itself (a cycle). Restate it as "every Python service except `api`".
  - Owner question Q-1.

**R-12 Monitoring breaks with replicas and does not see the new services (03c 6A-4, 6B-6).**
- *What.*
  - Prometheus scrapes the static target `api:8000`. With `--scale api=2`, Docker DNS returns
    either replica per scrape, so the multiprocess counters of two independent tmpfs directories
    alternate and `rate()` produces garbage. The same applies to the new Celery metric ports.
  - Alloy keeps logs only for the services named in its regexes (`api`, `triton-server`, the
    curation workers, `segmenter`). `jobs-worker`, `periodic-worker`, `open-vocab-worker`,
    `jobs-beat`, `postgres`, `valkey` and the GPU consumers ship no logs.
  - The logs dashboard selects by `container`, whose values change once `container_name` is gone.
  - 6B owns only `monitoring/prometheus.yml`.
- *Evidence.*
  - `monitoring/prometheus.yml:16-19` (static `api:8000`).
  - `monitoring/alloy-config.alloy:58`, `:87` (keep regexes).
  - `monitoring/dashboards/logs.json` (`container=~"$container"` selectors).
  - Prometheus `dns_sd_configs` / `docker_sd_configs`
    (https://prometheus.io/docs/prometheus/latest/configuration/configuration/#docker_sd_config).
- *Fix.*
  - 6B also owns `monitoring/alloy-config.alloy` and the dashboards.
  - Prometheus uses `docker_sd_configs` filtered by the Compose project and service labels, with an
    `instance` label per container.
  - Alloy keeps by the Compose `service` label for every Python service and the two stores.
  - Dashboards select by `service`.
  - Add a render test that every service in `TaskQueues` consumers has a scrape and a log rule.

### 2.3 Medium

**R-13 Row-level security details (02a sections 1.4, 1.5, 7; 02c sections 6, 8).**
- `op_owner` is the image's `POSTGRES_USER`, which is a superuser. Superusers bypass RLS whether or
  not `FORCE` is set, so "FORCE makes the policy apply to the owner as well" protects nothing here,
  and the upgrade `load` (deployment scope) runs with RLS off.
- With `set_config(..., true)`, a session that commits and then auto-begins a new transaction
  silently loses its scope. Later reads return zero rows instead of an error, so a bug shows up as
  a wrong 404, not a failure.
- The consumer policy on `model_shares` (`USING (op_scope_allows(consumer_project_id))`) is
  permissive for every command. A consumer project could therefore insert a share of a model it
  does not own, because foreign-key checks bypass RLS. The plan also does not say how the consumer
  reads the shared `models` row.
- `pg_restore --no-owner --role=op_owner` restores `GRANT ... TO op_app` before `db-migrate`
  (re)creates `op_app` on a fresh host, so the restore reports errors.
- *Evidence.*
  - PostgreSQL row security: "Superusers and roles with the BYPASSRLS attribute always bypass the
    row security system" (https://www.postgresql.org/docs/current/ddl-rowsecurity.html).
  - Docker official image: `POSTGRES_USER` "will create the specified user with superuser power"
    (https://hub.docker.com/_/postgres).
  - SQLAlchemy session events `after_begin`
    (https://docs.sqlalchemy.org/en/21/orm/events.html#sqlalchemy.orm.SessionEvents.after_begin).
- *Fix.*
  - Bootstrap a non-superuser `op_owner` (schema owner, `BYPASSRLS` so `pg_dump` works) from the
    image superuser, and keep the superuser only for bootstrap.
  - Set the scope in an `after_begin` listener from a context variable, so every transaction of a
    scoped session carries it.
  - Make `op_scope_allows` raise when `op.scope` is unset, instead of returning false (still fail
    closed, but loud).
  - Make the `model_shares` consumer policy `FOR SELECT`, and give `models` a consumer `SELECT`
    policy through `EXISTS (model_shares ...)`.
  - Restore with `--no-privileges`, then let `db-migrate` re-apply the grants.

**R-14 Test-database fixture and the fake repositories (02c section 7; 03a 1A-12).**
- "One container per xdist worker" means 24 Postgres containers on the shared host at the usual
  `-n 24`.
- With `OP_TEST_DATABASE_URL` (one CI service container), every xdist worker would build
  `op_template` on one server, which races.
- The Docker-free tier tests services against "in-memory repository fakes". That is a second
  implementation of every repository, which drifts from SQL semantics (exactly the semantics the
  plan says SQLite cannot model). The plan does not size the fakes or the rewrite of the existing
  OpenSearch-fake tests (`tests/_fake_project_registry.py`, `tests/_registry_doc_fake.py`,
  `tests/_fake_lease_opensearch.py`, 13 `FileJob` test modules).
- *Fix.*
  - One container per session, shared by all xdist workers (start it in the controller, pass the URL
    through `PYTEST_XDIST_WORKER`-keyed environment), with `op_template_<worker>` names.
  - Fakes only where a test checks no SQL behaviour.
  - Every service test whose outcome depends on a constraint, lock or RLS is `db`.
  - Add a line to each stage file listing the fake-based test modules it deletes or rewrites.

**R-15 The 1-vs-N equivalence test is either flaky or weak (05 section 7; 02c section 7).**
- "Pass: an empty diff" over "event sequences per stream" after "start, cancel and race jobs"
  cannot be deterministic: which racer wins and how concurrent producers interleave legitimately
  differ between runs, even with one replica. A scenario made sequential to get an empty diff no
  longer tests concurrency.
- 02c names two configurations and 05 three.
- *Fix.* Split the test:
  - (a) a sequential scenario with an exact diff of normalized records (class ids by name, config
    revisions, terminal job states);
  - (b) a concurrent scenario judged by invariants (exactly one winner per singleton, no
    lost or duplicated event per cursor, `next_class_id` equals the distinct names).
  Use the three configurations of 05 in both files.

**R-16 Pins part A is coupled into the B2 PR (00_master_plan W2.1; 13 section 7.1).**
- 13 section 7 says the control-plane packages are kept out of the bump series "so that a Celery
  or database problem never blocks a dependency bump". The master plan then puts pins part A on
  `int/b2-control-plane`, so the coupling runs the other way. A pure-Python bump regression now
  blocks or confuses the largest PR of the program, and the B2 sweep cannot tell the two apart.
- *Fix.* Land part A as its own small PR to `main` before W2.1 (CPU-only, one offline sweep, which
  PR #477 makes cheap), then cut `int/b2-control-plane`. Owner question Q-4.

**R-17 GPU window 1 has no contingency (00_master_plan section 5; 04 section 6).**
- About 1.5 GPU days carry:
  - the Ultralytics pin and the contract change;
  - torch 2.15, OpenCV 5, numpy 2.5 and pandas 3, CuPy 14 and the ORT 1.30 / modelopt set;
  - every image smoke;
  - the B2 stage-5 smoke.
  The failure rule is "bisect by reverting pins part B commits ... re-run only failing steps",
  and each bisect step needs engine rebuilds. Several G11 items (`pip check`, imports, the
  segmenter weight download) need no GPU.
- *Fix.*
  - Move the non-GPU parts of G11 into the W3 sweep (CPU image builds).
  - Set a written cap (for example +1 day) and the named fallback pins of 13 section 8 as the first
    action on a failure, not a bisect.
  - Add `src/utils/scrfd_decode.py`'s `cv2.dnn.blobFromImage` to the OpenCV 5 parity set
    (`cv2.dnn` was reworked in 5.x).

**R-18 Lane interface dependencies are not handled (03a, 03d, 03e, 04).**
- File ownership is disjoint, but some lanes call code another lane is still writing:
  - 1B-3 calls 1A's `schema_check` from `worker_init`, while 1A runs in parallel.
  - 1C-7's budget test reads 1A's `DatabaseSettings`.
  - 3A (cut from the stage-1 head) needs project lookups that 2A's repository provides.
  - 2E writes the upgrade with the ORM models directly "not the stage-2 repositories", which gives
    a second write path with its own class-name normalization (see R-3).
- `scripts/curation/{repair_merged_active_classes,seed_class_registry,seed_live_harness}.py` call
  the registry API and are owned by no lane. 2B's ownership grep covers `src` only, so the live
  harness seed breaks when 2B deletes the registry document code.
- 04 section 4 says "B3 touches no B2 file". That is false: B3-E2 edits `export/export_worker.py`,
  which 5B rewrites, and stage A edits `docker/trainer/Dockerfile` and the root `Dockerfile`, which
  5A and 5B edit.
- *Fix.*
  - Merge order 1A before 1B and 1C, or have 1B and 1C code against a stub signature that 1A owns.
  - 3A cut after 2A merges.
  - 2E switches to the repositories in a coordinator commit after stage 2 merges.
  - Add the three scripts to 2B.
  - Correct 04 section 4, and require the B3 tracks to re-verify every file and symbol on the
    post-B2 head (its "verified against" note predates B2).

**R-19 No resource limits on any new service; Valkey has no `maxmemory` (02c section 4; 02b section 7.2).**
- `maxmemory-policy noeviction` acts only when `maxmemory` is set. With the default 0, Valkey
  grows until the host or the container is killed, which loses the AOF tail.
- No service gets a memory or CPU limit. That includes the `jobs-worker` that runs UMAP and cuML
  children.
- Celery's documented `worker_max_memory_per_child` is not used.
- *Evidence.* Valkey "maxmemory" configuration (https://valkey.io/topics/lru-cache/); Celery
  `worker_max_memory_per_child`
  (https://docs.celeryq.dev/en/stable/userguide/configuration.html#worker-max-memory-per-child).
- *Fix.*
  - Set `maxmemory ${OP_VALKEY_MAXMEMORY:-1gb}` with `noeviction`, plus an alert on `used_memory`.
  - Add `deploy.resources.limits.memory` for `postgres`, `valkey`, `jobs-worker`, `periodic-worker`
    and `jobs-beat`, configurable from `.env`.
  - Use `--max-memory-per-child` on `jobs-worker`.
  - Have the compose render test assert all three.

**R-20 The GPU arbiter's Docker socket shares a queue with every periodic task (03c 6B-4).**
- The overlay mounts the Docker socket into `periodic-worker`, which also consumes the reaper,
  ingest dispatch, cache prune and the other periodic tasks.
- Precondition (neutral): any process that can enqueue to the `periodic` queue directs a worker
  that holds host-level container control.
- *Fix.* A dedicated `arbiter` queue and a one-process `gpu-arbiter-worker` that consumes only that
  queue gets the socket. `periodic-worker` never does.

**R-21 B4 project creation is specified as a "plain task idempotent by the project record's `op_type=create`" (05 section 6).**
- That is an OpenSearch-record concept that B2 deletes.
- Every long job is a `jobs` row through `start_job` (D-41, 02b section 2.4), and the single-entry
  ratchet would reject a plain task.
- The kind table has no `project_create` row.
- *Fix.* Make it kind `project_create` (deployment-level, exclusive per slug, auto-resume), and
  register it in `src/config/job_kinds/b4.py`.

**R-22 Six v0.7.0 issues still have no wave (00b section 2; 00_master_plan MQ-6, MQ-7).**
- #465 and #468-#472 sit in the v0.7.0 milestone, but MQ-6 and MQ-7 are asked only before W2.2
  and W4.
- #472 (project isolation and purge for the core text index) is an isolation item, which the
  vision puts under adversarial review. Deciding it late risks a release blocker.
- *Fix.* Ask MQ-6 and MQ-7 before W2.1 (move them to section 9.1). If accepted, cut the lanes and
  check disjointness at the W4 cut. If not, move the issues to v0.8.0 now.

**R-23 Traceability and cross-file contradictions (00b; 02b; 06; 10; 00_master_plan).**
- 02b section 4 says "R-final RF-42 is updated" to the 150 s busy window. RF-42 still says
  "after at most `LEASE_TTL_S` (120 s)", and "kill the API container mid-job" no longer kills a
  job once jobs run in `jobs-worker`.
- 06 risk rows 301-303 still give rollback and the registry migration to "B1-1B", which B2
  deletes.
- 10 section 3 says "the pin is 0.143.3"; the pin is 0.162.0.
- The master plan W2.4 says "budget 4 h, the 16 steps"; 03 section 9 has 18 steps and 4 h 30 min.
- 02c section 7 names CLAUDE.md's command wrongly (R-4).
- 02b and 13 disagree on the kombu and redis-py range (R-10).
- *Fix.*
  - Rewrite RF-42 (kill a `jobs-worker` replica; 150 s).
  - Re-point the 06 risks to 2E.
  - Correct 10.
  - Make the master plan cite 03's budget and step count by reference instead of restating them.

**R-24 SoftTimeLimitExceeded and soft shutdown inside `asyncio.run` are not spiked (02b section 3.1; 03a 1B-1).**
- Celery raises `SoftTimeLimitExceeded` from a signal handler in the main thread, which here is
  running an asyncio event loop. Where the exception lands (inside a callback, inside a C call
  under `asyncio.to_thread`) decides whether `finally` blocks and the terminal write run.
- S1 checks requeue on cold shutdown but not this.
- *Fix.* Add S1 item 7: a handler awaiting `asyncio.sleep` and one awaiting `asyncio.to_thread`
  both end `failed (time_limit)` with the engine disposed. If either does not, the wrapper turns
  the soft limit into a cooperative cancel (a timer that sets the cancel event) and keeps
  `time_limit` as the backstop.

**R-25 The 0.6.0 `load` transaction has no stated size or bulk-load method (02c section 2.2 step 2; 02c T-14).**
- The 0.6.0 `load` is "one transaction" with every ledger row: `dataset_import_images`,
  `label_operation_members` and pending `ingest_source_files` (which can reach millions of rows).
  T-14 asks for 5,000-row batches elsewhere and stays silent here.
- A single transaction is right for atomicity, so state the expected size and time and test the
  fixture at 1,000,000 rows.
- *Fix.* Add "the upgrade `load` is the one sanctioned large transaction; nothing else runs during
  it (step precondition); COPY via psycopg `cursor.copy()` for the ledger tables"
  (https://www.psycopg.org/psycopg3/docs/basic/copy.html) instead of row inserts.

**R-26 Multi-host statements need two limits named (02 section 2 rule 8; 02c section 5).**
- The GPU arbiter is single-host by construction: Docker socket, `gpu_reservations` keyed by host.
- The schema-head equality check makes every upgrade stop-the-world: an old replica that restarts
  after `db-migrate` refuses to start, and new replicas cannot start before it.
- Both are acceptable for 0.7.0, but the "scales to N replicas on many hosts without structural
  change later" goal needs them written down.
- *Fix.* State in INSTALLATION (6D):
  - upgrades are stop-the-world in 0.7.0;
  - the arbiter overlay is single-host;
  - from v0.8.0 migrations follow expand-then-contract, so a check of "database head is at or
    ahead of the code head and marked compatible" can replace strict equality.

**R-27 The kind of database driver build for images is not in the image plan (13 section 4).**
- 13 recommends `psycopg[c]` in images, which needs a compiler in the builder stage and `libpq5` in
  every runtime stage. The plan lists that for the api image only.
- The trainer (Python 3.12, its own Dockerfile) and the export and evaluator targets also import
  `src/db`.
- *Fix.* 5A and 5B add the builder and `libpq5` steps to those images. Or use `psycopg[binary]`
  in the three GPU images, which run only short control queries, and record why.

### 2.4 Low

- **R-28** 02b, 02c and 03a link the SQLAlchemy 2.0 docs (`/en/20/`), while 13 pins SQLAlchemy
  2.1.8, which has a migration guide (https://docs.sqlalchemy.org/en/21/changelog/migration_21.html).
  Re-point the links and have 1A read the 2.1 notes first.
- **R-29** The Alembic async `env.py` template is meant for async applications. The migration runner
  is a one-shot script that holds a session advisory lock on one connection. A sync `env.py` on a
  psycopg sync connection, as in the cookbook "sharing a connection" recipe, is simpler and avoids
  `run_sync` plumbing. Not wrong as written.
- **R-30** Flower: 02c offers `--basic-auth-file`. Flower documents `--basic-auth` (argv) and the
  `FLOWER_BASIC_AUTH` environment variable, and I found no file option. Verify in S1. If there is
  none, read the file in a two-line entrypoint and keep Flower loopback-only.
- **R-31** The upgrade bundle is `.jsonl.zst`. Python 3.13 has no standard-library zstd (it arrives
  in 3.14), and no zstd package is in 13. Use gzip, or pin `zstandard` explicitly.
- **R-32** The single-beat lock connection needs TCP keepalives (`keepalives_idle`,
  `tcp_user_timeout`). Otherwise a half-open connection after a network fault keeps the server-side
  lock while beat believes it still holds it, or the reverse.
- **R-33** The producer app and `gpu_app.make_app` must carry exactly the same
  `broker_transport_options`. kombu's Redis transport restores unacked messages from any consumer
  using *that* consumer's visibility timeout, and a GPU app on the default would restore `jobs`
  messages after an hour. Extend 1B-3's config test to every app factory.
- **R-34** Event retention is 24 h. A client that resumes with an older `Last-Event-ID` silently
  misses rows. Send an explicit `reset` event when the cursor is older than the retained minimum.
- **R-35** The connection budget omits the sync heartbeat connection that each in-request inline
  job (label op, undo) opens in an API process.
- **R-36** No policy covers the next PostgreSQL major (18 to 19): `pg_upgrade` versus dump and
  restore. Add one line to INSTALLATION: majors move only with a release, through
  `db backup` / `db restore`.
- **R-37** `stream_worker_claims` uses fencing tokens, but nothing fences the stream workers'
  OpenSearch writes with them. The per-item OCC carries correctness. Say so, so nobody assumes the
  token protects writes.
- **R-38** The pool healthcheck uses `inspect stats`, a broadcast over the broker. With N replicas,
  give it `destination=[own hostname]` so health probes do not fan out to every worker.

## 3. Notes per review dimension

### 3.1 Operations and Docker/compose

- **Service definitions, `depends_on`, healthchecks.**
  - Good: TCP `pg_isready` against the first-init trap, the Valkey ping with the password out of
    argv, the beat schedule-mtime check and the pool healthcheck.
  - Findings: R-11 (inference tied to the control plane), R-38.
- **Resource limits.** R-19.
- **Restart policies.** No findings: `unless-stopped`, `db-migrate` set to `"no"`, and
  `stop_grace_period` 30 s above the soft-shutdown timeout for every consumer (6B-1).
- **Secrets.** No findings in design: files with `umask 077`, mounted as read-only directories,
  only `db-migrate` and backup get the owner password, nothing in argv. R-30 is the one open item.
- **Non-root.** No findings for the Python services (the api image user). The S0 capability record
  for Postgres is the right gate; `cap_drop: ALL` would break its entrypoint, as 6C anticipates.
- **Published ports.** No findings (only Flower on loopback).
- **Migrate service.** Right pattern: FastAPI docs and the OpenTranscribe runner, with an advisory
  lock on a dedicated session. R-7 extends it to OpenSearch.
- **Backups and restore.** `pg_dump -Fc` and the "database first, then files" order are right.
  R-13 covers the restore role ordering. The restore test is offline argv plus S0 plus R-final,
  which is enough if R-final restores into a *fresh* install.
- **Logs and metrics.** R-12. The Prometheus multiprocess handling (own directory per service,
  `mark_process_dead` in `worker_process_shutdown`) is correct per the client docs.
- **Upgrade and rollback from 0.6.0.** R-9. The 0.6.0 CLI already hands off to the newest
  installer (the v0.6.0 `setup-openprocessor.sh` bootstrap downloads and runs the latest
  release's script), so the hand-off risk in 06 is covered.
- **What an operator does differently.**
  - Back up `secrets/` and the Postgres volume.
  - Size `OP_PG_MAX_CONNECTIONS` with replicas.
  - Run `db verify` after any restore.
  - Use a shared POSIX volume for multi-host.
  - Plan stop-the-world upgrades (R-26).
  6D's list should carry all of these.

### 3.2 Scalability

- **Postgres connection math and PgBouncer.** R-6, R-35.
- **Prefetch, acks and visibility timeout versus the longest job.**
  - No defect: prefetch 1, `acks_late`, `reject_on_worker_lost`, and the 6 h timeout against the
    72 h training limit.
  - The analysis in 02b and T-11 holds: a restored duplicate waits behind the running task in a
    `-c 1` consumer, or lands on another replica, and the claim acks it.
  - Recovery after a worker kill goes through reaper rules a and b, not through the visibility
    timeout.
  - R-33 is the one gap.
- **GPU queue concurrency.** No findings. `-c 1 --max-tasks-per-child=1` releases GPU memory by
  design, and capacity comes from replicas.
- **Beat as one instance.** The session advisory lock is a sound way to make "one scheduler" a
  fact. R-32.
- **Valkey memory and persistence.** AOF every second plus `noeviction` is right, and the reaper
  rebuilds the queue (D-24). R-19 (`maxmemory`).
- **Event fan-out cost.** R-6. At today's scale the safety tick (one query per subscribed project
  per process every 2 s) is affordable. With many replicas it multiplies by every API process.
- **OpenSearch load from replicas.** R-7.
- **Storage seam for multi-host.** No findings for 0.7.0. Keys in rows, no absolute paths, and a
  ratchet make an S3 backend a later addition, not a rewrite.

### 3.3 Design correctness

- **Job claim and fencing.**
  - Correct: one conditional `UPDATE`, the fencing token in every write's `WHERE`, database-clock
    expiry, and the message sent after commit.
  - Findings: R-5 (aborted transactions; the missing transition) and R-24.
- **Partial unique index for singletons.** Correct (`NULLS NOT DISTINCT`, active set only). R-5
  covers the transaction handling around it.
- **Row-level security.** Transaction-local `set_config`, the app role without `BYPASSRLS` and the
  pooled-connection reuse test are right. R-13 covers the superuser owner, auto-begin, the
  `model_shares` policy and restore.
- **LISTEN/NOTIFY and the resume cursor.**
  - The `(xid8, id)` cursor with `xid < pg_snapshot_xmin(pg_current_snapshot())` is correct. A new
    transaction always gets an xid at or above the current `xmin`, so no row can later sort before
    the cursor.
  - Read-only snapshots (including `pg_dump`) hold no xid and do not stall the feed; only an open
    *writing* transaction does, as T-13 says.
  - The 7,000-byte payload cap (the notification limit is 8,000 bytes) and the 2 s safety tick
    handle loss while disconnected.
  - Findings: R-6 (connections and the commit lock), R-34.
- **Reaper.** R-5.
- **Cancel.** No findings. One statement, the row is the truth, and never `terminate=True`, per
  the Celery workers guide.
- **Idempotency.** Design fine. R-5 covers the aborted-transaction read.
- **The 0.6.0 data migration.** Atomic load, verify before delete and idempotent deletes are good.
  Findings: R-9 and R-25.
- **Class identity by name.** R-3. The outbox projection (D-47), with one writer path and a
  verifier, is sound.
- **Project isolation.** Two layers (repository filter plus RLS). OpenSearch guard ownership comes
  from `project_indexes`. R-13.

### 3.4 Code organization and lanes

- **Layering.** The new `src/db` layer and its contracts fit `check_layering.py`'s model: clients
  -> services -> db is allowed, and the permanent guard exception still resolves. R-2 is the
  contradiction. Also add a contract "`src/routers` does not import `src.db.repos`", so handlers go
  through services.
- **Disjoint lanes.** Spot-checked owned paths:
  - W2.1: 1-0, part A, 1A, 1B, 1C, 1E.
  - W2.2: 2A-2E, 3A, 3B (`busy.py` in 3A only, `lifespan.py` in 3B only, `flow.py` and
    `vlm_worker.py` in 3A, `fairness.py` in 2A, `runtime.py` in 2C).
  - W2.3: 4A-4D, 5A-5D (job kinds and routes split per lane; the Dockerfile only in 5B; trainer
    files only in 5A).
  - W2.4: 6A and 6B.
  - W3: part B and `frame`.
  All are file-disjoint, apart from the declared append-only files and `contracts/`. The gaps are
  interface dependencies and unowned files (R-18, R-12).
- **File size and mixed concerns.** No findings. `src/db/models/` and `src/db/repos/` are split by
  concern, the shell additions go to new `scripts/lib/` files instead of the 3,099-line installer,
  and `src/db/repos/jobs.py` is long but cohesive.
- **Dead code and shims.**
  - No findings beyond the planned transition. Between W2.2 and W2.4 `FileJob`, the leases and
    the jobs table coexist on the integration branch, but never on `main`. The done-when `rg` in
    03 section 1 enforces the deletion.
  - The 0.6.0 reader in `src/services/upgrade/` is the one sanctioned reader of deleted sources.

### 3.5 Best practice against the docs and libraries

- **SQLAlchemy.**
  - Async in the API and sync in Celery children is the right split. The SQLAlchemy asyncio docs
    and the fork guidance (`dispose(close=False)` in `worker_process_init`) are followed.
  - Per-task async engines with `NullPool` avoid the "attached to a different loop" error.
  - R-28 (2.1 docs).
- **Alembic.** Naming convention before the first revision, explicit `op.execute` for triggers and
  policies, and `alembic check` in tests: right. R-29.
- **psycopg 3.** Correct `postgresql+psycopg://` and prepared-statement handling behind PgBouncer.
  R-27.
- **Celery 5.6.** The configuration matches the documented options for soft shutdown, the Redis
  transport caveats and routing, with no result backend. R-10, R-24, R-33.
- **kombu `redis://` against Valkey.** Correct and documented (no `valkey://` scheme in kombu).
- **FastAPI DB session dependency.** R-1. FastAPI's built-in SSE (`fastapi.sse`) exists in the
  pinned FastAPI and is the right library choice.
- **Hand-rolled work that a library does.**
  - The job runtime on Postgres is custom, but the owner chose Celery. A Postgres-native task queue
    (for example procrastinate) would remove the broker but is out of scope after D-17 and D-18.
  - The single-beat lock and the cursor are small and justified.
  - `worker_max_memory_per_child` should replace any hand-rolled leak guard (R-19).
- **Comparable projects.**
  - CVAT, Label Studio, Immich and OpenTranscribe are cited accurately as far as checked.
  - OpenTranscribe's lessons are applied: the naming convention, advisory-locked migrations, the
    partial-unique active job and queue settings.
  - Immich's "database before files" order is applied.
  - The one Immich practice not taken: dump schedules on by default (D-61 keeps them off; this is
    an owner preference, not a defect).

### 3.6 Tests and verification

- **The Postgres test strategy.** Realistic with R-4 and R-14 fixed. Use testcontainers locally and
  a service container in CI, not Docker-in-Docker; that matches the GitHub Actions docs.
- **Wave exit gates.** Provable: every gate is a command, an `rg`, or a named test. The live steps
  in 03 section 9 have pass columns and budgets.
- **Red-then-green tests for the risky parts.** They exist for the claim, fencing, the races, RLS,
  the transition trigger, the cursor and the upgrade. Missing: R-5's busy-group reaper case, R-1's
  commit-failure-is-5xx case, R-3's normalized-name cases and R-24.
- **The 1-vs-N test.** R-15.
- **GPU window budget.** R-17.

### 3.7 Traceability honesty

I checked these rows of 00b against `git show origin/main:docs/design/program/<file>`:

| Original | Claimed | Result |
|---|---|---|
| old 03b 2A-1..2A-6 | 4A-1..4A-4 | holds (scores, probe and selection merged into 4A-3) |
| 2B-1 | 4B-1 | holds |
| 2B-2 | 5D | holds |
| 2B-3 | 4B-3 | holds |
| 2B-4 | 4B-2 | holds |
| 2B-5 | 4B-4 | holds |
| 2C-1..2C-6 | 4C-2..4C-7 | holds, one to one |
| 2C-7 | 4A-6 | holds |
| old 03a W1-12 | 1A-11, 1B-7, with the restored clauses | holds |
| old 03a W1-13 | 1B-5, 6B-1 | holds |
| old 05 4A-6 | 5C | holds |
| old 05 4C-0 | retired | holds |
| old 05 4C-1 | 3B | holds |
| FE-7 | 3B-4 | holds |
| D-21, D-30 | retired | holds |
| RF-42 | "updated" | **does not hold** (R-23) |

Retired items: the 16 reasons hold. Two need a caveat:
- "E15 `flock` on non-local volumes" holds only once the file-lock baseline is actually empty
  (6A-3).
- "same `visibility_timeout` in both places" is right about the result backend, but R-33 adds a
  different "two places" (the app factories).

Open issues: `gh issue list --state open` returns 44, and all 44 are placed in 00b section 2. Six
are placed only conditionally (R-22).

### 3.8 Package pins

- **torch 2.15.0.**
  - On PyPI, uploaded 2026-10-28, `requires_python >=3.11`; the default Linux wheel depends on
    `nvidia-*-cu13` packages.
  - The API image still carries CUDA 12 libraries (`faiss-gpu-cu12`, `cupy-cuda12x`), so the 13
    section 8 risk 1 note is right and needs the G11 provider check.
  - The host driver floor does not change: the Triton 26.x base already needs a CUDA 13 driver.
- **OpenCV 5.0.0.93.** The dev venv already runs OpenCV 5.0.0 and numpy 2.5.2 offline. Add
  `cv2.dnn.blobFromImage` (`src/utils/scrfd_decode.py:289`) to the parity set (R-17).
- **onnx / onnxruntime / modelopt / TensorRT.** The holds are justified by modelopt 0.48.0's
  declared ranges. TensorRT moves only with the Triton base. Fine.
- **numpy 2.5 and pandas 3.1.** Scheduling them with the GPU gate (part B, commit 6) is right.
  pandas 3's copy-on-write and string dtype need the trainer smoke, which is planned.
- **huggingface-hub 2.4.** Reasonable, with a named fallback (1.33.0) and the segmenter start-up
  test.
- **opensearch-py 3.2.** Low risk: the dev venv already runs it, and the floor `>=2.3.0` already
  resolves to 3.x in fresh builds.
- **kombu 5.7 with Celery 5.6.3.** R-10.
- **Scheduling.** Part B with the B3 gate is sensible. Part A inside B2 is not (R-16).
- **Freshness.** About 15 pins are under a week old (13 finding 5). The "re-run the survey on the
  day, step back one patch" rule is right. Apply it to kombu by holding the minor (R-10).

## 4. Questions that need the owner

| # | Question in plain words | Recommendation |
|---|---|---|
| Q-1 | Should the inference routes (`/detect`, `/embed`, `/faces`, `/ocr`) keep starting and serving when PostgreSQL or Valkey is down, with only curation and job routes answering 503? (R-11) | Yes. The inference engine stays standalone; workers and beat keep the hard dependency |
| Q-2 | Should the browser event wake-up travel over Valkey pub/sub (Postgres stays the durable event table and cursor) instead of one Postgres `LISTEN` connection per API process? (R-6) | Yes. It removes the connection term no pooler can carry and the `NOTIFY` commit lock, and changes no wire format |
| Q-3 | Hold kombu at 5.6.2 (and redis-py at 6.4.x) until a Celery release declares support for kombu 5.7? (R-10) | Yes. It is the one pin where "newest" means "untested by its own framework" |
| Q-4 | Land pins part A as its own small PR before W2.1 instead of inside the B2 PR? (R-16) | Yes. A dependency regression then never blocks or muddies the control-plane PR |
| Q-5 | Answer MQ-6 and MQ-7 (#465, #468-#472) before W2.1 rather than before W2.2 and W4? (R-22) | Yes. #472 is an isolation item, and its lane is best known before the projects code is rewritten |

## 5. Planner responses

Written by the planner on 2026-11-04, after re-checking the key findings against the code and the
official docs. Every finding is **fixed** in the plan text (where named below). None is rejected.
Two are fixed with a different mechanism than the one suggested, with the evidence. The owner's
answers to section 4 (all "yes", 2026-11-04) are recorded in [00a_decision_register.md](00a_decision_register.md) section 6
as D-64 (Q-1), D-65 (Q-2), D-66 (Q-3) and D-67 (Q-4); Q-5 was answered with MQ-6 and MQ-7.

**Verified first.**
- R-1: `inspect.signature(fastapi.Depends)` in the repository venv (FastAPI 0.143.3) has
  `scope: Literal['function', 'request']`, documented as "`function`: ... end the dependency after
  the path operation function ends, but **before** the response is sent back to the client"; the pin
  is 0.162.0 (`requirements.txt:2`). FastAPI docs:
  https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/#early-exit-and-scope.
- R-3: `src/utils/class_names.py::normalize_class_name` lowercases and collapses every run outside
  `[a-z0-9]` to one `_`; `resolve_class_by_name` lets an active class win over a deprecated one with
  the same name; `src/clients/curation_opensearch/registry.py::_add` refuses a duplicate only against
  an active class. The finding holds.
- R-4: `pyproject.toml` `addopts` uses `-m "not live and not families"`; so do
  `.github/workflows/ci.yml:69` and `:102`, `CONTRIBUTING.md:39` and `:42`, and `CLAUDE.md:32` and
  `:226`. The `Makefile` `test` target relies on `addopts`.

| Finding | Response | Where |
|---|---|---|
| R-1 | fixed: `Depends(get_db, scope="function")`, explicit commits in `project_tx`, no session on SSE routes, red tests "commit failure is a 5xx" and "SSE holds no pooled connection" | [02b](02b_jobs_locking_events.md) section 8; [03a](03a_stage1_foundation.md) 1A-10; [03c](03c_stage3_job_runtime.md) 3B-4 |
| R-2 | fixed: the heartbeat, fenced writes, event row and Valkey wake-up live in `src/tasks/runtime/` and `src/db/repos/`; the OpenSearch factory is injected by the handler module; a planted-import red test | [02](02_control_plane_database.md) section 6; 02b section 3.1; 03e 3A, 3B; 03a 1A-11; [03d](03d_stage4_job_families.md) recipe step 3; [03e](03e_stage5_gpu_queues.md) |
| R-3 | fixed: `name_key` with the partial unique index over active classes; the `ensure_class` rule of today; fixture and test cases | [02a](02a_postgres_schema.md) section 3; [03b](03b_stage2_control_data.md) 2B-1; [02c](02c_migration_ops_tests.md) section 2.4 |
| R-4 | fixed: lane 1-0 changes `addopts`; 1D changes both `ci.yml` commands, PR #477's shard command, `CONTRIBUTING.md` and the `Makefile`; 6D changes `CLAUDE.md`; 1D-3 asserts the default selection | 03a 1-0, 1D; 02c section 7 |
| R-5 | fixed: savepoints in `start_job` and per reaper row, the busy-group exclusion, `interrupted -> failed` in the transition table, the red test | 02b sections 1, 2.2, 4; 03e 3A-3, 3A-7 |
| R-6 | fixed with option (a), owner decision D-65: Valkey pub/sub wake-up, no `LISTEN`, no `NOTIFY` commit lock; the API pool is a deployment total over workers and replicas; the budget test takes scale factors; PgBouncer by formula | 02b sections 6, 8, 8.1; 03e 3B-1, 3B-3, 3B-5; 03a 1D-6; [05](05_b4_stateless_api.md) 4B-2 |
| R-7 | fixed, with one difference: index creation and mapping migrations move into the one-shot `db-migrate` (OpenSearch step) and the API checks fingerprints read-only. The kNN warm-up also moves there, once per deployment, rather than "one process per replica": the warm-up loads graphs on the OpenSearch side, so one call per deployment warms them for every API process; 5a's cold-query numbers confirm it | 02c section 1.2; 03d 2F-4 |
| R-8 | fixed: liveness rows, release of the excess, the token bumps only on takeover, sweep step 13 starts the replicas one after the other | 02b section 5; 03e 3A-10; [03](03_b2_overview.md) section 9 step 13 |
| R-9 | fixed (a-e): the `upgrade_state` gate, `--reset`, rollback abandons the Postgres data, the rollback order, run logs into storage keys; RF-63 reworded | 02c sections 2.2, 2.3, 2.4; 03d 2E-2, 2E-3, 2E-4, 2E-7; [07](07_release_acceptance.md) RF-63 |
| R-10 | fixed, owner decision D-66: `kombu==5.6.2`, `redis==6.4.0`; the 02b sentence corrected; S1 checks `pip check` | [01](01_package_pins.md) sections 1, 4; 02b section 7.1; 03a 1-0, 1B-1; 02c T-17 |
| R-11 | fixed, owner decision D-64: `api` depends on neither `db-migrate` nor `valkey`; "control plane down" mode; refuse only on a reachable database at the wrong head; `/health` and `/ready` name the dependency; "every Python service except `api`" | 02 rule 7; 02c section 1.2; 03a 1A-9, 1C-2, 1D-1; 03 section 9 steps 8-9, R2-13 |
| R-12 | fixed with a different discovery mechanism: Prometheus `dns_sd_configs` (type A) on each service name instead of `docker_sd_configs`. Docker's embedded DNS answers a Compose service name with the address of every replica, and Prometheus' DNS discovery makes one target per record ([dns_sd_config](https://prometheus.io/docs/prometheus/latest/configuration/configuration/#dns_sd_config)), so each replica is its own `instance` without mounting the Docker socket into Prometheus. Alloy keeps by the `service` label; dashboards select by `service`; 6B owns Alloy and the dashboards; a render test | [03f](03f_stage6_scheduler_scaleout.md) 6B owned files, 6B-6 |
| R-13 | fixed: `op_super` for bootstrap only, a non-superuser `op_owner` with `BYPASSRLS`, the `after_begin` scope listener, `op_scope_allows` raises when unset, `FOR SELECT` consumer policies on `model_shares` and `models`, restore with `--no-privileges` | 02a sections 1.4, 1.5, 7; 02c sections 4, 6; 03a 1A-6, 1A-7, 1A-8, 1C-4 |
| R-14 | fixed: one container shared by the xdist workers, a template per worker, fakes only where no SQL behaviour is checked, each stage lists the fake-based modules it rewrites | 02c section 7; 03a 1A-12; 03d 2A, 2B; 03b recipe step 6 |
| R-15 | fixed: a sequential scenario with an exact diff and a concurrent one judged by invariants, in the three configurations of 05 | 05 section 7 and done-when; 02c section 7 |
| R-16 | fixed, owner decision D-67: pins part A is wave W2.0, its own PR before W2.1 | 13 section 7.1; [00_master_plan.md](00_master_plan.md) W2.0; 03a header |
| R-17 | fixed: the CPU parts of G11 move into the W3 sweep, a written cap of plus one day, fallback pins first, `cv2.dnn.blobFromImage` in the parity set | [04](04_b3_ultralytics_one_version.md) sections 6 and 7; master plan section 5 |
| R-18 | fixed: the worker-start schema check and the budget test move to 1D; 3A is cut after 2A merges; 2F-5 switches 2E's load to the repositories; 2B owns the three registry scripts; 04 section 4 corrected and every B3 track re-verifies on the post-B2 head | 03a header, 1B-3, 1D-5, 1D-6; 03e header; 03d 2B, 2F-5; 04 section 4 |
| R-19 | fixed: Valkey `maxmemory` with `noeviction` and an alert, memory limits on the new services, `--max-memory-per-child` on `jobs-worker`, a render test | 02b sections 7.1, 7.2; 02c section 4; 03a 1C-2, 1C-5; 03c 6B-6 |
| R-20 | fixed: a dedicated `arbiter` queue and a one-process `gpu-arbiter-worker` that alone mounts the socket | 02b sections 2.1, 7.3; 02c section 4; 03a 1B-2; 03c 6A-1, 6B-4; 03 section 5 |
| R-21 | fixed: `project_create` is a `jobs` kind (deployment-level, exclusive per slug, auto-resume) in `job_kinds/b4.py` and the kind table | 05 section 6; 02b section 2.1 |
| R-22 | fixed, owner decision of 2026-11-04 (MQ-6, MQ-7): the W4 privacy lane and the W5a-W5c typed-response lanes; #469 and #471 listed as open owner items OI-1, OI-2 | master plan W4, W5a-W5c, section 9; [00b](00b_traceability.md) section 2 |
| R-23 | fixed: RF-42 rewritten, the 06 risk rows re-pointed to 2E, 10's FastAPI pin corrected, the master plan cites 03's budget by reference, 02c's command and the kombu range corrected | 06; [10](10_standards_and_libraries.md) section 3; master plan W2.4; 02b section 4; 00b section 1.8 |
| R-24 | fixed: spike S1 item 7, and the wrapper's cooperative soft-limit fallback | 02b section 3.1; 03a 1B-1 |
| R-25 | fixed: the load is the one sanctioned large transaction, `COPY` for the ledgers, a 1,000,000-row test | 02c sections 2.2, 2.4; 03d 2E-3 |
| R-26 | fixed: INSTALLATION states stop-the-world upgrades, the single-host arbiter, expand-then-contract from v0.8.0 | 02c section 1.2; 03c 6D |
| R-27 | fixed with the first option: `psycopg[c,pool]` and `libpq5` in the trainer, export and evaluator images too (one build kind in every image) | 03f header; 13 section 4 |
| R-28 | fixed: links moved to the SQLAlchemy 2.1 docs; 1A reads the 2.1 migration notes first | 02b section 8; 03a 1A-6; 02a, 02c, 10 |
| R-29 | fixed: the sync `env.py` on the runner's shared connection | 02c section 1.1; 03a 1A-3 |
| R-30 | fixed: Flower reads the auth file from its `--conf` Python file, never argv or the environment; S1 checks it | 02c section 4; 03a 1B-1 |
| R-31 | fixed: the bundle is gzip | 02c section 2.2; 03d 2E-2 |
| R-32 | fixed: TCP keepalives and `tcp_user_timeout` on the beat lock connection | 02b section 4.1; 03c 6A-2 |
| R-33 | fixed: one `broker_transport_options()` for every app factory, with a test | 02b section 7.1; 03a 1B-2, 1B-3; 03f |
| R-34 | fixed: a `reset` event for a cursor older than retention | 02b section 6.3; 03e 3B-4; 03 step 11 |
| R-35 | fixed: the inline-job heartbeat connection is a row of the budget table | 02b section 8 |
| R-36 | fixed: PostgreSQL majors move only with a release, through `db backup` and `db restore` | 03c 6D |
| R-37 | fixed: the stream-claim token does not fence OpenSearch writes; per-item OCC does, said explicitly | 02b section 5 |
| R-38 | fixed: the pool healthcheck sends `inspect stats` to its own hostname only | 03a 1B-6 |
| section 3.4 note (`src/routers` imports no `src.db.repos`) | fixed: a layering contract and a planted red test | 02 section 6; 03a 1A-11 |
| section 3.7 caveats (E15, the visibility-timeout sub-item) | fixed: both caveats added to the retired rows | 00b section 1.9 |

### 5.1 Responses to the re-check (section 6) and to the owner correction of 2026-11-04

The owner corrected decision D-64 the same day ("the system only works if everything is up, seems
silly to run the app with no backend databases"). D-69 replaces it: the API and every worker require
PostgreSQL, Valkey and OpenSearch (plus Triton for inference), wait with bounded retries, then
refuse to start naming the missing dependency; `/live` and `/ready` keep the liveness and readiness
split; there is no degraded mode. The plan text, tests and tasks that existed only for the degraded
mode are deleted: the `api` exemption from `db-migrate` and `valkey`, `test_control_plane_down.py`,
the `/detect`-during-outage assertions of live steps 8 and 9, and risk R2-13.

| Finding | Response | Where |
|---|---|---|
| RC-1 | fixed, re-evaluated under D-69: with every service waiting on `db-migrate`, the OpenSearch step stays in `db-migrate` after Alembic (no separate service). It reads the projects from Postgres, does only the global indexes while the 0.6.0 data is not loaded (`upgrade_state` `in_progress` or `op_projects` present), and the installer runs `db-migrate` again after the control-data step, so the upgraded projects are stamped before `up`. Each index's `_meta` carries the existing `SCHEMA_META_KEY` fingerprint (`index_bootstrap.py`); the lifespan reads all stamps in one `_mapping` call and refuses on any mismatch. `project_create` stamps its own indexes | 02c section 1.2; 03d 2E-5, 2F-4; 05 section 6; 00_index D-69 |
| RC-2 | fixed: `src.tasks.runtime` is a leaf layer that `src/services`, `src/tasks`, `docker/trainer`, `export` and `scripts/curation/bakeoff` may import; any other `src.tasks` module stays private to `src/tasks`; 1A-11 plants both directions | 02 section 6; 03a 1A-11 |
| RC-3 | fixed: beat schedules the `gpu_arbiter` tick only when `OP_GPU_ARBITER_CONTAINERS` is set (the overlay sets it; checked in `src/config/gpu_arbiter.py`, the only arbiter settings that exist); the consumers test treats overlay queues as conditional; the rule "no periodic message to a queue without a consumer" is written down | 02b sections 2.1, 7.3; 03c 6A-1 |
| RC-4 | fixed: `deploy.replicas: ${OP_API_REPLICAS:-1}` is the one source of the replica count (no `openprocessor scale`); `pool_timeout` 5 s maps to 503 `database_busy`; the two polls get their own one-connection engine, counted in the budget (total 197); the budget test also fails when workers times replicas exceed the total; 4B-5 records pool wait | 02b section 8; 02c section 4; 03a 1A-6, 1C-2, 1D-6; 05 4B-2, 4B-5 |
| RC-5 | fixed: two modules, `b4_fanout` (track 4A) and `b4_project` (track 4D), in `job_kinds`, `task_routes` and `src/tasks`, all created empty by 1B-2, so W4 ownership is disjoint | 03a 1B-2; 05 section 6 and the 4A brief (section 17); 00b section 1 |
| RC-6 | fixed: the socket goes into `gpu-arbiter-worker` only in all three places | 03 section 9 step 17; 08 stage-6 row; 03c 6D |
| RC-7 | accepted, documented: the first query after an OpenSearch restart pays the graph load; INSTALLATION says so and 5a measures the cold query. No re-warm task (it would need restart detection for a one-time cost) | 02c section 1.2; 03c 6D |
| RC-8 | fixed: dashboards select and aggregate by `service` or `job`, never `container` or `instance`; Prometheus joins the stack's Compose network | 03c 6B-6 |
| RC-9 | fixed: the policy is `(SELECT op_scope_is_deployment()) OR project_id = (SELECT op_scope_project())`; each scalar subquery is an InitPlan evaluated once per statement, so `project_id` indexes apply; `op_scope_project()` keeps the `scope_not_set` raise; `op_scope_allows` remains a SQL wrapper for the EXISTS policies; an `EXPLAIN` test checks the plan shape | 02a section 1.4 |
| RC-10 | fixed: the W2.1 lanes are cut after the W2.0 PR merges; the "may code from `main`" line is gone | 00_master_plan W2.0 |
| RC-11 | fixed: K-20 names one Valkey pub/sub subscription per API process (D-65) | 05 section 14 |
| RC-12 | fixed: 2B-1 tests that restoring a deprecated name with an active twin answers 409 `class_name_taken` through a savepoint | 03d 2B-1 |
| RC-13 | fixed in docs: INSTALLATION states that messages for a queue whose profile never started stay in Valkey after the reaper fails the job | 02b section 7.3; 03c 6D |
| owner answers (OI-1, OI-2, MQ-3, MQ-4, MQ-8, Q-3) | recorded and applied: #469 and #471 close as "decided: keep current behaviour" with no code change and leave the privacy lane, which gains a SECURITY.md statement (the CURATION.md sentence goes to track 4D, which owns that file in W4); the 4-shard offline suite at every B2 part check; Pillow, simplejpeg, the Hugging Face stack and MLflow in pins part B; GPU-15 leftover in B3-D3; Flower on loopback 4613 and the gateway subpath under `monitoring` | 00_index section 6; 00_master_plan sections 6, 9; 00b sections 2, 4; 03 section 4; 04 B3-D3; 08 ENV-18, Q-3; 13 section 7.1 |

## 6. Re-check (2026-11-04)

Written by the independent reviewer on `plan/recheck` (from `plan/master` at `88b09f2df`), after the
planner's fix pass (`357594495`, `88b09f2df`). Same rules as sections 1-3: every claim was checked
against the plan text, the code and the docs, not against the response table in section 5.

### 6.1 Verdict

**Yes, with fixes.** Wave W2.0 can start now.

All twelve blockers and highs (R-1 to R-12) are resolved in the plan text, not just mentioned. The
fix pass did introduce three new high findings. Each is a contradiction between two fixes, or
between a fix and an existing rule. Each is a short text change. Make the changes before the lanes
they touch are cut: RC-2 before W2.1 (lane 1A-11 writes the contract), RC-1 before W2.2 (lane 2F),
RC-3 before W2.4 (lane 6A).

Counts of the new findings: **0 blockers, 3 high, 3 medium, 7 low.**

### 6.2 Blockers and highs: resolved?

| Finding | Resolved | Checked where |
|---|---|---|
| R-1 | yes. `Depends(get_db, scope="function")`, explicit commits in `project_tx`, SSE routes take no session, and the two red tests | 03a 1A-10; 02b sections 6.3 and 8 |
| R-2 | yes for the GPU images (heartbeat, fenced writes, event row and wake-up live in `src/tasks/runtime/` and `src/db/repos/`; the OpenSearch factory is injected). **The fix creates RC-2** | 02 section 6; 03e 3A and 3B owned files |
| R-3 | yes. `name_key`, the partial unique index over active classes, and the lookup rule matching `resolve_class_by_name` | 02a section 3, against `src/utils/class_names.py:157-214` |
| R-4 | yes. `addopts` in lane 1-0; `ci.yml` (both commands and the PR #477 shard command), `CONTRIBUTING.md` and the `Makefile` in 1D; `CLAUDE.md` in 6D; 1D-3 asserts the default selection | 03a lanes 1-0 and 1D |
| R-5 | yes. Savepoints in `start_job` and per reaper row, the busy-group `NOT EXISTS`, and `interrupted -> failed` in the transition table | 02b sections 1, 2.2 and 4 |
| R-6 | yes. Valkey pub/sub wake-up (D-65), so no `LISTEN` and no `NOTIFY` lock; the pool is a total over workers and replicas; the budget test takes scale factors. **The implementation leaves RC-4** | 02b sections 6 and 8; 03e 3B |
| R-7 | yes as specified: bootstrap and mapping migrations in the one-shot step, read-only check in the API. **It contradicts R-11: RC-1** | 02c section 1.2; 03d 2F-4 |
| R-8 | yes. Liveness rows, release of the excess, a token bump on takeover only, and sequential start in sweep step 13 | 02b section 5 |
| R-9 | yes (a-e). The `upgrade_state` gate, `--reset`, rollback order and the abandoned database, run logs moved into storage, RF-63 reworded | 02c sections 2.2 and 2.3 |
| R-10 | yes. `kombu==5.6.2` and `redis==6.4.0` (on PyPI; 6.4.0 is the only 6.4.x release, uploaded 2025-08-07, not yanked); the 02b sentence corrected | 13 sections 1 and 4; 02b section 7.1 |
| R-11 | yes. `api` depends on neither `db-migrate` nor `valkey`; "control plane down" mode; refuses to start only at the wrong head; `/health` and `/ready`. **See RC-1** | 02c section 1.2; 03a 1D-1; D-64 |
| R-12 | yes, with the alternative mechanism (6.3) | 03c 6B |

### 6.3 The two alternative mechanisms

- **R-7, kNN warm-up once per deployment in the migrate step: accepted.**
  - The OpenSearch k-NN warm-up loads the native graphs into the data nodes' memory on the server
    side, so one call warms them for every client. The repo's own docstring says the same
    (`src/services/curation/index_bootstrap.py:93-98`). One call per deployment is enough, and it
    is better than one per replica.
  - Caveat (RC-7, low): the warm state is lost when OpenSearch restarts, when the graph cache
    expires or evicts, and for segments written after the warm-up. The migrate step does not run
    again in those cases, so the first query after them pays the load.
- **R-12, Prometheus `dns_sd_configs` (type A) instead of `docker_sd_configs`: accepted. It is
  better than my suggestion.**
  - Docker's embedded DNS answers a Compose service name with one A record per running replica.
  - Prometheus' DNS discovery makes one target per record
    ([dns_sd_config](https://prometheus.io/docs/prometheus/latest/configuration/configuration/#dns_sd_config)).
  - It needs no Docker socket in Prometheus.
  - Caveat (RC-8, low): `instance` is the container IP, which changes when a container is
    recreated. Dashboards should aggregate by `job` or service, not by `instance`. Prometheus must
    sit on the same Compose network as the targets.

### 6.4 Mediums spot-checked

| Finding | Resolved | Note |
|---|---|---|
| R-13 | yes: non-superuser `op_owner` with `BYPASSRLS`, the `after_begin` scope listener, `scope_not_set` raised, `FOR SELECT` consumer policies, restore with `--no-privileges` | see RC-9 |
| R-14 | yes: one container shared by the xdist workers, `op_template_<worker>` | - |
| R-15 | yes: a sequential exact-diff scenario plus a concurrent invariant scenario, in the three configurations of 05 | - |
| R-19 | yes: Valkey `--maxmemory` with `noeviction`, a container limit above it, memory limits on the new services, `--max-memory-per-child` | - |
| R-20 | yes: a dedicated `arbiter` queue and `gpu-arbiter-worker` | **creates RC-3 and RC-6** |

### 6.5 New findings

#### High

**RC-1 The OpenSearch migrate step contradicts "inference stands alone" and misses the upgraded projects (02c section 1.2; 03d 2F-4; D-64).**
- *What.* Five gaps:
  1. Index creation for the **global** inference indexes (`/search`, `/ingest`) now happens only in
     `db-migrate`. That service waits on `postgres`, connects as `op_super` and `op_owner`, and
     fails when Postgres is down.
  2. The API refuses to start on any mapping-fingerprint mismatch. So on a fresh install with
     Postgres down, the inference API never starts, or starts with no visual-search indexes. D-64
     says the opposite.
  3. Because `api` no longer waits on `db-migrate` (R-11), it can start before the step has run.
     It then refuses (`opensearch_mapping_not_current`) and loops on its restart policy.
  4. On a 0.6.0 upgrade, the installer order is `db-migrate`, then the control-data step. The
     OpenSearch step therefore runs while Postgres holds no projects. The 0.6.0 `images` and
     `items` indexes are never migrated or stamped, and the API refuses after `up -d`.
  5. Where the fingerprint is stored is unspecified. Projects created later by the
     `project_create` job (B4 4D) are not covered by a check that runs once at start.
- *Evidence.* 02c section 1.2 (the R-7 bullet and the "who waits on whom" bullet); 02c section 1.2
  `openprocessor upgrade` order; `src/core/dependencies.py:140-152` (today's bootstrap also creates
  the global indexes); 00_index D-64.
- *Fix.*
  - Make the OpenSearch step independent of Postgres. Either run it as its own one-shot service
    that `api` *does* wait on, since it needs only OpenSearch, or let `db-migrate` run the
    OpenSearch part even when the Postgres part fails.
  - Store each fingerprint in the index mapping's `_meta`. That is the existing schema stamp the
    bootstrap already reads ("Indexes already stamped with the current schema cost one read"), so
    the check needs no Postgres.
  - Refuse to start only on a **global** index mismatch. A project index mismatch makes that
    project's curation routes answer 503 (fail closed per project), checked when the project is
    first used.
  - The installer runs the OpenSearch step again after the 0.6.0 control-data step.
  - `project_create` stamps its two indexes.

**RC-2 The layering contract forbids the imports the R-2 fix requires (02 section 6; 03a 1A-11; 03e 3B-1; 03f 5C).**
- *What.* 02 section 6 says "nothing under `src/` except `src/tasks/` imports `src.tasks`", and
  1A-11 keeps "nothing under `scripts/` imports `src.tasks`". But after the R-2 fix:
  - `src/services/jobs/` imports the heartbeat class from `src/tasks/runtime/` for inline jobs (02
    section 6 says so in the same table);
  - `src/services/events/publish.py` calls `src/tasks/runtime/event_wakeup.py::wake` after commit
    (03e 3B-1);
  - the evaluator's handler `scripts/curation/bakeoff/celery_tasks.py` (5C) must call
    `register_job` from `src/tasks/runtime/register.py`. This last one predates the fix; I missed
    it in the first review.
  Lane 1A-11 writes the contract in W2.1, so lanes 3A, 3B and 5C fail its pre-commit hook in W2.2
  and W2.3.
- *Evidence.* 02 section 6 (the `src/services/jobs/` row and the contract paragraph); 03a 1A-11;
  03e 3B-1; 03f lane 5C.
- *Fix.* Declare `src.tasks.runtime` a leaf layer in `check_layering.py`. It may be imported by
  `src/services/`, `src/tasks/` and the GPU handler modules (`docker/trainer/`, `export/`,
  `scripts/curation/bakeoff/`). Every other import of `src.tasks` stays forbidden. Add one planted
  test per allowed and forbidden direction to 1A-11.

**RC-3 The arbiter tick has no consumer on a default install and fills Valkey (02b sections 2.1, 7.3; 03c 6A-1; 02c section 4).**
- *What.* Five problems:
  1. `op.periodic.gpu_arbiter` is scheduled by beat every 15 s.
  2. It now routes to the `arbiter` queue, whose only consumer exists with the opt-in GPU-arbiter
     overlay.
  3. Beat's `expires` option is checked by a worker when it *receives* the message. Undelivered
     messages stay in the Valkey list, about 5,760 a day, forever.
  4. With `maxmemory` and `noeviction` (R-19), Valkey eventually refuses every write. Every job
     start then answers 503: an outage on the default install after weeks or months.
  5. `tests/test_queue_consumers_from_compose.py` requires a consumer "in core or in the profile
     `QUEUE_PROFILE` names", and the overlay is neither, so the test also fails.
- *Evidence.* 03c 6A-1 (the `op.periodic.gpu_arbiter` row); 02b section 7.3 (the `arbiter` row,
  "only with the GPU-arbiter overlay"); Celery periodic tasks, `expires`
  (https://docs.celeryq.dev/en/stable/userguide/calling.html#expiration).
- *Fix.*
  - Beat adds the `gpu_arbiter` entry only when the overlay's setting is present (read at beat
    start; the overlay sets it).
  - The consumers test treats overlay queues as conditional on that setting.
  - Note in 02b section 7.3 that a queue without a live consumer must never receive periodic
    messages.

#### Medium

**RC-4 The R-6 pool works only as a configuration, not as a guarantee (02b section 8; 05 4B-2).**
- *What.*
  - Per API process the pool is now 2 connections (64 / 32) with no overflow. Request handlers,
    the 1 s registry poll, the 5 s config poll and the SSE catch-up queries all share those 2.
    Under load, requests wait for SQLAlchemy's default `pool_timeout` (30 s) instead of failing
    fast.
  - The `max(1, ...)` floor still exceeds the total once workers x replicas > total.
  - `OP_API_REPLICAS` is declared separately from `docker compose --scale`. A user who scales with
    compose directly leaves it stale, and the budget is silently exceeded.
  - `openprocessor scale` (02b section 8) has no owner lane.
- *Fix.*
  - Make `deploy.replicas: ${OP_API_REPLICAS:-1}` on `api` the single source, and have the budget
    test read it.
  - Set a short `pool_timeout` that maps to 503 `database_busy`.
  - Run the polls on a separate one-connection engine, counted in the budget.
  - Have the 4B-5 benchmark record pool wait time.
  - Give `openprocessor scale` to 6B, or drop it in favour of the `.env` value.

**RC-5 Two W4 lanes now share the B4 kind modules (05 sections 3, 4A-3, 6; 03a 1B-2).**
- *What.* 4A (the VLM batch kinds, 05 line 441) and 4D (`project_create`, added by R-21) are
  parallel tracks. Both register kinds in `src/config/job_kinds/b4.py`, and so `task_routes/b4.py`
  and `src/tasks/b4.py` too.
- *Fix.* 1B creates `b4_fanout.py` and `b4_project.py` (each in the three registries), one per
  track, or 4D is cut after 4A merges.

**RC-6 Three files still put the Docker socket into `periodic-worker` (contradicts the R-20 fix).**
- *What.* 03 section 9 step 17 ("the arbiter overlay on `periodic-worker`"), 08 section 4 (the
  stage-6 row) and 03c 6D's CHANGELOG "Breaking" text and INSTALLATION note all say
  `periodic-worker`. Lane 6D would publish the wrong upgrade note.
- *Fix.* Change all three to `gpu-arbiter-worker`.

#### Low

- **RC-7** The kNN warm-up is lost on an OpenSearch restart or cache eviction (6.3). Accept, or add
  a cheap beat task that re-warms after OpenSearch restarts.
- **RC-8** The `dns_sd` `instance` label churns when containers are recreated (6.3). Dashboards
  aggregate by service.
- **RC-9** `op_scope_allows` is now plpgsql, which the planner cannot inline. It runs once per
  row, and the policy can never use an index. Repository queries keep their explicit
  `project_id = :p`, so this is cost, not correctness. Optional: express the policy as
  `project_id = (SELECT op_scope_project())`, a scalar subquery evaluated once per statement as an
  InitPlan, with the raising function inside.
- **RC-10** The master plan W2.0 says the W2.1 lanes "may code from `main`" in parallel, while W2.1's
  precondition says W2.0 is merged before `int/b2-control-plane` is cut. Pick one. Lane 0 edits the
  same manifests as W2.0.
- **RC-11** 05 K-20 still says "one `LISTEN` connection per API process" (D-65 replaced it).
- **RC-12** Restoring a deprecated class whose `name_key` now matches an active class hits
  `uq_project_classes_active_name`. It must answer 409 through a savepoint. Add the case to 2B-1's
  tests.
- **RC-13** Messages for jobs whose queue has no consumer (a profile never started) stay in Valkey
  after the reaper fails the row (`queue_wait_exceeded`). The volume is small, but INSTALLATION
  should say so. RC-3 is the unbounded case of the same rule.

### 6.6 Cross-file consistency checked

- The master plan's waves match the batch files for:
  - W2.0 (pins part A as its own PR, D-67);
  - W2.1 (merge order 0, 1A, 1E, 1B, 1C, 1D; 1D owns `src/tasks/app.py`'s start check and the
    budget test after 1B and 1A merge);
  - W2.2 (3A cut after 2A; merge 3B then 3A; 2F does not edit `lifespan.py`, which 3B owns).
- The W2.4 sweep is cited by reference (R-23 holds).
- File-disjointness holds in W2.1, W2.2 and W2.3 after the ownership moves:
  - W2.1: `schema_check.py` (1A in W2.1, 2E in W2.2); `src/tasks/app.py` (1B, then 1D);
  - W2.2: `src/core/dependencies.py` and `scripts/db/migrate.py` (2F after 2A-2E merge).
  The exception is W4 (RC-5).
- No remaining text gives redis-py `<8.0` for kombu 5.7. 13 and 02b agree.
- No remaining `LISTEN` dependency in the B2 files: the direct host is kept only for the beat and
  migration locks. The one stale mention is 05 K-20 (RC-11).
