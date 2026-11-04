# 02a: the PostgreSQL schema

| Field | Value |
|---|---|
| Status | plan, not started |
| Part of | [02_control_plane_database.md](02_control_plane_database.md) (read its sections 2 and 3 first) |
| Built by | B2 stage 1 (all tables, one initial Alembic revision), used by stages 2-6 ([03_b2_overview.md](03_b2_overview.md)) |
| Server | PostgreSQL 18 (latest stable major on 2026-11-04; 19 is a release candidate). The schema needs at least 15 (`NULLS NOT DISTINCT`) and 18 (`uuidv7()`); the exact image tag is in `13_package_pins.md` (written on its own branch) |

The DDL below is the contract. Stage 1 writes it as SQLAlchemy models in `src/db/models/` and one
Alembic revision `0001_control_plane`, drafted with autogenerate and corrected by hand against this
file (Alembic does not detect everything:
[what autogenerate detects](https://alembic.sqlalchemy.org/en/latest/autogenerate.html)). A test
(`tests/db/test_schema_matches_plan.py`) introspects the migrated database and asserts every table,
unique constraint, partial index and row-level-security policy named here.

---

## 1. Conventions

### 1.1 Names, keys, types

- Tables plural snake_case in schema `public`. Constraint names from one naming convention on the
  `MetaData` (`ix_`, `uq_`, `ck_`, `fk_`, `pk_` as in
  [the Alembic naming guide](https://alembic.sqlalchemy.org/en/latest/naming.html#the-importance-of-naming-constraints)),
  set before the first revision because it cannot be added later without renaming every constraint
  (the OpenTranscribe lesson in [02](02_control_plane_database.md) section 8).
- Entity primary keys `uuid DEFAULT uuidv7()` (time-ordered, so B-tree inserts stay local;
  [UUID functions](https://www.postgresql.org/docs/18/functions-uuid.html)). A job's id is also its
  Celery `task_id`.
- Times `timestamptz NOT NULL DEFAULT now()`; every comparison against an expiry uses `now()` in SQL.
- Status columns, and kind columns whose set is closed, are `text` with a `CHECK (... IN (...))`
  (`jobs.kind` is the exception: it is validated against the kind table in code, so adding a job kind
  needs no migration), not `CREATE TYPE ... AS ENUM`:
  adding a value is then an ordinary migration, and Alembic does not track enum value changes.
  Every such list has one Python source (`src/domain/job_states.py`, ...)
  and a test asserts the `CHECK` equals it.
- `jsonb` only for document-shaped bodies that are validated by a pydantic model on write and on
  read (D-59). Identity, ownership, revision, status and every value a constraint needs are columns.
- No absolute file paths: blob references are `storage_key text` relative to a storage root
  ([02c](02c_migration_ops_tests.md) section 5).
- Creation order inside `0001_control_plane`: `projects` and its satellites, then `jobs`, then
  every table that references `jobs` (`search_projection_tasks`, `projection_states`, the dataset,
  training and model tables).
- `*_by text` records the actor (`api`, `worker:<host>`, `beat`, `reaper`, `cli`, `upgrade`).
  `*_by_user_id uuid NULL` columns exist now without a foreign key; the v0.8.0 authentication wave
  adds `users` and the foreign keys. No auth table is created in this program.

### 1.2 Foreign keys and cascades

- A project-scoped row references `projects(id) ON DELETE CASCADE`: deleting the project row
  removes all its control data in one statement
  ([constraints](https://www.postgresql.org/docs/current/ddl-constraints.html#DDL-CONSTRAINTS-FK)).
- A reference to a job from a domain row is `ON DELETE SET NULL` (history survives job pruning).
- A reference that must not dangle while in use is `ON DELETE RESTRICT` (an activation's revision, a
  promotion's model).

### 1.3 Concurrency

- Every editable document carries `revision bigint`. A write is
  `UPDATE ... SET revision = revision + 1 ... WHERE <key> AND revision = :expected` and zero rows is a
  409 (`revision_conflict` or the route's existing code). This replaces the OpenSearch
  `if_seq_no`/`if_primary_term` loops.
- Default isolation is READ COMMITTED
  ([isolation](https://www.postgresql.org/docs/current/transaction-iso.html#XACT-READ-COMMITTED)):
  a conditional `UPDATE` re-checks its `WHERE` after waiting for a concurrent writer, which is what
  the claim and OCC statements rely on. No code does "select, then insert if absent": uniqueness is
  a constraint and the insert handles `unique_violation`.

### 1.4 Project isolation in the database (D-55)

Every table with a `project_id` column has row-level security
([row security policies](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)):

```sql
ALTER TABLE <t> ENABLE ROW LEVEL SECURITY;
ALTER TABLE <t> FORCE ROW LEVEL SECURITY;
CREATE POLICY p_<t>_project ON <t>
  USING      (op_scope_allows(project_id))
  WITH CHECK (op_scope_allows(project_id));

-- review fix R-13: an unset scope raises (still fail closed, but loud, never a silent empty result)
CREATE FUNCTION op_scope_allows(pid uuid) RETURNS boolean
  LANGUAGE plpgsql STABLE AS $$
  BEGIN
    CASE current_setting('op.scope', true)
      WHEN 'project'    THEN RETURN pid = nullif(current_setting('op.project_id', true), '')::uuid;
      WHEN 'deployment' THEN RETURN true;
      ELSE RAISE EXCEPTION 'scope_not_set' USING ERRCODE = 'insufficient_privilege';
    END CASE;
  END $$;
```

- `src/db/session.py` exposes exactly two ways to open a transaction: `project_tx(session, project)`
  runs `SELECT set_config('op.scope','project',true), set_config('op.project_id', :id, true)` first.
  Review fix R-13: the scope is held in a context variable and re-applied by a SQLAlchemy
  `after_begin` session event ([session events](https://docs.sqlalchemy.org/en/21/orm/events.html#sqlalchemy.orm.SessionEvents.after_begin)),
  so a scoped session that commits and auto-begins again keeps its scope instead of silently losing
  it;
  `deployment_tx(session, reason)` sets `op.scope = 'deployment'` and is allowed only in the modules a
  ratchet lists (the reaper, beat tasks, the project registry, the upgrade step, admin CLIs).
  `set_config(..., true)` is transaction-local, so it also works behind a transaction pooler.
- A session that sets neither gets the `scope_not_set` error on its first row-level-secured
  statement: fail closed, and loud (a bug shows up as an error, not as a wrong 404).
- Tables with a nullable `project_id` (`jobs`, `events`, `config_documents`, `models`) allow
  `project_id IS NULL` rows only under `deployment` scope (the function returns false for NULL in
  `project` scope).
- The application role `op_app` is not the table owner and has `NOBYPASSRLS`. Review fix R-13:
  "Superusers and roles with the BYPASSRLS attribute always bypass the row security system"
  ([row security](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)), so `FORCE` only
  matters for a non-superuser owner; section 1.5 makes the schema owner a non-superuser.
- Tests: `tests/db/test_row_level_security.py` (a project-scoped transaction cannot read, update or
  insert another project's row in every RLS table; an unscoped statement raises `scope_not_set`; a
  scoped session keeps its scope across commit and auto-begin).

### 1.5 Roles

| Role | Used by | Rights |
|---|---|---|
| `op_super` (the image's `POSTGRES_USER`: the Docker image creates it "with superuser power", [postgres image](https://hub.docker.com/_/postgres)) | bootstrap only: `db-migrate` creates `op_owner` and `op_app` with it, then never again | everything |
| `op_owner` | `db-migrate` (DDL), `openprocessor db backup/restore` | owns the schema; not a superuser; `BYPASSRLS` so `pg_dump` sees every row (review fix R-13) |
| `op_app` | API, Celery workers, beat, stream workers | `SELECT, INSERT, UPDATE, DELETE` on the tables, `USAGE` on sequences, `EXECUTE` on `op_scope_allows`; no DDL, not a superuser, `NOBYPASSRLS` |

`db-migrate` connects as `op_super` only to create `op_owner` and `op_app` idempotently (passwords
from their secret files), then runs `alembic upgrade head` as `op_owner`, and re-applies the grants.
The superuser password file is mounted only into `postgres` and `db-migrate`.

## 2. Projects and deployment tables

```sql
CREATE TABLE project_slugs (
  slug        text PRIMARY KEY
              CHECK (slug ~ '^[a-z][a-z0-9-]{1,31}$' AND slug !~ '--' AND slug !~ '-$'),
  created_at  timestamptz NOT NULL DEFAULT now(),
  retired_at  timestamptz NULL              -- set when the project is deleted; the slug is never reused
);

CREATE TABLE projects (
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  lock_key           integer GENERATED ALWAYS AS IDENTITY UNIQUE,  -- second key of advisory locks
  slug               text NOT NULL UNIQUE REFERENCES project_slugs(slug),
  display_name       text NOT NULL,
  description        text NOT NULL DEFAULT '',
  status             text NOT NULL CHECK (status IN ('building','active','archived','deleting','failed')),
  pre_delete_status  text NULL CHECK (pre_delete_status IN ('active','archived','failed')),
  revision           bigint NOT NULL DEFAULT 1,        -- PATCH expected_revision
  config_epoch       bigint NOT NULL DEFAULT 0,        -- bumped by every config, activation, settings write
  paused_at          timestamptz NULL,                 -- replaces pipeline_paused.flag
  paused_by          text NULL,
  origin             jsonb NULL,                       -- combine provenance: [{project_id, slug}]
  index_prefix       text NOT NULL,                    -- computed once at create (rule kept from 0.6.0)
  storage_prefix     text NOT NULL,                    -- 'projects/<slug>'
  model_prefix       text NOT NULL,                    -- '<slug>__', '' for default
  mlflow_experiment  text NOT NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  updated_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  created_by_user_id uuid NULL,
  CHECK (slug NOT IN ('combine','new','all','none','projects','global','settings','vlm','health')),
  CHECK ((status = 'deleting') = (pre_delete_status IS NOT NULL))
);

CREATE TABLE project_indexes (
  project_id  uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  role        text NOT NULL CHECK (role IN ('images','items')),
  index_name  text NOT NULL UNIQUE,
  created_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, role)
);

CREATE TABLE deployment_counters (        -- 'projects_revision', 'global_config_epoch'
  name   text PRIMARY KEY,
  value  bigint NOT NULL DEFAULT 0
);

CREATE TABLE deployment_settings (        -- 'local_vlm_desired', 'vlm_paused_for_training', ...
  key         text PRIMARY KEY CHECK (key IN ('local_vlm_desired','vlm_paused_for_training')),
  value       jsonb NOT NULL,
  revision    bigint NOT NULL DEFAULT 1,
  updated_at  timestamptz NOT NULL DEFAULT now(),
  updated_by  text NOT NULL
);
```

Rules:

- **Create** (one transaction, deployment scope): `INSERT INTO project_slugs` (a primary-key
  conflict answers 409 `slug_taken`, or `slug_retired` when the existing row has `retired_at`), then
  `projects`, `project_indexes`, `class_registries`, `project_settings`, `project_keymaps`, then bump
  `deployment_counters('projects_revision')`. The OpenSearch indexes are created after the commit;
  the project stays `building` until they exist (B4-4D measures and may move this into a job).
  `project_slugs` makes "a deleted slug is never reused" a constraint instead of a check-then-insert
  race.
- **Delete** keeps today's lifecycle: `deleting` with `pre_delete_status`, the `project_delete` job
  drains jobs, deletes the indexes, the storage prefix, the promoted models and the MLflow
  experiment, then in one transaction sets `project_slugs.retired_at`, deletes the `projects` row
  (cascading every control row) and bumps `projects_revision`. The wire status `deleted` becomes a
  404 for a retired slug, as today.
- **Archive, unarchive, rename, describe**: `UPDATE ... WHERE id = :id AND revision = :expected AND
  status IN (...)`; zero rows is 409.
- The guard's ownership map (`index_name -> slug`) is built from `project_indexes` joined to
  `projects`; each process refreshes it when `projects_revision` changes (1 s poll, as today).

## 3. Class registry

```sql
CREATE TABLE class_registries (
  project_id     uuid PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
  next_class_id  integer NOT NULL DEFAULT 0 CHECK (next_class_id >= 0),
  revision       bigint NOT NULL DEFAULT 0,
  migrated_from  text NULL CHECK (migrated_from IN ('class_registry_file','opensearch_registry_doc')),
  migrated_at    timestamptz NULL,
  updated_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE project_classes (
  project_id     uuid NOT NULL REFERENCES class_registries(project_id) ON DELETE CASCADE,
  class_id       integer NOT NULL CHECK (class_id >= 0),
  name           text NOT NULL CHECK (length(name) BETWEEN 1 AND 200 AND name = btrim(name)),
  class_group    text NULL,
  deprecated     boolean NOT NULL DEFAULT false,
  merged_into    integer NULL,
  hotkey_letter  text NULL CHECK (hotkey_letter ~ '^[a-z0-9]$'),
  notes          text NOT NULL DEFAULT '',
  added_at       timestamptz NOT NULL DEFAULT now(),
  updated_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, class_id),
  name_key       text NOT NULL CHECK (name_key ~ '^[a-z0-9_]+$'),  -- normalize_class_name(name); R-3
  FOREIGN KEY (project_id, merged_into) REFERENCES project_classes(project_id, class_id),
  CHECK (merged_into IS NULL OR merged_into <> class_id),
  CHECK (merged_into IS NULL OR deprecated)
);
CREATE UNIQUE INDEX uq_project_classes_hotkey
  ON project_classes (project_id, hotkey_letter) WHERE hotkey_letter IS NOT NULL;
-- review fix R-3: names compare by their normalized key, and a name may be reused after deprecation
CREATE UNIQUE INDEX uq_project_classes_active_name
  ON project_classes (project_id, name_key) WHERE NOT deprecated;
CREATE INDEX ix_project_classes_name_key ON project_classes (project_id, name_key);

CREATE TABLE search_projection_tasks (       -- outbox for the class-name projection (D-47)
  id          uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id  uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind        text NOT NULL CHECK (kind IN ('class_rename','class_merge')),
  payload     jsonb NOT NULL,                -- {class_id, new_name} or {source_ids, target_id}
  job_id      uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  done_at     timestamptz NULL
);
CREATE INDEX ix_search_projection_tasks_pending ON search_projection_tasks (created_at) WHERE done_at IS NULL;
```

Rules:

- **`ensure_class(project, name)`** (import, combine, the VLM and open-vocabulary passes):
  `key = normalize_class_name(name)` (an empty key refuses with `class_name_invalid`);
  `SELECT ... FROM project_classes WHERE project_id = :p AND name_key = :key`, choosing as
  `resolve_class_by_name` does today: an active class wins over a deprecated one, then the exact
  spelling, then the lowest id. No active match: lock the registry row
  (`SELECT next_class_id FROM class_registries WHERE project_id = :p FOR UPDATE`), re-check, insert
  `(class_id = next_class_id, name = btrim(name), name_key = key)`, increment `next_class_id` and
  `revision`, commit. Two writers serialize on the registry row; ids are never reused; the partial
  unique index `uq_project_classes_active_name` is the last line of defence. `add_class` (the route)
  answers 409 when an active class has the same key, as today.
- **Name matching** is the current rule, read from the code (review fix R-3):
  `src/utils/class_names.py::normalize_class_name` lowercases, turns every run of characters outside
  `[a-z0-9]` into one `_` and trims edge underscores, so "Car", "car", "car " and "CAR" are one class
  and "traffic light" equals "traffic_light"; `resolve_class_by_name` says "a name may be reused
  after deprecation, so both can exist" (an active class wins); `registry.py::_add` strips the name and
  refuses a duplicate only against an active class. `name_key` is written only by
  `src/services/classes/resolve.py` from `normalize_class_name`; the stored `name` keeps the
  spelling. The 2E fixture includes a deprecated class and an active class with the same name, and
  2B-1's test pins "Car" / "car" / "car " / "traffic light" vs "traffic_light", a reused name after
  deprecation, and a name with no slug-safe character (refused).
- **No cap** (D-57): the 10,000-class limit and `registry_full` existed for the size of one
  OpenSearch document; both go. `registry_conflict` and `registry_not_migrated` go too (no OCC loop,
  no migration state at runtime).
- **Counts** (`sample_count`, `validated_count`) are not stored: the list route runs one terms
  aggregation on `items.class_id` (plus a filter aggregation for validated) and joins it.
- **Rename and merge** write the class rows and a `search_projection_tasks` row in one transaction;
  the `class_projection_sync` job applies it to items ([02](02_control_plane_database.md) section
  3.3). Merge also deprecates the source and sets `merged_into`.
- The per-process registry cache is keyed by `class_registries.revision` (one indexed read).

## 4. Config store, settings, keymap

```sql
CREATE TABLE config_documents (
  id               uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id       uuid NULL REFERENCES projects(id) ON DELETE CASCADE,   -- NULL: deployment-wide
  kind             text NOT NULL CHECK (kind IN
                   ('prompt_pack','region_profile','open_vocab_set','ingest_profile','vlm_endpoint')),
  name             text NOT NULL CHECK (name ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$'),
  current_revision integer NOT NULL CHECK (current_revision >= 1),
  description      text NOT NULL DEFAULT '',
  cloned_from_id   uuid NULL REFERENCES config_documents(id) ON DELETE SET NULL,
  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now(),
  updated_by       text NOT NULL,
  updated_by_user_id uuid NULL,
  UNIQUE NULLS NOT DISTINCT (project_id, kind, name),
  UNIQUE (id, kind),                                   -- target of the activation's composite FK
  CHECK ((kind = 'vlm_endpoint') = (project_id IS NULL))
);

CREATE TABLE config_revisions (
  config_id   uuid NOT NULL REFERENCES config_documents(id) ON DELETE CASCADE,
  revision    integer NOT NULL CHECK (revision >= 1),
  body        jsonb NOT NULL,
  body_sha256 text NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  created_by  text NOT NULL,
  PRIMARY KEY (config_id, revision)
);
-- trigger config_revisions_immutable: BEFORE UPDATE OR DELETE ... RAISE (a DELETE only by cascade:
-- the trigger allows it when the parent row is gone, checked with NOT EXISTS on config_documents)

CREATE TABLE config_activations (
  project_id    uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  axis          text NOT NULL CHECK (axis IN
                ('prompt_pack','detection_profile','vlm','open_vocab','ingest_profile')),
  kind          text NOT NULL,
  config_id     uuid NOT NULL,
  revision      integer NOT NULL,
  activated_at  timestamptz NOT NULL DEFAULT now(),
  activated_by  text NOT NULL,
  PRIMARY KEY (project_id, axis),
  FOREIGN KEY (config_id, kind)     REFERENCES config_documents(id, kind),
  FOREIGN KEY (config_id, revision) REFERENCES config_revisions(config_id, revision) ON DELETE RESTRICT,
  CHECK ((axis, kind) IN (('prompt_pack','prompt_pack'), ('detection_profile','region_profile'),
                          ('vlm','vlm_endpoint'), ('open_vocab','open_vocab_set'),
                          ('ingest_profile','ingest_profile')))
);

CREATE TABLE config_activation_events (
  id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  project_id     uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  axis           text NOT NULL,
  from_config_id uuid NULL, from_revision integer NULL,
  to_config_id   uuid NOT NULL, to_revision integer NOT NULL,
  reason         text NOT NULL CHECK (reason IN ('activate','rollback','clone','upgrade')),
  at             timestamptz NOT NULL DEFAULT now(),
  by             text NOT NULL
);
CREATE INDEX ix_config_activation_events_axis ON config_activation_events (project_id, axis, id DESC);

CREATE TABLE vlm_endpoint_probes (
  config_id  uuid NOT NULL REFERENCES config_documents(id) ON DELETE CASCADE,
  revision   integer NOT NULL,
  probed_at  timestamptz NOT NULL DEFAULT now(),
  ok         boolean NOT NULL,
  result     jsonb NOT NULL,
  PRIMARY KEY (config_id, revision),
  FOREIGN KEY (config_id, revision) REFERENCES config_revisions(config_id, revision) ON DELETE CASCADE
);

CREATE TABLE config_runtime_reports (       -- which process applied which epoch (was runtime:<process>:<host>)
  project_id    uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  process       text NOT NULL,
  host          text NOT NULL,
  config_epoch  bigint NOT NULL,
  applied_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, process, host)
);

CREATE TABLE project_settings (
  project_id      uuid PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
  defaults        jsonb NOT NULL DEFAULT '{}',
  ingest_policy   jsonb NULL,
  vlm_policy      jsonb NULL,
  vlm_scope_reps  jsonb NULL,
  revision        bigint NOT NULL DEFAULT 1,
  updated_at      timestamptz NOT NULL DEFAULT now(),
  updated_by      text NOT NULL
);

CREATE TABLE project_keymaps (
  project_id  uuid PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
  overrides   jsonb NOT NULL DEFAULT '{}',
  revision    bigint NOT NULL DEFAULT 1,
  updated_at  timestamptz NOT NULL DEFAULT now(),
  updated_by  text NOT NULL
);

CREATE TABLE project_state (                -- small internal markers, one pydantic model per key
  project_id  uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  key         text NOT NULL CHECK (key IN ('cluster_refresh','region_cluster_refine_marker',
              'region_cluster_partition_marker','last_clustering','region_drain','region_dependency_health')),
  value       jsonb NOT NULL,
  revision    bigint NOT NULL DEFAULT 1,
  updated_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, key)
);

CREATE TABLE projection_states (            -- UMAP view state (was doc 'current' + the umap_state index)
  project_id          uuid PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
  projection_version  integer NOT NULL,
  scope               text NOT NULL,
  cluster_id          integer NULL,
  n_points            integer NOT NULL,
  n_components        integer NOT NULL,
  metric              text NOT NULL,
  fitted_at           timestamptz NOT NULL,
  reducer_storage_key text NULL,
  job_id              uuid NULL REFERENCES jobs(id) ON DELETE SET NULL
);
```

Rules:

- **Save** (`save_config`): in one project-scoped transaction, `UPDATE config_documents SET
  current_revision = current_revision + 1 ... WHERE id = :id AND current_revision = :expected`
  (zero rows: `RevisionConflictError`, the existing 409), insert the `config_revisions` row, bump
  `projects.config_epoch` (or `deployment_counters('global_config_epoch')` for a VLM endpoint). The
  first save inserts the document with `current_revision = 1`; a name collision is the unique
  constraint. Revisions are immutable (trigger) and never reused (the primary key).
- **Activate**: `UPDATE config_activations SET config_id, revision, activated_at ... WHERE project_id
  AND axis AND config_id = :expected_config AND revision = :expected_revision` (zero rows: the
  existing `expected_active` 409); the first activation is an `INSERT ... ON CONFLICT DO NOTHING`
  whose zero-row result is the same 409. The event row and the epoch bump go in the same
  transaction. Rollback reactivates `from_*` of the latest event for the axis.
- **Settings and keymap writes** always carry `expected_revision` (the route already sends one for
  some axes; the `defaults` write that had no OCC gains it, and the route answers 409 on a stale
  revision; contracts updated in stage 2).
- **Polling**: every API and worker process polls `projects.config_epoch` for the projects it serves
  and `global_config_epoch` (`OP_CONFIG_POLL_S`, default 5 s) and reloads on change; one indexed
  read replaces the old counter document.

## 5. Jobs and events

The behaviour (state machine, claim, heartbeat, cancel, reaper) is in
[02b](02b_jobs_locking_events.md). The tables:

```sql
CREATE TABLE jobs (
  id                   uuid PRIMARY KEY DEFAULT uuidv7(),    -- also the Celery task_id
  project_id           uuid NULL REFERENCES projects(id) ON DELETE CASCADE,  -- NULL: deployment job
  kind                 text NOT NULL,                        -- validated against src/config/job_kinds/ in code; no CHECK, so a new kind needs no migration
  scope                text NOT NULL DEFAULT '',             -- e.g. a source id, an upload id, a slug
  exclusive_group      text NULL,                            -- from the kind table; NULL: not a singleton
  queue                text NOT NULL,                        -- from TASK_ROUTES; '' for inline jobs
  status               text NOT NULL CHECK (status IN ('queued','running','paused','cancelling',
                         'completed','completed_with_errors','failed','cancelled','interrupted')),
  params               jsonb NOT NULL DEFAULT '{}',
  checkpoint           jsonb NULL,                           -- resume cursor, written fenced
  progress             jsonb NOT NULL DEFAULT '{}',          -- {done, total, stage, counters}
  progress_updated_at  timestamptz NULL,
  result               jsonb NULL,
  error_code           text NULL,
  error_message        text NULL,
  attempt              integer NOT NULL DEFAULT 0,
  max_attempts         integer NOT NULL DEFAULT 3 CHECK (max_attempts BETWEEN 1 AND 10),
  auto_resume          boolean NOT NULL DEFAULT false,
  fencing_token        bigint NOT NULL DEFAULT 0,
  owner                text NULL,                            -- '<worker hostname>:<pid>'
  heartbeat_at         timestamptz NULL,
  lease_expires_at     timestamptz NULL,
  max_queue_wait       interval NOT NULL,
  dispatch_count       integer NOT NULL DEFAULT 0,
  last_dispatched_at   timestamptz NULL,
  cancel_requested_at  timestamptz NULL,
  cancel_requested_by  text NULL,
  pause_requested_at   timestamptz NULL,
  idempotency_key      text NULL CHECK (length(idempotency_key) BETWEEN 8 AND 200),
  parent_job_id        uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  requested_by         text NOT NULL,
  requested_by_user_id uuid NULL,
  created_at           timestamptz NOT NULL DEFAULT now(),
  queued_at            timestamptz NULL,
  started_at           timestamptz NULL,
  finished_at          timestamptz NULL,
  updated_at           timestamptz NOT NULL DEFAULT now(),
  CHECK ((status IN ('completed','completed_with_errors','failed','cancelled')) = (finished_at IS NOT NULL)),
  CHECK (status NOT IN ('running','cancelling') OR (owner IS NOT NULL AND lease_expires_at IS NOT NULL))
);
CREATE UNIQUE INDEX uq_jobs_one_active
  ON jobs (project_id, exclusive_group, scope) NULLS NOT DISTINCT
  WHERE exclusive_group IS NOT NULL AND status IN ('queued','running','paused','cancelling');
CREATE UNIQUE INDEX uq_jobs_idempotency
  ON jobs (project_id, kind, idempotency_key) NULLS NOT DISTINCT
  WHERE idempotency_key IS NOT NULL;
CREATE INDEX ix_jobs_project_recent ON jobs (project_id, kind, created_at DESC);
CREATE INDEX ix_jobs_lease ON jobs (lease_expires_at) WHERE status IN ('running','cancelling');
CREATE INDEX ix_jobs_queued ON jobs (queued_at) WHERE status = 'queued';
CREATE INDEX ix_jobs_interrupted ON jobs (updated_at) WHERE status = 'interrupted' AND auto_resume;
-- trigger jobs_status_transition_guard: BEFORE UPDATE OF status; raises unless (OLD.status, NEW.status)
-- is in the transition table of 02b section 1 (generated from src/domain/job_states.py)

CREATE TABLE job_attempts (
  job_id         uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  attempt        integer NOT NULL,
  fencing_token  bigint NOT NULL,
  owner          text NOT NULL,
  started_at     timestamptz NOT NULL DEFAULT now(),
  ended_at       timestamptz NULL,
  outcome        text NULL CHECK (outcome IN ('completed','completed_with_errors','failed',
                   'cancelled','paused','interrupted','lost','time_limit')),
  error_code     text NULL,
  error_message  text NULL,
  PRIMARY KEY (job_id, attempt)
);

CREATE TABLE job_items (                    -- explicit item lists (assist members, reprocess targets)
  job_id    uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  position  integer NOT NULL,
  item_id   text NOT NULL,
  PRIMARY KEY (job_id, position),
  UNIQUE (job_id, item_id)
);

CREATE TABLE events (
  id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  xid         xid8 NOT NULL DEFAULT pg_current_xact_id(),   -- commit-safe resume (02b section 6.3)
  project_id  uuid NULL REFERENCES projects(id) ON DELETE CASCADE,  -- NULL: the global stream
  job_id      uuid NULL REFERENCES jobs(id) ON DELETE CASCADE,
  type        text NOT NULL,                  -- 'job.state', 'job.progress', 'combine.progress', 'config.changed', ...
  payload     jsonb NOT NULL CHECK (octet_length(payload::text) <= 7000),
  created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ix_events_cursor ON events (project_id, xid, id);   -- the SSE cursor is (xid, id)
CREATE INDEX ix_events_created ON events (created_at);

CREATE TABLE stream_worker_claims (         -- D-53: detection and VLM worker replicas claim projects
  worker_kind      text NOT NULL CHECK (worker_kind IN ('detection','vlm')),
  project_id       uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  worker_id        text NOT NULL,
  fencing_token    bigint NOT NULL DEFAULT 1,
  lease_expires_at timestamptz NOT NULL,
  PRIMARY KEY (worker_kind, project_id)
);

CREATE TABLE worker_capabilities (          -- written by a worker at start; replaces .trainer_capabilities.json
  worker_id    text PRIMARY KEY,             -- '<queue>@<hostname>'
  queue        text NOT NULL,
  host         text NOT NULL,
  capabilities jsonb NOT NULL,               -- GPUs, families, versions
  seen_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gpu_reservations (             -- replaces training_gpus.lock (the arbiter is single-host)
  host           text NOT NULL,
  gpu_index      integer NOT NULL CHECK (gpu_index >= 0),
  holder_job_id  uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  reserved_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (host, gpu_index)
);
```

`job_attempts`, `job_items` and `gpu_reservations` reach the project through `jobs`; their RLS
policy is `USING (EXISTS (SELECT 1 FROM jobs j WHERE j.id = job_id AND op_scope_allows(j.project_id)))`.

## 6. Datasets, ingest sources, label operations, holdout

```sql
CREATE TABLE dataset_uploads (
  id             uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id     uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  filename       text NOT NULL,
  size_bytes     bigint NOT NULL CHECK (size_bytes >= 0),
  sha256         text NULL,
  storage_key    text NOT NULL UNIQUE,
  status         text NOT NULL CHECK (status IN ('receiving','received','extracting','extracted','failed','expired')),
  extract_job_id uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  created_at     timestamptz NOT NULL DEFAULT now(),
  expires_at     timestamptz NOT NULL
);

CREATE TABLE dataset_imports (
  id            uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id    uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id        uuid NULL UNIQUE REFERENCES jobs(id) ON DELETE SET NULL,
  undo_job_id   uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  upload_id     uuid NULL REFERENCES dataset_uploads(id) ON DELETE SET NULL,
  source_key    text NULL,                    -- a server path as a storage key, when not an upload
  format        text NOT NULL,
  mapping       jsonb NOT NULL,                -- [{dataset_class, action}], by name
  options       jsonb NOT NULL,
  scan          jsonb NULL,
  report        jsonb NULL,
  created_at    timestamptz NOT NULL DEFAULT now(),
  undone_at     timestamptz NULL,
  CHECK ((upload_id IS NULL) <> (source_key IS NULL))
);

CREATE TABLE dataset_import_images (         -- the ledger: resume and undo
  import_id  uuid NOT NULL REFERENCES dataset_imports(id) ON DELETE CASCADE,
  image_key  text NOT NULL,                  -- path inside the dataset
  image_id   text NULL,                      -- the OpenSearch image id once written
  item_ids   text[] NOT NULL DEFAULT '{}',
  status     text NOT NULL CHECK (status IN ('pending','written','skipped','failed','undone')),
  error      text NULL,
  PRIMARY KEY (import_id, image_key)
);
CREATE INDEX ix_dataset_import_images_pending ON dataset_import_images (import_id) WHERE status = 'pending';

CREATE TABLE combine_runs (
  job_id             uuid PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
  target_project_id  uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  holdout_mode       text NOT NULL CHECK (holdout_mode IN ('union','recompute','drop')),
  class_mapping      jsonb NOT NULL,
  created_at         timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE combine_sources (
  job_id             uuid NOT NULL REFERENCES combine_runs(job_id) ON DELETE CASCADE,
  position           smallint NOT NULL CHECK (position BETWEEN 0 AND 7),
  source_project_id  uuid NULL REFERENCES projects(id) ON DELETE SET NULL,
  source_slug        text NOT NULL,
  PRIMARY KEY (job_id, position)
);

CREATE TABLE ingest_sources (
  id          uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id  uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name        text NOT NULL,
  kind        text NOT NULL,                 -- the existing source kinds (directory, object-store events, webhook)
  config      jsonb NOT NULL,
  schedule    jsonb NOT NULL,                -- poll interval, cron, webhook
  enabled     boolean NOT NULL DEFAULT true,
  revision    bigint NOT NULL DEFAULT 1,
  last_poll_at timestamptz NULL,
  scan_state  jsonb NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (project_id, name)
);

CREATE TABLE ingest_source_files (           -- replaces the per-source queue.sqlite and the run ledger
  source_id    uuid NOT NULL REFERENCES ingest_sources(id) ON DELETE CASCADE,
  file_key     text NOT NULL,
  status       text NOT NULL CHECK (status IN ('pending','claimed','done','failed','skipped')),
  job_id       uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  attempts     integer NOT NULL DEFAULT 0,
  image_id     text NULL,
  error        text NULL,
  enqueued_at  timestamptz NOT NULL DEFAULT now(),
  updated_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (source_id, file_key)
);
CREATE INDEX ix_ingest_source_files_pending ON ingest_source_files (source_id, enqueued_at) WHERE status = 'pending';

CREATE TABLE label_operations (
  id            uuid PRIMARY KEY DEFAULT uuidv7(),   -- the op_id items carry in label_op_id
  project_id    uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind          text NOT NULL,
  status        text NOT NULL CHECK (status IN ('writing','written','undoing','undone','failed')),
  params        jsonb NOT NULL,
  member_count  integer NOT NULL DEFAULT 0,
  job_id        uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  created_at    timestamptz NOT NULL DEFAULT now(),
  created_by    text NOT NULL,
  undone_at     timestamptz NULL
);
CREATE TABLE label_operation_members (
  op_id    uuid NOT NULL REFERENCES label_operations(id) ON DELETE CASCADE,
  item_id  text NOT NULL,
  prior    jsonb NOT NULL,                   -- the item fields the op changed, for undo
  PRIMARY KEY (op_id, item_id)
);

CREATE TABLE holdout_freezes (
  id             uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id     uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  is_current     boolean NOT NULL,
  source         text NOT NULL,              -- 'manual', 'import:<id>', 'combine:<job id>', 'upgrade'
  item_count     integer NOT NULL,
  seed           bigint NULL,
  snapshot_storage_key text NULL,            -- the frozen member list, for audit
  frozen_at      timestamptz NOT NULL DEFAULT now(),
  frozen_by      text NOT NULL
);
CREATE UNIQUE INDEX uq_holdout_current ON holdout_freezes (project_id) WHERE is_current;
```

`dataset_import_images`, `combine_sources`, `ingest_source_files` and `label_operation_members`
reach the project through their parent; their RLS policy uses an `EXISTS` on the parent as in
section 5.

## 7. Training, models, promotion, export, bake-off

```sql
CREATE TABLE dataset_exports (
  id               uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id       uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id           uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  kind             text NOT NULL CHECK (kind IN ('yolo','single_class')),
  class_names      text[] NOT NULL,           -- in label order: the export's identity by name
  manifest_sha256  text NOT NULL,
  artifact_prefix  text NOT NULL,             -- storage key of the frozen export
  split_seed       bigint NOT NULL,
  created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE training_runs (
  id                    uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id            uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id                uuid NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE RESTRICT,  -- status lives on the job
  dataset_export_id     uuid NULL REFERENCES dataset_exports(id) ON DELETE SET NULL,
  spec                  jsonb NOT NULL,
  class_names           text[] NOT NULL,      -- the training pin, in output order
  family                text NOT NULL,
  base_model            text NOT NULL,
  artifact_prefix       text NOT NULL,        -- checkpoints, run.log, manifest file
  manifest              jsonb NULL,           -- lineage: dataset hash, code versions, evaluation
  mlflow_run_id         text NULL,
  gpu_ids               integer[] NOT NULL DEFAULT '{}',
  created_at            timestamptz NOT NULL DEFAULT now(),
  finished_at           timestamptz NULL
);

CREATE TABLE models (
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id         uuid NULL REFERENCES projects(id) ON DELETE CASCADE,   -- NULL: core model
  name               text NOT NULL,
  version            integer NOT NULL CHECK (version >= 1),
  origin             text NOT NULL CHECK (origin IN ('trained','uploaded','builtin')),
  training_run_id    uuid NULL REFERENCES training_runs(id) ON DELETE SET NULL,
  family             text NOT NULL,
  task               text NOT NULL,
  labels             text[] NOT NULL,         -- class names in output order; labels.txt is written from this
  artifact_storage_key text NOT NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  UNIQUE NULLS NOT DISTINCT (project_id, name, version)
);

CREATE TABLE model_engines (
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  model_id           uuid NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  triton_model_name  text NOT NULL,
  precision          text NOT NULL,
  input_size         integer NOT NULL,
  trt_version        text NOT NULL,
  gpu_arch           text NOT NULL,
  engine_storage_key text NULL,
  build_job_id       uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  status             text NOT NULL CHECK (status IN ('building','ready','failed')),
  built_at           timestamptz NULL,
  UNIQUE (triton_model_name, trt_version, gpu_arch)
);

CREATE TABLE promotions (
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id         uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  model_id           uuid NOT NULL REFERENCES models(id) ON DELETE RESTRICT,
  triton_model_name  text NOT NULL,           -- '<slug>__<name>'
  triton_version     integer NOT NULL,
  class_remap        jsonb NULL,              -- by name, for a class-subset run
  job_id             uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  promoted_at        timestamptz NOT NULL DEFAULT now(),
  promoted_by        text NOT NULL,
  superseded_at      timestamptz NULL
);
CREATE UNIQUE INDEX uq_promotions_live ON promotions (triton_model_name) WHERE superseded_at IS NULL;

CREATE TABLE model_shares (
  model_id             uuid NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  consumer_project_id  uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  created_at           timestamptz NOT NULL DEFAULT now(),
  created_by           text NOT NULL,
  PRIMARY KEY (model_id, consumer_project_id)
);

CREATE TABLE bakeoff_runs (
  job_id              uuid PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
  project_id          uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  eval_dataset_key    text NOT NULL,
  models              jsonb NOT NULL,
  results_storage_key text NULL,
  created_at          timestamptz NOT NULL DEFAULT now()
);
```

Rules: a model shared into another project maps classes by name (`models.labels` against the
consumer's `project_classes.name_key`), never by position, as today. Review fix R-13: the consumer
reads its shares through a second policy that is `FOR SELECT` only
(`CREATE POLICY p_model_shares_consumer ON model_shares FOR SELECT USING (op_scope_allows(consumer_project_id))`),
so a consumer can never insert a share of a model it does not own (foreign-key checks bypass RLS);
only the owning project's scope inserts or deletes shares. The consumer reads the shared `models`
row through `CREATE POLICY p_models_shared ON models FOR SELECT USING (EXISTS (SELECT 1 FROM model_shares s WHERE s.model_id = models.id AND op_scope_allows(s.consumer_project_id)))`.

## 8. Upgrade bookkeeping

```sql
CREATE TABLE upgrade_runs (
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  from_version       text NOT NULL,
  to_version         text NOT NULL,
  step               text NOT NULL CHECK (step IN ('export_bundle','load','verify','delete_sources')),
  status             text NOT NULL CHECK (status IN ('running','completed','failed')),
  source_fingerprint text NOT NULL,           -- sha256 of the bundle
  report             jsonb NULL,
  started_at         timestamptz NOT NULL DEFAULT now(),
  finished_at        timestamptz NULL
);
CREATE UNIQUE INDEX uq_upgrade_step_done ON upgrade_runs (from_version, step) WHERE status = 'completed';
```

## 9. Sizes and retention

| Table | Growth | Retention (the reaper, [02b](02b_jobs_locking_events.md) section 4) |
|---|---|---|
| `events` | progress at most 1 row per second per running job, plus lifecycle events | rows older than 24 h deleted in batches of 10,000 |
| `jobs`, `job_attempts` | one row per start | terminal rows older than 90 days deleted, except kinds referenced by `training_runs`, `dataset_imports`, `combine_runs`, `bakeoff_runs`, `promotions`, `dataset_exports` (those keep their job) |
| `ingest_source_files` | one row per discovered file | `done` rows older than 30 days deleted |
| `config_runtime_reports`, `worker_capabilities` | one row per process and host | rows not seen for 1 day deleted |
| `dataset_uploads` | one per upload | `expires_at` passed: blob and row removed |
| `dataset_import_images`, `label_operation_members` | one per image or item | kept with their parent (undo needs them) |

The retention constants live in `src/config/retention.py` (constants, not env, until a deployment
needs to tune one).
