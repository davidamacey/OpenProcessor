# 02a: the PostgreSQL schema

| Field | Value |
|---|---|
| Status | plan, not started |
| Part of | [02_control_plane_database.md](02_control_plane_database.md) (read its sections 2 and 3 first) |
| Built by | B2 stage 1 (all tables, one initial Alembic revision), used by stages 2-6 ([03_b2_overview.md](03_b2_overview.md)) |
| Research seams | the tables the seven research plans need are in [02d_research_seam_schema.md](02d_research_seam_schema.md) (revisions `0002`-`0004`); the changes they make to tables of this file are edited in place below and listed in 02d section 1 (approved by the owner 2026-11-04, RD-1 of [00d](reference/owner_questions.md)) |
| Server | PostgreSQL 18 (latest stable major on 2026-11-04; 19 is a release candidate). The schema needs at least 15 (`NULLS NOT DISTINCT`) and 18 (`uuidv7()`); the exact image tag is in `01_package_pins.md` (written on its own branch) |

The DDL below is the contract. Stage 1 writes it as SQLAlchemy models in `src/db/models/` and one
Alembic revision `0001_control_plane` (followed by the fixed chain of 02d section 0), drafted with
autogenerate and corrected by hand against this file (Alembic does not detect everything:
[what autogenerate detects](https://alembic.sqlalchemy.org/en/latest/autogenerate.html)). A test
(`tests/db/test_schema_matches_plan.py`) introspects the migrated database and asserts every table,
unique constraint, partial index and row-level-security policy named here.

---

## 0.1 Table index: who writes, who reads, in which wave

Tables of this file (the research tables are indexed in [02d](02d_research_seam_schema.md) section 0.1). Wave names are those of [00_master_plan.md](00_master_plan.md); all of them are created in revision `0001_control_plane` (W2.1) and filled by the stage that owns the writer.

| Table | Writer | Reader | Wave |
|---|---|---|---|
| `project_slugs`, `projects`, `project_indexes` | project service (create, patch, delete job) | every route (scope), the OpenSearch guard, workers | W2.1 stage 2 |
| `deployment_counters`, `deployment_settings` | project and settings routes, the VLM pause and serving profile activation | API and worker snapshot polls | W2.1 stage 2 |
| `class_registries`, `project_classes`, `search_projection_tasks` | class service (`ensure_class`, rename, merge); outbox rows in the same transaction | class routes, import, export, `class_projection_sync` and the reaper rules e and i | W2.1 stage 2 |
| `config_documents`, `config_revisions`, `config_activations`, `config_activation_events` | the config store | the config readers (region, VLM, ingest, prompt packs), the activation routes | W2.1 stage 2 |
| `vlm_endpoint_probes`, `config_runtime_reports` | the probe task and each process at start | health, settings and the config-epoch check | W2.1 stage 2 |
| `project_settings`, `project_keymaps`, `project_state`, `projection_states` | the settings and keymap routes; workers (markers); the UMAP job | the same routes, workers, the UMAP view | W2.1 stage 2 |
| `jobs`, `job_attempts`, `job_items` | `start_job` and the task wrapper | job routes, the reaper, the heartbeat | W2.1 stage 3 |
| `events` | job runtime and lifecycle routes | the SSE routes | W2.1 stage 3 |
| `stream_worker_claims`, `worker_capabilities` | detection and VLM stream workers; every worker at start | the same replicas; `openprocessor gpu plan`, consumer checks | W2.1 stage 3 |
| `gpu_devices`, `gpu_reservations` | GPU workers, `op.periodic.serving_inventory`; the arbiter and training | `openprocessor gpu plan`, the arbiter | W2.1 stage 5 |
| `dataset_uploads`, `dataset_imports`, `dataset_import_images` | the upload and import jobs | import status and undo routes | W2.3a |
| `combine_runs`, `combine_sources` | the combine job | combine status routes | W2.3a |
| `ingest_sources`, `ingest_source_files` | source poll and bulk ingest | bulk ingest status | W2.3a |
| `label_operations`, `label_operation_members` | `label_op` and `label_undo` jobs (Celery, D-75) | undo routes and status | W2.3a |
| `holdout_freezes` | the holdout freeze route and near-duplicate lanes | the holdout guard, export | W2.3a |
| `dataset_exports`, `training_runs`, `models`, `model_engines`, `promotions`, `model_shares`, `bakeoff_runs` | export, train, promote, model-export and bake-off jobs | model, promotion and training routes, Triton router, `gpu plan` | W2.1 stage 5 |
| `upgrade_runs` | the 0.6.0 upgrade steps | `openprocessor upgrade`, `db verify` | W2.1 stage 2 |

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
- Review RR-17: every `CHECK` a later revision alters has an explicit name under the `ck_`
  convention (`ck_config_documents_kind`, `ck_config_documents_scope`, `ck_config_activations_axis`,
  `ck_config_activations_axis_kind`, `ck_search_projection_tasks_kind`, `ck_project_state_key`), so
  `0003` and `0004` drop and recreate them by a stable name.
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

- A project-scoped row references `projects(id) ON DELETE CASCADE`. The cascade is the safety net
  behind the `project_delete` job's batched delete (4C-5), not the delete path itself (review RN-7):
  a large project is removed table by table in batches first, and the final `DELETE FROM projects`
  then finds little left
  ([constraints](https://www.postgresql.org/docs/current/ddl-constraints.html#DDL-CONSTRAINTS-FK)).
- A reference to a job from a domain row is `ON DELETE SET NULL` (history survives job pruning).
- A reference that must not dangle while in use keeps the default `NO ACTION`, not `RESTRICT`
  (research integration, 02d X-1; reason corrected by review RR-6): both refuse a delete of a row
  still referenced; the documented difference is that `NO ACTION` can be deferred and `RESTRICT`
  cannot, and `RESTRICT` refuses even when the final state would be valid
  ([foreign keys](https://www.postgresql.org/docs/current/ddl-constraints.html#DDL-CONSTRAINTS-FK),
  [CREATE TABLE](https://www.postgresql.org/docs/current/sql-createtable.html)). Nothing promises
  that a non-deferrable `NO ACTION` passes a cascade in every trigger order, so the project delete
  does not rely on one cascade: the `project_delete` job deletes the project's rows table by table
  in reverse foreign-key order in batches first (RR-7, lane 4C-5), and
  `tests/db/test_profile_fk_cascade.py` (02d section 6) enumerates every referencing pair from the
  catalog; a pair that still fails is declared `DEFERRABLE INITIALLY DEFERRED`.
- **Guarded deletes (review RN-1).** Every immutability trigger (`config_revisions_immutable`,
  `cluster_partitions_ready_immutable`, `cluster_children_ready_immutable`,
  `cluster_calibrations_immutable`, and any later one) allows a DELETE in exactly two cases: the
  parent row is gone (a cascade), or `op_project_delete_allows(OLD.project_id)` is true. That
  function (below) is true only inside the transaction of the `project_delete` job for that project:
  the job sets `set_config('op.project_delete', <project id>, true)` (transaction-local, like
  `op.scope`) and the project row must already be `deleting` (2A's lifecycle; only the delete route
  sets it). A role check is not used because every app process runs as the one role `op_app`
  (section 1.5); the status condition is what a forged setting cannot fake, and
  `tests/test_project_delete_flag_writers.py` allows the literal `op.project_delete` in one file only,
  `src/services/projects/delete.py`. So the batched delete may remove children before parents.
  Retention does not use the flag: it sets a superseded partition to `discarded` first (a status
  change the partition trigger allows) and then deletes its children in batches, which the children
  trigger allows because the parent is no longer `ready` (02d section 2.2). Red test:
  `tests/projects/test_project_delete_guarded.py` (`db`): a project holding a `ready` partition with
  clusters and memberships, a config document with two revisions and a project calibration deletes
  through the batched job; controls: the same child DELETE without the setting, with the setting
  naming another project, or with the project not `deleting`, raises.
  Examples: an activation's revision, a
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
  USING      ((SELECT op_scope_is_deployment()) OR project_id = (SELECT op_scope_project())
              OR (project_id IS NULL AND (SELECT op_scope_is_core())))        -- review RR-3
  WITH CHECK ((SELECT op_scope_is_deployment()) OR project_id = (SELECT op_scope_project())
              OR (project_id IS NULL AND (SELECT op_scope_is_core())));

-- review fix R-13: an unset scope raises (still fail closed, but loud, never a silent empty result)
CREATE FUNCTION op_scope_project() RETURNS uuid
  LANGUAGE plpgsql STABLE AS $$
  BEGIN
    CASE current_setting('op.scope', true)
      WHEN 'project'    THEN RETURN nullif(current_setting('op.project_id', true), '')::uuid;
      WHEN 'deployment' THEN RETURN NULL;
      WHEN 'core'       THEN RETURN NULL;   -- review RR-3: core sees only project_id IS NULL rows
      ELSE RAISE EXCEPTION 'scope_not_set' USING ERRCODE = 'insufficient_privilege';
    END CASE;
  END $$;
CREATE FUNCTION op_scope_is_deployment() RETURNS boolean
  LANGUAGE sql STABLE AS $$ SELECT coalesce(current_setting('op.scope', true) = 'deployment', false) $$;
CREATE FUNCTION op_scope_is_core() RETURNS boolean
  LANGUAGE sql STABLE AS $$ SELECT coalesce(current_setting('op.scope', true) = 'core', false) $$;
-- the EXISTS policies of sections 5 and 7 keep the one-argument form
CREATE FUNCTION op_scope_allows(pid uuid) RETURNS boolean
  LANGUAGE sql STABLE AS $$ SELECT op_scope_is_deployment() OR pid = op_scope_project()
                               OR (pid IS NULL AND op_scope_is_core()) $$;   -- review RN-2
-- review RN-1: the one exception every immutability trigger honours for DELETE
CREATE FUNCTION op_project_delete_allows(pid uuid) RETURNS boolean
  LANGUAGE sql STABLE AS $$ SELECT pid IS NOT NULL
    AND coalesce(current_setting('op.project_delete', true), '') = pid::text
    AND EXISTS (SELECT 1 FROM projects p WHERE p.id = pid AND p.status = 'deleting') $$;
```

- `src/db/session.py` exposes exactly three ways to open a transaction (the third, `core_tx`, below): `project_tx(session, project)`
  runs `SELECT set_config('op.scope','project',true), set_config('op.project_id', :id, true)` first.
  Review fix R-13: the scope is held in a context variable and re-applied by a SQLAlchemy
  `after_begin` session event ([session events](https://docs.sqlalchemy.org/en/21/orm/events.html#sqlalchemy.orm.SessionEvents.after_begin)),
  so a scoped session that commits and auto-begins again keeps its scope instead of silently losing
  it;
  `deployment_tx(session, reason)` sets `op.scope = 'deployment'` and is allowed only in the modules a
  ratchet lists (the reaper, beat tasks, the project registry, the upgrade step, admin CLIs, the
  GPU worker start in `src/tasks/runtime/gpu_app.py` for `gpu_devices` (review RN-8), and the
  `project_delete` job, which is a deployment-level row).
  `set_config(..., true)` is transaction-local, so it also works behind a transaction pooler.
- A session that sets neither gets the `scope_not_set` error on its first row-level-secured
  statement: fail closed, and loud (a bug shows up as an error, not as a wrong 404).
- Review RC-9: each function call in the policy is a scalar subquery, which PostgreSQL evaluates
  once per statement as an InitPlan, so the policy costs one call per statement instead of one per
  row. That is all the InitPlan does: the policy is an OR over the scope functions, and an OR over
  an InitPlan result is **not** what lets the planner use a `project_id` index. The index is usable
  because every repository adds the explicit `project_id = :p` filter to its statement (the policy is
  the safety net, the filter is the access path). The planner cannot inline a plpgsql function, which
  is why the raising function is wrapped this way. Test: `EXPLAIN` of a scoped select **with the
  explicit `project_id = :p` filter** on `images`, `jobs` and `events` shows an index condition on
  `project_id`; a second case without the filter asserts only that the result is still correct (RLS
  hides other projects), not that an index is used.
- Tables with a nullable `project_id` (`jobs`, `events`, `config_documents`, `models`, and the
  research tables `cluster_partitions`, `partition_clusters`, `cluster_memberships`,
  `cluster_identities`, `cluster_activations`, `cluster_activation_events`, `cluster_calibrations`,
  `image_contents`, `near_dup_runs`, `near_dup_groups`, `near_dup_members`, `near_dup_edges`,
  `near_dup_pending`) allow `project_id IS NULL` rows under `deployment` scope and under the third
  scope **`core`** (review RR-3), never under `project` scope. `core_tx(session)` sets
  `op.scope = 'core'` and sees only the NULL-project rows (core faces, the core global index's
  near-duplicates, core jobs); it is allowed only in the modules a ratchet lists (`src/routers/persons.py`,
  `src/services/face_identity*`, the core ingest services and routers, `src/services/duplicate_detection/`,
  `src/routers/query_duplicates.py`, the core near-duplicate and person-grouping task handlers).
  Deployment-default `cluster_calibrations` (project NULL) are also readable by a project-scoped
  run through a second policy `CREATE POLICY p_cluster_calibrations_default ON cluster_calibrations
  FOR SELECT USING (project_id IS NULL OR op_scope_allows(project_id))` (the `p_model_shares_consumer`
  pattern): read-only, so a project can never write a shared calibration.
- The application role `op_app` is not the table owner and has `NOBYPASSRLS`. Review fix R-13:
  "Superusers and roles with the BYPASSRLS attribute always bypass the row security system"
  ([row security](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)), so `FORCE` only
  matters for a non-superuser owner; section 1.5 makes the schema owner a non-superuser.
- Tests: `tests/db/test_row_level_security.py` (a project-scoped transaction cannot read, update or
  insert another project's row in every RLS table; an unscoped statement raises `scope_not_set`; a
  scoped session keeps its scope across commit and auto-begin; review RR-3: project scope cannot see
  core rows, core scope cannot see project rows, a core job (project NULL) writes its `job_attempts` and `job_items` rows under `core_tx` (review RN-2), a project-scoped run reads a deployment calibration
  and cannot insert one; `core_tx` outside the allowlist raises).

### 1.5 Roles

| Role | Used by | Rights |
|---|---|---|
| `op_super` (the image's `POSTGRES_USER`: the Docker image creates it "with superuser power", [postgres image](https://hub.docker.com/_/postgres)) | bootstrap only: `db-migrate` creates `op_owner` and `op_app` with it, then never again | everything |
| `op_owner` | `db-migrate` (DDL), `openprocessor db backup/restore` | owns the schema; not a superuser; `BYPASSRLS` so `pg_dump` sees every row (review fix R-13) |
| `op_app` | API, Celery workers, beat, stream workers | `SELECT, INSERT, UPDATE, DELETE` on the tables, `USAGE` on sequences, `EXECUTE` on `op_scope_project`, `op_scope_is_deployment`, `op_scope_is_core`, `op_scope_allows` and `op_project_delete_allows` (review RN-2); no DDL, not a superuser, `NOBYPASSRLS` |

#### 1.5.1 Session settings per role (D-72)

Set once in the role bootstrap of `db-migrate`, with `ALTER ROLE ... SET`, so they apply to every new
session of the role and are not repeated in application code. The parameters are ordinary
client-connection settings ([client connection defaults](https://www.postgresql.org/docs/current/runtime-config-client.html):
`idle_in_transaction_session_timeout` terminates a session idle inside an open transaction,
`statement_timeout` aborts a statement, `lock_timeout` aborts a lock wait, each with 0 meaning off by
default; the same page says not to set the latter two in `postgresql.conf` because that affects all
sessions, hence per role):

| Role | `idle_in_transaction_session_timeout` | `statement_timeout` | `lock_timeout` | `timezone` |
|---|---|---|---|---|
| `op_app` (API, workers, beat, stream workers) | 60 s | 30 s | 5 s | `UTC` |
| `op_owner` (`db-migrate`, `db backup/restore`) | 5 min | 0 (off: a migration may rebuild an index) | 60 s (a migration waits for a lock, never forever) | `UTC` |
| the reaper | not a role: its tick sets `SET LOCAL statement_timeout = '20s'` and `SET LOCAL lock_timeout = '2s'` inside its transaction, so one stuck row cannot hold the 30 s tick | | | |

Job workers need longer statements than the API (a batched backfill, a retention delete, a bulk
insert): the job wrapper raises `statement_timeout` to 30 min and `lock_timeout`
to 30 s on its own connections at connect time ([02b](02b_jobs_locking_events.md) section 3.1 step 1);
the API role keeps the short values, so a runaway request query ends in seconds. Every process sets
`application_name` in its connection arguments (`op-api-<pid>`, `op-job-<queue>-<pid>`, `op-beat`,
`op-migrate`), so `pg_stat_activity` names who holds a lock. Test (`tests/db/test_role_settings.py`):
a new `op_app` session shows the four values in `SHOW`; a statement that sleeps 31 s is cancelled; a
transaction idle for 61 s is terminated; the job wrapper's connection shows 30 min; a connection
shows its `application_name`.

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
  lock_key           integer GENERATED ALWAYS AS IDENTITY UNIQUE,  -- second key of advisory locks; never reused
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
  key         text PRIMARY KEY CHECK (key IN ('local_vlm_desired','vlm_paused_for_training',
              'active_serving_profile')),      -- research X-2: {config_id, revision} of the deployment serving profile
  value       jsonb NOT NULL,
  revision    bigint NOT NULL DEFAULT 1,
  updated_at  timestamptz NOT NULL DEFAULT now(),
  updated_by  text NOT NULL
);
```

`projects.lock_key` is an identity column: the database hands each project a new value and never
reissues one, so deleting a project leaves a gap and a later project with the same slug can never get
the same key (the slug is retired too, `project_slugs.retired_at`, and never reused). A leftover
advisory lock on a deleted project's key therefore cannot collide with a new project. A test creates,
deletes and recreates a slug and asserts the new `lock_key` is different. The 32-bit range is the
documented two-key advisory lock form, `pg_advisory_xact_lock(class, lock_key)`.

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
  kind        text NOT NULL CHECK (kind IN ('class_rename','class_merge','cluster_partition')),
  payload     jsonb NOT NULL,                -- {class_id, new_name}, {source_ids, target_id} or
                                             -- {target, level, embedding_field, partition_id} (02d section 1)
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
                   ('prompt_pack','region_profile','open_vocab_set','ingest_profile','vlm_endpoint',
                    'serving_profile','cluster_profile','assist_loop_preset')),   -- 02d section 1
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
  CHECK ((kind IN ('vlm_endpoint','serving_profile')) = (project_id IS NULL))
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
-- the trigger allows it when the parent row is gone, checked with NOT EXISTS on config_documents,
-- or when op_project_delete_allows(<the document's project_id>) is true: section 1.2, review RN-1)

CREATE TABLE config_activations (
  project_id    uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  axis          text NOT NULL CHECK (axis IN
                ('prompt_pack','detection_profile','vlm','open_vocab','ingest_profile',
                 'clustering_item','clustering_region_box','assist_loop_preset')),
  kind          text NOT NULL,
  config_id     uuid NOT NULL,
  revision      integer NOT NULL,
  activated_at  timestamptz NOT NULL DEFAULT now(),
  activated_by  text NOT NULL,
  PRIMARY KEY (project_id, axis),
  FOREIGN KEY (config_id, kind)     REFERENCES config_documents(id, kind),
  FOREIGN KEY (config_id, revision) REFERENCES config_revisions(config_id, revision),   -- NO ACTION (1.2)
  CHECK ((axis, kind) IN (('prompt_pack','prompt_pack'), ('detection_profile','region_profile'),
                          ('vlm','vlm_endpoint'), ('open_vocab','open_vocab_set'),
                          ('ingest_profile','ingest_profile'), ('clustering_item','cluster_profile'),
                          ('clustering_region_box','cluster_profile'),
                          ('assist_loop_preset','assist_loop_preset')))
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
  search          jsonb NULL,                  -- SearchSettings (embeddings plan 5.6; 02d section 1)
  location_policy text NOT NULL DEFAULT 'store'  -- owner 2026-11-04 (RD-51 reversed): coordinates stored
                  CHECK (location_policy IN ('drop','store_coarse','store')),
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
  key         text NOT NULL CHECK (key IN ('cluster_refresh','region_drain','region_dependency_health')),
              -- research X-11: the region-cluster markers and last_clustering are replaced by
              -- cluster_partitions (02d section 2.2); cluster_refresh is the beat's scheduling clock
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
  attempt              integer NOT NULL DEFAULT 0,             -- numbers job_attempts rows only
  retry_count          integer NOT NULL DEFAULT 0,             -- failed deliveries only (research G-1; 02b section 3.1)
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
  pause_reason         text NULL CHECK (pause_reason IN ('user','audit','budget','project_paused','vlm_unavailable')),
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
  CHECK (status NOT IN ('running','cancelling') OR (owner IS NOT NULL AND lease_expires_at IS NOT NULL)),
  CHECK (status <> 'paused' OR pause_reason IS NOT NULL)
);
CREATE UNIQUE INDEX uq_jobs_one_active
  ON jobs (project_id, exclusive_group, scope) NULLS NOT DISTINCT
  WHERE exclusive_group IS NOT NULL AND status IN ('queued','running','paused','cancelling');
CREATE UNIQUE INDEX uq_jobs_idempotency
  ON jobs (project_id, kind, idempotency_key) NULLS NOT DISTINCT
  WHERE idempotency_key IS NOT NULL;
CREATE INDEX ix_jobs_project_recent ON jobs (project_id, kind, created_at DESC);
CREATE INDEX ix_jobs_live ON jobs (status) WHERE status IN ('running','cancelling');   -- rule a of the reaper; deliberately NOT on lease_expires_at (see below)
CREATE INDEX ix_jobs_queued ON jobs (queued_at) WHERE status = 'queued';
CREATE INDEX ix_jobs_interrupted ON jobs (updated_at) WHERE status = 'interrupted' AND auto_resume;
ALTER TABLE jobs SET (fillfactor = 70, autovacuum_vacuum_scale_factor = 0.02,
                      autovacuum_vacuum_threshold = 50, autovacuum_vacuum_cost_delay = 2);
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
                   'cancelled','paused','interrupted','lost','time_limit','yielded')),
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
CREATE INDEX ix_events_cursor ON events (project_id, xid, id);   -- the SSE cursor is (xid, id); per-project stream
CREATE INDEX ix_events_cursor_global ON events (xid, id) WHERE project_id IS NULL;   -- the global stream: `project_id = :p` never matches NULL (02b section 6.3)
CREATE INDEX ix_events_created ON events (created_at);
ALTER TABLE events SET (autovacuum_vacuum_scale_factor = 0.05, autovacuum_vacuum_insert_scale_factor = 0.05);

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

CREATE TABLE gpu_devices (                  -- research GPU S-6: inventory by UUID; deployment data, no RLS
  gpu_uuid           text PRIMARY KEY,          -- NVML UUID; CUDA numbers cards fastest first, so an index is ambiguous
  host               text NOT NULL,
  nvml_index         integer NOT NULL,
  cuda_index         integer NULL,
  name               text NOT NULL,
  compute_capability text NOT NULL,
  memory_total_mib   integer NOT NULL,
  numa_node          integer NULL,
  pcie_gen           integer NULL,
  pcie_width         integer NULL,
  hw_jpeg_decoder    boolean NOT NULL,
  driver_version     text NOT NULL,
  cuda_version       text NOT NULL,
  seen_at            timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gpu_reservations (             -- replaces training_gpus.lock (the arbiter is single-host)
  gpu_uuid       text PRIMARY KEY REFERENCES gpu_devices(gpu_uuid) ON DELETE RESTRICT,  -- research S-7
  holder_job_id  uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  reserved_at    timestamptz NOT NULL DEFAULT now()
);
```

`gpu_devices` is deployment data with no RLS, so its upsert needs no scope and runs in whichever
transaction the writer already holds (review RN-8: the worker start writes it under `deployment_tx`
from `src/tasks/runtime/gpu_app.py`, which is added to the `deployment_tx` allow-list for that, and
touches only this table and `worker_capabilities`). It is
upserted by every GPU worker at `worker_ready` for every NVML-visible card in its container and on every heartbeat, by
`op.periodic.serving_inventory` for every card a Triton server reports (Triton's GPU metrics carry
the `gpu_uuid` label), and in the same transaction as every `gpu_reservations` insert, so a card
with no Celery GPU worker can still be reserved; `openprocessor gpu plan` only reads; review RR-9,
00b section 5.5). Research S-12: a GPU worker's
`worker_capabilities` row names the card's `gpu_uuid`.

**Hot-row churn on `jobs`** (review fix, 2026-11-05). A running job is updated about once a second
(progress, [02b](02b_jobs_locking_events.md) section 3.1) and every 15 s (heartbeat). A normal `UPDATE`
writes a whole new row version and a new entry in every index; a heap-only tuple (HOT) update avoids
both, but only when no indexed column changes and the page has free room
([HOT updates](https://www.postgresql.org/docs/current/storage-hot.html)). Three rules follow:

1. **`ix_jobs_lease` is dropped.** It indexed `lease_expires_at`, which every heartbeat changes, so
   every heartbeat was a non-HOT update. Reaper rule a now reads the small set of live jobs through
   `ix_jobs_live` (a partial index on `status`, which changes only at a state transition) and filters
   `lease_expires_at < now()` on those rows: there are at most a few dozen live jobs.
2. The columns heartbeat, progress and checkpoint write (`heartbeat_at`, `lease_expires_at`,
   `progress`, `progress_updated_at`, `checkpoint`) are in **no index**, and those writes never set
   `updated_at`, `status`, `queued_at` or any other indexed column; a test
   (`tests/db/test_jobs_hot_updates.py`) runs 1,000 progress updates and asserts
   `pg_stat_user_tables.n_tup_hot_upd` is at least 95 % of `n_tup_upd` for `jobs`.
3. `fillfactor = 70` keeps 30 % of each page free for the new row versions
   ([CREATE TABLE storage parameters](https://www.postgresql.org/docs/current/sql-createtable.html#SQL-CREATETABLE-STORAGE-PARAMETERS):
   a lower fillfactor "makes heap-only tuple updates more likely"), and per-table autovacuum
   (`scale_factor` 0.02, `threshold` 50, `cost_delay` 2 ms) vacuums the churning table far sooner than the
   server default of 20 %.

**`events`: batched deletes, not day partitions (decided 2026-11-05).** Retention is 24 h. A partition
per day would let the reaper drop a whole partition instead of deleting rows, but it forces the
partition key into the primary key and every unique constraint, complicates the `(xid, id)` cursor across
partitions, and costs a partition-creation job. The volume does not need it: at most one progress row
per second per running job is about 90,000 rows a day per job, and with the default install's few
concurrent jobs the table holds under a million rows. Rule f deletes 10,000 rows per statement, and the
`events` autovacuum settings above reclaim the space. Revisit (a new decision, not a tuning knob) if the
measured table exceeds about 20 million live rows or the delete batch cannot keep up; the first sign is
`n_dead_tup` growing across ticks, which the monitoring overlay already scrapes.

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
  job_id                uuid NOT NULL UNIQUE REFERENCES jobs(id),  -- NO ACTION (1.2); status lives on the job
  dataset_export_id     uuid NULL REFERENCES dataset_exports(id) ON DELETE SET NULL,
  spec                  jsonb NOT NULL,
  class_names           text[] NOT NULL,      -- the training pin, in output order
  family                text NOT NULL,
  base_model            text NOT NULL,
  artifact_prefix       text NOT NULL,        -- checkpoints, run.log, manifest file
  manifest              jsonb NULL,           -- lineage: dataset hash, code versions, evaluation
  mlflow_run_id         text NULL,
  gpu_uuids             text[] NOT NULL DEFAULT '{}',   -- research X-6: by identity, not CUDA index
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

CREATE TABLE model_engines (                -- research GPU S-1..S-4 (one row per built engine)
  id                  uuid PRIMARY KEY DEFAULT uuidv7(),
  model_id            uuid NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  triton_model_name   text NOT NULL,
  precision           text NOT NULL CHECK (precision IN ('fp32','fp16','int8')),
  input_size          integer NOT NULL,
  input_contract      jsonb NOT NULL,   -- EngineInputContract (src/schemas/models.py): dtype, layout, size, color,
                                        -- normalize, resize, interpolation, fill, pad_split, max_side
  output_contract     jsonb NOT NULL,   -- EngineOutputContract: kind, shapes, dtypes
  build_profile       text NOT NULL DEFAULT 'default' CHECK (build_profile IN ('default','small')),
  build_settings      jsonb NOT NULL,   -- workspace MiB, opt/max batch, builder level, precision recipe
  build_stamp         text NOT NULL,
  onnx_sha256         text NULL,
  engine_sha256       text NULL,
  device_mib_measured integer NULL,     -- measured by loading the engine once (lane 5B)
  max_batch           integer NOT NULL,
  opt_batch           integer NULL,
  cuda_graphs         boolean NOT NULL DEFAULT false,
  parity_ref          text NULL,        -- the capture tag that approved the engine
  parity_report_key   text NULL,        -- storage key
  trt_version         text NOT NULL,
  gpu_arch            text NOT NULL,
  engine_storage_key  text NULL,
  build_job_id        uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  status              text NOT NULL CHECK (status IN ('building','ready','failed')),
  built_at            timestamptz NULL,
  UNIQUE (triton_model_name, trt_version, gpu_arch, build_profile)   -- an opt-in INT8 or small build is a new row
);

CREATE TABLE promotions (
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id         uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  model_id           uuid NOT NULL REFERENCES models(id),    -- NO ACTION (1.2)
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

Research seam rule (GPU S-1, S-13; 02d X-5): an engine row without `input_contract` is refused (the
column is `NOT NULL`), the export job refuses an ONNX input that differs from the declared contract
(`contract_mismatch`), and an embedding model version's `input_spec.image` must equal the
`input_contract` of the ready engine of its Triton image model.

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
| `jobs`, `job_attempts` | one row per start | terminal rows older than 90 days deleted, except kinds referenced by `training_runs`, `dataset_imports`, `combine_runs`, `bakeoff_runs`, `promotions`, `dataset_exports`, `assist_loop_runs`, `embedding_reindex_runs` (those keep their job: each references it without `ON DELETE SET NULL`) |
| `ingest_source_files` | one row per discovered file | `done` rows older than 30 days deleted |
| `config_runtime_reports`, `worker_capabilities`, `serving_reports` | one row per process and host, or per server and model | rows not seen for 1 day deleted |
| `gpu_devices` | one row per card | `seen_at` refreshed by the worker heartbeat and by `op.periodic.serving_inventory` (review RR-9); `DELETE ... WHERE seen_at < now() - interval '1 day' AND NOT EXISTS (SELECT 1 FROM gpu_reservations r WHERE r.gpu_uuid = gpu_devices.gpu_uuid)`, so a referenced row never fails the batch |
| `cluster_partitions` and their clusters and memberships ([02d](02d_research_seam_schema.md) 2.2) | about 150 B per membership row, about 150 MB per 1M-item partition | keep the serving, the previous serving and at most 2 `ready` candidates per key; older `discarded` or superseded partitions deleted after 30 days, 10,000 rows per statement; a superseded `ready` partition is set `discarded` first, then its children go in batches and the partition row last (review RN-1) |
| `assist_loop_proposals`, `assist_loop_selections` (02d 2.4) | one row per asked item | rows of runs finished more than 90 days ago deleted (the node rows keep the counts); a run and its children stay while one of its label operations can be undone |
| `vlm_usage_hourly` | one row per project, endpoint, source and hour | 400 days |
| `near_dup_runs` and their groups and members (02d section 3) | one members row per image and item per run | the current run, the two newest superseded runs per project and level, and every run a `holdout_freezes` row references (review RR-22); older runs deleted by beat |
| `near_dup_pending` | one row per changed member | deleted when its rescore drains it |
| `dataset_uploads` | one per upload | `expires_at` passed: blob and row removed |
| `dataset_import_images`, `label_operation_members` | one per image or item | kept with their parent (undo needs them) |

The retention constants live in `src/config/retention.py` (constants, not env, until a deployment
needs to tune one).
