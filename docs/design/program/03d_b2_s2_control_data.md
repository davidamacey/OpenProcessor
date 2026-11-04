# B2 stage 2: control data moves to Postgres, and the 0.6.0 upgrade step

| Field | Value |
|---|---|
| Status | plan, not started |
| Depends on | stage 1 merged into `int/b2-control-plane` ([03a](03a_b2_w1_queue_infra.md)) |
| Design | [02](02_control_plane_database.md) sections 3-4, [02a](02a_postgres_schema.md) sections 2-4 and 6, [02c](02c_migration_ops_tests.md) sections 2-3 |
| Lanes | 2A, 2B, 2C, 2D, 2E in parallel; then 2F |
| Merge order | 2A, 2B, 2C, 2D, 2E, 2F |
| Runs alongside | stage 3 lanes code from the stage-1 head in parallel (disjoint files) and merge after this stage |
| Compose project | none in the lanes (unit and `db` tests only) |

After stage 2 every control record is read and written in Postgres, OpenSearch keeps only `images`
and `items` per project, and the 0.6.0 upgrade step exists and is tested. Job execution is unchanged
in this stage (in-process launches and leases stay until stages 3-5).

Common rules for every lane: [03_b2_overview.md](03_b2_overview.md) section 4. A file that a lane
needs but does not own goes to the coordinator; the lane does not edit it.

---

## Lane 2A `b2/projects-pg`: projects, slugs, index ownership, pause

**Owns:** `src/db/repos/projects.py`; `src/config/projects.py`, `src/config/project_context.py`,
`src/config/curation_accessors.py`; `src/services/projects/{registry,bootstrap,lifecycle,delete,delete_paths,delete_models,mlflow_cleanup,guard,guard_policy,script_binding}.py`;
`src/routers/curation/projects.py`, `src/routers/curation/_project_deps.py`,
`src/routers/curation/_project_pause.py`, `src/schemas/curation/projects.py`;
`scripts/curation/worker/fairness.py`; their tests under `tests/projects/` and `tests/db/repos/test_projects_repo.py`;
the fake it retires: `tests/_fake_project_registry.py` and the tests built on it (rewritten as `db`
tests where they check a constraint, R-14).

| ID | Task | Red test | Control |
|---|---|---|---|
| 2A-1 | `src/db/repos/projects.py`: create (slug row, project, indexes, registry, settings and keymap rows, counter bump, one transaction), get by slug, list, patch with `expected_revision`, archive and unarchive (allowed `from` states as `projects.py:200-202`), mark deleting, finish delete (retire slug, delete row, bump counter), the ownership map query | `tests/db/repos/test_projects_repo.py` (`db`): duplicate slug -> `slug_taken`; a retired slug -> `slug_retired`; 10 concurrent creates of one slug -> one row; stale revision -> conflict; finish delete cascades every control row | list and get return the created project |
| 2A-2 | the registry snapshot (`registry.py`) reads Postgres: poll `deployment_counters('projects_revision')` every 1 s (unchanged interval), reload on change; transitional statuses re-read by id; `op_projects`, `meta:projects_revision` and `bind_registry_admin` deleted | `tests/projects/test_registry_snapshot_pg.py`: a create in one session is visible to a second snapshot within one poll; no OpenSearch call (a fake client that raises on any call) | a steady snapshot does one indexed read per poll |
| 2A-3 | the guard's ownership map from `project_indexes`; `guard_policy._check_targets` loses the `op_projects` and `op_global_configs` branches (those indexes no longer exist); `make_script_opensearch` builds its map from Postgres | `tests/projects/test_guard_ownership_pg.py`: a request naming another project's index raises `CrossProjectAccess`; a request naming `op_projects` raises (unowned `op_` index while bound) | the bound project's own indexes pass |
| 2A-4 | lifecycle routes over the repository: create (Postgres first, then the OpenSearch indexes; `building` until both exist; a failure leaves `failed`), rename and describe, archive, delete (dry run, confirm; the finish still launched as today until stage 4C) | `tests/projects/test_lifecycle_routes_pg.py`: create then GET; a concurrent PATCH with a stale revision is 409 | the existing lifecycle tests |
| 2A-5 | pause: `POST .../pause` and `.../resume` write `projects.paused_at`; the workers (`fairness.py`) read the project row; `pipeline_paused.flag` deleted | `tests/projects/test_pause_pg.py`: pause makes the worker skip the project; no flag file written | resume clears it |
| 2A-6 | `project_context.py`: storage prefix from `projects.storage_prefix` (stage 1's storage keys); the `.project` marker stamping of directories kept only where a directory is still created | `tests/projects/test_project_context_pg.py` | - |

**2A done when:** `rg -n "op_projects|OP_PROJECTS_INDEX|projects_revision'|bind_registry_admin|pipeline_paused" src scripts` lists only `src/services/upgrade/`; tests green.

## Lane 2B `b2/classes-pg`: the class registry

**Owns:** `src/db/repos/classes.py`; new `src/services/classes/{__init__,registry,resolve}.py`;
`src/clients/curation_opensearch/registry.py`, `registry_mirror.py`, `registry_transfer.py` (all
deleted) and the registry models in `src/clients/curation_opensearch/__init__.py`;
`src/services/curation/class_ensure.py`; `src/routers/curation/{classes,classes_write,classes_merge,classes_crops,_class_hotkeys,_class_models,class_seed}.py`;
every call site of the registry API (`rg -ln "get_registry|ensure_class|add_class|ClassRegistry" src scripts`)
outside files another stage-2 lane owns, including the scripts no lane owned before (review fix
R-18): `scripts/curation/repair_merged_active_classes.py`, `scripts/curation/seed_class_registry.py`,
`scripts/curation/seed_live_harness.py` (the live harness seed would break otherwise); tests under
`tests/classes/`; the fake it retires: `tests/_registry_doc_fake.py` (R-14).

| ID | Task | Red test | Control |
|---|---|---|---|
| 2B-1 | repository: list, get by id and by name, `ensure_class` (registry row `FOR UPDATE`, re-check, allocate), `add_class` (409 on an existing name), update (name, group, notes, hotkey), deprecate, restore, merge (sources deprecated with `merged_into`), revision bump on every write; the name normalization re-read from the current registry and pinned ([02a](02a_postgres_schema.md) section 3) | `tests/db/test_class_registry.py` (`db`, [02c](02c_migration_ops_tests.md) section 7): 10 concurrent `ensure_class` for 50 names; duplicate hotkey -> conflict; `merged_into` must be deprecated; review fix R-3: "Car", "car", "car " are one class and "traffic light" equals "traffic_light" (one `name_key`); a deprecated "bus" and a new active "bus" coexist and `ensure_class("Bus")` picks the active one; a name with no slug-safe character is refused | sequential adds give ids 0, 1, 2 |
| 2B-2 | `src/services/classes/resolve.py`: name to id and id to name, cached per process by `class_registries.revision`; every item writer that sets a class goes through it (the D-47 single writer path) | `tests/classes/test_resolve_single_path.py`: AST scan, no module outside `src/services/classes/` writes `class_name` onto an item without calling `resolve` (planted fixture fails) | the real tree passes |
| 2B-3 | rename and merge write a `search_projection_tasks` row in the same transaction; until stage 4B adds the `class_projection_sync` job, the item rewrite stays what the route does today (read it: `classes_merge.py:128-157`) | `tests/classes/test_rename_writes_outbox.py` (`db`) | - |
| 2B-4 | the class list route joins counts from one terms aggregation (`items.class_id`) plus a validated filter aggregation; no stored counts; the `classes` mirror writes and `POST /curation/projects/{project}/classes/sync_to_opensearch` deleted (contracts) | `tests/classes/test_class_counts_from_items.py` (fake OpenSearch aggregation) | names and ids unchanged |
| 2B-5 | error codes `registry_not_migrated`, `registry_full`, `registry_conflict` deleted (D-57), with their models and tests; the 10,000 cap gone; contracts regenerated | `tests/classes/test_no_registry_cap.py`: 10,001 classes are accepted (`db`) | - |
| 2B-6 | `class_registry.json` is written only into export snapshots and training pins (the six readers in [02](02_control_plane_database.md) section 4.3 read the snapshot they were given) | `rg`-equivalent test: no runtime reader of `OP_PROJECTS_DATA_ROOT/<slug>/class_registry.json` | export snapshot tests unchanged |

**2B done when:** `rg -n "registry_mirror|registry_transfer|registry_not_migrated|registry_full|registry_conflict|MAX_REGISTRY_CLASSES" src` is empty; contracts regenerated.

## Lane 2C `b2/config-pg`: config store, VLM endpoints, settings, keymap, view state

**Owns:** `src/db/repos/{config,settings}.py`; `src/services/config_store/**`;
`src/services/curation/{keymap,settings_doc,policy_doc_store}.py`; the VLM endpoint, activation and
snapshot modules (`rg -ln "def (save_endpoint|probe_endpoint|activate_vlm|local_vlm)" src/services`);
`src/services/curation/clustering/refresh_state_store.py`; `src/services/curation/umap_state.py` and
the projection-state read and write in `embedding_viz_read.py`; `scripts/curation/worker/runtime.py`
(runtime reports); the config routers (`src/routers/curation/{settings,vlm,vlm_activation,vlm_catalog,config_test,config_vocabulary,_models_vlm,_vlm_endpoint_models}.py`
and the prompt-pack, region-profile, ingest-profile and keymap routers:
`rg -ln "prompt_packs|region_profiles|ingest_profiles|keymap" src/routers/curation`); tests under
`tests/config_store/`.

| ID | Task | Red test | Control |
|---|---|---|---|
| 2C-1 | config documents and revisions: save with `expected_revision`, immutable revisions, `cloned_from`, list and get (current and by revision), validate (pydantic body models unchanged), epoch bump | `tests/db/test_config_repo.py` (`db`): two saves with one expected revision -> one wins, one `RevisionConflictError`; an `UPDATE` of a revision row raises (trigger); a revision number is never reused | load, save, read back |
| 2C-2 | activations: `activate` with `expected_active`, the event row, `rollback` to the previous event, the axis/kind composite check | `tests/db/test_activation_repo.py` (`db`): a stale `expected_active` is 409; activating a `region_profile` on the `vlm` axis is refused by the database | rollback restores the previous revision |
| 2C-3 | VLM endpoints deployment-wide (`project_id` NULL), probes per `name@revision`, `local_vlm_desired` in `deployment_settings`; keys still referenced as `secret:<slug>` (never stored) | `tests/config_store/test_vlm_endpoints_pg.py` | the probe-cap behaviour unchanged |
| 2C-4 | polling: `store_poll.py` reads `projects.config_epoch` and `global_config_epoch`; runtime reports go to `config_runtime_reports` | `tests/config_store/test_poll_epoch.py`: an activation in one session reloads a second process's store within one poll; no OpenSearch call | an unchanged epoch reloads nothing |
| 2C-5 | settings (`defaults`, `ingest_policy`, `vlm_policy`, `vlm_scope_reps`) and keymap with `expected_revision` on every write; the `defaults` write that had no OCC (`settings_doc.py:122-129`) answers 409 on a stale revision (contracts: the route gains `expected_revision` where it lacked it) | `tests/config_store/test_settings_occ.py` (`db`): two `PUT /settings` with the same revision: one 409. Red today (last write wins) | a single write succeeds |
| 2C-6 | cluster-refresh state in `project_state`; UMAP view state in `projection_states`, the reducer bytes through the storage seam (`projects/<slug>/umap/<state_id>.bin`); the `umap_state` reader and writer deleted | `tests/config_store/test_projection_state_pg.py` | the viz read route returns the same shape |

**2C done when:** `rg -n "doc_type|meta:config_revision|activation_event:|runtime:|keymap:default|op_global_configs|OP_GLOBAL_CONFIGS_INDEX" src` lists only `src/services/upgrade/`; contracts regenerated.

## Lane 2D `b2/records-pg`: holdout freezes, label-operation records, project markers

**Owns:** `src/db/repos/records.py`; `src/services/curation/holdout.py`, `holdout_freeze.py`;
`src/services/curation/label_ops/store.py` (records and members only: the lease and the in-request
execution stay until stage 4A); `src/services/curation/region_drain.py`,
`region_dependency_health.py`, `clustering/last_run.py`; tests under `tests/records/`. 2D-4 reads the
items' near-duplicate fields through the existing OpenSearch client (no file of another lane).

| ID | Task | Red test | Control |
|---|---|---|---|
| 2D-1 | holdout freezes: freeze, history, current, the "freeze fails closed" rule kept (a database error refuses the freeze); membership stays the items' `test_holdout` flag | `tests/records/test_holdout_pg.py` (`db`): two concurrent freezes -> one current row (partial unique index) | history lists both |
| 2D-2 | label-operation records and members in Postgres (`label_operations`, `label_operation_members`); the served `interrupted` and `undo_interrupted` states unchanged | `tests/records/test_label_ops_pg.py` (`db`): a 10,000-member op writes in batches of 5,000 (trap T-14); undo reads the prior state from the table | the existing label-op route tests |
| 2D-3 | `project_state` keys `region_drain`, `region_dependency_health`, `last_clustering` with OCC on `revision` | `tests/records/test_project_state_pg.py` | read after write |
| 2D-4 | #473 part c (folded in, [00b](00b_traceability.md) section 3): the freeze checks near-duplicates of the held-out items (the items' near-duplicate group fields); owner decision 2026-11-04: the whole near-duplicate group goes into the holdout together, and the freeze record counts the moved items | `tests/records/test_holdout_near_duplicates.py` (`db` + fake OpenSearch): an item near-duplicate to a held-out item ends on the holdout side; red today (the freeze ignores groups) | an item with no near-duplicate is unchanged |

**2D done when:** `rg -n "test_holdout/|label_ops/.*state.json|members.json|last_clustering.json|dependency_health.json" src` lists only `src/services/upgrade/`.

## Lane 2E `b2/upgrade-060`: the one-step upgrade and its rollback

**Owns:** new `src/services/upgrade/**`; new `scripts/db/{upgrade_from_060,rollback_to_060,make_v060_fixture,verify}.py`;
`scripts/lib/cli_db.sh` (`db verify`); new `scripts/lib/upgrade_060.sh`; `setup-openprocessor.sh`
(the upgrade and rollback hooks; removing the B1 registry-migration hooks);
`scripts/lib/registry_migration.sh` and `scripts/curation/migrate_class_registry.py` (deleted);
`scripts/lib/cli_core.sh` (`curation migrate-registry` removed); `src/services/projects/legacy_repair.py`
(moved into the upgrade as a step, then deleted); `src/db/schema_check.py` (the gate, 2E-7);
`tests/upgrade/**`; `tests/installer/test_upgrade_060_hook.py`.

Uses the ORM models of stage 1 directly (not the stage-2 repositories, which are in flight), under
`deployment_tx` (the upgrade module is in the allowlist).

| ID | Task | Red test | Control |
|---|---|---|---|
| 2E-1 | `make_v060_fixture.py` and the committed fixture ([02c](02c_migration_ops_tests.md) section 2.4) | - | - |
| 2E-2 | `export_bundle` (unguarded read-only admin client: `GET`, `search` with `search_after`, `_mapping`; files read only), gzip from the standard library (R-31); the training-run logs and every other `/jobs` file a kept row points to copied to storage keys and listed (R-9e); `MANIFEST.json` with counts and sha256 | `tests/upgrade/test_export_bundle.py`: the bundle of the fixture has the expected counts; the client issues no write | - |
| 2E-3 | the `upgrade_state` gate first (R-9a, [02c](02c_migration_ops_tests.md) section 2.2); `load` in one transaction, the ledger tables through psycopg `COPY` (R-25); `verify`; `delete_sources` (idempotent; sets `upgrade_state` completed); `--reset` after a committed load whose verify failed, refused after `delete_sources` (R-9b); `upgrade_runs` bookkeeping; the legacy-repair step per project | `tests/upgrade/test_load_from_060.py`, `test_delete_sources_idempotent.py` (`db`); a forced verify mismatch deletes nothing (trap T-25); `tests/upgrade/test_upgrade_gate.py` and `test_load_large_ledger.py` (1,000,000 ledger rows) of 02c section 2.4; `tests/upgrade/test_leftover_job_files_removed.py`: a 0.6.0 fixture with a leftover `trigger.json` and `running.lock` ends with neither (carried from the original 2E-1 step 6) | a re-run after a load failure gives the same rows |
| 2E-4 | `rollback_to_060.py` in the order of [02c](02c_migration_ops_tests.md) section 2.3 (OpenSearch only, recreate indexes from the bundle mappings, re-index, restore files and run logs, rename `class_registry.json.migrated-0.7.0` back, verify, mark the Postgres data abandoned after a confirmation, then 0.6.0 services; review fix R-9c, R-9d) | `tests/upgrade/test_rollback_bundle_roundtrip.py` | - |
| 2E-5 | installer: `upgrade` from 0.6.0 runs backup -> `db-migrate` -> `upgrade_from_060` -> start; prints the report; `--rollback` runs `rollback_to_060`; the B1 registry hooks and CLI removed | `tests/installer/test_upgrade_060_hook.py`: the order of calls (fake compose); a failed verify stops before `up` with the report path | a fresh install never runs the step |
| 2E-6 | `openprocessor db verify` ([02c](02c_migration_ops_tests.md) section 6) | `tests/upgrade/test_verify_reports_class_drift.py` | a clean fixture reports nothing |
| 2E-7 | the start gate (R-9a): `src/db/schema_check.py::check_control_plane()` (1A's module; 2E owns it in this wave) also refuses with `upgrade_incomplete` while `upgrade_state` is `in_progress` or `op_projects` still exists | `tests/upgrade/test_upgrade_gate.py` | a completed upgrade starts |

**2E done when:** the four upgrade tests and the installer test are green; `rg -n "migrate-registry|registry_migration|migrate_class_registry|legacy_repair" src scripts setup-openprocessor.sh openprocessor` lists only `src/services/upgrade/`.

## Lane 2F `b2/opensearch-control-removal` (cut after 2A-2E merged)

**Owns:** `src/clients/curation_opensearch/{bodies_other.py,lifecycle.py}`,
`src/config/curation_roles.py`, `src/services/curation/index_bootstrap.py`, the capacity and shard
code (`rg -ln "shards_after_create|shard_budget" src`), the remaining readers of the deleted roles
(`rg -ln "labels_confirmed_index|classes_index|umap_state_index|configs_index|settings_index" src`),
`src/config/retired_env.py` (the stage's retirements), `env.template` (the same), the OpenSearch
bootstrap in `src/core/dependencies.py` and new `scripts/opensearch/migrate.py` plus its call in
`scripts/db/migrate.py` (R-7; 1A merged it in W2.1), `src/services/upgrade/**` for 2F-5 (2E has
merged), tests under `tests/opensearch/`.

| ID | Task | Red test | Control |
|---|---|---|---|
| 2F-1 | delete the `configs`, `classes`, `umap_state`, `labels_confirmed` and legacy settings bodies, roles and bootstrap migrations; new projects get two indexes; the stale comment at `curation_roles.py:43-44` goes with the roles | `tests/opensearch/test_project_index_roles.py`: a new project creates exactly `images` and `items` | the mapping fingerprint test for the two roles |
| 2F-2 | capacity: two shards per project in the budget and `shards_after_create` | `tests/opensearch/test_shard_budget_two_per_project.py` | the existing capacity 409 test |
| 2F-3 | retire `OP_PROJECTS_INDEX`, `OP_GLOBAL_CONFIGS_INDEX` and every env var only the deleted code read (`removed:` entries naming the replacement) | `tests/config/test_retired_env.py` additions | `tests/test_env_surface.py` |
| 2F-4 | review fix R-7: index creation, mapping migrations and the kNN warm-up leave the API lifespan (they ran in every uvicorn worker of every replica, and again after each `--limit-max-requests` recycle: `src/core/dependencies.py:140-158`) and run once in the one-shot `db-migrate` step after Alembic (`scripts/opensearch/migrate.py`, idempotent; re-check RC-1 under D-69: every service already waits on `db-migrate`, so one migrate step serves both stores): it reads the projects from the Postgres `projects` table, does only the global indexes while the 0.6.0 data is not loaded (`upgrade_state` `in_progress` or `op_projects` present), and stamps each mapping's `_meta` with the existing `SCHEMA_META_KEY` fingerprint of `src/services/curation/index_bootstrap.py` ([02c](02c_migration_ops_tests.md) section 1.2); the API lifespan reads every stamp in one `_mapping` call with `filter_path` and refuses to start with `opensearch_mapping_not_current` naming the index when one is wrong or missing; the check replaces the bootstrap inside `src/core/dependencies.py` (the function the lifespan already calls), so 2F does not edit `src/app/lifespan.py`, which 3B owns in this wave | `tests/opensearch/test_lifespan_no_bootstrap.py` (AST: no `create_all_indexes`, `ensure_indexes` or warm-up call in the lifespan); `tests/opensearch/test_migrate_idempotent.py` (fake OpenSearch: two runs, the second issues no write); `tests/opensearch/test_migrate_waits_for_upgrade.py` (with `op_projects` present only the global indexes are touched; after the load the project indexes are stamped); `tests/opensearch/test_mapping_gate.py` (a project index with a stale stamp refuses the start naming the index) | a fresh run creates the global and project indexes |
| 2F-5 | review fix R-18: 2E's `load` switches from the ORM models to the stage-2 repositories (so class names get their one normalization path, R-3) | `tests/upgrade/test_load_from_060.py` stays green; an AST test that `src/services/upgrade/` writes `project_classes` only through `src/db/repos/classes.py` | the fixture round trip |

## Stage 2 exit criteria

- every lane's tests green on the merged branch; `-m db` green; `make contracts-check` green;
- `rg -n "op_projects|op_global_configs|class_registry'|lease:class|doc_type" src` lists only
  `src/services/upgrade/`;
- a project created on the merged branch has two OpenSearch indexes and its control rows in Postgres
  (coordinator check in the `-m db` run: `tests/projects/test_lifecycle_routes_pg.py`).

## Lane briefs

Use the common block of [03a](03a_b2_w1_queue_infra.md) ("Lane brief"), with: "B2 stage-2 lane",
"cut from int/b2-control-plane after stage 1 merged" (2F: "after 2A-2E merged"), the reading list
`03d_b2_s2_control_data.md` (your lane) plus the 02 sections it cites, and "Edit only the files your
lane owns; a file you need but do not own goes to the coordinator." Compose project: none. GPU: none.
