# B1 track 1B: class registry document with OCC, and the upgrade migration

| Field | Value |
|---|---|
| Status | plan, not started |
| Batch / release | B1, part of the single v0.7.0 release |
| Depends on | B0 merged; `int/b1-correctness` cut; track 1A merged before 1B merges (1A owns the `class_registry` mapping key, `02a_b1_job_leases.md` task 1A-1) |
| Decisions applied | D-9 (J-REG and upgrade), owner decision 3 (supported upgrade 0.6.0 -> 0.7.0 directly; no intermediate release), class identity is the name |
| Source IDs | the former master plan (rev 4, in git history) 3a (Q-P1.3), 3c (Q-D3), 3d, 6.2, 7.1 (1B row, risks, rollback, upgrade impact); `codebase_audit_plan.md` 4.7 CC-2 (fix, traps, tests); `task_queue_plan.md` section 3 D3, Phase 1 P1.3 (superseded); issue N2 (`09_github_actions.md`) |
| Size | L |
| Lane | branch `b1/registry`, worktree `<repo>-wt-b1-registry`, compose project `op-b1-registry` |

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read only, with `rg`).
Consolidated 2026-11-03.

## 1. Goal

Two concurrent class writes (two users, or a user plus an import, a combine or the open-vocab
pass) can no longer lose a class or give two classes one id. The authority moves from
`class_registry.json` (a read-modify-write file with no version check) to one OpenSearch document
per project written with optimistic concurrency. The file becomes an export snapshot only. A 0.6.0
install upgrades directly to 0.7.0 with every class id preserved, and refuses class-dependent work
(fail closed) until its registry is migrated.

## 2. Preconditions

- Lane worktree: `git worktree add <repo>-wt-b1-registry -b b1/registry int/b1-correctness`.
- Starts in parallel with 1A (justified in `02_b1_overview.md` section 5). 1B never edits
  `bodies_other.py` or `index_bootstrap._migrations()`; 1A adds the `class_registry` object key to
  both bodies. Before 1B merges, the lane rebases onto `int/b1-correctness` after 1A's merge and
  re-runs its targeted tests.
- On folded projects (`configs` index, `dynamic: false`) the doc works without the mapping key
  (unmapped fields are kept in `_source`). The key matters for legacy settings indexes (dynamic
  mapping), so the live registry test runs only in the batch sweep, after 1A.

## 3. Documented mechanisms

| Mechanism | Official doc |
|---|---|
| Index with `if_seq_no` / `if_primary_term` (409 on mismatch); `op_type=create` | https://docs.opensearch.org/latest/api-reference/document-apis/index-document/ |
| Realtime get (returns `_seq_no`, `_primary_term`; `_source=false` supported) | https://docs.opensearch.org/latest/api-reference/document-apis/get-documents/ |
| `http.max_content_length` default 100 MB (size bound argument) | https://docs.opensearch.org/latest/install-and-configure/configuring-opensearch/network-settings/ |
| Bulk API item errors (mirror write) | https://docs.opensearch.org/latest/api-reference/document-apis/bulk/ |
| Python `os.replace` (atomic rename, for the reverse step) via `src/utils/atomic_write.py` | https://docs.python.org/3/library/os.html#os.replace , https://docs.python.org/3/library/tempfile.html |
| FastAPI dependencies (one snapshot per request) | https://fastapi.tiangolo.com/tutorial/dependencies/ |

Repo precedents: `src/clients/occ.py` (`_fetch_with_version`, the conflict test,
`_RETRY_BACKOFF_SEC`); the project registry's `ensure_fresh` per-process snapshot (audit Appendix B
found it correct); `config_store` revisioned documents (`src/services/config_store/`);
`tests/projects/conftest.py::FakeRegistryOpenSearch` (enforces `op_type` and `if_seq_no`).

## 4. Current state (recounted)

- `src/clients/curation_opensearch/registry.py` (316 lines): `ClassRegistry` with an mtime cache,
  `_atomic_write` (writes a timestamped backup `class_registry.<ts>.json` next to the file on every
  write, then `atomic_write_bytes`), `next_id`, `validate_id`, `get`, `add_class` (raises
  `ClassRegistryError` on a duplicate non-deprecated name), `rename_class`, `merge_class`,
  `set_deprecated`, `sync_to_opensearch` (mirror to the per-class `classes` index), and
  `get_class_registry()` (one instance per project per process). File models `RegistryClassEntry`
  (`class_id`, `class_name`, `group`, `sample_count`, `validated_count`, `added_at`, `deprecated`,
  `notes`, `merged_into`, `hotkey_letter`) and `ClassRegistryFile` (`version`, `updated_at`,
  `classes`) live in `src/clients/curation_opensearch/__init__.py`.
- `rg -n "class_registry_path|class_registry\.json" src scripts docker`: **75 lines in 33 files**
  (the plan's "about 75" holds; the ledger's "about 140" counts both patterns below).
- `rg -n "get_class_registry\(|ClassRegistry\(" src scripts docker`: **60 lines in 49 files**
  (48 outside `registry.py`).
- Mutator call sites (`rg -n "\.(add_class|rename_class|merge_class|set_deprecated)\(" src scripts`):
  11, plus direct `_atomic_write` writes outside `registry.py`: `routers/curation/classes_write.py`
  (2: hotkey after create, group/hotkey on update), `routers/curation/keymap.py` (2:
  `_unbind_classes`, `_restore_classes`), `scripts/curation/seed_class_registry.py` (1).

| Writer | Site |
|---|---|
| `POST /classes` (create + optional hotkey, two writes) | `classes_write.py::create_registry_class` (also used by the new-class-proposal resolve route) |
| `PUT /classes/{id}` (rename, then group/hotkey, two writes) | `classes_write.py::update_class` |
| deprecate / restore | `classes_write.py` (`set_deprecated` x2) |
| merge | `classes_merge.py` (`merge_class`) |
| keymap save unbinds class hotkeys, and restores the snapshot on failure | `keymap.py::_unbind_classes`, `_restore_classes` |
| open-vocab and VLM class creation | `services/curation/class_ensure.py` |
| dataset import prepare / undo | `dataset_import/prepare.py` (`add_class`), `dataset_import/undo.py` (`set_deprecated`) |
| combine | `projects/combine/execute.py` (`add_class`) |
| project clone | `services/projects/clone.py` (copies the source file to the target) |
| scripts | `seed_class_registry.py`, `seed_live_harness.py`, `repair_merged_active_classes.py` |

- Readers (48 files) by area:
  - routers: `_common.py` (the `_registry_dep` dependency), `assist.py`, `bakeoff.py`,
    `classes.py`, `classes_crops.py`, `classes_merge.py`, `class_seed.py`, `config_test.py`,
    `config_vocabulary.py`, `crops_write.py`, `health.py`, `ingest_policy.py`, `keymap.py`,
    `_models_class_mapping.py`, `review.py`, `review_proposals.py`, `review_resolve.py`,
    `stats.py`, `vlm.py`;
  - services: `config_store/profile_dependency_checks.py`, `curation/assist/prepare.py`,
    `class_ensure.py`, `class_registry_names.py`, `dataset_import/project_source.py`,
    `eval_datasets.py`, `export.py`, `export_single_class.py`, `index_bootstrap.py`,
    `open_vocab_run.py`, `open_vocab_sweeper.py`, `probe_predictions.py`, `region_class.py`,
    `vlm_class_batch.py`, `projects/clone.py`, `projects/combine/{execute,plan}.py`,
    `projects/stats.py`, `training/{jobs,preflight_checks,preflight_run,promote_gate}.py`;
  - scripts and workers: `scripts/curation/worker/state.py` (detection worker),
    `_vlm_activity.py`, `reclassify_after_registry_growth.py`, `repair_merged_active_classes.py`,
    `seed_live_harness.py`; path readers `scripts/curation/bakeoff/class_map.py`,
    `seed_class_registry.py`.
  - Most call `.load()` inside sync helpers. Some read **another** project's file by path:
    `combine/plan.py`, `dataset_import/project_source.py`, `projects/stats.py`,
    `probe_predictions.py`.
- File snapshots that must stay (they are export artifacts, not the authority):
  `services/curation/export_writers.py::write_class_registry` (export `class_registry.json` with
  `export_id_map`), `dataset_import/op_export.py` (reads an export's `class_registry.json` by name),
  `export_models.py` (artifact names), the trainer `docker/trainer/dataset_prep.py` and
  `subset_dataset.py` (read the **export's** file, have no OpenSearch client), and
  `services/training/jobs.py::_pin_registry_snapshot_sync` (pins a job-scoped copy at train start).
- `ProjectResources.class_registry_path` (`src/config/projects.py`) is persisted in project records
  (`services/projects/registry.py`) and shown by `routers/curation/_project_views.py::resources_wire`
  (a free-form dict, not in `contracts/`, not read by Cropwright).
- 0.6.0 used the same layout (`git show v0.6.0:src/config/projects.py`: SETTINGS folded onto
  configs, `class_registry_path = data_root / 'class_registry.json'`), so the 0.6.0 -> 0.7.0
  migration is the same as from any 0.7 pre-release.
- `setup-openprocessor.sh` is 3112 lines, exactly its `file_size.txt` baseline; it sources
  post-install libraries from `scripts/lib/` (`model_setup.sh`, `install_summary.sh`, ...).
  `do_rollback` restores installer files from `backups/` only; project data is untouched.
- `scripts/release/acceptance_phases.sh::ph_installer_upgrade` upgrades, checks `/health`, then
  runs `--rollback` and checks `/health` again.

## 5. Design (J-REG, D-9)

### 5.1 The document

- Id `class_registry`, in the project's SETTINGS role index (`index_name(cfg, IndexRole.SETTINGS)`,
  folded or legacy, as in `02a_b1_job_leases.md` 5.1). Mapping key
  `"class_registry": {"type": "object", "enabled": false}` (added by 1A-1).
- Source:
  `{"class_registry": {"version": 1, "classes": [<RegistryClassEntry>...], "next_class_id": int,
  "registry_revision": int, "updated_at": iso, "migrated_from_file_at": iso | null,
  "reversed_to_file_at": iso | null}}`. Entries keep today's `RegistryClassEntry` fields, so the
  export snapshot and the trainer see the same shape.
- `MAX_REGISTRY_CLASSES = 10_000` (module constant; about 1-2 MB of `_source`, far below the 100 MB
  default `http.max_content_length`). `add_class` past it raises 409 `registry_full`. A 100k-class
  project is out of scope (risk M22).

### 5.2 Store (`src/clients/curation_opensearch/registry.py`, rewritten)

- `ClassRegistry` becomes the document store. Delete the file I/O (`_read_disk`, the mtime cache,
  `_atomic_write`, the per-write timestamped backups).
- Reads: `async def snapshot(client) -> ClassRegistryFile`. A per-process cache keyed by the doc's
  `(_seq_no, _primary_term)`, checked with a realtime `get(..., _source=False)`; the full source is
  fetched only when the version changed. The snapshot is immutable (a frozen copy); sync helpers take
  it as an argument.
- Absence rules (fail closed):
  - doc present: it is the authority (a leftover file is ignored and logged once);
  - doc absent, `class_registry.json` present: raise `RegistryNotMigratedError`
    (a `RequestRejectedError` with 409 and code `registry_not_migrated`, message naming
    `openprocessor curation migrate-registry --project <slug>`);
  - doc absent, file absent (a new project, or a 0.6.0 project that never had classes): create the
    empty doc with `op_type='create'` (a 409 means another worker created it: re-read).
- Mutations: one `async def _mutate(client, fn)` used by every mutator. Realtime `get` with version,
  apply the pure function `fn` to a deep copy, `index(..., if_seq_no, if_primary_term,
  refresh=False)`, `registry_revision += 1`. On 409, re-read and re-apply, **at most 5 attempts**
  (backoff from `occ._RETRY_BACKOFF_SEC`), then 409 `registry_conflict`. Any other error
  propagates (503 path).
- Mutators: `add_class(name, group, notes, hotkey_letter=None)` (refuses a duplicate active name:
  the `POST /classes` 409 stays), `ensure_class(name, group, notes) -> (id, created)` (idempotent by
  name: an existing non-deprecated class with that name returns its id; used by every automated
  writer, so a retry never duplicates), `rename_class`, `merge_class`, `set_deprecated`,
  `update_class(class_id, name=None, group=None, hotkey_letter=None)` (one mutation for the whole
  `PUT`), `set_hotkeys({class_id: letter | None})` (keymap unbind and its compensating restore).
- Refinement of master 6.2 ("`add_class` is idempotent by name"): the route keeps its documented
  409 on a duplicate name, so idempotency lives in `ensure_class`, which the automated writers
  (class_ensure, import, combine) call. Both are retry-safe because the function re-runs on the
  re-read state.
- Ids: `new_id = next_class_id`, then `next_class_id += 1`. Ids are never reused (merges and
  deprecations keep entries).
- Mirror: after each successful write, `sync_to_opensearch` writes the changed entries to the
  per-class `classes` index (bulk, item errors checked with `failed_bulk_items`). A mirror failure
  is logged with counts; the bootstrap self-heal (`index_bootstrap.py`) re-syncs from the doc.
- Async: combine, import and `class_ensure` `await` the store; no `asyncio.to_thread`.
- Cross-project reads (combine plan, project-source import, stats, probe predictions) bind the
  other project before reading (`src/config/project_context.py::bind_project`), because the guard
  refuses an index the bound project does not own.

### 5.3 The file

- `class_registry.json` stops being an authority (CC-2 option (a)). It is written only:
  - as the export snapshot (`export_writers.write_class_registry`, from the doc snapshot taken for
    that export);
  - as the training-job pin (`training/jobs.py`, from the doc, now async, no thread);
  - by the reverse migration on rollback (5.5).
- The trainer is unchanged: it reads the export's own file and has no OpenSearch client, so there
  is no ordering problem between the trainer image and the API.
- Every other reader of the project-level file is converted to the store or deleted.
- `scripts/curation/seed_class_registry.py` writes the doc (`ensure_class` per name, append-only;
  `--check` reads the doc). `repair_merged_active_classes.py`: read its header; port it to the store
  if the state it repairs can still occur, else delete it with its tests and its `scripts/README.md`
  row. `seed_live_harness.py`, `reclassify_after_registry_growth.py`, `_vlm_activity.py`,
  `bakeoff/class_map.py`: read the store.
- `.gitignore` keeps `/data/class_registry.json`; the timestamped backup pattern
  (`/data/class_registry.[0-9]*Z.json`) and `tests/test_class_registry_backup_gitignore.py` go
  with the per-write backups (no writer is left). Owner-visible change: per-write file backups end;
  the doc's `registry_revision`, every export snapshot and the `.migrated` file remain.
- `ProjectResources.class_registry_path` stays: it locates the legacy file for the 409 check, the
  migration and the reverse step. `resources_wire` keeps showing it (left as is: no consumer).

### 5.4 Fail closed until migrated

- Class-dependent routes depend on the snapshot (`_common.py::_registry_dep` becomes an async
  dependency returning the snapshot), so a project in the "file present, doc absent" state answers
  409 `registry_not_migrated` on every one of them. Inference routes and other projects are
  unaffected (they do not read the registry).
- Worker containers (detection worker, VLM worker, cluster refresh) treat the error as "skip this
  project and log once per minute"; they never write classes for it.
- No runtime read-shim that falls back to the file (owner rule: no back-compat shims). The
  migration is a required data migration, not a shim.

### 5.5 Migration, upgrade hook and rollback

1. `scripts/curation/migrate_class_registry.py` (new; run as `python -m
   scripts.curation.migrate_class_registry` like `scripts/purge_face_data`), flags
   `--project <slug> | --all`, `--dry-run`, `--reverse`, `--opensearch <url>` (default from env).
   Uses `make_script_opensearch` and binds each project in turn.
   - Forward, per project:
     - file absent: nothing to do (the doc is created empty on first use);
     - doc absent: seed the doc from the file with `op_type='create'`, ids preserved exactly,
       `next_class_id = max(id) + 1`, `migrated_from_file_at = now`; then rename the file to
       `class_registry.json.migrated` (never delete);
     - doc present and equal to the file (same entries): rename the file, no write (idempotent);
     - doc present, `reversed_to_file_at` set (a rollback happened since): the file is the
       authority of the downgrade period; replace the doc's classes from the file with
       `if_seq_no` (ids preserved), clear `reversed_to_file_at`, rename the file;
     - doc present and different, no `reversed_to_file_at`: error for that project (exit non-zero,
       report names both versions); nothing is written for it.
   - Reverse (rollback to a version without the doc), per project: write the doc's current classes to
     `class_registry.json` through `atomic_write_json` (so classes added after the migration are not
     lost), keep the doc, set `reversed_to_file_at`, keep any `.migrated` file as
     `class_registry.json.migrated.<ts>`.
   - `--dry-run` writes nothing and prints the per-project plan (action, class count, id range,
     conflicts).
   - Reports: JSON to stdout and to `<project_state_dir>/registry_migration/<ts>-{dry-run,run,
     reverse}.json`.
2. CLI: `./openprocessor curation migrate-registry [--project <slug> | --all] [--dry-run]
   [--reverse]` in `scripts/lib/cli_core.sh::cmd_curation`, implemented as
   `dc run --rm --no-deps -T -w /app api python -m scripts.curation.migrate_class_registry ...`
   (the `cmd_faces` pattern). OpenSearch must be up; say so in the error.
3. Installer hook, in a post-install library function (new `scripts/lib/registry_migration.sh`,
   sourced with the other post-install libraries):
   - `upgrade` and `repair`: after `run_health` passes, run `--all --dry-run`, save the output under
     `.install/registry_migration/<ts>-dry-run.json`, then run it for real. A failure stops with the
     report path and exit `EXIT_HEALTH`; the stack stays in its fail-closed 409 state.
   - `rollback` (`do_rollback`), before the files are restored and while the current API image is
     still available: run `--all --reverse` (open question Q-1B-1 below; recommended).
   - Fresh installs need nothing: the doc is created at project creation
     (`services/projects/lifecycle.py::create_project` creates the empty doc with
     `op_type='create'`).
   - Trap: `setup-openprocessor.sh` is at its size baseline. The `source` line and the two calls add
     lines; take the same number out of it by moving a cohesive post-install function into a
     library in the same commit. Never collapse lines, never raise the baseline.
4. Downgrade note (INSTALLATION.md and the CHANGELOG Breaking bullet): `openprocessor` rollback runs
   the reverse step. For a manual downgrade, run `./openprocessor curation migrate-registry --all
   --reverse` first; without it, rename `class_registry.json.migrated` back to `class_registry.json`
   in each project's data dir (classes added after the migration then exist only in the doc; export
   them first).

### 5.6 Interplay

- BA-1 (`02d_b1_api_frontend_tests.md`): the per-process snapshot keyed by `_seq_no` is the same
  "revision stamp" pattern that fixes the class-name staleness class.
- #365 (embedding swap, deferred): no B1 work; when scheduled, its dual-field migration reuses this
  pattern (fail-closed state, upgrade hook, rename-not-delete).

## 6. Tasks

One commit per task (split a task into sub-commits by area where it says so). Red test first, seen
failing on the base, with a paired control.

### 1B-1 Store and document (M)

- Files: `src/clients/curation_opensearch/registry.py`, `src/clients/curation_opensearch/__init__.py`
  (models; add `next_class_id`, `registry_revision`, the migration timestamps), the error code
  members `registry_not_migrated`, `registry_full`, `registry_conflict` in
  `src/schemas/curation/config_common.py::ErrorCode` (`make contracts` in this commit).
- Steps: 5.1, 5.2.
- Red: `tests/test_class_registry_cross_process.py` (the Q-P1.3 name, rewritten): 8 spawned
  processes call `add_class(f'c{i}')` at a `multiprocessing.Barrier` against a
  `multiprocessing.Manager`-backed fake that enforces `if_seq_no`; assert 8 classes, 8 distinct ids,
  each id maps to its own name. Run first against today's file-backed registry: it is red
  intermittently, so repeat it 20 times in the test (CC-2).
- Red: `ensure_class` called twice concurrently for one name returns one id; `add_class` past
  `MAX_REGISTRY_CLASSES` gives `registry_full`; 6 forced 409s give `registry_conflict`.
- Control: sequential adds give ids 0..N-1.
- Done when: 100 runs of the race give N classes and N distinct ids (CC-2 done-when), and no lock
  was added.

### 1B-2 Fail closed, empty-doc creation, project create and clone (S)

- Files: `registry.py` (absence rules), `services/projects/lifecycle.py::create_project`,
  `services/projects/clone.py` (copy the doc with `op_type='create'` into the target instead of the
  file), `routers/curation/_common.py` (async snapshot dependency).
- Red: `tests/test_registry_not_migrated.py`: file present and doc absent gives 409
  `registry_not_migrated` on a class route of that project; another project and `/detect` keep
  working; doc and file both absent creates the empty doc.
- Control: a migrated project answers 200.

### 1B-3 Route writers as single mutations (S)

- Files: `routers/curation/classes_write.py` (create + hotkey = one `add_class(..., hotkey_letter)`;
  rename + group + hotkey = one `update_class`), `classes_merge.py`, `keymap.py`
  (`_unbind_classes` = one `set_hotkeys` mutation; `_restore_classes` becomes a compensating
  `set_hotkeys` that restores only the letters it cleared, and only where the letter is still `None`,
  instead of writing back a whole snapshot, which would erase concurrent writes).
- Red: a concurrent rename and hotkey set on two different classes both persist (today one is lost
  by the second write); a failed keymap save restores the cleared letters without erasing a class
  added in between.
- Control: the existing classes and keymap route tests pass (`rg --files tests | rg
  "classes|keymap"`).

### 1B-4 Service writers (S)

- Files: `services/curation/class_ensure.py`, `dataset_import/prepare.py`, `dataset_import/undo.py`,
  `projects/combine/execute.py`: `await registry.ensure_class(...)` / `set_deprecated`.
- Red: an import and a combine adding overlapping names concurrently give one id per name.
- Control: a single import maps its classes by name as before (`rg --files tests | rg
  "dataset_import|combine"`).

### 1B-5 Readers to the snapshot (M; sub-commits: routers, services, training, scripts and workers)

- Files: the 48 reader files of section 4. Sync helpers take the snapshot as a parameter; async
  callers `await registry.snapshot(client)` once per request or job step.
- Cross-project readers bind the other project first (5.2 last bullet).
- `training/jobs.py::_pin_registry_snapshot_sync` becomes an async pin from the doc; keep it after
  the duplicate-job check. Today it is best-effort (returns `(None, None)` on any failure); with the
  fail-closed rule a `registry_not_migrated` project refuses the train start instead (the
  class-dependent route already refuses), and an OpenSearch error answers 503.
- Red: `tests/test_registry_snapshot_cache.py`: a write from process A is visible to process B's
  next snapshot (seq_no changed), with one `get(_source=False)` per check when unchanged.
- Control: each area's existing tests pass.

### 1B-6 Mirror and bootstrap self-heal (S)

- Files: `registry.py::sync_to_opensearch`, `services/curation/index_bootstrap.py` (its registry
  self-heal reads the doc).
- Red: after a mutation the `classes` index holds the new entry; a bulk with `errors: true` is
  logged with the failed count (not silently counted as upserted, as today).
- Control: an empty registry syncs nothing.

### 1B-7 File as export snapshot only (S)

- Files: `services/curation/export_writers.py`, `export.py`, `export_single_class.py`,
  `dataset_import/op_export.py` (unchanged reader of an export's file; verify), `.gitignore`,
  delete `tests/test_class_registry_backup_gitignore.py` with the backup writer.
- Red: an export after a class added through the doc carries that class in its
  `class_registry.json` with the right `export_id_map`.
- Control: `tests/curation/test_export_*` (golden, id map, single class) pass.
- Done when: `rg -n "class_registry_path|class_registry\.json" src scripts docker` lists only the
  export writers and readers of export snapshots, `op_export.py`, the trainer, `training/jobs.py`
  pin, `config/` and `projects/registry.py` record fields, the migration script and the 409 check.

### 1B-8 Scripts (S)

- Files: `scripts/curation/seed_class_registry.py`, `repair_merged_active_classes.py`,
  `seed_live_harness.py`, `reclassify_after_registry_growth.py`, `_vlm_activity.py`,
  `bakeoff/class_map.py`, `scripts/README.md`, `docs-site/docs/operations/curation-workflow.mdx` and
  `docs-site/docs/guides/use-your-own-domain.mdx` (they name `seed_class_registry.py`).
- Red: `tests/curation/test_seed_class_registry.py` adapted: seeding twice from one ONNX is a no-op
  on the doc; `--check` fails on class-order drift.
- Control: the seed of a fresh project gives ids in ONNX order.

### 1B-9 Migration script (M)

- File: `scripts/curation/migrate_class_registry.py` (new). Steps: 5.5 item 1.
- Red: `tests/test_migrate_class_registry.py`: dry-run writes nothing; the real run creates the doc
  with equal ids and renames the file; a rerun is a no-op; a conflicting doc is an error and writes
  nothing; reverse writes the doc's current classes to the file; a forward run after a reverse
  re-seeds from the file (ids preserved).
- Control: a project with no file is reported "nothing to do".

### 1B-10 CLI subcommand (S)

- Files: `scripts/lib/cli_core.sh::cmd_curation`, the help text, `CLAUDE.md` and `INSTALLATION.md`
  CLI lists.
- Red: `tests/installer/test_cli_curation_migrate.py` (installer harness with the docker shim):
  `curation migrate-registry --all --dry-run` runs the module with exactly those flags; an unknown
  flag exits 2.
- Control: `curation status` is unchanged.

### 1B-11 Installer hooks: upgrade, repair, rollback (M)

- Files: new `scripts/lib/registry_migration.sh`, `setup-openprocessor.sh` (source it; call it
  after `run_health` for upgrade/repair and in `do_rollback` before restoring files; take lines out
  to stay within the size baseline, trap in 5.5).
- Red: `tests/installer/test_upgrade_registry_migration.py`: the upgrade path calls the dry-run then
  the real migration, saves the dry-run under `.install/registry_migration/`, and stops with the
  report path and `EXIT_HEALTH` when the migration fails; rollback calls `--reverse` before the
  file restore.
- Control: a fresh install calls no migration.

### 1B-12 Acceptance phase change (S; the run itself is R-final in `06_release_acceptance.md`)

- File: `scripts/release/acceptance_phases.sh::ph_installer_upgrade` (and its seeding helper).
- Steps: before the upgrade, create a project with classes on the old version (with a gap in the
  ids, a deprecated class and a hotkey); after the upgrade, assert the doc has the same ids and
  names (`GET .../classes`), the file is `.migrated`; after `--rollback`, assert the old version
  serves the same classes (the reverse step).
- Red: run in `--dry-run`/shellcheck form in the lane (`tests/release/` if a harness exists for the
  phases: `rg --files tests/release`); the live run is R-final.

### 1B-12b Acceptance phases `dataset_import` and `combine` (S)

The registry migration touches import and combine (`dataset_import/prepare.py`,
`combine/execute.py`), and no scripted phase covers either today. Origin:
`release_acceptance_plan.md` phases E, H and the import-undo part of I.

- File: `scripts/release/acceptance_phases.sh`: two new `ph_` functions, registered so
  `acceptance_run.sh --list` names them. Pattern: the existing phases' helper style (shell, `curl`
  and `jq` against the run's API); counts through the OpenSearch `_count` API (docs.opensearch.org,
  "Count"), not by scrolling.
- `dataset_import`: build a YOLO dataset from the run's own `export` phase output and a COCO
  dataset from the license-filtered COCO subset (`scripts/datasets/fetch_coco_subset.py` already
  parses `instances_val2017.json`). For each: call
  `/curation/projects/{project}/datasets/preview`, then `/curation/projects/{project}/datasets/imports`.
  Assert classes map by name (compare the project's class names and ids before and after); a second
  import of the same files reports no new items;
  `/curation/projects/{project}/datasets/imports/{import_id}/undo` keeps a label a person set after
  the import; a second undo dry run changes nothing.
- `combine`: `/curation/projects/combine/preview`, then `/curation/projects/combine` into a new
  throwaway project from two sources. Assert counts, mapping by name, the test split preserved, and
  both sources unchanged (index document counts and a checksum of each source's registry document
  before and after).
- Red: `acceptance_run.sh --list` names both phases (fails today); `bash -n` and shellcheck pass (via
  pre-commit). The live run is R-final RF-70 and RF-71 (`06_release_acceptance.md`).
- Done when: both phases exist, are listed and lint clean; the sweep's docs/static step passes.

### 1B-13 Live registry test (S, runs in the batch sweep)

- File: `tests/live/test_class_registry_live.py` (marker `live`).
- The live harness runs the API with `--workers=1` (`docker/test/README.md`: some cancel endpoints
  and the SSE hub keep in-process state), so the cross-process race runs at the store level: 8
  spawned processes call `add_class` (distinct names) and then `ensure_class` (one shared name)
  against the harness OpenSearch, where the real server enforces `if_seq_no`. Pass: 8 + 1 classes,
  distinct ids, 5 repetitions (the B1 live smoke check is capped at 30 minutes,
  [00_index.md](00_index.md) section 8.1; the full repetition count is local and in B2's single live
  run). Then one HTTP sequence through the API (`POST /classes`,
  `PUT /classes/{id}` with a hotkey, `GET /classes`) checks the routes read the doc.
- Correction: CC-2 and the former master plan's 6.2 said "the live two-worker scenario"; the harness has one worker,
  and a second API replica is not part of it.

### 1B-14 Docs (S)

- `docs/ARCHITECTURE.md` (registry document is the authority), `docs/CURATION.md`
  (`registry_not_migrated`, the migrate command, `registry_full`, `registry_conflict`),
  `docs/opensearch_schema_design.md` (doc id, shape, mapping key), `INSTALLATION.md` (upgrade runs
  the migration; rollback runs the reverse; manual downgrade note), `docs/design/curation_api_contract.md`
  (new error codes), `scripts/README.md`.
- Add `docs/design/curation_design_rationale.md` sections 2.1 and 12: today 2.1 says the class
  registry lives on disk and 12 describes class identity over the file; update both to say the
  registry is an OpenSearch document and the file is an export snapshot.

## 7. Gate for this lane

- Targeted red/green per task; the spawn race 20 times on the base (to see red) and 10 times after.
- `.venv/bin/pre-commit run --files <touched files>`; `bash -n` and shellcheck through pre-commit for
  the shell files.
- `make contracts` in 1B-1 (error codes) and any commit that changes a wire model.
- No full suite, no `--all-files`, no Docker in the lane (the installer tests use the shim harness).

## 8. Done when

- No `class_registry.json` read outside export snapshots, the trainer, the training pin, the
  migration and the 409 check (1B-7 grep).
- `registry_not_migrated` is tested offline and the upgrade path is tested in the installer harness.
- The cross-process race passes 100 runs; the live registry test passes in the sweep.
- `ph_installer_upgrade` carries the id-equality and rollback assertions, and the `dataset_import`
  and `combine` phases exist (1B-12b); all run in R-final.
- The closing B1 PR moves the keep-list below to its targets and deletes this file
  ([00_index.md](00_index.md) "Plan lifecycle").

### 8.1 What to keep after merge

| Fact or section | Target doc |
|---|---|
| the registry document: id, shape, OCC rules, `registry_full` bound, `ensure_class` vs `add_class` (sections 5.1-5.2) | `docs/ARCHITECTURE.md`, `docs/opensearch_schema_design.md` |
| migration, upgrade hook, rollback reverse step and the manual downgrade note (5.5) | `INSTALLATION.md` |
| error codes `registry_not_migrated`, `registry_full`, `registry_conflict` | `docs/design/curation_api_contract.md`, `docs/CURATION.md` |
| the Breaking entry (section 9) | `CHANGELOG.md` |

## 9. Docs, contracts, CHANGELOG

- Contracts: `ErrorCode` members (1B-1); no route added (the CLI is not an HTTP route).
- CHANGELOG `[Unreleased]`:
  - Fixed: "concurrent class writes no longer lose a class or reuse an id" (N2).
  - **Breaking:** "the class registry moves from `class_registry.json` to an OpenSearch document per
    project. `openprocessor upgrade` migrates it (dry-run report under `.install/`); until then
    class routes answer 409 `registry_not_migrated`. The file is kept as
    `class_registry.json.migrated`. Per-write backups of the registry file end. Rollback writes the
    registry back to the file."
- `frontend/CHANGELOG.md`: only if Cropwright needs a change to show the 409 message (check in the
  sweep's browser pass; the generic error display should already show `detail.message`).

## 10. Upgrade impact

- 0.6.0 -> 0.7.0 (supported, direct): the installer runs the migration after the stack is healthy;
  ids preserved; until it runs (or if it fails) class routes answer 409 and inference keeps working.
- A deployment that upgrades with `git pull` + `make up` (a checkout, no installer) must run
  `./openprocessor curation migrate-registry --all`; the 409 message says so.

## 11. Rollback

- Before merge: revert the 1B merge commit alone (1B is the last track; nothing depends on it).
- After users migrated: `openprocessor` rollback runs the reverse step; a manual downgrade follows the
  note in 5.5 item 4. The doc stays and is inert for older code.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Migration of the owner's real projects (M7) | dry-run saved first, idempotent, ids preserved, file renamed not deleted, fail-closed 409 until it runs, reverse on rollback |
| A reader still reads the file after B1 (M8) | 1B-7 done-when grep; dead-code and layering ratchets |
| More than 10,000 classes (M22) | `registry_full`; per-class docs would be a separate design (not planned) |
| Async conversion of 48 reader files is wide | mechanical sub-commits by area; sync helpers take the snapshot as a parameter, so most call sites change one line |
| Cross-project reads hit the guard | bind the other project first (5.2); a test per cross-project reader |
| Rollback leaves the old version with no classes (found while verifying: `do_rollback` restores installer files only) | reverse step on rollback (Q-1B-1) |
| `setup-openprocessor.sh` at its size cap | move a cohesive function out in the same commit |

## 13. Corrections found while verifying

- `keymap.py` writes the registry directly twice (`_unbind_classes`, `_restore_classes`); the
  former master plan named only the two `classes_write.py` writes. `_restore_classes` must become a compensating
  mutation, not a snapshot write-back.
- `services/projects/clone.py` copies the registry file to the clone: a writer the plan did not list.
- `services/training/jobs.py` pins a registry snapshot at train start: a reader the plan did not list
  (it is best-effort today; fail closed after B1).
- Four readers read another project's file by path; with the doc they must bind that project.
- `add_class` cannot be "idempotent by name" without breaking `POST /classes`'s documented 409: split
  into `add_class` (route) and `ensure_class` (automated writers).
- Rollback: `ph_installer_upgrade` rolls back after the upgrade, and `do_rollback` does not touch
  project data, so without a reverse step the older version would serve an empty registry.

## 14. Open question for the owner

- **Q-1B-1 Rollback after the registry migration.** (a) Recommended: `openprocessor` rollback runs
  `migrate-registry --all --reverse` before restoring files, writing the doc's current classes back
  to `class_registry.json`; a later upgrade re-seeds from the file. (b) The former master plan's manual note
  only: rollback leaves the older version with no registry file until the operator renames
  `.migrated` back, and classes added after the migration are lost to it. If (b) is chosen, drop the
  reverse parts of 1B-9, 1B-11 and 1B-12.

## 15. Lane agent brief (paste as is)

```
You are the implementation agent for B1 track 1B (class registry document) of OpenProcessor.
Plan: docs/design/program/02b_b1_class_registry.md (this file). Execute it standalone.
Read first: CLAUDE.md, docs/VISION_AND_GOALS.md, docs/design/engineering_standards.md,
docs/design/program/02_b1_overview.md sections 3-6, docs/design/codebase_audit_plan.md 4.7 CC-2.

Worktree: <repo>-wt-b1-registry, branch b1/registry (created by the coordinator from
int/b1-correctness). Work ONLY there. Never push, never merge, never check out another branch.
Compose project for any Docker command: -p op-b1-registry (the lane should need none; the
installer tests use the shim harness). GPU: none. Deadline: <coordinator fills in>.

Rules:
- One commit per task 1B-1 .. 1B-14 including 1B-12b (sub-commits by area for 1B-5), conventional commits, hooks on.
- Red test first, run on the unchanged code and SEE IT FAIL (the cross-process race: 20 repeats
  to see the intermittent red), paste the red output into the commit body; then fix; paired
  control green.
- Targeted tests only (.venv/bin/python -m pytest <files> -q --no-cov) and
  .venv/bin/pre-commit run --files <touched files>. No full suite, no --all-files.
- Do NOT edit src/clients/curation_opensearch/bodies_other.py or index_bootstrap._migrations():
  track 1A adds the class_registry mapping key. When the coordinator says 1A is merged, rebase
  onto int/b1-correctness and re-run your targeted tests.
- OCC with if_seq_no/if_primary_term and op_type=create only (cite the OpenSearch index API doc);
  no file locks, no in-process locks, no asyncio.to_thread around the store, no fallback to the
  file (fail closed with 409 registry_not_migrated). Class identity is the name; ids are preserved
  exactly by the migration.
- make contracts in the same commit as the ErrorCode change.
- setup-openprocessor.sh is at its size baseline: move a cohesive function into scripts/lib/ to
  make room; never collapse lines, never raise a baseline.
- Never use pgrep -f, pkill -f, or sleep/until polling loops.

Report at the end: commits (sha + subject), red outputs in one line each, the recounted reader and
writer sites you changed, the answer you assumed for Q-1B-1, and any plan correction you found.
```
