# B1 track 1A: OpenSearch job leases replace every start lock

| Field | Value |
|---|---|
| Status | plan, not started |
| Batch / release | B1 (correctness foundation), part of the single v0.7.0 release |
| Depends on | B0 merged to main (`01_b0_close_out.md`); `int/b1-correctness` cut from main (`02_b1_overview.md`) |
| Decisions applied | D-8 (J-LEASE), D-13 (order), D-15 (trainer create-only generation claims), owner rule "no in-process or file locks for cross-process state" |
| Source IDs | the former master plan (rev 4, in git history) 3a (Q-P1.0, P1.1, P1.2, P1.5, P1.6, P1.7), 3b (B1 rows), 3c (Q-D2, D4, D7 trainer part, D9, D10, D12), 3d, 6.1, 6.3; `task_queue_plan.md` Phase 1 and section 3 (D2, D4, D7, D9, D12); `codebase_audit_plan.md` 4.2 OS-3, 4.7 CC-3 (cut) |
| Size | L |
| Lane | branch `b1/leases`, worktree `<repo>-wt-b1-leases`, compose project for any live check `op-b1-leases` |

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read only, with `rg`).
Consolidated 2026-11-03. Line numbers
are not used; re-find every site with the `rg` pattern given.

## 1. Goal

Two requests on two API workers can no longer start one singleton job twice, a sibling's live job
is never flipped to `interrupted`, a blocked event loop no longer costs a live job its liveness, and
no file lock (`fcntl.flock`) guards cross-process state in the B1 scope. The claim is an OpenSearch
lease document per project, written with create-only and optimistic-concurrency writes. The same
lease and the same heartbeat thread are what the B2 Celery task wrapper reuses
(`03_*` B2 file), so nothing here is rewritten later.

Mission link: correct under concurrency (32 uvicorn workers plus worker containers).

## 2. Preconditions

- B0 is merged into main and `int/b1-correctness` exists (`02_b1_overview.md`, section 4).
- The lane worktree is created by the coordinator:
  `git worktree add <repo>-wt-b1-leases -b b1/leases int/b1-correctness`.
- The venv exists in the worktree (`.venv/`, CPU torch, `requirements-test.txt`, per CONTRIBUTING.md).
- No Docker is needed for the lane. The live lease test runs in the batch sweep only.

## 3. Documented mechanisms this track uses (cite them in code comments where non-obvious)

| Mechanism | Official doc | Used for |
|---|---|---|
| OpenSearch index API `op_type=create` ("indexes a document only if it does not already exist") and `if_seq_no` / `if_primary_term` (a mismatch returns 409) | https://docs.opensearch.org/latest/api-reference/document-apis/index-document/ | acquire, renew, transfer, release, takeover |
| OpenSearch get API is realtime by default (no refresh needed between `get` and `index`) | https://docs.opensearch.org/latest/api-reference/document-apis/get-documents/ | reading a lease; never `search` |
| OpenSearch Job Scheduler lock document (seq_no / primary_term lock doc, prior art) | https://docs.opensearch.org/latest/monitoring-your-cluster/job-scheduler/locks/ | design precedent |
| Redis `SET NX PX` is discouraged in favour of Redlock | https://redis.io/docs/latest/commands/set/ | why not Redis (and Redis exists only from B2) |
| Python `threading.Thread(daemon=True)`, `threading.Event` | https://docs.python.org/3/library/threading.html | heartbeat thread and its stop / lost flags |
| Python `asyncio.run` (new event loop per call; one loop per thread) and "asyncio objects are not thread-safe" | https://docs.python.org/3/library/asyncio-runner.html , https://docs.python.org/3/library/asyncio-dev.html#concurrency-and-multithreading | the thread runs its own loop and its own client |
| Python `contextvars.copy_context()` / `Context.run` (a new thread starts with an empty context) | https://docs.python.org/3/library/contextvars.html | carry the bound project into the thread |
| Python `os.open(path, O_CREAT \| O_EXCL)` raises `FileExistsError` when the file exists | https://docs.python.org/3/library/os.html#os.open | trainer generation claims, API training-job create |
| Python `multiprocessing` `spawn` start method, `Barrier`, `Manager` | https://docs.python.org/3/library/multiprocessing.html | race tests (test scaffolding only) |
| OpenSearch bulk API item errors (`errors: true`, per-item `error`) | https://docs.opensearch.org/latest/api-reference/document-apis/bulk/ | OS-3 |

Repo precedents to reuse (do not hand-roll a second version):
- `src/clients/occ.py` (`_fetch_with_version`, `OCCFinalConflictError`, the conflict test
  `'Conflict' in type(exc).__name__`).
- `src/services/projects/guard.py::make_script_opensearch` (the only factory for a non-request
  client; the isolation guard stays installed).
- `export/export_worker.py::_heartbeat_loop` (an existing `threading.Thread` heartbeat with a
  `threading.Event` stop): the same shape, now renewing a lease.
- `src/services/projects/combine/execute.py::ensure_owner` / `FencedError` (cooperative fencing by
  claim; becomes lease based).
- `tests/projects/conftest.py::FakeRegistryOpenSearch` (a fake that enforces `op_type='create'` and
  `if_seq_no`); the cross-process tests need a `multiprocessing.Manager`-backed variant.

## 4. Current state (recounted)

`rg -n "exclusive_start_lock|exclusive_file_lock|fcntl\.flock" src docker` finds **55 hits in 19
files** (the former master plan said 51 in 18; corrected). `docker/` has none. Per file:

| File | Hits | Kind of hit |
|---|---|---|
| `src/services/curation/job_lock.py` | 10 | the two helper definitions and their docstrings (stays until B4) |
| `src/services/curation/label_ops/store.py` | 5 | `write_lock`, `undo_lock` (2 calls) and `_lock_is_held` (direct `fcntl.flock`) |
| `src/services/curation/dataset_import/runner.py` | 4 | 3 calls (start, resume, undo) + import |
| `src/services/training/promote_json.py` | 4 | 2 `exclusive_file_lock` calls + import + docstring |
| `src/services/curation/assist_loop/job.py` | 3 | 2 calls (start, resume) + import |
| `src/services/curation/item_scores/job.py` | 3 | 1 call + import + docstring |
| `src/services/curation/probe_job.py` | 3 | 1 call + import + docstring |
| `src/services/curation/assist/job.py`, `reprocess_job.py`, `bulk_ingest/runner.py`, `bulk_ingest/source_scan.py`, `projects/combine/service.py`, `projects/legacy_repair.py`, `training/promote_job.py` | 2 each | 1 call + import |
| `src/services/curation/file_job.py` | 1 | `_state_lock` (direct `fcntl.flock`) |
| `src/services/curation/bulk_ingest/dispatcher.py` | 2 | B2 (beat) |
| `src/services/curation/open_vocab_sweeper.py` | 2 | B2 (beat + `-c 1` queue) |
| `src/services/curation/source_image_cache.py` | 2 | direct `fcntl.flock`, B2 (beat prune) |
| `src/services/curation/event_hub.py` | 2 | direct `fcntl.flock`, B4 (Redis Streams) |

Call sites removed by this track (`rg -n "exclusive_(file|start)_lock\(" src | rg -v job_lock.py`
lists 20 today; this track removes 18 of them, plus the 2 direct `flock` uses in `file_job.py` and
`label_ops/store.py`):

| Site (re-find by `rg`) | Calls | Lease kind |
|---|---|---|
| `reprocess_job.py` create path | 1 | `reprocess` |
| `assist/job.py` start | 1 | `assist_run` |
| `assist_loop/job.py` start and resume | 2 | `assist_loop` |
| `dataset_import/runner.py` start, resume, undo | 3 | `dataset_import` |
| `item_scores/job.py` start | 1 | `item_scores` |
| `probe_job.py` start | 1 | `probe` |
| `training/promote_job.py` start | 1 | `promote` |
| `projects/combine/service.py` resume | 1 | `combine` (scope = combine job id) |
| `bulk_ingest/runner.py` start | 1 | `bulk_ingest` |
| `bulk_ingest/source_scan.py` scan | 1 | `source_scan` (scope = source id) |
| `projects/legacy_repair.py` repair | 1 | `legacy_repair` |
| `label_ops/store.py` `write_lock`, `undo_lock` | 2 | `label_op` (scope = op id), `label_undo` |
| `training/promote_json.py` `update_sharing`, `write_promote_backpointer` | 2 | `promote_json` (scope = model name) |

Left after B1 (the done-when grep lists only these): `bulk_ingest/dispatcher.py` and
`open_vocab_sweeper.py` calls, `source_image_cache.py` and `event_hub.py` direct `flock`, and the
definitions in `job_lock.py` (B2/B4, `05_b4_stateless_api.md`).

Other facts the design depends on (all verified):
- `FileJob` (`src/services/curation/file_job.py`): `HEARTBEAT_STALE_S = 30.0`,
  `HEARTBEAT_TICK_S = 10.0`, `_state_lock` (blocking `flock` on `state.lock`), `update()` and
  `repair_if_stale()` hold it, `reconcile()` calls `job_reconcile.reconcile_stale_running`, and
  `async def heartbeat_ticker` is an asyncio loop.
- `heartbeat_ticker` is started by `asyncio.create_task` in 11 places:
  `rg -n "create_task\(heartbeat_ticker" src` (datasets router resume, promote_job, combine/execute,
  dataset_import/runner x2, reprocess_job, assist/runner, bulk_ingest/runner, embedding_viz,
  probe_job, assist_loop/driver). `autolabel/job.py` has its own `_tick()` ticker task.
- Startup reconcile: `src/app/lifespan.py::_reconcile_orphaned_state` calls the **sync**
  `startup_reconcile.reconcile_all_jobs()`, which runs each module's `reconcile_orphaned_jobs()` for
  every project (`JOB_MODULES` tuple) and combine once (combine jobs are global).
- `job_reconcile.reconcile_stale_running` treats a missing heartbeat as stale (CC-3 / Q-D4).
- Region clustering (`clustering/region_cluster_jobs.py`) keeps a `running` flag in `job.json` with
  `_JOB_STALE_S = 1800.0`; `start_region_cluster_job` and `start_region_fp_centroid_job` are
  check-then-write.
- `selection/job.py` and `embedding_viz.py` start with `_is_busy()` then write; `autolabel/job.py`
  `start_job` checks `is_busy()` then writes `state.json` and `trigger.json`.
- `label_ops/store.py::served_status` derives `interrupted` / `undo_interrupted` from
  `_lock_is_held` (the flock is held for the whole write or undo).
- `promote_json.py` has **two** writers: the sharing route (`routers/curation/_models_sharing.py`
  `update_sharing`) and a re-promote (`triton_promote.py` `write_promote_backpointer`). Its temp
  name is `.{name}.{pid}.tmp`, not `atomic_write_json`.
- Trainer: `docker/trainer/job_protocol.py::list_pending_jobs` lists non-terminal jobs; there is no
  claim (`rg O_EXCL docker/trainer src/services/training` finds nothing). API
  `src/services/training/jobs.py::write_job` does `target.exists()` then writes.
- `export/export_worker.py::reconcile_claimed` fails **every** job in `claimed/` at worker start;
  there is only a worker-wide `worker.heartbeat`, no per-task heartbeat.
- The SETTINGS role: `src/config/projects.py::resources_for_new` points SETTINGS (and
  UMAP_VIZ_STATE) at the `configs` index for every new project (folded layout); older stored
  records may keep a separate settings index (legacy layout). `_configs_body()` has
  `dynamic: False` and maps `kind`, `name`, `revision`; `_settings_body()` has **no**
  `dynamic: False` (dynamic mapping).
- The bootstrap (`src/services/curation/index_bootstrap.py`) stamps each index's mapping `_meta`
  with `schema_fingerprint()`, a hash of every index body and of the migration list; a body change
  plus a migration in `_migrations()` makes every project re-run the migrations once.
  `tests/curation/test_index_schema_marker.py` holds bodies and migrations to each other.
- Error codes are a `Literal` in `src/schemas/curation/config_common.py` (`ErrorCode`); the busy
  codes are per family (`reprocess_busy`, `import_busy`, `assist_busy`, `assist_loop_busy`,
  `probe_busy`, `bulk_ingest_busy`, `label_op_busy`, `source_busy`, ...).

## 5. Design (J-LEASE, D-8)

### 5.1 Where the lease lives

- In the bound project's SETTINGS role index, resolved only as
  `index_name(get_curation_config(), IndexRole.SETTINGS)`. Never build the name.
  - Folded layout (every project made by `resources_for_new`, `default` included): the configs
    index `{prefix}{slug}__configs`.
  - Legacy layout (stored resources older than the fold): the separate settings index.
- One top-level object key, `lease`, so nothing collides with `kind`, `name`, `revision` (mapped on
  configs for config-store rows). No `doc_type` field, so a `doc_type`-filtered config-store query
  never sees a lease.
- Mapping: `"lease": {"type": "object", "enabled": false}` in `_configs_body()` **and**
  `_settings_body()`. This track also adds `"class_registry": {"type": "object", "enabled": false}`
  to both bodies, in the same commit, for track 1B (`02b_b1_class_registry.md`), so the two tracks
  never both edit the bodies or the migration list.
- An additive migration in `index_bootstrap._migrations()` (for example
  `ensure_settings_object_keys`) puts both keys on existing configs and legacy settings indexes. On
  the dynamic legacy index this stops OpenSearch from mapping lease or registry fields on its own.

### 5.2 Document

- Id: `lease:<kind>` or `lease:<kind>:<scope>`.
- Source: `{"lease": {"kind", "scope", "job_id", "owner", "generation", "acquired_at",
  "heartbeat_at", "ttl_s", "state"}}` where `state` is `held` or `released`, `owner` is
  `<hostname>:<pid>:<thread>` (no secrets), times are epoch seconds (float).

### 5.3 Code layout (layering: domain imports only config and utils)

| Module | Holds |
|---|---|
| `src/domain/job_lease.py` | `LEASE_TTL_S = 120.0`, `HEARTBEAT_TICK_S = 15.0`, `FENCE_MARGIN_S = 30.0`, `LeaseKind` (the table in 5.4), `doc_id(kind, scope)`, `is_expired(lease, now)` (`heartbeat_at + ttl_s < now`), `next_generation()`, the frozen result types `Acquired`, `Busy`, `Renewed`, `Lost` |
| `src/clients/job_lease.py` | the OpenSearch operations in 5.5, async, on a passed client; the conflict test reused from `occ.py` |
| `src/services/curation/lease_heartbeat.py` | `LeaseHeartbeat` (5.6), the thread the B2 wrapper reuses unchanged |
| `src/services/curation/file_job.py` | `FileJob` loses `_state_lock`; `HEARTBEAT_STALE_S` becomes `LEASE_TTL_S` (imported from the domain module); `heartbeat_ticker` is deleted |

### 5.4 Kind and scope per start site

The project is implicit (the doc lives in the project's index). "Held" says how long the lease is
held: for the whole run (renewed by the heartbeat thread) or only for the start critical section.

| Kind | Scope | Held | Start sites |
|---|---|---|---|
| `reprocess` | none | run | `reprocess_job.py` |
| `assist_run` | none | run | `assist/job.py` (runner `assist/runner.py`) |
| `assist_loop` | none | run | `assist_loop/job.py` start and resume (driver `assist_loop/driver.py`) |
| `dataset_import` | none | run | `dataset_import/runner.py` start, resume, undo; `routers/curation/datasets.py` resume rescan |
| `item_scores` | none | run | `item_scores/job.py` |
| `probe` | none | run | `probe_job.py` |
| `promote` | none | run | `training/promote_job.py` |
| `promote_json` | `<model_name>` | start (short critical section) | `training/promote_json.py` both writers (sharing route and re-promote), in the model owner's project |
| `combine` | `<combine_job_id>` | run | `projects/combine/service.py` resume and `combine/execute.py` run; doc in the **target** project's index (combine jobs are global) |
| `bulk_ingest` | none | run | `bulk_ingest/runner.py` |
| `source_scan` | `<source_id>` | run | `bulk_ingest/source_scan.py` |
| `legacy_repair` | none | run | `projects/legacy_repair.py` (32 workers run it at startup: one wins, the rest skip) |
| `label_op` | `<op_id>` | run | `label_ops/store.py` `write_lock` (callers `label_ops/record.py`, `label_ops/undo.py`) |
| `label_undo` | none | run | `label_ops/store.py` `undo_lock` |
| `selection` | none | run | `selection/job.py` |
| `embedding_viz` | none | run | `embedding_viz.py` |
| `region_cluster` | none | run | `clustering/region_cluster_jobs.py::start_region_cluster_job` |
| `region_fp_centroids` | none | run | `clustering/region_cluster_jobs.py::start_region_fp_centroid_job` |
| `autolabel` | none | start | `autolabel/job.py::start_job` (the run happens in the auto-label worker container; B2 Phase 5 replaces the trigger) |

For `autolabel` in B1 the lease closes the two-POST race only: acquire, run the existing
`is_busy()` check, write `state.json` and `trigger.json`, release. The record-based busy check stays
the guard against a second start while the worker runs. B2 makes the worker take the lease over.

### 5.5 Operations (`src/clients/job_lease.py`)

| Operation | Calls | Outcome |
|---|---|---|
| `acquire(client, kind, scope, job_id, owner, ttl_s=LEASE_TTL_S)` | realtime `get`; absent -> `index(op_type='create')`; present and (`released` or expired) -> `index(if_seq_no, if_primary_term)` with `generation + 1` | `Acquired(generation, seq_no, primary_term)`; any 409, or a live holder -> `Busy(job_id, heartbeat_at)` |
| `renew(client, lease)` | `index` of the same doc with the lease's own `if_seq_no` / `if_primary_term`, new `heartbeat_at` | `Renewed(seq_no, primary_term)`, or `Lost` on 409 |
| `fence(lease)` | see 5.7 | proceed only when it passes |
| `transfer(client, lease, new_owner)` | as `renew`, changing `owner` | used by the B2 task wrapper |
| `release(client, lease)` | `index` with `state='released'` and `if_seq_no` | a 409 means someone took over: log it, do not overwrite |
| `takeover_expired(client, kind, scope, job_id, owner)` | as `acquire`, but only when the doc is expired or absent | used by `repair_if_stale`, reconcile, and later the B2 reaper |

- `get` on an absent id raises `opensearchpy.NotFoundError`: catch exactly that class for "absent".
  Every other error (connection, 5xx) propagates: the start answers **503** (fail closed), never
  "proceed without a lease".
- Writes use `refresh=False`: reads are realtime `get`s by id, so no refresh is needed.
- Do not add a retry loop: a 409 on acquire is an answer (`Busy`), not a retry case.

### 5.6 Heartbeat thread (`LeaseHeartbeat`)

- `LeaseHeartbeat(lease, *, job: FileJob | None, hosts: list[str])` with `start()`, `stop()`,
  `lost: threading.Event`, `last_renew_ok: float` (monotonic).
- `start()` creates `threading.Thread(target=ctx.run, args=(self._thread_main,), daemon=True)` with
  `ctx = contextvars.copy_context()`, so the bound project (`src/config/project_context.py`
  `_BOUND` ContextVar) is visible in the thread. `_thread_main` calls `asyncio.run(self._run())`:
  the thread has its own event loop.
- `_run()` builds its **own** client with `make_script_opensearch(hosts, timeout=...)` (guard
  installed; never the request loop's client; no new sync client factory), then every
  `HEARTBEAT_TICK_S` (15 s), waiting on the stop event rather than sleeping blind:
  1. `job.touch_heartbeat()` (the file heartbeat the UI shows);
  2. `renew(...)`; on `Renewed` update `last_renew_ok`; on `Lost` set `lost` and exit;
  3. on a transport error, log and try again next tick; if `now - last_renew_ok` exceeds
     `LEASE_TTL_S - FENCE_MARGIN_S` (90 s), set `lost` and exit (self-fence before anyone can take
     the lease over);
  4. `await client.close()` in `finally`.
- `stop()` sets the stop event and joins with a timeout of one tick; then the caller releases the
  lease (on the request loop's client) or, for a terminal write, fences first.
- Delete `file_job.heartbeat_ticker` and `autolabel/job.py`'s `_tick()` ticker in the same commit
  as the last caller moves. All 11 `create_task(heartbeat_ticker(...))` sites become
  `LeaseHeartbeat(...).start()` / `stop()` around the run.
- Why a thread (Q-D9): CPU work on the loop (CC-5 upload extract and directory ingest until B2,
  `selection` `k_center_greedy`) stops an asyncio ticker. A blocked loop must not cost a live job its
  lease.

### 5.7 Fencing (cooperative)

- On `lost` the job stops at its next progress write (check `heartbeat.lost.is_set()`).
- Progress writes to `state.json`: allowed only when `not lost` and
  `now - last_renew_ok < LEASE_TTL_S - FENCE_MARGIN_S`. This is a local check: with one host clock
  and a renew younger than 90 s, no other process can see the lease as expired (it expires at
  `heartbeat_at + 120 s`). This keeps progress writes free of a round trip.
- Terminal writes (`completed`, `failed`, `cancelled`, `interrupted`) are preceded by a real
  `renew` round trip; write only on `Renewed`.
- Every record written under a lease carries `lease_generation` (an internal `state.json` key;
  check the job status serializer drops unknown keys, or add it to the wire model with
  `make contracts`). A taker-over always writes a higher generation, so the trail shows a
  superseded writer.
- `combine/execute.py::ensure_owner` / `FencedError` becomes the lease check (same call sites, no
  second mechanism).
- Residual risk (documented in `docs/CURATION.md`): a holder that stalls longer than the margin
  between its check and its write can still write once. A write takes milliseconds; accepted.

### 5.8 Rules

- The API acquires the lease **before** writing the job record (`queued` / `running`) and answers
  409 with the family's existing busy code on `Busy` (add a code to `ErrorCode` only for a family
  that has none: `selection`, `embedding_viz`, `region_cluster`, `region_fp_centroids`,
  `item_scores`, `autolabel` if they lack one; `make contracts` in the same commit).
- `LEASE_TTL_S = 120` s (at least 8 missed ticks; OpenTranscribe's measured thread-heartbeat lease
  is about 90 s, `task_queue_plan.md` 4d). `HEARTBEAT_STALE_S` equals it, so the file heartbeat and
  the lease agree.
- Busy window after a crash: a dead holder blocks a new start for at most 120 s after its last
  renew; then the next start takes over. Document it in `docs/CURATION.md`.
- Only the lease holder (or a taker-over) writes `state.json`. Readers never write (CC-1, B0).
  `repair_if_stale` becomes `takeover_expired` + write `interrupted` + `release`.
- Mid-upgrade jobs: a `running` record with **no lease doc** and a heartbeat older than
  `LEASE_TTL_S` is taken over (acquire creates the doc) and repaired to `interrupted` (resumable
  families) by the acquirer. With no lease doc and a **fresh** heartbeat it is busy until the
  heartbeat ages past `LEASE_TTL_S` (only the first two minutes after an upgrade).
- Cancel stays cooperative (`cancel.flag`); it needs no lease.
- OpenSearch unreachable: the start is refused with 503.
- A project delete removes the lease docs with the index; the running job's next renew is `Lost`
  and it stops (delete already waits for or cancels jobs).
- Clocks: one host, so they agree. Multi-host would need the TTL to exceed the clock skew; note it
  in INSTALLATION.md under multi-host (not supported today).

### 5.9 Traps

- `get` is realtime; never read a lease with `search`.
- No top-level lease fields named `kind`, `name`, `revision`.
- The heartbeat thread must not share the request loop's client or any asyncio object.
- A forked child must not inherit a running heartbeat thread: the tests use `spawn`.
- `reconcile_all_jobs` is sync today and lease operations are async: make it async and await it in
  the lifespan (it is called from an async lifespan already). Do not add `asyncio.run` inside the
  lifespan.
- 32 workers run the startup reconcile and `legacy_repair` at once: each call must be a takeover
  attempt, so one wins and the others see `Busy` and skip.
- `label_ops` liveness changes: a dead writer was visible at once (a released flock); now
  `served_status` reports `interrupted` once the lease expires (up to 120 s). Update the docstring
  and any test that expects an instant flip.

## 6. Tasks

One commit per task, conventional commit messages, red test first and seen failing on the base
(record the red output for the PR description), paired control passing before and after. Run only
the targeted tests and static checks on touched files in the lane (no full suite, no
`--all-files`).

### 1A-1 Mapping keys and bootstrap migration (S)

- Files: `src/clients/curation_opensearch/bodies_other.py` (`_configs_body`, `_settings_body`),
  `src/services/curation/index_bootstrap.py` (`_migrations`), a small `ensure_*` helper next to
  the existing ones (`rg -n "def ensure_" src/clients/curation_opensearch/ensure_fields.py`).
- Steps: add `lease` and `class_registry` as `{"type": "object", "enabled": false}` to both
  bodies; add one additive migration that puts both keys on the configs index and on a legacy
  settings index (skip when SETTINGS resolves to the configs index, so it is not put twice).
- Red: `tests/test_lease_folded_and_legacy_index.py::test_mapping_has_object_keys` (both bodies
  carry both keys); `tests/curation/test_index_schema_marker.py` fails until the migration exists.
- Control: `tests/projects/test_configs_mapping_union.py` stays conflict-free;
  `tests/projects/test_folded_roles_by_id_only.py` gains the `lease:*` and `class_registry` ids
  and still passes.
- Done when: both bodies carry both keys, the marker test passes, a bootstrap run on a fake legacy
  settings index issues the put once.

### 1A-2 Domain rules (S)

- File: `src/domain/job_lease.py` (new).
- Steps: constants, `LeaseKind` enum or table (5.4), `doc_id`, `is_expired`, result dataclasses.
- Red: `tests/domain/test_job_lease_rules.py` (or `tests/test_job_lease_rules.py`; follow where
  domain tests live: `rg --files tests | rg domain`): expiry at `heartbeat_at + ttl_s`, doc ids per
  kind and scope.
- Control: a sequential acquire, release, acquire gives generations 1 and 2 (in 1A-3's test).

### 1A-3 Lease client (M)

- File: `src/clients/job_lease.py` (new).
- Steps: 5.5. Reuse the conflict classification of `src/clients/occ.py`; do not copy its retry
  loop.
- Red: `tests/test_job_lease_cross_process.py`: 8 `spawn` processes acquire one lease at a
  `multiprocessing.Barrier` against a fake OpenSearch whose documents live in a
  `multiprocessing.Manager` dict, with per-document atomic `index` that enforces `op_type='create'`
  and `if_seq_no` (test scaffolding emulating the server, not production locking). Pass: exactly
  one `Acquired`, 7 `Busy`. Fails today in the sense that there is no lease: write the test against
  the real start functions in 1A-6/1A-7 (it then fails on the base for the four unlocked starts).
- Red: `tests/test_job_lease_takeover.py`: a renew after a takeover returns `Lost`; the old holder's
  terminal fence fails and it writes nothing more; reconcile on a fresh lease leaves it alone; on an
  expired lease it takes over and marks the job `interrupted` (replaces Q-P1.5 /
  `tests/test_reconcile_sibling_job.py`, which is not written).
- Control: sequential acquire, release, acquire gives generations 1 and 2; a connection error on
  `get` raises (no silent proceed).

### 1A-4 Heartbeat thread (M)

- File: `src/services/curation/lease_heartbeat.py` (new); delete `heartbeat_ticker` from
  `file_job.py` (and its `__all__` entry) once the last caller moved (1A-6).
- Steps: 5.6.
- Red: `tests/test_lease_heartbeat_thread.py`: with `LEASE_TTL_S` monkeypatched to 2 s and the tick
  to 0.2 s, a coroutine that blocks the event loop (`time.sleep(3)`) keeps its lease (a concurrent
  `acquire` from a spawned process gets `Busy`). Fails with the asyncio ticker.
- Control: after `stop()` and `release`, a new acquire succeeds with generation + 1; a renew 409
  sets `lost`.

### 1A-5 FileJob single writer; reconcile by takeover (M)

- Files: `src/services/curation/file_job.py`, `src/services/curation/job_reconcile.py`,
  `src/services/curation/startup_reconcile.py`, `src/app/lifespan.py`
  (`_reconcile_orphaned_state`), every `reconcile_orphaned_jobs` and `repair_if_stale` caller
  (`rg -n "repair_if_stale|\.reconcile\(" src`: reprocess, promote, dataset_import store, combine
  service and store, assist, bulk_ingest store, assist_loop, item_scores, selection, embedding_viz,
  probe, autolabel).
- Steps:
  1. Delete `_state_lock` and the `fcntl` import. `update()` becomes read-modify-write by the lease
     holder only (document the single-writer rule in the docstring).
  2. `repair_if_stale` -> `takeover_expired` + write `interrupted` + `release`; it needs the client
     and the kind, so it becomes `async` and every caller awaits it.
  3. `reconcile` -> the same takeover. `job_reconcile.reconcile_stale_running` either becomes the
     pure "is this record repairable" helper used by the takeover or is deleted (no dead code).
  4. `reconcile_all_jobs` becomes async; the lifespan awaits it.
  5. `HEARTBEAT_STALE_S = LEASE_TTL_S`.
- Red: `tests/test_lease_upgrade_no_doc.py`: a `running` record with no lease doc and a heartbeat
  older than `LEASE_TTL_S` is acquired and repaired; with a fresh heartbeat it is busy.
- Red: a sibling's freshly started job (lease held, no heartbeat file yet) is left alone by a second
  process's startup reconcile (Q-D4; fails today because a missing heartbeat reads as stale).
- Control: the B0 tests `tests/test_job_state_read_only.py` and `tests/test_atomic_write_concurrent.py`
  still pass (status polls do not write; a stale dataset import still resumes).

### 1A-6 Convert the 14 start-lock call sites (M)

- Files: the 11 files of the "Calls" table in section 4 other than `label_ops/store.py` and
  `promote_json.py` (those are 1A-8 and 1A-9).
- Steps per site: replace `with exclusive_start_lock(...) as acquired:` by
  `result = await acquire(client, kind, scope, job_id, owner)`; on `Busy` raise the existing busy
  error; write the record; start the run with a `LeaseHeartbeat`; release (or fenced terminal
  write) in `finally`. Delete the `start.lock` / `scan.lock` paths and the `job_lock` import.
  Resume paths take the lease over (`takeover_expired`) before rewriting the state to `queued`.
- Red per family, in `tests/test_job_lease_cross_process.py` (parametrised): 8 spawned
  `create_job` / `start_job` calls at a barrier give exactly one start.
- Control: the existing per-family tests of each module (`rg --files tests | rg
  "reprocess|assist|dataset_import|item_scores|probe|promote|combine|bulk_ingest|legacy_repair"`)
  pass unchanged except where they asserted `start.lock`.

### 1A-7 The four unlocked starts (Q-D2, Q-D12) (M)

- Files: `src/services/curation/autolabel/job.py`, `selection/job.py`, `embedding_viz.py`,
  `clustering/region_cluster_jobs.py` (both start functions).
- Steps:
  1. `selection`, `embedding_viz`: acquire before `_is_busy()` and the `running` write; heartbeat
     thread from the first write (today `running` is written with no heartbeat, Q-D4).
  2. `autolabel`: start-only lease (5.4). Leave `get_state`'s `_fail_unclaimed` repair as it is:
     B2 wave 3 deletes it (B0 open choice, recorded here as "left for B2").
  3. Region clustering and FP centroids move to `FileJob` (status, heartbeat, cancel) under the
     lease; delete `_JOB_STALE_S` and the `running` flag logic. Keep the served `job.json` wire
     shape; if it must change, update `contracts/` (`make contracts`) and Cropwright in the same
     commit.
- Red: the 8-process race for each of the four (today more than one starts); for autolabel,
  exactly one `trigger.json` and one `job_id` in `state.json`.
- Control: a single start still returns 202 and its snapshot; a second start after completion
  succeeds.

### 1A-8 Label operations (S)

- File: `src/services/curation/label_ops/store.py` (`write_lock`, `undo_lock`, `_lock_is_held`,
  `served_status`), callers `label_ops/record.py`, `label_ops/undo.py`.
- Steps: `write_lock(op_id)` -> lease `label_op:<op_id>` held with a heartbeat thread for the
  write; `undo_lock()` -> lease `label_undo`; `served_status` reads the lease (absent or expired
  while the record says `writing` / `undoing` -> `interrupted` / `undo_interrupted`), so it becomes
  async. Delete `_lock_is_held` and the `fcntl` import.
- Doc sync (same commit): update `docs/design/cq3_label_operations.md` sections 3 (the locks row),
  6.4 and 9 (crash during undo) to name the leases `label_op` and `label_undo` instead of
  `write.lock`, `undo.lock` and `flock`.
- Red: a `writing` record whose lease expired is served `interrupted`; two concurrent undos give one
  run and one `label_op_busy`.
- Control: a live writer's record is served `writing`.

### 1A-9 `promote.json` writers (S)

- File: `src/services/training/promote_json.py`; callers `routers/curation/_models_sharing.py`,
  `services/training/triton_promote.py`.
- Correction to the former master plan's 3b: "only the holder of the `promote` lease writes it" misses the
  sharing route, which is a second writer outside any promote job. Design: both writers take the
  short `promote_json:<model_name>` lease in the model owner's project for the read-compare-write,
  then release. The `sharing_revision` compare stays (it is the caller-facing OCC token). Writes go
  through `atomic_write_json` (delete `_write_atomic`). The functions become async (they hold an
  OpenSearch lease), so drop the `asyncio.to_thread` wrappers at the callers.
- Red: two concurrent `update_sharing` calls with the same `expected_revision` give one success and
  one `SharingRevisionConflictError`, in two spawned processes; a re-promote racing a sharing write
  keeps the new `shared` value.
- Control: a single sharing write bumps the revision by 1.

### 1A-10 Trainer and training-job claims, create-only generations (D-15) (S)

- Files: `docker/trainer/job_protocol.py`, `docker/trainer/trainer.py` (`process_one`),
  `src/services/training/jobs.py` (`write_job`).
- Steps:
  1. Claim: before `run_job`, create `<job_id>.claim.g1` with `os.open(path, O_CREAT | O_EXCL |
     O_WRONLY)`; on `FileExistsError` skip the job. The holder is the highest generation present.
     Write `{owner, generation, claimed_at}` and touch the job heartbeat.
  2. Stale reclaim: when the holder's heartbeat is older than the stale window, create
     `<job_id>.claim.g<N+1>` with `O_EXCL`; exactly one reclaimer wins, the rest get
     `FileExistsError` and back off; the winner then removes generations at or below N. Never
     unlink-then-create, never rename over a live claim.
  3. Fencing: before each status write the holder checks that no `g > N` exists; if one does it
     stops.
  4. Delete the claim files on a terminal state.
  5. API create: write the job spec with `O_EXCL` instead of `target.exists()` then write; a
     collision raises the existing duplicate error (409). Keep the order "duplicate check before
     pinning the registry snapshot" (the docstring explains why).
- Red: `tests/curation/trainer_protocol/test_job_claim_generations.py` (the trainer tests live in
  `tests/curation/trainer_protocol/`, not `tests/trainer/`; corrected): 4 spawned reclaimers on one
  stale claim, exactly one wins `g2`; the stale holder's fence fails; two `process_one` calls in two
  processes, one runs; the API create collision gives 409.
- Control: a single trainer claims `g1`, runs, and removes its claim.
- Note: the trainer image changes, but only the claim protocol; the Ultralytics pin and the
  trainer's export do not change until B3 (`04_b3_ultralytics_one_version.md`).

### 1A-11 Export worker reconcile fails only stale claims (S)

- File: `export/export_worker.py` (`reconcile_claimed`, `_heartbeat_loop`).
- Correction: there is no per-task heartbeat today. Add one: the existing heartbeat thread also
  touches `claimed/<task_id>.heartbeat` for the job it runs. `reconcile_claimed` fails only claimed
  jobs whose task heartbeat is missing or older than the stale window.
- Red: a claimed job with a fresh task heartbeat (a second worker's job) is not failed at start.
- Control: a claimed job with a stale heartbeat is failed with the existing message.

### 1A-12 OS-3: three bulk writers report item errors (S)

- Files: `src/services/curation/item_scores/job.py` (`opensearch.bulk(` in the score write),
  `embedding_viz.py` (counts `len(chunk_ids)` regardless of outcome), `probe_predictions.py` (the
  early-return flush).
- Library: the B0 OS-2 helper `failed_bulk_items(resp)` in `src/clients/opensearch/bulk.py`.
- Steps: use the helper at the three sites; a job with failed items reports the failed count and
  ends `failed` when nothing was written.
- Red per site: a fake bulk with `errors: true` makes the reported count or status reflect the
  failure.
- Control: an all-OK bulk is unchanged.
- Done when: a one-off test asserts no bare awaited `bulk(` result is discarded in
  `src/services/curation` (a test, not a ratchet; audit OS-3).

### 1A-13 Shrink-only file-lock ratchet (S)

- Files: `scripts/codegen/check_file_locks.py` (new), `scripts/codegen/baselines/file_locks.txt`
  (new), `.pre-commit-config.yaml` (one hook, `pass_filenames: false`, `always_run: true`, like
  `check-process-ids`), CONTRIBUTING.md (one line in the ratchet list).
- Pattern: the existing ratchets and `scripts/codegen/_baseline.py` (`key<TAB>value<TAB>reason`;
  `compare()` fails on new debt, growth, loosened or stale entries).
- Scope: tracked `.py` files under `src`, `scripts`, `export`, `docker`. Count per file the matches
  of `fcntl\.(flock|lockf)\(`, `exclusive_(start|file)_lock\(`, and imports of `filelock` or
  `portalocker`.
- Baseline after this track (only the B2/B4 rows): `bulk_ingest/dispatcher.py`,
  `open_vocab_sweeper.py`, `source_image_cache.py`, `event_hub.py`, `job_lock.py`, each with its
  reason and the batch that removes it.
- Red: a fixture tree with a new `fcntl.flock(` in a scoped file fails the guard.
- Control: the real tree passes.

### 1A-14 Live lease test (S, runs in the batch sweep)

- File: `tests/live/test_job_lease_live.py` (marker `live`).
- Steps: the same 8-process race against the live harness OpenSearch, 5 repetitions in the B1 live
  smoke check (cap 30 minutes, [00_index.md](00_index.md) section 8.1); record p50 and p99 acquire
  latency in the test output. The 20-repetition form is a local run, not a harness run.
- Red: not applicable on the base (the module is new); the paired control is one sequential
  acquire / release.

### 1A-15 Docs (S)

- `docs/CURATION.md`: busy window after a crash (at most 120 s), 409 busy codes, label-op
  `interrupted` delay, fencing residual risk.
- `docs/ARCHITECTURE.md`: the lease document and the heartbeat thread.
- `docs/opensearch_schema_design.md`: lease doc id, shape, `lease` mapping key in both layouts.
- `CLAUDE.md` "Lock rule" bullet and the curation rules: the cross-process claim is the OpenSearch
  lease; no file locks (name the ratchet).
- `INSTALLATION.md`: one line under multi-host (TTL must exceed clock skew; not supported today).
- Run `.venv/bin/python scripts/docs/check_docs_vs_code.py --only <file>` per file touched.

## 7. Gate for this lane

- Targeted: every red test above seen failing on the base and passing after, plus its control; the
  spawn races (lease, trainer claim) run 10 times in a loop in one pytest invocation
  (`pytest --count` is not installed; parametrise with `range(10)`).
- Static on touched files only: `.venv/bin/pre-commit run --files <touched files>` (ruff, mypy,
  bandit, layering, file size, the new file-lock ratchet).
- `make contracts` in any commit that changes `ErrorCode` or a served job shape.
- No full offline suite, no `--all-files`, no Docker in the lane. The batch sweep is in
  `02_b1_overview.md` section 7.

## 8. Done when

- `rg -n "exclusive_(file|start)_lock\(" src | rg -v job_lock.py` lists only
  `bulk_ingest/dispatcher.py` and `open_vocab_sweeper.py`.
- `rg -n "fcntl" src` lists only `job_lock.py`, `source_image_cache.py`, `event_hub.py`.
- `rg -n "heartbeat_ticker|_JOB_STALE_S|_state_lock" src` finds nothing.
- The file-lock baseline lists only the B2/B4 rows and `job_lock.py`.
- Every red test is green with its control; the live lease test passes in the sweep.
- The closing B1 PR moves the keep-list below to its targets and deletes this file
  ([00_index.md](00_index.md) "Plan lifecycle").

### 8.1 What to keep after merge

| Fact or section | Target doc |
|---|---|
| the lease document, kinds and scopes, TTL 120 s, self-fence 90 s, heartbeat thread, fencing generation (sections 5-6) | `docs/ARCHITECTURE.md`, `docs/opensearch_schema_design.md` |
| busy window, 409 busy codes, label-op `interrupted` delay, fencing residual risk | `docs/CURATION.md`, `docs/design/curation_api_contract.md` |
| the lock rule and the file-lock ratchet | `CLAUDE.md`, `CONTRIBUTING.md` |
| p50 and p99 acquire latency | `docs/PERFORMANCE.md` |
| the Breaking entry (OpenSearch needed to start a job; 120 s busy window) | `CHANGELOG.md` |

## 9. Docs, contracts, CHANGELOG

- Docs: section 6, task 1A-15.
- Contracts: only if a busy `ErrorCode` is added or the region-cluster job shape changes.
- CHANGELOG `[Unreleased]`: "Fixed: two requests can no longer start one singleton job twice;
  a reconcile never flips a sibling's live job; a blocked event loop no longer loses a job's
  liveness" and a **Breaking** bullet: "a crashed job blocks a new start of the same kind for up to
  120 s (was 30 s); OpenSearch must be reachable to start a job (503 otherwise)". Trainer: "a
  trainer replica claims a job create-only". OS-3 in Fixed.

## 10. Upgrade impact (0.6.0 -> 0.7.0)

- The two mapping keys are applied by the bootstrap on first start (schema fingerprint changes, so
  every project runs its migrations once).
- Lease docs appear on first start of each job kind.
- In-flight in-API jobs die with the container recreate, as on any upgrade; resumable ones show
  `interrupted` after the mid-upgrade rule (5.8).
- Trainer: a job file written by 0.6.0 with no claim file is claimed `g1` by the new trainer.

## 11. Rollback

- Before the PR merges: the coordinator reverts the 1A merge commit (`git revert -m 1 <merge>`) on
  `int/b1-correctness`. 1B depends on 1A (mapping keys), so revert 1B first.
- After release: lease docs and the two mapping keys are inert for older code (unmapped object
  keys in a fixed-id doc). No data migration to undo.

## 12. Risks

| Risk | Mitigation |
|---|---|
| A holder pauses longer than the TTL (GC, a blocked thread) and is taken over (M4) | thread heartbeat survives a blocked loop; TTL 120 s; self-fence at 90 s; fenced terminal writes; `lease_generation` on records |
| OpenSearch outage blocks every job start (M5) | fail closed with 503 (rule); `/health` already reports OpenSearch |
| Lease doc lost with a project delete while a job runs (M6) | next renew is `Lost`, the job stops |
| A round trip per start and per 15 s tick | negligible next to a job; measured in the live test (p50/p99) |
| A lane edits the same mapping lines as 1B | 1A owns both mapping keys; 1B never edits `bodies_other.py` or `_migrations()` |
| Async conversion of `repair_if_stale` / `served_status` ripples into sync callers | the callers are routes or async jobs already; check each with `rg` before converting, keep the change mechanical |

## 13. Corrections found while verifying

- 3b count: 55 hits in 19 files (not 51 in 18).
- Combine's start lock is per combine job (`store.directory / 'start.lock'` in `resume`), and combine
  jobs are global: scope `<combine_job_id>`, doc in the target project's index (master said scope
  none).
- `promote_json.py` has two writers (sharing route and re-promote); a `promote` lease does not cover
  the sharing route: lease `promote_json:<model_name>` instead.
- Trainer tests live in `tests/curation/trainer_protocol/`; `tests/trainer/` does not exist.
- `export_worker` has no per-task heartbeat to check; 1A-11 adds one.
- `autolabel/job.py` has its own `_tick()` ticker besides `heartbeat_ticker`.
- `reconcile_all_jobs` is sync; lease operations make it async.

## 14. Lane agent brief (paste as is)

```
You are the implementation agent for B1 track 1A (job leases) of OpenProcessor.
Plan: docs/design/program/02a_b1_job_leases.md (this file). Execute it standalone.
Read first: CLAUDE.md, docs/VISION_AND_GOALS.md, docs/design/engineering_standards.md,
docs/design/program/02_b1_overview.md sections 3-6, and audit plan
docs/design/codebase_audit_plan.md 4.2 (OS-3) for detail.

Worktree: <repo>-wt-b1-leases, branch b1/leases (created by the coordinator from
int/b1-correctness). Work ONLY there. Never push, never merge, never check out another branch.
Compose project for any Docker command: -p op-b1-leases (you should need none). GPU: none.
Deadline: <coordinator fills in>.

Rules:
- One commit per task 1A-1 .. 1A-15, conventional commits, hooks on (never --no-verify).
- For each task: write the red test, run it on the unchanged code and SEE IT FAIL, paste the red
  output into the commit body; then fix; then the paired control passes.
- Targeted tests only: .venv/bin/python -m pytest <files> -q --no-cov. No full suite, no
  pre-commit --all-files. Static checks: .venv/bin/pre-commit run --files <touched files>.
- Cite the official doc for each mechanism (section 3) where the code is not obvious.
- No in-process or file locks; no asyncio fan-out in request handlers; fail closed (OpenSearch
  error = 503, never "proceed without a lease"); no dead code, no shims; class identity by name.
- Never use pgrep -f, pkill -f, or sleep/until polling loops. Wait on a process by its PID or use
  the tool's background notification.
- make contracts in the same commit as any ErrorCode or wire change.
- Do NOT edit bodies_other.py beyond 1A-1, and do not touch class_registry code (track 1B owns it).
- If a file would grow past its scripts/codegen/baselines/file_size.txt value, split it by concern;
  never raise a baseline.

Report at the end: commits (sha + subject), each red output in one line, anything you could not
do and why, and any correction to this plan you found.
```
