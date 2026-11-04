# B2 stage 4: the API-hosted job families move to Celery

| Field | Value |
|---|---|
| Status | plan, not started |
| Depends on | stage 3 merged ([03e](03e_b2_s3_job_runtime.md)): `start_job`, `register_job`, `JobContext`, the reaper, `publish` |
| Design | [02b](02b_jobs_locking_events.md) sections 1-3 (the kind table is authoritative for queue, exclusivity, auto-resume and limits) |
| Lanes | 4A, 4B, 4C, 4D in parallel; 4E (frontend) after 4A-4D merged and `make contracts-check` is green |
| Merge order | 4A, 4B, 4C, 4D, 4E |
| Runs alongside | stage 5 ([03f](03f_b2_s5_gpu_queues.md)), disjoint files |
| Compose project | none in the lanes; compose changes go to lane 6B through the PR notes |

---

## 1. The recipe (every family, one family per commit)

1. **Kind and route.** Add the kind to the lane's `src/config/job_kinds/<lane>.py` and its route to
   `src/config/task_routes/<lane>.py`, exactly as [02b](02b_jobs_locking_events.md) section 2.1.
2. **Start.** The start route calls `start_job(kind, project, params=..., scope=..., idempotency_key=...)`
   and returns 202 with the job id. Domain rows (import, combine, run) are inserted through
   `start_job`'s `with_rows=` hook in the same transaction. Delete the module's `asyncio.create_task`,
   its `_tasks` / `_active_tasks` / `_TASKS` dict, its B1 lease acquire and release, its
   `reconcile_orphaned_jobs`, and its entry in `startup_reconcile.JOB_MODULES`.
3. **Task.** In `src/tasks/<lane module>.py`: `register_job('<kind>', handler)` where `handler(ctx)`
   calls the **existing** run coroutine with clients from `ctx` (never from a request).
4. **State.** Replace `FileJob` reads and writes: progress through `ctx.progress`, resume cursors
   through `ctx.checkpoint`, cancel through `ctx.cancelled()`, results as the handler's return value.
   Family-specific phases become `progress.stage`. Delete the family's job directory constant.
5. **Status and cancel routes** read the `jobs` row (plus the family's domain row) through the one
   serializer; add `queued` to the status union where it is missing (`rg -n "Literal\[" src/routers/curation/_*_models.py src/schemas/curation | rg running`),
   `make contracts` in the same commit. Cancel calls `cancel_job`.
6. **Bounded concurrency (CC-9).** Where the family's code has an uncapped `gather` over a large
   list, rewrite it as `asyncio.TaskGroup` with an `asyncio.Semaphore` bound
   ([task groups](https://docs.python.org/3/library/asyncio-task.html#task-groups)).

**Per-family tests (red before the change):**
- the start route inserts one `queued` row and sends one message with `task_id == job_id` (fake
  producer); a second start while active is 409 `job_busy` (for exclusive kinds);
- broker down -> 503 `queue_unavailable`, row `failed`;
- the registered task runs the existing coroutine once through the wrapper (`db` fixture, fake
  OpenSearch);
- cancel while `queued` -> `cancelled`; cancel while running -> `cancelling` then `cancelled`;
- for resumable kinds: an `interrupted` job resumed by the reaper continues from its checkpoint (the
  item count at the end equals the total, no duplicate writes on a fake);
- control: the family's existing status-route test passes (with `queued` added where the model
  changed).

## 2. Lane 4A `b2/families-a`

**Owns:** `src/services/curation/{reprocess_job,reprocess_images,reprocess_open_vocab,probe_job,embedding_viz}.py`,
`src/services/curation/assist/**`, `src/services/curation/item_scores/**`,
`src/services/curation/selection/**`, `src/services/curation/label_ops/{record,undo}.py`, their
routers (`rg -ln "reprocess_job|assist\.runner|item_scores|probe_job|selection\.job|embedding_viz|label_ops" src/routers`),
`src/tasks/jobs_a.py`, `src/config/job_kinds/families_a.py`, `src/config/task_routes/families_a.py`,
tests under `tests/curation/` for these families. `item_scores/near_dup.py` is in 4A too (moved from
4C with its task, so every near-duplicate change sits in one lane).

| ID | Kind | Notes |
|---|---|---|
| 4A-1 | `reprocess` | the template family; resumable; its target list in `job_items` when given by id; CC-9 in `reprocess_open_vocab.py` |
| 4A-2 | `assist_run` | members in `job_items` (was `members.json`); the wait on the region worker is fine in a worker; CC-9 in `assist/sample_eval.py`; doc sync note for 6D: `selectable_assist_plan.md` "Deviations" names the reaper instead of `reconcile_orphaned_jobs` |
| 4A-3 | `item_scores`, `probe`, `selection` | add `queued`; `k_center_greedy` runs in the worker |
| 4A-4 | `embedding_viz` | writes `projection_states` and the reducer through the storage seam (stage 2C API); UMAP and cuML import inside the task (trap T-29) |
| 4A-5 | `label_op`, `label_undo` | inline jobs (`start_inline_job`): exclusivity and history in `jobs`, records already in Postgres (2D); the B1 `LABEL_OP` and `LABEL_UNDO` leases deleted |
| 4A-6 | fail-closed config (was 4C-8) | `item_scores/near_dup.py::_threshold()` silently uses the default on an invalid `OP_CROP_DUP_THRESHOLD`; validate it in the startup config load and drop the fallback. Red: `abc` raises at load; control: unset gives 0.97 (the clustering plan's N-8 is the same fix) |
| 4A-7 | #473 part a (folded in, [00b](00b_traceability.md) section 3) | a re-score recomputes the crop `dup_*` fields from the new vectors instead of leaving the old group fields. Red: `tests/curation/test_rescore_refreshes_dup_fields.py` (fake OpenSearch): after a re-score with changed vectors an item leaves its old group; red today. Control: a re-score with unchanged vectors keeps the groups |

**4A done when:** `rg -n "_active_tasks|_tasks\[|file_job|job_lease" src/services/curation/{reprocess_job,probe_job,embedding_viz}.py src/services/curation/{assist,item_scores,selection,label_ops}` is empty.

## 3. Lane 4B `b2/families-b`

**Owns:** `src/services/curation/clustering/region_cluster_jobs.py`,
`src/services/curation/dataset_import/**`, `src/services/projects/combine/**`,
`src/services/classes/projection_sync.py` (new), `src/routers/curation/{datasets,classes_merge}.py`,
the combine and region-cluster routers, `src/tasks/jobs_b.py`, `src/config/job_kinds/families_b.py`,
`src/config/task_routes/families_b.py`, tests for these families.

| ID | Kind | Notes |
|---|---|---|
| 4B-1 | `region_cluster`, `region_fp_centroids` | markers (`refine_marker`, `partition_marker`) to `project_state`; `job.json`, `fp_job.json` deleted |
| 4B-2 | `dataset_upload_extract` | the upload route streams to storage (`dataset_uploads` row `receiving` -> `received`) and starts the job; 202 with the job id (CC-5); no extraction in the request |
| 4B-3 | `dataset_import`, `dataset_import_undo` | `dataset_imports` and the `dataset_import_images` ledger (pending rows claimed in batches with `FOR UPDATE SKIP LOCKED` by the one running job, so a resume continues where it stopped); undo keeps anything a human edited since; the resume route's inline rescan (`heartbeat_ticker` in `routers/curation/datasets.py`) moves into the task |
| 4B-4 | `combine` | `combine_runs`, `combine_sources`; the job runs in the target project's scope and binds each source in turn inside `execute.py` (Q-E20); progress chunks in `ctx.checkpoint`; the global `/jobs/combine` directory deleted |
| 4B-5 | `class_merge`, `class_projection_sync` | merge leaves the request (it ran an `update_by_query` inline); `class_projection_sync` consumes `search_projection_tasks` rows with one `update_by_query` per row filtered by `class_id`, marks `done_at`; the D-47 verifier shares its query |

Red tests beyond the recipe: `tests/curation/test_upload_extract_job.py` (no `safe_extract` call in
the handler; the task extracts); `tests/curation/test_dataset_import_resume_job.py` (no rescan in the
route); `tests/curation/test_combine_task_binds_target.py`;
`tests/classes/test_projection_sync.py` (`db` + fake OpenSearch: a rename updates exactly the items
with that `class_id`; a re-run is a no-op).

**4B done when:** `rg -n "_TASKS\[|_tasks\[store|ImportStore|/jobs/combine|heartbeat_ticker|file_job|job_lease" src/services/curation/dataset_import src/services/projects/combine src/services/curation/clustering/region_cluster_jobs.py src/routers/curation/datasets.py` is empty.

## 4. Lane 4C `b2/families-c`

**Owns:** `src/services/curation/bulk_ingest/**` (except the dispatcher's tick loop, which 6A turns
into a beat task: 4C changes only what the tick launches), `src/services/curation/open_vocab_ingest.py`,
`src/services/curation/upload_ingest.py` (its open-vocab scheduling call), `src/services/projects/delete.py`
(the finish; 2A owned the lifecycle states), `src/routers/curation/projects.py` (the DELETE route's
launch only), `src/routers/curation/ingest_sources.py`, `src/routers/curation/ingest.py` (its
open-vocab call), `src/routers/ingest/directory.py`, `src/routers/persons.py`, the face grouping in
`src/services/face_identity*`,
`src/tasks/jobs_c.py`, `src/config/job_kinds/families_c.py`, `src/config/task_routes/families_c.py`,
`tests/test_no_module_level_async_locks.py`, tests for these families.

| ID | Kind | Notes |
|---|---|---|
| 4C-1 | sources and the file queue | `ingest_sources` replaces `source.json` and `state.json`; `ingest_source_files` replaces `queue.sqlite` and the run ledger (`failures.jsonl` becomes `status = 'failed'` with `error`); the per-source scan state in `ingest_sources.scan_state` |
| 4C-2 | `bulk_ingest_run` | claims pending files in batches (`FOR UPDATE SKIP LOCKED`, 500 per batch); sends to `OP_API_LOOPBACK_URL` (`http://api:8000` in the worker env, for 6B); deletes `run_coroutine_threadsafe`, `_create_task`, the `loop=` argument, `start.lock` (`bulk_ingest/runner.py:117`); `start_run_now` stops passing its own address |
| 4C-3 | `source_poll` | a poll action starts the job (exclusive per source id) instead of `create_task(_poll)`; `_scans`, `_loops`, `_on_task_loop`, `scan.lock` (`source_scan.py:28`) deleted |
| 4C-4 | `open_vocab_drain` | one exclusive job per project: a start while one is queued or running hits the unique index and is reported as "already pending" (this replaces both `_PASS_LOCK` and the debounce marker of the old plan); `wait_for_scheduled` and `_tasks` deleted; `ALLOWED` in `tests/test_no_module_level_async_locks.py` becomes empty (the last CC-11 site); doc-sync note for 6D: `sam3_full_image_detection_plan.md` section 16 Wave 5 |
| 4C-5 | `project_delete` | deployment-level row (`project_id` NULL, scope slug), idempotent (the project row's status is the claim); `_BACKGROUND_DELETE_TASKS` and `_FINISH_IN_PROGRESS` deleted; drains (cancels and waits for) the project's active jobs before removing indexes and storage (trap T-27) |
| 4C-6 | `directory_ingest` | D-22a: deployment-level row; `POST /ingest/directory` returns 202 + job id and a `GET` status route; first prove the visual-search ingest is idempotent per image (`rg -n "image_id\|def .*hash" src/services/visual_search/ingest_batch.py`): a re-run must not duplicate documents, else stop and report |
| 4C-7 | `persons_cluster` | `POST /persons/cluster` returns 202 + job id; the O(n^2) range search and connected components run in the worker; `GET` reads the result |
| 4C-8 | (moved to 4A-6 with `near_dup.py`) | - |

Red tests beyond the recipe: `tests/curation/test_ingest_source_files_claim.py` (`db`: two runners
claim disjoint batches); `tests/curation/test_dispatcher_starts_poll_job.py` (no asyncio task);
`tests/curation/test_open_vocab_coalesce.py` (two ingest requests -> one pending job);
`tests/curation/test_project_delete_finish_idempotent.py`; `tests/test_directory_ingest_job.py`;
`tests/test_persons_cluster_job.py`.

**4C done when:** `rg -n "_PASS_LOCK|_BACKGROUND_DELETE_TASKS|_FINISH_IN_PROGRESS|run_coroutine_threadsafe|_scans\b|_loops\b|queue\.sqlite|start\.lock|scan\.lock" src` is empty; the file-lock baseline loses the `runner.py` and `source_scan.py` lines.

## 5. Lane 4D `b2/trigger-workers`: auto-label and the assist loop become queue consumers

**Owns:** `src/services/curation/autolabel/**`, `src/services/curation/assist_loop/**`,
`scripts/curation/auto_label_worker.py` and `scripts/curation/assist_loop_worker.py` (both deleted:
the services run Celery), `scripts/curation/worker_liveness.py` entries for those two,
`src/tasks/trigger.py`, `src/config/job_kinds/trigger.py`, `src/config/task_routes/trigger.py`,
tests for both.

| ID | Kind | Notes |
|---|---|---|
| 4D-1 | `autolabel` (queue `autolabel`) | `start_job` replaces `trigger.json`; `_run_one`'s body becomes the handler; the poll loop, `_claim_trigger_for`, `_oldest_pending_trigger`, `running.lock`, the exit-code file, `QUEUE_MAX_AGE_S` and the in-memory IVF `last_check` deleted (the reaper's rule d and the beat task of 6A replace them); `job_history` archive replaced by `jobs` history; CC-9 in `autolabel/vlm_stage.py`; the embed stage reads only `TRITON_URL` (ENV-2a: `OP_TRITON_URL` retired) |
| 4D-2 | `assist_loop` (queue `assist_loop`) | resumable; pause and resume through `pause_job` / `resume_job`; `claim_trigger`, the trigger file, `pause.flag`, `events.jsonl` replay (events are `events` rows), `budget.json` (into `jobs.params` and `checkpoint`), `repair_orphans` and the `ASSIST_LOOP` short lease deleted |
| 4D-3 | one liveness mechanism | the two services' healthcheck becomes the pool healthcheck (for 6B); the `worker_liveness.py` heartbeats of these two are deleted; if a dashboard reads the auto-label worker gauge (`ops_metrics.WORKER_LABELS`; `rg -n "auto_label_worker" monitoring/`), it is written from Celery's `heartbeat_sent` signal instead, else the label is deleted (ENV-20) |

Red tests: `tests/curation/test_autolabel_enqueue.py` (no trigger file; a second start is 409);
`tests/tasks/test_autolabel_task.py` (two deliveries -> one run; cancel at a stage boundary);
`tests/curation/test_assist_loop_enqueue.py`, `tests/tasks/test_assist_loop_task.py` (resume after
`interrupted` continues from the checkpoint; pause stops at a boundary). Doc-sync note for 6D:
`auto_label_loop_plan.md` sections 0, 1, 3.1 and 8.

**4D done when:** `rg -n "trigger\.json|claim_trigger|_claim_trigger_for|running\.lock|QUEUE_MAX_AGE_S|pause\.flag" src scripts` is empty.

## 6. Lane 4E `b2/frontend-queued` (after 4A-4D merged)

- **Scope:** Cropwright renders `queued` (waiting, with cancel) for every family whose status union
  gained it, follows the 202 + job id flows (upload, directory ingest is API-only, person grouping if
  the UI calls it), shows `cancelling` and `paused`, reads SSE ids as opaque strings, and drops calls
  to deleted routes (`classes/sync_to_opensearch`). `frontend/CHANGELOG.md`.
- **Rules:** `$lib` aliases, 2-space indent, `npm` only; thin frontend (the backend says the state).
- **Tests:** a unit test per controller (red first); lane checks `npm run check`, lint, the touched
  tests, `npm run build`; the full frontend set and the browser check run in the B2 sweep.
- **Done when:** no status union in `frontend/src` lacks a value the contract has.

## 7. Merge and gate

The coordinator merges 4A, 4B, 4C, 4D (`--no-ff`), running `make contracts-check`, the merged lanes'
tests and `-m db` after each; then 4E. Facts for 6B (compose) and 6D (docs) are collected from the
PR notes: the worker env (`OP_API_LOOPBACK_URL`), the two curation services' commands and
healthchecks, the removed volumes and env vars.

## 8. Lane briefs

Use the common block of [03a](03a_b2_w1_queue_infra.md), with: "B2 stage-4 lane <4A..4E>", "cut
from int/b2-control-plane after stage 3 merged", the reading list: this file (sections 1 and your
lane), [02b](02b_jobs_locking_events.md) sections 1-3, and "Register tasks only in your lane's
modules; write compose and docs facts in your PR notes, do not edit compose files or shared docs."
4E adds the frontend rules above. Compose project: none. GPU: none.
