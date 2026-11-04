# B2 stage 4: the API-hosted job families move to Celery

| Field | Value |
|---|---|
| Status | plan, not started |
| Depends on | stage 3 merged ([03e](03e_b2_s3_job_runtime.md)): `start_job`, `register_job`, `JobContext`, the reaper, `publish` |
| Design | [02b](02b_jobs_locking_events.md) sections 1-3 (the kind table is authoritative for queue, exclusivity, auto-resume and limits) |
| Lanes | two parts (review RR-13, owner Q-A 2026-11-04), each closed by a part check. **W2.3a** (the B2 families, ports of today's behaviour): 4A, 4B, 4C, 4D in parallel with stage 5's 5A-5D. **W2.3b** (the research families, cut from the W2.3a head after its part check): 4F, ND-A and ES-B in parallel; then ND-B (cut after ND-A and 4F merged); then 4E (frontend, after every other W2.3a and W2.3b lane merged and `make contracts-check` is green) |
| Merge order | W2.3a: 5D, 4D, 4A, 4B, 4C, 5A, 5B, 5C. W2.3b: 4F, ND-A, ES-B, ND-B, 4E ([00_master_plan.md](00_master_plan.md) W2.3a and W2.3b). Why: 4D fixes the VLM client signature other callers use (4A-8); 4F fixes `record_partition`, `group_lineage` and `assign.place_new` that ND-A calls; ES-B reuses 4A's reprocess decode path (merged in W2.3a) and queues 4F's and ND-A's kinds; ND-B edits the item serializer and query builders after 4F and calls ND-A's groups |
| Wave size | W2.3a: 8 lanes, no cross-lane signature except 4D's VLM keyword. W2.3b: 3 lanes at once, then 2 in sequence; every W2.3b cross-lane call is against a signature fixed in this file |
| Research seams | folded 2026-11-04, approved by the owner 2026-11-04 (RD-1, every seam now; [00d](00d_owner_questions.md)): the clustering family becomes its own lane 4F (split from 4B, which keeps import, combine and class merge); the embedding re-index job (ES-B); the near-duplicate machinery (ND-A) and holdout guard (ND-B, replacing 2D-4); the assist loop's own record, pause, budgets and slices in 4D (gaps G-2, G-3); 4E follows every new control and wire. Seam map: [00b](00b_traceability.md) section 5.2 |
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
3. **Task.** In `src/tasks/<lane module>.py`: `register_job('<kind>', handler, client_factories=...)` (the OpenSearch factory is injected here, never imported by `src/tasks/runtime/`, R-2) where `handler(ctx)`
   calls the **existing** run coroutine with clients from `ctx` (never from a request).
4. **State.** Replace `FileJob` reads and writes: progress through `ctx.progress`, resume cursors
   through `ctx.checkpoint`, cancel through `ctx.cancelled()`, results as the handler's return value.
   Family-specific phases become `progress.stage`. Delete the family's job directory constant.
5. **Status and cancel routes** read the `jobs` row (plus the family's domain row) through the one
   serializer; add `queued` to the status union where it is missing (`rg -n "Literal\[" src/routers/curation/_*_models.py src/schemas/curation | rg running`),
   `make contracts` in the same commit. Cancel calls `cancel_job`.
6. **Tests that used fakes (review fix R-14).** The family's `FileJob` test modules (13 today) are
   rewritten in the same commit: a test whose outcome depends on a constraint, lock or RLS becomes a
   `db` test; only tests that check no SQL behaviour keep an in-memory fake. The lane's PR notes list
   every test module it deleted or rewrote.
7. **Bounded concurrency (CC-9).** Where the family's code has an uncapped `gather` over a large
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
tests under `tests/curation/` for these families. Wave W2.3a. `item_scores/near_dup.py` is ported
unchanged with the other scorers; ND-A deletes it in W2.3b (the research near-duplicate plan
replaces the bucketed scorer with run-scoped groups), so no W2.3a lane depends on a research lane.

| ID | Kind | Notes |
|---|---|---|
| 4A-1 | `reprocess` | the template family; resumable; its target list in `job_items` when given by id; CC-9 in `reprocess_open_vocab.py` |
| 4A-2 | `assist_run` | members in `job_items` (was `members.json`); the wait on the region worker is fine in a worker; CC-9 in `assist/sample_eval.py`; doc sync note for 6D: `selectable_assist_plan.md` "Deviations" names the reaper instead of `reconcile_orphaned_jobs` |
| 4A-3 | `item_scores`, `probe`, `selection` | add `queued`; `k_center_greedy` runs in the worker |
| 4A-4 | `embedding_viz` | writes `projection_states` and the reducer through the storage seam (stage 2C API); UMAP and cuML import inside the task (trap T-29) |
| 4A-5 | `label_op`, `label_undo` | inline jobs (`start_inline_job`): exclusivity and history in `jobs`, records already in Postgres (2D); the B1 `LABEL_OP` and `LABEL_UNDO` leases deleted |
| 4A-6 | (moved to ND-A as ND-9, W2.3b) | the `near_dup` scorer is ported unchanged here; ND-A deletes it, retires `OP_CROP_DUP_THRESHOLD` and adds the `enqueue_pending(..., reason='vector_changed')` calls to the reprocess and re-embed paths, all after 4A merged |
| 4A-7 | (superseded by ND-4 and ND-5) | #473 part a: groups are run-scoped and the projection clears every field of a superseded run, so a re-score can no longer leave stale group fields; the red test moves to ND-A (`tests/near_duplicates/test_rescore_clears_stale_groups.py`) |
| 4A-8 | VLM usage at the assist-run call site (research auto-label 5.5) | `src/services/curation/assist/` passes `usage_source='assist_run'` to the VLM client against the signature 4D fixes (4D merges first). Red: `tests/curation/test_assist_run_records_usage.py` (`db`): one assist run of 10 crops adds 10 crops to `vlm_usage_hourly` under `assist_run` |

**4A done when:** `rg -n "_active_tasks|_tasks\[|file_job|job_lease" src/services/curation/{reprocess_job,probe_job,embedding_viz}.py src/services/curation/{assist,item_scores,selection,label_ops}` is empty.

## 3. Lane 4B `b2/families-b`

**Owns:** `src/services/curation/dataset_import/**`, `src/services/projects/combine/**`,
`src/services/classes/projection_sync.py` (new), `src/routers/curation/{datasets,classes_merge}.py`,
the combine routers, `src/tasks/jobs_b.py`, `src/config/job_kinds/families_b.py`,
`src/config/task_routes/families_b.py`, tests for these families. (Region clustering moved to lane
4F, research clustering plan section 8.)

| ID | Kind | Notes |
|---|---|---|
| 4B-1 | (moved to 4F) | `region_cluster` becomes `cluster_run(target='region_box')`; `region_fp_centroids` moves with its file; the markers are never ported (they become partitions) |
| 4B-2 | `dataset_upload_extract` | the upload route streams to storage (`dataset_uploads` row `receiving` -> `received`) and starts the job; 202 with the job id (CC-5); no extraction in the request |
| 4B-3 | `dataset_import`, `dataset_import_undo` | `dataset_imports` and the `dataset_import_images` ledger (pending rows claimed in batches with `FOR UPDATE SKIP LOCKED` by the one running job, so a resume continues where it stopped); undo keeps anything a human edited since; the resume route's inline rescan (`heartbeat_ticker` in `routers/curation/datasets.py`) moves into the task |
| 4B-4 | `combine` | `combine_runs`, `combine_sources`; the job runs in the target project's scope and binds each source in turn inside `execute.py` (Q-E20); progress chunks in `ctx.checkpoint`; the global `/jobs/combine` directory deleted. Research embeddings 5.4 (pending RD-33): vectors are copied only for spaces whose `space_key` exists in the target, the rest marked `deferred`; the preview warning `embedding_model_mismatch` compares space keys, not dimensions (the registry is ES-A's, merged in W2.2); the near-duplicate pending rows (`added`) for every copied image and item are added by ND-A in W2.3b (ND-3) |
| 4B-5 | `class_merge`, `class_projection_sync` | merge leaves the request (it ran an `update_by_query` inline); `class_projection_sync` consumes `search_projection_tasks` rows with one `update_by_query` per row filtered by `class_id`, marks `done_at`; the D-47 verifier shares its query |

Red tests beyond the recipe: `tests/curation/test_upload_extract_job.py` (no `safe_extract` call in
the handler; the task extracts); `tests/curation/test_dataset_import_resume_job.py` (no rescan in the
route); `tests/curation/test_combine_task_binds_target.py`;
`tests/classes/test_projection_sync.py` (`db` + fake OpenSearch: a rename updates exactly the items
with that `class_id`; a re-run is a no-op).

**4B done when:** `rg -n "_TASKS\[|_tasks\[store|ImportStore|/jobs/combine|heartbeat_ticker|file_job|job_lease" src/services/curation/dataset_import src/services/projects/combine src/routers/curation/datasets.py` is empty.

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
| 4C-5 | `project_delete` | deployment-level row (`project_id` NULL, scope slug), idempotent (the project row's status is the claim); `_BACKGROUND_DELETE_TASKS` and `_FINISH_IN_PROGRESS` deleted; drains (cancels and waits for) the project's active jobs before removing indexes and storage (trap T-27). Review RR-7: the rows go **per table in reverse foreign-key order, read from the catalog** (`pg_constraint`, so a later revision's table is covered without an edit), in batches of 10,000 rows per transaction with progress in `ctx.checkpoint`, and the `projects` row last; every batch transaction first sets `op.project_delete` to the project id, which the immutability triggers honour only while the project is `deleting` ([02a](02a_postgres_schema.md) section 1.2, review RN-1); the cascade stays as the safety net. Red: `tests/projects/test_project_delete_batched.py` (`db`): a project with 200,000 `cluster_memberships` and `job_items` rows deletes under `SET statement_timeout = '30s'` and resumes after a kill mid-table; `tests/projects/test_project_delete_guarded.py` (`db`, RN-1): a project holding a `ready` partition, a config document with two revisions and a calibration deletes, and the same child deletes without the setting raise |
| 4C-6 | `directory_ingest` | D-22a: deployment-level row; `POST /ingest/directory` returns 202 + job id and a `GET` status route; first prove the visual-search ingest is idempotent per image (`rg -n "image_id\|def .*hash" src/services/visual_search/ingest_batch.py`): a re-run must not duplicate documents, else stop and report |
| 4C-7 | `persons_cluster` | `POST /persons/cluster` returns 202 + job id; the O(n^2) range search and connected components run in the worker; the result is kept where today's code keeps it, read by `GET` through one function `persons_grouping_result(session)` in the face grouping module. In W2.3b 4F-9 switches that function and the write to a `cluster_partitions` row (`target = 'face'`, `method = 'union_find'`, project NULL, deployment scope; research clustering 5.2, the plan's smaller option) |
| 4C-8 | (moved to ND-A as ND-9 with `near_dup.py`) | - |

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
tests for both; research auto-label plan section 5: new `src/db/repos/assist_loop.py`, new
`src/domain/assist_loop_events.py` and `src/domain/assist_loop_metrics.py` (pure),
`src/schemas/curation/assist_loop.py` (the loop's wire models, built in
`src/services/curation/assist_loop/`, not in `wire.py`, which serializes items and boxes and is 4F's
in this wave), the assist-loop routers, `src/services/labeling/vlm_client.py`
(`usage` capture and `record_vlm_usage`), and every VLM call site whose file no other W2.3a lane owns
(the coordinator lists them at the cut with `rg -n "vlm_client" src scripts`; the VLM worker and
region verification are expected there).

| ID | Kind | Notes |
|---|---|---|
| 4D-1 | `autolabel` (queue `autolabel`) | `start_job` replaces `trigger.json`; `_run_one`'s body becomes the handler; the poll loop, `_claim_trigger_for`, `_oldest_pending_trigger`, `running.lock`, the exit-code file, `QUEUE_MAX_AGE_S` and the in-memory IVF `last_check` deleted (the reaper's rule d and the beat task of 6A replace them); `job_history` archive replaced by `jobs` history; CC-9 in `autolabel/vlm_stage.py`; the embed stage reads only `TRITON_URL` (ENV-2a: `OP_TRITON_URL` retired). The worker's IVF retrain check (`_ivf_retrain_trigger` in `scripts/curation/auto_label_worker.py`) is ported unchanged into one new module `src/services/curation/autolabel/ivf_check.py` (`maybe_retrain(session, project)`) with one call in `autolabel/job.py`, so lane 4F deletes it in W2.3b without touching anything else (it becomes a `cluster_run`); `autolabel` is pausable between stages and VLM chunks (pending RD-69) |
| 4D-2 | `assist_loop` (queue `assist_loop`) | **research G-3: the loop's durable record moves to the `assist_loop_*` tables** ([02d](02d_research_seam_schema.md) section 2.4; [research_auto_label_assist_plan.md](../research_auto_label_assist_plan.md) section 5); `events` rows are the 24-hour UI feed only, and `jobs.params` holds only the normalized request reference; `claim_trigger`, the trigger file, `pause.flag`, `events.jsonl` replay, `budget.json`, `repair_orphans`, the `ASSIST_LOOP` short lease, `assist_loop/rollback.py` and `store.py` deleted (rollback is the label-operation undo over the run's propagation ops) |
| 4D-3 | one liveness mechanism | the two services' healthcheck becomes the pool healthcheck (for 6B); the `worker_liveness.py` heartbeats of these two are deleted; if a dashboard reads the auto-label worker gauge (`ops_metrics.WORKER_LABELS`; `rg -n "auto_label_worker" monitoring/`), it is written from Celery's `heartbeat_sent` signal instead, else the label is deleted (ENV-20) |
| 4D-4 | loop control (auto-label 5.3, 5.8) | the pause route `POST /curation/projects/{project}/assist/loops/{loop_id}/pause` and the budget route `PATCH .../{loop_id}/budget` (both new; confirm the prefix against the existing routes in `src/routers/curation/assist_loop.py` first); resume answers 409 `resume_blocked` with the reason; cancel is terminal (pending RD-70); `AssistLoopControl` flags (`can_pause`, `can_resume`, `can_cancel`, `can_edit_budget`, `can_rollback`, `resume_blocked_reason`) decided by the backend; `on_budget: pause` default (pending RD-71) |
| 4D-5 | budgets (auto-label 5.5) | one SQL reservation per unit before every upstream call (crops, requests, completion tokens as the exact `max_tokens`, prompt tokens as an estimate), settle from the endpoint's `usage`, `usage_missing` charged at the reserved upper bound, forfeit unsettled reservations on resume; `record_vlm_usage` for every VLM call (`usage_source` is a required keyword of the client; callers in other W2.3a lanes add it against this signature) |
| 4D-6 | slices and system pauses (auto-label 5.3, gap G-2) | the handler checks `ctx.paused()` before every reservation and node admission, returns `Yield` at a node boundary after `slice_s` (600 s, pending RD-77), and requests its own pause on a failed audit, a reached budget, a paused project or a VLM taken for training |
| 4D-7 | events (auto-label 5.4) | `assist_loop.nodes`, `.budget`, `.audit`, `.iteration`, `.warning` and the `pause_reason`, `stop_reason`, `control` fields on `job.state`, each a snapshot under 7,000 bytes; the loop reads clusters only through `assist_loop/partition.py`'s `PartitionReader` (today's `ClusterFieldPartition`, unchanged here); 4F-10 adds the partition-backed reader that serves `(partition_id, cluster_uid)` in W2.3b, so no other loop file changes then |

Red tests: `tests/curation/test_autolabel_enqueue.py` (no trigger file; a second start is 409);
`tests/tasks/test_autolabel_task.py` (two deliveries -> one run; cancel at a stage boundary);
`tests/curation/test_assist_loop_enqueue.py`, `tests/tasks/test_assist_loop_task.py` (resume after
`interrupted` continues from the checkpoint; pause stops at a boundary). Research red tests (each red
on the base): `tests/curation/test_assist_loop_record_in_tables.py` (AST and `db`: no loop budget,
node or answer is read from `jobs.params`, `jobs.checkpoint` or `events`; gap G-3);
`tests/curation/test_assist_loop_budget_sql.py` (a randomized 200-seed crash test against the SQL
reservation never exceeds a limit); `tests/curation/test_assist_loop_kill_resume.py` (kill after each
step, then resume: no node asked twice, no crop charged twice beyond the forfeited in-flight ones);
`tests/curation/test_assist_loop_pause_cycles.py` (three pause cycles complete the run; gap G-1 end to
end); `tests/curation/test_assist_loop_two_projects.py` (gap G-2 end to end on one consumer);
`tests/curation/test_assist_loop_lock_rule.py` (carried from #286: one case per locked kind, never
written); `tests/curation/test_vlm_usage_recorded.py` (`usage_missing` counted, charged at the upper
bound). Doc-sync note for 6D: `auto_label_loop_plan.md` sections 0, 1, 3.1, 5.8, 8, 10 and 11 are
superseded for the rebuild by the research auto-label plan.

**4D done when:** `rg -n "trigger\.json|claim_trigger|_claim_trigger_for|running\.lock|QUEUE_MAX_AGE_S|pause\.flag|budget\.json|events\.jsonl|_ivf_retrain" src scripts` is empty (the ported check is named `maybe_retrain` and is 4F's to delete).

## 6. Lane 4E `b2/frontend-queued` (W2.3b, last: after every other W2.3a and W2.3b lane merged)

- **Scope:** Cropwright renders `queued` (waiting, with cancel) for every family whose status union
  gained it, follows the 202 + job id flows (upload, directory ingest is API-only, person grouping if
  the UI calls it), shows `cancelling` and `paused`, reads SSE ids as opaque strings, and drops calls
  to deleted routes (`classes/sync_to_opensearch`). `frontend/CHANGELOG.md`.
- **Research scope** (each control decided by the backend, the frontend renders it): the assist
  loop's Pause button and paused state with its reason, the budget edit dialog, the event log fed
  by SSE, every button driven by `AssistLoopControl`; the clusters pages on the new wire
  (`cluster_uid`, `display_no`, state counts, the serving partition, the activation conflict 409);
  the near-duplicate status (`stale_reasons`, the rescore button), the freeze's 409
  `near_dup_stale` and `moved_by_near_dup`, and `hide_near_duplicates` on the new fields.
- **Rules:** `$lib` aliases, 2-space indent, `npm` only; thin frontend (the backend says the state).
- **Tests:** a unit test per controller (red first); lane checks `npm run check`, lint, the touched
  tests, `npm run build`; the full frontend set and the browser check run in the B2 sweep.
- **Done when:** no status union in `frontend/src` lacks a value the contract has.

## 6a. Lane 4F `b2/clustering` (W2.3b): the clustering family on partitions (research clustering plan sections 5 and 8; approved, RD-1)

**Owns:** new `src/services/clustering/{__init__,partitions,run,carry_over,activation,projection,assign}.py`;
new `src/domain/group_lineage.py` (Jaccard matching with `scipy.optimize.linear_sum_assignment`,
also used by ND-A); `src/services/curation/clustering/**` (deletes `region_cluster_jobs.py` after
moving `region_fp_centroids`, `last_run.py`, `refresh_state_store.py`'s interim body, `cluster_ids.py`
and the `RESIDUAL_CLUSTER_ID_OFFSET` namespace); `src/config/cluster_profile.py`;
`src/config/job_kinds/clustering.py`, `src/config/task_routes/clustering.py`, `src/tasks/clustering.py`;
`src/clients/curation_opensearch/cluster_fields.py` (from 2F-8) and its mapping migration;
`src/domain/region_boxes.py` (the cluster keys of a box element); `src/services/curation/wire.py`
(the item and box serializer: the one W2.3b lane that edits it at once) and the review and search query
builders `semantic_search.py` and `review_request.py` (the cluster filters; ND-B changes the
near-duplicate filter after 4F merged); the cluster and region-cluster
routers and their wire models; from W2.3a's merged lanes: `src/services/curation/autolabel/ivf_check.py`
(deleted) and its one call line in `autolabel/job.py`, `src/services/curation/assist_loop/partition.py`
(the partition-backed reader), and the face grouping module's result function with `src/routers/persons.py`
(4F-9); `src/services/upgrade/from_060/steps/clustering.py`;
`src/config/retired_env.py` and `env.template` (append rules); tests under `tests/clustering/`,
`tests/test_no_cluster_id_namespace.py`.

| ID | Task | Red test (fails on the base) | Control |
|---|---|---|---|
| 4F-1 | `record_partition(session, *, project, target, level, space, method, params, members) -> partition_id` (COPY for memberships) and `group_lineage.match_groups(old, new, min_jaccard=0.5)`; signatures fixed here for ND-A and 4F-9 | `tests/clustering/test_record_partition.py` (`db`): a 10,000-member partition writes in one COPY and is `ready` | read back |
| 4F-2 | `cluster_run` (queue `cluster`): IVF for items and MiniBatchKMeans for region boxes through the partition seam; carry-over by Jaccard >= 0.5 and Hungarian matching; anchored identities pinned; the GPU backend refused without a passing `cluster_backend_validations` row | `tests/clustering/test_carry_over.py`: the same vectors in permuted order keep every `cluster_uid` and give ARI 1.000 (today's IVF gives 0.306) | an IVF run on a 1,000-item fixture equals today's assignment |
| 4F-3 | activation and projection: the OCC activation (409 `partition_conflict`), the `search_projection_tasks` row of kind `cluster_partition`, `cluster_projection_sync` fenced by `pending_partition_id`, the served flip at its end | `tests/clustering/test_activation_occ.py`, `tests/clustering/test_projection_fenced.py` (an older projection cannot overwrite a newer partition's fields) | a single activation projects once |
| 4F-4 | auto-activation guard (L1b; pending RD-24): auto only with 0 anchored or validated moves and id stability >= 0.95 | `tests/clustering/test_auto_activation_guard.py`: a run that moves an anchored member stays `ready` | stability 1.0 activates |
| 4F-5 | incremental bands (L1c): `assign.place_new(session, *, project, target, item_ids, vectors) -> list[Placement]` (signature fixed here) places a new item by the serving partition's prototypes with the calibrated bands; the one call in `src/services/curation/ingest_index.py` (its cluster placement) is made by ND-A, which owns that file in W2.3b, against this signature (4F merges first) | `tests/clustering/test_incremental_bands.py`: below the review band an item is `unassigned` | without a calibration, behaviour equals today's |
| 4F-6 | the cluster split of research finding "cluster_id is overloaded": assigner (IVF centroids), versioned partition (Postgres) and review state (names, anchors) are three things; items and boxes carry `cluster_partition_id`, `cluster_uid`, `cluster_state`, `cluster_score`, `cluster_margin`, `cluster_assigned_by`; integer `cluster_id`, `cluster_subid`, `excluded_prior_cluster_*`, the class-cluster overload (`cluster_id == class_id`) deleted with their readers; one nested `cluster` object on the item and box wire; the routes of clustering 5.5 S-core, with `regions/cluster`, `regions/cluster/status` and both `refine` routes deleted; `make contracts` | `tests/test_no_cluster_id_namespace.py`: no `RESIDUAL_CLUSTER_ID_OFFSET`, `cluster_subid` or integer `cluster_id` in `src/` (red today) | `tests/test_no_legacy_region_scalars.py` stays green |
| 4F-7 | `cluster_profile` read path and RD-29: the refresh and IVF-retrain env values move into the profile and are retired through `src/config/retired_env.py`; `project_state('cluster_refresh')` keeps only the beat's clock | retired-env test additions | `OP_CLUSTER_REFRESH_INTERVAL_S` still read as the beat interval |
| 4F-8 | upgrade step `clustering.py` ([02c](02c_migration_ops_tests.md) section 2.2) and RLS coverage of the clustering tables | `tests/upgrade/test_research_steps.py` (clustering case); `tests/db/test_row_level_security.py` covers the tables by introspection | the served view after the upgrade equals the 0.6.0 assignment |

| 4F-9 | person grouping as a partition (research clustering 5.2): the face grouping module's write and `persons_grouping_result` go through `record_partition` (`target = 'face'`, `method = 'union_find'`, project NULL, deployment scope, `deployment_tx`) | `tests/test_persons_cluster_partition.py` (`db`): one run writes one `ready` face partition and `GET /persons/cluster` reads it | 4C-7's job test stays green |
| 4F-10 | the auto-label consumers on partitions: `autolabel/ivf_check.py` and its call deleted (the retrain is a `cluster_run` the beat of 6A starts); `assist_loop/partition.py` gains the partition-backed `PartitionReader` and `ClusterFieldPartition` is deleted | `tests/curation/test_assist_loop_reads_partition.py`: the loop's roots carry `(partition_id, cluster_uid)` and no integer `cluster_id` is read | 4D's loop tests stay green |

**4F done when:** `rg -n "RESIDUAL_CLUSTER_ID_OFFSET|cluster_subid|region_cluster_jobs|last_clustering|IVF_RETRAIN|ivf_check|ClusterFieldPartition" src scripts` lists only `src/services/upgrade/`.

## 6b. Lane ES-B `b2/embedding-reindex` (W2.3b): the re-index job and the space routes (research embeddings plan 5.5, 5.7, 8.1; approved, RD-1)

**Owns:** new `src/config/job_kinds/b2_embedding.py`, `src/config/task_routes/b2_embedding.py`, new
`src/tasks/embedding_reindex.py`, new `src/services/embedding_spaces/reindex.py`, new
`src/routers/curation/embedding_spaces.py` and `_embedding_space_models.py`, `contracts/` (append
rule), tests under `tests/embedding_spaces/test_reindex*.py`.

| ID | Task | Red test | Control |
|---|---|---|---|
| ES-B1 | `embedding_reindex` (queue `jobs`, exclusive per `space_key`, auto-resume, pausable at batch boundaries, `slice_s` 600 with a 30 / 35 minute limit so it never holds a `jobs` slot for hours (review RR-20), `gpu_affinity = 'any_card'`): PIT and `search_after` over ids only, one decode per image through 4A's reprocess decode path, encode on Triton, OCC partial updates of the new field and its state, a fenced checkpoint per batch, the verify step (a sampled `exists` count and 200 re-encodes at cosine >= 0.999) before `ready` | `tests/embedding_spaces/test_reindex_resume.py`: kill mid-run and resume, no finished doc re-encoded (a fake Triton counts calls); a moved box is skipped as `deferred` | a 100-doc fake build completes |
| ES-B2 | routes of embeddings 5.7 (catalog list, spaces list and register, promote, retire, evaluate, run status); `Idempotency-Key` on register; the start refuses when the OpenSearch data path lacks twice the `bytes_estimate` (fail closed) | `tests/embedding_spaces/test_space_routes.py`: a second `build` for one space is 409; cross-project access refused | contracts regenerated in the commit |
| ES-B3 | the flip queues the cluster refresh and the near-duplicate re-score for that space (the dependent artefacts read the primary space only) | `tests/embedding_spaces/test_flip_queues_dependents.py` | a flip with no dependents queues nothing |

## 6c. Lane ND-A `b2/near-duplicates` (W2.3b): one near-duplicate service with two levels (research near-duplicates plan sections 5 and 8; approved, RD-1)

**Owns:** the paths of the near-duplicate plan section 7.2 (new `src/services/near_duplicates/**`,
`src/domain/near_dup.py`, `src/config/job_kinds/near_dup.py`, `src/config/task_routes/near_dup.py`,
`src/tasks/near_dup.py`, `src/routers/curation/near_duplicates.py`,
`src/schemas/curation/near_duplicates.py`); the revision `0003_near_duplicates` and its models file
`src/db/models/near_dup.py` (the only revision-writing lane of W2.3a and W2.3b, [02d](02d_research_seam_schema.md)
section 0); the source backfill (new `src/services/source_backfill/{__init__,job,content}.py`, the
pre-created `src/tasks/source_backfill.py`, `src/config/job_kinds/source_backfill.py`,
`src/config/task_routes/source_backfill.py`; review RR-10); from W2.3a's merged lanes:
`src/services/curation/item_scores/near_dup.py` (deleted) and its registry entry, and the
`enqueue_pending` call sites in the reprocess and re-embed paths (`reprocess_job.py`,
`reprocess_embed.py`), dataset import, combine (`src/services/projects/combine/execute.py`), curation
ingest (`src/services/curation/ingest.py` and `ingest_index.py`), the region stage and the item and
image deletes (the coordinator lists the files with `rg -n "def (index_items|ingest_image|delete_)" src/services`
at the cut); `src/clients/curation_opensearch/near_dup_fields.py` (from 2F-8) and its migration;
`src/services/detection/frame_dedup.py`, `export_images.py`, `export_single_class.py`,
`strategy_builders.py` (the dedup lines only, as the plan says; the coordinator confirms no other
W2.3b lane owns them at the cut); `src/services/upgrade/from_060/steps/near_duplicates.py`; `env.template` and
`src/config/retired_env.py` (append rules); `contracts/`; tests under `tests/near_duplicates/`.

Tasks ND-1..ND-10 exactly as the near-duplicate plan section 8 names them, each with its red test:
ND-1 schema and RLS (the X-1 changes of 02d section 3); ND-2 the `near_dup_profile` kind and its
seeding from code defaults equal to today's behaviour (0.98 cosine until P1, pending RD-47); ND-3
`src/services/near_duplicates/pending.py::enqueue_pending(session, project, level, member_ids,
reason)` (signature fixed here) and its calls in every writer of the Owns list, each written before
the OpenSearch write it describes (the W2.3a lanes have merged, so ND-A owns those files now; in
`ingest_index.py` it also makes 4F-5's `assign.place_new` call); ND-4 the score, group and rescore tasks (exact FAISS GPU kNN k = 64 for a full
build, OpenSearch kNN for an incremental one, connected components with the size guard and a
complete-linkage split, lineage through 4F's `group_lineage`); ND-5 the projection sync and the
deletion of every `dup_*` field; ND-6 the routes, wire and contracts; ND-7 the export's
`collapse_near_duplicates` (pending RD-45); ND-8 the `content_sha256` exact key in curation ingest
(pending RD-43), first confirming that the stored bytes equal the uploaded bytes (no re-encode at
ingest; review RR-10; red `tests/near_duplicates/test_stored_bytes_equal_upload.py`), else stop and
report; ND-9 the `near_dup` scorer deleted (was 4A-6): `item_scores/near_dup.py`, its registry entry
and the `/curation/projects/{project}/scores/compute` option, contracts in the same commit,
`OP_CROP_DUP_THRESHOLD` retired (`removed:` naming the `near_dup_profile` kind; the former
fail-closed fix, clustering plan N-8, retires with it; red `tests/curation/test_no_near_dup_scorer.py`,
control: the other scorers run); ND-10 the upgrade step; ND-11 the `source_backfill` job (pausable,
`slice_s` 600, one read per stored image) with its content step writing `content_sha256` for every
image the 0.6.0 upgrade carried over, started from the upgrade report (RD-53); the W4 capture lane
adds the capture step to the same pass (review RR-10; red
`tests/near_duplicates/test_backfill_then_reingest_refused.py`: an upgraded fixture image re-ingested
after the backfill is refused as an exact duplicate).
Red tests named by the plan, plus `tests/near_duplicates/test_rescore_clears_stale_groups.py` (#473
part a, moved from 4A-7) and `tests/near_duplicates/test_incremental_equals_full.py` (risk R4).

## 6d. Lane ND-B `b2/near-dup-holdout` (W2.3b, cut after 4F and ND-A merged): the holdout guard (replaces 2D-4)

**Owns:** `src/services/curation/wire.py`, `semantic_search.py` and `review_request.py` (ND-B5, ND-B6; 4F has merged); new `src/services/near_duplicates/holdout.py`; the freeze call sites in
`src/services/curation/holdout_freeze.py` and `holdout.py` (2D merged in W2.2); the training-export
and cohort gate call; tests `tests/near_duplicates/test_holdout_*.py`.

| ID | Task | Red test | Control |
|---|---|---|---|
| ND-B1 | #473 part c and MQ-5 (moved from 2D-4, finding C8): at freeze, `H = H0` plus every member of every current item-level `leak` group that holds an item of `H0`; `test_holdout = true` through the OCC writers; one `holdout_near_dup_moves` row per moved item; the freeze record's run ids and `moved_by_near_dup`; all inside the freeze's fail-closed path | `tests/near_duplicates/test_holdout_near_duplicates.py` (`db` + fake OpenSearch): a validated item whose near-duplicate crop is in the cohort's leak group ends on the holdout side and is counted (red today) | an item with no near-duplicate is unchanged |
| ND-B2 | absorption after the freeze (pending RD-42): every item-level group run, before it flips current, moves the non-held-out members of mixed leak groups into the holdout and emits `holdout.near_dup_absorbed` | `tests/near_duplicates/test_holdout_absorption.py` | a run with no mixed group moves nothing |
| ND-B3 | the freeze and the training export or cohort build refuse with 409 `near_dup_stale` while a level has no current run, is `rebuild_required`, or has pending rows (pending RD-41) | `tests/near_duplicates/test_freeze_refuses_stale.py` | a fresh project freezes |
| ND-B4 | the status route's `stale_reasons` (`pending`, `rebuild_required`, `no_run`), which 4E renders | `tests/near_duplicates/test_status_route.py` | - |
| ND-B5 | the item wire for near-duplicates in `src/services/curation/wire.py` (4F owned it while ND-A coded and has merged): the `dup_*` keys go, `near_dup_group_id`, `near_dup_is_representative` and `near_dup_group_size` are served; `make contracts` | `tests/near_duplicates/test_item_wire.py`: no `dup_` key on any serialized item | review and search answers otherwise unchanged |
| ND-B6 | `hide_near_duplicates` in `semantic_search.py` and `review_request.py` keeps documents whose `near_dup_is_representative` is true or missing (the stale-window rule of the plan's 5.3) | `tests/near_duplicates/test_hide_filter.py`: a non-representative member is hidden, an ungrouped item shown | the review list without the filter is unchanged |

## 7. Merge and gate

**W2.3a.** The coordinator merges 5D, 4D, 4A, 4B, 4C, 5A, 5B, 5C (`--no-ff`), running
`make contracts-check`, the merged lanes' tests and `-m db` after each, then the W2.3a part check
(03b and 03f exit criteria, the Docker-free suite in 4 shards). **W2.3b.** The lanes cut from that
head; the coordinator merges 4F, ND-A, ES-B, then ND-B (cut after 4F and ND-A merged), with the same
checks after each; then 4E; then the W2.3b part check. Facts for 6B (compose) and 6D (docs) are collected from the
PR notes: the worker env (`OP_API_LOOPBACK_URL`), the two curation services' commands and
healthchecks, the removed volumes and env vars.

## 8. Lane briefs

Use the common block of [03a](03a_b2_w1_queue_infra.md), with: "B2 stage-4 lane <4A..4F, ES-B, ND-A, ND-B>", "cut
from int/b2-control-plane after stage 3 merged" for the W2.3a lanes 4A-4D, "after the W2.3a part
check" for 4F, ND-A and ES-B, "after 4F and ND-A merged" for ND-B, the research plan of the
lane in the reading list for 4F, 4D, ES-B, ND-A and ND-B (its sections 5 and 8), the reading list: this file (sections 1 and your
lane), [02b](02b_jobs_locking_events.md) sections 1-3, and "Register tasks only in your lane's
modules; write compose and docs facts in your PR notes, do not edit compose files or shared docs."
4E adds the frontend rules above. Compose project: none. GPU: none.
