# B2 wave 2: move the request-started job families (tracks 2A, 2B, 2C, 2G)

| Field | Value |
|---|---|
| Status | plan, not started |
| Depends on | W1 merged into `int/b2-celery` (`03a_b2_w1_queue_infra.md`): `src/config/queue.py`, `src/tasks/app.py`, `src/tasks/job_task.py` (`register_job`), `src/clients/task_queue.py`, the queued marker, the compose services |
| Decisions applied | D-8, D-17; recommendations of D-20a (24 h limits for long kinds), D-22, D-22a, D-22b, D-25 |
| Source IDs | `task_queue_plan.md` Phase 3 (five steps, the 11-row table), section 2a; audit CC-4, CC-5 (upload extract, directory ingest), CC-9, CC-11 (last site `_PASS_LOCK`); Q-D1, Q-D6, Q-E3, E11, E14, E20; former master plan 7.2 W2 rows |
| Branches / worktrees | `b2/families-a` in `<repo>-wt-b2-families-a`; `b2/families-b` in `<repo>-wt-b2-families-b`; `b2/families-c` in `<repo>-wt-b2-families-c`; `b2/frontend-queued` in `<repo>-wt-b2-frontend-queued` (starts when 2A-2C are merged and `make contracts-check` is green) |
| Compose project | none in the lanes (unit tests with mocks only); the live checks run in the B2 sweep |
| Size | 2A M, 2B M, 2C M, 2G S |

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03.
Consolidated 2026-11-03. Line numbers in the task plan
are from `5b95591f`; every site below is given as an `rg` pattern. The shared contract (states,
lease, marker, wrapper) is `03_b2_overview.md` section 9.

---

## 1. The recipe (every FileJob family, one family per commit)

B1 already replaced each family's start lock with the lease. W2 changes the launch lines only
(accepted churn, former master plan A.5 #3).

1. **Start:** in the family's start function, keep B1's `acquire` but pass
   `state='queued', ttl_s=MAX_QUEUE_WAIT_S[queue]`; write the record `status=queued`; replace
   `asyncio.create_task(run_x(...))` with
   `task_queue.enqueue('op.jobs.<kind>', project=slug, job_id=job_id)`; then `mark_queued`.
   Broker down -> record `failed` (`queue_unavailable`), release the lease, route 503
   `queue_unavailable` (Q-E3).
2. **Task:** register the handler with `register_job('<kind>', handler, resumable=<D-25>,
   soft_time_limit=..., time_limit=...)` in the track's module (`src/tasks/jobs_a.py`,
   `jobs_b.py`, `jobs_c.py`; one module per track avoids merge conflicts) and add the route in
   `src/config/queue.py` `TASK_ROUTES` (in the track's delimited block). The handler calls the
   **existing** `run_x(job, ctx)` coroutine; it builds its OpenSearch client and service from
   `ctx`, never from a request.
3. **Delete** the module's `_tasks` / `_active_tasks` dict, its `reconcile_orphaned_jobs`, and its
   entry in `startup_reconcile.JOB_MODULES` (2D deletes the file itself).
4. **Cancel:** `queued` -> `task_queue.revoke(job_id)`, write `cancelled`, release, clear the
   marker; `running` -> touch `cancel.flag` (unchanged).
5. **Status:** the status route reports `queued`. Where the wire model lacks `queued`, add it and
   run `make contracts` in the same commit (Q-E14). Find the model with
   `rg -n "Literal\[" src/routers/curation/_*_models.py src/schemas/curation | rg running`.

**Per-family tests (each red before the change):**
- unit: the start route enqueues exactly once with `task_id == job_id` (mock `task_queue`); the
  record is `queued`; the lease doc is `queued`;
- unit: a broker-down `enqueue` gives 503 `queue_unavailable`, the record `failed`, the lease
  released;
- unit: cancel while `queued` revokes and writes `cancelled`;
- unit: the registered task runs the existing coroutine once through `JobTask` (mocked lease);
- control: the family's existing status-route test still passes (with `queued` added where the
  model changed).
- live (in the sweep only): the per-family smoke in overview section 14 step 11.

**Rule for every track:** where a family's job code has an uncapped `gather` over a large list,
rewrite it as `asyncio.TaskGroup` with an `asyncio.Semaphore` bound when the family moves (CC-9;
Python docs, "Task Groups"). Only where the list is large and uncapped. Red: with N=1000 inputs at
most the bound is in flight (count a fake); control: N=3.

---

## 2. Track 2A (`b2/families-a`): families 1-6

| ID | Family | Files (re-find with `rg`) | Notes | Auto-resume | Time limit |
|---|---|---|---|---|---|
| 2A-1 | `reprocess` | `src/services/curation/reprocess_job.py` (`create_job`, `start_job`, `cancel_job`, `run_job`); its router (`rg -ln "reprocess_job" src/routers`) | the template family (already split create/run). **CC-9:** `reprocess_open_vocab.py` `gather` over docs -> TaskGroup + Semaphore | yes | 24 h (D-20a) |
| 2A-2 | `assist_run` | `src/services/curation/assist/runner.py` (`start_run`, `run_assist`), `assist/job.py` | `_segment_stage` waits on the region worker: fine in a worker. **CC-9:** `assist/sample_eval.py` `gather` over chunks -> TaskGroup + Semaphore | no | default |
| 2A-3 | `item_scores` | `src/services/curation/item_scores/job.py` (`start_job`, `cancel_job`, `run_scoring_job`) | add `queued` to the status model | no | default |
| 2A-4 | `probe` | `src/services/curation/probe_job.py` (`start_probe_job`, `cancel_probe_job`) | add `queued` | no | default |
| 2A-5 | `selection` | `src/services/curation/selection/job.py` (`start_job`, `cancel_job`, `run_selection_job`) | `k_center_greedy` now runs in the worker (no loop concern); add `queued` | no | default |
| 2A-6 | `embedding_viz` | `src/services/curation/embedding_viz.py` (`start_job`, `cancel_job`, `run_projection_job`, `fit_projection`) | UMAP/cuML in the worker child: no CUDA in the parent (Q-E10); `jobs-worker` has the `API_GPU_ID` reservation; add `queued` | no | default |

**2A doc sync (in the `assist_run` family commit):** update the `docs/design/selectable_assist_plan.md`
"Deviations" line that says a restart ends a run `interrupted` through `reconcile_orphaned_jobs` so
it names the reaper rule (`03_b2_overview.md` section 9.5) and the `assist_run` task.

**2A done when:** the six families run only through `op.jobs.*`; `rg -n "_active_tasks|_tasks\[" src/services/curation/{reprocess_job,item_scores/job,probe_job,selection/job,embedding_viz}.py src/services/curation/assist/runner.py` is empty; contracts regenerated; the per-family tests green.

## 3. Track 2B (`b2/families-b`): families 7-12

| ID | Family | Files | Notes | Auto-resume | Time limit |
|---|---|---|---|---|---|
| 2B-1 | `region_cluster`, `region_fp_centroids` | `src/services/curation/clustering/region_cluster_jobs.py` (`start_region_cluster_job`, `start_region_fp_centroid_job`, the status functions) | B1 made them `FileJob`s with leases (the 1800 s `_JOB_STALE_S` gone); add `queued` to `job.json`'s wire shape or update contracts and Cropwright in the same commit | no | default |
| 2B-2 | `promote` | `src/services/training/promote_job.py` (`claim`, `start`, `run_detached`), `src/routers/curation/train/promote.py` | delete `run_detached` and `_detached`; **D-22b:** the `wait=true` synchronous mode is deleted (only tests use it: `rg -n "wait=true" tests \| rg promote`); those tests move to the async path or call `execute_promote` directly; contracts (query parameter removed); CHANGELOG "Breaking" | no | default |
| 2B-3 | `dataset_import`, `dataset_import_undo`, `dataset_import_resume` | `src/services/curation/dataset_import/runner.py` (the `_tasks[store.import_id]` sites), `src/routers/curation/datasets.py` (the resume route's inline rescan with `heartbeat_ticker`) | the resume rescan moves into `op.jobs.dataset_import_resume`; the route only acquires, writes `queued`, enqueues | yes | 24 h |
| 2B-4 | `dataset_upload_extract` (new job, CC-5 upload) | `src/services/curation/dataset_import/upload.py` (`receive_archive`, `safe_extract`), `POST /curation/projects/{project}/datasets/uploads` | the route streams the archive to disk and enqueues the extraction; lease kind `dataset_upload`, scope `<upload_id>`; the response becomes 202 with a job id and a status route (contracts). No interim `to_thread` (CC-5 says so) | no | default |
| 2B-5 | `combine` | `src/services/projects/combine/service.py` (`start`, `cancel`, `resume`, `_TASKS`, `_spawn`), `combine/execute.py` | the job dir is global; the wrapper binds the **target** project; `execute.py` keeps binding sources in turn (Q-E20). The task plan's "fix OS-4 in the same commit" is void: OS-4 was refuted (former master plan Appendix B) | yes | 24 h |

**2B red tests beyond the recipe:**
- `tests/curation/test_upload_extract_job.py`: the upload route returns 202 and does no
  extraction in the request (assert `safe_extract` is not called in the handler); the task
  extracts. Red today (it extracts inline).
- `tests/curation/test_dataset_import_resume_job.py`: the resume route does no rescan inline.
- `tests/curation/test_promote_no_wait_mode.py`: `wait=true` is rejected (422 unknown parameter,
  or absent from OpenAPI). Red today.
- `tests/curation/test_combine_task_binds_target.py`: the task binds the target project.

**2B done when:** `rg -n "run_detached|_TASKS\[|_tasks\[store" src/services` is empty; the upload
and resume routes do no blocking extraction or rescan; contracts regenerated.

## 4. Track 2C (`b2/families-c`): families 13-17

| ID | Family | Files | Notes | Auto-resume | Time limit |
|---|---|---|---|---|---|
| 2C-1 | `bulk_ingest_run` | `src/services/curation/bulk_ingest/runner.py` (`claim_run`, `spawn`, `_create_task`, the `run_coroutine_threadsafe` plumbing), `bulk_ingest/source_runs.py` (`start_source_run`), `src/routers/curation/ingest_sources.py` (`start_run_now`: it builds `base_url` from the request's own address) | the task sends to `OP_API_LOOPBACK_URL` (`http://api:8000` in the worker env); delete `run_coroutine_threadsafe`, `_create_task`, the `loop=` argument, `_tasks`; `start_run_now` stops passing its own address | yes | 24 h |
| 2C-2 | `source_poll` (new, K-4) | `src/services/curation/bulk_ingest/dispatcher.py` (`_poll`, `_execute`, `_scans`, `_loops`, `_on_task_loop`), `bulk_ingest/source_scan.py` (`scan_source`) | `_execute` for a `poll` action enqueues `op.jobs.source_poll` under the `source_scan:<source_id>` lease instead of `create_task(_poll)`; delete `_scans`, `_loops`, `_on_task_loop`. The `dispatch.lock` flock stays until 2D moves the tick to beat | no (the next tick re-polls) | default |
| 2C-3 | `open_vocab_drain` (queue `open_vocab`) | `src/services/curation/open_vocab_ingest.py` (`drain`, `schedule_open_vocab_after_ingest`, `wait_for_scheduled`, `_PASS_LOCK`, `_tasks`); callers `src/routers/curation/ingest.py`, `src/services/curation/upload_ingest.py` | a plain task (overview 9.4); one message per ingest request with its image ids, debounced by `SET op:open_vocab:pending:<slug> NX EX 300` ([V5]; a lost marker only costs one extra drain, the sweeper is the backstop). Delete `_PASS_LOCK`, `_tasks`, `wait_for_scheduled` and the `ALLOWED` entry in `tests/test_no_module_level_async_locks.py` (the last CC-11 site; `ALLOWED` becomes empty) | n/a | default |
| 2C-4 | `project_delete_finish` | `src/routers/curation/projects.py` (`_BACKGROUND_DELETE_TASKS`, the `_finish` task in the DELETE route), `src/services/projects/delete.py` (`_FINISH_IN_PROGRESS`, the finish function) | a plain task; idempotent through the project record OCC; delete both per-process collections. Celery does not de-duplicate by `task_id` (K-7) | n/a (re-DELETE retries) | default |
| 2C-5 | `directory_ingest` | `src/routers/ingest/directory.py` (`ingest_directory`: inline `rglob` + `read_bytes`, CC-5) | D-22a: a plain task with a deployment-level record under `/jobs/ingest_directory/<job_id>`; `POST /ingest/directory` returns 202 + job id; a `GET` status route; contracts and README. **First prove idempotency:** `rg -n "image_id|def .*hash" src/services/visual_search/ingest_batch.py` shows how ids are derived; a re-run of the same file must not duplicate docs, else stop and escalate | no | 24 h |

**2C red tests beyond the recipe:**
- `tests/curation/test_dispatcher_enqueues_poll.py`: a poll action enqueues `op.jobs.source_poll`
  and creates no asyncio task. Red today.
- `tests/curation/test_open_vocab_single_consumer.py`: two ingest requests produce at most one
  pending drain message per project while one is pending; the module has no `asyncio.Lock`.
- `tests/test_no_module_level_async_locks.py`: `ALLOWED` empty (the existing test, now strict).
- `tests/curation/test_project_delete_finish_idempotent.py`: two finish tasks for one project end
  in one `deleted` record and no error.
- `tests/test_directory_ingest_job.py`: the route returns 202 and does no file reads in the
  handler; a repeat run indexes no duplicates.

**2C doc sync (in the `open_vocab_drain` commit):** update `docs/design/sam3_full_image_detection_plan.md`
section 16 Wave 5 (API background task, serialized per process, sweeper in every worker) to the
`open_vocab` queue (`-c 1`) and the beat sweep task.

**2C done when:** `rg -n "_PASS_LOCK|_BACKGROUND_DELETE_TASKS|_FINISH_IN_PROGRESS|run_coroutine_threadsafe|_scans\b|_loops\b" src` is empty; contracts regenerated.

## 5. Track 2G (`b2/frontend-queued`): the frontend

- **Starts after** 2A, 2B and 2C are merged and `make contracts-check` is green (the TypeScript
  contracts are generated).
- **Scope:** Cropwright renders the new `queued` status (as waiting, with a cancel button that
  calls the existing cancel route) for item scores, probe, selection, embedding viz, region
  clustering, region FP centroids, upload extraction; the upload flow follows the 202 + job id;
  `frontend/CHANGELOG.md`. Find the status renderers with
  `rg -l "'interrupted'" frontend/src` and the job controllers under `frontend/src/lib/`.
- **Rules:** `$lib` aliases, 2-space indent, `npm` only (owner frontend rules); thin frontend: the
  backend says the state, the frontend renders it.
- **Tests:** a unit test per controller that `queued` renders as waiting and cancel is offered
  (red before); `npm run check`, lint, `npm run test`, `npm run build` on the touched files' package
  in the lane; the full frontend set and the browser check (1440 and 430 px) run in the B2 sweep.
- **Done when:** the unit tests are green and no status union in `frontend/src` lacks `queued`
  where the contract has it.

---

## 6. Decision used in this wave

D-22b (promote `wait=true` deleted, no shim; the work moves to the worker, only tests use it,
Cropwright and the acceptance phase use the async path; one query parameter removed, tests
rewritten) is registered in [00_index.md](00_index.md) section 6; the detail is in
`03_b2_overview.md` section 12. Needed by the start of 2B.

## 7. Merge order and gate

- The coordinator merges 2A, then 2B, then 2C (`--no-ff`, one merge commit each), running
  `make contracts-check` and the targeted tests after each; then 2G (T0 tier,
  [00_index.md](00_index.md) section 8.1).
- Lanes: only their red/green tests and `pre-commit run --files <touched>`. The full suite, the
  frontend full set and the single live run (B1 and B2 together) run once after stage W3
  (`03_b2_overview.md` section 14).

## 8. Docs, contracts, CHANGELOG (W2)

- `contracts/` regenerated in each wire commit (`queued`, upload 202, directory ingest 202, the
  promote parameter).
- `docs/CURATION.md`: job states per family, auto-resume list, the upload flow.
- `README.md`: `/ingest/directory` returns 202 + job id.
- `docs/design/curation_api_contract.md`: the changed routes.
- `CHANGELOG.md` `[Unreleased]` "Breaking": `queued` status, upload 202, directory ingest 202,
  promote `wait` removed. `frontend/CHANGELOG.md` (2G).
- Doc syncs of the older plans: the 2A and 2C notes above (`selectable_assist_plan.md`,
  `sam3_full_image_detection_plan.md`).
- What to keep after merge: the per-family table and the recipe go to `docs/ARCHITECTURE.md`; the
  changed routes to `docs/design/curation_api_contract.md`; this file is deleted with the other B2
  plan files in the closing PR (`03_b2_overview.md` section 17.1, [00_index.md](00_index.md)
  "Plan lifecycle").

## 9. Risks (W2)

| Risk | Mitigation |
|---|---|
| a family keeps a hidden in-process task | the done-when `rg` lines per track; overview section 5.4 allowlist |
| contract drift between 2A-2C | `make contracts` per commit; contracts-check after each merge; 2G last |
| directory ingest re-run duplicates documents | 2C-5 proves idempotency first, else escalates |
| a handler built clients from the request | the recipe step 2; unit test runs the handler with only `ctx` |
| bulk ingest loopback URL wrong inside the worker | `OP_API_LOOPBACK_URL` in the compose env (W1-8); live smoke |

## 10. Ready-to-paste lane briefs

### Track 2A

```
You are the B2 wave-2 lane agent for track 2A (Sonnet). Worktree <repo>-wt-b2-families-a,
branch b2/families-a, cut from int/b2-celery after W1 merged. Read: docs/design/program/03_b2_overview.md
(sections 5, 9, 12), docs/design/program/03b_b2_w2_job_families.md (sections 1, 2, 6-9),
CLAUDE.md, docs/design/engineering_standards.md. Move families 2A-1..2A-6 with the recipe,
one family per commit, plus the CC-9 rewrites in reprocess_open_vocab.py and assist/sample_eval.py.
Rules: red test on the unchanged base first (record the failure), then the fix and the control;
`make contracts` in the same commit as any wire change; lane checks only (`pytest <files> -q
--no-cov`, `pre-commit run --files <touched>`); no full suite, no --all-files; never --no-verify,
push, merge or another checkout. No Docker, no GPU in this lane (compose project: none).
Do NOT use `pgrep -f`, `pkill -f`, or sleep/until loops. Register tasks only in src/tasks/jobs_a.py
and the 2A block of TASK_ROUTES. Stop and report if B1's lease API differs from overview section 2.
Deadline: <set by the coordinator>. Report: commits, red outputs, contract changes.
```

### Track 2B

```
You are the B2 wave-2 lane agent for track 2B (Sonnet). Worktree <repo>-wt-b2-families-b,
branch b2/families-b, cut from int/b2-celery after W1 merged. Read the same files as 2A, using
section 3 of 03b. Move 2B-1..2B-5 (one family per commit); apply D-22b (delete promote wait=true)
only if the owner approved it, else stop and ask. Same rules as 2A (red first, contracts in the
wire commit, lane checks only, no --no-verify/push/merge/checkout, no Docker, no GPU, compose
project: none, no `pgrep -f`/`pkill -f`/sleep loops). Register tasks only in src/tasks/jobs_b.py
and the 2B block of TASK_ROUTES. Deadline: <set by the coordinator>.
```

### Track 2C

```
You are the B2 wave-2 lane agent for track 2C (Sonnet). Worktree <repo>-wt-b2-families-c,
branch b2/families-c, cut from int/b2-celery after W1 merged. Read the same files as 2A, using
section 4 of 03b. Move 2C-1..2C-5 (one family per commit). For 2C-5 prove idempotency of the
visual-search ingest first; if a re-run duplicates documents, stop and report. Delete _PASS_LOCK
and empty ALLOWED in tests/test_no_module_level_async_locks.py. Same rules as 2A (red first,
contracts in the wire commit, lane checks only, no --no-verify/push/merge/checkout, no Docker,
no GPU, compose project: none, no `pgrep -f`/`pkill -f`/sleep loops). Register tasks only in
src/tasks/jobs_c.py and the 2C block of TASK_ROUTES. Deadline: <set by the coordinator>.
```

### Track 2G

```
You are the B2 Cropwright lane agent (Sonnet). Worktree <repo>-wt-b2-frontend-queued, branch
b2/frontend-queued, cut from int/b2-celery after 2A, 2B and 2C merged and contracts-check is green.
Read docs/design/program/03b_b2_w2_job_families.md section 5, frontend/CLAUDE.md if present,
CLAUDE.md. Render `queued` (waiting + cancel) wherever the regenerated contracts carry it, and the
upload 202 flow; update frontend/CHANGELOG.md. Red unit test first per controller. Lane checks:
npm run check, lint, the touched tests, npm run build. No browser run in the lane (the sweep does
it). No Docker, no GPU. Do NOT use `pgrep -f`, `pkill -f`, or sleep/until loops. Never push,
merge or check out another branch. Deadline: <set by the coordinator>.
```
