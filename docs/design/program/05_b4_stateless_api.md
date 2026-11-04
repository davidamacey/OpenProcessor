# B4: stateless API (the worker count becomes a tuning knob)

| Field | Value |
|---|---|
| Status | plan, not started |
| Release | part of the single v0.7.0 release; B5 follows B4 and R-final (`06_release_acceptance.md`) runs last, after B5 (D-37 approved, [00_index.md](00_index.md) section 6); the tag is the owner's word |
| Depends on | B2 merged (`03_b2_overview.md`) and B3 merged (`04_b3_ultralytics_one_version.md`); the CC-10 evidence (4A-0) before 4A-3 |
| Decisions applied | D-13 order (approved); owner rules: no asyncio fan-out in request handlers (bulk OCC instead; the ingest data plane is the documented `TaskGroup` exemption Q-J7), no in-process or file locks for cross-process state (the last `flock` goes here), the API ends stateless, the worker count is a tuning knob only |
| Open decisions | D-29, D-29a, D-32, D-32a, D-33 (D-30 replaced by D-48 and D-31 approved, both delivered by B2): status in [00_index.md](00_index.md) section 6; section 9 holds the detail |
| Source IDs | `task_queue_plan.md` section 4e, Phase 6, Phase 7, J7, J13, J14, E26, section 2e, 2f; audit CC-7, CC-10, BA-6 (deferred), BA-8 (documented in B1); Q-D14 (remaining part); former master plan rev 4 sections 3b (event-hub and `job_lock.py` rows), 3c D14, 7.4, 7.7 (BA-6), 12 (T2 B4, T5), 15 (M15, M16), 16 (B4 row), Appendix A (A.4 #23, A.5 #6, A.7 #10, M-4) |
| Branches | `int/b4-stateless` from main after B2 and B3 merged; tracks `b4/bulk-occ` (4A), `b4/worker-knob` (4B), `b4/project-create` (4D), each in `<repo>-wt-b4-<track>` |
| Size | M-L (4A M, 4B M, 4D S-M); the former events track 4C is done by B2 stage 3 |

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read only, with `rg`). B2 and
B3 are not implemented there; every B2 name used here is defined in `03_b2_overview.md`.
Consolidated 2026-11-03. Updated 2026-11-04 for the PostgreSQL control plane
([02_control_plane_database.md](02_control_plane_database.md)): B2 now delivers the job store, the
event feed and the last file lock, so replicas no longer wait on anything from B4.

---

## 1. Goal and done-when

**Goal (task plan section 4e).** A design is correct only if it gives the same answer with 1 API
worker and with 32. After B2 the jobs and periodic loops are out of the API. B4 removes the rest:
request fan-out, the process-sized pools, the shared JSONL event log with its `flock`, and the
hard-coded worker counts. Then `API_WORKERS` is a pure tuning knob sized to the host's cores.

**B4 is done when:**
- (verify only; B2 delivered it) `rg -n "fcntl" src` is empty, `job_lock.py` is gone and the
  file-lock ratchet asserts zero;
- no lifespan side effects (the B2 rule in `03_b2_overview.md` section 5.5 holds; the ratchet in
  4B-4 enforces it);
- no request handler fans out (`rg -n "asyncio\.gather" src/routers` is empty; services reached
  only from a handler have no `gather` except the J7 ingest data plane, written as `TaskGroup`);
- the 1-vs-N live check (section 7) passes: an empty diff for the sequential scenario and every invariant for the concurrent one (R-15);
- the installer default `API_WORKERS` (D-33) is set provisionally as `min(nproc, 32)`, and its
  benchmark against the 32-worker baseline retaken on the B3 engine is recorded in
  `docs/PERFORMANCE.md` in GPU window 2 (step 4B-5; it needs Triton, and only three GPU windows
  exist, [00_index.md](00_index.md) section 8.1);
- the B4 gate (section 7) is green and the PR merged with a merge commit. B5 follows; R-final
  runs last;
- the closing PR moves the section 8.1 keep-list to its targets and deletes this file and its rows
  in `00_index.md` and `docs/design/README.md` ([00_index.md](00_index.md) "Plan lifecycle").

## 2. Preconditions

| Check | How | Expect |
|---|---|---|
| B2 merged | `rg -n "op.periodic.reaper" src/tasks` | present |
| B3 merged | `rg -n "venv-y11" Dockerfile requirements*.txt` | empty (Stage B) or the Stage A state recorded in 04 |
| file locks after B2 | `rg -n "fcntl" src` | empty |
| host | `11_risks_cleanup_host.md` | peers told; no other heavy run; no GPU needed in B4 (the benchmark is in GPU window 2) |

## 3. Tracks and merge order

| Track | Branch / worktree | Scope | Merge order |
|---|---|---|---|
| 4A fan-out | `b4/bulk-occ`, `<repo>-wt-b4-bulk-occ` | section 4 | 1 |
| 4D #439 | `b4/project-create`, `<repo>-wt-b4-project-create` | section 6 | 2 |
| 4B worker knob | `b4/worker-knob`, `<repo>-wt-b4-worker-knob` | section 5 | last (its ratchet allowlist must see every other track's final code) |
| privacy and security items | `b4/privacy-hardening` | [00_master_plan.md](00_master_plan.md) W4; owns `SECURITY.md` in W4, so it also writes the location rules of section 6c (coordinates stored under the project's `location_policy`, default `store`; exports keep the embedded GPS, as today; owner 2026-11-04, RD-51 reversed, RD-54) and the identity-text section's operator text | after 4D |
| capture metadata and the location policy at ingest (research spatial L-C1; RD-5 approved) | `b4/capture-metadata` | section 6c | after the privacy lane (cut after 4A, 4D and the privacy lane merged: 4A rewrites the ingest data plane, 4D owns `docs/CURATION.md` until it merges, the privacy lane may edit `src/services/curation/wire.py` for #470) |
| identity-text privacy baseline (research plate 5.8) | `b4/identity-text-privacy` | section 6c | after the capture lane, before 4B |

Same branch model as B2: one agent per worktree, lanes run only targeted red/green tests and static
checks on touched files, the coordinator merges `--no-ff` (one merge commit per track) and runs
**one** full gate for B4 (section 7). Opus plans, Sonnet implements and reviews. Every brief
forbids `pgrep -f`, `pkill -f` and sleep/until loops, and names its `-p`, GPU, worktree and
deadline.

---

## 4. Track 4A: request fan-out (Q-Phase 6, J7)

Inventory re-run on the branch (`rg -n "asyncio\.gather|TaskGroup\(" src/routers src/services`),
classified:

| Site | Reached from | Disposition |
|---|---|---|
| `routers/curation/review_resolve.py` (`gather` + `Semaphore(16)`, one OCC write per crop) | `POST /curation/projects/{project}/review/new_class_proposals/resolve` | 4A-1: one `mget` + one bulk |
| `routers/curation/edit_undo.py` (`gather` + `Semaphore(32)`, one undo per crop; "Edit history has no bulk merger yet") | `POST /curation/projects/{project}/crops/region/undo_batch` | 4A-2: bulk merger + one bulk |
| `services/labeling/vlm_labeler_classify.py` (two `gather`s over chunks) | `POST /curation/projects/{project}/vlm/label_batch` | 4A-3 (after the 4A-0 evidence) |
| `services/labeling/vlm_labeler_verify.py` (`gather` over chunks) | `POST /curation/projects/{project}/vlm/verify_region_batch` (**missing from the task plan's list**, K-18) | 4A-3 |
| `services/curation/ingest_batch.py` (4 sites), `ingest_detect.py` (2), `ingest_index.py` `_cache_crops`, `visual_search/ingest_batch.py` (2) | `/ingest`, curation ingest, bulk ingest's loopback requests | 4A-4: J7 exemption as `TaskGroup` (CC-7) |
| `services/curation/semantic_search.py` (kNN + count in parallel) | curation semantic search | 4A-5: one `_msearch` |
| `routers/health.py` (two probes in parallel, plus B2's queue probe) | `/health` | 4A-5: sequential with the existing per-probe timeouts |
| `services/resource_links.py` `refresh_reachability` (`gather` over service probes, in a request-spawned refresh) | curation health, global status, settings | 4A-5: a beat task writes the result to a Valkey key; the API reads it (one probe per deployment, the same answer in every worker) |
| job and worker code (`reprocess_*`, `open_vocab_run`, `assist_loop/*`, `clustering/*`, `bulk_ingest/*`, `vlm_labeler_{combined,visibility,assist}`, `region_detector`) | Celery workers or the detection/VLM workers after B2 | not request path; CC-9 bounds landed in B2 where they mattered |

### 4A-0 CC-10 evidence (first; no code change)

- With a stub VLM endpoint (the live harness's `fake-vlm`, or an offline `httpx` mock transport),
  send a maximum-size `label_batch` and a maximum-size `verify_region_batch`; record the peak
  in-flight upstream requests and the wall time. Offline is enough; no GPU.
- Record the numbers in the 4A PR. They size the bound in 4A-3; they do not decide whether the
  fan-out leaves the handler (the owner rule does, D-29a).

### 4A-1 `review_resolve`: one `mget` + one bulk with per-item OCC

- **Doc:** OpenSearch `if_seq_no`/`if_primary_term` -> 409 per item [O1]; the bulk API reports
  item-level errors (audit OS-2/OS-3 citations).
- **Library in the repo:** `src/clients/occ_bulk.py` `occ_update_bulk` (human-write bulk OCC: one
  `mget` page plus one bulk, re-reading and retrying a 409; already called from
  `src/clients/occ_upsert.py`). Not `occ_skip_on_conflict_bulk` (worker semantics, skips a 409;
  that is what `label_batch` uses). Do not write a new loop.
- **Steps:** replace the `gather` and the semaphore with one `occ_update_bulk` call over the crop
  ids with the existing merger; per-item final conflicts become `conflicts` entries (the response
  model already has them); above `OP_BATCH_SYNC_MAX` (default 500) the route returns 202 with a
  Celery job (`op.jobs.review_resolve_batch`, the B2 recipe in `03b` section 1, exclusive group
  `review_resolve`, scope ''); contracts in the same commit.
- **Red test:** `tests/curation/test_review_resolve_bulk.py`: the handler issues one `mget` and
  one `bulk` for 100 crops (count calls on a fake; today 100 single writes); a per-item 409 that
  stays conflicting lands in `conflicts`; 501 crops give 202 and a job id. **Control:** a
  single-crop resolve is unchanged.

### 4A-2 `edit_undo` batch: a bulk merger

- **Steps:** add the bulk form of `edit_undo_merger` (the merger already computes the undo from the
  current doc; it needs no per-crop call) and call `occ_update_bulk`; keep the per-crop outcome
  vocabulary (`undone`, `nothing_to_undo`, `conflicts`, `not_found`); above `OP_BATCH_SYNC_MAX`
  -> 202 + `op.jobs.edit_undo_batch`.
- **Red test:** `tests/curation/test_undo_batch_bulk.py`: one `mget` + one `bulk` for 100 crops;
  outcomes match the per-crop version on a fixture with each outcome. **Control:** the single-crop
  undo route.

### 4A-3 VLM batch routes (after 4A-0)

- **D-29a recommendation:** both routes return 202 with a Celery job (`op.jobs.vlm_label_batch`,
  `op.jobs.vlm_verify_region_batch`), and the chunk fan-out runs in the worker as
  `asyncio.TaskGroup` + `asyncio.Semaphore(<bound from 4A-0>)` (Python docs, "Task Groups").
  Cropwright barely calls them (`rg -l "label_batch|verify_region_batch" frontend/src`), so the
  UI cost is small; contracts change; CHANGELOG "Breaking".
- **Red test:** `tests/curation/test_vlm_batch_routes_enqueue.py`: the handler enqueues and makes
  no upstream call; the task keeps at most the bound in flight for N=1000 crops (count a fake).
  **Control:** N=3 completes with the same verdicts as today.

### 4A-4 The ingest data plane as `TaskGroup` (J7 exemption, CC-7)

- **Doc:** Python asyncio "Task Groups" (a child failure cancels the siblings; errors arrive as an
  `ExceptionGroup`).
- **Steps:** rewrite the `gather`s in `ingest_batch.py`, `ingest_detect.py`, `ingest_index.py`
  `_cache_crops` and `visual_search/ingest_batch.py` as `async with asyncio.TaskGroup()`; in
  `ingest_batch._one`, resolve the representative's future on error so no follower is left
  pending (CC-7); callers that catch one exception type use `except*` or re-raise the single child
  at the boundary (check with `rg -n "ingest_one|IngestBatch" src`).
- **Document** the exemption in `docs/design/engineering_standards.md`: request-scoped, bounded by
  the batch, awaited before the response, `TaskGroup` only, and the only exemption.
- **4A-4b, #473 part b, rewritten as ND-C1..ND-C4 (research near-duplicates plan sections 5.2, 5.6
  and 8, lane ND-C; pending RD-44, RD-50; the core pending rows and runs are written under
  `core_tx`, the `core` RLS scope of [02a](02a_postgres_schema.md) section 1.4, review RR-3; the job
  kind and task go in the modules 1B pre-created, `src/config/job_kinds/near_dup_core.py`,
  `src/config/task_routes/near_dup_core.py` and `src/tasks/near_dup_core.py`, review RR-4):** while 4A-4 rewrites `visual_search/ingest_batch.py`,
  core ingest stops grouping in the request. ND-C1: it writes `near_dup_pending` rows (project NULL,
  level `core`) and the single-writer `near_dup_core_group` job (exclusive group `near_dup_core`)
  groups them, so concurrent batches can no longer race. ND-C2: the ingest response lists
  `near_duplicate_candidates` (id and cosine above the core cut); the group appears once the job has
  run; the `near_duplicate_threshold` form field on the three ingest routes goes. ND-C3: the literal
  thresholds, the stale comment about Immich's threshold and the swallowed exception in
  `src/services/duplicate_detection/` deleted; the core index fields renamed to `near_dup_*` with
  the `/query/duplicates` wire shapes kept. ND-C4: the core `content_sha256` key. 4A owns
  `src/services/visual_search/ingest_single.py`, `src/services/duplicate_detection/`, the threshold
  fields of `src/routers/ingest/*.py`, the reads in `src/routers/query_duplicates.py` and the core
  index mapping for these tasks. Red: `tests/visual_search/test_core_grouping_serialized.py`: two
  concurrent batches with a shared near-duplicate pair give one group, not two (red today). Control:
  `GET /query/duplicates` answers the same shape.
- **Red test (CC-7):** a stubbed `ingest_one` raising for the representative leaves no pending
  follower task after the request fails. **Control:** an all-success batch is unchanged (same
  docs, same timings within noise).

### 4A-5 The small parallel sites

- `semantic_search.py`: the kNN query and the unembedded count become one `_msearch` (one round
  trip; OpenSearch multi-search API), with the vector field and state field of the bound project's
  space (research embeddings 5.4, ES-C's `embedding_space()`). Red: one client call instead of two
  concurrent ones.
- `routers/health.py`: probes run sequentially with their existing timeouts (health is not
  latency-critical). Red: no `create_task` or `gather` in the module (AST test).
- `resource_links.py`: `op.periodic.resource_reachability` (beat, 30 s) probes the services and
  writes `op:resource_reachability` (Valkey, `SET ... EX 90`; a short-lived cache, the kind of value Valkey owns per [02](02_control_plane_database.md) section 3.1); the routes read it; a missing key
  reads "unknown", never "down". Delete the per-process `_cache.inflight` task. Red: two API
  processes (two app instances in a test) give the same answer from one fake Valkey. The beat entry
  goes into `src/tasks/beat_schedule.py` and `tests/periodic/test_beat_schedule.py` gains its name in
  the same commit (the B2 stage-6 test lists the stage-6 names).

### 4A-6 Evaluator

Done in B2 stage 5 (lane 5C, [03f](03f_b2_s5_gpu_queues.md)): the evaluator is a `gpu_eval` consumer
(D-31 approved 2026-11-04).

**4A done when:** `rg -n "asyncio\.gather" src/routers` is empty; the 4A red tests and controls
are green; contracts regenerated; `engineering_standards.md` names the one exemption.

---

## 5. Track 4B: the worker knob (Q-Phase 7, J14)

### 4B-1 One knob

- **Doc:** uvicorn settings: `--workers` "Defaults to the `$WEB_CONCURRENCY` environment variable
  if available, or 1" [U1]; FastAPI "Server Workers" and "FastAPI in Containers" (one process per
  container under an orchestrator; several in one container on a single Compose host) [F2][F3].
- **Steps:** remove `--workers=32` and the "64" and "benchmark" comments from the `api` command in
  `docker-compose.yml`; add `WEB_CONCURRENCY=${API_WORKERS:-8}` to the `api` environment; remove
  `"--workers", "16"` from the Dockerfile `CMD` (the image default becomes uvicorn's
  `WEB_CONCURRENCY`, else 1, which is what an orchestrator wants); keep
  `--limit-max-requests` and the jitter (D-26).
- **Installer:** `setup-openprocessor.sh` writes `API_WORKERS` once from `nproc` (default D-33);
  `upgrade` adds it when missing; INSTALLATION documents it with the orchestration note
  (`API_WORKERS=1` with several replicas works from B2 on: jobs, events and claims are in
  Postgres and Valkey; see the 1-vs-N test in section 7).
- **Red tests (`tests/compose/`):** the `api` command has no `--workers`; the `api` environment
  sets `WEB_CONCURRENCY` from `API_WORKERS`; the Dockerfile `CMD` has no `--workers`.
  **Control:** `tests/compose/test_worker_recycle_jitter.py`.

### 4B-2 Pools sized per process from totals (Q-D14 remaining part)

- **Files:** `src/app/lifespan.py` `_start_pools` (today `ThreadPoolExecutor(max_workers=64)` and
  `AsyncTritonPool(pool_size=4, max_concurrent=64)` per process), a new `src/config/api_pools.py`.
- **Steps:** totals (`EXECUTOR_THREADS_TOTAL`, `TRITON_CHANNELS_TOTAL`,
  `TRITON_INFLIGHT_TOTAL`) as constants sized for today's 32 workers (64 x 32 threads,
  4 x 32 channels); per process = `max(min_floor, total // API_WORKERS)`; the config reads
  `WEB_CONCURRENCY` (the same value uvicorn uses) and fails closed when it is not a positive int.
  Record in INSTALLATION that `--limit-concurrency=512` stays per process (the benchmark shows
  whether it needs sizing too).
- The database pool follows the same rule as a real deployment total (review fix R-6):
  `floor(OP_DB_API_CONNECTIONS_TOTAL / (API_WORKERS * OP_API_REPLICAS))`, at least 1, no overflow
  ([02b](02b_jobs_locking_events.md) section 8); the connection-budget test of B2 (1D-6, which takes
  the scale factors) is re-run with the chosen default and with two replicas. Re-check RC-4: the
  replica count has one source, `.env` `OP_API_REPLICAS` read by compose as `deploy.replicas`
  (B2 1C-2); the pool timeout answer 503 `database_busy` and the separate poll engine come from
  B2 1A-6; 4B adds no scale command.
- **Red test:** `tests/config/test_api_pools.py`: `API_WORKERS=1` gives the totals, 32 gives
  today's per-process values, 0 or junk fails. **Control:** the lifespan smoke test.

### 4B-3 Document the remaining per-process limits

- BA-8 (probe cap) was documented in B1; the remaining per-process caches (registry, config,
  settings, stats) keep their revision checks or staleness windows (B1-1D). 4B lists them in
  `docs/ARCHITECTURE.md` "Per-process state" with the reason each is correct per process.

### 4B-4 The statelessness ratchet

- **File:** `scripts/codegen/check_process_state.py` with a shrink-only baseline in
  `scripts/codegen/baselines/process_state.txt` (the same pattern as the existing ratchets;
  CONTRIBUTING.md). It fails on:
  - `asyncio.create_task` / `loop.create_task` in `src/routers`, `src/services`, `src/app` outside
    the allowlist (the B2 list in `03_b2_overview.md` section 5.4, each with its reason);
  - a new module-level `asyncio`/`threading` lock or semaphore, or a module-level mutable dict, set
    or list, outside the allowlist (each entry with a reason: "per-process cache with a revision
    check", "local throttle", ...).
- This is the statelessness ratchet the former master plan keeps in 4B. The asyncio-audit `spawn` helper
  stays cut (former master plan Appendix B); the ratchet uses an allowlist, not a helper.
- **Red test:** a fixture module with a new module-level dict fails; the real tree passes with the
  baseline. **Control:** an allowlisted entry with its reason passes.

### 4B-6 The Triton router (research GPU S-15, S-23)

`src/clients/triton_client_manager.py` takes its server list from the active serving profile (one
server by default) and picks per call the server with the fewest outstanding requests in this
process among those whose `serving_reports` say the model is ready; the image-prep call and every
follow-up for that image go to the same server; a gauge `op_triton_router_outstanding{server}`.
Red: `tests/clients/test_triton_router.py`: with two fake servers 100 concurrent calls split within
10 %; a server whose report lacks the model gets no call; one image's follow-up calls stay on its
server. Control: one server behaves as today.

### 4B-7 The in-process text encoders are deleted (research embeddings ES-E; D-69 consequence; pending RD-31)

Under D-69 every API process requires Triton, so the in-process `torch` and `onnx` PE text backends
are dead code. Delete them from `src/clients/pe_text_backends.py` (keep only Triton) and
`src/clients/pe_encoder.py`; retire `OP_PE_TEXT_BACKEND` and `OP_PE_TEXT_ONNX_PATH` as `removed:`
entries in `src/config/retired_env.py` and `env.template`; the 503 text at
`src/routers/curation/search.py:116` names Triton's text model. Red:
`tests/curation/test_search_requires_triton_text.py`: with the Triton text model not ready the
search refuses with an error naming it and no `torch` import happens anywhere in the API image path;
the retired variables fail at startup. Control: search with Triton ready is unchanged.

### 4B-5 Benchmarks (T5; run in GPU window 2, the B5 window)

The scripts and the procedure are written and reviewed in B4; the run needs Triton and a GPU, so
it happens as the first step of the B5 window ([00_index.md](00_index.md) section 8.1;
`07a_b5_baseline.md`), on the stack as merged after B4.

1. **Retake the 32-worker ingest and search baseline on the B3 engine** (the baseline in
   `docs/PERFORMANCE.md` was taken on the 0.6.0 engine; comparing against it would mix engines).
   Public data only (COCO / Open Images), the existing benchmark scripts, GPU 0, its own `-p`,
   host load recorded.
2. Run the installer default (D-33) against that retaken baseline; repeat only a disputed point.
   Record the database pool wait time (SQLAlchemy pool checkout time, p50 and p99) and the count of
   503 `database_busy` answers next to the throughput (re-check RC-4).
3. This validates the knob; it is not the B5 baseline (`07_*`).

**4B done when:** the compose and Dockerfile tests, the pool test and the ratchet are green; the
benchmark scripts exist and run on a dry configuration; the numbers and the confirmed default are
in `docs/PERFORMANCE.md` once GPU window 2 has run (5a retakes the same baseline, so the 4B-5
numbers are its first rows).

---

## 6. Track 4D: #439 project creation takes 13-20 s

- **4D-0 measure first:** wrap the stages of `POST /curation/projects` (`src/routers/curation/projects.py`
  and the create service it calls) with `src/utils/stage_timing.py` `stage_timer` (the #414
  pattern); run it 10 times inside the single B4 live run (`-p op-b4-live`, section 7; CPU is enough, no extra harness run) and post the stage
  table to #439 (the coordinator posts; the lane writes the text).
- **D-32 recommendation:** project creation becomes a `jobs` row of kind `project_create` through
  `start_job` (review fix R-21: the `op_type=create` record was an OpenSearch concept B2 deletes, and
  the single-entry ratchet refuses a plain task): deployment-level (`project_id` NULL, scope = slug),
  exclusive per slug, auto-resume (each step checks what already exists: the Postgres rows from 2A,
  then the two OpenSearch indexes, R-7), registered in `src/config/job_kinds/b4_project.py` (with `task_routes/b4_project.py` and
  `src/tasks/b4_project.py`, created empty by B2 lane 1B; 4A owns the `b4_fanout` modules, RC-5);
  it creates and stamps its two OpenSearch indexes (RC-1) and listed in the 02b kind table; the route returns 202 with the
  record in **`building`** (D-32a: reuse the existing project status, which combine already uses
  for a target being built, instead of a new `creating` value); Cropwright shows `building` as it
  does for combine. Alternative: keep it synchronous if one slow step explains the time and can be
  fixed in place. No request-path fan-out either way.
- **Red test:** `tests/curation/test_project_create_job.py`: the route returns 202 and creates no
  index in the handler; the task creates them; a duplicate slug is 409 before enqueue.
  **Control:** project list and get routes unchanged.
- **Docs and contracts:** the 202 + `building` in contracts, `docs/CURATION.md`, Cropwright
  (projects page), both CHANGELOGs ("Breaking": create returns 202). 4D also adds one sentence to
  the external-endpoint part of `docs/CURATION.md` (owner decisions OI-1 and OI-2 of 2026-11-04):
  when external use is enabled, external VLM endpoints and the hosted research judge may receive
  any region kind and contact-sheet image; the privacy lane writes the matching SECURITY.md text.

## 6c. Research lanes in W4 (approved by the owner 2026-11-04, RD-1 of [00d](00d_owner_questions.md))

**`b4/capture-metadata`** (research spatial plan sections 5.1 and 8, lane L-C1; RD-51 reversed and
RD-54 answered by the owner 2026-11-04: GPS coordinates are stored now under the project's
`location_policy`, default `store`, and exports keep the embedded GPS as today; RD-5 approved;
pending RD-52, RD-53). Why at ingest: the capture-date filter on the crops list and the capture time that
the plate burst consensus of W5e groups by (review 3.6; the upgrade backfill reads the stored files,
so waiting loses nothing for files still present). Owns: new `src/domain/capture_metadata.py` (`CaptureMetadata`,
`parse_capture(exif_ifd0, exif_ifd, gps_ifd)`, pure, no Pillow import); the header read in
`src/services/curation/image_decode.py` (`DecodedImage.capture`; 5b-D1 moves it in W5b); the capture
part of the ingest writer; new `src/clients/curation_opensearch/capture_fields.py` and its
mapping-migration registration (2F's additive registry); the `capture` wire block and the
`taken_from` / `taken_to` filter on the crops list; the capture and location-purge steps of the
`source_backfill` job that ND-A created in W2.3b (new `src/services/source_backfill/{capture,location_purge}.py`
and their entries in its step list; no new job kind, review RR-10); new `src/domain/location_policy.py`
(the one visibility helper: `apply_location_policy(meta, policy) -> CaptureMetadata`, pure);
the `location_policy` change hook in 2C's project-settings service (a switch to `drop` starts the
purge step); `src/services/curation/wire.py` (the `capture` block); `docs/CURATION.md`,
`docs/design/curation_api_contract.md`, `CHANGELOG.md`; tests `tests/curation/test_capture_*.py`,
`tests/domain/test_capture_metadata.py`. Precondition: the spatial prototype P-1 (CPU, the Pillow
parser against ExifTool) read before the cut.

| ID | Task | Red test | Control |
|---|---|---|---|
| CM-1 | the parse rules of spatial 5.1: IFD0 `DateTime` is never capture time; the offset only from `OffsetTimeOriginal` (or the digitized pair) or the GPS UTC delta rounded to 15 minutes within -12:00..+14:00; (0, 0) and out-of-range coordinates are no location; a malformed date gives `exif_status = malformed`, never a failed ingest; the broad-except ratchet does not grow | `tests/domain/test_capture_metadata.py`, one case per rule | a clean header parses |
| CM-2 | every ingest path carries `capture` (upload, directory and bulk, dataset import, combine copies `capture.*` verbatim); items get `capture_taken_local` and `capture_taken_utc`; the coordinates go to the images index only, through `apply_location_policy` (`store` as parsed, `store_coarse` rounded to 0.01 degree, about 1 km, `drop` none); **no coordinate is ever copied to items, labels, manifests, training files, a VLM call or a log** | `tests/curation/test_capture_paths.py`; the location canary appears only in the image `capture` block under `store` or `store_coarse` and never in an item response, manifest, training file, VLM request or log | - |
| CM-3 | the capture step of `source_backfill` for upgraded projects (header-only read through the storage `source` root, in the same pass as ND-A's content step; a missing file counted, not failed; pending RD-53) | `tests/curation/test_capture_backfill.py` | a re-run writes nothing |
| CM-4 | the location policy (owner 2026-11-04, RD-51 reversed): the writer and the image wire go through the one helper; a switch to `drop` starts the location-purge step (every `capture.location*` value of the project cleared, counts-only audit); the former export GPS strip is **dropped** (RD-54: exports keep the embedded GPS, as today, and `SECURITY.md` says so) | `tests/curation/test_location_policy.py`: under `drop` no coordinate reaches the images index; `store_coarse` rounds; a switch to `drop` purges every coordinate and a re-run writes nothing | `has_location` is unchanged by the policy |

**`b4/identity-text-privacy`** (research plate plan 5.8; a separate lane from `b4/privacy-hardening`,
which is scoped from private notes; the coordinator checks the two own disjoint files before the
cut). Owns: `src/core/logging.py` (the structlog processor `redact_region_text` in
`shared_processors`, so structlog and foreign stdlib records both pass through it), new
`src/config/identity_text.py` (`OP_IDENTITY_TEXT_MAX_MODE`, `OP_IDENTITY_TEXT_SECRETS_DIR`;
malformed refuses startup), the sink-inventory canary `tests/region_text/test_sink_inventory.py`,
`env.template` (append rule). Red: a log record carrying `text`, `text_raw`, `text_frame`,
`text_vlm`, `text_ocr` or a model reply under those keys is masked (red today); a malformed
`OP_IDENTITY_TEXT_MAX_MODE` refuses startup; the master file must be a regular non-symlink file of
exactly 32 bytes. The `SECURITY.md` text ("Privacy: identity text") is written by
`b4/privacy-hardening`, which owns that file in W4.

## 6b. Events (moved)

The event transport is decided (D-48 as approved 2026-11-04: a Postgres `events` table, a Valkey pub/sub wake-up, resumable SSE ids) and
built in B2 stage 3 ([03e](03e_b2_s3_job_runtime.md)); the JSONL log, its `flock` and `job_lock.py`
are deleted there. B4 has no events track.

## 7. Gate (once for B4)

- One offline sweep on the merged `int/b4-stateless` (tracks got T0 only): `.venv/bin/pre-commit run
  --all-files` and the pre-push hooks; the full offline suite (`-n 8`); `make contracts-check`;
  `check_docs_vs_code.py`; the frontend full set (4A conflicts display, 4D `building`) and a
  browser check of every page that streams events at 1440 and 430 px.
- **Live harness, `-p op-b4-live`, no GPU (the harness has no Triton), budget 2 h; the third and
  last live-harness run** ([00_index.md](00_index.md) section 8.1):
  - **the 1-vs-N diff (Q-E26), workers and replicas:** one scripted scenario run three times:
    `API_WORKERS=1` with one `api` replica and one `jobs-worker`; `API_WORKERS=32` with one replica;
    `API_WORKERS=4` with two `api` replicas and two `jobs-worker` replicas (no `container_name`
    since B2, so `--scale` works). In each run: start, cancel and race jobs; edit, rename and merge classes; save and
    activate configs; resolve and undo a 600-crop batch (sync and 202 paths); create a project;
    subscribe to SSE and reconnect; read resource links and health. Review fix R-15: two
    scenarios per configuration. (a) **Sequential**: the same steps one at a time; diff the
    normalized records (class ids by name, config revisions, terminal job states). **Pass: an empty
    diff.** (b) **Concurrent**: the races (20 starts of one singleton, concurrent class edits, two
    SSE clients) judged by invariants, because which racer wins legitimately differs between runs:
    exactly one winner per singleton start, no lost or duplicated event per cursor, `next_class_id`
    equals the number of distinct class names, no 5xx. **Pass: every invariant holds** in all three
    configurations.
  - SSE reconnect with `Last-Event-ID` across replicas (a client moved from one replica to the other
    misses no event).
- **Benchmarks (4B-5)** are not part of this gate: they run in GPU window 2 (the B5 window).
- Trivy and the image-vs-git audit if an image changed.
- Then one PR from `int/b4-stateless`, CI green, a merge commit. B5 follows.

## 8. Rollback, upgrade impact, docs

- **Rollback:** revert the PR merge before the release; each track reverts alone.
- **Upgrade impact (0.6.0 -> 0.7.0):** `API_WORKERS` is written into `.env` on upgrade (default
  D-33); SSE clients reconnect once; the JSONL event history is not migrated; `POST
  /curation/projects` returns 202 (Cropwright ships in the same release).
- **Docs, contracts, CHANGELOG:** `docs/ARCHITECTURE.md` (worker model,
  per-process state), `docs/PERFORMANCE.md` (retaken baseline, the knob), `INSTALLATION.md`
  (`API_WORKERS`, orchestration note), `docs/design/engineering_standards.md` (the J7 exemption),
  `docs/CURATION.md` (`conflicts`, 202 paths, project `building`), `contracts/` (4A `conflicts` and
  202s, 4D, 4A-3), `CHANGELOG.md` `[Unreleased]` and `frontend/CHANGELOG.md`.

### 8.1 What to keep after merge

| Fact or section | Target doc |
|---|---|
| stateless-API rules and the ratchet allowlist (section 4B) | `docs/ARCHITECTURE.md`, `docs/design/engineering_standards.md` |
| bulk OCC semantics (`conflicts`, 202 paths), project `building` status on create | `docs/design/curation_api_contract.md`, `docs/CURATION.md` |
| `API_WORKERS` default, orchestration note | `INSTALLATION.md` |
| the 1-vs-32 diff result, the spike numbers, the retaken baseline and the confirmed default | `docs/PERFORMANCE.md` |
| the Breaking entries (project create 202, 202 batch routes) | `CHANGELOG.md`, `frontend/CHANGELOG.md` |

## 9. Decision detail (status lives only in [00_index.md](00_index.md) section 6)

| ID | Decision | Recommendation | Consequence | Needed by |
|---|---|---|---|---|
| D-29 | Q-J7 request fan-out | handler fan-outs become bulk OCC (`occ_update_bulk`); the ingest data plane is the one documented `TaskGroup` exemption | `/ingest` keeps its latency; review and undo batches use two round trips instead of N | B4 kick-off |
| D-29a | the VLM batch routes (`label_batch`, `verify_region_batch`) (new) | 202 + Celery job; the bounded `TaskGroup` runs in the worker | two wire changes; small UI cost (few Cropwright callers). Alternative: widen the J7 exemption to these two routes (the owner must say so) | 4A-3 |
| D-30 | Q-J13 event transport | replaced by D-48 (Postgres `events` + a Valkey wake-up), built in B2 stage 3 | - | - |
| D-31 | Q-J6 evaluator | approved 2026-11-04 (owner decision 4); built in B2 stage 5 | - | - |
| D-32 | #439 approach | measure first (4D-0); then project creation as a Celery job (202); alternative: keep it synchronous if one slow step explains the time | | after 4D-0 |
| D-32a | the status during creation (new) | reuse `building` (already used for a combine target) instead of a new `creating` | no new status vocabulary | 4D |
| D-33 | Q-J14 API workers | one knob `API_WORKERS` (`WEB_CONCURRENCY`); installer default `min(nproc, 32)`; compose fallback 8 when unset; confirmed by the 4B-5 benchmark against the retaken baseline; replicas need nothing more after B2 (D-42) | a lower count reduces throughput unless the benchmark says otherwise (M15) | 4B-5 |

## 10. Risks

| # | Risk | Mitigation |
|---|---|---|
| R4-1 | a lower default worker count reduces throughput (M15) | the benchmark against the retaken baseline decides the default |
| R4-2 | SSE rewrite regressions (M16) | the spike gates it; ids give ordered resume per stream; browser check of every live page |
| R4-3 | Postgres becomes load-bearing for the UI feed (since B2) | `/ready` 503 naming `database` (D-69); SSE 503; the UI shows the outage |
| R4-4 | `TaskGroup` changes the exception type callers see (`ExceptionGroup`) | `except*` or re-raise at the boundary; `rg` every caller first |
| R4-5 | bulk OCC changes conflict semantics | outcome vocabulary kept; fixture comparing the per-crop and bulk outcomes |
| R4-6 | the ratchet allowlist hides a new process state | each entry needs a reason; the baseline only shrinks |
| R4-7 | the benchmarks run on a loaded host | record host load; repeat only the disputed point |

## 11. Config handed to `08_environment_and_setup.md`

| Name | Kind | Default | Read by | Purpose |
|---|---|---|---|---|
| `API_WORKERS` | env (installer-written) | installer `min(nproc, 32)`; compose fallback 8 | compose `WEB_CONCURRENCY` | D-33 |
| `WEB_CONCURRENCY` | env on `api` | `${API_WORKERS:-8}` | uvicorn [U1]; `src/config/api_pools.py` | the worker count |
| `OP_BATCH_SYNC_MAX` | env | `500` | `review_resolve`, `edit_undo` (and the VLM batch routes if they stay sync) | above it the route returns 202 + job |
| `op:resource_reachability` | Valkey key | `EX 90` | `resource_links.py` | 4A-5 |
| removed | compose `--workers=32`; Dockerfile `--workers 16`; the JSONL event log path and its rotation constant | - | - | 08 deletes them from `env.template` and docs |

## 12. Release checks handed to R-final (`06_release_acceptance.md`)

- The 1-vs-32 diff re-run on the release stack (or its result carried from the B4 gate if the SHA
  is the same; R-final decides).
- `installer_upgrade` from 0.6.0 writes `API_WORKERS`; a fresh install writes it from `nproc`.
- `rg -n "fcntl" src` is empty on the release SHA (delivered by B2; re-checked).
- SSE: kill and reconnect a browser page mid-job; no lost or duplicated event; screenshots at 1440
  and 430 px of every live page.
- Project creation returns 202 and reaches `active`; the 4D stage numbers are on #439.
- `docs/PERFORMANCE.md` carries the retaken 32-worker baseline and the default's numbers.
- The bake-off acceptance phase runs through `gpu_eval` (B2 stage 5).

## 13. Deferred (after B4; not batched)

- **BA-6** (71 operations without a response model; 62 under `/curation`): one router per PR after
  B4, so the routers B2 and B4 rewrote are typed once (audit BA-6: model today's exact shape, set
  `response_model=`, `make contracts` in the same commit; red: a per-router test that its operations
  have non-empty schemas). Tracked by issue N15 in the index file's issue list.

## 14. Corrections (B4-specific)

| # | Where | Correction |
|---|---|---|
| K-18 | task plan 2e, audit CC-10 | `verify_region_batch` and `resource_links` fan-outs were missing (also in `03_b2_overview.md` section 21) |
| K-19 | task plan Phase 6 | "VLM chunk fan-out: bound it with a semaphore" would keep the fan-out in the handler, which owner rule 6 forbids; D-29a moves it to a worker |
| K-20 | former master plan 7.4 4C (Streams consumers) | obsolete: events moved to B2 stage 3 on Postgres (D-48); the per-process tail idea survives as one Valkey pub/sub subscription per API process (D-65; RC-11) |
| K-21 | former master plan 7.4 4D "record in `creating`" | the project status vocabulary already has `building` for the same meaning (D-32a) |
| K-22 | former master plan 7.4 "Then R-0.9" | obsolete (owner decision 2): R-final (after B5, D-37) owns the release acceptance; section 12 lists the B4 checks |

## 15. Former master plan content absorbed by this file

| Former master plan item | Lands in |
|---|---|
| 3b event-hub and `job_lock.py` rows | B2 stage 3 (3B) and stage 6 (6A) |
| 3c D14 (B4 part) | 4B-2 |
| 7.4 dependencies, tracks 4A-4D, test cycle, benchmarks, done-when, size, rollback, upgrade, docs, risks | sections 1-10 |
| 7.4 R-0.9 | section 12 (handed to R-final) |
| 7.7 BA-6 | section 13 |
| 12 T2 (B4 diff), T5 (B4 knob) | sections 5, 7 |
| 15 M15, M16 | section 10 |
| 16 B4 row | section 8 |
| Appendix A: A.4 #23 (spike, now void), A.5 #6 (baseline retake), A.7 #10, M-4 (pools) | -, 4B-5, 4B-2 |
| N1 (epic, B4 children), N14 (CC-10 box) | 4A-0, sections 1-6; the issue list stays with the index file |

## 16. Sources

- [U1] uvicorn settings (`--workers` and `WEB_CONCURRENCY`, `--limit-max-requests-jitter`):
  https://github.com/encode/uvicorn/blob/master/docs/settings.md
- [F2] FastAPI Server Workers: https://fastapi.tiangolo.com/deployment/server-workers/ ;
  [F3] FastAPI in Containers: https://fastapi.tiangolo.com/deployment/docker/
- Python asyncio Task Groups: https://docs.python.org/3/library/asyncio-task.html#task-groups
- [O1] OpenSearch index API (`if_seq_no`/`if_primary_term`):
  https://docs.opensearch.org/latest/api-reference/document-apis/index-document/ ; multi-search:
  https://docs.opensearch.org/latest/api-reference/search-apis/multi-search/
- [SSE] HTML Living Standard, server-sent events (`id:` field, `Last-Event-ID`):
  https://html.spec.whatwg.org/multipage/server-sent-events.html
- Celery, PostgreSQL and SQLAlchemy docs: as cited in [02b](02b_jobs_locking_events.md).

## 17. Ready-to-paste lane briefs

Common to every B4 brief: read `docs/design/program/05_b4_stateless_api.md` (your track's section
and sections 1-3, 9), `docs/design/program/03_b2_overview.md` section 5, `02b_jobs_locking_events.md` sections 2 and 8, CLAUDE.md,
CONTRIBUTING.md, `docs/design/engineering_standards.md`. Red test on the unchanged base first
(record it), then the fix and the control; `make contracts` in the same commit as a wire change;
lane checks only (`pytest <files> -q --no-cov`, `pre-commit run --files <touched>`); no full suite,
no `--all-files`; never `--no-verify`, push, merge or another checkout. Do NOT use `pgrep -f`,
`pkill -f`, or sleep/until loops; wait with healthchecks or the Monitor tool. Never touch another
compose project. Deadline: set by the coordinator at kick-off.

```
Track 4A (b4/bulk-occ, <repo>-wt-b4-bulk-occ): do 4A-0 first (offline, no Docker, no GPU), then
4A-1, 4A-2, 4A-4, 4A-5; 4A-3 only after the owner answered D-29a. The 202 paths use B2's
`start_job` (kinds `review_resolve_batch`, `edit_undo_batch`, `vlm_label_batch`,
`vlm_verify_region_batch` in `src/config/job_kinds/b4_fanout.py`, with `task_routes/b4_fanout.py`
and `src/tasks/b4_fanout.py`, all created empty by B2 lane 1B; 4D uses the `b4_project` modules, so
the two tracks never share a file, RC-5).
Compose project: none (offline). GPU: none.
```

```
Track 4D (b4/project-create, <repo>-wt-b4-project-create): write the 4D-0 stage timer and its report script; it runs inside the single B4 live run (no separate harness run), with
`docker compose -p op-b4-live` (CPU only, GPU: none), then post the stage table text in your report;
implement the D-32 option the owner chose.
```

```
Track 4B (b4/worker-knob, <repo>-wt-b4-worker-knob): 4B-1..4B-4 offline (compose project: none,
GPU: none). 4B-5 benchmarks are run by the coordinator in GPU window 2 (the B5 window) with its own -p;
write the benchmark commands and the PERFORMANCE.md table skeleton only.
```

