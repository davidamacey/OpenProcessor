# B4: stateless API (the worker count becomes a tuning knob)

| Field | Value |
|---|---|
| Status | plan, not started |
| Release | part of the single v0.7.0 release; B5 follows B4 and R-final (`06_release_acceptance.md`) runs last, after B5 (D-37 approved, [00_index.md](00_index.md) section 6); the tag is the owner's word |
| Depends on | B2 merged (`03_b2_overview.md`) and B3 merged (`04_b3_ultralytics_one_version.md`); the CC-10 evidence (4A-0) before 4A-3; the 4C spike before the rest of 4C |
| Decisions applied | D-13 order (approved); owner rules: no asyncio fan-out in request handlers (bulk OCC instead; the ingest data plane is the documented `TaskGroup` exemption Q-J7), no in-process or file locks for cross-process state (the last `flock` goes here), the API ends stateless, the worker count is a tuning knob only |
| Open decisions | D-29, D-29a, D-30, D-31, D-32, D-32a, D-33: status in [00_index.md](00_index.md) section 6; section 9 holds the detail |
| Source IDs | `task_queue_plan.md` section 4e, Phase 6, Phase 7, J7, J13, J14, E26, section 2e, 2f; audit CC-7, CC-10, BA-6 (deferred), BA-8 (documented in B1); Q-D14 (remaining part); former master plan rev 4 sections 3b (event-hub and `job_lock.py` rows), 3c D14, 7.4, 7.7 (BA-6), 12 (T2 B4, T5), 15 (M15, M16), 16 (B4 row), Appendix A (A.4 #23, A.5 #6, A.7 #10, M-4) |
| Branches | `int/b4-stateless` from main after B2 and B3 merged; tracks `b4/bulk-occ` (4A), `b4/worker-knob` (4B), `b4/events` (4C), `b4/project-create` (4D), each in `<repo>-wt-b4-<track>` |
| Size | L (4A M, 4B M, 4C M, 4D S-M) |

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read only, with `rg`). B2 and
B3 are not implemented there; every B2 name used here is defined in `03_b2_overview.md`.
Consolidated 2026-11-03.

---

## 1. Goal and done-when

**Goal (task plan section 4e).** A design is correct only if it gives the same answer with 1 API
worker and with 32. After B2 the jobs and periodic loops are out of the API. B4 removes the rest:
request fan-out, the process-sized pools, the shared JSONL event log with its `flock`, and the
hard-coded worker counts. Then `API_WORKERS` is a pure tuning knob sized to the host's cores.

**B4 is done when:**
- `rg -n "fcntl" src` is empty and `src/services/curation/job_lock.py` is deleted; the file-lock
  ratchet baseline is zero;
- no lifespan side effects (the B2 rule in `03_b2_overview.md` section 5.5 holds; the ratchet in
  4B-4 enforces it);
- no request handler fans out (`rg -n "asyncio\.gather" src/routers` is empty; services reached
  only from a handler have no `gather` except the J7 ingest data plane, written as `TaskGroup`);
- the 1-vs-32 live diff (section 7) is empty;
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
| B2 merged | `rg -n "op.periodic.lease_reaper" src/tasks` | present |
| B3 merged | `rg -n "venv-y11" Dockerfile requirements*.txt` | empty (Stage B) or the Stage A state recorded in 04 |
| file-lock ratchet after B2 | `rg -n "exclusive_(file\|start)_lock\(\|fcntl\.flock" src` | only `event_hub.py` and `job_lock.py` |
| host | `11_risks_cleanup_host.md` | peers told; no other heavy run; no GPU needed in B4 (the benchmark is in GPU window 2) |

## 3. Tracks and merge order

| Track | Branch / worktree | Scope | Merge order |
|---|---|---|---|
| 4A fan-out | `b4/bulk-occ`, `<repo>-wt-b4-bulk-occ` | section 4 | 1 |
| 4D #439 | `b4/project-create`, `<repo>-wt-b4-project-create` | section 6 | 2 |
| 4B worker knob | `b4/worker-knob`, `<repo>-wt-b4-worker-knob` | section 5 | 3 (its ratchet allowlist must see 4A's and 4D's final code) |
| 4C events | `b4/events`, `<repo>-wt-b4-events` | section 6b | 4 (risky; reverts alone) |

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
| `services/resource_links.py` `refresh_reachability` (`gather` over service probes, in a request-spawned refresh) | curation health, global status, settings | 4A-5: a beat task writes the result to a Redis key; the API reads it (one probe per deployment, the same answer in every worker) |
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
  Celery job (`op.jobs.review_resolve_batch`, the B2 recipe in `03b` section 1, lease kind
  `review_resolve`, scope none); contracts in the same commit.
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
- **Red test (CC-7):** a stubbed `ingest_one` raising for the representative leaves no pending
  follower task after the request fails. **Control:** an all-success batch is unchanged (same
  docs, same timings within noise).

### 4A-5 The small parallel sites

- `semantic_search.py`: the kNN query and the unembedded count become one `_msearch` (one round
  trip; OpenSearch multi-search API). Red: one client call instead of two concurrent ones.
- `routers/health.py`: probes run sequentially with their existing timeouts (health is not
  latency-critical). Red: no `create_task` or `gather` in the module (AST test).
- `resource_links.py`: `op.periodic.resource_reachability` (beat, 30 s) probes the services and
  writes `op:resource_reachability` (Redis, `SET ... EX 90`); the routes read it; a missing key
  reads "unknown", never "down". Delete the per-process `_cache.inflight` task. Red: two API
  processes (two app instances in a test) give the same answer from one fake Redis.

### 4A-6 Evaluator (only if D-31 = move)

- In the evaluator image add the Celery pins; run `-Q gpu_eval -c 1`; the API enqueues
  `op.gpu_eval.bakeoff` instead of writing `bakeoff_jobs/*.job.json`; `gpu_arbiter.bakeoff_active`
  reads the job records. Removes the at-least-once re-run (task plan 2c).
- Red: `tests/curation/test_bakeoff_enqueue.py` (no job file written; one message); a redelivery is
  a lease no-op. Control: the bake-off status route.

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
  (`API_WORKERS=1` + replicas needs a multi-host job store first, D-21).
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

### 4B-5 Benchmarks (T5; run in GPU window 2, the B5 window)

The scripts and the procedure are written and reviewed in B4; the run needs Triton and a GPU, so
it happens as the first step of the B5 window ([00_index.md](00_index.md) section 8.1;
`07a_b5_baseline.md`), on the stack as merged after B4.

1. **Retake the 32-worker ingest and search baseline on the B3 engine** (the baseline in
   `docs/PERFORMANCE.md` was taken on the 0.6.0 engine; comparing against it would mix engines).
   Public data only (COCO / Open Images), the existing benchmark scripts, GPU 0, its own `-p`,
   host load recorded.
2. Run the installer default (D-33) against that retaken baseline; repeat only a disputed point.
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
- **D-32 recommendation:** project creation becomes a `project_create` Celery job (the B2 recipe,
  plain task idempotent by the project record's `op_type=create`): the route returns 202 with the
  record in **`building`** (D-32a: reuse the existing project status, which combine already uses
  for a target being built, instead of a new `creating` value); Cropwright shows `building` as it
  does for combine. Alternative: keep it synchronous if one slow step explains the time and can be
  fixed in place. No request-path fan-out either way.
- **Red test:** `tests/curation/test_project_create_job.py`: the route returns 202 and creates no
  index in the handler; the task creates them; a duplicate slug is 409 before enqueue.
  **Control:** project list and get routes unchanged.
- **Docs and contracts:** the 202 + `building` in contracts, `docs/CURATION.md`, Cropwright
  (projects page), both CHANGELOGs ("Breaking": create returns 202).

## 6b. Track 4C: events on Redis Streams (Q-J13, D-30)

### 4C-0 The spike (gate for the rest of 4C)

- **Docs:** Redis Streams (`XADD` with approximate trimming `MAXLEN ~`, `XREAD BLOCK`, `XRANGE`)
  [RS][RX][RR]; the SSE spec's `id:` field and the `Last-Event-ID` reconnect header [SSE].
- **Setup:** on the B2 stack's Redis, in its own project `-p op-b4-spike`, CPU only.
- **Load:** 10,000 events/s for 60 s from 4 producer processes; 50 SSE consumers; kill and
  reconnect consumers with `Last-Event-ID`.
- **Measure both consumer models:** (i) one `XREAD BLOCK` per SSE connection; (ii) one per-process
  tail (`XREAD BLOCK` over the subscribed streams) fanning out to that process's SSE clients, with
  `XRANGE` catch-up from `Last-Event-ID` on reconnect. Model (ii) matches today's per-process tail
  and holds one Redis connection per API process instead of one per browser tab.
- **Pass:** no lost or reordered event per stream; resume correct; p99 delivery < 250 ms; Redis
  memory bounded by `MAXLEN ~`; record connections used per model.
- **The owner decides D-30 on the result before the JSONL log is deleted.**

### 4C-1 After D-30 = Streams

- **Files:** `src/services/curation/event_hub.py` (the JSONL log, its append with `fcntl.flock`, the
  rotation to `.1`, `tail`), `src/routers/curation/events.py`, `pipeline_events.py`,
  `global_status.py` (the SSE routes), `frontend/src/lib/sse*.ts` (only if the client must read
  `id:`; the browser `EventSource` re-sends `Last-Event-ID` on its own [SSE]).
- **Steps:** publishers (API and Celery workers) `XADD op:events:<project>` (and a global stream)
  with `MAXLEN ~ <cap>`; the per-process tail (model ii unless the spike says otherwise) dispatches
  to local subscribers; every SSE message carries the stream id as `id:`; a reconnect with
  `Last-Event-ID` replays with `XRANGE` then joins the tail (this is FE-7's root cause; B1 fixed
  the keymap/config resync symptom); delete the JSONL log, its rotation and the last `flock`;
  delete `src/services/curation/job_lock.py` and its tests; the file-lock ratchet baseline becomes
  empty, then the ratchet asserts zero.
- **Failure mode:** Redis down -> `/health` reports `queue: down` (degraded); SSE routes answer 503
  and the UI shows the outage; a refetch fallback only if the owner asks for it.
- **Red tests:** `tests/curation/test_event_stream_resume.py` (fake Redis: publish 100, disconnect
  at 40, reconnect with `Last-Event-ID` of 40 -> 41..100 in order, no duplicates);
  `tests/curation/test_event_publish_from_worker.py` (a worker-side publish reaches an API
  subscriber); `tests/test_no_fcntl.py` (`rg`-equivalent AST scan: no `fcntl` import in `src/`).
  **Control:** the existing SSE route tests pass with ids added.
- **Upgrade:** SSE clients reconnect once; the old JSONL history is not migrated (it is transient).

**If D-30 = keep the JSONL log:** the log keeps an allowlisted `flock` (it violates the owner rule;
not recommended), `job_lock.py` stays with only that caller, and the B4 done-when "`rg fcntl src`
is empty" is waived by the owner in writing.

**4C done when:** the spike passed and D-30 answered; `rg -n "fcntl" src` is empty;
`job_lock.py` deleted; the resume test green; the browser check of every live page passes.

---

## 7. Gate (once for B4)

- One offline sweep on the merged `int/b4-stateless` (tracks got T0 only): `.venv/bin/pre-commit run
  --all-files` and the pre-push hooks; the full offline suite (`-n 8`); `make contracts-check`;
  `check_docs_vs_code.py`; the frontend full set (4A conflicts display, 4D `building`, 4C SSE) and a
  browser check of every page that streams events at 1440 and 430 px.
- **Live harness, `-p op-b4-live`, no GPU (the harness has no Triton), budget 2 h; the third and
  last live-harness run** ([00_index.md](00_index.md) section 8.1):
  - **the 1-vs-32 diff (Q-E26):** one scripted scenario run twice, with `API_WORKERS=1` and
    `API_WORKERS=32`: start, cancel and race jobs; edit, rename and merge classes; save and
    activate configs; resolve and undo a 600-crop batch (sync and 202 paths); create a project;
    subscribe to SSE and reconnect; read resource links and health. Diff the normalized outcomes
    (records, class ids by name, config revisions, event sequences per stream). **Pass: an empty
    diff.**
  - the 4C SSE resume check against real Redis.
- **Benchmarks (4B-5)** are not part of this gate: they run in GPU window 2 (the B5 window).
- Trivy and the image-vs-git audit if an image changed (the evaluator image under D-31).
- Then one PR from `int/b4-stateless`, CI green, a merge commit. B5 follows.

## 8. Rollback, upgrade impact, docs

- **Rollback:** revert the PR merge before the release. 4C reverts alone (its merge commit is last):
  the JSONL log comes back with its allowlisted `flock` and `job_lock.py`.
- **Upgrade impact (0.6.0 -> 0.7.0):** `API_WORKERS` is written into `.env` on upgrade (default
  D-33); SSE clients reconnect once; the JSONL event history is not migrated; `POST
  /curation/projects` returns 202 (Cropwright ships in the same release).
- **Docs, contracts, CHANGELOG:** `docs/ARCHITECTURE.md` (events on Streams, worker model,
  per-process state), `docs/PERFORMANCE.md` (retaken baseline, the knob), `INSTALLATION.md`
  (`API_WORKERS`, orchestration note), `docs/design/engineering_standards.md` (the J7 exemption),
  `docs/CURATION.md` (`conflicts`, 202 paths, project `building`), `contracts/` (4A `conflicts` and
  202s, 4D, 4A-3), `CHANGELOG.md` `[Unreleased]` and `frontend/CHANGELOG.md`.

### 8.1 What to keep after merge

| Fact or section | Target doc |
|---|---|
| events on Redis Streams: stream names, `MAXLEN ~`, per-process tail, `Last-Event-ID` resume; stateless-API rules and the ratchet allowlist (sections 4B, 6b) | `docs/ARCHITECTURE.md`, `docs/design/engineering_standards.md` |
| bulk OCC semantics (`conflicts`, 202 paths), project `building` status on create | `docs/design/curation_api_contract.md`, `docs/CURATION.md` |
| `API_WORKERS` default, orchestration note | `INSTALLATION.md` |
| the 1-vs-32 diff result, the spike numbers, the retaken baseline and the confirmed default | `docs/PERFORMANCE.md` |
| the Breaking entries (project create 202, 202 batch routes) | `CHANGELOG.md`, `frontend/CHANGELOG.md` |

## 9. Decision detail (status lives only in [00_index.md](00_index.md) section 6)

| ID | Decision | Recommendation | Consequence | Needed by |
|---|---|---|---|---|
| D-29 | Q-J7 request fan-out | handler fan-outs become bulk OCC (`occ_update_bulk`); the ingest data plane is the one documented `TaskGroup` exemption | `/ingest` keeps its latency; review and undo batches use two round trips instead of N | B4 kick-off |
| D-29a | the VLM batch routes (`label_batch`, `verify_region_batch`) (new) | 202 + Celery job; the bounded `TaskGroup` runs in the worker | two wire changes; small UI cost (few Cropwright callers). Alternative: widen the J7 exemption to these two routes (the owner must say so) | 4A-3 |
| D-30 | Q-J13 event transport | Redis Streams with a per-process tail and `XRANGE` resume, **decided after the 4C spike passes**; alternative: keep the JSONL log with an allowlisted `flock` (violates the rule; not recommended) | the last file lock goes; Redis becomes load-bearing for the UI | after 4C-0 |
| D-31 | Q-J6 evaluator | move the bake-off evaluator to `-Q gpu_eval -c 1`; alternative: keep the file protocol | removes its at-least-once re-run; the evaluator image gains Celery | B4 kick-off |
| D-32 | #439 approach | measure first (4D-0); then project creation as a Celery job (202); alternative: keep it synchronous if one slow step explains the time | | after 4D-0 |
| D-32a | the status during creation (new) | reuse `building` (already used for a combine target) instead of a new `creating` | no new status vocabulary | 4D |
| D-33 | Q-J14 API workers | one knob `API_WORKERS` (`WEB_CONCURRENCY`); installer default `min(nproc, 32)`; compose fallback 8 when unset; confirmed by the 4B-5 benchmark against the retaken baseline; `API_WORKERS=1` + replicas needs a multi-host job store first | a lower count reduces throughput unless the benchmark says otherwise (M15) | 4B-5 |

## 10. Risks

| # | Risk | Mitigation |
|---|---|---|
| R4-1 | a lower default worker count reduces throughput (M15) | the benchmark against the retaken baseline decides the default |
| R4-2 | SSE rewrite regressions (M16) | the spike gates it; ids give ordered resume per stream; browser check of every live page |
| R4-3 | Redis becomes load-bearing for the UI | `/health` degraded; the UI shows the outage; refetch fallback only on the owner's word |
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
| `op:events:<project>` and the global stream | Redis keys | `MAXLEN ~` cap (constant) | `event_hub.py` | D-30 |
| `op:resource_reachability` | Redis key | `EX 90` | `resource_links.py` | 4A-5 |
| removed | compose `--workers=32`; Dockerfile `--workers 16`; the JSONL event log path and its rotation constant | - | - | 08 deletes them from `env.template` and docs |

## 12. Release checks handed to R-final (`06_release_acceptance.md`)

- The 1-vs-32 diff re-run on the release stack (or its result carried from the B4 gate if the SHA
  is the same; R-final decides).
- `installer_upgrade` from 0.6.0 writes `API_WORKERS`; a fresh install writes it from `nproc`.
- `rg -n "fcntl" src` is empty on the release SHA; `job_lock.py` absent.
- SSE: kill and reconnect a browser page mid-job; no lost or duplicated event; screenshots at 1440
  and 430 px of every live page.
- Project creation returns 202 and reaches `active`; the 4D stage numbers are on #439.
- `docs/PERFORMANCE.md` carries the retaken 32-worker baseline and the default's numbers.
- If D-31 moved the evaluator: the bakeoff acceptance phase runs through `gpu_eval`; Trivy on the
  evaluator image.

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
| K-20 | former master plan 7.4 4C "consumers `XREAD BLOCK` per SSE connection" | one blocking Redis connection per browser tab across ~32 processes; the spike measures it against a per-process tail (today's pattern), which is recommended |
| K-21 | former master plan 7.4 4D "record in `creating`" | the project status vocabulary already has `building` for the same meaning (D-32a) |
| K-22 | former master plan 7.4 "Then R-0.9" | obsolete (owner decision 2): R-final (after B5, D-37) owns the release acceptance; section 12 lists the B4 checks |

## 15. Former master plan content absorbed by this file

| Former master plan item | Lands in |
|---|---|
| 3b event-hub and `job_lock.py` rows | 4C-1 |
| 3c D14 (B4 part) | 4B-2 |
| 7.4 dependencies, tracks 4A-4D, test cycle, benchmarks, done-when, size, rollback, upgrade, docs, risks | sections 1-10 |
| 7.4 R-0.9 | section 12 (handed to R-final) |
| 7.7 BA-6 | section 13 |
| 12 T2 (B4 diff), T5 (B4 knob) | sections 5, 7 |
| 15 M15, M16 | section 10 |
| 16 B4 row | section 8 |
| Appendix A: A.4 #23 (spike), A.5 #6 (baseline retake), A.7 #10, M-4 (pools) | 4C-0, 4B-5, 4B-2 |
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
- [RS] Redis Streams: https://redis.io/docs/latest/develop/data-types/streams/ ; [RX] `XADD`:
  https://redis.io/docs/latest/commands/xadd/ ; [RR] `XREAD`: https://redis.io/docs/latest/commands/xread/ ;
  `XRANGE`: https://redis.io/docs/latest/commands/xrange/
- [SSE] HTML Living Standard, server-sent events (`id:` field, `Last-Event-ID`):
  https://html.spec.whatwg.org/multipage/server-sent-events.html
- Celery docs: as in `03_b2_overview.md` section 25.

## 17. Ready-to-paste lane briefs

Common to every B4 brief: read `docs/design/program/05_b4_stateless_api.md` (your track's section
and sections 1-3, 9), `docs/design/program/03_b2_overview.md` sections 5 and 9, CLAUDE.md,
CONTRIBUTING.md, `docs/design/engineering_standards.md`. Red test on the unchanged base first
(record it), then the fix and the control; `make contracts` in the same commit as a wire change;
lane checks only (`pytest <files> -q --no-cov`, `pre-commit run --files <touched>`); no full suite,
no `--all-files`; never `--no-verify`, push, merge or another checkout. Do NOT use `pgrep -f`,
`pkill -f`, or sleep/until loops; wait with healthchecks or the Monitor tool. Never touch another
compose project. Deadline: set by the coordinator at kick-off.

```
Track 4A (b4/bulk-occ, <repo>-wt-b4-bulk-occ): do 4A-0 first (offline, no Docker, no GPU), then
4A-1, 4A-2, 4A-4, 4A-5; 4A-3 only after the owner answered D-29a; 4A-6 only if D-31 = move.
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

```
Track 4C (b4/events, <repo>-wt-b4-events): 4C-0 spike with `docker compose -p op-b4-spike`
(CPU only, GPU: none); stop and report the measurements; continue with 4C-1 only after the owner
answered D-30.
```
