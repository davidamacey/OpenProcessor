# Task queue plan: background and concurrent work across processes

> **Execution order and overrides:** see `docs/design/program/00_index.md` (the program set replaced the former master plan, rev 4, in git history), which wins where the two disagree. It REPLACES this plan's Phase 1 file locks (P1.0, P1.3, and `exclusive_start_lock` in P1.1/P1.2/P1.6) with OpenSearch leases and the class-registry document (`program/02a_b1_job_leases.md` and `program/02b_b1_class_registry.md`). It schedules Phases 2-5 in B2 and the API worker model in B4.

Status: plan only; owner decisions are in the master plan's decision register. Committed 2026-11-03.
Code read from main at `5b95591f` (pinned tree `.claude/worktrees/audit-r2`).
All `file:line` references are to that tree.
Inputs folded in: `artifacts_local/audit/asyncio.md` (R1-R10),
`artifacts_local/audit/opensearch.md` (OS-1..OS-10) and `artifacts_local/audit/api.md`
(API-1, 2, 4-8, 12-14).

---

## 0. TL;DR

- **Answer to "do we need Celery?": yes, for one specific class of work, and no for the rest.**
  - The defects come from **heavy jobs that run as `asyncio` tasks inside the 32 uvicorn
    request workers** (13 job families), and from **periodic loops that run once per worker**.
  - Uvicorn recycles each worker every 10,000 requests (`docker-compose.yml:264`), with
    30 s of graceful shutdown (`:276`). Any job running in that worker dies.
  - The fix is to run these jobs in a dedicated worker process. They need a real queue with
    routing, retries, revoke, and one periodic scheduler. Celery provides all of these, and
    FastAPI's own docs point here: "heavy background computation ... you might benefit
    from using other bigger tools like Celery" [F1].
- **Recommended architecture (hybrid):**
  1. **Celery 5.6 with a Redis 8 broker**, the same stack the owner already runs in
     OpenTranscribe (section 4d). It runs one small container with no published port and a
     required password. It handles every job that is enqueued from a request today and runs
     inside the API.
     Also for the trigger-file workers (auto-label, assist loop), and later the bake-off.
  2. **Celery beat**, as exactly one instance, for every periodic job. Today these are 32×
     lifespan loops plus the cluster-refresh daemon.
  3. **Our job record stays the source of truth** for status, progress, cancel and the
     exclusive claim (`FileJob` + a lease). The broker is transport only.
     - Celery delivery is at-least-once, so exactly-once *effect* still needs a domain
       claim [C2][C4].
     - Because the broker holds nothing the UI reads, it needs no backup.
  4. **Do not migrate:**
     - the detection and VLM workers. The item state in OpenSearch is their queue, and their
       writes are OCC.
     - the trainer. It runs hour-long GPU work through a file protocol with heartbeats and
       the GPU arbiter.
     - the model-export worker. Its `rename` claim is already correct.
     - per-process caches and pollers.
- **Fix first, no new service (Phase 1).** These live cross-process races exist today and
  need fixing whatever queue is chosen:
  - job-record writes that collide on one temp file (API-4, reproduced);
  - unlocked singleton starts (auto-label, selection, embedding-viz, region clustering);
  - class-registry lost updates (API-2);
  - startup reconcile that can interrupt a sibling's job (API-5);
  - config first-create (OS-1).
- **The recycle loss is proven.** The API audit reproduced a request-started task dying at
  `--limit-max-requests` (API-6).
- **Make the API stateless (section 4e).** Then the uvicorn worker count (32 in compose, 16
  in the Dockerfile, "64" in a comment) becomes a pure tuning knob, `API_WORKERS`, sized to
  CPU cores, that never affects correctness.
- **What the owner must decide** (section 6):
  - J1: library. The owner has confirmed Celery, aligned with OpenTranscribe's conventions.
  - J2: broker. Redis 8, same as OpenTranscribe. Valkey is the fallback.
  - J3: whether the broker, the jobs workers and beat join the default install. Recommended
    yes.
  - J4: delivery semantics.
  - J5: job-state store.
  - Also: what not to migrate, the fan-out rule, the migration order, and the resume policy.

---

## 1. Constraints from the project (read before deciding)

- **Deployment:**
  - Single-host Docker Compose and a one-line installer (`setup-openprocessor.sh`).
  - No git clone and no host Python (VISION "A one-line installer").
  - `docker-compose.yml` is deploy-safe (no `build:`).
- **Datastore:** OpenSearch only. There is no Redis, Postgres or RabbitMQ today.
  `requirements*.txt` has no `celery`, `redis`, `dramatiq` or `rq`.
- **API:** `uvicorn --workers=32 --limit-max-requests=10000 --timeout-graceful-shutdown=30`
  (`docker-compose.yml:253-282`). There is no `--limit-max-requests-jitter`.
  - Uvicorn: "Maximum number of requests to service before terminating the process" [U1].
- **Shared state today:**
  - named volumes `openprocessor-jobs:/jobs` and `openprocessor-state:/var/lib/openprocessor`,
    on one host;
  - `fcntl.flock` works across containers that share a local volume.
- **Workers:**
  - every worker service has a fixed `container_name`, so `docker compose --scale` cannot
    run two replicas;
  - single-replica assumptions in worker *containers* are therefore latent, not live;
  - the 32 API processes are live concurrency.
- **Standards:**
  - fail closed;
  - no shims or compatibility layers;
  - delete the superseded code in the same change;
  - class identity by name;
  - contracts regenerated (`make contracts`) when a wire model changes;
  - layering, file-size and unused-code ratchets in `scripts/codegen/`.
- **Network posture:** loopback by default. `OP_BIND_ADDRESS` publishes beyond it. OpenSearch
  runs with its security plugin disabled (`docker-compose.yml:1238`), so internal services
  rely on network isolation. A broker must follow the same pattern: internal network only,
  no published port, authenticated anyway.

---

## 2. Inventory of every background and long-running mechanism

Legend:
- **Claim** is how one executor gets exclusive ownership.
- **X-proc?** says whether that claim holds across the 32 API processes or across replicas.
- **Recovery** is what happens when the executor dies mid-job.
- "FileJob" means `src/services/curation/file_job.py`. It provides:
  - `state.json` (atomic rename),
  - `heartbeat` (mtime; stale after 30 s, ticks every 10 s from an *asyncio* task),
  - `cancel.flag`,
  - `state.lock` (flock around read-modify-write).
- "start lock" means `exclusive_start_lock` (`src/services/curation/job_lock.py:235`): a
  non-blocking flock around check-busy-then-write.

### 2a. Jobs enqueued by a request and executed **inside an API worker process** (asyncio task)

All of these share these properties:
- **Execution:** the uvicorn worker that took the POST. They die when that worker is
  recycled or crashes.
- **Concurrency:** no limit across processes beyond the per-project singleton.
- **Shutdown:** not cancelled or awaited at shutdown (asyncio R5; `lifespan.py:365-382`
  only stops the periodic loops).
- **UI visibility:** a status route reads the file, plus SSE where noted.

| Job | Enqueue (file:line) | Claim / X-proc? | Retries | Cancel | Progress | Crash recovery |
|---|---|---|---|---|---|---|
| Reprocess | `reprocess_job.py:121-182` (`create_job` then `create_task(run_job)` :181) | start lock :132, yes | none | cancel.flag :185 | FileJob counts | heartbeat stale → `interrupted` (lazy + startup reconcile); manual resume |
| Assist run | `assist/runner.py:71-90` (`create_task` :87) | start lock `assist/job.py:80`, yes | none | cancel.flag `assist/job.py:155` | FileJob | interrupted; manual |
| Dataset import (+ undo, resume) | `dataset_import/runner.py:113,345,450` claims; tasks :398, :414 | start lock :123/:350/:456, yes | none | `cancel_import` :485 | FileJob + SSE | interrupted; `POST .../resume`. The resume rescan runs inline in the request (`routers/curation/datasets.py:458`) |
| Dataset upload extraction | `dataset_import/upload.py:240-268` | n/a, runs in the request | n/a | n/a | n/a | request dies. Blocking extract on the event loop (asyncio R1) |
| Combine | `projects/combine/service.py:113-172` (`_spawn` :169) | unique job dir + target create (`op_type=create`); cancel lock :230 | none | `cancel` :209 | FileJob + `combine.progress` events | interrupted; resumable (target left `building`) |
| Bulk ingest run | `bulk_ingest/source_runs.py:36` → `runner.py:111` claim, task :490-520 | start lock :117, yes | dispatcher re-drains `interrupted` | `cancel_run` :179 | FileJob | interrupted → next dispatcher `drain` resumes |
| Embedding viz (UMAP) | `embedding_viz.py:198-230` (`create_task` :227) | **none**: `_is_busy()` then write (:213-225). **NO** | none | cancel.flag :233 | FileJob | heartbeat stale → failed/interrupted |
| Selection (k-center) | `selection/job.py:123-148` (`create_task` :145) | **none**: `_is_busy()` then write. **NO** | none | cancel.flag :151 | FileJob | same. `k_center_greedy` runs **on the event loop** (:218) |
| Item scores | `item_scores/job.py:159-194` | start lock :179, yes | none | :200 | FileJob | interrupted |
| Probe | `probe_job.py:229-285` | start lock :258, yes | none | :300 | FileJob | interrupted |
| Promote | `training/promote_job.py:135-210` | start lock :141, yes | none | no | FileJob | interrupted; `run_detached` shields the work from request cancel |
| Region clustering, FP centroids | `clustering/region_cluster_jobs.py:158-260`, `:316-345` | **none**: read `running` flag then write. **NO** | none | none | `job.json` file | stale after **1800 s** (:55): a dead job blocks re-runs for 30 min |
| Open-vocab ingest drain | `open_vocab_ingest.py:63-91` (`create_task` :87) after every ingest | `_PASS_LOCK = asyncio.Lock()` :39, **per process only** | sweeper re-runs stale `pending` | none | `open_vocab_status` on the image | sweeper (2b) picks up stale `pending` |
| Project delete finish | `routers/curation/projects.py:296-336` (`create_task` :325) | per-process dict `_BACKGROUND_DELETE_TASKS`; record OCC in `projects/delete.py:361-519` | re-DELETE | n/a | `project.deleted` event | record stays `deleting`; re-DELETE retries |
| Cluster-geometry backfill | `crop_orders.py:76-91` from a GET | per-process `_BACKFILLING` set | none | none | none | idempotent cache fill; lost on recycle (harmless) |
| Directory ingest | `routers/ingest/directory.py:115-190` | n/a, inline `rglob` + `read_bytes` in the request (asyncio R1) | n/a | n/a | n/a | request dies |

### 2b. Periodic loops started in the API lifespan, running **once per uvicorn worker (×32)**

| Loop | Where | Interval | Cross-process guard | Effect of ×32 |
|---|---|---|---|---|
| Project registry poll | `projects/bootstrap.py:211,266`; `lifespan.py:103` | config | none needed (read-only cache) | OK: per-process cache |
| Config store poll | `config_store/store_poll.py:117-149`; `lifespan.py:119` | `config_poll_interval_s` | none needed (cache) | OK |
| Ops metrics refresh | `ops_metrics_refresh.py:229-267`; `lifespan.py:108` | 30 s | `_claim_turn` mtime stamp (:229): one refresher per interval, best effort | OK (a race means two refreshes, never none). Dead-worker multiprocess files are never cleaned (API-14) |
| Index bootstrap + kNN warmup | `core/dependencies.py:140-160`; `lifespan.py:113` | once per process start | per-process `_ensure_indexes_lock` (`index_bootstrap.py:133`) | idempotent `put_mapping` ×32 at every recycle; warmup task never cancelled (asyncio R9) |
| Startup job reconcile | `startup_reconcile.py:31-50`; `lifespan.py:135-142` | once per process start | none | **D4**: can mark a sibling's live job `interrupted` |
| GPU-arbiter reconcile + promoted-model reload | `lifespan.py:145-213`; `training/gpu_arbiter.py:249` | 15 s | none ("idempotent ... safe re-enforcement", `lifespan.py:156`) | **D5**: 32 processes issue docker stop/start and Triton loads |
| Ingest dispatcher | `bulk_ingest/dispatcher.py:204-263`; `background_jobs.py:397` | 5 s ± 20 % | flock `dispatch.lock` per project per tick (:211), correct | 32 pollers; loop dies on a `for_each_project` error (R9) |
| Open-vocab sweeper | `open_vocab_sweeper.py:92-125` | `OP_OPEN_VOCAB_SWEEP_S` | flock held across the sweep (:99), correct | same as above; holds a flock across awaits |
| Event bus tail | `event_hub.py:444`; `lifespan.py:122-132` | continuous | none needed (each process serves its own SSE clients) | OK by design |

### 2c. Worker containers (profile `curation`, `export`, `evaluator`, `training`)

| Worker | Entry / claim | X-replica? | Retries | Cancel | Progress | Crash recovery | GPU |
|---|---|---|---|---|---|---|---|
| Detection worker | `scripts/curation/worker/runner.py`; polls `pending_*` items, in-process `in_flight` set (:271-275), OCC writes | No (in-process set); enforced by `container_name` | items stay pending; no-verdict counter | pause flag | `worker_liveness` heartbeat, metrics | items stay pending; picked up again | via Triton, segmenter, VLM |
| VLM worker | `scripts/curation/vlm_worker.py`; same pattern, `in_flight_lock` :146 | same | same | pause flag | same | same | via VLM |
| Auto-label worker | API `autolabel/job.py:454-511` writes `trigger.json`; worker `auto_label_worker.py:300-323` claims by read then `unlink` | **No**: two readers can both read before either unlinks; `suppress(FileNotFoundError)` hides the loss | none | cancel.flag at stage boundaries `autolabel/job.py:514` | `state.json`, SSE by mtime | heartbeat stale → `failed` (lazy `get_state` :239) or `interrupted` (startup). The trigger is gone (at-most-once) | CPU/IVF |
| Assist-loop worker | `assist_loop/job.py:168-200` (start lock, correct); worker claim `claim_trigger` :142-156 (read then `unlink`) | No (same pattern) | none | `cancel_loop` :230 | FileJob | interrupted; resume rewrites the trigger | via VLM |
| Cluster refresh daemon | `scripts/curation/cluster_refresh_daemon.py`; a timer that calls API routes to start auto-label | single container | backoff 600 s | pausing a project cancels | n/a | stateless restart | none |
| Model export worker | API `model_export.py:196,286` queues `queue/*.job.json`; worker `export/export_worker.py:71-92` claims with an atomic `rename` into `claimed/` | **Yes** for the claim; startup `reconcile_claimed` :95 fails *every* claimed job (single-worker assumption) | none | none | task file | failed at next start | `API_GPU_ID` |
| Trainer | API `training/jobs.py:116-201` writes `<id>.job.json` (`target.exists()` check :152, then write; not atomic); trainer `docker/trainer/job_protocol.py:102` lists non-terminal jobs, **no claim** | No (enforced by `container_name`) | re-runs a non-terminal job on restart | `.cancel` file between epochs (`jobs.py:204`) | `status.json` + `heartbeat_at`; API marks `lost` (`jobs.py:222`) | re-run from scratch | `OP_TRAIN_GPU_ORDER`; GPU arbiter stops other GPU services |
| Evaluator (bake-off) | API writes `bakeoff_jobs/*.job.json`; runner `bakeoff_runner.py:404-421` runs every pending file, renames to `done/` **after** finishing | No claim; crash re-runs the job (at-least-once, no dedupe) | none | not found | MLflow, report files | re-run | `EVALUATOR_GPU_ID`; arbiter `bakeoff_active` |

### 2d. CLI and script jobs

- These run on the operator's command (`docker compose run` / `exec`).
- Examples: `scripts/curation/*` backfills, `scripts/curation/bakeoff/*`, the
  `openprocessor models install|export`, and `sample coco`.
- They bind a project with `--project` and use `make_script_opensearch`. They write
  through the same OCC helpers.
- **Out of scope for the queue.** One-shot operator tools should stay plain CLIs.
- The exception is the IVF retrain and cluster refresh, which are periodic (2b/2c).

### 2e. Request-path fan-out (owner rule: no `asyncio` fan-out in request handlers)

| Site | Shape |
|---|---|
| `routers/curation/review_resolve.py:221-227` | `gather` over crop ids, semaphore 16, one OCC write per crop |
| `routers/curation/edit_undo.py:126-143` | `gather` over crop ids, semaphore 32, one undo per crop |
| `services/curation/ingest_batch.py:92,134,158,306`; `ingest_detect.py:151,319`; `visual_search/ingest_batch.py:320,339` | the core ingest data plane: decode, then batched Triton calls inside one request |
| `services/labeling/vlm_labeler_classify.py:65,291` (behind `POST /curation/vlm/label_batch`) | VLM chunk fan-out, unbounded beyond the token bucket (R6) |
| `services/curation/semantic_search.py:291` | two queries in parallel (kNN + count) |
| `routers/health.py:159-160` | two probes in parallel |

### 2f. Per-process state that callers treat as global

These come from `artifacts_local/audit/api.md`. A queue does not fix them, but they belong to
the same "one process assumed" class, and Phase 1 must not build on them.

| Site | What it assumes | Audit |
|---|---|---|
| `clients/curation_opensearch/settings_doc.py:34-54,122-124` | 5 s settings cache, invalidated only in the writing process; it also caches `{}` on error | API-7 |
| `services/projects/capacity.py:24-30,192-231` | 10 s capacity cache used for the create-time 409; invalidated only locally | API-7 |
| `services/curation/diverse_order.py:142-198`, `clustering/outliers.py:45-132`, `routers/curation/regions_fp.py:68-71`, `services/curation/field_coverage.py:22-23` | unbounded or TTL caches keyed by query; stale across workers | API-7 |
| `services/curation/eval_datasets.py:161-300` | cache key narrower than its inputs | API-8 |
| `clients/model_residency.py:41,61-74` | `_ready` set per process, never reset after a Triton restart | API-12 |
| `services/labeling/vlm_endpoint_probe.py:40,326` | `MAX_CONCURRENT_PROBES = 2` per process, so 64 across the API | API-13 |
| `PROMETHEUS_MULTIPROC_DIR` (`docker-compose.yml:195,250`) | dead-worker files never cleaned (`mark_process_dead` unused); grows with every recycle | API-14 |
| Class-name cache invalidated only in the promoting worker | per process | API-1 |

---

## 3. Defects found (single-process assumptions and related)

Severity uses the project's scale. "Live" means it is reachable with today's default compose.

- **D1 (high, live): heavy jobs run inside recyclable request workers.**
  - Every job in 2a runs as an `asyncio` task in the uvicorn worker that received the POST.
  - `--limit-max-requests=10000` (`docker-compose.yml:264`) terminates a worker after 10,000
    requests, with 30 s of graceful shutdown. Without jitter, all 32 workers recycle at
    about the same point under even load.
  - A bulk ingest (thousands of loopback requests) therefore recycles workers *while* a
    reprocess, import or combine is running in one of them.
  - The job dies. Resumable jobs show `interrupted`; the others show `failed` with a
    heartbeat-stale reason.
  - **Reproduced** (API-6): with a minimal uvicorn app at `--limit-max-requests 6`, a
    request-started 10 s `asyncio` task logged ticks 0-5 and never reached DONE after
    "Maximum request limit of 6 exceeded. Terminating process".
  - UI status polls, thumbnails and SSE all count toward the 10,000, so recycling happens in
    normal use, not only under ingest load.
  - The API also never cancels or awaits these tasks at shutdown (R5) and keeps unpruned
    `_tasks` dicts.
  - The CPU-heavy parts compete with request latency in the same event loop.
    `selection/job.py:218` runs `k_center_greedy` on the loop.
- **D2 (medium, live): singleton starts without a cross-process lock.** A check-then-write
  with no start lock lets two POSTs on two workers both start a job.
  - `autolabel/job.py:454-511`: `is_busy()`, then write `state.json` and `trigger.json`.
    - The second POST overwrites the first trigger.
    - The first caller got a 202 and a `job_id` that never runs.
  - `selection/job.py:138-148` and `embedding_viz.py:211-230`: both start an `asyncio` task
    on the same state file. The runs overwrite each other's state.
  - `clustering/region_cluster_jobs.py:158-260`, `:316-345`: read the `running` flag, then
    write it. Two 7-8 min clustering runs then write assignments concurrently.
- **D3 (high, live): class-registry lost update across processes.**
  - `ClassRegistry.add_class` (`clients/curation_opensearch/registry.py:116-140`),
    `rename_class` and `merge_class` load the file, compute
    `new_id = max(id) + 1`, then atomically replace the file. No cross-process lock.
  - Concurrent writers can be any of:
    - two `POST .../classes` on different API workers;
    - the open-vocab pass (`class_ensure.py:57-65`, running in up to 32 processes; see D6);
    - dataset import (`dataset_import/prepare.py:419`);
    - combine (`combine/execute.py:143`).
  - Two writers that read the same version cause one of two failures:
    - both assign the same `class_id` to different names;
    - one class silently disappears while its caller already returned the id.
  - This violates "class identity is the name" at the storage layer.
  - `_registry_lock = asyncio.Lock()` (:313) is declared and unused (dead code).
- **D4 (low/medium, live): startup reconcile assumes it is the only process.**
  - Each recycled worker runs `reconcile_all_jobs()` (`lifespan.py:135-142`). The module
    docstrings claim "nothing in this process can legitimately hold running" (for example
    `selection/job.py:98-105`), but sibling processes can.
  - `reconcile_stale_running` (`job_reconcile.py:354-366`) treats a **missing** heartbeat
    as stale.
  - `selection` and `embedding_viz` write `running` with no heartbeat (`selection/job.py:142-145`).
  - So a sibling's freshly started job can be flipped to `interrupted` while it runs. Its
    next state write flips it back, and a resume may start a second run in between.
- **D5 (medium, live): the GPU-arbiter loop runs in 32 processes.**
  - `_arbiter_reconcile_loop` (`lifespan.py:145-167`) calls docker stop/start (with the
    arbiter overlay) and Triton reloads from every worker every 15 s.
  - Each process decides from its own directory scan. Around a run's start or end, one
    process can `start` containers another has just `stop`ped. The design accepts this
    flapping because it calls the loop "idempotent".
  - It is ×32 load on the Docker socket and Triton.
- **D6 (medium, live): serialization is per process.**
  - The open-vocab `_PASS_LOCK` (`open_vocab_ingest.py:39`) is meant to give "one
    ingest-time pass at a time ... instead of multiplying the segmenter load".
  - It serializes within one process only, so up to 32 passes run against the segmenter.
- **D7 (latent): worker-container claims are correct only with one replica.**
  - Auto-label and assist-loop claim by read then `unlink`.
  - The trainer has no claim.
  - The evaluator has no claim and renames only after finishing.
  - The export worker's startup reconcile fails every claimed job.
  - Today the fixed `container_name` blocks scaling. Any "add a second worker" change breaks
    these silently.
- **D8 (low, live): at-most-once auto-label claim.**
  - The worker unlinks the trigger (`auto_label_worker.py:300-323`) before it writes
    `running`.
  - A crash in between loses the job. The state stays `queued` until `QUEUE_MAX_AGE_S`
    (3600 s, `autolabel/job.py:114`), then fails.
- **D9 (medium, live): a blocked event loop falsifies liveness.**
  - The job heartbeat is an *asyncio* task (`file_job.py:186-190`).
  - CPU work on the loop (D1, asyncio R1) stops heartbeats for its duration. Once that
    exceeds 30 s, another process's status read repairs the live job to
    `interrupted`/`failed` (`file_job.py:127-153`; `embedding_viz.py:183-195`).
- **D10 (medium): job success states lie on bulk item errors.**
  - OS-2 and OS-3: item-level bulk errors are dropped in
    `item_scores/job.py:226`, `embedding_viz.py:407`, `probe_predictions.py:238`,
    `combine/holdout.py:87` and `holdout_freeze.py:106`.
  - Background jobs report `completed` with partial writes.
  - The queue migration should not carry these forward; fix them in the same family PR.
- **D11 (medium): config first-create race (OS-1).**
  - `config_store/index.py` `save_config`/`activate` and `curation/keymap.py` write a missing
    doc without `op_type=create`.
  - Two first-writers both succeed.
- **D12 (low): region-cluster stale window.**
  - A worker that dies mid-run leaves `running: true`, which blocks new runs for 1800 s
    (`region_cluster_jobs.py:55`).
- **D13 (medium, live, reproduced): the job-state writers race on one temp file (API-4).**
  - `FileJob.write` (`file_job.py:79-83`) always writes `state.tmp`, then `replace()`.
    Two processes writing the same job collide.
    - In the audit's 4-process repro, every process hit `FileNotFoundError` thousands of
      times.
  - About 100 call sites call `write()` directly, bypassing `update()`'s flock. One example
    is `get_state` repairs that run on any status poll.
  - `job_reconcile.reconcile_stale_running` (`:376-378`) does an unlocked read-modify-write
    on the same temp name.
  - The same fixed-name pattern appears in `holdout.py:334-337`,
    `routers/curation/bakeoff.py:122-125`, `dataset_import/store.py:62-64`,
    `label_ops/store.py:190-192` and `registry.py:84`.
  - **Why it matters here.** The job record is the lease and the source of truth in the
    recommended design. A job runner (on a jobs worker) and a status poll (on an API worker)
    will keep writing the same record from different processes. This must be fixed before
    any job moves (P1.0).
- **D14 (medium, live): per-process caches and caps behave as if global (section 2f).**
  - The queue does not cure this.
  - It is listed so Phase 1 does not lean on these caches, for example a cached
    "model ready" or a cached capacity check inside a job.
  - These are fixed in the API-audit series, not here.
- **D15 (low, latent): Prometheus multiprocess files from recycled workers accumulate
  (API-14).**
  - Moving jobs out of the API does not change the recycle rate.
  - The new Celery worker containers must not reuse the API's `PROMETHEUS_MULTIPROC_DIR`.
  - The Celery prefork children use `worker_max_tasks_per_child`, so they need the same
    `mark_process_dead` handling if they export metrics in multiprocess mode.

Cross-references to the API audit:
- D1 = API-6;
- D3 = API-2;
- D4 = API-5;
- D13 = API-4.

Preconditions to keep neutral, not exploitable details:
- Any broker added must sit on the internal compose network with no published port and
  require authentication.
- Workers must accept JSON only [C6].
- Who can reach the broker can enqueue work that runs with the worker's privileges [C6].

---

## 4. Options compared (official docs, fetched 2026-11-03)

### 4a. What each option offers against our requirements

| Requirement | Celery 5.6 + Redis 8 | Celery + RabbitMQ | Dramatiq | RQ | arq | Huey | Temporal | Keep files/OpenSearch, fix coordination |
|---|---|---|---|---|---|---|---|---|
| Maintained | yes, 5.6.3 docs [C0] | yes | yes | yes | **maintenance-only** [A2] | yes (single maintainer) | yes | n/a |
| New service | broker (Redis 8) | RabbitMQ (Erlang) | Redis or RabbitMQ | Redis | Redis | **none with SQLite/file storage** [H1] | server + Postgres + Elasticsearch [T1] | none |
| Routing to GPU queues | `task_routes` + `-Q` [C5] | same | per-actor queue | per-queue workers | per-queue workers | one instance per queue | task queues | hand-rolled |
| Long jobs (min to hours) | ack timeout = visibility timeout, default 1 h; redelivered "again and again" past it [C1] | `consumer_timeout` default 30 min; quorum queues only from 4.3 [R1] | worker heartbeat 60 s [D1]; TimeLimit default 10 min, must be raised [D2] | per-job timeout, fork per job [Q1] | pessimistic: "may be called more than once" [A1] | timeouts | activity heartbeats (best) | heartbeat lease (have it) |
| Crash recovery | `acks_late` plus `task_reject_on_worker_lost` [C2] | same | acks only on finish [D0] | registries | re-run | retries | durable | reaper (have it) |
| Cancel running | revoke skips queued only; `terminate` is a last resort and "you must never call this programmatically" [C3] | same | `dramatiq-abort` (third party) [D3] | `send_stop_job_command` [Q1] | `Job.abort()` with `allow_abort_jobs` [A1] | revoke | cancellation | cancel.flag (have it) |
| Progress | `update_state` custom states [C2] (needs a result backend) | same | none built in | job meta | none | none | queries | FileJob (have it) |
| Periodic | beat, **exactly one** scheduler [C4] | same | none (needs a third-party package) | `CronScheduler` beta; multi-instance safety undocumented [Q2] | cron built in [A1] | built in | schedules | hand-rolled |
| asyncio code | no native `async def` tasks in 5.6; wrap with `asyncio.run` [S1] | same | AsyncIO middleware [D2] | no | native | no | native | n/a |
| Retries/backoff | `autoretry_for`, `retry_backoff`, `retry_jitter`, max 600 s [C2] | same | defaults: 20 retries, 15 s min backoff [D2] | yes | `Retry` | yes | yes | hand-rolled |
| Ecosystem/ops | Flower, very wide use, owner's ask | same | smaller | medium | small | small | large, heavy | n/a |

### 4b. Key doc-backed pitfalls that shape the design

- **At-least-once is the best any of them promises.**
  - Celery: with `acks_late` "the task may then run more than once if the worker crashes
    mid-execution", and tasks should be idempotent [C2].
  - arq: "Jobs may be called more than once!" [A1].
  - So **every option still needs our domain claim (lease)** for exactly-once *effect*.
- **Visibility timeout (Celery on Redis or Valkey).**
  - An unacknowledged task "will be redelivered to another worker and executed" after the
    timeout (default 1 h) [C1].
  - The timeout must be set in `broker_transport_options`, `result_backend_transport_options`
    and `app.conf.visibility_timeout` [C1].
  - The broker must not evict keys: set `maxmemory-policy` (we use `noeviction`) [C1].
  - **Mitigation in this plan:** a redelivered message finds a live lease and exits without
    running. Our heartbeat reaper (Phase 4), not the visibility timeout, detects dead workers.
- **Prefetch.**
  - For long tasks use `worker_prefetch_multiplier = 1` together with `task_acks_late = True` [C7].
- **Revoke.**
  - Revoke only skips tasks that have not started.
  - Revoked ids are in memory unless the worker runs with `--statedb` [C3].
  - Our cancel stays cooperative (`cancel.flag`). The job record's `cancelled` status makes a
    stale revoke irrelevant.
- **Shutdown.**
  - TERM triggers a warm shutdown that "will finish all currently executing tasks" [C8].
  - Soft shutdown (`worker_soft_shutdown_timeout`) is off by default [C8].
  - On SIGKILL, running tasks "will be lost" unless `acks_late` [C8].
- **Beat.**
  - Only one scheduler may run, or tasks are duplicated [C4].
  - The default schedule store is a local file [C4]. Put it on the state volume.
- **Serializers.** JSON is the default since 4.0; pickle is "inherently insecure"; pin
  `accept_content=['json']` [C6].
- **Testing.**
  - `task_always_eager` "is by definition not suitable for unit tests".
  - Use the `celery_worker` fixture (a real worker in a thread) for integration tests,
    and mocks for unit tests [C9].
- **Licences.**
  - Redis 8 is tri-licensed RSALv2/SSPLv1/AGPLv3 [V1]. The repo is AGPL, so the AGPLv3
    option fits it, and OpenTranscribe already ships Redis 8 (section 4d).
  - Valkey is BSD and stewarded by the Linux Foundation [V2]. It is the fallback if a
    downstream redistributor objects to the Redis licence.
- **Persistence (if we keep the queue durable).**
  - AOF `appendfsync everysec` is the default and loses at most about 1 s of writes [V3].
  - Because our job records are the source of truth and the reaper re-enqueues `queued` jobs
    (Phase 4), broker loss costs a short delay, not data.
- **Broker security.**
  - Redis is "designed to be accessed by trusted clients inside trusted environments".
  - Bind and firewall it, and use `requirepass` or ACLs.
  - Protected mode applies only when no password is set and it binds all interfaces [V4].

### 4c. The "no new library" option

- **The documented OpenSearch pattern.** OCC claims (`if_seq_no`/`if_primary_term` → 409;
  `op_type=create` "indexes a document only if it does not already exist") [O1].
  OpenSearch's own Job Scheduler acquires a lock document with seq_no/primary_term before
  running a job [O2].
- **What it would require.** A leased jobs index per project, plus our own
  dispatcher/worker loop, retry/backoff, routing and a periodic scheduler with leader
  election.
- **Why it is not recommended.** It is exactly the "hand-roll what a library does" the owner
  rules out.
- **The useful part to keep.** OCC/`op_type=create` stays the right tool for *domain*
  records (OS-1 fix) and for the per-job lease if J5 picks OpenSearch.

### 4d. Proven patterns in OpenTranscribe (`<OpenTranscribe checkout>`, read only)

The owner runs Celery in production in OpenTranscribe. Its setup is the reference this plan
aligns with. Paths below are relative to that repo.

**Stack**
- Broker and result backend: Redis 8 (`redis:8.2.2-alpine3.22`, `docker-compose.yml:95-125`).
  - The password is optional (`REDIS_PASSWORD`).
  - The port is published on loopback only (`127.0.0.1:${REDIS_PORT:-5177}`).
  - Data lives on volume `redis_data`. No AOF flags are passed.
- Celery app: `backend/app/core/celery.py`. Workers start with
  `celery -A app.core.celery worker -Q <queues> --hostname <role>@%h -E`.

**Queues**
- One constants class, `CeleryQueues` (`backend/app/core/constants.py:27-64`):
  `gpu, download, cpu, nlp, embedding, utility, cloud-asr, cpu-transcribe, gpu-transcribe,
  gpu-diarize, redaction, celery`.
- `task_queues` is declared explicitly. `task_create_missing_queues=False` turns a typo into
  an error at dispatch instead of a phantom queue (`celery.py:154-157, 454`).
- Every task needs a `task_routes` entry, and a task must never put `queue=` in its decorator.
  Enforced by `tests/unit/test_task_routes_complete.py`; rules in `app/tasks/CLAUDE.md`
  ("Conventions / patterns").
- **Lesson ("a valid route is not a drained route"):** the lite overlay scaled the GPU
  consumers to 0, so tasks routed there sat in a queue nothing drained while the API had
  already answered 200. `tests/unit/test_lite_mode_queue_consumers.py` now derives the consumer
  set from the compose files and fails on a route into a dead queue.

**Workers per type** (`docker-compose.yml`)
- GPU (`celery-worker`, :350-455): `-Q gpu --pool=threads --concurrency=1`, a configurable
  `stop_grace_period`.
- download (:456): prefork `-c 5`.
- cpu (:510): `-Q cpu,utility,cpu-transcribe -c 8 --max-tasks-per-child=20`.
- nlp (:755): `-Q nlp,celery -c 4`.
- embedding (:805): `-c 1`.
- split GPU workers (:945, :1019) under a profile.
- Flower (:1164).
- `worker_prefetch_multiplier=1` is set globally, which is GPU-safe (`celery.py:318-325`).

**Acks and timeouts** (`celery.py:340-455`)
- `visibility_timeout=21600` (6 h). The comment says it "MUST exceed the longest task":
  with the 1 h default, "ANY transcription over one hour was silently duplicated", on the GPU,
  concurrently (issue #284).
- `task_soft_time_limit=3 h` and `task_time_limit=3 h 15 m` stay *below* the visibility
  timeout, "so a timeout kill happens before redelivery". SIGALRM limits are unreliable under
  `--pool=threads`.
- Worker loss is recovered in minutes, not 6 h: transcription stages hold an app lease, and a
  beat orphan reaper returns a dead holder's message (`app/core/broker_orphans.py`).
- **Lesson:** Celery's cold-shutdown cancel *acks* a running `acks_late` task on Redis
  (verified on 5.6 in their live test). So the reaper must also recover runs whose message is
  gone, from the application's own records (`broker_orphans.py` docstring).

**Leases and duplicates**
- `app/core/task_liveness.py` keeps two Redis markers per run:
  - a `queued` marker;
  - a heartbeat lease, refreshed by a daemon thread, about 90 s TTL, that records which
    message holds it.
- Reads fail safe: Redis unreachable means `UNKNOWN`, never "dead".
- `app/tasks/transcription/run_ownership.py` stands a stage down when its run was superseded,
  or when another delivery of the same message holds the lease. This is the same lease idea
  as section 5, rule 3.

**Shutdown and robustness** (`celery.py:181-215, 351-399`)
- `worker_soft_shutdown_timeout=30`, `worker_enable_soft_shutdown_on_idle=true`.
- `worker_cancel_long_running_tasks_on_connection_loss=False`: on Redis an unacked message
  survives a blip.
- `broker_connection_retry_on_startup=True`, `broker_connection_max_retries=None`.
- `worker_proc_alive_timeout=30`. Issue #631: a slow `worker_process_init` caused 69,231
  fork/SIGKILL cycles over 10 h 46 m with zero tasks consumed. Rule: no network, locks or
  model loads in `worker_process_init`.
- Broker socket keepalive, `socket_timeout=30`, `health_check_interval=25`, and
  `retry_on_timeout=False` for the broker. Issue #1144: a half-open socket wedged a consumer
  forever.

**Cancellation** (`app/core/task_cancellation.py:1-60`, `app/tasks/CLAUDE.md:190`)
- `revoke(terminate=True)` was removed: "dead on every pool type".
  - It cannot signal a thread-pool task.
  - It revoked the *application* id, not the Celery message id, because `apply_async` minted
    its own.
- Cancel is cooperative: a per-run Redis key that the task checks at stage boundaries.
- A user cancel must not requeue (unlike a shutdown abort).

**Progress to the UI**
- A Redis-backed progress tracker (`app/services/progress_tracker.py`).
- Redis pub/sub to the WebSocket manager (`app/utils/websocket_notify.py`,
  `app/api/websockets.py`).

**Health**
- `backend/scripts/celery_pool_healthcheck.py`: `inspect ping` is answered by the
  MainProcess even when the prefork pool is wedged. The script reads `inspect stats` and
  detects a dead pool or a respawn storm (pid set replaced while the task counter is frozen).
- Beat health: schedule-file mtime under 300 s (`docker-compose.yml:876-890`).
- **Lesson:** nothing acts on `unhealthy`; `restart: always` fires only when the main process
  exits (`app/tasks/CLAUDE.md`).

**API process model**
- The backend runs **one uvicorn process per container** with no `--workers`
  (`backend/Dockerfile.prod:550`).
- All background work runs in Celery workers.

**Docs**
- `docs/ARCHITECTURE.md:49-102`: services, Celery-with-Redis rationale, worker table.
- `docs-site/static/architecture/celery-queues.html` and `app/tasks/CLAUDE.md`: conventions
  and gotchas.

**How this plan aligns, and where it differs on purpose**

| Topic | OpenTranscribe | This plan | Why different (if) |
|---|---|---|---|
| Broker | Redis 8 | **Redis 8, same family** (J2) | same |
| Password | optional | **required**, generated by the installer | our fail-closed rule and LAN posture |
| Published port | loopback | **none** | nothing outside compose needs it |
| Persistence | default (no AOF flags) | AOF everysec, `noeviction` [C1][V3] | queued jobs survive a restart; eviction breaks kombu [C1] |
| Result backend | Redis, `result_expires=86400`, `task_track_started` | **none** (`task_ignore_result`) | our job record is the UI's source; `worker_deduplicate_successful_tasks` is replaced by the lease's terminal-state check |
| Queue constants + `task_create_missing_queues=False` + route-completeness test | yes | **adopt** | |
| Consumer-for-every-queue test from compose | yes (lite lesson) | **adopt** | |
| `prefetch_multiplier=1`, `acks_late`, `visibility_timeout` above the longest task, time limits below it | yes, 6 h | **adopt, 6 h** (bulk ingest, import and combine can run for hours) | |
| App lease + reaper for fast recovery, including a lost message | Redis keys | `FileJob` lease + beat reaper (J5) | our job records are already files that every status route reads |
| Cooperative cancel; no `terminate=True` | Redis key | `cancel.flag` (existing); `revoke` only for queued work, with `task_id=job_id` so the id is the Celery id | |
| Shutdown, keepalive, `proc_alive_timeout` settings | yes | **adopt verbatim** | |
| Pool health script + beat mtime check | yes | **port both** | |
| `--hostname <role>@%h -E`, Flower | yes | adopt; Flower under the `monitoring` profile | Flower is an extra web UI |
| Priorities (`queue_order_strategy: priority`) | yes | not now | our queues are few and FIFO is fine |
| Progress via Redis pub/sub | yes | keep `FileJob` + JSONL event log for now (J13) | avoids rewriting every status route in this series |

### 4e. The API worker model, in plain language

**What `--workers` does.**
- `uvicorn --workers N` starts one parent "process manager" and N separate Python
  *processes* [U1][F2]. Each is a full copy of the API.
- The operating system hands each incoming connection to whichever process accepts it first.
  Two requests from the same browser can land on two different processes.
- The processes share nothing in memory, so each one has its own:
  - event loop;
  - module-level variables, caches, locks and flags;
  - Triton gRPC pool (4 channels);
  - OpenSearch client;
  - 64-thread CPU executor (`lifespan.py:60-74`);
  - lifespan, which runs once per process;
  - `--limit-concurrency=512`, which is per process, so 16,384 in total at 32 workers;
  - `--limit-max-requests` counter.

**Why it exists.**
- A Python process runs Python code on one core at a time (the GIL).
- Several processes are the standard way to use more cores for request parsing, validation,
  JSON and the CPU-side preprocessing the API does before Triton.
- FastAPI's docs list "Docker Compose on a single server" as a case where several worker
  processes in one container make sense [F3].

**What we run today.** Three different numbers:
- `--workers=32` in `docker-compose.yml:263`;
- `--workers 16` in the image's default `CMD` (`Dockerfile:312`);
- a comment saying "Production: 64 workers" (`docker-compose.yml:260`).

None is derived from the host's cores. `nproc` is 48 on the dev host.

**What it means for state.** Anything kept in one process's memory is invisible to the other
N-1. A design is correct only if it gives the same answer with 1 worker and with 32. Today
these do not (sections 2-3):
- jobs running inside a worker (D1);
- in-process "only one at a time" locks (D2, D6);
- read-modify-write without a cross-process lock (D3, D13);
- startup repair that assumes no siblings (D4);
- periodic loops in every worker (D5);
- caches and ready flags that other workers never see invalidated (API-1, 7, 12);
- caps that are really 32× larger (API-13).

**Recommendation: make the API stateless, so the worker count is only a tuning knob.**
1. **No job execution in the API.** Requests write a job record and enqueue (Phases 3-5).
2. **No periodic side effects in the API.** Beat owns them (Phase 4). Lifespan only creates
   per-process resources (pools, clients, warm-ups); it never repairs or reconciles.
3. **No exclusion from process memory.**
   - Use start locks or flock for file-backed state, OCC for OpenSearch docs, and the lease
     for jobs.
   - Module-level `asyncio.Lock`/`Semaphore` is allowed only to coalesce a *cache* or bound
     *local* resource use. Each one carries a comment saying so.
4. **Caches never carry correctness.** They use a revision check (the project-registry and
   config-store pattern) or a documented staleness window. They are bounded
   (`cachetools.TTLCache`) and never cache an error as "empty" (API-7, API-8).
5. **Global limits are global.** Use a Redis-backed limit (the broker is now there), or
   document the limit as per process (API-13).
6. **Metrics** stay in Prometheus multiprocess mode, plus `mark_process_dead` (API-14).
7. **One knob.**
   - Remove `--workers` from the compose command and the Dockerfile `CMD`.
   - Set `WEB_CONCURRENCY=${API_WORKERS}`; uvicorn reads it when `--workers` is absent [U1].
   - The installer writes `API_WORKERS` from the host's cores.
     - Default: `min(nproc, 32)`.
     - Validate the default with the existing ingest and search benchmarks
       (`docs/PERFORMANCE.md` has the 32-worker baseline).
   - Size per-process pools *from* it: the 64-thread executor and Triton channels become
     `total / API_WORKERS`, so a smaller worker count does not starve the machine.
   - Delete the "64" comment.
8. **Orchestrated deployments** (Kubernetes, Swarm, Nomad):
   - Set `API_WORKERS=1`, one process per container, and scale replicas at the cluster level,
     as the FastAPI docs recommend [F2][F3].
   - OpenTranscribe already runs this way (`backend/Dockerfile.prod:550`).
   - Precondition: `/jobs` file records and flock need one host (E15). Multi-host
     orchestration therefore needs J5's OpenSearch or Redis job store first. Say so in the
     install docs.
9. **Proof.**
   - The cross-process tests from Phases 1-3 run with N spawned processes.
   - A live check runs the same scripted scenario (start, cancel and race jobs, edit classes,
     save configs) with `API_WORKERS=1` and `API_WORKERS=32`, and diffs the outcomes.
   - Add a ratchet in `scripts/codegen/`: no `asyncio.create_task` in `src/routers`/
     `src/services` outside the asyncio-audit `spawn` helper's allowlist, and no new
     module-level `asyncio.Lock`/`Semaphore`/mutable dict without an allowlist reason.

---

## 5. Recommended architecture

```
 request ──> api (API_WORKERS uvicorn processes; stateless, section 4e)
               │ validate, create job record (FileJob dir, status=queued) under start lock
               │ enqueue by NAME: celery send_task('op.jobs.<kind>', [project, job_id], task_id=job_id)
               ▼
         redis 8 (internal network, no port, requirepass, AOF, noeviction)
               │
   ┌───────────┼────────────────────────┬──────────────────────────┐
   ▼           ▼                        ▼                          ▼
 jobs-worker  autolabel-worker         assist-loop-worker          (later) evaluator
 -Q jobs      -Q autolabel -c1         -Q assist_loop -c1          -Q gpu_eval -c1
 api image    api image                api image                   evaluator image
   │ task wrapper: bind project → lease claim (FileJob) → heartbeat THREAD →
   │ asyncio.run(handler) → cooperative cancel.flag → terminal state
   ▼
 job record (files on /jobs and the state volume) = status, progress, cancel, result → existing routes + SSE

 jobs-beat (exactly one): ingest dispatch tick, open-vocab sweep, GPU-arbiter reconcile + promoted reload,
                          lease reaper (stale running → interrupted; stale queued → re-enqueue),
                          cluster-refresh check, IVF retrain check  → all send tasks to queue `periodic`
 periodic tasks run on jobs-worker (-Q jobs,periodic)

 unchanged: detection worker, VLM worker (item state machine + OCC), trainer (file protocol),
            model-export worker (rename claim), per-process caches/pollers, event hub
```

### Design rules

1. **Enqueue by name.**
   - Add one thin client, `src/clients/task_queue.py`, with
     `enqueue(task_name, *, project, job_id)` (queue from `task_routes` only) and
     `revoke(job_id)`. The Celery `task_id` is always the job id: OpenTranscribe's revoke
     was dead because the two ids differed.
   - Services call it and never import task modules. This keeps the layering ratchet
     intact: tasks import services, never the reverse.
   - If the broker is down, the enqueue raises. The route returns 503 `queue_unavailable`
     and marks the job record `failed` (fail closed).
2. **The job record is the truth; the broker is transport.**
   - Set `task_ignore_result=True`. There is no result backend, and the UI never reads Celery.
   - Every wire status route keeps reading `FileJob`, so status wire models change only where
     a job gains a `queued` state (selection, viz, item scores, probe, region clustering).
     Those need `make contracts` and a Cropwright check.
3. **One task wrapper, `src/tasks/job_task.py`, for every job kind.** It runs these steps:
   1. Resolve and bind the project (fail closed if missing or archived).
   2. Build OpenSearch with `make_script_opensearch` (guarded).
   3. **Lease claim:** under the job's start lock, accept only `queued`, or `interrupted`
      when the kind is resumable and `attempt < 3`. Write `status=running`,
      `owner=<hostname:pid>`, `attempt+=1`, and touch the heartbeat.
      - A live lease held by someone else means this is a redelivery: log it and return,
        which acks the message.
      - A terminal status means return.
   4. Start a **heartbeat thread**, so blocking code cannot starve it (fixes D9).
   5. `asyncio.run(handler(job, ctx))`.
   6. Map exceptions to `failed` with the message, `CancelledError` to `cancelled`, and
      success to `completed`/`completed_with_errors`.
   7. Stop the heartbeat in `finally`.
4. **Celery config** (`src/tasks/app.py`), aligned with OpenTranscribe
   `backend/app/core/celery.py` (section 4d):
   - **Queues:**
     - a `TaskQueues` constants class in `src/tasks/queues.py` (`jobs`, `periodic`,
       `open_vocab`, `autolabel`, `assist_loop`, `gpu_eval`);
     - `task_queues` declared from it, `task_create_missing_queues=False`;
     - `task_routes` with an entry for every task by exact name; no `queue=` in decorators
       [C5].
   - **Connection:**
     - `broker_url` from `OP_QUEUE_URL`, password read from a secret file, not argv;
     - `broker_connection_retry=True`, `broker_connection_retry_on_startup=True`,
       `broker_connection_max_retries=None`.
   - **Serialization:** `task_serializer=result_serializer='json'`,
     `accept_content=['json']` [C6].
   - **Delivery:**
     - `task_acks_late=True`, `task_reject_on_worker_lost=True` [C2];
     - `worker_prefetch_multiplier=1` [C7].
   - **Broker transport options:** `visibility_timeout=21600` (6 h, above the longest job;
     the same value in `app.conf.visibility_timeout`) [C1]; socket keepalive with
     `TCP_KEEPIDLE/INTVL/CNT` and `TCP_USER_TIMEOUT`; `socket_timeout=30`,
     `socket_connect_timeout=10`, `health_check_interval=25`, `retry_on_timeout=False`.
     These are copied with their rationale from OpenTranscribe `celery.py:181-215, 417-455`.
   - **Time limits:** `task_soft_time_limit=10800`, `task_time_limit=11700`, below the
     visibility timeout. Per-task overrides only where a job legitimately runs longer.
   - **Workers:**
     - `task_ignore_result=True` (no result backend; section 4d table);
     - `worker_max_tasks_per_child=50` [C3];
     - `worker_proc_alive_timeout=30`, with `worker_process_init` kept free of I/O;
     - `worker_soft_shutdown_timeout=30`, `worker_enable_soft_shutdown_on_idle=True`,
       `worker_cancel_long_running_tasks_on_connection_loss=False` [C8];
     - `worker_send_task_events=True`, `task_send_sent_event=True` (Flower).
5. **Beat.**
   - One `jobs-beat` container with a fixed `container_name`, so it cannot be scaled [C4].
   - The schedule file goes on the state volume.
   - Each periodic task is cheap and idempotent, and enqueues real jobs through the same
     start locks as a request.
6. **Concurrency.**
   - `jobs`: `--concurrency=4` (J9). This replaces "as many as there are API workers".
   - `autolabel`, `assist_loop`, `gpu_eval`: `-c 1`.
   - The open-vocab drain moves to its own queue `open_vocab` with `-c 1`, which is the
     documented intent behind `_PASS_LOCK` (fixes D6).
7. **GPU.**
   - `jobs-worker` gets the same reservation as `api` (`API_GPU_ID`), because UMAP/cuML and
     any in-process encoder fallback ran in `api` before.
   - The GPU queues pin through compose `device_ids`, exactly as the existing GPU services do.
8. **Redis service** (default install, no profile; J2/J3):
   - Image: `redis:8.2.x-alpine`, the same family OpenTranscribe pins, with a pinned digest.
   - Flags: `--appendonly yes --appendfsync everysec --maxmemory-policy noeviction`.
   - Password from a config file the installer generates (mode 600). The password is
     **required**, unlike OpenTranscribe's optional one.
   - No `ports:` (OpenTranscribe publishes on loopback; we need no host access). On
     `triton_net`.
   - Healthcheck: `redis-cli` ping with auth.
   - Named volume `openprocessor-queue`. No backup requirement (documented).
9. **Worker conventions from OpenTranscribe.**
   - `--hostname <role>@%h -E`.
   - A per-service `stop_grace_period`.
   - A pool healthcheck ported from `backend/scripts/celery_pool_healthcheck.py` (a ping
     cannot see a wedged prefork pool).
   - Beat health from the schedule file's mtime.
   - Prefork for `jobs`. `--pool=threads -c 1` only where a GPU model is held in-process
     (none in this plan's first phases).

### What NOT to migrate, and why

- **Detection worker and VLM worker.**
  - These are data-plane streams over the item state machine (`pending_detection`,
    `pending_verification`) with OCC writes.
  - One Celery message per item would duplicate state that already lives in OpenSearch, add
    millions of messages, and lose the producer/consumer batching that keeps the GPUs fed.
  - Keep them. If more than one replica is ever wanted, add an OCC lease field per item
    batch at that time (D7, J6).
- **Trainer.**
  - Hours-long GPU runs in a separate image with the GPU arbiter.
  - The file protocol already gives status, heartbeats, cancel and lineage, and
    `container_name` enforces one trainer.
  - Under Celery-on-Redis the visibility timeout would have to exceed the longest run [C1],
    or RabbitMQ's `consumer_timeout` would have to be raised per queue [R1]. That adds risk
    and gains nothing.
  - Add only an atomic claim (Phase 1, P1.6) so a second trainer is safe.
- **Model-export worker.** The `rename` claim is correct. Only make `reconcile_claimed` safe
  for more than one worker (P1.6). Migrating it is optional uniformity (J6).
- **Per-process caches** (registry, config, stats, index bootstrap) and the event-hub tail.
  These are correct per process and are not exclusion.
- **`ingest_writer` group commit.** It is request-scoped and the caller awaits its own ack.
- **Operator CLIs.**

---

## 6. Judgment calls for the owner

- **J1. Queue library.**
  - *Owner confirmed: Celery 5.6*, with OpenTranscribe's conventions (section 4d). Reuse its
    patterns, not its code: the repos are separate, and nothing is ported into
    transcribe-app.
  - Options:
    - Celery: beat, routing, revoke, Flower, broadest docs, the owner's ask.
    - Dramatiq: acks-late by default, heartbeat-based worker liveness [D0][D1], AsyncIO
      middleware. But cancel and periodic scheduling need third-party packages.
    - Huey: SQLite/file storage with no new service [H1]. But it has one maintainer, and one
      Huey instance per queue.
    - arq: excluded, maintenance-only [A2].
    - Temporal: excluded; it needs Postgres and Elasticsearch [T1].
  - *Trade-off:* Celery's weak spots are no native `async def` [S1] and the visibility timeout
    [C1]. Both are neutralised by the wrapper (`asyncio.run`) and the lease.
- **J2. Broker.**
  - *Recommend Redis 8*, the same broker OpenTranscribe runs (`docker-compose.yml:95-125`
    there).
  - Why:
    - Celery documents the Redis transport as feature complete [C0].
    - The AGPLv3 licence option fits this AGPL repo [V1].
    - One broker across the owner's two products means one set of operational knowledge.
  - Hardening beyond OpenTranscribe: a required password, no published port, AOF on,
    `noeviction` [C1][V3][V4].
  - Fallbacks:
    - Valkey (BSD) [V2], only if a redistributor objects to the Redis licence. It needs the
      transport spike.
    - RabbitMQ: rejected. It is a heavier service, and its `consumer_timeout` defaults to
      30 min [R1].
- **J3. Default install.**
  - *Recommend* these in the core (no profile): `redis`, `jobs-worker`, `open-vocab-worker`
    and `jobs-beat`. Reprocess, import, combine, selection and the rest are core curation
    routes today and would 503 otherwise.
  - The `autolabel`/`assist_loop` workers replace the existing curation-profile containers
    in place (Phase 5).
  - Flower goes under `monitoring`.
  - A compose-derived test asserts that every routed queue has a consumer in the default
    install. This is OpenTranscribe's lite lesson.
  - *Cost:*
    - 4 more containers;
    - Redis has a small idle footprint;
    - the installer generates one more secret;
    - `openprocessor status/health` grows four rows.
  - *Alternative:* a `jobs` profile. That gives a smaller default but a broken-by-default
    curation surface.
- **J4. Delivery semantics.**
  - *Recommend* `acks_late` + `reject_on_worker_lost` + the lease + an attempt cap of 3.
    A redelivery to a live lease is a no-op; one to a stale lease resumes (if resumable) or
    marks `interrupted`.
  - *Alternative:* early ack (Celery's default) plus a reaper that re-enqueues `queued`. That
    is simpler, but a crash between ack and claim relies on the reaper alone.
- **J5. Job-state store.**
  - *Recommend* keeping `FileJob` on the shared volumes (single host; flock works across
    containers; every status route and wire model reads it already).
  - *Alternative:* a per-project OpenSearch `jobs` index with OCC leases [O1][O2]. It is
    multi-host-ready but a large rewrite of every status route, and multi-host is out of scope.
- **J6. Scope of migration.**
  - *Recommend* migrating 2a (all), the 2b periodic loops (not the caches), auto-label and
    assist loop.
  - Make the evaluator optional in Phase 6.
  - Keep the trainer, export worker and detection/VLM workers (section 5).
- **J7. Request fan-out rule.**
  - *Recommend:*
    - Convert `review_resolve` and `edit_undo` to one `mget` plus one bulk with per-action
      `if_seq_no`. Batches above `OP_BATCH_SYNC_MAX` (default 500) become a job.
    - Bound the VLM chunk fan-out (R6).
    - **Exempt the inference/ingest data plane** (decode → batched Triton) as request-scoped
      parallelism that is awaited before the response, written with `TaskGroup` (R3).
  - *Trade-off:* a strict reading of the rule would push `/ingest` latency through a queue,
    contradicting the performance goals.
- **J8. Migration order.** *Recommend* Phase 1 (fixes, no service) first, then infra, jobs,
  periodic and workers (section 7). Each phase ships on its own.
- **J9. `jobs` concurrency.**
  - *Recommend 4*, tunable with `OP_JOBS_CONCURRENCY`.
  - *Trade-off:* this is lower than today's effective parallelism (one per busy API process).
    Per-project singletons already limit most jobs to one per project.
- **J10. Redis durability.**
  - *Recommend* AOF everysec [V3], no backup.
  - The reaper re-enqueues any `queued` job record older than 2 min with no claim, so losing
    the broker loses no work.
- **J11. Auto-resume after a crash.**
  - *Recommend* auto-resume only for kinds already designed to resume (reprocess, dataset
    import, bulk ingest, combine, assist loop).
  - The others (selection, viz, item scores, probe, region clustering, promote) end
    `interrupted` and need a user restart, as today.
- **J12. API recycling.**
  - *Recommend* keeping `--limit-max-requests` once no job lives in the API, and adding
    `--limit-max-requests-jitter=1000` [U1] so the workers do not recycle in lockstep.
- **J13. Progress transport.**
  - *Recommend* keeping `FileJob` + the JSONL event log in this series.
  - Later, consider OpenTranscribe's Redis pub/sub (`app/utils/websocket_notify.py`) to
    replace the shared JSONL tail, once Redis is in the default install.
  - *Trade-off:* one less file protocol, against rewriting the SSE plumbing.
- **J14. API worker count** (section 4e).
  - *Recommend* one knob, `API_WORKERS` (passed as `WEB_CONCURRENCY`), written by the
    installer as `min(nproc, 32)` and validated by benchmark. Per-process pools are sized from
    it.
  - Use `API_WORKERS=1` plus replicas for orchestrated deployments, which need a multi-host
    job store first (J5).
  - Remove the hard-coded 32, 16 and "64".

---

## 7. Phased migration plan

**Branch:** one branch per phase, off the current integration branch
(`git rev-parse --abbrev-ref @{u}`).

**Git rules for every phase:**
- conventional commits;
- `.venv/bin/pre-commit run --all-files` before each commit;
- `make contracts` in the same commit as any wire change;
- never squash.

**Live checks:**
- Use a separate compose project: `docker compose -p opq-<phase> ...`. Never use the default
  project name.
- Use GPUs 0/2.
- Wait with healthchecks or the Monitor tool. **No `pgrep -f`, `pkill -f` or sleep loops.**

### Phase 1: cross-process correctness fixes (no new service). Ships alone.

- **P1.0 Atomic writes with a unique temp file (D13 / API-4).** This goes first because
  everything else writes job records.
  - Add `src/utils/atomic_write.py` with `atomic_write_json(path, obj)`, using
    `tempfile.NamedTemporaryFile(dir=path.parent, delete=False)` + `os.replace`.
  - Use it in `FileJob.write`, `job_reconcile.reconcile_stale_running`, `holdout.py:334-337`,
    `routers/curation/bakeoff.py:122-125`, `dataset_import/store.py:62-64`,
    `label_ops/store.py:190-192` and `registry.py:84`.
  - Make `FileJob.write` take `_state_lock()`. Add an inner `_write_unlocked` for `update()`:
    flock is per open file description, so a nested open in the same process would block.
  - Convert direct `write()` repairs in `get_state` functions (for example
    `item_scores/job.py:155`, `embedding_viz.py:183-195`) to `repair_if_stale`.
  - Test: `tests/test_file_job_concurrent_write.py`. 4 spawned processes × 3,000 writes to one
    job must give 0 errors. It is red today.
- **P1.1 One lease primitive.**
  - Add `FileJob.claim(active_statuses, *, owner, resumable, max_attempts)` in
    `src/services/curation/file_job.py`, using the existing `exclusive_start_lock`
    (`job_lock.py`).
  - It returns `Claimed | Busy | Terminal`.
  - It records `owner`, `attempt`, `claimed_at` and touches the heartbeat *inside* the lock.
- **P1.2 Unlocked starts (D2).** Route these through a start lock plus a busy check:
  - `autolabel/job.py:454-511`: hold the start lock across the `is_busy()` check, the
    `state.json` write and the `trigger.json` rename. Refuse (409) when `trigger.json`
    exists.
  - `selection/job.py:123-148` and `embedding_viz.py:198-230`: start lock, and touch the
    heartbeat at create.
  - `clustering/region_cluster_jobs.py:158-260, 316-345`:
    - move to `FileJob` (status, heartbeat, cancel) under a start lock;
    - delete the 1800 s `_JOB_STALE_S` flag logic (D12);
    - keep the `job.json` wire shape, or update the contract and Cropwright in the same
      commit.
- **P1.3 Class registry (D3).**
  - In `clients/curation_opensearch/registry.py`, wrap `add_class`, `rename_class`,
    `merge_class` and any other mutator in `exclusive_file_lock(<registry dir>/registry.lock)`
    (`job_lock.py:263`).
  - Re-`load()` from disk *inside* the lock, bypassing the mtime cache: the read must happen
    under the lock.
  - Callers in async code call through `asyncio.to_thread`, because the lock blocks.
  - Delete the unused `_registry_lock` (:313).
  - Check every caller for blocking calls on the event loop (`classes_write.py:49,94`,
    `classes_merge.py:108`, `dataset_import/prepare.py:419`, `combine/execute.py:143`,
    `class_ensure.py:57`).
- **P1.4 OS-1.** Add `op_type='create'` on the absent-doc branch in `config_store/index.py`
  (`save_config`, `activate`, the immutable `<kind>:<name>@<rev>` copy) and
  `curation/keymap.py`. Map `ConflictError` to the existing 409 errors.
- **P1.5 Startup reconcile (D4).**
  - In `job_reconcile.py:354-366`, treat a missing heartbeat as stale only when `state.json`'s
    own mtime is older than `HEARTBEAT_STALE_S`. A sibling's just-written job is then never
    flipped.
  - This code is deleted in Phase 4. It ships now because it is live.
- **P1.6 Worker-container claims (D7/D8).**
  - Auto-label and assist-loop `claim_trigger`: replace read+unlink with
    `os.rename(trigger.json, claimed/<job_id>.json)`. The first rename wins; the loser gets
    `FileNotFoundError`. Write `running` before deleting the claimed file.
    - These are deleted in Phase 5, so skip this item if Phase 5 lands within the same
      release (J8).
  - Trainer `list_pending_jobs`/`process_one` (`docker/trainer/job_protocol.py:102`,
    `trainer.py:267`): claim by `os.open(<id>.claim, O_CREAT|O_EXCL)` before running.
    Delete the claim on a terminal state, and treat a stale heartbeat as reclaimable.
  - API `training/jobs.py:152-187`: write the job file with `O_EXCL` instead of `exists()`
    then write.
  - `export_worker.reconcile_claimed` (`:95`): fail only claimed jobs whose task heartbeat is
    stale.
- **P1.7 D10 in the same series.** OS-2/OS-3: use the shared `async_bulk` helper
  (`src/clients/opensearch/bulk.py`) in the five sites, and fail the job or route on item
  errors.

**Tests (offline, `tests/`; each must fail before its fix):**

- `tests/test_job_claim_cross_process.py`:
  - `multiprocessing` with the `spawn` context;
  - 8 processes call `FileJob.claim` / `reprocess_job.create_job` on the same `tmp_path`
    at a `multiprocessing.Barrier`;
  - assert exactly one `Claimed` and 7 `Busy`.
  - Repeat for `autolabel.start_job` (exactly one trigger, one `job_id` in `state.json`),
    `selection.start_job` and `embedding_viz.start_job` (with `run_*_job` monkeypatched to a
    no-op coroutine run in each process's own loop).
- `tests/test_class_registry_cross_process.py`:
  - 8 spawned processes each `add_class(f'c{i}')` on one registry file at a barrier;
  - assert 8 classes, 8 unique ids, and every returned id maps to its own name.
  - Red today.
- `tests/test_reconcile_sibling_job.py`: state `running`, no heartbeat, state mtime fresh,
  so reconcile leaves it alone. With an old mtime it is reconciled.
- `tests/test_config_store_first_create.py`: the OS-1 test from `opensearch.md` (a stateful
  fake raising `ConflictError` on create of an existing id).
- Trainer claim: two `process_one` calls in two processes on one job file. One runs; the
  other sees the claim.

**Gate:**
- `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'` (targeted files first, the
  full suite once at the end);
- pre-commit;
- `make contracts-check`.

### Phase 2: queue infrastructure (no job moved yet)

Model this phase on OpenTranscribe. Read these files there first, read only:
- `backend/app/core/celery.py`;
- `backend/app/core/constants.py:27-64`;
- `backend/scripts/celery_pool_healthcheck.py`;
- `docker-compose.yml:95-125, 350-455, 868-890`;
- `app/tasks/CLAUDE.md` ("Conventions / patterns").

Copy patterns and rationale comments, not the transcription-specific code.

- **S1 (spike, 1 h). Done when:**
  - Celery 5.6.x with `celery[redis]` connects to the pinned `redis:8.2.x-alpine` image with
    `requirepass` and AOF.
  - An `acks_late` task killed with SIGKILL stays unacked, and is redelivered after a
    *lowered* test visibility timeout.
  - A cold shutdown during an `acks_late` task acks it, which reproduces the OpenTranscribe
    finding; the reaper design depends on it.
  - `revoke(task_id=job_id)` of a queued task skips it.
  - Record the versions in the PR.
- **Dependencies:**
  - Add `celery[redis]==5.6.*` (pin exact, matching OpenTranscribe's pin where possible) to
    `requirements.txt` and `requirements-test.txt`.
  - Check licences: Celery/kombu/billiard BSD-3, redis-py MIT, Redis 8 under its AGPLv3
    option [V1].
  - Add the rows to `ATTRIBUTION.md`.
  - Trivy-scan the new images.
- **New files:**
  - `src/tasks/__init__.py`;
  - `src/tasks/queues.py`: the `TaskQueues` constants class with `ALL`, mirroring
    `CeleryQueues`;
  - `src/tasks/app.py`: the config from section 5, rule 4, plus a startup check that every
    registered task has a route, mirroring `_validate_task_routes`;
  - `src/tasks/job_task.py`: the wrapper;
  - `src/clients/task_queue.py`: `enqueue(task_name, *, project, job_id)`, `revoke(job_id)`
    and `ping`. Routing comes only from `task_routes`; there is no queue argument.
  - `src/config/queue.py`: `OP_QUEUE_URL`, `OP_QUEUE_PASSWORD_FILE`, `OP_JOBS_CONCURRENCY`,
    validated at startup and failing closed when malformed.
  - `scripts/healthcheck/celery_pool_healthcheck.py`: a port of OpenTranscribe's script,
    with attribution in the header.
- **Layering:** register the `tasks` layer in the layering ratchet. `tasks` may import
  services, clients, config and domain; nothing imports `tasks`.
- **Compose (`docker-compose.yml`):**
  - **`redis`:** see section 5, rule 8.
  - **`jobs-worker`:**
    - api image;
    - command `celery -A src.tasks.app worker -Q jobs,periodic --concurrency=${OP_JOBS_CONCURRENCY:-4} --max-tasks-per-child=50 --hostname jobs@%h -E`;
    - the same volumes and env as `api` (plus `OP_API_LOOPBACK_URL=http://api:8000`);
    - the `API_GPU_ID` reservation;
    - `stop_grace_period: ${OP_STOP_GRACE_JOBS:-60}s`;
    - healthcheck: the pool healthcheck script against `jobs@$$HOSTNAME`.
  - **`open-vocab-worker`:** the same image, `-Q open_vocab -c 1 --hostname open-vocab@%h -E`
    (D6).
  - **`jobs-beat`:**
    - `celery -A src.tasks.app beat --schedule=/var/lib/openprocessor/celerybeat-schedule`;
    - a fixed `container_name`;
    - healthcheck: schedule-file mtime under 300 s (OpenTranscribe `docker-compose.yml:876-890`).
  - **`depends_on: redis: service_healthy`** for `api`, the workers and `jobs-beat`.
  - **Flower** under the `monitoring` profile, internal only.
- **Dev overlay:** source mounts for the new services in `docker-compose.dev.yml`. Celery
  does not hot-reload, so document `docker compose restart jobs-worker`.
- **Installer and CLI:**
  - `setup-openprocessor.sh` and `scripts/setup.sh` generate `secrets/queue/redis.conf` once
    (mode 600, `requirepass <random 32 bytes>`).
  - `env.template` gets the "Curation quick-config" keys.
  - `openprocessor status/health` lists `redis`, `jobs-worker`, `open-vocab-worker` and
    `jobs-beat`.
  - `/health` reports `queue: ok|down`: degraded, not fatal for inference routes.
- **Docs:**
  - `docs/ARCHITECTURE.md` (services table, "Workers, jobs and events", and a queues table
    like OpenTranscribe's `docs/ARCHITECTURE.md:69-102`);
  - `CLAUDE.md` services table and a short Celery conventions block (routes, no decorator
    `queue=`, no I/O in `worker_process_init`, no `terminate=True`);
  - `INSTALLATION.md`;
  - `docs/CURATION.md`;
  - `SECURITY.md` (broker preconditions);
  - `docs-site`.
  - Run `.venv/bin/python scripts/docs/check_docs_vs_code.py`.
- **Tests:**
  - `tests/test_job_task_wrapper.py`, unit level with mocks, as the Celery docs advise [C9]:
    - claims and runs once;
    - a live lease means a no-op ack;
    - a terminal state means a no-op;
    - `attempt` reaching 3 means `failed`;
    - an exception means `failed`;
    - a cancel flag means `cancelled`;
    - the heartbeat thread keeps ticking while the handler blocks past `HEARTBEAT_STALE_S`
      (monkeypatched short value).
  - `tests/test_task_routes_complete.py`: every registered task has a `task_routes` entry,
    and no task decorator passes `queue=` (OpenTranscribe `tests/unit/test_task_routes_complete.py`).
  - `tests/test_queue_consumers_from_compose.py`: parse `docker-compose.yml` without
    profiles; every queue in `TaskQueues.ALL` that a route targets has a `-Q` consumer
    (OpenTranscribe `tests/unit/test_lite_mode_queue_consumers.py`).
  - `tests/test_celery_config.py`:
    - `visibility_timeout` > `task_time_limit` > `task_soft_time_limit`;
    - `prefetch` is 1;
    - `accept_content` is `['json']`;
    - `task_create_missing_queues` is False.
  - `tests/live/test_queue_infra.py`: in the live harness (`docker/test/compose.yml` gains
    redis and jobs-worker), enqueue a no-op task and see it run.

### Phase 3: move the API-hosted jobs (2a), one family per commit

For each family:
1. Replace `asyncio.create_task(run_x(...))` with `task_queue.enqueue('op.jobs.<kind>', project=slug, job_id=id)`. The queue comes from `task_routes`.
   after the record is written under the start lock.
2. Add `@app.task(name='op.jobs.<kind>', base=JobTask)` in `src/tasks/jobs.py`. It calls the
   *existing* `run_x(job, ctx)` coroutine, which needs its OpenSearch client and service
   built from `ctx`, not from the request.
3. Delete the module's `_tasks` / `_active_tasks` dict, its `reconcile_orphaned_jobs`, and its
   entry in `startup_reconcile.JOB_MODULES`.
4. Change cancel: if the record is `queued`, call `task_queue.revoke(job_id)` and write
   `cancelled`. If it is `running`, touch `cancel.flag` (unchanged).
5. Write a status that a status route can report while queued.

| Order | Kind (task name) | Files to change | Notes |
|---|---|---|---|
| 1 | `reprocess` | `reprocess_job.py:170-182`, its router | already split into `create_job`/`run_job`: the template family |
| 2 | `assist_run` | `assist/runner.py:71-90`, `assist/job.py` | `_segment_stage` waits on the region worker: fine in a worker |
| 3 | `item_scores`, `probe`, `selection`, `embedding_viz` | the four job modules | selection: run `k_center_greedy` in the worker (no loop concern any more); add `queued` status, contracts |
| 4 | `region_cluster`, `region_fp_centroids` | `region_cluster_jobs.py` | after P1.2 they are FileJobs |
| 5 | `promote` | `training/promote_job.py:168-210` | `run_detached` no longer needed: delete it |
| 6 | `dataset_import`, `dataset_import_undo`, `dataset_import_resume` | `dataset_import/runner.py:398,414`, `routers/curation/datasets.py:430-480` (move the resume rescan into the task), `dataset_import/upload.py:240-268` (extraction becomes a `dataset_upload_extract` job; R1) | the upload route stores the archive and enqueues |
| 7 | `combine` | `projects/combine/service.py:165-172`, `execute.py` | fix OS-4 in the same commit (partial `update` with checked errors) |
| 8 | `bulk_ingest_run` | `bulk_ingest/runner.py:490-520`, `source_runs.py:36` | the loopback URL must be `http://api:8000` (`OP_API_LOOPBACK_URL` in jobs-worker env); delete `run_coroutine_threadsafe` plumbing |
| 9 | `open_vocab_drain` (queue `open_vocab`) | `open_vocab_ingest.py:63-104` | delete `_PASS_LOCK`, `_tasks` and `wait_for_scheduled`; enqueue one task per ingest request with its image ids (D6) |
| 10 | `project_delete_finish` | `routers/curation/projects.py:296-336` | delete `_BACKGROUND_DELETE_TASKS`; dedupe by `task_id=f'delete:{slug}'` plus the record OCC |
| 11 | `directory_ingest` | `routers/ingest/directory.py:115-190` | the route returns 202 and a job id (wire change and contracts) |

- **Not moved:** the `crop_orders` backfill. It is an idempotent cache fill. Keep it in
  process with its dedupe, but wrap it with the asyncio-audit `spawn` helper (R5).
- **Tests per family**, red against the pre-change code where possible:
  - Unit: the route enqueues exactly once with `task_id=job_id` (mock `task_queue`). A
    broker-down `enqueue` raises → 503, and the record is `failed`.
  - `tests/live/test_jobs_cross_process.py` (live harness, two `jobs-worker` containers
    `-p opq-p3 --scale` is impossible with `container_name`, so the harness compose defines
    `jobs-worker-a` and `jobs-worker-b`):
    - **Race:** `celery send_task` the same `reprocess` `job_id` twice (two messages, distinct
      `task_id` suffix) → exactly one run in `state.json` history, and the other worker logs
      `lease_busy`.
    - **Kill mid-job:** start a reprocess of 2,000 images and `docker kill jobs-worker-a`
      after progress > 0. Within `HEARTBEAT_STALE_S` + one reaper tick, the state goes
      `interrupted`, then (resumable, J11) `running` on `jobs-worker-b` with `attempt=2`,
      and it finishes. The image count equals the total, with no duplicate writes (OCC
      no-ops).
    - **Cancel during a run:** POST cancel during progress. The state is `cancelled` within
      one progress step and no further writes follow.
    - **Cancel while queued:** stop both workers, start a job, cancel it → `cancelled`.
      Start the workers: the job never runs (revoke or lease terminal check).
    - **API recycle does not matter:** run the API with `--limit-max-requests=50`, drive 500
      requests during a reprocess, and the job completes.
  - Each family has a quick live smoke: start, status shows progress, completes.

### Phase 4: periodic work to beat (2b) and the reaper

`src/tasks/periodic.py` with `beat_schedule`:

| Task | Every | Replaces (delete in the same commit) |
|---|---|---|
| `op.periodic.ingest_dispatch` | `OP_INGEST_DISPATCH_S` (5 s) | `bulk_ingest/dispatcher.py` `_loop`/`start_ingest_dispatcher` (:251-263); keep `run_project_tick` and its flock (a harmless second guard) |
| `op.periodic.open_vocab_sweep` | `OP_OPEN_VOCAB_SWEEP_S` | `open_vocab_sweeper.py:111-125`; the sweep enqueues `open_vocab_drain` |
| `op.periodic.gpu_arbiter` | 15 s | `lifespan.py:145-213` loop (the startup reconcile also moves here; the arbiter overlay mounts the Docker socket into `jobs-worker` instead of `api`, so update `docker-compose.gpu-arbiter.yml` and its header) |
| `op.periodic.reload_promoted` | 15 s | the reload inside the arbiter loop and `lifespan.py:216-228` (keep the on-demand route) |
| `op.periodic.lease_reaper` | 30 s | `startup_reconcile.py`, every module's `reconcile_orphaned_jobs`, `lifespan.py:135-142`. For each project and job dir: a stale `running` lease becomes `interrupted` (+ re-enqueue if resumable, J11); `queued` older than 120 s with no live message is re-enqueued (`task_id` = job id, so a duplicate is a lease no-op) |
| `op.periodic.cluster_refresh` | `OP_CLUSTER_REFRESH_INTERVAL_S` | `scripts/curation/cluster_refresh_daemon.py` main loop and the `curation-cluster-refresh` service; move the schedule logic (`clustering/refresh_schedule.py`) into the task |
| `op.periodic.ivf_retrain_check` | `AUTO_RETRAIN_CHECK_INTERVAL_S` | `auto_label_worker.py` `_next_retrain` (:398-420) |

- **Delete:** `background_jobs.py` (start and stop), `AppResources.curation_background_jobs`,
  `arbiter_task`, `ops_metrics_task`, and the matching shutdown steps.
- **Keep:** the registry poll, config poll, event tail and ops-metrics refresh in the API.
  - The ops-metrics refresh is already elected by `_claim_turn`. Its gauges live in the
    API's Prometheus multiprocess registry.
  - Add the `mark_process_dead` cleanup for API-14 in the API-audit series.
- **Tests:**
  - Unit: each periodic task body runs once against fakes.
  - Unit: the schedule has exactly the names above.
  - Compose check (`tests/test_compose_services.py`): `jobs-beat` has a `container_name` and
    no `deploy.replicas`.
  - Live: kill `jobs-worker` mid-reprocess with no API restart. The reaper (not a lifespan)
    marks it `interrupted`.
  - Live: run with the arbiter overlay. Training start stops the configured containers once;
    the Docker events log shows one stop per container, not 32.

### Phase 5: trigger-file workers to queues

- **Auto-label.**
  - `autolabel/job.py:454-511`: enqueue `op.autolabel.run` with `task_id=job_id` instead of
    writing `trigger.json`.
  - The `curation-auto-label-worker` command becomes
    `celery -A src.tasks.app worker -Q autolabel -c 1`.
  - Move `_run_one` (`auto_label_worker.py:202-297`) into the task.
  - Delete the poll loop, `_claim_trigger_for`, `_oldest_pending_trigger`, `running.lock`,
    the trigger file and `QUEUE_MAX_AGE_S` handling (the reaper covers `queued`).
- **Assist loop.** Do the same with `op.assist_loop.run` and queue `assist_loop`. Delete
  `claim_trigger`, the trigger file and the worker poll loop. Resume enqueues again.
- **Liveness.** Keep the `worker_liveness` heartbeats, or replace the compose healthchecks
  with `celery inspect ping`. Pick one and delete the other.
- **Tests:**
  - Two auto-label workers (harness only) + one job → one run.
  - Kill mid-stage → `interrupted`, then manual restart (not resumable, J11).
  - Cancel at a stage boundary.

### Phase 6 (optional): evaluator; request fan-out (J7)

- **Evaluator (J6).** In the evaluator image, add celery and run `-Q gpu_eval -c 1`. The API
  enqueues `op.gpu_eval.bakeoff` instead of dropping `*.job.json`. Update `bakeoff_active`
  in `gpu_arbiter.py:91` to read the job records.
- **`review_resolve` and `edit_undo`.**
  - One `mget`, then one bulk of `update` actions with `if_seq_no`/`if_primary_term` per
    item [O1] through the OCC helpers' merge.
  - A 409 per item becomes a `conflicts` entry. A batch above `OP_BATCH_SYNC_MAX` returns 202
    with a job.
  - Tests: per-item conflict handling, and a batch over the limit gives 202.
- **VLM chunk fan-out:** bound it with a semaphore (R6).
- **Ingest data plane:** `TaskGroup` (R3), and document the exemption in
  `docs/design/engineering_standards.md`.

### Phase 7: API worker count as a tuning knob (section 4e, J14)

Do this after Phases 3-4 and the API-audit fixes for API-1, 7, 8, 12, 13 and 14.

- **`docker-compose.yml`:**
  - drop `--workers=32` and the 64/benchmark comments;
  - add `WEB_CONCURRENCY=${API_WORKERS:-8}` to `api` environment [U1].
- **`Dockerfile:309-319`:** drop `"--workers", "16"`. The image default is uvicorn's
  `WEB_CONCURRENCY`, else 1, which is what an orchestrator wants [F3].
- **Installer and `env.template`:** `API_WORKERS`, defaulting to `min(nproc, 32)`, written
  once and documented in INSTALLATION with the orchestration note.
- **`src/app/lifespan.py:57-74`:** size the shared executor and the Triton pool from the
  totals divided by `API_WORKERS` (new config in `src/config/`).
- **Ratchet** (`scripts/codegen/`): no `asyncio.create_task` outside the spawn helper's
  allowlist; no new module-level lock, semaphore or mutable-dict state without an
  allowlisted reason.
- **Tests:**
  - an offline test that the compose `api` command has no `--workers`;
  - a live diff test running the same scripted scenario with `API_WORKERS=1` and
    `API_WORKERS=32`;
  - ingest and search benchmarks at the installer default against the 32-worker baseline in
    `docs/PERFORMANCE.md`.

**Release gate (once, at the end of the series):**
- the full offline suite;
- the live harness;
- `make contracts-check`;
- the docs check;
- Trivy on the new images;
- an image-vs-git audit before any push;
- Cropwright `npm run check` + `npm run build`;
- a browser check of the jobs UI (progress, cancel, queued state) at desktop and 430 px.

---

## 8. Risks and edge-case register

| # | Risk / edge case | Mitigation |
|---|---|---|
| E1 | Redelivery after the visibility timeout while a job is still running (job > 2 h) [C1] | the lease is live, so the redelivered message is a no-op ack; the heartbeat thread keeps the lease fresh |
| E2 | A poison message that kills the worker (OOM) loops with `reject_on_worker_lost` | `attempt` cap 3 in the lease, then `failed` with the last error; `worker_max_tasks_per_child` |
| E3 | Broker down at enqueue | 503 `queue_unavailable`; record `failed`; nothing half-started (fail closed) |
| E4 | Broker data lost (volume wiped) | the reaper re-enqueues `queued` records older than 120 s; running jobs keep their lease |
| E5 | Beat duplicated (two beats) [C4] | fixed `container_name`; compose test; every periodic body is idempotent and guarded by the same start locks |
| E6 | Revoke lost on worker restart (in-memory) [C3] | the cancel truth is the record's `cancelled`; the wrapper refuses terminal records |
| E7 | `terminate=True` temptation for a hard cancel [C3] | forbidden in code (add a ratchet: no `terminate=` in `src/`); cooperative `cancel.flag` only |
| E8 | Docker stop of jobs-worker during a long job (warm shutdown waits forever) [C8] | `worker_soft_shutdown_timeout`, then cold; the message is requeued (`acks_late`) and the lease becomes stale, so it resumes or ends `interrupted` |
| E9 | asyncio code in Celery prefork children [S1] | `asyncio.run` per task in the wrapper; never share an event loop or `AsyncOpenSearch` across tasks; build clients per task |
| E10 | CUDA and fork | the parent worker never initialises CUDA (no model import at module scope in `src/tasks`); GPU work goes through Triton/segmenter, or happens inside the child |
| E11 | Project archived or deleted while its job is queued | the wrapper binds the project; an unbound or archived project means `failed: project unavailable` (fail closed) |
| E12 | Job record written but enqueue not attempted (API crash in between) | the reaper re-enqueues `queued` after 120 s |
| E13 | Two projects' jobs starve each other on `jobs` | per-project singletons + concurrency 4; FIFO; revisit with per-project routing only if measured |
| E14 | Wire change: new `queued` status for selection, viz, item scores, probe, region clustering | `make contracts`; Cropwright handles `queued` (render as waiting); browser check |
| E15 | `flock` semantics on non-local volumes | single-host local volumes only; document it in INSTALLATION (NFS-backed `/jobs` is unsupported) |
| E16 | The class-registry lock blocks the event loop | callers use `asyncio.to_thread`; the lock is held only for the read-modify-write |
| E17 | Redis reachable from the LAN | no `ports:` in compose; required `requirepass` from a 600-mode file; JSON-only content [C6][V4] |
| E18 | A route into a queue nothing consumes (OpenTranscribe's lite incident) | `task_create_missing_queues=False` + the compose consumer test (Phase 2) |
| E23 | Wedged prefork pool reports healthy via `inspect ping` (OpenTranscribe #631) | port the pool healthcheck; `worker_proc_alive_timeout=30`; no I/O in `worker_process_init`; note that nothing auto-restarts `unhealthy` |
| E24 | Half-open broker socket stalls a consumer forever (OpenTranscribe #1144) | the keepalive/socket-timeout transport options, copied with their rationale |
| E25 | Cold shutdown acks a running `acks_late` task, so its message is gone | the reaper recovers from the job record, not the broker (Phase 4); S1 verifies it |
| E26 | API behaviour differs with `API_WORKERS=1` vs 32 | the statelessness rules and diff test in section 4e; the ratchet on module-level state |
| E19 | Metrics from Celery workers (prefork children recycled by `worker_max_tasks_per_child`) | the worker gets its own metrics dir/endpoint, never the API's `PROMETHEUS_MULTIPROC_DIR`; apply the API-14 `mark_process_dead` hook there too |
| E22 | Job runner (jobs worker) and status poll (API) write one record from two processes | P1.0 unique temp file + `FileJob.write` under `state.lock` (D13); the lease claim and repairs re-check under the lock |
| E20 | Combine/import jobs that touch two projects | the wrapper binds the *job's* project; combine already binds sources in turn inside `execute.py`; keep that |
| E21 | Dev hot-reload | `celery worker` does not auto-reload; the dev overlay documents `docker compose restart jobs-worker` after task code changes |

---

## 9. Sources (fetched 2026-11-03)

**Celery**
- [C0] Celery introduction, 5.6.3 docs: https://docs.celeryq.dev/en/stable/getting-started/introduction.html
- [C1] Celery, Using Redis (visibility timeout, eviction): https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html
- [C2] Celery Tasks (acks_late, reject_on_worker_lost, idempotence, retries, update_state): https://docs.celeryq.dev/en/stable/userguide/tasks.html
- [C3] Celery Workers (revoke, terminate, statedb, max_tasks_per_child): https://docs.celeryq.dev/en/stable/userguide/workers.html
- [C4] Celery Periodic Tasks (single scheduler, schedule file): https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html
- [C5] Celery Routing (task_routes, -Q, Redis priorities): https://docs.celeryq.dev/en/stable/userguide/routing.html
- [C6] Celery Security (broker trust, JSON default, pickle, accept_content): https://docs.celeryq.dev/en/stable/userguide/security.html
- [C7] Celery Optimizing (prefetch multiplier 1 + acks_late for long tasks): https://docs.celeryq.dev/en/stable/userguide/optimizing.html
- [C8] Celery Workers, stopping (warm, cold, soft shutdown, SIGKILL): https://docs.celeryq.dev/en/stable/userguide/workers.html
- [C9] Celery Testing (eager not for tests, celery_worker fixture): https://docs.celeryq.dev/en/stable/userguide/testing.html
- [S1] Async tasks are not native in Celery 5.x; deferred to 6.0: https://github.com/celery/celery/issues/3884 , https://github.com/celery/celery/issues/6603

**Other queue libraries**
- [D0] Dramatiq motivation (acks on completion; comparison, dated 2019): https://dramatiq.io/motivation.html
- [D1] Dramatiq RedisBroker `heartbeat_timeout` 60 s (requeue mechanics not stated on the page; verify if J1 picks Dramatiq): https://dramatiq.io/reference.html
- [D2] Dramatiq TimeLimit (10 min default), Retries defaults, AsyncIO middleware: https://dramatiq.io/reference.html
- [D3] Dramatiq cookbook (dramatiq-abort, ConcurrentRateLimiter): https://dramatiq.io/cookbook.html
- [Q1] RQ workers (fork per job, timeouts, stop/kill job commands): https://python-rq.org/docs/workers/
- [Q2] RQ cron (beta; multi-instance safety undocumented): https://python-rq.org/docs/cron/
- [A1] arq docs (job_id uniqueness, pessimistic execution, abort, cron): https://arq-docs.helpmanual.io/
- [A2] arq README, "In maintenance only mode": https://github.com/python-arq/arq
- [H1] Huey (storage: redis/valkey, postgres, sqlite, file, memory; periodic, retries, locking, priorities): https://huey.readthedocs.io/en/latest/
- [T1] Temporal self-hosted deployment (server, UI, PostgreSQL, Elasticsearch): https://docs.temporal.io/self-hosted-guide/deployment

**Brokers**
- [R1] RabbitMQ consumers, delivery acknowledgement timeout (30 min default; quorum-only from 4.3): https://www.rabbitmq.com/docs/consumers
- [V1] Redis licences (RSALv2 / SSPLv1 / AGPLv3 from Redis 8): https://redis.io/legal/licenses/
- [V2] Valkey (BSD, Linux Foundation): https://valkey.io/
- [V3] Redis persistence (AOF everysec default, about 1 s loss): https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/
- [V4] Redis security (trusted environments, protected mode, requirepass/ACL): https://redis.io/docs/latest/operate/oss_and_stack/management/security/

**OpenSearch**
- [O1] OpenSearch Index document API (if_seq_no / if_primary_term → 409; op_type=create): https://docs.opensearch.org/latest/api-reference/document-apis/index-document/
- [O2] OpenSearch Job Scheduler locks (lock document with seq_no/primary_term before running a job): https://docs.opensearch.org/latest/monitoring-your-cluster/job-scheduler/locks/ , https://github.com/opensearch-project/job-scheduler

**Web framework and server**
- [F1] FastAPI Background Tasks, caveat recommending Celery for heavy work: https://fastapi.tiangolo.com/tutorial/background-tasks/
- [F2] FastAPI Server Workers (`--workers`; "on Kubernetes ... a single Uvicorn process per container"): https://fastapi.tiangolo.com/deployment/server-workers/
- [F3] FastAPI in Containers ("Replication - Number of Processes", "Containers with Multiple Processes and Special Cases": Docker Compose on a single server): https://fastapi.tiangolo.com/deployment/docker/
- [OT] OpenTranscribe (read only, local): `<OpenTranscribe checkout>`, files as cited in section 4d
- [U1] Uvicorn settings (--workers, --limit-max-requests, --limit-max-requests-jitter, --timeout-graceful-shutdown): https://github.com/encode/uvicorn/blob/master/docs/settings.md
