# B2 overview: jobs off the API on Celery

| Field | Value |
|---|---|
| Status | plan, not started |
| Release | part of the single v0.7.0 release (no intermediate tag; R-final in `06_release_acceptance.md`) |
| Depends on | B1 merged into main (`02_*` files: the J-LEASE lease, its thread heartbeat, the trainer claims, the file-lock ratchet) |
| Decisions applied | D-8 J-LEASE (approved: the task wrapper transfers the API-acquired lease and reuses B1's heartbeat thread), D-13 order B0, B1, B2, B3, B4, B5, R-final last (approved; D-37 = yes), D-17 Celery 5.6 (approved), owner decisions of 2026-11-03 (plan-only round, no intermediate tags, branch model, owner rules); the register is [00_index.md](00_index.md) section 6, the test and GPU budget is section 8.1 |
| Open decisions | D-18..D-26: status in [00_index.md](00_index.md) section 6; section 12 here holds only the design detail behind each row |
| Source IDs | `task_queue_plan.md` Phases 2-5, J1-J12, D1-D15, E1-E26, section 4d, section 5; audit CC-4, CC-5 (upload, directory ingest), CC-9, CC-11 (last site), OPS-2, SEC-4, SEC-5; the former master plan (rev 4, in git history) sections 3a (last row), 3b (B2 rows), 3c, 7.2, 12 (T2, T4), 15 (M9-M11), 16 (B2 row), Appendix A (A.4 #19-21, A.5 #3, A.6, A.7 #16, C-6) |
| Files of this batch | this file (shared contract, inventory, topology, config, gate), `03a_b2_w1_queue_infra.md`, `03b_b2_w2_job_families.md`, `03c_b2_w3_beat_reaper_workers.md` |

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read only, with `rg`). B1 is not
implemented on that branch, so every B1 artifact named here is a precondition, checked with the
`rg` line in section 2.
Consolidated 2026-11-03.

Naming: "W1/W2/W3" in this file are the three **internal stages** of B2 (stage W1 = queue
infrastructure, and so on), not the program waves of [00_index.md](00_index.md) section 4 (B2 is
program wave W2).

---

## 1. Goal and done-when

**Goal.** Every job a request starts, every periodic loop and the two trigger-file workers run in
Celery workers, not in the uvicorn request processes. The job record (`FileJob` files) and the
OpenSearch lease (B1) stay the truth. The broker (Redis 8) is transport only.

**Why (mission).** Jobs die when an API worker recycles (CC-4, reproduced). Periodic side effects
run once per API worker, so 32 times (Q-D5). Claims hold only with one worker container (Q-D7,
Q-D8). Heavy work competes with request latency in the same event loop.

**B2 is done when:**
- no job family runs as an asyncio task inside an API worker (CC-4 done-when). Check:
  `rg -n "create_task\(" src/services src/routers` lists only the allowlisted per-process
  helpers of section 5.4;
- the API lifespan creates only per-process resources, caches and pollers (section 5.5);
- `startup_reconcile.py`, every `reconcile_orphaned_jobs`, `background_jobs.py` and the
  lifespan arbiter loop are deleted;
- the file-lock ratchet lists only `event_hub.py` and `job_lock.py`
  (`rg -n "exclusive_(file|start)_lock\(|fcntl\.flock" src | rg -v "job_lock.py|event_hub.py"`
  is empty);
- `tests/test_no_module_level_async_locks.py` has an empty `ALLOWED` set;
- the B2 sweep (section 14) is green, including the single live-harness run, which covers B1 and B2
  together ([00_index.md](00_index.md) section 8.1);
- the PR is merged with a merge commit. No tag (owner decision 2; D-37).
- the closing PR moves the section 17.1 keep-list to its target docs and deletes the B2 plan files
  (`03_*` through `03c`) and their rows in `00_index.md` and `docs/design/README.md`
  ([00_index.md](00_index.md) "Plan lifecycle").

---

## 2. Preconditions

| Check | Command (read only) | Expect |
|---|---|---|
| B1 merged into main | `git log --oneline main \| rg -i "b1\|lease"` | the B1 merge commit |
| lease client | `rg -n "def (acquire\|renew\|fence\|transfer\|release\|takeover_expired)" src/clients/job_lease.py` | six functions |
| lease rules | `rg -n "LEASE_TTL_S\|HEARTBEAT_TICK_S" src/domain/job_lease.py src/services/curation/lease_heartbeat.py` | `LEASE_TTL_S = 120`, `HEARTBEAT_TICK_S = 15` |
| heartbeat thread | `rg -n "class .*Heartbeat" src/services/curation/lease_heartbeat.py` | one class; the asyncio `heartbeat_ticker` is gone |
| file-lock ratchet | `ls scripts/codegen/check_file_locks.py scripts/codegen/baselines/` | the script and its baseline |
| start locks converted | `rg -n "exclusive_(file\|start)_lock\(" src \| rg -v job_lock.py` | only `bulk_ingest/dispatcher.py`, `open_vocab_sweeper.py`, `source_image_cache.py`, `event_hub.py` (the B1 done-when) |
| OPS-2 evidence | `03a` task W1-0 | recorded before 2D starts |
| decisions | section 12 | D-18..D-26 answered (the "needed by" column says which can wait) |
| host | `13`-style rules in `11_risks_cleanup_host.md` | peers told; no GPU needed for the live run; B3's GPU window not running |

If B1 changed a name above, re-find it by the concept (lease acquire, heartbeat thread) and record
the new name in the W1 PR. Do not rebuild a B1 artifact.

---

## 3. Branch model, waves and agents

- Integration branch **`int/b2-celery`**, cut from `main` after B1 merged.
- Waves W1, W2, W3. Each track is one agent in its own worktree `<repo>-wt-b2-<track>` on branch
  `b2/<track>`, cut from `int/b2-celery` at the start of its wave (after the previous wave merged).
- Lanes run only their targeted red/green tests and static checks on touched files
  (`.venv/bin/pre-commit run --files <touched files>`, `ruff`, `mypy` on those files). **No full
  suite, no `--all-files` in a lane.** One commit per fix or concern, hooks on, never
  `--no-verify`, never push, never merge, never check out another branch.
- The coordinator merges finished lanes into `int/b2-celery` with `--no-ff`, one merge commit per
  track, in the order below. After each stage merge it runs targeted tests and static checks only
  (T0 tier, [00_index.md](00_index.md) section 8.1), and **one** full offline sweep plus the single
  live run for the whole batch after stage W3 (section 14).
  A failing track reverts alone with `git revert -m 1 <merge>`.
- Opus plans, Sonnet implements and reviews.
- Every brief forbids `pgrep -f`, `pkill -f` and sleep/until loops, and names its compose project
  (`-p`), its GPU, its worktree and its deadline.

| Wave | Track (branch) | Worktree | Scope file | Merge order | Size |
|---|---|---|---|---|---|
| W1 | `b2/queue-infra` | `<repo>-wt-b2-queue-infra` | `03a_b2_w1_queue_infra.md` | 1 | L |
| W2 | `b2/families-a` | `<repo>-wt-b2-families-a` | `03b`, track 2A | 2 | M |
| W2 | `b2/families-b` | `<repo>-wt-b2-families-b` | `03b`, track 2B | 3 | M |
| W2 | `b2/families-c` | `<repo>-wt-b2-families-c` | `03b`, track 2C | 4 | M |
| W2 (after the contracts settle) | `b2/frontend-queued` | `<repo>-wt-b2-frontend-queued` | `03b`, track 2G | 5 | S |
| W3 | `b2/beat-reaper` | `<repo>-wt-b2-beat-reaper` | `03c`, track 2D | 6 | M |
| W3 | `b2/trigger-workers` | `<repo>-wt-b2-trigger-workers` | `03c`, track 2E | 7 | M |
| W3 | `b2/hardening` | `<repo>-wt-b2-hardening` | `03c`, track 2F | 8 (last: it edits every service block) | S |

After each W2 merge the coordinator runs `make contracts-check` on the integration branch
(contracts drift is caught per merge, M19). Overlap rule (global standing rule): while the B2
sweep runs, the next batch's lanes may already be coding.

---

## 4. Architecture (after B2)

```
request -> api (uvicorn processes; no job runs here)
             | acquire lease (state=queued, queue-wait TTL) -> write FileJob record status=queued
             | task_queue.enqueue('op.jobs.<kind>', project, job_id)  (task_id = job_id)
             | set the queued marker  op:queued:<job_id>  (Redis key with a TTL)
             v
         redis 8 (internal network, no port, required password, AOF everysec, noeviction)
             |
   +---------+-------------+---------------+------------------+-------------------+
   v                       v               v                  v                   v
 jobs-worker          periodic-worker  open-vocab-worker  auto-label-worker  assist-loop-worker
 -Q jobs -c 4         -Q periodic -c 2 -Q open_vocab -c 1 -Q autolabel -c 1  -Q assist_loop -c 1
 (core)               (core)           (core)             (curation profile) (curation profile)
   | JobTask wrapper: bind project -> transfer lease (state=running, TTL 120 s)
   |   -> B1 heartbeat thread -> asyncio.run(handler) -> fence + terminal write -> release lease
   v
 FileJob record (status, progress, cancel.flag, result) -> existing status routes + SSE

 jobs-beat (exactly one): ingest dispatch, open-vocab sweep, cache prune, GPU arbiter,
   promoted reload, lease reaper, cluster refresh, IVF retrain check  -> queue `periodic`

 unchanged: detection worker, VLM worker (item state machine + OCC), trainer (file protocol,
   B1 create-only claims), model-export worker (rename claim), evaluator (D-31, B4)
```

`periodic` has its own consumer (`periodic-worker`). This corrects the task plan, which put
`periodic` on `jobs-worker`: with `-c 4` and four long jobs running, the reaper and the dispatch
tick would wait behind them (correction K-6).

---

## 5. Inventory (verified on the branch)

Greps re-run on `fix/411-r2-round` at `af62c947` (the task plan used `5b95591f`):
- jobs started by requests: `rg -n "create_task\(" src` and `rg -n "exclusive_start_lock\(" src`;
- lifespan loops: `rg -n "create_task|start_|reconcile" src/app/lifespan.py src/services/curation/background_jobs.py`;
- worker containers: `rg -n "^  [a-z0-9-]+:$" docker-compose.yml`;
- request fan-outs: `rg -n "asyncio\.gather|TaskGroup\(" src/routers src/services`;
- per-process state: `rg -n "^_[A-Za-z_]+\s*(:\s*[^=]+)?=\s*(\{\}|dict\(|set\(|asyncio\.(Lock|Semaphore))" src`.

### 5.1 Request-started jobs (today an asyncio task in an API worker)

"Auto-resume" is the D-25 recommendation: after a crash the reaper re-enqueues the job; the
others end `interrupted` for a manual restart. "New `queued` wire" means the status model gains
`queued` (contracts + Cropwright, Q-E14). Lease kinds are B1's table (former master plan 6.1).

| # | Family (task name) | Start site today (`rg` pattern) | Lease kind (scope) | Queue | Wave / track | Auto-resume | New `queued` wire |
|---|---|---|---|---|---|---|---|
| 1 | reprocess (`op.jobs.reprocess`) | `rg -n "create_task\(run_job" src/services/curation/reprocess_job.py` | `reprocess` | `jobs` | W2 / 2A | yes | no (has it) |
| 2 | assist run (`op.jobs.assist_run`) | `rg -n "create_task" src/services/curation/assist/runner.py` | `assist_run` | `jobs` | W2 / 2A | no | check |
| 3 | item scores (`op.jobs.item_scores`) | `rg -n "_active_tasks\[" src/services/curation/item_scores/job.py` | `item_scores` | `jobs` | W2 / 2A | no | yes |
| 4 | probe (`op.jobs.probe`) | `rg -n "_active_tasks\[" src/services/curation/probe_job.py` | `probe` | `jobs` | W2 / 2A | no | yes |
| 5 | selection, k-center (`op.jobs.selection`) | `rg -n "_active_tasks\[" src/services/curation/selection/job.py` | `selection` | `jobs` | W2 / 2A | no | yes |
| 6 | embedding viz, UMAP (`op.jobs.embedding_viz`) | `rg -n "_active_tasks\[" src/services/curation/embedding_viz.py` | `embedding_viz` | `jobs` | W2 / 2A | no | yes |
| 7 | region clustering (`op.jobs.region_cluster`) | `rg -n "create_task\(_run" src/services/curation/clustering/region_cluster_jobs.py` (first site) | `region_cluster` | `jobs` | W2 / 2B | no | yes |
| 8 | region FP centroids (`op.jobs.region_fp_centroids`) | same file, second site | `region_fp_centroids` | `jobs` | W2 / 2B | no | yes |
| 9 | promote (`op.jobs.promote`) | `rg -n "_tasks\[|_detached|run_detached" src/services/training/promote_job.py` | `promote` | `jobs` | W2 / 2B | no | no (has `queued`) |
| 10 | dataset import, undo, resume (`op.jobs.dataset_import`, `op.jobs.dataset_import_undo`, `op.jobs.dataset_import_resume`) | `rg -n "_tasks\[store.import_id\]" src/services/curation/dataset_import/runner.py`; resume rescan inline: `rg -n "heartbeat_ticker" src/routers/curation/datasets.py` | `dataset_import` | `jobs` | W2 / 2B | yes | check |
| 11 | dataset upload extract (`op.jobs.dataset_upload_extract`) | `rg -n "def " src/services/curation/dataset_import/upload.py` (runs in the request, CC-5) | `dataset_upload:<upload_id>` (new scope) | `jobs` | W2 / 2B | no | new job shape |
| 12 | combine (`op.jobs.combine`) | `rg -n "_TASKS\[|create_task" src/services/projects/combine/service.py` (job dirs are global, not per project) | `combine` (target project) | `jobs` | W2 / 2B | yes | no |
| 13 | bulk ingest run (`op.jobs.bulk_ingest_run`) | `rg -n "def spawn\|run_coroutine_threadsafe" src/services/curation/bulk_ingest/runner.py`; callers `rg -n "start_source_run" src` | `bulk_ingest` | `jobs` | W2 / 2C | yes | no |
| 14 | **source poll / scan** (`op.jobs.source_poll`), **new in this inventory** | `rg -n "create_task\(_poll" src/services/curation/bulk_ingest/dispatcher.py` (module dicts `_scans`, `_loops`) | `source_scan` (`<source_id>`) | `jobs` | W2 / 2C | no (the next tick re-polls) | no |
| 15 | open-vocab drain (`op.jobs.open_vocab_drain`) | `rg -n "_PASS_LOCK|create_task" src/services/curation/open_vocab_ingest.py` | none: idempotent over item state (plain task, section 9.4) | `open_vocab` | W2 / 2C | n/a (the sweeper re-drains) | no |
| 16 | project delete finish (`op.jobs.project_delete_finish`) | `rg -n "_BACKGROUND_DELETE_TASKS|create_task" src/routers/curation/projects.py`; `rg -n "_FINISH_IN_PROGRESS" src/services/projects/delete.py` | none: the project record OCC is the claim (plain task) | `jobs` | W2 / 2C | n/a (re-DELETE retries) | no |
| 17 | directory ingest (`op.jobs.directory_ingest`) | `POST /ingest/directory` in `src/routers/ingest/directory.py` (inline `rglob` + `read_bytes`, CC-5) | none: not project-bound (D-22a) | `jobs` | W2 / 2C | no | 202 + job id (wire change) |

| 18 | **person grouping** (`op.jobs.persons_cluster`), **new in this inventory** (research finding, `docs/design/research_clustering_plan.md` N-1) | `POST /persons/cluster` in `src/routers/persons.py` runs an O(n^2) exact FAISS range search and connected components inside the request (about 4 min at 200k faces) | `persons_cluster` (global scope; faces are core-only) | `jobs` | W2 / 2C | no | 202 + job id (wire change) |

The family count above is 17 plus this row; counts elsewhere in this file that say "17 families"
read as 18 for the person-grouping family.

Not moved (stay in process, allowlisted in B4's ratchet with a reason):

| Site | Why it stays |
|---|---|
| `crop_orders.py` cluster-geometry backfill (`_BACKFILLING`, `_TASKS`) | idempotent per-process cache fill; lost on recycle is harmless. The task plan's "wrap with the asyncio-audit spawn helper (R5)" is superseded: former master plan Appendix B cut the spawn helper (K-9) |
| `pipeline_events.py` stats coalescing (`_stats_inflight`) | per-process cache coalescing (CC-11 fix) |
| `resource_links.py` reachability refresh | per-process cache; its `gather` fan-out is a B4-4A item (`05_b4_stateless_api.md`) |
| `vlm_factory.py` labeler close tasks | per-process resource cleanup |
| `ingest_writer.py` group commit | request-scoped; the caller awaits its own ack (task plan section 5, "What NOT to migrate") |
| `ingest_bulk_preview.py` disconnect watcher | request-scoped helper of one streaming response |
| `label_ops` record and undo | request-path writes under B1's `label_op` / `label_undo` leases; no background task (`rg -n create_task src/services/curation/label_ops` is empty) |
| `projects/legacy_repair.py` | runs from the registry bootstrap (`rg -n repair_legacy_projects src`), under B1's `legacy_repair` lease; idempotent; not a request job |

### 5.2 Periodic loops (today once per API worker, so 32 times)

| Loop | Where today | Destination | Wave / track |
|---|---|---|---|
| ingest dispatcher tick | `bulk_ingest/dispatcher.py` `_loop` via `background_jobs.start_background_jobs` | beat `op.periodic.ingest_dispatch` (5 s) | W3 / 2D |
| open-vocab sweeper | `open_vocab_sweeper.py` `_loop` via `background_jobs` | beat `op.periodic.open_vocab_sweep` | W3 / 2D |
| source-image cache prune (`.prune.lock`) | `source_image_cache.py` | beat `op.periodic.cache_prune` (**missing from the task plan's Phase 4 table**, K-3) | W3 / 2D |
| GPU-arbiter reconcile | `lifespan.py` `_arbiter_reconcile_loop` (15 s) | beat `op.periodic.gpu_arbiter` | W3 / 2D |
| promoted-model reload | inside the arbiter loop and `lifespan._reload_promoted_models` | beat `op.periodic.reload_promoted` (the on-demand route stays) | W3 / 2D |
| startup job reconcile | `lifespan._reconcile_orphaned_state` -> `startup_reconcile.reconcile_all_jobs` | beat `op.periodic.lease_reaper` (30 s) | W3 / 2D |
| cluster refresh | `scripts/curation/cluster_refresh_daemon.py` (service `curation-cluster-refresh`, curation profile) | beat `op.periodic.cluster_refresh` | W3 / 2D |
| IVF retrain check | `auto_label_worker.py` `_next_retrain` | beat `op.periodic.ivf_retrain_check` | W3 / 2D |

Stay in the API lifespan (per-process, correct per process): project registry poll
(`projects/bootstrap.py`), config store poll (`config_store/store_poll.py`), event-bus tail
(`event_hub.start_tail`; B4-4C replaces the transport), ops-metrics refresh (`ops_metrics_refresh.py`,
best-effort elected by `_claim_turn`, writes only its own Prometheus files), index bootstrap and
kNN warm-up (`core/dependencies.py`, create-if-absent with a marker since CC-11).

### 5.3 Worker containers

| Service (compose) | Profile | After B2 |
|---|---|---|
| `curation-detection-worker` | curation | unchanged (item state machine) |
| `curation-vlm-worker` | curation | unchanged |
| `curation-auto-label-worker` | curation | Celery consumer `-Q autolabel -c 1` (2E) |
| `curation-assist-loop-worker` | curation | Celery consumer `-Q assist_loop -c 1` (2E) |
| `curation-cluster-refresh` | curation | deleted; beat task (2D) |
| `curation-evaluator` | evaluator | unchanged in B2 (D-31 decides in B4) |
| `model-export-worker` | export | unchanged (rename claim; B1 made `reconcile_claimed` stale-only) |
| `curation-trainer` | training | unchanged (B1 create-only generation claims) |
| new: `redis`, `jobs-worker`, `periodic-worker`, `open-vocab-worker`, `jobs-beat` | none (core, D-19) | W1 |
| new: `flower` | monitoring | W1 |

### 5.4 Allowlisted `create_task` sites after B2

After W2 and W3, `rg -n "create_task\(" src/services src/routers src/app` may list only: the
section 5.1 "Not moved" rows, the request-scoped probes in `routers/health.py` (B4-4A decides),
the per-process pollers (`projects/bootstrap.py`, `config_store/store_poll.py`,
`event_hub.py` tail), `core/dependencies.py` warm-up, `lifespan.py` ops-metrics task, and
coroutine-internal helpers inside job code that now runs in a worker (`reprocess_images.py`,
`assist_loop/driver.py`, `bulk_ingest/runner.py` workers, `clustering/*`, `autolabel/job.py`
`_tick`). B4-4B turns this list into the ratchet allowlist.

### 5.5 Lifespan rule after B2

Allowed in lifespan: per-process pools, clients, caches, pollers, the event tail, idempotent
create-if-absent bootstrap with a marker, warm-ups. Forbidden: job execution, reconcile or repair
of shared records, periodic writes to shared state, Docker or Triton control. The B2 done-when
checks `src/app/lifespan.py` against this rule (no `_reconcile_orphaned_state`,
`_start_gpu_arbiter`, `_reload_promoted_models`, `start_background_jobs`).

---

## 6. Queue and concurrency topology

| Queue | Consumer service | Pool, concurrency | `--max-tasks-per-child` | Profile | GPU | `stop_grace_period` | Max queue wait (`MAX_QUEUE_WAIT_S`) |
|---|---|---|---|---|---|---|---|
| `jobs` | `jobs-worker` | prefork, `-c ${OP_JOBS_CONCURRENCY:-4}` (D-23) | 50 | core | `API_GPU_ID` reservation (UMAP/cuML ran in `api`) | 60 s | 86,400 s |
| `periodic` | `periodic-worker` | prefork, `-c 2` | 200 | core | none (the arbiter overlay mounts the Docker socket here, not in `api`) | 30 s | 300 s (plus beat `expires`) |
| `open_vocab` | `open-vocab-worker` | prefork, `-c 1` (the intent of `_PASS_LOCK`, Q-D6) | 50 | core | none (calls the segmenter over HTTP) | 30 s | 3,600 s |
| `autolabel` | `curation-auto-label-worker` | prefork, `-c 1` | 20 | curation | none | 60 s | 3,600 s (today's `QUEUE_MAX_AGE_S`) |
| `assist_loop` | `curation-assist-loop-worker` | prefork, `-c 1` | 20 | curation | none | 60 s | 3,600 s |
| `gpu_eval` | `curation-evaluator` | only if D-31 = move (B4) | - | evaluator | `EVALUATOR_GPU_ID` | - | - |

Rules:
- Global serialization (open-vocab drain, auto-label, assist loop) is single-consumer routing
  (`-c 1` and one container with a fixed `container_name`). The lease still guards per-project
  exclusivity (former master plan 3a, last row).
- A profile-gated queue (`autolabel`, `assist_loop`) can receive a message while its profile is
  down. The job then waits `queued` and the reaper fails it after `MAX_QUEUE_WAIT_S` with reason
  `queue_wait_exceeded`. That is today's behaviour (trigger unclaimed for `QUEUE_MAX_AGE_S`),
  kept on purpose. `tests/test_queue_consumers_from_compose.py` checks every routed queue has a
  consumer in the core install or in the profile `QUEUE_PROFILE` names for it (OpenTranscribe's
  "a valid route is not a drained route" lesson, Q-E18).
- `MAX_QUEUE_WAIT_S` is a constant table in `src/config/queue.py`, not an env var.
- E13 (two projects starving each other on `jobs`): FIFO with `-c 4` and per-project singletons;
  per-project routing only if measured.

---

## 7. Celery configuration (authoritative; `src/tasks/app.py`)

Cited docs (task plan section 9 IDs, plus the new ones in section 17). [C3] workers, [C4] periodic
tasks and [C1] Redis were re-fetched on 2026-11-03; the others are cited as the task plan fetched
them, and W1 re-reads the configuration reference [C10] before writing the module.

| Key | Value | Why | Doc |
|---|---|---|---|
| `broker_url` | built from `OP_QUEUE_URL` + the password read from `OP_QUEUE_PASSWORD_FILE` (never argv, never env) | the password stays in a mode-600 file | [C10], [V4] |
| `task_queues` | declared from `TaskQueues.ALL` (`src/config/queue.py`) | single source of truth | [C5] |
| `task_create_missing_queues` | `False` | a typo raises at dispatch instead of making a phantom queue | [C5], OpenTranscribe `celery.py` |
| `task_routes` | `TASK_ROUTES` from `src/config/queue.py`; an entry for every task by exact name; never `queue=` in a decorator | a decorator queue silently overrides the route | [C5], OpenTranscribe `app/tasks/CLAUDE.md` |
| `task_serializer`, `result_serializer` | `'json'` | pickle is insecure | [C6] |
| `accept_content` | `['json']` | whoever reaches the broker can enqueue work | [C6] |
| `task_ignore_result` | `True` (no result backend) | the job record is the UI's source | [C2] |
| `task_acks_late` | `True` | a worker crash leaves the message unacked | [C2] |
| `task_reject_on_worker_lost` | `True` | a killed prefork child requeues its message (needs the parent alive) | [C2] |
| `worker_prefetch_multiplier` | `1` | long tasks; with `acks_late` | [C7] |
| `broker_transport_options.visibility_timeout`, `app.conf.visibility_timeout` | `21600` (6 h), the same int in both (no result backend, so no `result_backend_transport_options`) | above the longest normal job; "redelivered to another worker and executed" after it | [C1] |
| `broker_transport_options` socket keys | `socket_keepalive=True`, `socket_keepalive_options` (`TCP_KEEPIDLE=30`, `TCP_KEEPINTVL=10`, `TCP_KEEPCNT=3`, `TCP_USER_TIMEOUT=60000`, each only if the platform has it), `socket_timeout=30`, `socket_connect_timeout=10`, `health_check_interval=25`, `retry_on_timeout=False` | a half-open socket wedged a consumer forever (OpenTranscribe #1144); copy the rationale comment | OpenTranscribe `celery.py`, [C10] |
| `broker_connection_retry`, `broker_connection_retry_on_startup` | `True`, `True` | survive broker restarts | [C10] |
| `broker_connection_max_retries` | `None` | the default 100 makes a worker give up after a long outage | [C10], OpenTranscribe |
| `task_soft_time_limit`, `task_time_limit` | `10800`, `11700` (3 h, 3 h 15 m) default; per-kind overrides for the long kinds (D-20a) | a timeout kill happens before redelivery | [C2], OpenTranscribe |
| `worker_max_tasks_per_child` | `50` (per-service CLI flag overrides, section 6) | bounds leaks; prefork only | [C3] |
| `worker_proc_alive_timeout` | `30` | a slow child start looped 69,231 fork/kill cycles (OpenTranscribe #631); no I/O in `worker_process_init` | [C10], OpenTranscribe |
| `worker_soft_shutdown_timeout` | `30` | on a **cold** shutdown wait 30 s, then cancel and requeue | [C3] "Soft Shutdown", [C8] |
| `worker_enable_soft_shutdown_on_idle` | `True` | also requeue reserved messages when idle | [C3] |
| `worker_cancel_long_running_tasks_on_connection_loss` | `False` | on Redis an unacked message survives a broker blip | [C10], OpenTranscribe |
| `worker_send_task_events`, `task_send_sent_event` | `True`, `True` | Flower | [C10] |
| `timezone`, `enable_utc` | `'UTC'`, `True` | beat schedules | [C4] |
| `beat_schedule` | section 9 of `03c` | exactly one scheduler | [C4] |
| env `REMAP_SIGTERM=SIGQUIT` on every worker service | so `docker stop` (SIGTERM) starts a cold shutdown, which is the only path the soft shutdown applies to (D-20b) | a plain SIGTERM is a warm shutdown that "will finish all currently executing tasks" and never applies the soft timeout; Docker then SIGKILLs at `stop_grace_period` | [C3] "Cold Shutdown" note, OpenTranscribe live test of shutdown |

Dependency pins (W1-2): `celery[redis]==5.6.*` exact (OpenTranscribe pins `celery==5.6.3`),
`billiard>=4.3.1` exact pin (OpenTranscribe: billiard 4.3.0 **acked** a running `acks_late` task on
cold shutdown and dropped it; 4.3.1 requeues it; verified in S1), `redis` (redis-py) exact, Flower
in the monitoring image only. Licences: Celery, kombu, billiard BSD-3; redis-py MIT; Flower BSD;
Redis 8 under its AGPLv3 option [V1].

---

## 8. Redis service (authoritative; W1-8)

| Setting | Value | Doc |
|---|---|---|
| image | `redis:8.2.x-alpine` with a pinned digest (OpenTranscribe pins `redis:8.2.2-alpine3.22`; take the newest 8.2 patch at W1 kick-off) | [V1], Docker Hub `redis` |
| persistence | `appendonly yes`, `appendfsync everysec` (about 1 s of loss) | [V3] |
| eviction | `maxmemory-policy noeviction` (eviction breaks kombu bindings) | [C1] "Key eviction" |
| auth | `requirepass` from `secrets/queue/redis_password` (mode 600, generated by the installer); the server reads it from a tmpfs config written at start (no password in argv) | [V4] |
| network | `triton_net` only; **no `ports:`** | [V4] |
| healthcheck | `redis-cli` ping with auth read from the file | Docker Hub `redis` |
| volume | `openprocessor-queue` (named), no backup requirement (documented) | [V3] |
| hardening | `security_opt: no-new-privileges:true` (OpenTranscribe sets it on Redis) | - |

Trap (W1 S1 verifies): the official image starts as root and drops to the `redis` user; the API
image runs as `appuser` (uid 1000). A mode-600 host file must be readable by both the Redis start
script (as root, before it drops privileges) and the Celery clients (uid 1000). Follow the
existing generated-secret pattern (`rg -n "secrets/cropwright" docker-compose.yml setup-openprocessor.sh`)
and record the ownership rule in INSTALLATION.

---

## 9. Shared contract: job record, lease and queued marker

This section is the contract every B2 track follows. W1 builds it; W2 and W3 use it.

### 9.1 States

| Record status | Lease state (B1 doc) | Lease TTL | Who writes |
|---|---|---|---|
| `queued` | `queued`, owner `api:<host>:<pid>` | `MAX_QUEUE_WAIT_S[queue]` | the API at start; the reaper on re-enqueue |
| `running` | `running`, owner `<worker hostname>:<pid>` | `LEASE_TTL_S` (120 s), renewed every 15 s by the B1 heartbeat thread | the task wrapper (the lease holder) |
| `interrupted` (+ `auto_resume: true\|false`) | `released` | - | the wrapper on a worker shutdown; the reaper after taking over an expired lease |
| `completed`, `completed_with_errors`, `failed`, `cancelled` | `released` | - | the wrapper (after `fence`), or the cancel route for a queued job |

The B1 lease doc already has `state` and `ttl_s`. W1 extends `acquire` to take
`state='queued'` and `transfer` to set `state='running'` and `ttl_s=LEASE_TTL_S` (an extension of
B1's functions, not a rewrite; correction K-1).

### 9.2 Start (API, every FileJob family)

1. `acquire(kind, scope, job_id, owner, ttl_s=MAX_QUEUE_WAIT_S[queue], state='queued')`.
   `Busy` -> 409 `job_busy` (existing per-family code). OpenSearch down -> 503 (fail closed).
2. Write the `FileJob` record `status=queued`, `attempt=0`, `queued_at`, `lease_generation`.
3. `task_queue.enqueue(name, project=slug, job_id=job_id)`. The Celery `task_id` is the job id.
   Broker down -> the record becomes `failed` (`queue_unavailable`), the lease is released, and the
   route answers 503 `queue_unavailable` (Q-E3).
4. `SET op:queued:<job_id> 1 EX <MAX_QUEUE_WAIT_S[queue]>` (after the send, so a crash between
   send and marker makes a harmless duplicate, never a loss). Redis `SET` with `EX` [V5].

### 9.3 Task wrapper (`JobTask`, every FileJob family)

1. Bind the project from the message (`project`, `job_id`). Missing or archived -> record `failed`
   (`project_unavailable`), ack (Q-E11).
2. Build OpenSearch with `make_script_opensearch` (guarded), inside `asyncio.run`, never shared
   across tasks (Q-E9).
3. Read the record. Terminal -> ack, no-op (Q-E6). `interrupted` with `auto_resume` false, or a
   non-resumable kind -> ack, no-op.
4. `transfer(lease, owner=<hostname>:<pid>, state='running', ttl_s=LEASE_TTL_S)`. A live
   `running` lease held by another owner -> log `lease_busy`, ack, no-op (redelivery; Q-E1).
   `attempt >= 3` -> record `failed` with the last error (Q-E2).
5. Write `status=running`, `attempt += 1`, `owner`, `lease_generation`; `DEL op:queued:<job_id>`.
6. Start the B1 heartbeat thread unchanged (`lease_heartbeat.py`); it renews every 15 s and sets
   the stop flag on `Lost`.
7. `asyncio.run(handler(job, ctx))`. The handler builds its clients from `ctx`, never from a
   request.
8. Outcome -> `fence(lease)` -> terminal write -> `release(lease)`:
   success -> `completed` / `completed_with_errors`; `CancelledError` or cancel flag ->
   `cancelled`; `SoftTimeLimitExceeded` -> `failed` (`time_limit`); other exception -> `failed`
   with the message. A failed `fence` (`Lost`) -> write nothing, log `lease_lost`.
9. Worker shutdown (the soft-shutdown cancel raises `SystemExit` in the child): `fence`, write
   `interrupted` with `auto_resume = (kind in AUTO_RESUME)`, release the lease, re-raise so
   billiard reports a lost worker and Celery requeues the message (S1 verifies; Q-E8, Q-E25).
10. `finally`: stop the heartbeat thread.

### 9.4 Plain task (no FileJob: `open_vocab_drain`, `project_delete_finish`, `directory_ingest`, periodic tasks)

Bind the project when there is one, `asyncio.run`, idempotent body, no lease. Duplicates are
harmless by construction (item state machine, record OCC, or create-if-absent). Celery does **not**
de-duplicate by `task_id`; never rely on it (correction K-7).

### 9.5 Reaper (`op.periodic.lease_reaper`, 30 s; built in W3, authoritative rules here)

For each project, every FileJob family directory, and the global combine directory:
- **(a)** `running` and the lease expired (`heartbeat_at + ttl_s < now`) ->
  `takeover_expired` -> record `interrupted`, `auto_resume = (kind in AUTO_RESUME and attempt < 3)`.
  A `running` record with **no** lease doc follows B1's mid-upgrade rule (stale heartbeat ->
  take over; fresh -> live until it ages past `LEASE_TTL_S`).
- **(b)** `interrupted`, `auto_resume` true, no `op:queued:<job_id>`, older than 120 s ->
  re-acquire the lease as `queued`, write `queued`, enqueue, set the marker.
- **(c)** `queued`, no marker, older than 120 s -> enqueue again and set the marker (covers a lost
  broker volume, Q-E4, and an API crash between record and enqueue, Q-E12). The duplicate is a
  lease no-op.
- **(d)** `queued` and the queued lease expired (`MAX_QUEUE_WAIT_S`) -> record `failed`
  (`queue_wait_exceeded`), release.
- Redis unreachable -> conclude nothing, change nothing (fail safe; OpenTranscribe
  `task_liveness.py` "UNKNOWN, never dead").
- OpenSearch unreachable for a project -> skip that project this tick, log it.

### 9.6 Cancel

`queued` -> `task_queue.revoke(job_id)`, write `cancelled`, release the lease, `DEL` the marker.
`running` -> touch `cancel.flag` (unchanged; cooperative). Never `revoke(terminate=True)`:
"you must never call this programmatically" [C3]; ratchet in W1 (Q-E7). Revoke ids are in memory
only ([C3] "Persistent revokes"); the record's terminal status is the truth (Q-E6).

---

## 10. Edge-case map (Q-E1..E26 -> the task that handles it)

| E | Edge case | Handled by |
|---|---|---|
| E1 | redelivery after the visibility timeout while running | wrapper step 4 (`03a` W1-6); test with a lowered timeout; long kinds need D-20a |
| E2 | poison message loops with `reject_on_worker_lost` | wrapper attempt cap 3 (W1-6); `--max-tasks-per-child` (W1-8) |
| E3 | broker down at enqueue | `task_queue.enqueue` raises, 503, record `failed` (W1-5); per-family unit test (W2) |
| E4 | broker data lost | reaper rule (c) with the queued marker (`03c` 2D) |
| E5 | two beats | fixed `container_name`, compose test (W1-8); idempotent bodies (2D) |
| E6 | revoke lost on restart | terminal-status refusal (W1-6); cancel path (W2 step 4) |
| E7 | `terminate=True` temptation | ratchet test `tests/test_no_revoke_terminate.py` (W1-12) |
| E8 | `docker stop` during a long job | `REMAP_SIGTERM=SIGQUIT` + soft shutdown 30 s + teardown write (W1-6, W1-8, D-20b); S1 |
| E9 | asyncio in prefork children | `asyncio.run` per task, clients per task (W1-6) |
| E10 | CUDA and fork | no model or CUDA import at module scope in `src/tasks` (W1-4, test `tests/tasks/test_no_cuda_import.py`) |
| E11 | project archived or deleted while queued | wrapper step 1 (W1-6) |
| E12 | record written, enqueue never attempted | reaper rule (c) (2D) |
| E13 | projects starve each other on `jobs` | topology note (section 6); measure before acting |
| E14 | new `queued` status | contracts per family (W2), Cropwright (2G), browser check (sweep) |
| E15 | `flock` on non-local volumes | INSTALLATION "single host, local volumes" note (W1-14); after B4 only the `FileJob` files need one host (D-21) |
| E16 | class-registry lock blocks the loop | superseded: B1-1B registry doc with async OCC (no lock) |
| E17 | Redis reachable from the LAN | no `ports:`, required password, JSON only (W1-8, section 8) |
| E18 | route into an undrained queue | `task_create_missing_queues=False` + `tests/test_queue_consumers_from_compose.py` with `QUEUE_PROFILE` (W1-13) |
| E19 | metrics from Celery children | OPS-2 evidence (W1-0), then per-service metrics dir + `mark_process_dead` (2D) |
| E20 | combine and import touch two projects | wrapper binds the job's (target) project; combine keeps binding sources in turn inside `execute.py` (2B) |
| E21 | dev hot-reload | dev overlay mounts + "restart the worker after task edits" doc (W1-8, W1-14) |
| E22 | runner and status poll write one record | superseded by B1: single writer (lease holder), readers never write (CC-1); `fence` before writes (W1-6) |
| E23 | wedged prefork pool reports healthy | pool healthcheck port (W1-9) |
| E24 | half-open broker socket | transport socket options (section 7, W1-4) |
| E25 | cold shutdown acks a running task | **corrected**: billiard >= 4.3.1 requeues it (OpenTranscribe); S1 verifies; the reaper still recovers from records (2D) |
| E26 | 1 vs 32 API workers differ | B4-4B ratchet + live diff (`05_b4_stateless_api.md`); B2 removes the job and periodic part (section 13) |

## 11. Defect map (Q-D, B2 and B4 parts)

| Q-D | Defect | Fixed in |
|---|---|---|
| D1 | jobs inside recyclable request workers (CC-4) | W2 (every family); the interim jitter landed in B0 |
| D4 | startup reconcile assumes it is alone | B1 lease takeover; **deleted** in 2D |
| D5 | GPU arbiter in 32 processes | 2D |
| D6 | per-process serialization (`_PASS_LOCK`) | 2C (`open_vocab` `-c 1`) |
| D7 | worker-container claims need one replica | trainer and training jobs: B1; auto-label and assist loop: 2E |
| D8 | at-most-once auto-label claim | 2E |
| D9 | blocked loop falsifies liveness | B1 thread heartbeat, reused unchanged by the wrapper |
| D14 | per-process caches and caps | B1-1D (caches); B4-4B (pools per `API_WORKERS`) |
| D15 | Prometheus multiprocess files accumulate | 2D (after the W1-0 evidence) |

---

## 12. Decision detail (status, owner answers and approvals live only in [00_index.md](00_index.md) section 6)

This table is the recommendation and consequence text behind the B2 register rows; it does not
carry status. Where it disagrees with the register, the register wins.

| ID | Decision | Recommendation | Consequence | Needed by |
|---|---|---|---|---|
| D-18 | Q-J2 broker | Redis 8 under its AGPLv3 option [V1], the same family as OpenTranscribe; Valkey (BSD) only if a redistributor objects, and then after a transport spike | one more image in the default install; one set of operating knowledge across the owner's two products | W1 kick-off |
| D-19 | Q-J3 default install | core (no profile): `redis`, `jobs-worker`, **`periodic-worker`** (new, K-6), `open-vocab-worker`, `jobs-beat`; Flower under `monitoring`, published on `${OP_BIND_ADDRESS:-127.0.0.1}:4613` with basic auth from `secrets/queue/flower_auth`, or not published at all if the owner prefers | **five** more containers (the former master plan said four) and two secrets; curation routes work by default; `openprocessor status/health` grows five rows. Alternative (a `jobs` profile) leaves core curation routes broken by default | W1 kick-off |
| D-20 | Q-J4 delivery | `acks_late` + `reject_on_worker_lost` + lease + attempt cap 3 + soft shutdown 30 s | at-least-once delivery made idempotent by the lease | W1 kick-off |
| D-20a | time limits for long kinds (new) | default 3 h / 3 h 15 m; `bulk_ingest_run`, `dataset_import*`, `combine`, `reprocess`, `directory_ingest` get 24 h / 24 h 15 m per task. Above the 6 h visibility timeout these rely on E1 (redelivery finds a live lease and acks) | without overrides a 1M-image ingest or a large combine is killed at 3 h 15 m; with them a wedged job holds its slot up to 24 h (the thread heartbeat keeps a wedged loop's lease) | W1 (wrapper), W2 (per kind) |
| D-20b | `REMAP_SIGTERM=SIGQUIT` on worker services (new) | yes, with `stop_grace_period` at least 30 s above `worker_soft_shutdown_timeout` (60 s for `jobs`) | `docker stop` and `openprocessor upgrade` interrupt running jobs after 30 s and requeue them (resumable kinds resume elsewhere at once); without it a `docker stop` waits the grace period and SIGKILLs, and recovery waits for the lease TTL plus a reaper tick | W1 (S1 result first) |
| D-21 | Q-J5 job-state store | status, progress and cancel stay in `FileJob` files on the shared volumes; exclusivity is the OpenSearch lease; the queued marker is a Redis key | `/jobs` stays single host until a multi-host store is wanted (blocks `API_WORKERS=1` + replicas, D-33) | W1 kick-off |
| D-22 | Q-J6 scope | migrate all 17 request-started families (section 5.1, including the newly listed source poll), the periodic loops (including the cache prune), auto-label and assist loop; keep detection/VLM workers, trainer, export worker; the evaluator is D-31 | as planned | W2 kick-off |
| D-22a | directory ingest (`POST /ingest/directory`), not project-bound (new) | a plain Celery task with a deployment-level job record under `/jobs/ingest_directory/<job_id>`, no lease; 202 + job id; 2C proves the visual-search ingest is idempotent per image before it ships (else escalate). Alternative: delete the legacy route (it duplicates bulk ingest), which needs the owner's word since `/ingest` is a documented core route | one wire change on a core route (contracts, README) | W2 (2C) |
| D-22b | promote `wait=true` (new; synchronous promote inside the request, shielded by `run_detached`) | delete the mode (no shim); only tests use it (`rg -n "wait=true" tests \| rg promote`); Cropwright and the acceptance phase use the async path | one query parameter removed (contracts, CHANGELOG "Breaking") | W2 (2B) |
| D-23 | Q-J9 `jobs` concurrency | 4, tunable with `OP_JOBS_CONCURRENCY` | lower than today's "one per busy API process"; per-project singletons already bound most jobs | W1 kick-off |
| D-24 | Q-J10 Redis durability | AOF everysec, no backup | worst case a short delay: the reaper re-enqueues from the records (rules b, c) | W1 kick-off |
| D-25 | Q-J11 auto-resume | `AUTO_RESUME = {reprocess, dataset_import, dataset_import_resume, bulk_ingest_run, combine, assist_loop}` | the others end `interrupted` and need a manual restart, as today | W2 kick-off |
| D-26 | Q-J12 recycling | keep `--limit-max-requests` with the jitter that landed in B0 | no change; harmless once no job lives in the API | none (recorded) |

---

## 13. The 1-vs-32 equivalence goal (B2 part)

Goal (task plan section 4e): the API gives the same answers with 1 uvicorn worker and with 32, so
the worker count is a tuning knob only. B2 removes the job and periodic part of the difference:
- no job runs in an API worker, so recycling and the worker count no longer change job outcomes;
- periodic effects run once per deployment (one beat, one `periodic` consumer group);
- claims are the OpenSearch lease, not process memory.

B2 proves its part in the live sweep (section 14): the API-recycle test, and the race and kill tests
with two job workers. The scripted 1-vs-32 diff of the whole API runs in B4
(`05_b4_stateless_api.md`), after caches and pools are sized per `API_WORKERS`.

---

## 14. The B2 sweep (once, after stage W3 merged)

Run by the coordinator on `int/b2-celery`. Announce it to peer sessions first; one heavy run at a
time; it never overlaps B3's GPU window (`11_risks_cleanup_host.md`). Budget rules:
[00_index.md](00_index.md) section 8.1. Stage merges before this point got T0 checks only.

**T1 (offline, about 35 min):**
- `.venv/bin/pre-commit run --all-files`, and the pre-push hooks;
- `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live' -n 8`;
- `make contracts-check`; `.venv/bin/python scripts/docs/check_docs_vs_code.py`;
- frontend full set (2G changed `frontend/`): `npm ci`, `npm run check`, lint, knip, `npm run test`,
  `npm run build`, `npm run test:e2e` (CLAUDE.md "Frontend");
- the unit tests this batch adds (they run inside the suite): `tests/tasks/test_job_task_wrapper.py`,
  `tests/tasks/test_task_routes_complete.py`, `tests/test_queue_consumers_from_compose.py`,
  `tests/tasks/test_celery_config.py` (asserts the soft-shutdown keys), per-family "enqueues once
  with `task_id=job_id`" and "broker down -> 503, record `failed`", the beat schedule tests, and
  `tests/compose/test_beat_single_instance.py` (`jobs-beat` cannot scale).

**T2 live harness: the one run that covers B1 and B2 together, `-p op-b2-live`, budget 3 h.**
Because B1 had only a 30-minute lease/registry smoke check, this run first executes the full
`tests/live` set plus the B1 live scenarios of B1 (merged; see CHANGELOG.md) section 7.1 (lease race,
registry race and HTTP sequence, OS-6 hit count, blocked-loop check), then the B2 steps below; it
is the only live run between the B1 smoke check and B4's 1-vs-32 run. The harness needs no GPU
(the arbiter step uses the overlay, not a model). `docker/test/compose.yml` gains
`redis`, `jobs-worker-a`, `jobs-worker-b` (two containers, because `container_name` blocks
`--scale`), `periodic-worker`, `open-vocab-worker`, `jobs-beat`, and the two trigger consumers.
Wait with healthchecks or the Monitor tool; no sleep loops.

| Step | Scenario | Pass | Budget |
|---|---|---|---|
| 1 | bring-up, `/health` shows `queue: ok` | all healthchecks green | 15 min |
| 2 | `tests/live/test_queue_infra.py` | a no-op task runs | 5 min |
| 3 | race: `send_task` the same reprocess `job_id` twice | one run in the history; the other worker logs `lease_busy` | 10 min |
| 4 | kill mid-job: reprocess of 2,000 images, `docker kill jobs-worker-a` after progress > 0 | `interrupted` within `LEASE_TTL_S` + one reaper tick, then `running` on b with `attempt=2`, completes; image count equals the total; no duplicate writes | 25 min |
| 5 | cold-shutdown requeue: `docker stop jobs-worker-a` mid-job (REMAP + soft shutdown) | the message is requeued and resumes on b without waiting for the TTL (D-20b) | 15 min |
| 6 | cancel running; cancel queued (both workers stopped, start, cancel, start workers) | `cancelled` within one progress step; the queued job never runs | 15 min |
| 7 | API recycle: API at `--limit-max-requests=50`, 500 requests during a reprocess | the job completes | 15 min |
| 8 | broker loss: stop Redis, delete the queue volume, start it | queued jobs re-enqueued by the reaper (rule c); running jobs keep their lease | 10 min |
| 9 | GPU arbiter with the overlay: training start | one stop per configured container in the Docker events log, not 32 | 15 min |
| 10 | two auto-label consumers (harness only) + one job; kill mid-stage; cancel at a stage boundary | one run; `interrupted` and manual restart; `cancelled` | 20 min |
| 11 | per-family smoke (start, progress, completes) for the 17 families | each completes | 20 min |
| 12 | two B1 scenarios that were never scripted, written as `tests/live/` tests before this run (see the two tasks after this table) | both pass | 10 min |
| 13 | acceptance subset: `acceptance_run.sh --only ingest,cluster,holdout,label_ops,loop,export,train,bakeoff,promote` on this stack | PASS | 15 min |

**Live-test gap carried from B1 (tasks for the B2 live-run lane).** B1 merged with two of its
planned live checks never scripted as tests. Write both under `tests/live/` (harness rules in
`docker/test/README.md`; red first where a pre-fix state is reproducible) and run them as step 12:

- **OS-6 filtered-kNN hit count.** On a legacy visual-search index with more than `k`
  documents of which a restrictive class filter matches only a few, a filtered
  `/search/objects` query returns `k` hits. The filter sits inside the `knn` clause now
  (`src/clients/opensearch/search.py`); the same query with the filter wrapped in `bool.filter`
  beside the `knn` clause returns fewer. Record both hit counts in `docs/PERFORMANCE.md` and
  `docs/opensearch_schema_design.md`.
- **Blocked-loop lease heartbeat.** Start an API-hosted leased job (a reprocess or an item
  score), then block the API event loop with a large dataset upload; the lease document's
  `heartbeat_at` keeps advancing during the block (the heartbeat thread has its own loop and
  client, `src/services/curation/lease_heartbeat.py`) and the job is not marked `interrupted`.
  Also record the lease p50 and p99 acquire latency in `docs/PERFORMANCE.md`.

Total 3 h. A step that overruns stops the sweep; the coordinator reports it instead of extending
the budget silently.

**Images:** Trivy on every image B2 adds or changes (`redis`, the api image with Celery, the
monitoring image with Flower), and the image-vs-git audit. Add the Redis, Celery, kombu, billiard,
redis-py and Flower rows to `ATTRIBUTION.md` (W1-2).

**Browser check:** the jobs UI (progress, cancel, the new `queued` state) at 1440 and 430 px,
full-page screenshots viewed.

Then one PR from `int/b2-celery` against main, CI green (local CI parity first), a merge commit.

---

## 15. Upgrade impact (0.6.0 -> 0.7.0 directly)

Owner decision 2: the supported upgrade is 0.6.0 -> 0.7.0 in one step. Redis, the new services,
the queue secret and B1's registry migration all arrive at once.
- `openprocessor upgrade` and `repair` create `secrets/queue/redis_password` (and the Flower auth
  file if D-19 publishes Flower) when missing, not only on fresh installs (M11).
- They add the new services and the new keys from `env.template` (08 owns the table).
- **In-flight jobs.** A 0.6.0 stack runs jobs inside the API and has no lease docs. The installer
  stops the API (`stop_grace_period`) and the in-API jobs die. On first start the reaper applies
  B1's mid-upgrade rule (no lease doc + stale heartbeat -> take over), marks them `interrupted`,
  and re-enqueues the D-25 kinds (rule b). The installer prints which jobs were interrupted.
- **Trigger files from 0.6.0.** A `trigger.json` left by the 0.6.0 auto-label or assist-loop API is
  never read again. The record says `queued` with no marker, so the reaper re-enqueues it (rule c)
  and 2E deletes the stale trigger file and `running.lock` in the same step.
- The upgrade order with B1: the registry migration runs after the stack is healthy (B1); the
  reaper's first tick may re-enqueue jobs before the migration finished; class-dependent jobs then
  fail with 409 `registry_not_migrated` and are resumable once it ran. R-final tests this order.
- Tested by `installer_upgrade` from 0.6.0 in R-final.

## 16. Rollback

**B2 is forward-only.** Once the in-API runners are deleted there is no switch back. Before the
release: revert the B2 PR merge commit on main (or a track's merge commit on `int/b2-celery`).
After v0.7.0 ships, the only earlier release is 0.6.0; going back is a reinstall of 0.6.0 with
B1's registry downgrade note (`02_*`). Redis and the new services are simply not started by 0.6.0;
the job records are files 0.6.0 reads (a `queued` status on a family that lacked it shows as
unknown in 0.6.0's Cropwright; documented in the CHANGELOG "Breaking" section).

## 17. Docs, contracts, CHANGELOG (B2)

- `docs/ARCHITECTURE.md`: services table, "Workers, jobs and events", a queues table (as in
  OpenTranscribe's architecture doc).
- `CLAUDE.md`: services table (the five core services, Flower), a short Celery conventions block
  (routes in `src/config/queue.py`, no decorator `queue=`, no I/O in `worker_process_init`, no
  `terminate=True`, restart the worker after task edits).
- `INSTALLATION.md` (Redis, the secret, single host, ownership of the secret file, upgrade note),
  `docs/design/curation_design_rationale.md` section 6 ("Known gaps": update the worker list and
  drop the "experimental for v0.4.0" line, since jobs now run on Celery workers),
  `docs/CURATION.md` (job states, `queued`, `interrupted` + auto-resume, the busy window),
  `SECURITY.md` (broker preconditions; SEC-4 and SEC-5 neutral lines), `docs-site` (the same pages).
- `contracts/`: `queued` for the families in section 5.1, the upload-extract job, directory ingest
  202. Run `make contracts` in the same commit as each wire change.
- `CHANGELOG.md` `[Unreleased]` (root): "Added" (Celery workers, beat, Redis), "Changed" (jobs
  survive API recycling), "Breaking" (new core services and secret, `queued` status, directory
  ingest returns 202). `frontend/CHANGELOG.md` (2G).
- `ATTRIBUTION.md`: Celery, kombu, billiard, redis-py, Flower, Redis 8 (AGPLv3 option).
- Docs check: `check_docs_vs_code.py` in the sweep.

### 17.1 What to keep after merge

| Fact or section | Target doc |
|---|---|
| queues, topology, Celery configuration, the job record/lease/queued-marker contract, reaper rules (sections 4, 6, 7, 9) | `docs/ARCHITECTURE.md` |
| job states, `queued`, `interrupted` and auto-resume (D-25), busy window | `docs/CURATION.md`, `docs/design/curation_api_contract.md` |
| Redis service (no published port, AOF), secrets, upgrade and rollback notes (sections 8, 15, 16) | `INSTALLATION.md`, `SECURITY.md` |
| the family list and per-family recipe (`03b`), beat table and trigger workers (`03c`) | `docs/ARCHITECTURE.md`; the user-facing parts `docs-site` |
| measured request latency under a running job; kill/race results of the live run | `docs/PERFORMANCE.md` |
| the Breaking entries | `CHANGELOG.md`, `frontend/CHANGELOG.md` |

## 18. Config handed to `08_environment_and_setup.md`

The definitions here are authoritative; 08 carries them in the cross-batch table.

| Name | Kind | Default | Read by | Purpose |
|---|---|---|---|---|
| `OP_QUEUE_URL` | env | `redis://redis:6379/0` | `src/config/queue.py` | broker URL without the password |
| `OP_QUEUE_PASSWORD_FILE` | env | `/run/secrets/queue/redis_password` | `src/config/queue.py` | password file (mode 600) |
| `OP_JOBS_CONCURRENCY` | env | `4` | compose `jobs-worker` command | D-23 |
| `OP_STOP_GRACE_JOBS` | env (compose only) | `60` | compose `jobs-worker` | must exceed the 30 s soft shutdown |
| `OP_API_LOOPBACK_URL` | env (exists) | `http://api:8000` in the worker services | `bulk_ingest/dispatcher.py` `loopback_url` | workers reach the API by service name |
| `REMAP_SIGTERM` | env on worker services | `SIGQUIT` | Celery | D-20b |
| `secrets/queue/redis_password` | generated secret (mode 600) | random 32 bytes | redis, every Celery client | D-19 |
| `secrets/queue/flower_auth` | generated secret | random | flower | only if D-19 publishes Flower |
| `openprocessor-queue` | named volume | - | redis | AOF data |
| services `redis`, `jobs-worker`, `periodic-worker`, `open-vocab-worker`, `jobs-beat` | compose, core | - | installer, CLI status/health | D-19 |
| service `flower`, port 4613 | compose, `monitoring` | loopback | - | D-19 |
| `celerybeat-schedule*` on the state volume | file | - | `jobs-beat` | beat store; the healthcheck globs `celerybeat-schedule*` (Python 3.13 `shelve` uses SQLite WAL) |

Removed by B2 (08 deletes them from `env.template`; `src/config/retired_env.py` rejects the env
ones): the `curation-cluster-refresh` service; any env read only by the deleted loops (2D lists
them with `rg -n "env_(str|float|int)\('OP_" src/services/curation/{background_jobs,open_vocab_sweeper}.py scripts/curation/cluster_refresh_daemon.py`).
Kept and re-pointed: `OP_INGEST_DISPATCH_S`, `OP_OPEN_VOCAB_SWEEP_S`,
`OP_CLUSTER_REFRESH_INTERVAL_S`, `AUTO_RETRAIN_CHECK_INTERVAL_S` become beat intervals.

## 19. Release checks handed to R-final (`06_release_acceptance.md`)

- `installer_upgrade` from 0.6.0: the queue secret is created, the five core services start
  healthy, in-flight jobs end `interrupted` and the D-25 kinds resume; stale trigger files are gone;
  the registry migration and the first reaper tick do not race into a wrong state.
- Fresh install: `openprocessor status/health` lists the five services; `/health` reports `queue`.
- `installer_repair` recreates a deleted queue secret.
- Every acceptance phase runs with jobs on Celery (not only the B2 subset).
- Kill a job worker mid-reprocess on the release stack: resumes with `attempt=2`.
- `docker compose ... ps` shows no published Redis port; Flower only under `monitoring`.
- Trivy and the image-vs-git audit on `redis`, the api image and the monitoring image; the
  ATTRIBUTION rows present.
- Browser check of the `queued` state and the cancel of a queued job at 1440 and 430 px.
- The CHANGELOG `[Unreleased]` "Breaking" lines for B2 are present and accurate.
- Rollback rehearsal: reinstall 0.6.0 over the 0.7.0 data dir per the downgrade notes.

## 20. Risks

| # | Risk | Mitigation |
|---|---|---|
| R2-1 | redelivery storm (E2) | attempt cap 3; `--max-tasks-per-child` |
| R2-2 | a cold-shutdown ack drops a message (E25) | billiard >= 4.3.1 pin; S1 check; reaper rules a-c recover from the records |
| R2-3 | a route into a queue nothing drains (E18) | `task_create_missing_queues=False`; compose consumer test with `QUEUE_PROFILE`; queue-wait failure |
| R2-4 | Cropwright misses `queued` (E14) | contracts per family; 2G; browser check |
| R2-5 | the Redis secret is unreadable by a container user | S1 trap check; follow the Cropwright secret pattern; INSTALLATION |
| R2-6 | an install upgraded from 0.6.0 lacks the secret (M11) | `upgrade` and `repair` create it; R-final tests it |
| R2-7 | `periodic` blocked behind long jobs | its own consumer (K-6) |
| R2-8 | a wedged job keeps its lease (thread heartbeat survives a wedged loop) | per-kind hard time limits (D-20a); cancel stays available |
| R2-9 | 12 start paths edited twice (B1 claim lines, B2 launch lines) | accepted (former master plan A.5 #3): the second edit is mechanical (`create_task` -> `enqueue`) |
| R2-10 | contract drift between W2 tracks (M19) | `make contracts` per commit; contracts-check after each merge; 2G after the contracts settle |
| R2-11 | the live sweep overruns the 3 h budget | stop and report; never extend silently |
| R2-12 | Cropwright changes of B1, B2 and B4 (M24) | all ship in the one v0.7.0 release, so one frontend release |

## 21. Corrections (stale or wrong references found while verifying)

| # | Where | Correction |
|---|---|---|
| K-1 | former master plan 7.2, D-8 | the API-acquired lease has no heartbeat while the job waits in the queue, so with B1's 120 s TTL it would expire and a second start could win. The lease is acquired as `state='queued'` with the queue-wait TTL and transferred to `running` / 120 s by the wrapper (section 9.1) |
| K-2 | task plan section 4d, E25; former master plan 7.2 risks | "cold shutdown acks a running `acks_late` task" was true for billiard 4.3.0. OpenTranscribe now pins billiard >= 4.3.1, under which the cold-shutdown cancel **requeues** the message. Pin billiard; S1 verifies |
| K-3 | task plan Phase 4 table | the source-image cache prune (`source_image_cache.py`, `.prune.lock`) is a periodic loop missing from the table; former master plan 3b already schedules it for beat. Added as `op.periodic.cache_prune` |
| K-4 | task plan 2a | the dispatcher's in-process source poll (`dispatcher.py` `_poll`, `_scans`, `_loops`) is a request-independent background job missing from the inventory. Added as family 14 `source_poll` |
| K-5 | task plan section 5 rule 4, E8 | soft shutdown applies only to a cold shutdown (SIGQUIT, or SIGTERM with `REMAP_SIGTERM=SIGQUIT`); a plain `docker stop` is a warm shutdown that waits for the task with no limit ([C3]). D-20b |
| K-6 | task plan section 5 diagram, Phase 2 compose | `periodic` on `jobs-worker -Q jobs,periodic` would queue the reaper and the dispatch tick behind four long jobs. `periodic` gets `periodic-worker` (D-19 becomes five containers) |
| K-7 | task plan Phase 3 row 10 | "dedupe by `task_id=f'delete:{slug}'`" is not a Celery feature; two messages with one `task_id` both run. Idempotency comes from the project record OCC; the per-process `_FINISH_IN_PROGRESS` set in `projects/delete.py` is deleted too |
| K-8 | task plan Phase 2 "New files" | `src/tasks/queues.py` would force `src/clients/task_queue.py` to import `src.tasks`, which the layering rule forbids. Queue names, routes and `QUEUE_PROFILE` live in `src/config/queue.py` (pure data); `src/tasks/app.py` and the client both read them |
| K-9 | task plan Phase 3 "Not moved" | the crop-orders backfill "wrapped with the asyncio-audit spawn helper (R5)": the helper was cut (former master plan Appendix B). It stays a plain `create_task`, allowlisted in B4's ratchet |
| K-10 | former master plan 3b, W-9 | the flock count is now **19 files, 55 lines** (`rg -c "exclusive_start_lock\|exclusive_file_lock\|fcntl\.flock" src`), not 18 / 51 |
| K-11 | task plan Phase 2 time limits vs former master plan 7.2 "jobs longer than 6 h" | with `task_time_limit=11700` no job can outlive the 6 h visibility timeout; the 6 h scenario exists only with per-kind overrides (D-20a) |
| K-12 | task plan 4d "Beat health: schedule-file mtime" | Python 3.13 `shelve` writes SQLite WAL, so the base file's mtime goes stale on a working beat (OpenTranscribe observed it). The healthcheck takes the newest mtime across `celerybeat-schedule*` and fails on an empty glob |
| K-13 | task plan Phase 3 row 11; Q-section 2a | directory ingest is not project-bound (legacy visual search), so it has no project lease index. D-22a |
| K-14 | former master plan 7.2 upgrade | "0.8.0 -> 0.9.0" and "users downgrade to 0.8.0" are obsolete (owner decision 2): the upgrade is 0.6.0 -> 0.7.0 and the earlier release is 0.6.0 |
| K-15 | the docs check | RESOLVED: the checker no longer applies its route and env checks to `docs/design/` (commit `0cc2b3ac`), so plans may name the variables and routes they will add (`OP_QUEUE_URL`, `OP_JOBS_CONCURRENCY`, `OP_BATCH_SYNC_MAX`). Once B2 adds them to code and `env.template`, the user docs (not the plans) must name them correctly |

| K-16 | task plan Phase 4 "Delete ... `ops_metrics_task`" | contradicts the same phase's "Keep: ... the ops-metrics refresh in the API"; the keep wins (`03c` 2D-3) |
| K-17 | task plan Phase 3 row 7 | "fix OS-4 in the same commit" is void: OS-4 was refuted (former master plan Appendix B) |
| K-18 | task plan 2e, audit CC-10 | `POST /curation/projects/{project}/vlm/verify_region_batch` has the same unbounded chunk fan-out as `label_batch` (`vlm_labeler_verify.py` `gather`), and `resource_links.refresh_reachability` has a `gather` in a request-spawned cache refresh; both were missing from the fan-out list. They go to B4-4A |

## 22. OpenTranscribe re-read (2026-11-03, read only; the former master plan's "alignment unverified" item)

Confirmed (unchanged in the plan):
- Celery 5.6 pinned exactly (`celery==5.6.3`), `worker_prefetch_multiplier=1` globally,
  `task_create_missing_queues=False`, explicit `task_queues` from one constants class
  (`CeleryQueues` with `ALL`), `task_routes` per task, "never put `queue=` in a task decorator"
  (Celery merges the route under the decorator's options).
- `visibility_timeout=21600` with the "MUST exceed the longest task" rationale;
  `task_soft_time_limit=10800`, `task_time_limit=11700` below it.
- `worker_soft_shutdown_timeout=30`, `worker_enable_soft_shutdown_on_idle=true`,
  `worker_cancel_long_running_tasks_on_connection_loss=False`,
  `broker_connection_retry_on_startup=True`, `broker_connection_max_retries=None`,
  `worker_proc_alive_timeout=30` (issue #631), the keepalive and socket options with
  `retry_on_timeout=False` for the broker (issue #1144).
- A thread-refreshed run lease of about 90 s TTL that records which message holds it; reads fail
  safe (Redis unreachable means UNKNOWN, never dead).
- The pool healthcheck reads `inspect stats` and catches a dead pool and a respawn storm (pid set
  replaced while the task counter is frozen); it self-detects the pool type.
- Tests to mirror: `tests/unit/test_task_routes_complete.py`, `tests/unit/test_lite_mode_queue_consumers.py`.
- Redis 8 (`redis:8.2.2-alpine3.22`), Flower with basic auth under a URL prefix.

Corrected or new:
- billiard >= 4.3.1: the cold-shutdown cancel requeues instead of acking (K-2).
- Soft shutdown only on a cold shutdown; `REMAP_SIGTERM=SIGQUIT` makes `docker stop` cold (K-5).
- Beat healthcheck must glob `celerybeat-schedule*` (K-12).
- OpenTranscribe also runs an unacked-orphan reaper that restores a dead holder's message from
  kombu's `unacked` hash with `QoS.restore_by_tag`. This plan does **not** copy it: our reaper
  re-enqueues from the job records, and the stale unacked copy is a lease no-op when it surfaces
  after 6 h. Recorded as a deliberate difference (simpler; recovery waits `LEASE_TTL_S` plus one
  tick).
- A `queued` marker per run in Redis (`transcription_queued:<id>`, TTL = the queue-wait ceiling):
  adopted as `op:queued:<job_id>` (section 9.2), which fixes a gap in the task plan's reaper rule
  "re-enqueue `queued` older than 120 s" (it would re-enqueue every 30 s while a job legitimately
  waits behind four long ones).
- `worker_max_tasks_per_child` is env-driven and off by default there, with per-service CLI values
  (20, 50, 100, 500); our per-service values are in section 6.
- OpenTranscribe publishes Redis on loopback with an optional password; we keep no port and a
  required password (unchanged decision).
- Flower 2.x `/api/workers` is a one-shot snapshot at Flower start (their issue #609); do not build
  health on it.

## 23. What NOT to migrate (task plan section 5, kept)

- Detection and VLM workers: data-plane streams over the item state machine with OCC writes; one
  message per item would duplicate OpenSearch state and lose batching.
- Trainer: hours-long GPU runs through the file protocol; B1's create-only claims make a second
  trainer safe.
- Model-export worker: its `rename` claim is correct; B1 made `reconcile_claimed` stale-only.
- Per-process caches and pollers, the event-hub tail (B4 replaces its transport), the
  `ingest_writer` group commit, operator CLIs.

## 24. Former master plan content absorbed by the B2 files

| Former master plan item | Lands in |
|---|---|
| 3a last row (Celery exclusive starts) | section 9, `03a` W1-6 |
| 3b dispatcher, sweeper, cache-prune rows | `03c` 2D |
| 3c D1, D4, D5, D6, D7, D8, D9, D15 | section 11 |
| 7.2 dependencies, waves, W1-W3 rows, family recipe, jobs longer than 6 h, test cycle, done-when, size, rollback, upgrade, docs, risks | sections 1-3, 9, 14-20; `03a`, `03b`, `03c` |
| 12 T2 (B2 3 h), T4 (B2 subset) | section 14 |
| 15 M9, M10, M11, M24 | section 20 |
| 16 B2 row | section 17 |
| 17 Celery, Redis, FastAPI sources | section 25 |
| Appendix A: A.4 #19-21, A.5 #3 and #5, A.6 soft shutdown, A.7 #16, C-6 | sections 7, 14-16, 20, 22 |
| R-0.9's B2 checks | section 19 |
| N1 (epic), N11, N12 (SEC-4, SEC-5), N14 (OPS-2 box) | `03c` 2D, 2F; the issue list itself stays with the index file |

## 25. Sources

Task plan section 9 IDs: [C0]-[C9] Celery, [S1] async tasks not native in 5.x, [V1] Redis
licences, [V3] Redis persistence, [V4] Redis security, [O1], [O2] OpenSearch, [F1]-[F3] FastAPI,
[U1] uvicorn settings. Re-fetched 2026-11-03: [C1]
https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html ("Visibility
Timeout", "Key eviction", "Soft Shutdown" caveat), [C3]
https://docs.celeryq.dev/en/stable/userguide/workers.html ("Warm Shutdown", "Cold Shutdown" with
the `REMAP_SIGTERM` note, "Soft Shutdown", "Persistent revokes", "Max tasks per child"), [C4]
https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html (one scheduler; `-s` schedule
file; entry `options` such as `expires`).
New in this plan (W1 reads them before writing code):
- [C10] Celery configuration reference: https://docs.celeryq.dev/en/stable/userguide/configuration.html
- [C11] Celery signals (`worker_process_init`, `worker_process_shutdown`): https://docs.celeryq.dev/en/stable/userguide/signals.html
- [V5] Redis `SET` (`EX`, `NX`): https://redis.io/docs/latest/commands/set/
- [REDIS-EVICT] Redis eviction policies: https://redis.io/docs/latest/develop/reference/eviction/
- [P1] prometheus_client multiprocess mode (`mark_process_dead`): https://prometheus.github.io/client_python/multiprocess/
- [DH] Redis official image: https://hub.docker.com/_/redis
- [DC] Compose secrets: https://docs.docker.com/compose/how-tos/use-secrets/
- OpenTranscribe (public sibling project, read only): `backend/app/core/celery.py`,
  `backend/app/core/constants.py`, `backend/app/core/task_liveness.py`,
  `backend/app/core/broker_orphans.py`, `backend/scripts/celery_pool_healthcheck.py`,
  `docker-compose.yml` (redis, worker, beat, flower blocks), `backend/app/tasks/CLAUDE.md`,
  `backend/tests/integration/test_orphaned_delivery_recovery_live.py`.
