# B2 wave 3: beat, the lease reaper, the trigger workers, hardening (tracks 2D, 2E, 2F)

| Field | Value |
|---|---|
| Status | plan, not started |
| Depends on | W1 and W2 merged into `int/b2-celery` (`03a`, `03b`) |
| Decisions applied | D-8, D-17; recommendations of D-19, D-20b, D-24, D-25 |
| Source IDs | `task_queue_plan.md` Phase 4 (beat table, deletes, keeps, tests), Phase 5 (auto-label, assist loop, liveness, tests); Q-D4, D5, D7, D8, D15; Q-E4, E5, E12, E19, E25; audit OPS-2 (act part), SEC-4, SEC-5; former master plan 3b (dispatcher, sweeper, cache-prune rows), 7.2 W3 rows |
| Branches / worktrees | `b2/beat-reaper` in `<repo>-wt-b2-beat-reaper` (2D); `b2/trigger-workers` in `<repo>-wt-b2-trigger-workers` (2E); `b2/hardening` in `<repo>-wt-b2-hardening` (2F) |
| Compose project | none in the lanes (offline renders and unit tests); live checks in the single B2 live run (`-p op-b2-live`; no GPU needed; [00_index.md](00_index.md) section 8.1) |
| Merge order | 2D, 2E, 2F (2F last: it edits every service block) |
| Size | 2D M, 2E M, 2F S |

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03.
Consolidated 2026-11-03. The reaper rules are
authoritative in `03_b2_overview.md` section 9.5; 2D implements them.

---

## 1. Track 2D (`b2/beat-reaper`): periodic work to beat, the reaper, OPS-2

### 2D-1 `src/tasks/periodic.py` and the schedule

- **Docs:** [C4] periodic tasks (one scheduler; entry `options` accept any `apply_async`
  argument, such as `expires`); [C5] routing.
- Every periodic task is a plain task (overview 9.4) on queue `periodic`, idempotent, with
  `soft_time_limit=120`, `time_limit=150`, and its beat entry has `options={'expires': <interval>}`
  so a backlog behind a stalled consumer is dropped, not replayed.
- Route every name in `TASK_ROUTES` (`periodic` block).

| Task | Every | Replaces (delete in the same commit) | Exclusion |
|---|---|---|---|
| `op.periodic.ingest_dispatch` | `OP_INGEST_DISPATCH_S` (5 s) | `bulk_ingest/dispatcher.py` `_loop`, `start_ingest_dispatcher`, `run_tick`'s `_loops` use | the `dispatch.lock` flock is replaced by a per-project B1 lease `ingest_dispatch` (TTL 60 s) acquired and released per tick; `Busy` -> skip the project this tick (former master plan 3b) |
| `op.periodic.open_vocab_sweep` | `OP_OPEN_VOCAB_SWEEP_S` | `open_vocab_sweeper.py` `_loop`, `start_open_vocab_sweeper`, the `open_vocab_sweep.lock` flock | none needed: the sweep only finds stale `pending` and enqueues `op.jobs.open_vocab_drain` (single consumer `-c 1`) |
| `op.periodic.cache_prune` | the prune interval the cache already uses (`rg -n "prune" src/services/curation/source_image_cache.py`) | the `.prune.lock` flock and the in-process prune trigger | none: the prune is idempotent; `unlink` tolerates a missing file |
| `op.periodic.gpu_arbiter` | 15 s | `lifespan._arbiter_reconcile_loop`, `_start_gpu_arbiter` (the startup reconcile also becomes this task's first run) | one beat, one message per tick |
| `op.periodic.reload_promoted` | 15 s | the reload inside the arbiter loop and `lifespan._reload_promoted_models`; keep the on-demand `reload_promoted` route | idempotent |
| `op.periodic.lease_reaper` | 30 s | `startup_reconcile.py`, every remaining `reconcile_orphaned_jobs`, `lifespan._reconcile_orphaned_state`, and `job_reconcile.reconcile_stale_running` if the dead-code ratchet agrees | rules a-d of overview 9.5 |
| `op.periodic.cluster_refresh` | `OP_CLUSTER_REFRESH_INTERVAL_S` | `scripts/curation/cluster_refresh_daemon.py` main loop and the `curation-cluster-refresh` service; the schedule logic stays in `clustering/refresh_schedule.py` and `refresh_state_store.py` | the task calls the auto-label service function directly (not the API over HTTP); B1's `autolabel` lease makes a busy project a skip |
| `op.periodic.ivf_retrain_check` | `AUTO_RETRAIN_CHECK_INTERVAL_S` | `auto_label_worker.py` `_next_retrain`, `_maybe_auto_retrain`, the in-memory `last_check` dict | the policy's existing cooldown (`should_retrain_centroids`); starts the IVF pipeline through `autolabel.start_job` (lease) |

**Consumer gate for the two auto-label starters.** `cluster_refresh` and `ivf_retrain_check` used to
run only with the `curation` profile up. In beat they run in the core install, so each first checks
that an `autolabel` consumer is live with `celery_app.control.inspect(timeout=1).active_queues()`
([C3] "Inspecting workers"); no consumer -> skip with a debug log. This keeps today's behaviour
(nothing auto-starts without the curation workers) and never queues work nothing drains (Q-E18).

### 2D-2 The lease reaper

- **Contract:** overview 9.5 rules (a) expired `running` -> takeover -> `interrupted` (+
  `auto_resume`); (b) resumable `interrupted` with no marker, older than 120 s -> re-enqueue;
  (c) `queued` with no marker, older than 120 s -> re-enqueue; (d) `queued` past its queue-wait
  TTL -> `failed` (`queue_wait_exceeded`). Redis `Unknown` -> do nothing. OpenSearch down for a
  project -> skip it.
- **Scope of the scan:** every project from `for_each_project()`, every FileJob family directory
  (use one registry of family roots, `src/services/curation/job_families.py`, built from the
  `register_job` kinds, so a new family cannot be forgotten), and the global combine directory.
- **Traps:** take over with `takeover_expired` (OCC), never by writing the record first; the
  reaper is the only writer of a record whose holder is gone; a fresh lease is never touched.
- **Red tests:** `tests/tasks/test_lease_reaper.py` (fakes): each rule a-d acts; a fresh `running`
  lease is untouched; Redis `Unknown` changes nothing; a `running` record with no lease doc and a
  fresh heartbeat is left alone (B1 mid-upgrade rule); the reaper scans the combine directory.
  `tests/tasks/test_job_families_registry.py`: every `register_job` kind has a family root.
  **Control:** a clean project with only terminal records gives no writes.

### 2D-3 Deletes and keeps

- **Delete:** `src/services/curation/background_jobs.py`; `AppResources.curation_background_jobs`
  and `arbiter_task`; `lifespan._stop_curation_jobs`, `_stop_arbiter`, `_reconcile_orphaned_state`,
  `_start_gpu_arbiter`, `_reload_promoted_models` and their shutdown steps;
  `src/services/curation/startup_reconcile.py`; the remaining `reconcile_orphaned_jobs`
  functions (`rg -n "def reconcile_orphaned_jobs" src`); the three B2 flock sites
  (`dispatcher.py`, `open_vocab_sweeper.py`, `source_image_cache.py`).
- **Keep (correction K-16):** `AppResources.ops_metrics_task` and the ops-metrics refresh in the
  API. The task plan's Phase 4 "Delete ... `ops_metrics_task`" contradicts its own "Keep: ... the
  ops-metrics refresh in the API"; the keep wins (it writes only the API's own Prometheus files).
- **Keep:** the project registry poll, the config store poll, the event tail, index bootstrap and
  warm-up (overview 5.5).
- **File-lock ratchet:** the baseline in `scripts/codegen/baselines/` shrinks to `event_hub.py` and
  `job_lock.py` only.
- **Arbiter overlay:** `docker-compose.gpu-arbiter.yml` mounts the Docker socket into
  `periodic-worker` instead of `api`; rewrite its header (the security trade-off text stays, now
  naming the worker).
- **Red tests:** `tests/test_lifespan_has_no_side_effects.py` (AST over `src/app/lifespan.py`: no
  call to the deleted functions, no `reconcile`, no arbiter); `tests/compose/test_arbiter_overlay.py`
  (the socket is mounted only into `periodic-worker`); `tests/tasks/test_beat_schedule.py` (the
  schedule has exactly the eight names, each routed to `periodic`, each with `expires`).
  **Control:** `tests/compose/test_overlay_render.py`.

### 2D-4 OPS-2 (act only on the W1-0 verdict) and worker metrics (Q-E19, D15)

- **If W1-0 said "act":** call `prometheus_client.multiprocess.mark_process_dead(os.getpid())` in
  the API lifespan shutdown (uvicorn has no parent hook) [P1]. Red test: a fake multiproc dir with
  a dead pid's live gauge files is cleaned after the shutdown step.
- **In every case, for Celery workers:** each worker service has its own `PROMETHEUS_MULTIPROC_DIR`
  tmpfs (W1-8), and a `worker_process_shutdown` handler [C11] calls `mark_process_dead` for the
  child that exits under `--max-tasks-per-child`. Metrics that a dashboard reads from job code:
  list them with `rg -n "OP_[A-Z_]+\.(labels|inc|set|observe)" src/services/curation | rg -v ingest`
  and the dashboards under the monitoring config; if one is job-only (it moved out of the API with
  the job), expose the worker's registry with `prometheus_client.start_http_server` from a
  `worker_init` handler on an internal port and add a scrape job; otherwise no endpoint.
- **Done when:** the verdict is applied and recorded in the PR.

**2D done when:** the schedule test, the reaper tests, the lifespan test and the overlay test are
green; `rg -n "exclusive_(file|start)_lock\(|fcntl\.flock" src | rg -v "job_lock.py|event_hub.py"`
is empty; `rg -n "reconcile_orphaned_jobs|startup_reconcile|background_jobs" src` is empty.

---

## 2. Track 2E (`b2/trigger-workers`): auto-label and assist loop become queue consumers

### 2E-1 Auto-label

- **Files:** `src/services/curation/autolabel/job.py` (`start_job`, `_trigger_file`,
  `QUEUE_MAX_AGE_S`, the trigger-aware busy and repair code), `scripts/curation/auto_label_worker.py`
  (`_run_one`, `_claim_trigger_for`, `_oldest_pending_trigger`, `_running_lock`, `_main_loop`,
  `_heartbeat_loop`), compose service `curation-auto-label-worker`, a new `src/tasks/autolabel.py`.
- **Steps:**
  1. `start_job` acquires B1's `autolabel` lease as `queued` (TTL `MAX_QUEUE_WAIT_S['autolabel']`),
     writes `queued`, enqueues `op.autolabel.run` with `task_id=job_id`, sets the marker. No
     `trigger.json`.
  2. Move `_run_one`'s body into the task handler (through `JobTask`; not resumable, D-25).
  3. The service command becomes `celery -A src.tasks.app worker -Q autolabel -c 1
     --max-tasks-per-child=20 --hostname auto-label@%h -E` (profile `curation` unchanged,
     `REMAP_SIGTERM=SIGQUIT`, the pool healthcheck).
  4. Delete the poll loop, `_claim_trigger_for`, `_oldest_pending_trigger`, `running.lock` (and its
     "backward-compat" comment), the trigger file handling and `QUEUE_MAX_AGE_S` (the reaper's
     rule d with `MAX_QUEUE_WAIT_S` covers it).
  5. **CC-9:** `autolabel/vlm_stage.py` `gather` over chunks -> TaskGroup + Semaphore.
  6. **Upgrade:** a leftover `trigger.json` or `running.lock` from 0.6.0 is deleted by the reaper
     when it re-enqueues that project's `queued` record (overview section 15); a unit test covers it.
- **Red tests:** `tests/curation/test_autolabel_enqueue.py` (start enqueues once, writes no trigger
  file; a second start is 409 `job_busy`); `tests/tasks/test_autolabel_task.py` (two deliveries of
  one job -> one run; a cancel flag at a stage boundary -> `cancelled`); a leftover trigger file is
  removed on re-enqueue. **Control:** the auto-label status route test.

### 2E-2 Assist loop

- **Files:** `src/services/curation/assist_loop/job.py` (`claim_trigger`, start and resume),
  `scripts/curation/assist_loop_worker.py` (`repair_orphans`, `_oldest_pending`, `run_next`,
  `_main_loop`), compose `curation-assist-loop-worker`, `src/tasks/assist_loop.py`.
- **Steps:** the same as 2E-1 with `op.assist_loop.run`, queue `assist_loop`, lease `assist_loop`;
  resumable (D-25); resume enqueues again; delete `claim_trigger`, the trigger file, the poll loop
  and `repair_orphans` (the reaper owns repair).
- **Doc sync (same commit as the auto-label and assist-loop worker changes):** update
  `docs/design/auto_label_loop_plan.md` sections 0 (D4), 1 (worker pattern), 3.1 (`start.lock`,
  `trigger.json`) and 8 (one loop per project) to the lease and the `-Q assist_loop -c 1` consumer
  (trigger files, `claim_trigger` and the poll loop no longer exist).
- **Red tests:** `tests/curation/test_assist_loop_enqueue.py`, `tests/tasks/test_assist_loop_task.py`
  (two deliveries -> one run; resume after `interrupted` re-enqueues). **Control:** the existing
  assist-loop status test.

### 2E-3 One liveness mechanism

- **Recommendation:** the compose healthcheck of both services is the pool healthcheck (W1-9);
  delete their `worker_liveness.py check ... --max-age 120` healthchecks and the asyncio heartbeat
  loops. The `auto_label_worker` heartbeat also feeds an ops gauge (`ops_metrics.WORKER_LABELS`);
  if a dashboard or alert reads it (`rg -n "auto_label_worker|worker=\"curation\"" ` over the
  monitoring config), write that heartbeat from Celery's `heartbeat_sent` worker signal [C11]
  instead of a loop; if nothing reads it, delete the label too.
- **Done when:** each service has exactly one healthcheck and at most one heartbeat writer.

**2E done when:** `rg -n "trigger\.json|claim_trigger|_claim_trigger_for|running\.lock|QUEUE_MAX_AGE_S" src scripts` is empty; both services run Celery commands; Q-D7 and Q-D8 are closed.

---

## 3. Track 2F (`b2/hardening`): SEC-4 and SEC-5 (neutral)

- **SEC-4:** reduce the privileges of an opt-in monitoring service.
- **SEC-5:** pin a build input and review per-service container hardening.
- Done here because B2 already edits compose and adds images.
- **Details stay private.** The lane reads the maintainers' notes in the owner's gitignored
  evidence directory (`artifacts_local/audit/security.md` and `security.verdicts.md` in the main
  checkout) and applies the right-sized option recorded there. Commit messages, PR text, CHANGELOG
  and docs use only the two neutral lines above.
- **In scope publicly:** `security_opt: no-new-privileges:true` on every service B2 added
  (`redis`, `jobs-worker`, `periodic-worker`, `open-vocab-worker`, `jobs-beat`, `flower`), and
  `cap_drop: ALL` where the service needs no capability (verify each starts in the sweep). No
  blanket change to the GPU, Triton or OpenSearch services in this batch.
- **Red tests:** `tests/compose/test_new_service_hardening.py` (each B2 service sets
  `no-new-privileges`); the private-note fixes carry their own neutral-named test
  (`tests/compose/test_build_input_pinned.py` style). **Control:** the overlay render test.
- **Docs:** `SECURITY.md` neutral lines; `CHANGELOG.md` `[Unreleased]` "Security" with the two
  neutral lines. The naming-leak scan and a manual read of the diff before commit (M17).
- **Done when:** the tests are green and the diff carries no security detail.

---

## 4. W3 gate, docs, upgrade, rollback, risks

- **Lane gate:** only the targeted tests above and `pre-commit run --files <touched>` (T0); then the
  coordinator runs the one B2 sweep (`03_b2_overview.md` section 14), whose live steps 4, 5, 8, 9
  and 10 exercise this wave.
- **Keep after merge, then delete:** the beat table and reaper rules go to `docs/ARCHITECTURE.md`,
  the trigger-worker description to `docs/CURATION.md`; the closing B2 PR deletes the B2 plan files
  (`03_b2_overview.md` section 17.1).
- **Docs:** `docs/ARCHITECTURE.md` (beat and the reaper), `docs/CURATION.md` (auto-resume, the
  busy window after a crash is `LEASE_TTL_S` plus one reaper tick), `CLAUDE.md` (services table: the
  cluster-refresh service removed), `docker-compose.gpu-arbiter.yml` header, `INSTALLATION.md`
  (the arbiter overlay now targets `periodic-worker`), `SECURITY.md` (2F), `CHANGELOG.md`
  `[Unreleased]` ("Removed": the cluster-refresh service and the trigger files; "Changed": the
  arbiter overlay target).
- **Upgrade impact:** the `curation-cluster-refresh` container is removed by `docker compose up
  --remove-orphans` in `openprocessor upgrade` (08 checks the installer does this); leftover trigger
  files are cleaned by the reaper; the arbiter overlay users must re-apply the overlay (it now
  targets a different service): a CHANGELOG "Breaking" line.
- **Rollback:** revert a track's merge commit on `int/b2-celery` (2F alone; 2E alone; 2D needs 2E
  and 2F reverted first because the compose blocks overlap).
- **Risks:** a periodic task overruns its interval (`expires` drops the backlog; the per-project
  dispatch lease prevents overlap); the arbiter flaps because two ticks overlap (one beat, `-c 2`
  on `periodic`: the arbiter task takes a deployment lease `gpu_arbiter` in the default project's
  settings index with TTL 30 s, or runs `-c 1` on its own queue if the 2D agent finds overlap in
  the unit test; record the choice); a dashboard loses the curation worker gauge (2E-3 check).

## 5. Ready-to-paste lane briefs

### Track 2D

```
You are the B2 wave-3 lane agent for track 2D (Sonnet). Worktree <repo>-wt-b2-beat-reaper,
branch b2/beat-reaper, cut from int/b2-celery after W2 merged. Read docs/design/program/03_b2_overview.md
(sections 5.2, 5.5, 9.5, 21), docs/design/program/03c_b2_w3_beat_reaper_workers.md section 1, the
W1 PR's OPS-2 verdict, CLAUDE.md, CONTRIBUTING.md (ratchets only shrink). Do 2D-1..2D-4, one
concern per commit. Red test on the unchanged base first, then fix and control. Lane checks only
(`pytest <files> -q --no-cov`, `pre-commit run --files <touched>`); no full suite, no --all-files;
never --no-verify, push, merge or another checkout. No Docker run, no GPU (compose project: none;
offline renders only). Do NOT use `pgrep -f`, `pkill -f`, or sleep/until loops. Deadline:
<set by the coordinator>. Report: commits, red outputs, the OPS-2 action taken, the arbiter
exclusion choice, the shrunk file-lock baseline.
```

### Track 2E

```
You are the B2 wave-3 lane agent for track 2E (Sonnet). Worktree <repo>-wt-b2-trigger-workers,
branch b2/trigger-workers, cut from int/b2-celery after W2 merged. Read docs/design/program/03_b2_overview.md
(sections 6, 9, 15), 03c section 2, CLAUDE.md. Do 2E-1..2E-3. Same rules as 2D (red first, lane
checks only, no --no-verify/push/merge/checkout, no Docker, no GPU, compose project: none, no
`pgrep -f`/`pkill -f`/sleep loops). Deadline: <set by the coordinator>. Report: commits, red
outputs, the liveness choice and what reads the heartbeat.
```

### Track 2F

```
You are the B2 wave-3 lane agent for track 2F (Sonnet). Worktree <repo>-wt-b2-hardening, branch
b2/hardening, cut from int/b2-celery after W2 merged. Read 03c section 3, then the private notes
artifacts_local/audit/security.md and security.verdicts.md in the main checkout (SEC-4, SEC-5
only). Apply the right-sized options recorded there and the public scope in 03c section 3. Never
write a security detail into any tracked file, commit message or PR text: use only the two
neutral lines. Run the naming-leak scan via `pre-commit run --files <touched>`. Same lane rules
(red first, lane checks only, no --no-verify/push/merge/checkout, no Docker, no GPU, compose
project: none, no `pgrep -f`/`pkill -f`/sleep loops). Deadline: <set by the coordinator>.
```
