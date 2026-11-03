# B2 wave 1: queue infrastructure (track `b2/queue-infra`)

| Field | Value |
|---|---|
| Status | plan, not started |
| Depends on | `03_b2_overview.md` section 2 preconditions (B1 merged; `int/b2-celery` cut from main) |
| Decisions applied | D-8 (wrapper transfers the API lease, reuses B1's heartbeat thread), D-17 Celery 5.6; recommendations of D-18, D-19, D-20, D-20a, D-20b, D-21, D-23, D-24 (stop if the owner answered otherwise) |
| Source IDs | `task_queue_plan.md` Phase 2 (S1, dependencies, new files, compose, installer, docs, tests), section 4d, section 5 rules 1-9; Q-E1-E3, E5-E11, E15, E17, E18, E21, E23-E25; audit OPS-2 (evidence part) |
| Branch / worktree | `b2/queue-infra` in `<repo>-wt-b2-queue-infra`, cut from `int/b2-celery` |
| Compose project | `-p op-b2-s1` (spike only), CPU only, no GPU |
| Size | L |

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03.
Consolidated 2026-11-03. The S1 spike (CPU only, `-p op-b2-s1`) is a spike, not one of the three
live-harness runs of [00_index.md](00_index.md) section 8.1. The shared contract (states,
lease, marker, wrapper, reaper rules) is `03_b2_overview.md` section 9; this file builds it. No job
family moves in W1.

---

## Task order

| ID | Task | Size | Depends on |
|---|---|---|---|
| W1-0 | OPS-2 evidence (offline measurement) | S | - |
| W1-1 | Spike S1 | S (1 h) | - |
| W1-2 | Dependencies, pins, licences | S | W1-1 |
| W1-3 | `src/config/queue.py` | S | W1-2 |
| W1-4 | `src/tasks/app.py` | M | W1-3 |
| W1-5 | `src/clients/task_queue.py` | S | W1-3 |
| W1-6 | `src/tasks/job_task.py` wrapper + lease extension | M | W1-4, W1-5 |
| W1-7 | queued marker helpers | S | W1-5 |
| W1-8 | compose services, dev overlay, live harness | M | W1-4 |
| W1-9 | pool and beat healthchecks | S | W1-8 |
| W1-10 | installer, CLI, secret | M | W1-8 |
| W1-11 | `/health` queue status | S | W1-5 |
| W1-12 | layering ratchet and guard tests | S | W1-4 |
| W1-13 | the Phase 2 tests | M | all above |
| W1-14 | docs | S | all above |

---

### W1-0 OPS-2 evidence (before any B2 coding acts on it)

- **Source:** audit OPS-2 (API-14), task plan D15/E19.
- **Doc:** prometheus_client multiprocess mode [P1] (`mark_process_dead`, `livemostrecent`).
- **Steps:** in a temporary `PROMETHEUS_MULTIPROC_DIR` (a `tmp_path`, no stack), simulate 1,000
  recycled processes by writing their `.db` files with the project's own metric definitions
  (`src/services/curation/ops_metrics.py`), then time `/metrics` generation
  (`prometheus_client.multiprocess.MultiProcessCollector`) at 0, 100 and 1,000 dead pids; 5 runs each.
- **Output:** the numbers and the host load (`uptime`) in the W1 PR description, plus a one-line
  verdict: "act" if generation time at 1,000 grows more than 2x the spread over the 0 baseline,
  else "no action". 2D (`03c`) reads the verdict.
- **Done when:** the verdict is in the PR. No code change in W1.

### W1-1 Spike S1 (1 h, throwaway, not committed)

- **Docs:** [C1] Redis transport, [C3] workers (shutdown sections), [C2] `acks_late`.
- **Setup:** a scratch compose under the lane's own directory with the pinned `redis:8.2.x-alpine`
  digest (`requirepass`, `appendonly yes`) and one worker built from the api image plus
  `celery[redis]==5.6.*` and `billiard>=4.3.1`; `docker compose -p op-b2-s1`.
- **Prove, and record versions and outputs in the PR:**
  1. the worker connects with the password read from a mounted file (not argv, not env);
  2. an `acks_late` task killed with SIGKILL of the **whole** container stays unacked and is
     redelivered only after a lowered test `visibility_timeout` (for example 30 s);
  3. a cold shutdown (`REMAP_SIGTERM=SIGQUIT`, `docker stop`, soft shutdown 10 s) during an
     `acks_late` task requeues the message (billiard >= 4.3.1) rather than acking it; also run it
     with billiard 4.3.0 to see the old ack (proves the pin matters);
  4. a plain SIGTERM without `REMAP_SIGTERM` is a warm shutdown that ignores the soft timeout;
  5. `revoke(task_id)` of a queued task skips it;
  6. secret ownership: the Redis start script (root) and a uid-1000 client can both read the
     password file the installer would write; record the working ownership/mode rule.
- **Done when:** all six recorded. A failure of 2, 3 or 6 stops W1 and goes to the coordinator.
- **Teardown:** `docker compose -p op-b2-s1 down -v`. Never touch another project.

### W1-2 Dependencies, pins, licences

- **Files:** `requirements.txt`, `requirements-test.txt`, the monitoring image or a Flower image
  pin, `ATTRIBUTION.md`.
- **Steps:** add `celery[redis]==<5.6.x exact>`, `billiard==<4.3.1 or newer, exact>`,
  `redis==<exact>` (match OpenTranscribe's pins where they still resolve; record the S1 versions).
  `requirements-test.txt` gets the same pins so the offline suite imports Celery. Do not add
  `requirements.txt` to the venv instructions (CLAUDE.md). ATTRIBUTION rows: Celery, kombu,
  billiard (BSD-3), redis-py (MIT), Flower (BSD), Redis 8 (AGPLv3 option) [V1].
- **Red test:** `tests/test_celery_pins.py` asserts the exact pins in both requirements files and
  `billiard>=4.3.1`; red before the edit. **Control:** the existing Ultralytics pin test passes.
- **Done when:** both green; `pip check` in the api image is part of the sweep.

### W1-3 `src/config/queue.py` (pure data and settings)

- **Why here:** the layering rule forbids the client importing `src.tasks` (K-8). Config may import
  only `config` and `utils`.
- **Contents:** `TaskQueues` constants (`JOBS='jobs'`, `PERIODIC='periodic'`,
  `OPEN_VOCAB='open_vocab'`, `AUTOLABEL='autolabel'`, `ASSIST_LOOP='assist_loop'`, and `ALL`);
  `TASK_ROUTES: dict[str, str]` (task name -> queue), filled in W1 for the W1 tasks only
  (`op.infra.noop`); `QUEUE_PROFILE` (queue -> compose profile or `None` for core);
  `MAX_QUEUE_WAIT_S` (overview section 6); `AUTO_RESUME` (D-25); `QueueSettings` reading
  `OP_QUEUE_URL`, `OP_QUEUE_PASSWORD_FILE` (validated at startup; a malformed URL or a missing or
  empty file fails closed with a named error).
- **Red test:** `tests/config/test_queue_settings.py`: a missing password file raises; a URL with
  userinfo is refused (the password must come from the file); `ALL` equals the set of
  `TASK_ROUTES` values plus declared-but-unrouted queues. **Control:** a valid file and URL load.
- **Done when:** green; `mypy` clean on the file.

### W1-4 `src/tasks/app.py` (the Celery app)

- **Docs:** [C10] configuration reference for every key; [C5] routing; [C6] security; [C7]
  prefetch; [C8]/[C3] shutdown; [C11] signals.
- **Steps:** build `celery_app = Celery('openprocessor', broker=<from QueueSettings>)` with the
  overview section 7 table exactly; `task_queues` from `TaskQueues.ALL`; `task_routes` from
  `TASK_ROUTES`; the socket keepalive dict built only from the `socket` attributes the platform
  has (copy OpenTranscribe's rationale comment, attributed); a `worker_ready` hook that refuses to
  start if a registered task has no route (mirror OpenTranscribe `_validate_task_routes`);
  `include=` lists the task modules (W1: `src.tasks.infra` with `op.infra.noop`).
- **Rules:** nothing at module scope opens a socket, imports torch, cuML, Triton or the PE
  encoder (Q-E10); `worker_process_init` does no I/O (OpenTranscribe #631).
- **Red tests:** `tests/tasks/test_celery_config.py` asserts `visibility_timeout >
  task_time_limit > task_soft_time_limit`, `worker_prefetch_multiplier == 1`, `accept_content ==
  ['json']`, `task_create_missing_queues is False`, `task_acks_late`, `task_reject_on_worker_lost`,
  `task_ignore_result`, `worker_soft_shutdown_timeout == 30`,
  `worker_enable_soft_shutdown_on_idle is True`,
  `worker_cancel_long_running_tasks_on_connection_loss is False`,
  `broker_connection_max_retries is None`, the same `visibility_timeout` in both places.
  `tests/tasks/test_no_cuda_import.py`: importing `src.tasks.app` in a fresh interpreter leaves
  `torch`, `cuml`, `tritonclient` out of `sys.modules`. **Control:** `src.tasks.app` imports.
- **Done when:** green.

### W1-5 `src/clients/task_queue.py`

- **API:** `enqueue(task_name, *, project, job_id) -> None` (queue from `TASK_ROUTES` only; no
  queue argument; `task_id=job_id`; `send_task` by name, so services never import task modules);
  `revoke(job_id) -> None` (no `terminate`); `ping() -> bool` (bounded timeout). Raises
  `QueueUnavailable` on broker errors.
- **Producer app:** a minimal `Celery` instance built from `src/config/queue.py` (same broker
  options), not an import of `src.tasks.app` (K-8).
- **Red test:** `tests/clients/test_task_queue.py` with a fake producer: `enqueue` sends exactly one
  message with `task_id == job_id` and the routed queue; an unknown task name raises before any
  send; a broker error raises `QueueUnavailable`. **Control:** `ping` on a reachable fake.
- **Done when:** green.

### W1-6 Task wrapper `src/tasks/job_task.py` and the lease extension

- **Contract:** overview section 9.1-9.3 and 9.6. Implement exactly those steps.
- **Files:** `src/tasks/job_task.py` (`JobTask` base and a `register_job(kind, handler, *,
  resumable, soft_time_limit=None, time_limit=None)` helper used by W2), `src/clients/job_lease.py`
  and `src/domain/job_lease.py` (B1's signatures per `02a_b1_job_leases.md`:
  `acquire(client, kind, scope, job_id, owner, ttl_s=LEASE_TTL_S)` already takes `ttl_s`; add a
  keyword `state='running'` that the start path passes as `'queued'`;
  `transfer(client, lease, new_owner)` gains keywords `state` and `ttl_s`),
  `src/services/curation/lease_heartbeat.py` (`LeaseHeartbeat`, with its `FENCE_MARGIN_S`
  self-fence) **unchanged**.
- **Steps:**
  1. bind the project (fail closed), build clients inside `asyncio.run`;
  2. terminal or not-resumable `interrupted` -> ack, no-op;
  3. `transfer` to `running`; a live lease held by another owner -> log `lease_busy`, ack;
     `attempt >= 3` -> `failed`;
  4. write `running`, `attempt += 1`, delete the queued marker (W1-7);
  5. start the B1 heartbeat thread; `asyncio.run(handler)`;
  6. `fence` before the terminal write; map outcomes (overview 9.3 step 8);
  7. on `SystemExit` (soft-shutdown cancel): `fence`, write `interrupted` with
     `auto_resume = kind in AUTO_RESUME`, release, re-raise;
  8. `finally`: stop the heartbeat thread.
- **Per-kind time limits:** `register_job` passes `soft_time_limit`/`time_limit` to the task
  options; W2 sets 24 h / 24 h 15 m for the D-20a kinds.
- **Traps:** the heartbeat thread must not share the task loop's client (B1 rule); a prefork child
  must not inherit a running thread (start it in the task, never in `worker_process_init`); never
  read the lease with `search` (realtime `get` only).
- **Red tests:** `tests/tasks/test_job_task_wrapper.py` (unit, mocks, as [C9] advises; never
  `task_always_eager`):
  - claims and runs once; a live lease -> no-op ack; a terminal record -> no-op;
  - `attempt` reaching 3 -> `failed`; an exception -> `failed` with the message; cancel flag ->
    `cancelled`; `SoftTimeLimitExceeded` -> `failed` (`time_limit`);
  - the heartbeat thread keeps renewing while the handler blocks the loop past `LEASE_TTL_S`
    (monkeypatched to 2 s);
  - a redelivery while the first delivery holds a live lease is a no-op (Q-E1, with a lowered
    visibility timeout simulated by two deliveries);
  - `SystemExit` during the handler writes `interrupted` with the right `auto_resume` and
    re-raises;
  - a `Lost` fence writes nothing.
  `tests/test_job_lease_queued_state.py`: `acquire(state='queued', ttl_s=X)` then
  `transfer(state='running', ttl_s=120)` gives one doc with the new owner and TTL; a second
  `acquire` while queued is `Busy`; an expired queued lease is acquirable.
  **Control:** B1's sequential acquire, release, acquire still gives generations 1 and 2.
- **Done when:** green; red output recorded in the PR.

### W1-7 Queued marker helpers

- **Doc:** Redis `SET` with `EX` [V5]; OpenTranscribe `task_liveness.py` (queued marker, fail-safe
  reads).
- **Files:** `src/clients/task_queue.py` (`mark_queued(job_id, ttl_s)`, `clear_queued(job_id)`,
  `queued_state(job_id) -> Present | Absent | Unknown`).
- **Rule:** Redis unreachable -> `Unknown`; callers never act on `Unknown`.
- **Red test:** `tests/clients/test_queued_marker.py`: set, read `Present`, TTL honoured, clear,
  `Absent`; a connection error gives `Unknown`. **Control:** two job ids are independent.
- **Done when:** green.

### W1-8 Compose services, dev overlay, live harness

- **Docs:** [C3] (shutdown, `--max-tasks-per-child`), [C4] (one beat, `-s` file), [C1] (eviction),
  [V3], [V4], [DH], [DC]; OpenTranscribe `docker-compose.yml` worker, beat and redis blocks.
- **`docker-compose.yml` (deploy-safe: no `build:`, no source mounts):**
  - `redis`: overview section 8; start through a small wrapper that writes `requirepass` into a
    tmpfs config from the secret file, then execs the image entrypoint; no `ports:`;
    `restart: unless-stopped` (match the file's convention); volume `openprocessor-queue`.
  - `jobs-worker`: api image; `celery -A src.tasks.app worker -Q jobs
    --concurrency=${OP_JOBS_CONCURRENCY:-4} --max-tasks-per-child=50 --hostname jobs@%h -E`;
    the same volumes and env as `api` (copy the block, minus ports), plus
    `OP_API_LOOPBACK_URL=http://api:8000`, `REMAP_SIGTERM=SIGQUIT`; the `API_GPU_ID` reservation;
    `stop_grace_period: ${OP_STOP_GRACE_JOBS:-60}s`; its own `PROMETHEUS_MULTIPROC_DIR` tmpfs
    (never the API's, Q-E19).
  - `periodic-worker`: api image; `-Q periodic -c 2 --max-tasks-per-child=200 --hostname
    periodic@%h -E`; no GPU; `stop_grace_period: 30s`.
  - `open-vocab-worker`: api image; `-Q open_vocab -c 1 --max-tasks-per-child=50 --hostname
    open-vocab@%h -E`.
  - `jobs-beat`: api image; `celery -A src.tasks.app beat -s /var/lib/openprocessor/celerybeat-schedule`;
    a fixed `container_name`; no `deploy.replicas`.
  - `flower`: `monitoring` profile; basic auth from the secret; port per D-19.
  - `depends_on: redis: condition: service_healthy` for `api`, every worker and `jobs-beat`.
  - **No job is routed in W1** except `op.infra.noop`; the services start and idle.
- **`docker-compose.dev.yml`:** source mounts for the new services; a comment that Celery does not
  hot-reload (`docker compose -p <name> restart jobs-worker` after task edits, Q-E21).
- **`docker/test/compose.yml` (live harness):** add `redis`, `jobs-worker-a`, `jobs-worker-b`
  (two containers: `container_name` blocks `--scale`), `periodic-worker`, `open-vocab-worker`,
  `jobs-beat`; the harness keeps its own project name.
- **Red tests (`tests/compose/`, offline render):**
  - `test_queue_services.py`: `redis` has no `ports`, has `noeviction` and `appendonly`, reads a
    password file; every worker has `REMAP_SIGTERM=SIGQUIT` and a `stop_grace_period` at least 30 s
    above the soft timeout; no worker shares the API's `PROMETHEUS_MULTIPROC_DIR`;
  - `test_beat_single_instance.py`: `jobs-beat` has a `container_name` and no `deploy.replicas`
    (Q-E5).
  **Control:** `tests/compose/test_overlay_render.py` still passes.
- **Done when:** green; `docker compose config` renders for the default, dev and arbiter overlays
  (offline render only in the lane).

### W1-9 Pool and beat healthchecks

- **Source:** OpenTranscribe `backend/scripts/celery_pool_healthcheck.py` (dead pool; respawn
  storm = pid set replaced while `stats["total"]` is frozen; one prior sample persisted; pool type
  read from `pool.implementation`). Port it to `scripts/healthcheck/celery_pool_healthcheck.py`
  with an attribution header (AGPL to AGPL; public sibling project).
- **Compose:** each worker's healthcheck runs it against `<role>@$$HOSTNAME`; `jobs-beat`'s
  healthcheck takes the newest mtime across `celerybeat-schedule*` under 300 s and fails on an
  empty glob (K-12).
- **Note in the compose comment:** nothing acts on `unhealthy`; `restart:` fires only when the
  main process exits (OpenTranscribe lesson).
- **Red tests:** `tests/healthcheck/test_celery_pool_healthcheck.py` (fake `inspect stats`
  replies): empty `processes` -> unhealthy; pid set replaced with a frozen counter on two probes ->
  unhealthy; churn with an advancing counter -> healthy; a threads pool -> healthy on a reply.
  `tests/compose/test_beat_healthcheck.py`: the command globs `celerybeat-schedule*` and handles an
  empty glob. **Control:** a steady pool is healthy.
- **Done when:** green.

### W1-10 Installer, CLI, secret

- **Files:** `setup-openprocessor.sh`, `scripts/setup.sh`, `openprocessor`, `scripts/lib/cli_*.sh`,
  `scripts/lib/bind_dirs.sh`, `env.template` (08 owns the table; this task adds the keys).
- **Steps:** create `secrets/queue/redis_password` once (umask 077, random 32 bytes, the S1
  ownership rule) on fresh install **and** on `upgrade` and `repair` when missing (M11); the same
  for the Flower auth file if D-19 publishes Flower; `openprocessor status/health` list `redis`,
  `jobs-worker`, `periodic-worker`, `open-vocab-worker`, `jobs-beat`; `openprocessor uninstall`
  removes the queue volume with the others (respecting the existing secrets confirmation);
  `config show` redacts the password file contents (never print it).
- **Red tests:** `tests/installer/test_queue_secret.py`: upgrade on a dir without the secret creates
  it with mode 600; a rerun keeps the existing value; `repair` recreates a deleted one.
  `tests/installer/test_cli_status_lists_queue_services.py`. **Control:** the existing installer
  rerun test passes.
- **Done when:** green.

### W1-11 `/health` queue status

- **Files:** `src/routers/health.py` (and the curation health/status routes that list services, if
  they list them: `rg -n "resource_links|services" src/routers/curation/health.py`).
- **Rule:** `queue: ok | down`, degraded but not fatal for inference routes; the probe is
  `task_queue.ping()` with a short timeout. B4-4A owns the existing two-probe fan-out in
  `health.py`; W1 adds one probe without adding a new `gather`.
- **Red test:** `tests/test_health_queue.py`: a down fake broker gives `queue: down` and HTTP 200
  with a degraded status. **Control:** an up broker gives `ok`.
- **Done when:** green; `make contracts` if the health model changed.

### W1-12 Layering ratchet and guard tests

- **Files:** `scripts/codegen/check_layering.py` (add contracts: nothing under `src/` except
  `src/tasks/` itself, and nothing under `scripts/`, may import `src.tasks`; `src/tasks` must not
  import `src.routers`, `fastapi` or `starlette`); `tests/test_no_revoke_terminate.py`
  (AST scan of `src/` and `scripts/`: no `terminate=` keyword in any `revoke` call, Q-E7).
- **Red test:** a temp module under `src/services/` importing `src.tasks` makes the layering check
  fail; a `revoke(..., terminate=True)` fixture fails the guard. **Control:** the real tree passes.
- **Done when:** green.

### W1-13 Phase 2 tests

- `tests/tasks/test_task_routes_complete.py`: every registered task has a `TASK_ROUTES` entry; no
  task decorator passes `queue=` (mirrors OpenTranscribe `tests/unit/test_task_routes_complete.py`).
- `tests/test_queue_consumers_from_compose.py`: parse `docker-compose.yml`; for every queue a route
  targets, a `-Q` consumer exists in the core services or in the profile `QUEUE_PROFILE` names
  (mirrors OpenTranscribe `tests/unit/test_lite_mode_queue_consumers.py`, Q-E18). Red: a route to
  a queue with no consumer fails.
- `tests/live/test_queue_infra.py` (live harness only, runs in the sweep): enqueue
  `op.infra.noop` and see it run on `jobs-worker-a` or `-b`.

### W1-14 Docs

- `docs/ARCHITECTURE.md` (services table, a queues table, "Workers, jobs and events"), `CLAUDE.md`
  (services table rows, the Celery conventions block), `INSTALLATION.md` (Redis, secret ownership,
  single host / local volumes only, Q-E15), `SECURITY.md` (broker preconditions: internal network,
  required password, JSON only, whoever reaches the broker can enqueue work [C6]), `docs-site`
  (same pages), `CHANGELOG.md` `[Unreleased]` "Added" lines.
- Run `.venv/bin/python scripts/docs/check_docs_vs_code.py --only <the touched docs>` in the lane.

---

## Gate (lane level)

Only the targeted red/green tests above and static checks on touched files:
`.venv/bin/python -m pytest <the W1 test files> -q --no-cov` and
`.venv/bin/pre-commit run --files <touched files>`. The full suite, `--all-files` and the live
harness run once in the B2 sweep (`03_b2_overview.md` section 14); that single live run covers B1
and B2 together. What to keep after merge and the deletion of this file: `03_b2_overview.md`
section 17.1.

## Upgrade impact, rollback, risks

- Upgrade: W1 adds the services and the secret; no job moves, so nothing is interrupted by W1 alone.
- Rollback: revert the W1 merge commit on `int/b2-celery` (W2 and W3 depend on it, so they revert
  first).
- Risks: secret ownership (S1 item 6); Python 3.13 `shelve` WAL (K-12); a platform without
  `TCP_USER_TIMEOUT` (built only from present attributes).

## Ready-to-paste lane brief (W1)

```
You are the B2 wave-1 lane agent (Sonnet). Repo worktree: <repo>-wt-b2-queue-infra, branch b2/queue-infra,
cut from int/b2-celery. Read, in order: docs/design/program/03_b2_overview.md (sections 2, 6-9,
18), docs/design/program/03a_b2_w1_queue_infra.md (this task list), CLAUDE.md, CONTRIBUTING.md,
docs/design/engineering_standards.md. Execute W1-0 to W1-14 in order.

Rules:
- Read the cited Celery, Redis and Docker docs before each task; cite them in commit bodies.
- One commit per task (conventional commits); run each task's red test on the unchanged base
  first and paste the failure in the commit body or PR notes; then the fix and the control.
- Lane checks only: the task's test files with `.venv/bin/python -m pytest <files> -q --no-cov`
  and `.venv/bin/pre-commit run --files <touched files>`. No full suite, no --all-files.
- Never --no-verify, never push, never merge, never check out another branch.
- Docker: only `docker compose -p op-b2-s1` for the spike, CPU only, no GPU; `down -v` after.
  Never touch another compose project, container, volume or network.
- Waiting: healthchecks or the Monitor tool only. Do NOT use `pgrep -f`, `pkill -f`, or
  sleep/until loops.
- Do not read .env or any secret file contents. Do not edit B1's lease_heartbeat.py.
- Stop and report if S1 items 2, 3 or 6 fail, or if a B1 precondition in overview section 2 is
  missing.
Deadline: <set by the coordinator at kick-off>. Final report: commits, red outputs, S1 versions
and results, the W1-0 OPS-2 verdict, anything you could not do.
```
