# 02b: jobs, locking, cancel, the reaper, events, queues and pools

| Field | Value |
|---|---|
| Status | plan, not started |
| Part of | [02_control_plane_database.md](02_control_plane_database.md); tables in [02a_postgres_schema.md](02a_postgres_schema.md) section 5 |
| Built by | B2 stage 1 (Celery app, queues, producer), stage 3 (job runtime, reaper, events), used by stages 4-6 |
| Replaces | the shared contract of the former `03_b2_overview.md` section 9 (job files + OpenSearch lease + queued marker), B1's lease documents and heartbeat, the JSONL event hub |

---

## 1. The job state machine

| From | To | Who | Condition |
|---|---|---|---|
| (new) | `queued` | job service `start` | insert; the partial unique index `uq_jobs_one_active` refuses a second active job of the same exclusive group and scope |
| (new) | `running` | job service `run_inline` | inline kinds only (section 2.5) |
| `queued` | `running` | task wrapper `claim` | `cancel_requested_at IS NULL` and `attempt < max_attempts` |
| `queued` | `cancelled` | cancel route | - |
| `queued` | `failed` | reaper (`queue_wait_exceeded`), start (`queue_unavailable`), wrapper (`project_unavailable`, `attempts_exhausted`) | - |
| `running` | `completed`, `completed_with_errors`, `failed` | wrapper, fenced | the handler returned or raised |
| `running` | `cancelling` | cancel route | - |
| `running` | `paused` | wrapper, fenced | `pause_requested_at` set and the handler stopped at a boundary (assist loop only) |
| `running`, `cancelling` | `interrupted` | wrapper on worker shutdown (fenced); reaper when the lease expired | - |
| `cancelling` | `cancelled` | wrapper, fenced; reaper when the lease expired | - |
| `paused`, `interrupted` | `queued` | resume route; reaper for `auto_resume` kinds | the unique index again guards exclusivity |
| `paused`, `interrupted` | `cancelled` | cancel route | - |

Terminal: `completed`, `completed_with_errors`, `failed`, `cancelled`. Active (counted by the
singleton index): `queued`, `running`, `paused`, `cancelling`. `interrupted` is not active: a new
start is allowed and the interrupted job can still be resumed only if no other job of its group is
active (the index decides).

The table lives in `src/domain/job_states.py` (pure data: `TRANSITIONS`, `TERMINAL`, `ACTIVE`). The
database trigger `jobs_status_transition_guard` is generated from it, and
`tests/db/test_job_transitions.py` asserts every listed transition passes and every other pair
raises. Existing family status vocabularies (`waiting_region`, `exporting`, `paused_backpressure`,
`undoing`, the promote phases) become `progress.stage` values, not statuses; the wire models keep
their family-specific fields through the one serializer, and every status union gains `queued`
where it lacks it (contracts per family, stage 4).

## 2. Starting a job

### 2.1 The kind table (`src/config/job_kinds.py`, pure data)

| Kind | Project | Queue | Exclusive group, scope | Auto-resume | Soft / hard limit | Max queue wait |
|---|---|---|---|---|---|---|
| `reprocess` | yes | `jobs` | `reprocess`, '' | yes | 24 h / 24 h 15 m | 24 h |
| `assist_run` | yes | `jobs` | `assist_run`, '' | no | 3 h / 3 h 15 m | 24 h |
| `item_scores`, `probe`, `selection`, `embedding_viz` | yes | `jobs` | own kind, '' | no | 3 h / 3 h 15 m | 24 h |
| `region_cluster`, `region_fp_centroids` | yes | `jobs` | own kind, '' | no | 3 h / 3 h 15 m | 24 h |
| `dataset_import`, `dataset_import_undo` | yes | `jobs` | `dataset_import`, '' | import yes, undo no | 24 h / 24 h 15 m | 24 h |
| `dataset_upload_extract` | yes | `jobs` | `dataset_upload`, upload id | no | 3 h / 3 h 15 m | 24 h |
| `combine` | target | `jobs` | `combine`, '' | yes | 24 h / 24 h 15 m | 24 h |
| `bulk_ingest_run` | yes | `jobs` | `bulk_ingest`, '' | yes | 24 h / 24 h 15 m | 24 h |
| `source_poll` | yes | `jobs` | `source_poll`, source id | no (the next tick polls) | 1 h / 1 h 5 m | 1 h |
| `open_vocab_drain` | yes | `open_vocab` | `open_vocab_drain`, '' | no (the sweep re-drains) | 3 h / 3 h 15 m | 1 h |
| `class_merge`, `class_projection_sync` | yes | `jobs` | `class_registry_write`, '' | sync yes | 3 h / 3 h 15 m | 24 h |
| `promote` | yes | `jobs` | `promote`, Triton model name | no | 3 h / 3 h 15 m | 24 h |
| `autolabel` | yes | `autolabel` | `autolabel`, '' | no | 3 h / 3 h 15 m | 1 h |
| `assist_loop` | yes | `assist_loop` | `assist_loop`, '' | yes | 24 h / 24 h 15 m | 1 h |
| `train` | yes | `gpu_train` | none | no | 72 h / 72 h 15 m | 7 d |
| `model_export` | either | `gpu_export` | `model_export`, Triton model name | no | 4 h / 4 h 15 m | 24 h |
| `bakeoff` | yes | `gpu_eval` | none | no | 24 h / 24 h 15 m | 24 h |
| `project_delete` | NULL, scope slug | `jobs` | `project_delete`, slug | yes (idempotent) | 3 h / 3 h 15 m | 24 h |
| `directory_ingest` | NULL | `jobs` | none | no | 24 h / 24 h 15 m | 24 h |
| `persons_cluster` | NULL | `jobs` | `persons_cluster`, '' | no | 3 h / 3 h 15 m | 24 h |
| `label_op`, `label_undo` | yes | inline | `label_op`, op id; `label_undo`, '' | no | request-bound | - |

Auto-resume carries the D-25 recommendation forward. Limits above 3 h carry D-20a. `max_attempts`
is 3 for every kind (D-20). A test asserts every registered task name maps to exactly one kind and
every kind routes to a queue that has a consumer in compose (section 7.3).

### 2.2 The start (one function: `src/services/jobs/service.py::start_job`)

1. In a project-scoped transaction: `INSERT INTO jobs (...) VALUES (... status='queued',
   queued_at=now(), max_queue_wait=..., exclusive_group=..., queue=...) RETURNING id`. A
   `unique_violation` on `uq_jobs_one_active` answers 409 `job_busy` with the active job's id (read
   in the same transaction); on `uq_jobs_idempotency` it returns the existing job (section 2.3).
   Domain rows that belong to the job (`dataset_imports`, `combine_runs`, `training_runs`,
   `job_items`) are inserted in the same transaction. Postgres down: 503 `database_unavailable`.
2. Commit.
3. `task_queue.enqueue(task_name, job_id=id)`: `send_task` by name with `task_id = id`, the queue from
   the route table only. Then `SET op:queued:<id> 1 EX <max_queue_wait>` and
   `UPDATE jobs SET dispatch_count = dispatch_count + 1, last_dispatched_at = now()`.
4. Broker error at step 3: `UPDATE jobs SET status='failed', error_code='queue_unavailable',
   finished_at=now() WHERE id = :id AND status = 'queued'`, answer 503 `queue_unavailable`.
5. A crash between steps 2 and 3 leaves a `queued` row with `last_dispatched_at IS NULL`: the reaper
   dispatches it (section 4, rule c). The message is sent only after the commit, so a worker never
   receives a job id it cannot read.

Routes answer 202 with the job id and the existing per-family status route. The wire carries the
`queued` status.

### 2.3 Idempotency (D-58)

Start routes accept an optional `Idempotency-Key` header (8-200 characters). It is stored on the
row; a repeat within the retention window with the same key, project and kind returns 200 with the
first job (a different body with the same key answers 422 `idempotency_key_reused`, checked by
comparing `params`). Internal starters use it too: beat ticks use `<task>:<project>:<slot>` and the
reaper's creates use `<rule>:<row id>`, so a duplicate tick or a re-run of a rule never makes a
second job.

### 2.4 Who may start what

Every start goes through `start_job`; a ratchet (`tests/test_jobs_single_entry.py`, AST scan) fails
on any `INSERT INTO jobs` or `jobs` model construction outside `src/db/repos/jobs.py`, and on any
`send_task` outside `src/clients/task_queue.py`.

### 2.5 Inline jobs

`label_op` and `label_undo` stay request-bound (today they run in the request under B1's
`LABEL_OP`/`LABEL_UNDO` leases, `label_ops/record.py:39`, `undo.py:192-201`).
`start_inline_job` inserts the row directly as `running` with the API process as owner, runs the
heartbeat thread for its duration, and writes the terminal state fenced. A crash leaves an expired
lease that the reaper marks `interrupted`, which the label-op routes already serve as
`interrupted` / `undo_interrupted`.

## 3. Running a job (`src/tasks/runtime/job_task.py`)

### 3.1 The wrapper, step by step

1. Read `job_id` from the message (the only argument). Open a project-scoped session for the job's
   project (a NULL project uses deployment scope; the row is read first under deployment scope with
   only `SELECT project_id, kind`).
2. **Claim**, one statement:

   ```sql
   UPDATE jobs SET status = 'running', attempt = attempt + 1, fencing_token = fencing_token + 1,
          owner = :owner, heartbeat_at = now(), lease_expires_at = now() + :ttl,
          started_at = coalesce(started_at, now()), updated_at = now()
    WHERE id = :id AND status = 'queued' AND cancel_requested_at IS NULL AND attempt < max_attempts
   RETURNING fencing_token, attempt, params, checkpoint;
   ```

   Zero rows: read the row. Terminal, `running` (a redelivery while the first delivery is alive),
   `cancelling`, `paused` or `interrupted`: ack and return (log `claim_skipped` with the status).
   `queued` with `attempt >= max_attempts`: write `failed` (`attempts_exhausted`) and ack. Project
   missing or archived: `failed` (`project_unavailable`), ack. Insert the `job_attempts` row in the
   same transaction as the claim. `DEL op:queued:<id>` after the commit.
3. **Heartbeat thread** (`src/services/jobs/heartbeat.py`, the shape of B1's `LeaseHeartbeat`: its
   own thread and its own database connection; never the task's): every `HEARTBEAT_TICK_S` (15 s)

   ```sql
   UPDATE jobs SET heartbeat_at = now(), lease_expires_at = now() + :ttl
    WHERE id = :id AND fencing_token = :tok AND status IN ('running','cancelling')
   RETURNING status, cancel_requested_at, pause_requested_at;
   ```

   Zero rows: `Lost`; the stop event is set. A returned cancel or pause request sets the matching
   event. `may_write()` (kept from B1) answers false once the last good beat is older than
   `LEASE_TTL_S - FENCE_MARGIN_S` (120 - 30 s), so a holder that lost its lease stops OpenSearch writes
   before a reaper can take over. Constants move from `src/domain/job_lease.py` to
   `src/domain/job_timing.py` (`LEASE_TTL_S = 120`, `HEARTBEAT_TICK_S = 15`, `FENCE_MARGIN_S = 30`).
4. **Run** `asyncio.run(handler(ctx))`. `ctx` carries the job id, project, params, checkpoint, the
   fencing token, `progress()`, `checkpoint()`, `cancelled()`, `paused()`, `may_write()` and a client
   factory. Clients (OpenSearch through `make_script_opensearch`, an async database engine with
   `NullPool`) are built inside the loop and closed before it ends.
5. **Progress** `ctx.progress(done, total, stage, counters)`: throttled to one write per second per
   job (and on every stage change), one transaction:
   `UPDATE jobs SET progress = :p, progress_updated_at = now() WHERE id AND fencing_token AND status IN
   ('running','cancelling') RETURNING cancel_requested_at, pause_requested_at`, plus an `events` row
   (`job.progress`, a full snapshot, not a delta) and `pg_notify`. Zero rows: `Lost`.
6. **Checkpoint** `ctx.checkpoint(cursor)`: the same fenced update on `checkpoint`. Resumable kinds
   write it after each durable chunk and resume from it.
7. **Terminal write**, fenced: `UPDATE jobs SET status = :s, result, error_code, error_message,
   finished_at = now(), owner = NULL, lease_expires_at = NULL WHERE id AND fencing_token AND status IN
   ('running','cancelling')`, the attempt's `outcome`, an `events` row (`job.state`). Outcome map:
   return -> `completed` or `completed_with_errors`; `cancelled()` or `CancelledError` -> `cancelled`;
   `SoftTimeLimitExceeded` -> `failed` (`time_limit`); any other exception -> `failed` with the
   message. Zero rows (`Lost`): write nothing, log `lease_lost`.
8. **Worker shutdown**: the soft-shutdown cancel raises `SystemExit` in the child; the wrapper
   writes `interrupted` fenced (`auto_resume` from the kind table), the attempt outcome
   `interrupted`, and re-raises so Celery requeues the message (billiard behaviour verified in spike
   S1, [03a](03a_b2_w1_queue_infra.md)). The requeued message then finds `interrupted` and acks; the
   reaper is what re-queues a resumable job (rule b), so recovery does not depend on the requeue.
9. `finally`: stop the heartbeat thread, dispose the per-task engine.

Fencing is enforced by the database on every write (`fencing_token` in the `WHERE`), which closes
B1's residual "one write after a lost lease" window for job state (risk M4). OpenSearch writes
remain guarded by `may_write()` and the OCC rule.

### 3.2 Why the queued marker stays in Valkey

`op:queued:<id>` (TTL = the kind's max queue wait) shares fate with the broker message: if the
Valkey volume is lost, both disappear, and the reaper's rule c sees a `queued` row without a marker
and dispatches again. A Postgres flag could not tell "waiting behind four long jobs" from "message
lost". Valkey unreachable makes the marker state `Unknown`, and nothing acts on `Unknown`
(OpenTranscribe `task_liveness.py`: UNKNOWN, never dead). Duplicate messages are harmless: the claim
admits one.

### 3.3 Cancel and pause

| Status | Cancel route effect |
|---|---|
| `queued`, `paused`, `interrupted` | `cancelled` at once (`finished_at = now()`); best-effort `revoke(job_id)` (never `terminate=True`: "you must never call this programmatically", [workers guide](https://docs.celeryq.dev/en/stable/userguide/workers.html#revoking-tasks)); `DEL` the marker |
| `running` | `cancelling`, `cancel_requested_at = now()`; the handler sees it on its next progress write (within about 1 s) or heartbeat (15 s) and stops; the wrapper writes `cancelled` |
| `cancelling` | no-op 202 |
| terminal | 409 `job_finished` (existing per-family behaviour where a family already answers differently is kept and listed in stage 4) |

One statement does it:
`UPDATE jobs SET cancel_requested_at = now(), cancel_requested_by = :by, status = CASE WHEN status IN
('queued','paused','interrupted') THEN 'cancelled' ELSE 'cancelling' END, finished_at = CASE WHEN
status IN ('queued','paused','interrupted') THEN now() END WHERE id = :id AND status IN
('queued','running','paused','interrupted','cancelling') RETURNING status`.
Revoked ids are held in worker memory only ([persistent revokes](https://docs.celeryq.dev/en/stable/userguide/workers.html#persistent-revokes));
the row's status is the truth, which is why the claim checks `status = 'queued'`.

Pause (assist loop only): `pause_requested_at = now()`; the handler stops at its next boundary and
the wrapper writes `paused`. Resume: `UPDATE ... SET status = 'queued', pause_requested_at = NULL,
queued_at = now() WHERE id AND status IN ('paused','interrupted')`, then dispatch.

## 4. The reaper and stuck-job recovery

`op.periodic.reaper` (beat, every 30 s, queue `periodic`). Each rule is its own short transaction
under deployment scope; each scans with `FOR UPDATE SKIP LOCKED LIMIT 200`
([the locking clause](https://www.postgresql.org/docs/current/sql-select.html#SQL-FOR-UPDATE-SHARE):
suited to "multiple consumers accessing a queue-like table"), so two reapers never act on one row;
the tick also takes `pg_try_advisory_xact_lock(REAPER, 0)` and skips when another tick holds it.

| Rule | Selects | Action |
|---|---|---|
| a | `status IN ('running','cancelling') AND lease_expires_at < now()` | `cancelling` -> `cancelled`; `running` -> `interrupted`; attempt outcome `lost`; event |
| b | `status = 'interrupted' AND auto_resume AND updated_at < now() - interval '30 s'` | `attempt < max_attempts`: -> `queued`, dispatch after commit; else `failed` (`attempts_exhausted`) |
| c | `status = 'queued' AND (last_dispatched_at IS NULL OR last_dispatched_at < now() - interval '120 s')` and the marker is `Absent` | dispatch again (marker `Unknown`: skip) |
| d | `status = 'queued' AND queued_at + max_queue_wait < now()` | `failed` (`queue_wait_exceeded`) |
| e | `search_projection_tasks` with `done_at IS NULL AND job_id IS NULL` | start `class_projection_sync` (idempotency key `e:<row id>`) |
| f | retention ([02a](02a_postgres_schema.md) section 9) | batched deletes, 10,000 rows per statement |
| g | `stream_worker_claims` with an expired lease | delete (the next replica claims) |

All comparisons use the database clock. Postgres down: the tick fails and retries next tick.
Valkey down: rules a, b, d, f, g still run; c and the dispatch parts of b and e wait (enqueue
raises; the row stays `queued` and rule c retries).

Busy window after a crash: a dead holder blocks a restart for at most `LEASE_TTL_S` + one tick
(150 s), the same bound as B1's PR-02 (now 150 s instead of 120 s; R-final RF-42 is updated).

### 4.1 Periodic critical sections and the single beat (D-60)

- Each beat-driven task body that must not overlap itself takes a transaction-level advisory lock:
  `SELECT pg_try_advisory_xact_lock(:class, :obj)`, `class` from `src/config/advisory_locks.py`
  (`REAPER = 16`, `INGEST_DISPATCH = 10`, `OPEN_VOCAB_SWEEP = 11`, `CACHE_PRUNE = 12`,
  `GPU_ARBITER = 13`, `CLUSTER_REFRESH = 14`, `IVF_RETRAIN = 15`), `obj` the project's `lock_key`
  or 0. False: skip this tick. Transaction-level locks release at commit or rollback and work behind a
  transaction pooler ([advisory locks](https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS)).
  A lock taken in a transaction is held while that transaction does its (short) database work; a
  body that must call out (the arbiter's Docker calls) takes the lock, records its intent, commits,
  acts, and records the outcome in a second transaction, guarded by `gpu_reservations` rows.
- **One beat**: the Celery docs require "only a single scheduler is running for a schedule at a time"
  ([periodic tasks](https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html#introduction)).
  A `beat_init` signal handler ([signals](https://docs.celeryq.dev/en/stable/userguide/signals.html))
  opens a dedicated direct connection and runs `SELECT pg_try_advisory_lock(BEAT = 1, 0)` (session
  level); false: log `beat_already_running` and exit 3. The connection stays open for beat's life; if
  it drops, beat exits and compose restarts it. A second beat on another host therefore never
  schedules. The compose file still runs one `jobs-beat` replica.
- The beat schedule file is ephemeral (a tmpfs path): losing it only resets "last run" times of
  interval schedules.

## 5. Stream workers (detection and VLM) under replicas (D-53)

They stay stream workers over `region_status` and the VLM verdict fields (one Celery message per
item would duplicate OpenSearch state and lose batching). Each replica claims whole projects:

```sql
INSERT INTO stream_worker_claims (worker_kind, project_id, worker_id, lease_expires_at)
VALUES (:kind, :pid, :me, now() + :ttl)
ON CONFLICT (worker_kind, project_id) DO UPDATE
   SET worker_id = excluded.worker_id,
       fencing_token = stream_worker_claims.fencing_token + 1,
       lease_expires_at = excluded.lease_expires_at
 WHERE stream_worker_claims.lease_expires_at < now() OR stream_worker_claims.worker_id = excluded.worker_id
RETURNING fencing_token;
```

A replica renews its claims every 15 s (TTL 120 s), claims at most
`ceil(active_projects / live_replicas)` new projects per pass (live replicas = distinct `worker_id`
with a live claim, plus itself), skips paused projects (`projects.paused_at`), and releases its
claims on shutdown. With one replica, behaviour equals today's round-robin over every project. The
in-process in-flight exclusion (`flow.py:43-119`, `vlm_worker.py:68-197`) stays as a per-replica
optimisation. The worker's own heartbeat file (`worker_liveness.py`, container-local) stays its
healthcheck.

## 6. Events and progress to the browser (D-48)

### 6.1 The choice

| Option | For | Against |
|---|---|---|
| **Postgres `events` table + `NOTIFY` (chosen)** | the job state and its feed are one store; rows are project-scoped under RLS; resume is a query; works across hosts on one database; no new component on the UI path | `NOTIFY` is a wake-up only (not durable, 8,000-byte payload, delivered at commit: [NOTIFY](https://www.postgresql.org/docs/current/sql-notify.html)); `LISTEN` needs a session connection, not a transaction pooler ([PgBouncer feature map](https://www.pgbouncer.org/features.html)); insert load bounded by throttling |
| Valkey Streams (the former B4-4C plan) | fast appends, natural ids, `MAXLEN ~` trimming | a second store for UI-critical state; no RLS; the spike and the Streams code are extra work; state and feed can disagree |
| Valkey pub/sub | trivial | not resumable |

### 6.2 Write path

`src/services/events/publish.py::publish(session, project, type, payload)` inserts an `events` row
and runs `SELECT pg_notify('op_events', :channel_payload)` in the **caller's** transaction (so a
rolled-back change emits nothing, and the notify arrives after the commit). The notify payload is
only `<project_id or 'global'>`; the consumer reads the rows. Payloads are snapshots (the full job
progress or state), capped at 7,000 bytes by a `CHECK`. Producers: the job runtime (state, progress
throttled to 1/s per job), lifecycle routes (project, config, VLM changes), combine, and the
detection worker's pipeline stats (coalesced to at most 1/s per project, the existing
`_stats_inflight` coalescing in `pipeline_events.py:73`).

### 6.3 Read path and commit-safe resume

An identity column is assigned at insert, not at commit, so two concurrent writers can commit ids
out of order; a reader that resumes from `id > last` could skip a row that commits late. The cursor
is therefore `(xid, id)` with `xid xid8 DEFAULT pg_current_xact_id()`, and a reader emits only rows
whose transaction is older than every running one:

```sql
SELECT xid, id, type, payload FROM events
 WHERE project_id = :p AND (xid, id) > (:last_xid, :last_id)
   AND xid < pg_snapshot_xmin(pg_current_snapshot())
 ORDER BY xid, id LIMIT 500;
```

Every transaction below the snapshot's `xmin` has finished, so no later row can sort before the
cursor ([transaction ID and snapshot functions](https://www.postgresql.org/docs/current/functions-info.html#FUNCTIONS-PG-SNAPSHOT)).
The cost: a long write transaction anywhere delays the feed until it ends, which rule 4 of
[02](02_control_plane_database.md) section 2 forbids anyway (trap register T-14).

Each API process holds one `LISTEN op_events` connection (psycopg async, autocommit, direct DSN
`OP_DB_DIRECT_HOST`). On a notification, or every 2 s as a safety tick, it runs the query above once
per project that has subscribers in this process and fans the rows out to that process's SSE
clients. Reconnect with backoff; after a reconnect, the catch-up query covers the gap.

SSE routes (`GET /curation/events`, `GET /curation/projects/{project}/events`, the pipeline and
global-status streams) use FastAPI's built-in SSE (`fastapi.sse.EventSourceResponse`,
`ServerSentEvent(id=..., event=..., data=...)`; keep-alive every 15 s, `Cache-Control: no-cache`,
`X-Accel-Buffering: no`, resume through the `Last-Event-ID` header:
[server-sent events](https://fastapi.tiangolo.com/tutorial/server-sent-events/#resuming-with-last-event-id);
FastAPI 0.135.0 or newer). The SSE `id:` is `<xid>-<id>`. A reconnect replays from the cursor, then
joins the live fan-out. The two hand-written SSE framers (`events.py::sse_response`,
`pipeline_events.py::_sse`) are deleted (LIB-3 of [10](10_standards_and_libraries.md)).

Postgres down: SSE routes answer 503; `/health` reports `database: down`. The UI shows the outage.

## 7. Celery and Valkey configuration

### 7.1 Celery (`src/tasks/app.py`; the producer in `src/clients/task_queue.py` builds its own minimal app from `src/config/queue.py`)

Carried unchanged from the former 03 section 7 (it was checked against the Celery 5.6 docs and
OpenTranscribe): `task_queues` from `TaskQueues.ALL`, `task_create_missing_queues=False`,
`task_routes` from the route table only (never `queue=` in a decorator), JSON only,
`task_acks_late=True`, `task_reject_on_worker_lost=True`, `worker_prefetch_multiplier=1`,
`broker_transport_options.visibility_timeout=21600` (6 h), the socket keepalive and timeout options
with `retry_on_timeout=False`, `broker_connection_retry_on_startup=True`,
`broker_connection_max_retries=None`, default `task_soft_time_limit=10800` and `task_time_limit=11700`
with per-kind overrides from the kind table, `worker_proc_alive_timeout=30`,
`worker_soft_shutdown_timeout=30`, `worker_enable_soft_shutdown_on_idle=True`,
`worker_cancel_long_running_tasks_on_connection_loss=False`, task events on for Flower, UTC, and
`REMAP_SIGTERM=SIGQUIT` on every worker service (D-20b). Sources:
[configuration](https://docs.celeryq.dev/en/stable/userguide/configuration.html),
[Redis broker, visibility timeout and caveats](https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html#redis-visibility-timeout),
[workers: shutdown](https://docs.celeryq.dev/en/stable/userguide/workers.html),
[routing](https://docs.celeryq.dev/en/stable/userguide/routing.html).

Changed:

| Key | Value | Why |
|---|---|---|
| `broker_url` | `OP_QUEUE_URL` (default `redis://valkey:6379/0`) plus the password read from `OP_QUEUE_PASSWORD_FILE` | kombu has no `valkey://` scheme: issue [#2245](https://github.com/celery/kombu/issues/2245) is open and PR #2246 targets kombu 5.8; Valkey speaks the protocol of Redis OSS 7.2 ([migration](https://valkey.io/topics/migration/)), so the documented `redis://` transport and redis-py are used. No third-party transport |
| `result_backend` | unset; `task_ignore_result=True`; `task_track_started=False` | D-49: the `jobs` row is the only truth. A database result backend would add `celery_taskmeta` rows that duplicate status without project scope, progress or fencing ([database backend settings](https://docs.celeryq.dev/en/stable/userguide/configuration.html#conf-database-result-backend)) |
| `worker_deduplicate_successful_tasks` | not set | it needs a persistent result backend; the claim de-duplicates instead |
| `worker_process_init` handler | `engine.dispose(close=False)` on any engine object inherited from the parent; nothing else (no I/O; the docs give 4 s before the child is killed) | [pooling with fork](https://docs.sqlalchemy.org/en/20/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork) |
| `worker_init`, `beat_init` handlers | `reject_retired_env()`; `beat_init` also takes the single-beat lock | ENV-6 of [08](08_environment_and_setup.md); D-60 |
| `worker_ready` handler | refuses to start when a registered task has no route or no kind | OpenTranscribe `_validate_task_routes` |

Package traps for `13_package_pins.md` (verified 2026-11-04): Celery 5.6.3
requires `kombu>=5.6.0` without a cap and kombu 5.7.0 was released on 2026-11-03, so kombu is pinned
exactly; kombu 5.6.x accepts redis-py `<6.5` and kombu 5.7.x `<8.0`, while redis-py 8.1 is the
newest, so redis-py is pinned to what the chosen kombu accepts; billiard is pinned at 4.3.1 or newer
(the cold-shutdown requeue). Spike S1 runs on exactly these pins.

### 7.2 Valkey

`valkey/valkey:9.x-alpine` by digest (Valkey is BSD-3-Clause,
[COPYING](https://github.com/valkey-io/valkey/blob/unstable/COPYING); Immich runs `valkey/valkey:9`
by digest). Settings: `appendonly yes` (the default is `no`), `appendfsync everysec` (about one
second of loss, [persistence](https://valkey.io/topics/persistence/#how-durable-is-the-append-only-file)),
`maxmemory-policy noeviction` (eviction breaks kombu's bindings, Celery Redis caveats), a password
for the `default` user from a secret file, no published port. Durability: AOF, no backup (D-24): the
reaper rebuilds the queue from the `jobs` rows.

### 7.3 Queues and consumers

| Queue | Service | Pool, concurrency | `--max-tasks-per-child` | Profile | GPU |
|---|---|---|---|---|---|
| `jobs` | `jobs-worker` | prefork, `${OP_JOBS_CONCURRENCY:-4}` (D-23) | 50 | core | `API_GPU_ID` (UMAP, cuML) |
| `periodic` | `periodic-worker` | prefork, 2 | 200 | core | none |
| `open_vocab` | `open-vocab-worker` | prefork, 1 | 50 | core | none (calls the segmenter) |
| `autolabel` | `curation-auto-label-worker` | prefork, 1 | 20 | `curation` | none |
| `assist_loop` | `curation-assist-loop-worker` | prefork, 1 | 20 | `curation` | none |
| `gpu_train` | `curation-trainer` (trainer image) | prefork, 1 | 1 (a fresh child per run releases GPU memory) | `training` | `OP_TRAIN_GPU_ORDER` |
| `gpu_export` | `model-export-worker` (export image) | prefork, 1 | 1 | `export` | the export GPU |
| `gpu_eval` | `curation-evaluator` (evaluator image) | prefork, 1 | 5 | `evaluator` | `EVALUATOR_GPU_ID` |

Concurrency is capacity, never correctness: every exclusivity rule is the unique index, so any
service may run several replicas. A queue whose profile is down holds its messages; the job waits
`queued` and the reaper fails it after its max queue wait (`queue_wait_exceeded`), today's behaviour
for the trigger files. `tests/test_queue_consumers_from_compose.py` asserts every routed queue has a
consumer in core or in the profile `QUEUE_PROFILE` names (OpenTranscribe's "a valid route is not a
drained route").

The trainer, export and evaluator images import only `src/tasks/runtime/` (plus `src/db`,
`src/config`, `src/domain`, `src/utils`) and their own handler module; their Celery app is
`src/tasks/runtime/gpu_app.py`, built from the same `src/config/queue.py`. API-side code finds a live
consumer with `celery_app.control.inspect(timeout=1).active_queues()`
([inspecting workers](https://docs.celeryq.dev/en/stable/userguide/workers.html#inspecting-workers))
and reads `worker_capabilities`; this replaces the container-name probe and the
`.trainer_capabilities.json` file.

## 8. Connection pools per process type

SQLAlchemy 2.x with the psycopg 3 driver (`postgresql+psycopg://`), engines created lazily per
process (never at import, never in a Celery parent). Async in the API: one `AsyncSession` per
request through a `yield` dependency ([FastAPI SQL databases](https://fastapi.tiangolo.com/tutorial/sql-databases/#create-a-session-dependency),
[dependencies with yield](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/)),
`expire_on_commit=False` ([preventing implicit IO](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html#preventing-implicit-io-when-using-asyncsession)),
never shared between concurrent tasks ([concurrent tasks](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html#using-asyncsession-with-concurrent-tasks)).
`pool_pre_ping=True`, `pool_recycle=1800`, `pool_use_lifo=True`
([pooling](https://docs.sqlalchemy.org/en/20/core/pooling.html)).

| Process | Engine | Per process | Count (default install) | Connections |
|---|---|---|---|---|
| API worker | async, `pool_size = max(2, OP_DB_API_CONNECTIONS_TOTAL // API_WORKERS)`, `max_overflow = 1`; plus 1 `LISTEN` connection | 3 + 1 | 32 processes (`API_WORKERS`) with `OP_DB_API_CONNECTIONS_TOTAL=64` | 128 |
| Celery child (job wrapper) | sync, `pool_size=2` (wrapper + heartbeat thread), `max_overflow=0`; plus a per-task async engine with `NullPool` for handler queries (at most 2 at once) | 4 | 4 + 2 + 1 + 1 + 1 children in core and curation; + 3 GPU children | 48 |
| Celery parent, Flower | none | 0 | - | 0 |
| beat | 1 direct connection (single-beat lock) + sync `pool_size=1` | 2 | 1 | 2 |
| detection and VLM stream workers | sync, `pool_size=2` | 2 | 2 | 4 |
| `db-migrate`, backup | 1 direct, exits | 1 | - | 2 |
| reserve (maintenance, `psql`) | | | | 10 |
| `superuser_reserved_connections` | | | | 3 |
| **Total** | | | | **197** |

`postgres` runs with `max_connections=${OP_PG_MAX_CONNECTIONS:-300}` (default 100 is too low for 32 API
processes; it needs a restart to change:
[connection settings](https://www.postgresql.org/docs/current/runtime-config-connection.html#GUC-MAX-CONNECTIONS)).
`tests/config/test_db_connection_budget.py` computes the table from the compose file and the
settings and fails when the sum exceeds `OP_PG_MAX_CONNECTIONS - 13`. B4-4B folds
`OP_DB_API_CONNECTIONS_TOTAL` into its "pools sized from totals" rule.

### 8.1 More replicas or hosts: PgBouncer (D-50)

The default install has no pooler. For several API replicas or hosts, an opt-in overlay
(`docker-compose.pgbouncer.yml`, written in stage 6) puts PgBouncer in transaction mode in front of
Postgres for the pooled connections. Rules that make the code ready for it now:

- no session state on pooled connections: only `set_config(..., true)` (transaction-local), only
  `pg_try_advisory_xact_lock` (session advisory locks, `LISTEN` and SQL `PREPARE` are "never" in
  transaction mode: [features](https://www.pgbouncer.org/features.html));
- `LISTEN`, the single-beat lock and the migration lock use `OP_DB_DIRECT_HOST` (default: the same
  host), never the pooler;
- psycopg prepared statements: `prepare_threshold=None` when `OP_DB_POOLER=pgbouncer`, unless the
  pooler is 1.22+ with `max_prepared_statements > 0` and libpq 17
  ([prepared statements with PgBouncer](https://www.psycopg.org/psycopg3/docs/advanced/prepare.html#using-prepared-statements-with-pgbouncer));
- client side `NullPool` behind the pooler.
