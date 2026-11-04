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
| `interrupted` | `failed` | reaper rule b (`attempts_exhausted`) | `attempt >= max_attempts` (review fix R-5: the trigger refused this write before) |

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

### 2.1 The kind table (`src/config/job_kinds/`, one module per lane, pure data)

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
| `project_create` (B4 4D, R-21) | NULL, scope slug | `jobs` | `project_create`, slug | yes (idempotent: each step checks what exists) | 1 h / 1 h 5 m | 1 h |
| `review_resolve_batch`, `edit_undo_batch`, `vlm_label_batch`, `vlm_verify_region_batch` (B4 4A) | yes | `jobs` | own kind, '' | no | 3 h / 3 h 15 m | 1 h |
| `gpu_arbiter` tick (R-20) | NULL | `arbiter` (periodic task, not a `jobs` row); scheduled only when `OP_GPU_ARBITER_CONTAINERS` is set, which the overlay sets (RC-3) | advisory lock | - | 120 s / 150 s | - |

Auto-resume carries the D-25 recommendation forward. Limits above 3 h carry D-20a. `max_attempts`
is 3 for every kind (D-20). A test asserts every registered task name maps to exactly one kind and
every kind routes to a queue that has a consumer in compose (section 7.3).

### 2.2 The start (one function: `src/services/jobs/service.py::start_job`)

1. In a project-scoped transaction: `INSERT INTO jobs (...) VALUES (... status='queued',
   queued_at=now(), max_queue_wait=..., exclusive_group=..., queue=...) RETURNING id`, inside a
   savepoint (`session.begin_nested()`,
   [using SAVEPOINT](https://docs.sqlalchemy.org/en/21/orm/session_transaction.html#using-savepoint)).
   A `unique_violation` aborts only the savepoint (PostgreSQL refuses every further statement of an
   aborted transaction, [SAVEPOINT](https://www.postgresql.org/docs/current/sql-savepoint.html));
   after rolling back to it, the same transaction reads the active job's id and answers 409
   `job_busy`; on `uq_jobs_idempotency` it reads and returns the existing job (section 2.3). Review
   fix R-5.
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
3. **Heartbeat thread** (`src/tasks/runtime/heartbeat.py`, review fix R-2: it runs in the GPU
   images too, which import only `src/tasks/runtime/`, `src/db`, `src/config`, `src/domain`,
   `src/utils`; the shape of B1's `LeaseHeartbeat`: its own thread and its own database connection;
   never the task's; the inline jobs of section 2.5 use the same class from the API process): every
   `HEARTBEAT_TICK_S` (15 s)

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
   factory. Clients (an async database engine with `NullPool`, and OpenSearch where the image has it)
   are built inside the loop and closed before it ends. Review fix R-2: `src/tasks/runtime/` never
   names `src.services`; the OpenSearch factory (`make_script_opensearch`) is **injected** by the
   image-specific handler module (`src/tasks/jobs_*.py` for the api image passes it to
   `register_job(kind, handler, client_factories=...)`); a GPU handler that needs OpenSearch builds
   its own from its own module. `check_layering.py` checks function-level imports too, so a lazy
   import would not pass.
5. **Progress** `ctx.progress(done, total, stage, counters)`: throttled to one write per second per
   job (and on every stage change), one transaction:
   `UPDATE jobs SET progress = :p, progress_updated_at = now() WHERE id AND fencing_token AND status IN
   ('running','cancelling') RETURNING cancel_requested_at, pause_requested_at`, plus an `events` row
   (`job.progress`, a full snapshot, not a delta) through `src/db/repos/events.py`; after the commit,
   a Valkey wake-up (section 6.2). Zero rows: `Lost`.
6. **Checkpoint** `ctx.checkpoint(cursor)`: the same fenced update on `checkpoint`. Resumable kinds
   write it after each durable chunk and resume from it.
7. **Terminal write**, fenced: `UPDATE jobs SET status = :s, result, error_code, error_message,
   finished_at = now(), owner = NULL, lease_expires_at = NULL WHERE id AND fencing_token AND status IN
   ('running','cancelling')`, the attempt's `outcome`, an `events` row (`job.state`). Outcome map:
   return -> `completed` or `completed_with_errors`; `cancelled()` or `CancelledError` -> `cancelled`;
   `SoftTimeLimitExceeded` -> `failed` (`time_limit`); any other exception -> `failed` with the
   message. Zero rows (`Lost`): write nothing, log `lease_lost`. Review fix R-24: Celery raises
   `SoftTimeLimitExceeded` from a signal handler in the main thread, which here runs the event loop;
   spike S1 item 7 checks that a handler awaiting `asyncio.sleep` and one awaiting
   `asyncio.to_thread` both reach this terminal write with the engine disposed. If either does not,
   the wrapper arms its own timer at the soft limit that sets the cancel event (a cooperative cancel
   recorded as `failed (time_limit)`), and Celery's `time_limit` stays the backstop.
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
| b | `status = 'interrupted' AND auto_resume AND updated_at < now() - interval '30 s'` and no active job of the same group (`AND NOT EXISTS (SELECT 1 FROM jobs a WHERE a.project_id IS NOT DISTINCT FROM j.project_id AND a.exclusive_group = j.exclusive_group AND a.scope = j.scope AND a.status IN (<ACTIVE>))`) | `attempt < max_attempts`: -> `queued`, dispatch after commit; else `failed` (`attempts_exhausted`). Each row's update runs in its own savepoint, so a `unique_violation` from a race skips that row (it stays `interrupted`, logged once) instead of rolling back the batch (review fix R-5); resume (section 3.3) uses the same savepoint |
| c | `status = 'queued' AND (last_dispatched_at IS NULL OR last_dispatched_at < now() - interval '120 s')` and the marker is `Absent` | dispatch again (marker `Unknown`: skip) |
| d | `status = 'queued' AND queued_at + max_queue_wait < now()` | `failed` (`queue_wait_exceeded`) |
| e | `search_projection_tasks` with `done_at IS NULL AND job_id IS NULL` | start `class_projection_sync` (idempotency key `e:<row id>`) |
| f | retention ([02a](02a_postgres_schema.md) section 9) | batched deletes, 10,000 rows per statement |
| g | `stream_worker_claims` with an expired lease | delete (the next replica claims) |

All comparisons use the database clock. Postgres down: the tick fails and retries next tick.
Valkey down: rules a, b, d, f, g still run; c and the dispatch parts of b and e wait (enqueue
raises; the row stays `queued` and rule c retries).

Busy window after a crash: a dead holder blocks a restart for at most `LEASE_TTL_S` + one tick
(150 s), the same bound as B1's PR-02 (now 150 s instead of 120 s). R-final RF-42 kills a
`jobs-worker` replica mid-job and checks this bound (rewritten in this pass: review R-23).

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
  schedules. The compose file still runs one `jobs-beat` replica. Review fix R-32: the lock
  connection sets libpq TCP keepalives and a user timeout (`keepalives=1`, `keepalives_idle=10`,
  `keepalives_interval=5`, `keepalives_count=3`, `tcp_user_timeout=30000`;
  [libpq connection parameters](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-PARAMKEYWORDS)),
  and beat checks the connection every tick, so a half-open connection ends beat within about 30 s
  and the server releases the lock when the session dies.
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
       fencing_token = CASE WHEN stream_worker_claims.worker_id <> excluded.worker_id
                            THEN stream_worker_claims.fencing_token + 1
                            ELSE stream_worker_claims.fencing_token END,  -- bump on takeover only (R-8)
       lease_expires_at = excluded.lease_expires_at
 WHERE stream_worker_claims.lease_expires_at < now() OR stream_worker_claims.worker_id = excluded.worker_id
RETURNING fencing_token;
```

Every pass (review fix R-8):
1. the replica upserts its `worker_capabilities` row (`worker_id = '<kind>@<hostname>'`, `seen_at =
   now()`), so a replica with no claim yet still counts;
2. `live` = the `worker_capabilities` rows of its kind seen within the TTL (itself included);
   `fair = ceil(active_projects / live)`;
3. it renews the claims it holds (every 15 s, TTL 120 s);
4. if it holds more than `fair`, it releases the excess, oldest claim first, at a project boundary
   (`DELETE ... WHERE worker_id = :me AND project_id = :pid`), so a newly started replica gets work
   within two passes;
5. if it holds fewer than `fair`, it claims unclaimed or expired projects up to `fair`, skipping
   paused projects (`projects.paused_at`);
6. on shutdown it releases every claim.

With one replica, behaviour equals today's round-robin over every project. Review fix R-37: the
fencing token tells a replica it lost a project; it does not fence the OpenSearch writes. Their
correctness is the per-item OCC and the lock rule (`src/clients/occ.py`, `occ_locks.py`), as today. The
in-process in-flight exclusion (`flow.py:43-119`, `vlm_worker.py:68-197`) stays as a per-replica
optimisation. The worker's own heartbeat file (`worker_liveness.py`, container-local) stays its
healthcheck.

## 6. Events and progress to the browser (D-48)

### 6.1 The choice

| Option | For | Against |
|---|---|---|
| **Postgres `events` table (the durable record and the resume cursor) + a Valkey pub/sub wake-up (chosen; owner, 2026-11-04, after review R-6)** | the job state and its feed are one store; rows are project-scoped under RLS; resume is a query; works across hosts on one database; the wake-up uses each API process's existing Valkey client, so there is no unpoolable `LISTEN` connection per process and no `NOTIFY` commit lock | a lost wake-up (Valkey down, a disconnected subscriber) delays delivery until the 2 s safety tick; never loses an event, because the reader always re-reads from its cursor |
| Postgres `events` table + `NOTIFY` / `LISTEN` (the earlier choice) | no second component on the wake-up path | one `LISTEN` session connection per API process (workers x replicas direct connections that no transaction pooler can carry: [PgBouncer feature map](https://www.pgbouncer.org/features.html)); every `NOTIFY` commit takes a database-wide lock while it queues the notification ([NOTIFY](https://www.postgresql.org/docs/current/sql-notify.html)); `NOTIFY` is a wake-up only too |
| Valkey Streams (the former B4-4C plan) | fast appends, natural ids, `MAXLEN ~` trimming | a second store for UI-critical state; no RLS; the spike and the Streams code are extra work; state and feed can disagree |

### 6.2 Write path

`src/services/events/publish.py::publish(session, project, type, payload)` (API side) and the job
runtime both insert the `events` row through `src/db/repos/events.py::insert_event` in the
**caller's** transaction (so a rolled-back change emits nothing). After that transaction commits,
the writer sends a wake-up: `PUBLISH op:events:<project_id or global> 1` on Valkey
(`src/tasks/runtime/event_wakeup.py`, importable by every image; a failed publish is logged and
ignored, because the safety tick catches up). The wake-up carries no data; the consumer reads the
rows. Payloads are snapshots (the full job progress or state), capped at 7,000 bytes by a
`CHECK`. Producers: the job runtime (state, progress
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

Each API process subscribes (redis-py async pub/sub, on its existing Valkey connection settings)
to `op:events:<project>` for every project that has subscribers in this process, plus
`op:events:global`. On a wake-up, or every 2 s as a safety tick, it runs the query above once per
such project, on a pooled connection that it returns at once, and fans the rows out to that
process's SSE clients. Reconnect with backoff; after a reconnect, the catch-up query covers the gap.
Valkey down: delivery continues at the 2 s tick. Review fix R-34: when a client resumes with a
`Last-Event-ID` older than the oldest retained row (retention 24 h), the stream sends one explicit
`reset` event first, so the client reloads its state instead of silently missing rows.

SSE routes (`GET /curation/events`, `GET /curation/projects/{project}/events`, the pipeline and
global-status streams) use FastAPI's built-in SSE (`fastapi.sse.EventSourceResponse`,
`ServerSentEvent(id=..., event=..., data=...)`; keep-alive every 15 s, `Cache-Control: no-cache`,
`X-Accel-Buffering: no`, resume through the `Last-Event-ID` header:
[server-sent events](https://fastapi.tiangolo.com/tutorial/server-sent-events/#resuming-with-last-event-id);
FastAPI 0.135.0 or newer). The SSE `id:` is `<xid>-<id>`. A reconnect replays from the cursor, then
joins the live fan-out. The two hand-written SSE framers (`events.py::sse_response`,
`pipeline_events.py::_sse`) are deleted (LIB-3 of [10](10_standards_and_libraries.md)).

Postgres down: SSE routes answer 503; `/ready` answers 503 naming `database` (D-69). The UI shows
the outage.
SSE routes never declare the per-request `DbSessionDep` (section 8): a stream holds no pooled
connection between queries (review fix R-1).

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
`REMAP_SIGTERM=SIGQUIT` on every worker service (D-20b). Review fix R-33: the worker app, the
producer app of `src/clients/task_queue.py` and `src/tasks/runtime/gpu_app.py::make_app` build
`broker_transport_options` from one function in `src/config/queue.py`, because kombu's Redis
transport restores unacked messages from any consumer with *that* consumer's visibility timeout;
`tests/tasks/test_celery_config.py` asserts the three apps carry equal options. Review fix R-19:
`jobs-worker` runs with `--max-memory-per-child=${OP_JOBS_MAX_MEMORY_PER_CHILD_KB:-8000000}`
([worker_max_memory_per_child](https://docs.celeryq.dev/en/stable/userguide/configuration.html#worker-max-memory-per-child)).
Sources:
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
| `worker_process_init` handler | `engine.dispose(close=False)` on any engine object inherited from the parent; nothing else (no I/O; the docs give 4 s before the child is killed) | [pooling with fork](https://docs.sqlalchemy.org/en/21/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork) |
| `worker_init`, `beat_init` handlers | `reject_retired_env()`; `beat_init` also takes the single-beat lock | ENV-6 of [08](08_environment_and_setup.md); D-60 |
| `worker_ready` handler | refuses to start when a registered task has no route or no kind | OpenTranscribe `_validate_task_routes` |

Package pins (owner decision 2026-11-04 after review R-10; [13](13_package_pins.md) section 4):
Celery 5.6.3 requires `kombu>=5.6.0` without a cap, and kombu 5.7.0 (released 2026-11-03) has not
been released against by any Celery version, so **kombu is held at 5.6.2** and **redis-py at 6.4.0**
(kombu 5.6.2's `redis` extra accepts `redis>=4.5.2,<6.5`, excluding 4.5.5 and 5.0.2; kombu 5.7.0's
would accept `<9.0`), until a Celery release declares support for kombu 5.7. billiard is pinned at
4.3.1 or newer (the cold-shutdown requeue). Spike S1 runs on exactly these pins, and `pip check`
passes in the api image with the `redis` extra.

### 7.2 Valkey

the Valkey image (tag `9.1.3-alpine3.24`) by digest ([13](13_package_pins.md) section 4; Valkey is BSD-3-Clause,
[COPYING](https://github.com/valkey-io/valkey/blob/unstable/COPYING); Immich runs `valkey/valkey:9`
by digest). Settings: `appendonly yes` (the default is `no`), `appendfsync everysec` (about one
second of loss, [persistence](https://valkey.io/topics/persistence/#how-durable-is-the-append-only-file)),
`maxmemory ${OP_VALKEY_MAXMEMORY:-1gb}` with `maxmemory-policy noeviction` (review fix R-19: the
policy acts only when `maxmemory` is set, [eviction](https://valkey.io/topics/lru-cache/); eviction
would break kombu's bindings, Celery Redis caveats; a full Valkey refuses writes, so starts answer
503 and the monitoring overlay alerts on `used_memory`), a password
for the `default` user from a secret file, no published port. Durability: AOF, no backup (D-24): the
reaper rebuilds the queue from the `jobs` rows.

### 7.3 Queues and consumers

| Queue | Service | Pool, concurrency | `--max-tasks-per-child` | Profile | GPU |
|---|---|---|---|---|---|
| `jobs` | `jobs-worker` | prefork, `${OP_JOBS_CONCURRENCY:-4}` (D-23) | 50 | core | `API_GPU_ID` (UMAP, cuML) |
| `periodic` | `periodic-worker` | prefork, 2 | 200 | core | none |
| `arbiter` (R-20) | `gpu-arbiter-worker` (api image, only with the GPU-arbiter overlay) | prefork, 1 | 200 | overlay | none; the only service that mounts the Docker socket |
| `open_vocab` | `open-vocab-worker` | prefork, 1 | 50 | core | none (calls the segmenter) |
| `autolabel` | `curation-auto-label-worker` | prefork, 1 | 20 | `curation` | none |
| `assist_loop` | `curation-assist-loop-worker` | prefork, 1 | 20 | `curation` | none |
| `gpu_train` | `curation-trainer` (trainer image) | prefork, 1 | 1 (a fresh child per run releases GPU memory) | `training` | `OP_TRAIN_GPU_ORDER` |
| `gpu_export` | `model-export-worker` (export image) | prefork, 1 | 1 | `export` | the export GPU |
| `gpu_eval` | `curation-evaluator` (evaluator image) | prefork, 1 | 5 | `evaluator` | `EVALUATOR_GPU_ID` |

Concurrency is capacity, never correctness: every exclusivity rule is the unique index, so any
service may run several replicas. Review RC-3: a queue without a live consumer must never receive
periodic messages. Celery's `expires` is checked by the worker that receives a message, so beat
messages to a queue nobody drains pile up in Valkey forever; beat therefore schedules the
`gpu_arbiter` tick only when `OP_GPU_ARBITER_CONTAINERS` is set (read at beat start; the arbiter
overlay sets it), and the consumers test treats overlay queues as present only under that setting.
Review RC-13: a job whose queue's profile is never started is failed by the reaper
(`queue_wait_exceeded`), but its one broker message stays in Valkey until a consumer starts;
INSTALLATION says so (small, bounded by the number of such starts). A queue whose profile is down holds its messages; the job waits
`queued` and the reaper fails it after its max queue wait (`queue_wait_exceeded`), today's behaviour
for the trigger files. `tests/test_queue_consumers_from_compose.py` asserts every routed queue has a
consumer in core or in the profile `QUEUE_PROFILE` names (OpenTranscribe's "a valid route is not a
drained route").

The trainer, export and evaluator images import only `src/tasks/runtime/` (plus `src/db`,
`src/config`, `src/domain`, `src/utils`) and their own handler module (the heartbeat, the fenced
job writes, the event row and the wake-up are all in those packages: review fix R-2); their Celery app is
`src/tasks/runtime/gpu_app.py`, built from the same `src/config/queue.py`. API-side code finds a live
consumer with `celery_app.control.inspect(timeout=1).active_queues()`
([inspecting workers](https://docs.celeryq.dev/en/stable/userguide/workers.html#inspecting-workers))
and reads `worker_capabilities`; this replaces the container-name probe and the
`.trainer_capabilities.json` file.

## 8. Connection pools per process type

SQLAlchemy 2.1 with the psycopg 3 driver (`postgresql+psycopg://`), engines created lazily per
process (never at import, never in a Celery parent). Async in the API: one `AsyncSession` per
request through a `yield` dependency declared with **`scope="function"`**:
`DbSessionDep = Annotated[AsyncSession, Depends(get_db, scope="function")]` (review fix R-1). FastAPI
runs the exit code of a `yield` dependency "after the response is sent to the client" unless the
scope is `"function"`, which ends it "after the path operation function ends, but before the
response is sent back" ([dependencies with yield, early exit and scope](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/#early-exit-and-scope);
the `scope` argument is in the signature of `fastapi.Depends` in the repository venv's FastAPI
0.143.3, and the pin is 0.162.0). The dependency only opens and closes the session; every write
commits explicitly inside `project_tx` / `deployment_tx` (`async with session.begin()`) in the
service, before the handler returns, so a failed commit becomes a 5xx and never a 2xx. SSE and
streaming routes never declare it. `expire_on_commit=False` ([preventing implicit IO](https://docs.sqlalchemy.org/en/21/orm/extensions/asyncio.html#preventing-implicit-io-when-using-asyncsession)),
never shared between concurrent tasks ([concurrent tasks](https://docs.sqlalchemy.org/en/21/orm/extensions/asyncio.html#using-asyncsession-with-concurrent-tasks)).
`pool_pre_ping=True`, `pool_recycle=1800`, `pool_use_lifo=True`
([pooling](https://docs.sqlalchemy.org/en/21/core/pooling.html)). Read the SQLAlchemy 2.1 migration
notes first ([migration_21](https://docs.sqlalchemy.org/en/21/changelog/migration_21.html), R-28).

| Process | Engine | Per process | Count (default install) | Connections |
|---|---|---|---|---|
| API worker | async, `pool_size = max(1, OP_DB_API_CONNECTIONS_TOTAL // (API_WORKERS * OP_API_REPLICAS))`, `max_overflow = 0`, `pool_timeout = 5` s mapped to 503 `database_busy` (RC-4); no `LISTEN` connection (the wake-up is on Valkey, section 6) | 2 (the 64 total over 32 processes and 1 replica) | 32 processes, 1 replica, `OP_DB_API_CONNECTIONS_TOTAL=64` | 64 |
| API worker, polls | a separate sync engine with one connection for the 1 s registry poll and the 5 s config poll (RC-4), so the polls never wait behind requests | 1 | 32 | 32 |
| API worker, inline jobs (label op, undo) | the heartbeat thread's own sync connection while an inline job runs (R-35) | 0-1 | at most one per process at a time | 32 (worst case) |
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
`tests/config/test_db_connection_budget.py` computes the table from the compose file, the settings
and **the scale factors** (`deploy.replicas` per service, `${OP_API_REPLICAS:-1}` for `api` being the
single source; review fix R-6 and RC-4) and fails when the sum exceeds `OP_PG_MAX_CONNECTIONS - 13`,
including when `workers x replicas` exceeds the total so the per-process floor of 1 takes over.
`OP_DB_API_CONNECTIONS_TOTAL` is a real deployment total: every API process takes
`floor(total / (API_WORKERS * OP_API_REPLICAS))`, at least 1, no overflow (B4-4B folds it into its
"pools sized from totals" rule; `OP_API_REPLICAS` lives in `.env`, default 1, and compose reads it as
the `api` service's `deploy.replicas`, so there is no second place to scale; no `openprocessor scale`
command, RC-4). The 4B-5 benchmark records the pool wait time.

### 8.1 More replicas or hosts: PgBouncer (D-50)

The default install has no pooler. It becomes necessary (review fix R-6: the formula, not "several
replicas") when the connection sum of section 8 with the planned scale factors exceeds
`OP_PG_MAX_CONNECTIONS - 13` and raising `max_connections` is not wanted; then an opt-in overlay
(`docker-compose.pgbouncer.yml`, written in stage 6) puts PgBouncer in transaction mode in front of
Postgres for the pooled connections. Rules that make the code ready for it now:

- no session state on pooled connections: only `set_config(..., true)` (transaction-local), only
  `pg_try_advisory_xact_lock` (session advisory locks, `LISTEN` and SQL `PREPARE` are "never" in
  transaction mode: [features](https://www.pgbouncer.org/features.html));
- the single-beat lock and the migration lock use `OP_DB_DIRECT_HOST` (default: the same host),
  never the pooler (there is no `LISTEN` any more, section 6);
- psycopg prepared statements: `prepare_threshold=None` when `OP_DB_POOLER=pgbouncer`, unless the
  pooler is 1.22+ with `max_prepared_statements > 0` and libpq 17
  ([prepared statements with PgBouncer](https://www.psycopg.org/psycopg3/docs/advanced/prepare.html#using-prepared-statements-with-pgbouncer));
- client side `NullPool` behind the pooler.
