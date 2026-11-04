# B2 stage 6: beat and the periodic tasks, the deletions, scale-out compose, hardening, the docs

| Field | Value |
|---|---|
| Status | plan, not started |
| Depends on | stages 4 and 5 merged ([03d](03d_stage4_job_families.md), [03e](03e_stage5_gpu_queues.md)) |
| Design | [02b](02b_jobs_locking_events.md) section 4.1 (advisory locks, single beat), [02c](02c_migration_ops_tests.md) section 4 (services) |
| Lanes | 6A, 6B and ES-C in parallel; then `b2/frontend-spaces` (after ES-C); then 6C; then 6D |
| Merge order | 6A, 6B, ES-C, `b2/frontend-spaces`, 6C, 6D |
| Research seams | folded 2026-11-04, approved by the owner 2026-11-04 (RD-1; [00d](00d_owner_questions.md)): 6A gains the serving inventory, the cluster-run refresh and the near-duplicate ticks and loses the IVF retrain check; 6B the per-card Triton shape and the DCGM profiling fields; 6D the research docs; new lanes ES-C (readers to the embedding registry) and `b2/frontend-spaces`; the five stale Cropwright screenshots are recaptured in the B2 sweep ([03](03_b2_overview.md) section 9, step 23) |
| Compose project | none in the lanes (offline renders); the B2 sweep follows this stage |
| Decisions | D-19, D-20b, D-24, D-50, D-54, D-60, D-61; SEC-4, SEC-5 (neutral) |

---

## Lane 6A `b2/beat-periodic`: periodic work to beat, and the old infrastructure deleted

**Owns:** `src/tasks/periodic.py`, `src/tasks/beat_schedule.py`, `src/config/task_routes/periodic.py`;
`src/services/curation/bulk_ingest/dispatcher.py` (the tick loop), `src/services/curation/open_vocab_sweeper.py`,
`src/services/curation/source_image_cache.py`, `src/services/curation/ingest_index.py` (the inline
prune call), `scripts/curation/cluster_refresh_daemon.py` (deleted), `src/services/curation/clustering/refresh_schedule.py`,
`src/services/training/{gpu_arbiter_policy,pause_sentinel}.py`, `src/app/lifespan.py`,
`src/app/resources.py`, `src/services/curation/{background_jobs,startup_reconcile,job_reconcile,file_job,lease_heartbeat,job_lock}.py`
(deleted), `src/clients/job_lease.py` and `src/domain/job_lease.py` (deleted),
`scripts/codegen/check_file_locks.py` and `scripts/codegen/baselines/file_locks.txt`,
`src/services/curation/ops_metrics*.py` (OPS-2 act part only), tests under `tests/periodic/`,
`tests/test_lifespan_has_no_side_effects.py`.

### 6A-1 The beat schedule

Every periodic task is a plain task on queue `periodic`, idempotent, `soft_time_limit=120`,
`time_limit=150`, its beat entry with `options={'expires': <interval>}` (a backlog behind a stalled
consumer is dropped, not replayed: [periodic tasks](https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html);
the worker that receives the message checks `expires`, so this holds only for a queue with a live
consumer, which is why the `gpu_arbiter` entry exists only with the overlay, RC-3; lane 6A's red test
`tests/tasks/test_beat_schedule_arbiter_conditional.py`: without `OP_GPU_ARBITER_CONTAINERS` the
schedule has no `gpu_arbiter` entry, with it the entry is present),
and the advisory lock of [02b](02b_jobs_locking_events.md) section 4.1.

| Task | Every | Replaces (deleted in the same commit) | Lock |
|---|---|---|---|
| `op.periodic.reaper` | 30 s | (3A, already present) | `REAPER, 0` |
| `op.periodic.ingest_dispatch` | `OP_INGEST_DISPATCH_S` (5 s) | `dispatcher.py` `_loop`, `start_ingest_dispatcher`, the `dispatch.lock` flock | `INGEST_DISPATCH, <project lock_key>`; busy -> skip the project |
| `op.periodic.open_vocab_sweep` | `OP_OPEN_VOCAB_SWEEP_S` (120 s) | `open_vocab_sweeper.py` `_loop`, the `open_vocab_sweep.lock` flock | `OPEN_VOCAB_SWEEP, <project>`; starts `open_vocab_drain` (coalesced by the unique index) |
| `op.periodic.cache_prune` | the cache's interval | the `.prune.lock` flock and the inline prune from ingest | `CACHE_PRUNE, 0` |
| `op.periodic.gpu_arbiter` | 15 s, scheduled only when `OP_GPU_ARBITER_CONTAINERS` is set (the overlay sets it; RC-3) | `lifespan._arbiter_reconcile_loop`, `_start_gpu_arbiter`; `training_gpus.lock` and `pause.sentinel` become `gpu_reservations` and `deployment_settings('vlm_paused_for_training')` | `GPU_ARBITER, 0`; intent and outcome in two transactions around the Docker calls; routed to the dedicated `arbiter` queue, consumed only by `gpu-arbiter-worker` (review fix R-20), so no other periodic task runs in a process that holds the Docker socket |
| `op.periodic.reload_promoted` | 15 s | the reload in the arbiter loop and `lifespan._reload_promoted_models` (the on-demand route stays) | none (idempotent) |
| `op.periodic.cluster_refresh` | `OP_CLUSTER_REFRESH_INTERVAL_S` | `cluster_refresh_daemon.py` and its service; the schedule logic stays in `refresh_schedule.py` over `project_state('cluster_refresh')` (the beat's clock) and the serving partition's `n_items` | `CLUSTER_REFRESH, <project>`; research clustering 5.2: starts `cluster_run` (not `autolabel`) through `start_job` when the profile's growth rule fires and ingest is quiet, only when a `cluster` consumer is live (a fresh `worker_capabilities` row naming the `cluster` queue, never a broadcast `inspect()`; review RR-21), else skips |
| (deleted) `op.periodic.ivf_retrain_check` | - | the auto-label worker's `_next_retrain` and `last_check` are deleted with the worker in 4D; the IVF retrain is a `cluster_run` (research clustering 5.2), so this tick and the `IVF_RETRAIN` advisory lock do not exist | - |
| `op.periodic.serving_inventory` (research GPU S-14) | 60 s | the hand-copied serving facts | none (idempotent upsert); reads each server of the active serving profile (`/v2/repository/index`, model readiness, metrics) and upserts `serving_reports`, and upserts a `gpu_devices` row with a fresh `seen_at` for every card the server's metrics name (`gpu_uuid` label; review RR-9) |
| `op.periodic.near_dup_rescore` (research near-duplicates 5.2) | 30 s | the request-time grouping | `NEAR_DUP_RESCORE, <project lock_key>` (its own advisory-lock class, 02b; review RR-21); starts `near_dup_rescore` per (project, level) with pending rows, idempotency key `near_dup_rescore:<project>:<level>:<slot>` |
| `op.periodic.near_dup_full_build` | weekly | - | per project; a full `image` build (cheap at image level) to bound incremental drift (risk R4) |
| `op.periodic.near_dup_prune` | daily | - | deletes near-duplicate runs beyond the current and two superseded per project and level (02a section 9) |
| `op.periodic.db_backup` | `OP_DB_BACKUP_INTERVAL_H` if set | - | `pg_try_advisory_xact_lock` on a backup class; off by default (D-61) |

The two auto-label starters ran only with the `curation` profile up; the consumer gate keeps that
behaviour and never queues work nothing drains.

### 6A-2 Single beat (D-60)

`beat_init` handler: the session advisory lock on a direct connection; exit 3 when held; exit when
the connection drops; the lock connection carries TCP keepalives and `tcp_user_timeout` (review fix
R-32, [02b](02b_jobs_locking_events.md) section 4.1). Red: `tests/periodic/test_single_beat_lock.py`
(`db`): a second beat process exits with `beat_already_running`; killing the first lets a restarted
second one take the lock; the lock connection's DSN carries the keepalive parameters.

### 6A-3 Deletions (the old job infrastructure, now without callers)

`file_job.py`, `lease_heartbeat.py`, `job_lock.py`, `startup_reconcile.py`, `background_jobs.py`,
`job_reconcile.py` (if the dead-code ratchet agrees), `src/clients/job_lease.py`,
`src/domain/job_lease.py`, the lifespan steps `_stop_curation_jobs`, `_stop_arbiter`,
`_reconcile_orphaned_state`, `_start_gpu_arbiter`, `_reload_promoted_models`, the `AppResources`
fields `curation_background_jobs` and `arbiter_task`. The ops-metrics refresh stays (it writes only
its own Prometheus files). The file-lock baseline becomes empty and `check_file_locks.py` asserts
zero (the baseline file is deleted).

Red tests: `tests/test_lifespan_has_no_side_effects.py` (AST over `src/app/lifespan.py`: none of the
deleted calls, no `reconcile`, no arbiter); `tests/periodic/test_beat_schedule.py` (exactly the names
above, each routed to `periodic`, each with `expires`; B4 lane 4A adds `op.periodic.resource_reachability`
and extends this test in the same commit, [05](05_b4_stateless_api.md) 4A-5); `tests/codegen/test_file_locks_zero.py`
(a planted `fcntl.flock` fails); the done-when `rg` of [03_b2_overview.md](03_b2_overview.md)
section 1.

### 6A-4 OPS-2 and worker metrics

If 1B-0 said "act": `mark_process_dead(os.getpid())` in the API lifespan shutdown. In every case:
each Celery service has its own `PROMETHEUS_MULTIPROC_DIR` (6B), and a `worker_process_shutdown`
handler calls `mark_process_dead` for a child leaving under `--max-tasks-per-child`
([multiprocess mode](https://prometheus.github.io/client_python/multiprocess/)). A job-only metric
a dashboard reads is exposed with `start_http_server` from `worker_init` on an internal port, with a
scrape target (6B).

## Lane 6B `b2/scale-out-compose`

**Owns (research additions: 6B-7, 6B-8 below):** `docker-compose.yml`, `docker-compose.dev.yml`, `docker-compose.gpu-arbiter.yml`, new
`docker-compose.pgbouncer.yml`, `docker/test/compose.yml`, `monitoring/prometheus.yml`,
`monitoring/alloy-config.alloy`, `monitoring/dashboards/**` (review fix R-12), the shell
helpers that address containers by name (`rg -ln "COMPOSE_PROJECT_NAME.*-(api|triton|trainer|vlm|opensearch)" scripts/lib openprocessor`),
`src/services/labeling/vlm_url_policy.py` (own-container recognition), `tests/compose/**`
(new tests), `tests/installer/test_container_lookup_by_label.py`.

| ID | Task | Red test | Control |
|---|---|---|---|
| 6B-1 | the stage-4 and stage-5 service changes from the PR notes: `curation-auto-label-worker`, `curation-assist-loop-worker`, `curation-trainer`, `model-export-worker`, `curation-evaluator` run their Celery consumers with the pool healthcheck, `REMAP_SIGTERM=SIGQUIT` and a `stop_grace_period` above the soft timeout; `curation-cluster-refresh` deleted; every `/jobs` mount and the `openprocessor-jobs` volume deleted; `OP_API_LOOPBACK_URL=http://api:8000` on `jobs-worker` | `tests/compose/test_consumers_after_b2.py`: each queue of `TaskQueues.ALL` has exactly the service of [02b](02b_jobs_locking_events.md) section 7.3, with `arbiter` required only in the render that includes the arbiter overlay (RC-3); no service mounts `/jobs`; `tests/test_queue_consumers_from_compose.py` | `tests/compose/test_overlay_render.py` |
| 6B-2 | D-54: no service has a `container_name`; scripts and the CLI find containers through Compose (`docker compose -p <p> ps -q <service>`) or the labels `com.docker.compose.project` and `com.docker.compose.service` ([compose labels](https://docs.docker.com/reference/compose-file/services/#labels)); the GPU arbiter's `OP_GPU_ARBITER_CONTAINERS` names services, resolved by label | `tests/compose/test_no_container_name.py`; `tests/installer/test_container_lookup_by_label.py` (fake docker output) | the CLI status test |
| 6B-3 | scale: `docker compose up --scale api=2 --scale jobs-worker=2` renders; `jobs-beat` stays one replica (`deploy.replicas: 1`); the live harness scales `jobs-worker` instead of declaring two services | `tests/compose/test_scale_render.py` (offline `docker compose config` with the scale flags) | - |
| 6B-4 | the GPU-arbiter overlay adds the one-process `gpu-arbiter-worker` (queue `arbiter` only) and mounts the Docker socket into it alone; `periodic-worker` never gets the socket (review fix R-20); its header rewritten (the trade-off text stays, naming the worker) | `tests/compose/test_arbiter_overlay.py`: only `gpu-arbiter-worker` mounts the socket and it consumes only `arbiter` | - |
| 6B-5 | `docker-compose.pgbouncer.yml` (D-50, opt-in): PgBouncer in transaction mode for the pooled DSN, `OP_DB_POOLER=pgbouncer`, `OP_DB_DIRECT_HOST=postgres` kept for the beat and migration locks; the header gives the formula of [02b](02b_jobs_locking_events.md) section 8.1 for when it becomes required (R-6) | `tests/compose/test_pgbouncer_overlay.py`: the API's pooled host is the pooler and its direct host is Postgres | - |
| 6B-6 | monitoring for replicas and the new services (review fix R-12; the static `api:8000` target sees one replica per scrape and mixes their counters): Prometheus uses `dns_sd_configs` (type A) on each scaled service name, which Docker's embedded DNS answers with every replica, so each replica is its own `instance` (chosen over `docker_sd_configs`, which would need the Docker socket in Prometheus); Alloy keeps logs by the Compose `service` label for every Python service, `postgres` and `valkey`; the dashboards select and aggregate by `service` (or `job`), never by `container` or `instance`, because a `dns_sd` `instance` is a container IP that changes when the container is recreated (RC-8), and Prometheus joins the stack's Compose network so the names resolve; an alert on Valkey `used_memory` near `maxmemory` (R-19) | `tests/compose/test_monitoring_targets.py`: every service that consumes a queue of `TaskQueues.ALL`, plus `api`, has a scrape rule and a log rule; no dashboard query selects by `container` | render test |
| 6B-7 | research GPU S-16: the `triton-server` service reads its card and NUMA node from the env the installer generates from the active serving profile (`TRITON_GPU_ID` today); a second card is a second generated service `triton-server-<slot>`, never a hand edit; the `cluster-worker` consumer of 1C-8 joins the consumers test | `tests/compose/test_triton_service_shape.py`: no fixed `container_name`, the card from the generated env; `tests/compose/test_consumers_after_b2.py` lists `cluster` | the one-card render unchanged |
| 6B-8 | research GPU S-22 (pending RD-18): the DCGM exporter field list in the `monitoring` profile adds `DCGM_FI_PROF_SM_ACTIVE`, `DCGM_FI_PROF_SM_OCCUPANCY`, `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, `DCGM_FI_PROF_DRAM_ACTIVE`, `DCGM_FI_PROF_PCIE_TX_BYTES`, `DCGM_FI_PROF_PCIE_RX_BYTES`; the Grafana GPU dashboard shows them beside utilization; off in the minimal install | `tests/compose/test_dcgm_profiling_fields.py` | the monitoring render test |

## Lane ES-C `b2/embedding-readers`: every reader resolves its vector field through the registry (research embeddings plan 5.4, 8.1; approved, RD-1)

**Owns:** every file of `rg -l "pe_embedding|ITEM_EMBEDDING_FIELD|DEFAULT_KNN_FIELD" src scripts` that
6A and 6B do not own, **run on the merged W2.3b head** (W2.3a and W2.3b delete or move
`item_scores/near_dup.py` and `curation/clustering/*`; review 3.5) (about 40; the coordinator lists both lanes' files at the cut and moves any
shared file's change into one of them); `src/config/project_context.py` (`embedding_space(target,
key=None)`, refreshed on `config_epoch`); `src/domain/embedding_state.py`;
`src/services/curation/wire.py` (`embedding_states: {space_key: state}`); the search `space` and
`fuse` parameters on the three curation search routes; `contracts/`; new
`tests/curation/test_no_embedding_field_literals.py`.

| ID | Task | Red test | Control |
|---|---|---|---|
| ES-C1 | `ITEM_EMBEDDING_FIELD` and `DEFAULT_KNN_FIELD` deleted; every reader takes the bound project's space; the read projection excludes every registry vector field | `tests/curation/test_no_embedding_field_literals.py` fails on today's constant (the `test_embedding_state_consumers.py` pattern) | the existing search tests |
| ES-C2 | `build_knn_query` takes the space and the query vector's `space_key`; a mismatch raises `EmbeddingSpaceMismatch` | `tests/curation/test_space_mismatch_raises.py` | a matching query returns today's hits |
| ES-C3 | search wire: `space` (default the target's primary, echoed), `fuse` (RRF over optional spaces), `score_space`; item wire `embedding_states`; `make contracts` | contract test of the three routes | default answers unchanged |

## Lane `b2/frontend-spaces` (after ES-C merged)

Cropwright reads `embedding_states` instead of `embedding_state` and shows the space a search used;
no other change. Lane checks `npm run check`, lint, the touched unit tests, `npm run build`; red: a
unit test of the item badge reading `embedding_states` (fails on the old field).

## Lane 6C `b2/hardening` (after 6B merged; it edits every service block)

- **SEC-4:** reduce the privileges of an opt-in monitoring service. **SEC-5:** pin a build input and
  review per-service container hardening. The lane reads the maintainers' notes in the owner's
  gitignored evidence directory (`artifacts_local/audit/`) and applies the right-sized option
  recorded there; commit messages, PR text, CHANGELOG and docs use only these two neutral lines.
- **Public scope:** `security_opt: [no-new-privileges:true]` on every service B2 added or changed,
  `cap_drop: [ALL]` where the service starts without capabilities (the S0 and S1 records say which
  need some), no new published port. No blanket change to the GPU, Triton or OpenSearch services in
  this batch.
- **Red tests:** `tests/compose/test_new_service_hardening.py`; the private-note fixes carry their own
  neutral-named tests. **Control:** the overlay render test.
- The naming-leak scan and a manual read of the diff before every commit.

## Lane 6D `b2/docs` (last)

**Owns:** `docs/ARCHITECTURE.md`, `INSTALLATION.md`, `docs/CURATION.md`, `SECURITY.md`, `CLAUDE.md`,
`README.md`, root `CHANGELOG.md`, `ATTRIBUTION.md`, `docs-site/**`, `docs/opensearch_schema_design.md`,
`docs/design/curation_api_contract.md`, `docs/design/curation_design_rationale.md` section 6, the
doc-sync notes of stage 4 (`selectable_assist_plan.md`, `sam3_full_image_detection_plan.md`,
`auto_label_loop_plan.md`), `docs/PERFORMANCE.md` (the table skeleton for the live-run numbers).

Research content (approved, RD-1): `docs/ARCHITECTURE.md` gains the embedding registry, the cluster
partition record (assigner, versioned partition, review state), the near-duplicate service with two
levels, the assist loop's record in PostgreSQL and the serving profile; `docs/opensearch_schema_design.md`
the registry-named vector fields and the retired-field list, the `cluster_*` and `near_dup_*`
projections and the items normalizer; `docs/CURATION.md` spaces and the swap, cluster runs and
activation, near-duplicate status and the holdout rule, loop pause, budgets and slices; the header of
`auto_label_loop_plan.md` says its storage, events, budget and control sections are superseded for
the rebuild by [research_auto_label_assist_plan.md](../research_auto_label_assist_plan.md);
`CLAUDE.md` names the loop record and the cluster partition rule in the curation rules.

Content: the keep-list of [03_b2_overview.md](03_b2_overview.md) section 11; the services table and
the Celery conventions block in `CLAUDE.md` (routes in `src/config/task_routes/`, no decorator
`queue=`, no I/O in `worker_process_init`, no `terminate=True`, restart the worker after task edits,
one Alembic head, `project_tx` for every project query); INSTALLATION (services, secrets and their
ownership rule, `max_connections`, backup and restore, PgBouncer overlay, multi-host with a shared
volume, the 0.6.0 upgrade, the downgrade note; review fixes R-26 and R-36: upgrades are stop-the-world
in 0.7.0 (old and new processes never run together), the GPU-arbiter overlay is single-host, from
v0.8.0 migrations follow expand-then-contract; messages for a queue whose profile never started stay in Valkey after the reaper fails their job
(RC-13); the kNN warm-up is lost on an OpenSearch restart and the first query pays it (RC-7);
a PostgreSQL major version moves only with a release,
through `db backup` and `db restore`; the operator checklist of the review (back up `secrets/` and the
Postgres volume, size `OP_PG_MAX_CONNECTIONS` with replicas, `db verify` after any restore, a shared
POSIX volume for multi-host); every service requires Postgres, Valkey and OpenSearch, and `/ready` names the one that is down, D-69); SECURITY (neutral preconditions: the broker and the
database are internal-only with required passwords, JSON-only messages, whoever reaches the broker
can enqueue work); ATTRIBUTION rows (SQLAlchemy MIT, Alembic MIT, psycopg LGPL-3.0, Celery, kombu,
billiard BSD-3, redis-py MIT, Flower BSD-3, testcontainers Apache-2.0 (test only), Valkey BSD-3,
PostgreSQL licence; each confirmed from the installed wheel's metadata and Trivy's licence scanner
at pin time, [08](08_environment_and_setup.md) section 7.3); CHANGELOG `[Unreleased]`: Added
(Postgres, Valkey, Celery workers, beat, job listing routes, `Idempotency-Key`), Changed (jobs
survive API recycling; events over SSE with resumable ids), Removed (file protocols, the
cluster-refresh service, four index roles, `op_projects`, `op_global_configs`, the registry
migration CLI), Breaking (new core services and secrets, `queued` status, 202 for upload extraction,
directory ingest and person grouping, settings writes need `expected_revision`, promote `wait=true`
gone, removed error codes and the sync route, the GPU-arbiter overlay now mounts the Docker socket
into the new `gpu-arbiter-worker` only (RC-6) and must be re-applied by users who enabled it, carried from the original 2D-3
with a matching INSTALLATION note), Security (the two neutral lines).

Red test: `.venv/bin/python scripts/docs/check_docs_vs_code.py` over every touched doc;
`tests/test_doc_links.py`.

## Stage 6 exit criteria

- every lane's tests green on the merged branch; the done-when list of
  [03_b2_overview.md](03_b2_overview.md) section 1 holds by `rg`;
- then the B2 sweep ([03_b2_overview.md](03_b2_overview.md) section 9).

## Lane briefs

Use the common block of [03a](03a_stage1_foundation.md), with: "B2 stage-6 lane <6A..6D>", "cut
from int/b2-control-plane after stages 4 and 5 merged" (6C after 6B, 6D after 6C), this file (your
lane). 6C adds: "Read the private notes only for SEC-4 and SEC-5; never write a security detail
into a tracked file, a commit message or PR text." Compose project: none. GPU: none.
