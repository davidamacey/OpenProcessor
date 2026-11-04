# 02: the control plane on PostgreSQL (foundation)

| Field | Value |
|---|---|
| Status | plan, not started. Owner decisions 1-8 of 2026-11-04 are applied (approved) |
| Release | part of the single v0.7.0 release; built by B2 ([03_b2_overview.md](03_b2_overview.md)) |
| Files | this file (why, rules, the data-ownership table, the inventory with a destination for every item, the decisions); [02a_postgres_schema.md](02a_postgres_schema.md) (every table); [02b_jobs_locking_events.md](02b_jobs_locking_events.md) (jobs, claims, cancel, reaper, events, queues, pools); [02c_migration_ops_tests.md](02c_migration_ops_tests.md) (Alembic, the 0.6.0 upgrade and rollback, compose services, backup, storage seam, tests, traps) |
| Decisions | the register in [00a_decision_register.md](00a_decision_register.md) section 6: approved D-39..D-45 (the owner decisions), D-46..D-63 approved 2026-11-04 as a block (section 7 below keeps the plain question and the recommendation for each), D-64..D-67 added after the review [00c](00c_plan_review.md) |
| Package versions | not here: `01_package_pins.md` (written on its own branch) owns every exact pin. This set only names the packages |

Verified against `origin/main` at `7f3453de` on 2026-11-04 (read only, with `rg`). Line numbers move;
re-find code by the symbol named.

---

## 1. Why

The owner, 2026-11-04: "postgres needs to save ALL the relational data properly and correctly like a
real application"; job files are "a recipe for disaster and not cloud usable".

What is wrong today, verified in the code:

| Problem | Evidence | Consequence |
|---|---|---|
| Job state is JSON files on a shared volume | `FileJob` (`src/services/curation/file_job.py:58-186`) writes `state.json`, `heartbeat` and `cancel.flag`; 19 modules use it; 6 more job mechanisms use their own files (trainer, model export, bake-off, region cluster, promote, combine) | one host only; no query over jobs; a reader sees a stale job only by file mtime; every family re-implements status, cancel and resume |
| Control records live in a search engine | projects in `op_projects` (`src/services/projects/registry.py:32`), config store and settings in each project's `configs` index (`src/services/config_store/index.py:8-18`), the class registry as one document (`src/clients/curation_opensearch/registry.py:38`), leases as documents (`src/domain/job_lease.py:54-57`) | no transactions across two records, no foreign keys, no unique constraints except a document id, every write an OCC loop; 6 indexes (6 shards) per project, 4 of them control data |
| Some control writes have no concurrency control | the project settings `defaults` are written with `doc_as_upsert` and no OCC (`src/services/curation/settings_doc.py:122-129`) | two editors lose each other's change |
| Control state on local disk outside any job | the holdout freeze (`holdout.py:222-335`), label-operation records (`label_ops/store.py:53`), bulk-ingest sources with a per-source SQLite queue (`bulk_ingest/sources.py:5-9,161-175`), `promote.json` model records (`training/triton_promote.py:287-301`), the pause flag (`scripts/curation/worker/fairness.py:22-29`), GPU reservations (`training/gpu_arbiter_policy.py:28-38`) | none of this survives a second host or a container moved to another machine |
| Claims depend on one container | `curation-auto-label-worker` unlinks `trigger.json` (`scripts/curation/auto_label_worker.py:300-320`); the evaluator renames job files with no claim (`scripts/curation/bakeoff/bakeoff_runner.py:405-421`); the detection worker keeps in-flight ids in process memory only (`scripts/curation/worker/flow.py:43-119`); every compose service has a fixed `container_name` | a second replica double-runs work or cannot start at all (`--scale` is impossible) |
| Trainer reachability is a Docker container-name probe | `src/services/training/trainer_reachability.py:82-112`, `OP_GPU_ARBITER_TRAINER_CONTAINER` | not usable on a cluster without the Docker socket |

The fix: PostgreSQL holds every relational and transactional record; OpenSearch keeps the search
data; Valkey carries Celery messages and short-lived keys; files hold only blobs (images, crops,
models, datasets, exports), addressed by storage keys that Postgres records.

## 2. Rules (they apply to every later file)

1. **One owner per value.** Every field has exactly one store that owns it (section 3). The other
   side refers to it by an id. The single sanctioned exception is the class name projected onto
   items for search (section 3.3, D-47), which has one writer path and a verifier.
2. **Transactions, constraints and row locks, never process locks.** Uniqueness is a unique
   constraint; a singleton is a partial unique index; a claim is one conditional `UPDATE`; a queue
   scan is `FOR UPDATE SKIP LOCKED`; a short periodic critical section is a transaction-level
   advisory lock. No `asyncio.Lock`, `threading.Lock`, `fcntl` or file lock guards shared state.
   PostgreSQL docs: [explicit locking](https://www.postgresql.org/docs/current/explicit-locking.html),
   [the locking clause](https://www.postgresql.org/docs/current/sql-select.html#SQL-FOR-UPDATE-SHARE).
3. **The database clock decides expiry.** Lease and queue-wait expiry compare against `now()` in
   SQL, never a worker's wall clock, so hosts with skewed clocks agree.
4. **Never hold a transaction open across a network call.** No OpenSearch, Triton, VLM, segmenter
   or HTTP call happens inside a database transaction. A job writes its progress in a short
   transaction of its own.
5. **Project isolation holds in Postgres too.** Every control row that belongs to a project carries
   `project_id` with `ON DELETE CASCADE`, and row-level security refuses a row of another project
   (D-55; [02a](02a_postgres_schema.md) section 1.4). The OpenSearch guard
   (`src/services/projects/guard.py`) keeps guarding the data plane unchanged in purpose.
6. **Class identity is the name at every boundary.** Inside one project the registry's `class_id`
   is a stable foreign key (ids are never reused); import, combine, export, train, promote and model
   sharing map by name, as today.
7. **Fail closed.** The API and every worker require PostgreSQL, Valkey and OpenSearch, plus
   Triton for inference (owner decision D-69, which replaced D-64 on 2026-11-04): a process waits for
   them with bounded retries and then refuses to start, naming the missing one; there is no degraded
   mode. At run time `/ready` answers 503 and names each dependency that is down, `/live` says only
   that the process is up, and a route that reaches a down dependency answers 503. A database whose
   schema is not at the code's Alembic head refuses to start the process (`schema_not_current`). An unbound project raises
   `scope_not_set` in the database (R-13).
8. **Stateless processes.** The API, every Celery worker and beat keep no state another process
   needs: shared state lives in Postgres, Valkey or OpenSearch; blobs behind the storage seam.
   Scalable services carry no fixed `container_name` (D-54).
9. **No shims, no dual paths.** `FileJob`, the job files, the file protocols, the OpenSearch control
   documents and every file lock are deleted in the change that replaces them. The 0.6.0 data is
   moved once by the upgrade step and its OpenSearch copies deleted, never dual-read.
10. **Official docs and maintained libraries first.** SQLAlchemy 2.0, Alembic, psycopg 3, Celery
    5.6 with kombu's documented `redis://` transport and redis-py, FastAPI's own SSE support,
    testcontainers-python. Custom code is justified against them in
    [10_standards_and_libraries.md](10_standards_and_libraries.md) section 3.

## 3. Data ownership

### 3.1 The four stores

| Store | Owns | Never owns |
|---|---|---|
| **PostgreSQL** | projects and their settings, slugs, index names, pause; class registries; revisioned config (prompt packs, region and ingest profiles, open-vocabulary sets, VLM endpoints, activations, probes); jobs, attempts, progress, cancel, checkpoints, events; dataset uploads, imports and their ledgers; combine runs; ingest sources and their file queues; label operations; holdout freezes; training runs, models, engines (with their input and output contracts), promotions, shares, exports, bake-off runs; GPU devices and reservations; upgrade runs; the research-seam records of [02d](02d_research_seam_schema.md) (cluster partitions, identities, memberships, activations and calibrations; the embedding catalog, spaces and their re-index and evaluation runs; near-duplicate runs, groups, edges, decisions, pending rows and holdout moves; the assist-loop record and VLM usage; serving reports; later the plate reader runs, consensus groups and identity-text policies); later users and tokens | vectors, item documents, region boxes, item text, cluster assignments as search data |
| **OpenSearch** | `images` and `items` per project: every field a search, filter, kNN query or aggregation reads (geometry, class reference, region boxes and their text, OCR text, captions, VLM label text, vectors in the registry-named fields, the served cluster assignment, the near-duplicate projection, capture metadata, review flags such as `test_holdout`, `class_validated`, `class_excluded`); the five global visual-search indexes. Registration-plate text values live only here (never in PostgreSQL: one place to purge) | anything with a lifecycle, revision, owner or exclusivity |
| **Valkey** | Celery messages (kombu `redis://` transport); the queued marker per job (`op:queued:<job_id>`); short-lived caches with a TTL (resource reachability) | any record a restart may not lose; Celery results (no result backend, D-49) |
| **Files behind the storage seam** | source images (read only), uploads, crops cache, exports and their frozen snapshots (`class_registry.json` inside an export, `data.yaml`, `manifest.json`), training artifacts and logs, model files and engines, UMAP reducers, the upgrade bundle | any state that is read-modify-written; any absolute path in a database row |

### 3.2 Field by field, for everything that moves or is referenced across stores

| Value | Today | Owner after B2 | How the other side refers to it |
|---|---|---|---|
| project record (slug, display name, description, status, revision, origin, `pre_delete_status`) | `op_projects` doc `project:<slug>` (`registry.py:70-96`) | Postgres `projects`, `project_slugs` | OpenSearch indexes carry the slug in their name only; the guard maps index name to project from `project_indexes` |
| project index names (`resources.indexes`) | inside the project record (`projects.py:65-81`) | Postgres `project_indexes` (roles `images`, `items` only, D-56) | the guard's ownership map |
| project directories (`class_registry_path`, `export_root`, `upload_root`, `train_jobs_dir`, `autolabel_dir`, `bakeoff_jobs_dir`, `project_state_dir`) | inside the record | Postgres `projects.storage_prefix` (one relative key prefix); the job directories are deleted | storage keys `projects/<slug>/<area>/...` |
| projects revision counter | `meta:projects_revision` (`registry.py:26`) | Postgres `deployment_counters('projects_revision')` | API and worker snapshot polls |
| pause flag | `pipeline_paused.flag` file (`fairness.py:22-29`) | Postgres `projects.paused_at` | workers read the project row |
| class registry (classes, `next_class_id`, revision, hotkeys, merged_into) | OpenSearch doc `class_registry` (B1, unreleased) or `class_registry.json` (0.6.0) | Postgres `class_registries`, `project_classes` | items store `class_id`; see 3.3 for `class_name` |
| class `sample_count`, `validated_count` | stored in the registry doc and the `classes` mirror index (`registry_mirror.py:41,64`) | **nobody**: derived on read by a terms aggregation over `items.class_id` | the class list route joins the aggregation to the Postgres rows |
| `classes` mirror index | written by `registry_mirror.py` | deleted (D-56) | - |
| `labels_confirmed` index | no writer (`bodies_other.py:11-16`) | deleted (D-56) with its two readers (`classes_write.py:132`, `health.py:142`) | - |
| config documents, revisions, activations, activation events, `meta:config_revision`, runtime reports | per-project `configs` index (`config_store/index.py:8-18,107-123,301-385,406`) | Postgres `config_documents`, `config_revisions`, `config_activations`, `config_activation_events`, `projects.config_epoch`, `config_runtime_reports` | items keep `region_profile`, `region_profile_revision`, `vlm_prompt_pack`, `vlm_endpoint` as provenance stamps (`name@revision` strings; immutable once written, so they cannot drift) |
| VLM endpoint registry, probes, `local_vlm:desired` | `op_global_configs` (`global_store.py:21-125`, `vlm_endpoints.py:131-163`) | Postgres `config_documents` (`project_id` NULL, kind `vlm_endpoint`), `vlm_endpoint_probes`, `deployment_settings` | activation per project in `config_activations` |
| keymap overrides | `keymap:default` doc (`keymap.py:30,280`) | Postgres `project_keymaps` | - |
| project settings (`defaults`, `ingest_policy`, `vlm_policy`, `vlm_scope_reps`) | settings doc `default` (`settings_doc.py:15`, `policy_doc_store.py:77-84`) | Postgres `project_settings` (one revision, OCC on every write, fixing the gap in section 1) | - |
| cluster-refresh growth state | `cluster_refresh` key in the settings doc (`refresh_state_store.py:3-16`) | Postgres `project_state` key `cluster_refresh` (2C, interim body); from lane 4F the key holds only the beat's scheduling clock (deferral start, last attempt, backoff) and the trained count is the serving partition's `n_items` (research X-11) | - |
| UMAP projection state (`current`) and the reducer blob | `umap_viz_state` doc and the `umap_state` index (`bodies_other.py:103-144`, `umap_state.py:114`) | Postgres `projection_states`; the reducer bytes in storage (`projects/<slug>/umap/<state_id>.bin`) | items keep `viz_projection_version` as a stamp |
| job leases | `lease:<kind>[:<scope>]` docs (`job_lease.py:99-119`) | Postgres `jobs` (the partial unique index is the singleton, the row's `fencing_token` and `lease_expires_at` are the lease) | - |
| job records, progress, cancel, heartbeat, result | `FileJob` and the six other file mechanisms | Postgres `jobs`, `job_attempts`, `events`, `job_items` | items keep the job ids they already carry (`ingest_run_id`, `import_ids`, `region_run_id`, `label_op_id`) as references |
| holdout freeze record | `test_holdout/current.json` and history (`holdout.py:222-335`) | Postgres `holdout_freezes` | the membership stays the `test_holdout` flag on items (it is filtered in about 38 places) |
| label operations and their members (prior state for undo) | `label_ops/<op_id>/{state,members}.json` (`label_ops/store.py:53`) | Postgres `label_operations`, `label_operation_members` | items keep `label_op_id` |
| bulk-ingest sources, schedules, scan state, the per-source file queue | `source.json`, `state.json`, `queue.sqlite` (`bulk_ingest/sources.py:5-9,161-190`) | Postgres `ingest_sources`, `ingest_source_files` (claimed with `SKIP LOCKED`) | images keep `ingest_run_id` |
| dataset uploads, imports, mapping, scan, report, ledger | `datasets/uploads`, `ImportStore` files (`dataset_import/store.py:93-122`) | Postgres `dataset_uploads`, `dataset_imports`, `dataset_import_images` | items keep `import_ids` |
| combine runs and their sources | `/jobs/combine/cmb_*` (`combine/store.py:33`) | Postgres `combine_runs`, `combine_sources`; progress in `jobs` | items keep `origin_project`, `origin_item_id` |
| region drain, dependency health | files (`region_drain.py:68`; `region_dependency_health.py:54`) | Postgres `project_state` keys | - |
| region-cluster markers, `last_clustering.json`, the IVF retrain state inside `autolabel`, the `cluster_id`/`class_id` namespace overload | files (`region_cluster_jobs.py:56,123-127,284`; `last_run.py:28`), `cluster_ids.py` offsets | Postgres `cluster_partitions`, `cluster_identities`, `cluster_memberships`, `cluster_activations` (lane 4F, research clustering plan 5.1); the markers are never ported to `project_state` | items and region boxes carry the **served** assignment (`cluster_partition_id`, `cluster_uid`, `cluster_state`, `cluster_score`, `cluster_margin`, `cluster_assigned_by`) as a projection written by `cluster_projection_sync`; integer `cluster_id` and `cluster_subid` are deleted |
| encoder identity, per-project embedding spaces, re-index and evaluation runs | constants (`ITEM_EMBEDDING_FIELD`, `DEFAULT_KNN_FIELD`); combine compares only the dimension | Postgres `embedding_model_versions`, `embedding_spaces`, `embedding_reindex_runs`, `embedding_evaluation_runs` (lanes ES-A, ES-B) | the vectors and per-item state fields stay in OpenSearch under the registry-named fields; 0.6.0 names (`pe_embedding`, `embedding_state`, `region_box_embeddings`) stay as registry data |
| near-duplicate groups (curation export, crop scorer, core ingest) | per request, per IVF bucket, or online in the request; `dup_*` and `duplicate_*` fields | Postgres `near_dup_runs`, `near_dup_groups`, `near_dup_members`, `near_dup_edges`, `near_dup_decisions`, `near_dup_pending`, `holdout_near_dup_moves`, the exact key `image_contents` (lanes ND-A, ND-B, ND-C) | the `near_dup_*` fields and `content_sha256` are a search projection rewritten from the current run; images lose `imohash` and `phash` (pending RD-43) |
| assist-loop budget, event log, node state, proposals | `budget.json`, `events.jsonl`, the loop job directory | Postgres `assist_loop_*` tables and `vlm_usage_hourly` (lane 4D); the `events` table is the 24-hour UI feed only (research G-3) | items keep the label-operation ids of the loop's propagations |
| training jobs, status, manifest, registry snapshot, log | `<id>.{job,status,manifest,registry_snapshot}.json`, `<id>.run.log`, claim files (`training/job_files.py:30-87`) | Postgres `jobs` + `training_runs` (spec, class names in output order, lineage); the log and checkpoints in storage | - |
| model upload/export tasks | `/jobs/model_export/{queue,claimed}` (`model_export.py:8-52`) | Postgres `jobs` (kind `model_export`) + `models`, `model_engines` | Triton model names in `model_engines` |
| promotions, `promote.json`, class remap, sharing | files (`triton_promote.py:287-327`) | Postgres `promotions`, `models`, `model_shares` | Triton holds the engine; `labels.txt` is written from `models.labels` |
| bake-off jobs and results | `bakeoff_jobs/*.job.json` + `status.json` (`bakeoff_runner.py:385-421`) | Postgres `jobs` (kind `bakeoff`) + `bakeoff_runs`; result files in storage | - |
| GPU reservations and the VLM pause sentinel | `training_gpus.lock`, `vlm_worker/pause.sentinel` (`gpu_arbiter_policy.py:28-154`) | Postgres `gpu_devices` and `gpu_reservations` (keyed by the card's NVML UUID, research S-6, S-7) | - |
| serving layout (servers, load lists, instances, image-prep limits) | `config_templates/profiles/*.json`, `.env`, the installer's `config.pbtxt` edits | Postgres `config_documents` kind `serving_profile` (deployment), active one in `deployment_settings('active_serving_profile')`; `serving_reports` from the beat inventory (research S-5, S-14) | `config.pbtxt` instance counts are written from the profile by the installer |
| trainer capabilities | `.trainer_capabilities.json` (`preflight_checks.py:292`) | Postgres `worker_capabilities` (written by the `gpu_train` worker at start) | - |
| events for the UI | `events.jsonl` with `flock` (`event_hub.py:77,186,238,261`) | Postgres `events` + a Valkey wake-up (D-48) | SSE `id:` is the `events.id` |
| person names and merges on the global faces index | `visual_search_faces` (`src/clients/opensearch/indexes.py`) | stays in OpenSearch for this program | out of scope: core inference data, not project control data |

### 3.3 The class name on items

Items carry `class_id` and `class_name`, and `class_name` is read or written in 210 files (1,036
lines: `rg -c class_name src`). Dropping it from the items index is a data-plane rewrite of its own.
**Recommendation (D-47):** `class_id` is the reference; `class_name` on items becomes a *search
projection* owned by Postgres:

- every write that sets a class on an item resolves the name from `project_classes` in the same
  code path that resolves the id (one helper, `src/services/classes/resolve.py`);
- a rename or merge in Postgres writes a `search_projection_tasks` row **in the same transaction**,
  and a `class_projection_sync` job rewrites the affected items with one `update_by_query` filtered
  by `class_id`; the reaper creates the job if a row has none, so the rewrite cannot be forgotten;
- a verifier (`openprocessor db verify`, and the upgrade step) reports items whose
  `(class_id, class_name)` disagrees with the registry.

The alternative (items carry only `class_id`; every reader resolves names) removes the projection
but touches every reader; it is listed for v0.8.0 if the owner prefers it.

### 3.4 What stays on items although it looks like control

`region_status`, the `region_*` summary fields, `test_holdout`, `class_validated`,
`class_excluded`, `embedding_state` (one state field per registry space), `open_vocab_status`, the
served cluster assignment and the near-duplicate projection stay in OpenSearch:
they are filtered and aggregated together with search queries (about 38, 47 and 33 sites for the
first three), so moving one would turn every filter into a cross-store join. Each is written under
the existing OCC and lock rule (`src/clients/occ.py`, `occ_locks.py`). The detection and VLM stream
workers keep using `region_status` as their item state machine (D-53).

## 4. Inventory and destination (every item from the code)

### 4.1 `FileJob` users (19 modules under `src/`, none in `scripts/` or `docker/`)

| Module | Family | Destination (stage) |
|---|---|---|
| `curation/reprocess_job.py:34` | reprocess | `jobs` kind `reprocess`, queue `jobs` (stage 4, lane 4A) |
| `curation/assist/job.py:26`, `runner.py:50`, `segment.py:25` | assist run | `jobs` kind `assist_run` + `job_items` members (4A) |
| `curation/item_scores/job.py:67` | item scores | kind `item_scores` (4A) |
| `curation/probe_job.py:60` | probe | kind `probe` (4A) |
| `curation/selection/job.py:45` | selection | kind `selection` (4A) |
| `curation/embedding_viz.py:72` | embedding viz | kind `embedding_viz` + `projection_states` (4A) |
| `curation/dataset_import/store.py:26` | dataset import, undo, resume | kinds `dataset_import`, `dataset_import_undo` + `dataset_imports`, `dataset_import_images` (4B) |
| `projects/combine/execute.py:42` (reuses `ImportStore`) | combine | kind `combine` + `combine_runs`, `combine_sources` (4B) |
| `curation/bulk_ingest/store.py:25`, `runner.py:60` | bulk ingest | kind `bulk_ingest_run` + `ingest_source_files` (4C) |
| `curation/bulk_ingest/sources.py:36` | source bookkeeping | `ingest_sources` (stage 2, lane 2D) |
| `curation/label_ops/store.py:33` | label operation records | `label_operations`, `label_operation_members` (2D) |
| `curation/assist_loop/job.py:28`, `store.py:26` | assist loop | kind `assist_loop` (4D) |
| `curation/autolabel/job.py:67` | auto-label | kind `autolabel` (4D) |
| `training/promote_job.py:23` | promote | kind `promote` + `promotions` (stage 5, lane 5D) |
| `curation/lease_heartbeat.py:42` | fence and heartbeat | deleted; `src/tasks/runtime/heartbeat.py` (stage 3; in the runtime package so the GPU images can import it, R-2) |
| `projects/busy.py:204,216,364` (reads state files) | "is this project busy" | one query on `jobs` (stage 3) |

The 13 test modules that import `FileJob` (agent inventory of 2026-11-04) are rewritten with the
family that owns them; none survives as a file-protocol test.

### 4.2 Other job mechanisms (not `FileJob`)

| Mechanism | Where | Destination |
|---|---|---|
| trainer job protocol (`job.json`, `status.json`, `cancel`, claim generations) | `src/services/training/jobs.py`, `job_files.py:30-114`; `docker/trainer/job_protocol.py`, `job_claim.py:30-100`, `job_status.py`, `trainer.py:55,296-324` | queue `gpu_train`, kind `train` (stage 5, lane 5A) |
| model export (`queue/`, `claimed/`, heartbeats, `reconcile_claimed`) | `src/services/model_export.py:8-328`; `export/export_worker.py:44-150,364` | queue `gpu_export`, kind `model_export`, gains cancel (5B) |
| bake-off evaluator (`bakeoff_jobs/*.job.json`, no claim) | `scripts/curation/bakeoff/bakeoff_runner.py:385-421` | queue `gpu_eval`, kind `bakeoff` (5C; D-31 settled by owner decision 4) |
| region cluster and FP jobs (`job.json`, `fp_job.json`, markers) | `clustering/region_cluster_jobs.py:46-56,123-127,201,275,284,336,366` | kinds `region_cluster`, `region_fp_centroids`; markers in `project_state` (4B) |
| region drain, dependency health | `region_drain.py:57-68`, `region_dependency_health.py:44-54` | `project_state` keys (2D) |
| promote JSON | `training/promote_json.py:1-52` | `promotions` (5D) |
| startup reconcile over 9 modules | `startup_reconcile.py:22-71`, lifespan `:143` | deleted; the reaper (stage 3) |
| project delete finish | `routers/curation/projects.py:325`, `projects/delete.py` (`_FINISH_IN_PROGRESS`) | kind `project_delete` with `project_id` NULL and `scope` = slug (4C) |
| directory ingest (inline in the request) | `src/routers/ingest/directory.py` | kind `directory_ingest`, `project_id` NULL (4C, D-22a) |
| person grouping (inline in the request) | `src/routers/persons.py` | kind `persons_cluster`, `project_id` NULL (4C) |
| class merge (`update_by_query` in the request) | `src/routers/curation/classes_merge.py:64-157` | kind `class_merge` (+ `class_projection_sync`) (4B) |

### 4.3 Leases, registry, config store, projects (B1 code)

| Code | Destination |
|---|---|
| `src/clients/job_lease.py` (`acquire`, `takeover_expired`, `renew`, `release`, `is_live`; `transfer` has no callers) and `src/domain/job_lease.py` (19 kinds; `BULK_INGEST`, `SOURCE_SCAN` have no callers) | deleted in stage 3; the job claim, heartbeat and fencing in Postgres replace them ([02b](02b_jobs_locking_events.md) sections 2-4) |
| `src/services/curation/lease_heartbeat.py` (thread with its own loop and client, `may_write()` self-fence) | replaced by `src/tasks/runtime/heartbeat.py` (R-2): the same thread shape with its own database connection; `may_write()` stays for OpenSearch writes (stage 3) |
| lease callers outside jobs: `LABEL_OP`, `LABEL_UNDO` (`label_ops/record.py:39`, `undo.py:192-201`), `LEGACY_REPAIR` (`legacy_repair.py:95`), `PROMOTE_JSON` (`promote_json.py:52`), the `ASSIST_LOOP` short lease (`assist_loop/job.py:180,204`) | label op and undo become exclusive `jobs` rows with `requested_by='api'` that run inline (short, no Celery: [02b](02b_jobs_locking_events.md) section 2.5); legacy repair becomes a migration-time step and is deleted with the rest of the 0.6.0 repair code (stage 2); the other two disappear with their files |
| class registry: `registry.py`, `registry_mirror.py`, `registry_transfer.py`, `scripts/curation/migrate_class_registry.py`, `scripts/lib/registry_migration.sh`, CLI `curation migrate-registry` (`scripts/lib/cli_core.sh:236-258`), 409 `registry_not_migrated`, `registry_full`, `registry_conflict` | deleted in stage 2; `src/services/classes/` over `src/db/repos/classes.py`. The 0.6.0 file migrates in the upgrade step ([02c](02c_migration_ops_tests.md) section 2). `class_registry.json` stays only inside exports and training pins (the six readers in the inventory) |
| config store: `src/services/config_store/` (`index.py`, `store.py`, `global_store.py`, `store_poll.py`), `vlm_endpoints.py`, `vlm_activation.py`, `vlm_snapshot.py`, `keymap.py`, `settings_doc.py`, `policy_doc_store.py` | rewritten over `src/db/repos/config.py` and `settings.py` (stage 2); the poll stays (5 s, `OP_CONFIG_POLL_S`) but reads one integer, `projects.config_epoch` |
| projects: `src/config/projects.py`, `project_context.py`, `src/services/projects/{registry,bootstrap,lifecycle,delete,delete_paths,busy,legacy_repair}.py`, `guard.py`, `guard_policy.py` | registry and lifecycle over `src/db/repos/projects.py`; the guard's ownership map built from `project_indexes`; `op_projects` and `op_global_configs` leave the guard policy (stage 2) |

### 4.4 Background tasks and periodic loops

| Site | Destination |
|---|---|
| registry poll (`projects/bootstrap.py:211,266`, 1 s) | stays per process; reads `deployment_counters('projects_revision')` |
| config poll (`config_store/store_poll.py:132,149`, 5 s) | stays per process; reads `config_epoch` |
| event-hub tail (`event_hub.py:444`, 0.25 s) | replaced by one Valkey pub/sub subscription per API process plus a 2 s safety tick over the Postgres `events` table (stage 3; owner decision 2026-11-04, R-6) |
| ops-metrics refresh (`ops_metrics_refresh.py:248-267`) | stays (writes only its own Prometheus files) |
| kNN warm-up (`core/dependencies.py:160`) | stays |
| startup reconcile (`lifespan.py:143`) | deleted; reaper (stage 3) |
| GPU arbiter and promoted reload (`lifespan.py:146-229`) | beat tasks `op.periodic.gpu_arbiter`, `op.periodic.reload_promoted` (stage 6) |
| open-vocab sweeper (`open_vocab_sweeper.py:99-125`) | beat `op.periodic.open_vocab_sweep` (6) |
| ingest dispatcher (`bulk_ingest/dispatcher.py:189,211,251-263`) | beat `op.periodic.ingest_dispatch` (6) |
| crop-cache prune, inline from ingest (`source_image_cache.py:142-257`, `ingest_index.py:222`) | beat `op.periodic.cache_prune` (6) |
| cluster-refresh daemon (`scripts/curation/cluster_refresh_daemon.py`) | beat `op.periodic.cluster_refresh`; service deleted (6) |
| IVF retrain self-trigger (`auto_label_worker.py:130`) | beat `op.periodic.ivf_retrain_check` (6) |
| request-started jobs (`create_task` at `reprocess_job.py:196`, `assist/runner.py:89`, `dataset_import/runner.py:400,418`, `combine/service.py:180`, `promote_job.py:187`, `probe_job.py:280`, `item_scores/job.py:186`, `selection/job.py:154`, `embedding_viz.py:235`, `region_cluster_jobs.py:275,366`, `bulk_ingest/runner.py:447-519`, `routers/curation/projects.py:325`) | Celery (stage 4, stage 5 for promote) |
| per-process helpers (`crop_orders.py:87`, `vlm_factory.py:102`, `resource_links.py:166`, `pipeline_events.py:73`, `ingest_bulk_preview.py:177`, `health.py:159-160`, `ingest_writer.py:156,187,246`, `pit_pages.py:113`) | stay; B4-4B allowlists them with a reason |

49 `create_task(` lines exist in `src/services` and `src/routers` today.

### 4.5 File locks (`scripts/codegen/check_file_locks.py`, baseline `scripts/codegen/baselines/file_locks.txt`)

| Site | Guards | Destination |
|---|---|---|
| `curation/job_lock.py:43-84` | the helpers | deleted when the last caller goes (stage 3) |
| `bulk_ingest/runner.py:117` (`start.lock`) | one bulk run per project | exclusive `jobs` row (4C) |
| `bulk_ingest/dispatcher.py:211` (`dispatch.lock`) | one tick per project | transaction advisory lock per project in the beat task (6) |
| `bulk_ingest/source_scan.py:28` (`scan.lock`) | one scan per source | exclusive `jobs` row kind `source_poll`, scope = source id (4C) |
| `open_vocab_sweeper.py:99` | one sweep per project | advisory lock (6) |
| `event_hub.py:77,238,261` | JSONL append and rotate | deleted with the JSONL log (3) |
| `source_image_cache.py:31,173-224` (`.prune.lock`) | one prune | advisory lock (6) |
| `docker/trainer/job_claim.py:47` (`O_EXCL` generations) | trainer claim | the job claim `UPDATE` (5A) |
| `utils/atomic_write.py:41-60` (`os.link` create-only) | create-only train job files | the helper stays for blob writes; its job-file caller is deleted (5A) |

End state: the baseline file is empty after stage 6, and the ratchet asserts zero (B4 no longer owns
this; its done-when is met by B2).

### 4.6 Compose worker services

| Service | Today | After B2 |
|---|---|---|
| `api` | uvicorn, `--workers=32`, fixed `container_name`, mounts `openprocessor-jobs:/jobs` | no `container_name`, no `/jobs`; replicas allowed |
| `curation-detection-worker`, `curation-vlm-worker` | stream workers over item states, no cross-replica claim | stay stream workers; claim a project through `stream_worker_claims` so N replicas never process one project twice (D-53) |
| `curation-auto-label-worker`, `curation-assist-loop-worker` | trigger-file pollers | Celery consumers of `autolabel`, `assist_loop` |
| `curation-cluster-refresh` | daemon | deleted (beat task) |
| `curation-trainer` | file-protocol watcher | Celery consumer of `gpu_train` in the trainer image |
| `model-export-worker` | file-protocol watcher | Celery consumer of `gpu_export` |
| `curation-evaluator` | job-file watcher | Celery consumer of `gpu_eval` |
| new | - | `postgres`, `valkey`, `db-migrate` (one-shot), `jobs-worker`, `periodic-worker`, `open-vocab-worker`, `jobs-beat`, `flower` (monitoring) ([02c](02c_migration_ops_tests.md) section 4) |

Named volume `openprocessor-jobs` is deleted. `openprocessor-state` keeps only blobs (uploads,
caches) behind the storage seam.

### 4.7 OpenSearch index roles

| Role | Fields | Verdict |
|---|---|---|
| `images` | identity, geometry, `content_sha256` (the exact key; `imohash` and the writer-less `phash` deleted, pending RD-43), vectors, import and combine provenance, the `capture.*` object (W4) | stays (search data) |
| `items` | see the schema doc; control-looking fields stay per section 3.4 | stays |
| `classes` | mirror of the registry | deleted (D-56) |
| `labels_confirmed` | no writer | deleted (D-56) |
| `umap_state` | reducer blob | deleted; blob to storage (D-56) |
| `configs` (with the folded settings and UMAP view state) | all control | deleted (D-56) |
| `op_projects` | project registry | deleted |
| `op_global_configs` | VLM endpoints | deleted |
| `visual_search_*` | global inference indexes | unchanged |

Per project: 6 indexes become 2, so the shard budget per project drops from 6 to 2
(`docs/opensearch_schema_design.md` section 3).

## 5. Where the old plans' premises are now wrong

| Old premise | Where | Now |
|---|---|---|
| "The job record (`FileJob` files) and the OpenSearch lease (B1) stay the truth; the broker is transport only" | former 03 section 1 | Postgres `jobs` is the truth; `FileJob` and the leases are deleted; the broker is still transport only |
| D-21: status, progress and cancel stay in job files; `/jobs` stays single-host | former 03 section 12 | replaced by D-39 |
| D-22: the trainer, the export worker and the evaluator keep their file protocols | former 03 sections 5.3, 23 | all three become Celery GPU consumers (owner decision 4) |
| global serialization comes from `-c 1` and one container with a fixed `container_name` | former 03 section 6 | correctness comes from database exclusivity; `-c 1` is only a capacity choice, so every worker can scale |
| B4-4C moves events to Redis Streams after a spike (D-30) | 05 section 6b | events are a Postgres table plus a Valkey wake-up (it was `NOTIFY` until the review), built in B2 stage 3 (D-48); the JSONL log and the last `flock` go in B2 |
| B1's registry migration (`migrate-registry`, `.migrated` rename, 409 `registry_not_migrated`) is the upgrade path | B1, 08 section 5 step 11 | B1 never shipped in a release; 0.7.0 migrates `class_registry.json` straight to Postgres in the upgrade step, and the B1 registry document code is deleted |
| "Valkey only if a redistributor objects" (D-18) | 00 section 6 | Valkey 9 is approved (owner, 2026-11-04) |
| the queued marker and lease state live in OpenSearch documents | former 03 section 9 | the job row holds queue state; the queued marker stays a Valkey key because it must share fate with the broker message ([02b](02b_jobs_locking_events.md) section 3.2) |
| the reaper scans job directories per project | former 03c 2D-2 | one indexed query over `jobs` |
| the detection and VLM workers are "unchanged" | former 03 section 5.3 | they have no cross-replica claim today (`flow.py:43-119`, `vlm_worker.py:68-197`); scale-out needs the project claim of D-53 |
| per-source `queue.sqlite` | not in any plan | a local SQLite queue per ingest source; it moves to `ingest_source_files` |
| `OP_GPU_ARBITER_TRAINER_CONTAINER` is how the API finds the trainer | compose comment F-72, `trainer_reachability.py` | trainer reachability = a live consumer of `gpu_train` (Celery inspect) plus `worker_capabilities` |

## 6. Code layout (new packages)

| Path | Layer | Holds | May import |
|---|---|---|---|
| `src/db/base.py`, `src/db/types.py` | db | `MetaData` with the naming convention, `DeclarativeBase`, shared column types | `sqlalchemy`, `src/config`, `src/domain`, `src/utils` |
| `src/db/models/` (`projects.py`, `classes.py`, `config.py`, `jobs.py`, `datasets.py`, `training.py`, `ops.py`) | db | ORM models, one file per concern | as above |
| `src/db/engine.py`, `src/db/session.py` | db | lazy per-process engines, `async_sessionmaker`, the project-scoped transaction helper (`project_tx`) | as above |
| `src/db/repos/` | db | repositories: every SQL statement lives here | as above, plus `src/db/models` |
| `src/db/migrations/` + `alembic.ini` at the repo root | db | Alembic environment and versions | as above |
| `src/core/db_session.py` | core | the FastAPI dependency (`DbSessionDep`) | `src/db`, `fastapi` |
| `src/services/jobs/` | services | API side: start, cancel, resume, status, the reaper rules | `src/db`, `src/clients/task_queue.py`, `src/tasks/runtime/` (the heartbeat class, for inline jobs) |
| `src/services/classes/`, `src/services/events/` | services | class registry, events and SSE fan-out | `src/db` |
| `src/tasks/` | tasks | Celery app, the `JobTask` wrapper, task modules | services, db, config |
| `src/tasks/runtime/` | tasks | the wrapper and its bookkeeping, the heartbeat thread, progress and checkpoint writes (fenced, through `src/db/repos/jobs.py`), the event row (through `src/db/repos/events.py`) and the Valkey wake-up (`event_wakeup.py`); importable without FastAPI, OpenSearch or torch (the trainer image imports it); the OpenSearch client factory is injected by the handler module (review fix R-2) | `src/db`, `src/config`, `src/domain`, `src/utils` only (plus `redis`, `celery`) |
| `src/clients/storage/` | clients | the storage seam (local backend) | `src/config`, `src/utils` |
| `src/clients/task_queue.py` | clients | producer, revoke, queued marker | `celery`, `redis`, `src/config` |

Layering contracts added to `scripts/codegen/check_layering.py` (stage 1): `src/db` imports no
`src.services`, `src.clients`, `src.routers`, `fastapi`; `src/{config,domain,utils,schemas}` import no
`src.db`; `src.tasks.runtime` is a leaf layer (re-check RC-2): `src/services`, `src/tasks`,
`docker/trainer`, `export` and `scripts/curation/bakeoff` may import it, and nothing outside
`src/tasks/` imports any other `src.tasks` module (the Celery task modules); `src/tasks/runtime/` imports
only `src.db`, `src.config`, `src.domain`, `src.utils` (checked by a fresh-interpreter test, as in
`tests/tasks/test_no_cuda_import.py`, and by a planted-import red test in 1A-11: a runtime module
importing `src.services` fails, review R-2); `src/routers` imports no `src.db.repos` (handlers go
through services, review section 3.4).

## 7. Decisions (plain question first, then the code)

Approved by the owner on 2026-11-04: **D-39** PostgreSQL holds all control data (replaces D-21);
**D-40** OpenSearch keeps the search data; **D-41** every long job runs on Celery, training and model
export included, and every file protocol is deleted (D-22's scope); **D-42** cloud-ready scale-out;
**D-43** one-step 0.6.0 upgrade that moves the OpenSearch control data and then deletes it;
**D-44** exact pins at the latest installable versions (file 13); **D-45** plain names first. Also
approved that day: **D-17** Celery 5.6, **D-18** Valkey, **D-31** the evaluator on `gpu_eval`.

Approved by the owner on 2026-11-04 as a block (the rows below keep their text for the record; D-48
and D-50 as amended after the review of [00c](00c_plan_review.md)):

| ID | Plain question | Recommendation | Why | Detail |
|---|---|---|---|---|
| D-46 | Where do images, crops, models and exports live when there is more than one host: a shared volume, or an S3-compatible object store? | 0.7.0 ships the storage seam with the local backend, so a shared POSIX volume (NFS or similar) is the multi-host answer now; an S3-compatible backend follows in v0.8.0 | the seam (keys in Postgres, no absolute paths) is what avoids a structural change later; an S3 backend now would touch the trainer, Triton loading and every image read in the same release | [02c](02c_migration_ops_tests.md) section 5 |
| D-47 | Should items keep the class name next to the class id? | yes, as a search projection owned by Postgres, rewritten by one job after a rename or merge, checked by a verifier | removing it touches 210 files; the projection has one writer path and cannot be forgotten (outbox row in the same transaction) | section 3.3 |
| D-48 | How do job progress and other events reach the browser? | a Postgres `events` table (the durable record and the resume cursor), woken by Valkey pub/sub (owner, 2026-11-04, review R-6; not a per-process `LISTEN`), SSE ids are event ids | one store for state and its feed, project-scoped rows, works across hosts; no second system on the UI path | [02b](02b_jobs_locking_events.md) section 6 |
| D-49 | Does Celery keep task results anywhere? | no result backend (`task_ignore_result=True`); the `jobs` table is the only truth | a second state store would drift from the first | [02b](02b_jobs_locking_events.md) section 7 |
| D-50 | Does the default install include a connection pooler (PgBouncer)? | no; an opt-in overlay, required when the connection formula of 02b section 8 with the planned scale factors exceeds `max_connections` (R-6) | one host fits `max_connections` (section 8 of 02b); a pooler forbids the session features the beat and migration locks use | [02b](02b_jobs_locking_events.md) section 8 |
| D-51 | What happens to jobs that were running when a 0.6.0 install is upgraded? | they are recorded as `interrupted` without automatic resume; the user restarts them | resuming a 0.6.0 cursor in 0.7.0 code is untested ground | [02c](02c_migration_ops_tests.md) section 2 |
| D-52 | Which 0.6.0 job histories are moved into Postgres? | the ones with lasting meaning: training runs, promotions, models, dataset imports (undo needs them), exports, label operations, combine provenance; the transient ones (reprocess, probe, scores, selection, viz, cluster, assist runs) stay only in the upgrade bundle | keeps the migration small and tested | [02c](02c_migration_ops_tests.md) section 2 |
| D-53 | Do the detection and VLM workers become Celery tasks? | no: they stay stream workers over item states, and each replica claims whole projects through a Postgres lease so two replicas never work one project | one message per item would duplicate OpenSearch state and lose batching; project claims are the smallest change that makes replicas safe | [02b](02b_jobs_locking_events.md) section 5 |
| D-54 | Which services lose their fixed `container_name`? | all of them; scripts and the GPU arbiter find containers by the Compose labels | a fixed name blocks `--scale` and replicas; the label is documented Docker behaviour | [03f](03f_stage6_scheduler_scaleout.md) |
| D-55 | Does Postgres itself refuse another project's rows (row-level security), or only the repository code? | both: repositories always filter by the bound project, and row-level security is on for every project-scoped table | fail closed at the storage layer, the same idea as the OpenSearch guard | [02a](02a_postgres_schema.md) section 1.4 |
| D-56 | Are the per-project `configs`, `classes`, `umap_state` and `labels_confirmed` indexes deleted? | yes (6 indexes per project become 2; the UMAP reducer moves to storage) | no dead or duplicated stores; a third of the shards | section 4.7 |
| D-57 | Is the 10,000-class cap kept? | no: drop the cap and the `registry_full` error | the cap existed for the size of one OpenSearch document | [02a](02a_postgres_schema.md) section 3 |
| D-58 | Do job-start routes accept an `Idempotency-Key` header? | yes, optional, stored per project and kind | a retried start after a timeout returns the same job instead of a 409 | [02b](02b_jobs_locking_events.md) section 2.3 |
| D-59 | Are prompt packs, profiles and settings stored as validated JSON documents or as columns? | validated `jsonb` bodies (pydantic on write and read), with the identity, revision and activation as columns | the bodies are editor documents with their own schema; the relations are what need constraints | [02a](02a_postgres_schema.md) section 4 |
| D-60 | How is "exactly one beat" enforced on a cluster? | beat takes a Postgres session advisory lock at start and exits if another holds it; periodic task bodies also take transaction locks | the Celery docs require one scheduler; a lock makes it a fact, not a convention | [02b](02b_jobs_locking_events.md) section 4 |
| D-61 | Is Postgres backed up on a schedule by default? | no schedule by default: `openprocessor upgrade` always takes a `pg_dump`, `openprocessor db backup` does it on demand, and a beat task can be enabled with `OP_DB_BACKUP_INTERVAL_H` | backups cost disk the owner may not want on a workstation | [02c](02c_migration_ops_tests.md) section 6 |
| D-62 | How do tests get a real Postgres? | testcontainers-python on a developer machine, a service container in CI; database tests carry a `db` marker that the Docker-free offline command leaves out | SQLite cannot test `SKIP LOCKED`, partial unique indexes or row-level security | [02c](02c_migration_ops_tests.md) section 7 |
| D-63 | Does B2 add a cheap database test run at each stage merge? | yes: `pytest -m db` (minutes, no GPU) after each stage merge, still one full sweep for the batch | the schema and claim rules are the batch's foundation; catching a break per stage is cheaper than at the end | [03_b2_overview.md](03_b2_overview.md) section 9 |

## 8. What we take from comparable public projects (read 2026-11-04)

| Project | How it splits the stores | Taken | Not taken |
|---|---|---|---|
| CVAT ([docker-compose.yml](https://github.com/cvat-ai/cvat/blob/develop/docker-compose.yml), [settings/base.py](https://github.com/cvat-ai/cvat/blob/develop/cvat/settings/base.py), [engine/models.py](https://github.com/cvat-ai/cvat/blob/develop/cvat/apps/engine/models.py)) | PostgreSQL holds annotations as relational rows (`LabeledShape`, `LabeledTrack`, `Label`); RQ on Redis (AOF on) runs one worker container per queue (import, export, annotation, webhooks, quality reports, chunks) with a per-queue timeout; media on a shared `cvat_data` volume | one worker service per queue with its own time limit; broker persistence on | job status kept only in the broker (CVAT needs its own Redis data migrations for it) |
| Label Studio ([core/redis.py](https://github.com/HumanSignal/label-studio/blob/develop/label_studio/core/redis.py), [data_export/models.py](https://github.com/HumanSignal/label-studio/blob/develop/label_studio/data_export/models.py)) | PostgreSQL with JSON fields for task data and results; RQ for background work; export jobs keep their status in a database row | the database row as the status of record | `start_job_async_or_sync` runs the job inline in the request when Redis is down: fail open, against our rule 7 |
| Immich ([docker/docker-compose.yml](https://github.com/immich-app/immich/blob/main/docker/docker-compose.yml), [architecture](https://docs.immich.app/developer/architecture/#redis), [backup and restore](https://docs.immich.app/administration/backup-and-restore/#asset-types-and-storage-locations)) | PostgreSQL (with a vector extension) for metadata; Valkey 9 (by digest) under BullMQ for queues; a separate machine-learning container; one upload root split into originals and regenerable derivatives; automatic database dumps into the data volume; "back up the database first, then the files" | Valkey by digest as the broker; originals vs regenerable files in the storage layout; dump before files; the API and job-worker roles in one image | vectors in Postgres (OpenSearch keeps them here, owner decision 2) |
| FiftyOne ([delegated_operation.py](https://github.com/voxel51/fiftyone/blob/develop/fiftyone/factory/repos/delegated_operation.py), [config](https://docs.voxel51.com/user_guide/config.html#configuring-mongodb-connection)) | MongoDB holds samples and labels and the delegated-operation documents; state moves by `find_one_and_update` with a `required_state` guard; the open-source runner polls and runs one operation at a time | guarded state transitions (our claim `UPDATE ... WHERE status = 'queued'`) | one store for documents and control; a single polling executor |
| OpenTranscribe ([core/celery.py](https://github.com/davidamacey/OpenTranscribe/blob/master/backend/app/core/celery.py), [models/media.py](https://github.com/davidamacey/OpenTranscribe/blob/master/backend/app/models/media.py), [db/migrations.py](https://github.com/davidamacey/OpenTranscribe/blob/master/backend/app/db/migrations.py), [alembic/env.py](https://github.com/davidamacey/OpenTranscribe/blob/master/backend/alembic/env.py), [docker-compose.yml](https://github.com/davidamacey/OpenTranscribe/blob/master/docker-compose.yml)) | FastAPI + Celery + Redis + PostgreSQL + OpenSearch + S3-compatible storage; the `Task` table is keyed by the Celery task id with status and progress; a partial unique index (`uq_task_active_per_file_type`, `WHERE status IN ('pending','in_progress')`) allows one active job per file and type; migrations run under `pg_advisory_lock(42)` on a dedicated session; explicit queues, `task_create_missing_queues=False`, prefetch 1, soft shutdown 30 s, 6 h visibility timeout, 3 h / 3 h 15 m limits | the job row keyed by the Celery id; the partial-unique active-job guard; the advisory-locked migration runner; the queue and shutdown settings; storage keys in rows | migrations from the app lifespan (here a one-shot `db-migrate` service runs them, so no app process does DDL); a broker without persistence (their Redis has no AOF and no `maxmemory-policy`); starting without an Alembic naming convention (they cannot add one later without renaming constraints) |

The two products share these patterns on purpose (owner request): job row = Celery id, partial
unique active-job index, advisory-locked migrations, explicit queue table, one worker service per
queue.
