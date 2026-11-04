# B2 stage 5: GPU queues (training, model export, the evaluator) and the model registry

| Field | Value |
|---|---|
| Status | plan, not started |
| Depends on | stage 3 merged ([03e](03e_b2_s3_job_runtime.md)) |
| Design | [02b](02b_jobs_locking_events.md) sections 2.1, 3 and 7.3; [02a](02a_postgres_schema.md) section 7 |
| Decisions | D-41 (owner decision 4: training and model export run on Celery; the file protocols are deleted), D-31 (the evaluator on `gpu_eval`, approved), D-22b (promote `wait=true` deleted) |
| Lanes | 5A, 5B, 5C, 5D in parallel |
| Merge order | 5D, 5A, 5B, 5C |
| Runs alongside | stage 4, disjoint files |
| Compose project | none in the lanes; no GPU in any lane (handlers are tested with fakes; the GPU smoke runs in a budgeted GPU window, below) |

The three GPU images (trainer, export, evaluator) import only `src/tasks/runtime/` and the database
layer, and build their Celery app with `src/tasks/runtime/gpu_app.py::make_app(include=[<handler module>])`.
Each GPU consumer runs `-c 1 --max-tasks-per-child=1` (5 for the evaluator), so every run gets a
fresh child and releases GPU memory when it ends.

---

## Lane 5A `b2/gpu-train`: the trainer becomes a `gpu_train` consumer

**Owns:** `docker/trainer/{trainer.py,job_protocol.py,job_claim.py,job_status.py,trainer_callbacks.py,gpu_select.py}`
and new `docker/trainer/celery_tasks.py`; `docker/trainer/Dockerfile` (runtime deps and the copy of
`src/tasks/runtime`, `src/db`, `src/config`, `src/domain`, `src/utils`; B3 owns the Ultralytics line
and the training code, and rebases after B2); `src/services/training/{jobs,job_files,preflight_checks,preflight_scan,trainer_reachability,lineage,run_retention}.py`;
the train routers (`rg -ln "train/start|train/status|train/preflight" src/routers`);
`src/config/job_kinds/gpu_train.py`, `src/config/task_routes/gpu_train.py`; `docker/test/fake_trainer.sh`
replaced by a fake Celery handler under `docker/test/` (the harness compose edit itself is 6B's);
tests under `tests/training/` and `tests/trainer/`.

| ID | Task | Red test | Control |
|---|---|---|---|
| 5A-1 | `train/start` calls `start_job('train', ...)` with a `training_runs` row (spec, class names in output order from the registry, the export snapshot key) in the same transaction; no `job.json`, no `atomic_create_bytes` caller | `tests/training/test_train_start_job.py` (`db`): one row each in `jobs` and `training_runs`, one message on `gpu_train`; no file under the old jobs dir | preflight unchanged |
| 5A-2 | `docker/trainer/celery_tasks.py`: `register_job('train', handler)`; the handler runs the existing fit with progress from the epoch callback (`ctx.progress`), cancel checked between epochs (`ctx.cancelled()`, replacing `<id>.cancel`), the log and checkpoints written through the storage seam under the run's `artifact_prefix`, the manifest stored in `training_runs.manifest` | `tests/trainer/test_train_handler.py` (fake fit): progress per epoch, cancel at an epoch boundary -> `cancelled`, an exception -> `failed` with the message | a fake fit completes |
| 5A-3 | delete the file protocol: `job_protocol.py`, `job_claim.py` (the `O_EXCL` generations), `job_status.py` status files, the 2 s poll in `trainer.py`, `job_files.py`, the claim-file cleanup in `run_retention.py` | `tests/trainer/test_no_file_protocol.py`: `rg`-equivalent scan of `docker/trainer` and `src/services/training` for `.job.json`, `.status.json`, `claim.g`, `.cancel` is empty | - |
| 5A-4 | trainer reachability and capabilities: at worker start the trainer writes `worker_capabilities` (GPUs, families); `/train/preflight` reports "trainer available" when `inspect().active_queues()` shows a `gpu_train` consumer (timeout 1 s) and its capabilities row is fresh; `.trainer_capabilities.json` and the container-name probe (`trainer_reachability.py`, `lineage.py`'s `containers.get`) deleted; `OP_GPU_ARBITER_TRAINER_CONTAINER` retired | `tests/training/test_trainer_reachability_celery.py` (fake inspect): no consumer -> "not running"; consumer + fresh row -> available | - |
| 5A-5 | the training image imports: a fresh interpreter in the trainer environment imports `docker/trainer/celery_tasks.py` without FastAPI or OpenSearch (`tests/trainer/test_runtime_import_minimal.py`, run with the trainer requirements in CI's existing trainer job if present, else in the sweep's image check) | the test | - |

## Lane 5B `b2/gpu-export`: model upload and export become `gpu_export` jobs

**Owns:** `src/services/model_export.py`; `export/export_worker.py` (the watch loop, the
`queue/`/`claimed/` rename claim, `reconcile_claimed`, the heartbeats: all deleted) and new
`export/celery_tasks.py`; the root `Dockerfile` (the `export` and `evaluator` targets: Celery and the
runtime package; the `api` target already has them from stage 1); the model routers for
`POST /models/upload` and `POST /models/{model_name}/export` and the status route;
`src/schemas/models.py` (the export status); `src/config/settings.py` (the `/jobs/model_export`
setting removed); `src/config/job_kinds/gpu_export.py`, `src/config/task_routes/gpu_export.py`; tests
under `tests/models/`.

| ID | Task | Red test | Control |
|---|---|---|---|
| 5B-1 | upload and export call `start_job('model_export', ...)` (exclusive per Triton model name) with a `models` row (or a new version) and a `model_engines` row `building`; the "worker heartbeat older than 15 s -> 503" check becomes "no `gpu_export` consumer -> 503" (the routes keep answering 503 while the worker is down) | `tests/models/test_export_start_job.py` (`db`): one job, one message; a second export of the same name while active -> 409; no consumer -> 503 | - |
| 5B-2 | `export/celery_tasks.py`: the existing build-and-load as the handler, the `ExportStatus` phases (`validating`, `exporting`, `loading`) as `progress.stage`, the engine row `ready` or `failed`; **cancel added** (checked between phases; the export had none) | `tests/models/test_export_handler.py` (fake build): phases in order; cancel between phases -> `cancelled` and the partial engine removed | a fake build completes |
| 5B-3 | delete `/jobs/model_export`, both heartbeat files, the 4 h `EXPORT_TIMEOUT_S` constant (the kind's time limit replaces it), `reconcile_claimed` | `tests/models/test_no_export_job_files.py` | - |

## Lane 5C `b2/gpu-eval`: the bake-off evaluator becomes a `gpu_eval` consumer

**Owns:** `scripts/curation/bakeoff/bakeoff_runner.py` (the job-file watch and rename deleted) and
new `scripts/curation/bakeoff/celery_tasks.py`; the bake-off routers and the start service
(`rg -ln "bakeoff_jobs|bakeoff/run" src`); `src/config/job_kinds/gpu_eval.py`,
`src/config/task_routes/gpu_eval.py`; tests under `tests/bakeoff/`.

| ID | Task | Red test | Control |
|---|---|---|---|
| 5C-1 | `bakeoff/run` calls `start_job('bakeoff', ...)` with a `bakeoff_runs` row; no `bakeoff_jobs/*.job.json` | `tests/bakeoff/test_bakeoff_start_job.py` (`db`) | - |
| 5C-2 | the handler runs the existing evaluation; results under `results_storage_key`; at-least-once re-run removed (a redelivery is a claim no-op) | `tests/bakeoff/test_bakeoff_handler.py` (fake backend): one run for two deliveries | a fake evaluation completes |
| 5C-3 | `OP_BAKEOFF_JOBS_DIR` and the per-project `bakeoff_jobs` directories retired; 5C adds `src/db/repos/bakeoff.py::active_bakeoffs()`, which the GPU arbiter (6A's file) calls instead of scanning job files | retired-env test; `tests/bakeoff/test_active_bakeoffs.py` (`db`) | - |

## Lane 5D `b2/promote-models`: promotion and the model registry

**Owns:** `src/services/training/{promote_job,promote_json,triton_promote,model_classes}.py`,
`src/services/model_repository.py`, `src/routers/curation/train/promote.py`, the model-sharing route
(`rg -ln "models/{model_name}/sharing" src/routers`), new `src/db/repos/models.py`, `src/tasks/promote.py`,
`src/config/job_kinds/promote.py`, `src/config/task_routes/promote.py`, tests under `tests/promote/`.

| ID | Task | Red test | Control |
|---|---|---|---|
| 5D-1 | `promote` is a job on `jobs` (exclusive per Triton model name): `models` row (labels in output order from the run), `promotions` row with the previous live row `superseded_at` in the same transaction (partial unique index `uq_promotions_live`); `labels.txt` and the class remap written from the rows | `tests/promote/test_promote_job.py` (`db`): two concurrent promotes of one name -> one live promotion; the old one superseded | - |
| 5D-2 | delete `promote.json` and `promote_json.py` (the `PROMOTE_JSON` lease), `run_detached`, `_detached`; `model_classes.py` and `model_repository.py` read `models`, `promotions`, `model_shares` | `tests/promote/test_no_promote_json.py` | the promoted-model listing route returns the same shape |
| 5D-3 | D-22b: `wait=true` removed from the promote route (only tests used it: `rg -n "wait=true" tests \| rg promote`); those tests use the job path; contracts | `tests/promote/test_promote_no_wait_mode.py`: the parameter is absent from OpenAPI | - |
| 5D-4 | sharing: `PUT .../models/{model_name}/sharing` writes `model_shares`; the consumer maps classes by name (unchanged rule) | `tests/promote/test_model_share_by_name.py` | - |

## GPU check (no lane runs a GPU)

The stage's real-GPU smoke (one short training run on COCO-sized public data, one export of a small
model, one bake-off on a 20-image set) runs inside **GPU window 1** (the B3 gate,
[00_index.md](00_index.md) section 8.1) on the merged B2 + B3 stack, not as an extra window: B3's
engine export already exercises the export worker, and the training smoke takes about 30 minutes.

## Stage 5 exit criteria

- every lane's tests green on the merged branch; `-m db` green; contracts clean;
- `rg -n "\.job\.json|\.status\.json|claim\.g|/jobs/model_export|bakeoff_jobs|promote\.json|OP_TRAIN_JOBS_DIR|OP_GPU_ARBITER_TRAINER_CONTAINER" src scripts docker export`
  lists only `src/services/upgrade/` (the 0.6.0 reader);
- compose facts for 6B in the PR notes: the three consumer commands, the removed `/jobs` mounts, the
  trainer and export healthchecks.

## Lane briefs

Use the common block of [03a](03a_b2_w1_queue_infra.md), with: "B2 stage-5 lane <5A..5D>", "cut
from int/b2-control-plane after stage 3 merged", this file (your lane) and [02b](02b_jobs_locking_events.md)
sections 2-3 and 7.3; "No GPU: handlers are tested with fakes"; for 5A: "B3 also edits
`docker/trainer/`: touch only the files listed for 5A; leave the Ultralytics pin and the fit code to
B3." Compose project: none. GPU: none.
