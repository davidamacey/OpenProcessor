# 10: code standards and "no hand-rolling" (cross-cutting)

Status: plan, not started. Nothing in this file has been implemented.
Depends on: [00_index.md](00_index.md) and the batch files: B0 [01_b0_close_out.md](01_b0_close_out.md);
B1 tracks 1A B1 1A (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)), 1B B1 1B (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)),
1C B1 1C (merged; see [CHANGELOG](../../../CHANGELOG.md)), 1D B1 1D (merged; see [CHANGELOG](../../../CHANGELOG.md)),
1E B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md)); B2 [03_b2_overview.md](03_b2_overview.md)
(W1 [03a_b2_w1_queue_infra.md](03a_b2_w1_queue_infra.md), W2 [03b_b2_w2_job_families.md](03b_b2_w2_job_families.md),
W3 [03c_b2_w3_beat_reaper_workers.md](03c_b2_w3_beat_reaper_workers.md)); B3
[04_b3_ultralytics_one_version.md](04_b3_ultralytics_one_version.md); B4 [05_b4_stateless_api.md](05_b4_stateless_api.md);
R-final [06_release_acceptance.md](06_release_acceptance.md); B5 [07_b5_overview.md](07_b5_overview.md)
(5b [07b_b5_decode_once.md](07b_b5_decode_once.md)); [09_github_actions.md](09_github_actions.md).
Companion: [08_environment_and_setup.md](08_environment_and_setup.md) (ENV-n findings).
Source IDs: CC-1, CC-5, CC-7, CC-9, CC-10, CC-11, BA-3, BA-8, GPU-3b, GPU-8, GPU-12, OS-7
(`codebase_audit_plan.md`, including Appendix B "negative results", C "refuted" and D "cut as
churn"); Q-Phase 2..7, Q-J7 (`task_queue_plan.md`); U-Step 3, U-Step 5, U-Step 6
(`model_family_versions_plan.md`); the former master plan (rev 4, in git history) sections 3,
3b, 17, which [00_index.md](00_index.md) section 12 maps. Decisions are in the register at
[00_index.md](00_index.md) section 6 and are not restated here.

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read only; the later pin commit
`d1c6bab0` touches no file cited here).
Consolidated 2026-11-03.

What to keep after merge: sections 1 and 2 (the rules and the ratchets) move into
`docs/design/engineering_standards.md` and `CONTRIBUTING.md` as each ratchet lands; the library
register (section 3) moves to `docs/ARCHITECTURE.md` (mechanism, library, source). The files go as
described in [00_index.md](00_index.md) "Plan lifecycle".

## 0. How to use this file

- Section 1 is the rule set every batch is held to, with where each rule is written and what
  enforces it. It summarises; it does not replace
  [`docs/design/engineering_standards.md`](../engineering_standards.md),
  [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) or [`docs/VISION_AND_GOALS.md`](../../VISION_AND_GOALS.md).
- Section 2 lists the ratchets: the ones that exist and the ones the plans add.
- Section 3 is the library-per-mechanism register. A batch that builds a mechanism uses the
  register's choice or changes the register in the same PR, with the doc cited.
- Section 4 is the hand-roll audit (`LIB-n`), section 5 the standards findings (`STD-n`), section 6
  the custom code that is justified and stays.
- Every finding carries evidence verified now, the doc URL, the batch, a size and its proof test.
  Re-find code by symbol; line numbers move.

## 1. The standards (consolidated)

| # | Rule | Written in | Enforced by |
|---|---|---|---|
| 1 | **Correct first, under concurrency; speed measured once the stack stops changing** | VISION "core mission"; [00_index.md](00_index.md) section 1 | batch order (B1-B4 before B5); review |
| 2 | **Fail closed**: an outage, an unbound project, a stale registry or an ambiguous state refuses; a broad `except` that returns `None/False/[]/0` is a bug unless the fallback is the feature | VISION; engineering_standards 1.3 | broad-except ratchet (`check_ruff_ratchet.py`); ruff `S110`/`S112`; review |
| 3 | **No dead code, no duplicate logic, no shims or aliases**; a superseded route, field or helper is deleted in the same change | VISION; engineering_standards 1.5; CLAUDE.md | vulture ratchet (`check_vulture.py`); review; STD-1, STD-2 |
| 4 | **Retired settings are refused at startup**, naming the replacement; no alias is read | CLAUDE.md; `src/config/retired_env.py`; `scripts/lib/retired_env.sh`; engineering_standards 5.18 | entry-point calls (ENV-6 adds a test) |
| 5 | **Class identity is the name**, never a raw index, at every boundary (import, combine, export, train, promote, sharing) | VISION; CLAUDE.md | per-boundary tests; U-C10 in B1-1E |
| 6 | **Docs and maintained libraries before hand-rolling**: find the documented pattern or library, cite it, justify custom code against it | owner rule (2026-11-03); global agent instructions | section 3 register; STD-6 PR checkbox; review asks "is there a library for this?" |
| 7 | **No in-process or file locks for cross-process state**; use OCC, leases, idempotency, single-consumer queues | owner rule; the former master plan 3.2 (rev 4, in git history) | `tests/test_no_module_level_async_locks.py` (exists); file-lock ratchet (B1-1A, new); review |
| 8 | **No asyncio fan-out in request handlers**; where concurrency is truly needed, `asyncio.TaskGroup` with kept task references, bounded by a `Semaphore`; asyncio primitives never shared across loops | owner rule; Python docs (TaskGroup cancels siblings on failure, `gather` does not) | review; Q-J7 exemption documented in engineering_standards (B4-4A); `create_task` ratchet (B4-4B, new) |
| 9 | **Heavy or long work goes to workers** (Celery from B2); FastAPI `BackgroundTasks` only for small same-process work | owner rule; `task_queue_plan.md` section 5 | B2 done-when: no job runs inside an API worker |
| 10 | **Environment through `src/config/env.py`**; a malformed value refuses startup naming the variable | CONTRIBUTING (`TID251` row); engineering_standards 2.1 | ruff `TID251`; `tests/test_env_surface.py`; ENV-1, ENV-5 extend it |
| 11 | **`.env` settings tiered** (basic vs advanced) | owner rule | ENV-8 tier test (new, B4) |
| 12 | **`docker compose` with `-p` always** from a worktree or harness; local-only image tags for local builds | CLAUDE.md; CONTRIBUTING | review; ENV-9 |
| 13 | **Per-project `.venv`, binaries called by path**, never `source` | CLAUDE.md; CONTRIBUTING | review |
| 14 | **Production Dockerfiles**: multi-stage, slim base, non-root final `USER`, `FROM` pinned by digest; Trivy before publishing | owner rule; engineering_standards 3.2 | `tests/test_dockerfile_hygiene.py`; `security-scan.yml`; release gate |
| 15 | **Conventional commits, merge commits (never squash), never bypass hooks** | owner rule; CONTRIBUTING | `conventional-pre-commit` hook (`commit-msg`); review |
| 16 | **Evidence first**: every fix has a red test seen failing on the base, plus a paired control; PR descriptions record the red output; every new guard gets a plant-a-violation proof | engineering_standards 1.7; the former master plan 3.6 (rev 4, in git history) | review; guard unit tests in `tests/codegen/` |
| 17 | **Wire contract**: one serializer; `make contracts` in the same commit as any route or model change | CLAUDE.md | `api-contracts-drift`, `region-status-ts-drift` hooks |
| 18 | **Comments say non-obvious why**; no wave, round or work-package ids, no history names | engineering_standards 1.4 | `check_process_ids.py` |
| 19 | **Split by concern, never trim to fit** (700 lines; CLAUDE.md files 400) | engineering_standards 1.2 | `check_file_size.py` |
| 20 | **Layering**: routers -> services -> clients; `domain`, `config`, `utils`, `schemas` import neither services nor clients; only routers import FastAPI; from B2 a `tasks` layer that imports services and nothing imports | CONTRIBUTING; Q-Phase 2 | `check_layering.py` (B2-W1 registers `tasks`) |
| 21 | **Baselines only shrink**; fix and `--tighten`, never add a line to hide a finding | CONTRIBUTING; engineering_standards 3.1 | every ratchet's stale/loosened checks |
| 22 | **Public tree**: no private names, machine paths or secrets; public data (COCO, Open Images) in tests and docs; security findings as neutral one-liners | CONTRIBUTING; CLAUDE.md | `check_naming_leaks.py`; gitleaks; review |
| 23 | **Review depth follows risk**: isolation, locking, concurrency and security get characterization tests and an independent review | VISION; engineering_standards 1.8 | review |
| 24 | **Agent lanes**: one agent per worktree and branch; lanes run targeted tests only (the [Test and GPU budget](00_index.md): T0 per task, one sweep per wave, three live-harness runs, three GPU windows); briefs forbid `pgrep -f`, `pkill -f` and sleep/until loops and name the `-p` project, the GPU, the worktree and the PR deadline | global agent instructions; [00_index.md](00_index.md) section 7 | brief template; coordinator |

## 2. Ratchets and guards

### 2.1 Existing (do not weaken)

| Guard | What it stops | Where |
|---|---|---|
| `check_file_size.py` | files over 700 lines (warn band 640-700) | `scripts/codegen/`, baseline `file_size.txt` |
| `check_layering.py` | imports against the layer contract | `scripts/codegen/` (no baseline; one reasoned exception) |
| `check_ruff_ratchet.py` | complexity above 15; new `except Exception` | baselines `complexity.txt`, `broad_except.txt` |
| `check_process_ids.py`, `check_cited_paths.py`, `check_vulture.py`, `check_naming_leaks.py` | process ids, dead cited paths, dead code, private names | `scripts/codegen/` |
| `noqa-needs-reason` | a coded `noqa` without a reason | `.pre-commit-config.yaml` |
| `tests/test_no_module_level_async_locks.py` | module-level asyncio primitives (allowlist: `open_vocab_ingest.py::_PASS_LOCK`, deleted in B2-2C with its allowlist line) | `tests/` |
| `tests/test_no_legacy_region_scalars.py` | single-box region scalars | `tests/` |
| `tests/test_env_surface.py`, `check_docs_vs_code.py` | env drift both directions; docs naming routes or vars that do not exist | `tests/`, `scripts/docs/` |
| `tests/test_dockerfile_hygiene.py`, `tests/test_version_consistency.py`, `tests/test_requirements_contract.py`, `tests/test_dependency_manifest.py` | image hygiene, version drift, pin drift, undeclared imports | `tests/` |

### 2.2 Added by the plans

| Guard | Batch | Rule | Start state (verified now) | End state |
|---|---|---|---|---|
| `scripts/codegen/check_file_locks.py` + `baselines/file_locks.txt` | B1-1A | no new `exclusive_start_lock`, `exclusive_file_lock` or `fcntl.flock` hit; shrink-only | `rg -c "exclusive_start_lock\|exclusive_file_lock\|fcntl\.flock" src` gives **19 files, 55 matching lines** today (the master plan's "18 files, 51 hits" is older). Generate the baseline with the script; never copy a count from a plan (STD-4) | B1 done-when: only the event hub, dispatcher, sweeper, cache prune and `job_lock.py`; B2: event hub and `job_lock.py`; **B4: zero, baseline file deleted** |
| module-level state guard (extend `tests/test_no_module_level_async_locks.py`) | B4-4B | no new module-level lock, semaphore or mutable dict outside an allowlist with a reason (Q-Phase 7) | allowlist has one entry until B2-2C | allowlist empty or reasoned per entry |
| `create_task` ratchet | B4-4B | no `asyncio.create_task` outside an allowlist | 61 matching lines in 35 files under `src` today; most belong to in-API jobs that B2 deletes | shrink-only baseline taken **after** B2 merges (STD-3) |
| `tests/test_no_handrolled_atomic_write.py` | B1-1A (with LIB-1) | no temp-name-plus-`replace` outside `src/utils/atomic_write.py` | 7 sites in `src` (LIB-1) | zero |
| `tests/test_retired_env_entrypoints.py` | B1-1E / B2-W1 | every Python entry point a compose service runs calls `reject_retired_env` | 4 entry points miss it (ENV-6) | zero |
| env tier test | B4 | every `env.template` assignment sits under one tier header | none (ENV-8) | enforced |
| Celery guards: `tests/test_task_routes_complete.py`, `tests/test_queue_consumers_from_compose.py`, `tests/test_celery_config.py` | B2-W1 | every task routed, no decorator `queue=`, every routed queue consumed, the delivery and time-limit settings ordered | n/a | enforced |

## 3. Library-per-mechanism register

Version policy: **exact** (`==`, or a digest for an image) where the behaviour is part of a
contract or a gate; **floor** elsewhere, with dependabot's `increase-if-necessary`
([08_environment_and_setup.md](08_environment_and_setup.md) 7.1).

| Mechanism | Choice | Version policy | Official doc | Where it applies |
|---|---|---|---|---|
| Job transport, routing, delivery | Celery 5.6 on Redis 8 (`acks_late`, `reject_on_worker_lost`, `prefetch_multiplier=1`, `visibility_timeout` above the longest job, soft shutdown) | `celery[redis]==5.6.<x>`; Redis image by digest | https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html ; https://docs.celeryq.dev/en/stable/userguide/workers.html ; https://docs.celeryq.dev/en/stable/userguide/routing.html | B2: `src/tasks/`, `src/clients/task_queue.py`, compose services |
| Periodic work | Celery beat, exactly one `jobs-beat` container | as above | https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html | B2-W3: replaces the lifespan loops, dispatcher, sweeper, cache prune, cluster-refresh daemon |
| Leases and claims | OpenSearch optimistic concurrency: `op_type=create`, `if_seq_no`/`if_primary_term` -> 409 | client floor | https://docs.opensearch.org/latest/api-reference/document-apis/index-document/ ; lock-document precedent: https://docs.opensearch.org/latest/monitoring-your-cluster/job-scheduler/locks/ | B1-1A `src/clients/job_lease.py` over `src/clients/occ.py` |
| Claims without OpenSearch (trainer) | `os.open(path, O_CREAT \| O_EXCL)` generation files | stdlib | https://docs.python.org/3/library/os.html#os.open | B1-1A `docker/trainer/job_protocol.py`, `src/services/training/jobs.py` |
| OCC writes and bulk | the in-repo helpers in `src/clients/occ.py`, `occ_bulk.py` (Appendix B: correct); per-item `if_seq_no` in bulk | n/a | https://docs.opensearch.org/latest/api-reference/document-apis/bulk/ | every automated writer (lock rule) |
| Atomic files | `tempfile.NamedTemporaryFile(dir=parent, delete=False)` / `mkstemp`, `flush`, `os.fsync`, `os.replace`, through `src/utils/atomic_write.py` | stdlib | https://docs.python.org/3/library/tempfile.html ; https://docs.python.org/3/library/os.html#os.replace | every state file (LIB-1) |
| Caches | `functools.lru_cache` / `cache` for pure functions; `cachetools.TTLCache`/`LRUCache` with a `threading.Lock` (cachetools is not thread-safe) for keyed, expiring caches | cachetools floor | https://docs.python.org/3/library/functools.html#functools.lru_cache ; https://cachetools.readthedocs.io/en/stable/ | BA-3 sites, LIB-2 |
| Retries and backoff (HTTP) | `tenacity` (already used by `src/services/labeling/vlm_client.py`) | floor (`>=8.2`) | https://tenacity.readthedocs.io/en/latest/ | VLM HTTP; new HTTP retry code uses it |
| Retries (Triton) | the in-repo pool retry, fixed by GPU-8 (classify deterministic errors, never retry them); tenacity was **cut as churn** for it (audit Appendix D) | n/a | Triton client docs (GPU-8 in the audit plan) | `src/clients/triton_pool_core.py`, `src/utils/retry.py` |
| Retries (OpenSearch transport) | the client's own `max_retries`, `retry_on_status`, `retry_on_timeout` (`opensearchpy/transport.py` `Transport.__init__` docstring); per-call retries only for a status the transport must not retry globally (for example 429 on one admin call) | floor | opensearch-py `Transport` docstring (installed package) | factory in `src/clients/opensearch/` |
| Rate limiting | the per-process token bucket in `vlm_client.py` (kept, section 6); no cross-process limiter is planned | n/a | n/a | VLM calls |
| Bounded concurrency | `asyncio.TaskGroup` + `asyncio.Semaphore`; single-consumer Celery queues (`-c 1`) for global serialization; `asyncio.to_thread` / the shared CPU pool for blocking work in a handler | stdlib | https://docs.python.org/3/library/asyncio-task.html#task-groups ; https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore ; https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread | CC-5, CC-7, CC-9, Q-J7 exemption |
| Event streams | Redis Streams: `XADD ... MAXLEN ~`, `XREAD BLOCK`, stream ids as SSE ids | Redis image digest | https://redis.io/docs/latest/develop/data-types/streams/ | B4-4C, after the spike (D-30) |
| Server-sent events | `fastapi.sse.EventSourceResponse` + `ServerSentEvent` (in FastAPI since 0.135.0; the pin is 0.143.3) | fastapi `==` | https://fastapi.tiangolo.com/tutorial/server-sent-events/ | LIB-3, in B4-4C |
| Image decode and EXIF orientation | Pillow `ImageOps.exif_transpose`; OpenCV / simplejpeg for fast decode where orientation is handled; one oriented decode helper in `src/utils/` (GPU-3b) | Pillow floor (declare it, ENV-11) | https://pillow.readthedocs.io/en/stable/reference/ImageOps.html | `src/utils/image_processing.py`, `pe_preprocess.py`, `cascade_detect/preprocess.py`; GPU-3b in B5b |
| CPU letterbox | the in-repo `src/services/detection/geometry.py` (`letterbox_params`, `letterbox_to_square`, `undo_letterbox`), because callers need the scale and pad; Pillow `ImageOps.pad` where only the padded image is needed | n/a | https://pillow.readthedocs.io/en/stable/reference/ImageOps.html | LIB-5, LIB-6, STD-1, GPU-12 |
| GPU image transforms | today: nvJPEG through ctypes and CuPy `RawKernel` resamplers in `models/gpu_image_prep/1/` (a measured choice, section 6); candidates to evaluate before new kernels: CV-CUDA (`pillowresize`, `copymakeborder`, `resize_crop_convert_reformat`), DALI, nvImageCodec | pinned in the Triton image | https://cvcuda.github.io/CV-CUDA/ ; https://docs.nvidia.com/deeplearning/dali/user-guide/docs/ ; https://docs.nvidia.com/cuda/nvimagecodec/ ; https://docs.cupy.dev/en/stable/user_guide/kernel.html | LIB-8, B5b |
| Polygon offsetting (OCR DB unclip) | `pyclipper` (as PaddleOCR's `DBPostProcess`); OpenCV `findContours`, `minAreaRect`, `boxPoints` | `pyclipper==1.4.0` | https://pypi.org/project/pyclipper/ ; https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html | `models/ocr_pipeline/1/ocr_postprocess.py` (GPU-2, done) |
| ONNX export and NMS | Ultralytics export `nms` argument (`False` = NMS-free head where available, `True` = embedded NMS); `torchvision.ops.nms`/`batched_nms` in Python paths; `ultralytics.utils.nms.non_max_suppression` in the probe | ultralytics `==` | https://docs.ultralytics.com/modes/export/ ; https://docs.pytorch.org/vision/stable/ops.html | B1-1E, B3 (`export/ultralytics_export.py`); `src/services/detection/ensemble_nms.py` (Appendix B: correct) |
| TensorRT build | TensorRT Python API / `trtexec` in the export image; ModelOpt AutoCast for FP16 | `tensorrt-cu13==` the Triton image's TensorRT | https://docs.nvidia.com/deeplearning/tensorrt/latest/ ; https://nvidia.github.io/TensorRT-Model-Optimizer/ | `export/`, `scripts/lib/export*.sh` |
| Metrics | `prometheus_client`; multiprocess mode for uvicorn workers and Celery children (`PROMETHEUS_MULTIPROC_DIR`, `mark_process_dead`); `start_http_server` for side-car workers | floor | https://prometheus.github.io/client_python/multiprocess/ ; https://prometheus.github.io/client_python/exporting/http/ | `src/core/metrics.py`, `ops_metrics.start_worker_metrics_server`, OPS-2 in B2, LIB-4 |
| HTTP clients | `httpx` (async; VLM, segmenter, loopback); `requests` in sync scripts and the trainer; `aiohttp` only as opensearch-py's async transport | floor | https://www.python-httpx.org/ | everywhere; LIB-4 removes the one direct `aiohttp` server |
| Schema validation | pydantic v2 models for wire bodies and stored config documents; pydantic-settings only where the field list is the config (ENV-1) | `pydantic==` in tests | https://docs.pydantic.dev/latest/ ; https://docs.pydantic.dev/latest/concepts/pydantic_settings/ | `src/schemas/`, config store |
| Python CLIs | `argparse` | stdlib | https://docs.python.org/3/library/argparse.html | `scripts/` |
| Shell | bash with the shared `scripts/lib/*.sh` helpers; `shellcheck --severity=warning` | hook rev pinned | https://www.shellcheck.net/wiki/ | installer, CLI |
| Worker recycling and count | uvicorn `--limit-max-requests` + `--limit-max-requests-jitter` (0.41.0+); `WEB_CONCURRENCY` when `--workers` is absent | `uvicorn>=0.41.0` | https://github.com/encode/uvicorn/blob/master/docs/settings.md ; https://github.com/encode/uvicorn/releases/tag/0.41.0 | compose `api` (OPS-1 done); B4-4B |
| Vulnerability and licence scanning | Trivy (`fs`, `image`, `--scanners license`) | action and binary pinned | https://trivy.dev/latest/docs/scanner/license/ | CI `security-scan.yml`, release gate |

## 4. Hand-roll audit (LIB-n)

Method: `rg` for the shapes in the brief (retry loops, `OrderedDict`/`move_to_end`, TTL arithmetic,
`.tmp` + `replace`, `Semaphore`/token buckets, `isinstance(..., dict)` validation, letterbox,
EXIF, NMS/IoU, `pyclipper`/contours, `while True` + `sleep` loops, `text/event-stream`, bash
`case` parsing), then each hit read and checked against the audit's Appendix B (clean), C
(refuted) and D (cut). Only real findings are listed; the justified ones are in section 6.

| ID | Finding (evidence verified now) | Library or pattern (doc) | Covered? | Batch | Size | Risk of changing | Proof |
|---|---|---|---|---|---|---|---|
| LIB-1 | Hand-rolled temp-file writes remain outside `src/utils/atomic_write.py` after CC-1 (CC-1's search pattern missed them): `src/services/curation/export_support.py::atomic_write_text` (a **second** atomic-write helper, fixed `<path>.tmp`); `src/services/detection/fp_store.py::save` (fixed `centroids.faiss.tmp`, `metadata.json.tmp`); `src/services/visual_search/clustering_persist.py::_save_index` (fixed `<name>.index.tmp`, `<name>.meta.tmp.npy`); `src/services/curation/dataset_import/image_load.py` (fixed `.<name>.tmp` copy); `src/services/curation/image_serving.py` and `clustering/last_run.py` (pid-only names: two threads of one process share them); `src/services/training/promote_json.py` (pid-only; B1-1A makes it single-writer). Find with `rg -n "\.tmp['\"]\|with_suffix\([^)]*\.tmp\|\.tmp\.\{os.getpid" src` | stdlib `tempfile` + `os.replace` via `atomic_write_bytes`/`atomic_write_json`; for writers that need a path (`faiss.write_index`, `np.save`, `shutil.copyfile`) add one `atomic_write_via(path, write_fn)` helper using `mkstemp(dir=path.parent)`, as `export/trt_utils.py` already does | **CC-1** (its done-when says "every site"); this extends its site list | B1-1A | S | low: whole-file replace semantics unchanged; `clustering_persist` writes two files, keep its order (metadata, then index) | red: 4 spawned writers x 500 calls to `fp_store.save` and to `atomic_write_text` on one path give `FileNotFoundError` today, 0 after; the new `tests/test_no_handrolled_atomic_write.py` scan fails today on the 7 sites |
| LIB-2 | `src/utils/cache.py::EmbeddingCache` is a hand-rolled LRU+TTL (an `OrderedDict`, `move_to_end`, a timestamp per entry, an `RLock`), used by `src/services/embedding.py` and `src/services/inference.py` (`get_image_cache`, `get_text_cache`) | `cachetools.TTLCache(maxsize, ttl)` under a `threading.Lock` (cachetools docs: the classes are not thread-safe) | new; BA-3 adds the `cachetools` dependency in the same track | B1-1D (after the BA-3 commit) | S | low: keep `get_stats()` keys (`size`, `max_size`, `hits`, `misses`, `hit_rate`, `ttl_seconds`) and the defensive `copy()` on `set` | TTL expiry with an injected `timer`; `maxsize+1` inserts evict the oldest; hit and miss counts equal before and after on a scripted sequence |
| LIB-3 | Two hand-written SSE framers: `src/routers/curation/events.py::sse_response` (`': connected'`, `': hb'` every `_SSE_HEARTBEAT_SECONDS`, `event:`/`data:` lines, no `id:`) and `pipeline_events.py::_sse` (`': keepalive'`), each setting `Cache-Control`/`X-Accel-Buffering` by hand | `fastapi.sse.EventSourceResponse` with `ServerSentEvent(data=..., event=..., id=...)`: keep-alive ping every 15 s, `Cache-Control: no-cache`, `X-Accel-Buffering: no` built in; resume by reading `Last-Event-ID` as a header parameter (FastAPI SSE tutorial) | covered by **B4-4C** (Redis Streams; stream ids become SSE ids, FE-7's root cause); this names the library 4C uses | B4-4C | M (inside 4C) | medium: Cropwright's `EventSource` handlers (event names, the initial comment) must keep working; browser check of every live page | contract test: frames carry `event`, `data`, `id`; a reconnect with `Last-Event-ID` skips seen ids; heartbeat at 15 s; Cropwright live pages checked at 1440 and 430 px |
| LIB-4 | The region worker stands up its own `aiohttp.web` `/metrics` server (`scripts/curation/worker/runner.py::_start_metrics_http_server`), while the other workers use `src/services/curation/ops_metrics.py::start_worker_metrics_server`, a wrapper over `prometheus_client.start_http_server` (threaded, no event loop, a taken port is logged and skipped) | `prometheus_client.start_http_server` through the existing helper | new | B1-1C (CC-8 edits `runner.py` there) | S | low: same registry, same port variable (`OP_REGION_WORKER_METRICS_PORT`); `aiohttp` stays installed for opensearch-py | red: a test that the worker's metrics start goes through `start_worker_metrics_server` (patched) fails today; live: the Prometheus target is up in the B1 harness run |
| LIB-5 | `src/services/curation/probe_models.py::_letterbox_yolov5_objectness` re-implements `geometry.letterbox_to_square` (its docstring claims it is "not a second, potentially-drifted implementation") | the in-repo `src/services/detection/geometry.py::letterbox_to_square`, then scale to `[0, 1]` | covered by **U-Step 3** (B1-1E edits this probe to read its input size from the ONNX) | B1-1E | XS | low: same bilinear resize, centered pad, 114 | the probe tensor equals `letterbox_to_square(img, target)/255` on a COCO fixture; the U-Step 3 real-ONNX probe test passes |
| LIB-6 | `src/services/curation/export_support.py::_letterbox_pil` (training-export letterbox: LANCZOS resize, paste centered on a 114 canvas) is what Pillow's `ImageOps.pad(img, (t, t), method=LANCZOS, color=(114, 114, 114))` does | Pillow `ImageOps.pad` | new | backlog (when the file is next touched) | XS | low but real: Pillow centres by rounding while the code floors, so odd margins may shift by one pixel in exported training images; accept or keep | golden test on odd and even sizes; if any pixel differs, record it and decide |
| LIB-7 | The re-scan around an `update_by_query` in `src/services/projects/legacy_repair.py::_backfill_with_retry` retries a 429 by hand around `backfill_embedded_state` | OS-7's fix (run the update as a task and poll it) owns the retry; the client's transport retry is global and must not retry an `update_by_query` blindly | covered by **OS-7** (B1-1C); fold the 429 handling into the poll, no separate task | B1-1C | XS | low | OS-7's own red test; plus a 429 on the poll is retried and a 400 is not |
| LIB-8 | GPU image transforms: the CuPy `RawKernel` resamplers and blur in `models/gpu_image_prep/1/prep_kernels.py` and the ctypes nvJPEG lanes are a documented, measured choice (section 6), but issue #407 lists CV-CUDA (it has `pillowresize`, `copymakeborder`, `resize_crop_convert_reformat`), DALI and nvImageCodec as "not verified in this repo". This is the owner's recorded past cost (a GPU prep path hand-written before the NVIDIA libraries were found) | evaluate CV-CUDA and DALI against the current kernels under the #407 parity gate **before** any new kernel or the batched-kernel work | covered by **#407 / #40 in B5b**; this adds a gate, not a task | B5b (step 0 of #407) | M | medium: bit-exact Laplacian and PIL-matching resize are parity requirements | the #407 parity gate (identical boxes within 1 px, embedding cosine >= 0.999) and a written comparison table in the PR |

## 5. Standards findings (STD-n)

| ID | Finding (evidence verified now) | Fix | Batch | Size | Proof |
|---|---|---|---|---|---|
| STD-1 | Compatibility shim: `src/services/detection/cascade_detect/preprocess.py::_letterbox` is a "thin wrapper over `geometry.letterbox_to_square` ... kept here so call sites in this module don't need to change"; its caller is `cascade_detect/region_detector.py` | call `letterbox_to_square` directly and delete the wrapper (engineering_standards 1.5) | B3 (U-Step 6 rewrites `region_detector.py`; editing it in B1 would be churn) | XS | vulture finds no `_letterbox`; region tests unchanged |
| STD-2 | Three scalar IoU helpers in `src`: `services/detection/geometry.py::iou`, `services/curation/dataset_import/regions.py::iou`, `services/curation/region_eval_geometry.py::iou`. They differ on degenerate boxes (the last does not clamp a negative width or height in the union) | one pure helper in `src/domain/` (domain imports only `config` and `utils`, so every caller may use it); `torchvision.ops.box_iou` is for tensors and would pull torch into pure code, so it is not the answer here | B1-1C | S | parametrized test on normal, touching, disjoint and degenerate boxes against the old three; the three callers' tests pass |
| STD-3 | The B4 `create_task` ratchet (Q-Phase 7) looks like audit Appendix D's cut item "a `spawn(coro, registry)` helper plus a `create_task` ratchet". The cut reason was "it would cement jobs in API workers"; after B2 no job runs in an API worker, so the reason no longer applies | state this in the ratchet's docstring and in [05_b4_stateless_api.md](05_b4_stateless_api.md) so review does not flag it as redoing a cut item; take its baseline only after B2 merges | B4-4B | XS | the ratchet's plant-a-violation proof |
| STD-4 | Counts in plans go stale: the file-lock count is 19 files / 55 lines today, the former master plan said 18 / 51 | every new ratchet's first baseline is generated by running the guard on the tree, with a reason written per line by a person; a count quoted in a plan is never copied into a baseline | B1-1A | XS | the guard's own tests |
| STD-5 | The module-level asyncio guard allows `open_vocab_ingest.py::_PASS_LOCK` | delete the lock and the allowlist line together | covered by **B2-2C** ([03b_b2_w2_job_families.md](03b_b2_w2_job_families.md)) | XS | the guard with an empty allowlist |
| STD-6 | Owner rule 6 (library first) is enforced by nothing but review; `.github/PULL_REQUEST_TEMPLATE.md` has no item for it | add a checklist line: "A new mechanism uses a maintained library or a documented pattern; the doc is cited in the PR; any custom code says why the library does not fit" | B1 (with [09_github_actions.md](09_github_actions.md)) | XS | n/a (template) |
| STD-7 | Stale guard reference: the PR template and CONTRIBUTING's intro name a region-field ratchet (`scripts/codegen/check_no_literal_region_fields.py`, `PORTED_PATHS`) that does not exist (`ls scripts/codegen`; the only remaining mention outside docs is the naming-leak allowlist) | delete both mentions (same commit as ENV-9) | B1-1D | XS | `tests/test_doc_links.py`; `rg check_no_literal_region_fields` finds only history |

## 6. Custom code that stays (checked, with the reason)

| Code | Why it stays | Check against |
|---|---|---|
| `src/utils/retry.py`, the retry in `src/clients/triton_pool_core.py` | GPU-8 fixes the classifier; tenacity for it was cut as churn | audit Appendix D ("A retry library (tenacity) for the Triton retry") |
| OCC conflict loops in `src/clients/occ.py`, `occ_bulk.py`, `clustering/region_box_rows.py`, `assignment_writer.py`, `projects/bootstrap.py` | these re-read and re-apply on a 409; that is the OCC pattern, not a transient-error retry; the helpers were checked correct | audit Appendix B (OpenSearch: "The OCC helpers ... are correct") |
| Segmenter retry in `scripts/curation/worker/client.py`, upload retry in `scripts/curation/ingest_upload.py` | each attempt records timing bookmarks and a circuit-breaker metric; tenacity callbacks could carry them, but the gain is the same "40 lines" the audit cut | audit Appendix D reasoning |
| Download retries in `scripts/datasets/_common.py`, `fetch_hires_set.py` (urllib) | offline fetch tools, standard library only | n/a |
| `_TokenBucket` in `src/services/labeling/vlm_client.py` | 30 lines; `aiolimiter` would add a dependency for the same per-process semantics. The real limitation is that the limit is **per process** (x API workers), not a provider-wide cap: record it with BA-8's per-process caps and revisit when CC-10 moves `label_batch` off the API | BA-8 (documented per-process cap), CC-10 (B4) |
| LRU dicts in `src/clients/pe_encoder.py` (batched partial-hit lookup under one lock), `labeling/vlm_factory.py` (closes the evicted labeler), `projects/guard.py` (isolation-critical handle map) | each needs eviction side effects or batch semantics a plain `LRUCache` does not give; the guard is security-relevant and gets no churn | review depth by risk |
| Small single-value TTL caches (`capacity.py`, `regions_fp.py`, `field_coverage.py`, `resource_links.py`, `settings_doc.py`) | one value with a timestamp; BA-3's verdict leaves capacity, `regions_fp` and `field_coverage` alone, and fixes `settings_doc` | audit BA-3 fix step 3; Appendix D ("Capacity cache change") |
| `src/utils/scrfd_decode.py::_nms` | matches the SCRFD reference implementation | audit Appendix B (GPU: "SCRFD NMS and Umeyama match their references") |
| `scripts/curation/bakeoff/backends/yolo_post.py` (`letterbox`, greedy `nms` with the class-offset trick) | documented choice: "Kept deliberately small and dependency-light (numpy + cv2)"; bake-off results back published numbers, so a swap to `torchvision.ops.batched_nms` would need a parity run for no correctness gain | module docstring |
| Greedy NMS in `src/services/detection/region_candidates.py` | crop-frame IoU with a per-profile cap, not a plain NMS; the separate raw-head NMS in `region_detector.py` is deleted in B3 | U-Step 6 |
| `src/services/curation/dataset_import/scan.py::exif_transposed_size` (reads EXIF tag 0x0112 to swap width and height) | reads the header only; `ImageOps.exif_transpose` would decode every pixel just to learn a size | Pillow docs (exif_transpose returns a transposed image) |
| nvJPEG ctypes lanes and CuPy kernels in `models/gpu_image_prep/1/` | measured: nvImageCodec's GPU-only backend decodes nothing on a card without a hardware JPEG engine and its hybrid backend runs Huffman on the CPU; DALI and nvImageCodec ignore EXIF; see `docs/design/gpu_decode_plan.md` and the module docstrings. LIB-8 adds the evaluation gate before any new kernel | `gpu_decode_plan.md`; #407 |
| Periodic `while True` + `asyncio.sleep` loops in the lifespan and daemons | replaced by Celery beat | Q-Phase 4 (B2-W3) |
| Importers' `isinstance(..., dict)` checks (`dataset_import/coco.py`, `labelstudio.py`, `bulk_ingest/s3_events.py`) | they parse open external formats and must name the bad record; not reviewed in depth here. A pydantic model per format is a reasonable later refactor, not a library being re-implemented | backlog only if an importer is rewritten (#370) |
| Bash `case` argument parsing in `setup-openprocessor.sh::parse_args` and the CLI | `getopts` has no long options; the `case` loop is the idiomatic form; shellcheck runs on it | n/a |

## 7. Batch summary of this file's work

| Batch | IDs |
|---|---|
| B1-1A | LIB-1, STD-4, the file-lock ratchet, the atomic-write scan |
| B1-1C | LIB-4, LIB-7 (inside OS-7), STD-2 |
| B1-1D | LIB-2, STD-7 |
| B1-1E | LIB-5 (inside U-Step 3) |
| B1 (09) | STD-6 |
| B2 | STD-5 (inside 2C); the Celery guards |
| B3 | STD-1 (inside U-Step 6) |
| B4 | LIB-3 (inside 4C), STD-3, the `create_task` ratchet and the module-level state guard |
| B5b | LIB-8 (gate inside #407) |
| backlog | LIB-6 |
