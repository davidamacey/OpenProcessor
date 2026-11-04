# B1 track 1C: inference and search hygiene

| Field | Value |
|---|---|
| Status | plan, not started |
| Batch / release | B1, part of the single v0.7.0 release |
| Depends on | B0 merged; `int/b1-correctness` cut. Independent of 1A, 1B, 1D, 1E |
| Decisions applied | D-13 (order); owner rules: docs and maintained libraries first, no asyncio fan-out in request handlers (TaskGroup only in worker and clustering code), fail closed |
| Source IDs | `codebase_audit_plan.md` 4.2 (OS-5, OS-6, OS-7, OS-9, OS-10), 4.3 (GPU-4, GPU-5, GPU-8, GPU-9, GPU-10, GPU-16, GPU-17 with API-12), 4.7 (CC-6, CC-8); master plan 5.1 open choice (shared OCR detector-size helper, decided here), 7.1 1C row; issue N5 (`09_github_actions.md`) |
| Size | M |
| Lane | branch `b1/inference`, worktree `<repo>-wt-b1-inference`, compose project `op-b1-inference` |

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read only, with `rg`).
Consolidated 2026-11-03.

Not in this track (do not do them here): GPU-14's interim clip (cut; B3 deletes the heuristic,
`04_b3_ultralytics_one_version.md`), GPU-18 docs (B3), GPU-3b, GPU-7, GPU-12 (B5,
`07_*`), GPU-6 (owner decision D-35, B5), OS-3 (track 1A), OS-8 and GPU-19 (evidence first,
`11_risks_cleanup_host.md` backlog).

## 1. Goal

Inference and search stop doing avoidable damage: no multi-GB allocation from a sliver crop or a
huge-dimension JPEG, no retries of deterministic errors, no wait-forever sync Triton calls, no
stale "model ready" after an unload, filtered kNN returns `k` hits, list routes refuse an
unbounded page with 422 instead of a 503, and clustering stops leaking PIT and scroll work on an
error. Every item is small and independent; one commit each.

## 2. Preconditions

- Lane worktree: `git worktree add <repo>-wt-b1-inference -b b1/inference int/b1-correctness`.
- The venv in the worktree. No Docker, no GPU in the lane. OS-6's live hit count runs in the batch
  sweep.

## 3. Tasks

Each task: red test seen failing on the base (record the output for the PR), paired control
passing before and after, one commit. Citations are the docs to follow; cite them in a comment only
where the code is not obvious.

### 1C-0 Decide the shared OCR detector-size helper (evidence first, S)

- Open choice from B0: two sizing rules exist.
  - `/ocr`: `src/clients/triton_ocr.py` (the B0 GPU-1 fix): long side scaled into [640, 960], each
    side rounded to a multiple of 32 with a floor of 32, no padding.
  - Cascade: `src/services/detection/cascade_detect/ocr_recognizer.py::ocr_det_input_size`: long
    side 640, short side raised to at least 320, each side clamped to [320, 960].
  - Sharing one helper would change `/ocr` output, so it needs a defect, not taste.
- Evidence step: read the det engine's TensorRT profile as the installer builds it
  (`scripts/export_paddleocr.sh` sets `DET_MIN_SHAPES="x:1x3x32x32"` and max `4x3x960x960`; check
  `scripts/lib/paddleocr_export.sh`, `scripts/lib/trt_settings.sh` and
  `models/paddleocr_det_trt/config.pbtxt` for the profile the default install uses). The cascade
  comment claims "builds reject sides below 320".
- Decision rule:
  - If any default-install profile rejects a side below 320: `/ocr` can send a thin image the
    engine refuses (a live bug). Move `ocr_det_input_size` to `src/domain/ocr_geometry.py` (domain
    may import only `config` and `utils`) and use it in both paths. Red: a 2000x40 image through
    `/ocr`'s sizing gives a side below the engine minimum today. Control: a 1280x960 image gives the
    same size before and after. CHANGELOG: behaviour change for `/ocr` on thin images.
  - Otherwise: no code change. Record the finding in the PR description and in a one-line comment
    above each rule ("`/ocr` follows PaddleOCR `DetResizeForTest`; the cascade sizes crops for the
    region engine"), and close the open choice.

### 1C-1 GPU-4: `gpu_decode_ready` caches a transient failure forever (S)

- File: `src/services/curation/ingest_gpu_decode.py` (`rg -n "_READY\[key\]"`).
- Bug: `_READY[key] = ready` runs after the `try` too, so one failed Triton call during a restart
  pins that worker to the CPU path until it recycles.
- Fix: write `_READY[key]` only when a verdict is reached inside the `try`; on an exception return
  `False` without caching. Nothing else (no TTL re-validation: audit Appendix D).
- Red: the first call raises, the second succeeds; the second must return True.
- Control: a success stays cached (one probe call across two lookups).

### 1C-2 GPU-5: full bitmap allocated before the dimension check (S)

- File: `src/utils/image_processing.py` (`decode_image`, `validate_image`, `MAX_IMAGE_DIMENSION =
  16384`).
- Doc: Pillow `Image.open` is lazy; `.size` reads the header only
  (https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open).
- Fix: read the header size first (`Image.open(BytesIO(b)).size`, or the existing `megapixels()`
  helper if it reads the header) and reject anything over `MAX_IMAGE_DIMENSION` before
  `cv2.imdecode`. Keep `validate_image` as the post-check. Not `OPENCV_IO_MAX_IMAGE_PIXELS`
  (Appendix D).
- Red: a 20000x20000 JPEG raises before decode; monkeypatch `cv2.imdecode` and assert it is not
  called.
- Control: a 4000x3000 JPEG decodes.
- Reachable by: `/detect`, `/ocr`, the model test route.

### 1C-3 GPU-8: one retry predicate on `status()` (S)

- Files: `src/utils/retry.py` (`is_retryable_error`), `src/clients/triton_pool_core.py`
  (`AsyncTritonPool.infer` retry loop).
- Bug: `InferenceServerException` has `status()`, not `code()`, so the gRPC-code branch never runs
  (RESOURCE_EXHAUSTED and ABORTED are never retried), and the pool retries every exception,
  INVALID_ARGUMENT included.
- Doc: tritonclient `InferenceServerException.status()`
  (https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/utils/__init__.py).
- Fix: one predicate on `exc.status()` (retry UNAVAILABLE, RESOURCE_EXHAUSTED, ABORTED; never
  INVALID_ARGUMENT or DEADLINE_EXCEEDED), used by both `retry.py` and the pool. Delete the `.code()`
  branch and the string patterns. No retry library (Appendix D).
- Red: the status table gives True for UNAVAILABLE, RESOURCE_EXHAUSTED and ABORTED; the pool makes
  1 call on INVALID_ARGUMENT (3 today).
- Control: the pool makes 3 calls on UNAVAILABLE.

### 1C-4 GPU-9: a sliver crop makes the CLIP resize allocate up to about 3 GB (S)

- File: `src/utils/cpu_preprocess.py::center_crop_cpu`.
- Reachable by: `POST /embed/boxes` (up to 100 caller-chosen boxes) and
  `analyze_pipeline.crop_batch_tensor`.
- Fix: crop the long side first to the window that survives the centre crop, then resize only that
  window. Exact for a centre crop; no valid output changes.
- Red: `tracemalloc` peak for a (3000, 1) crop is under 20 MB (the audit measured 623 MB RSS).
- Control: 256x256 and 480x640 crops give byte-identical output to the current function.
- Done when: no resize intermediate exceeds about 4x the output.

### 1C-5 GPU-10: PIL CLIP preprocessors truncate and black-pad one row (S)

- Files: `src/clients/triton_preprocess.py::preprocess_for_mobileclip` and
  `src/services/embedding.py` (`preprocess_image`), both `scale = 256 / min(width, height)` with
  `int()` truncation.
- Fix: route both through `center_crop_cpu` on the array and delete the two copies. This changes the
  resampler on `/embed/image` and `TritonClient.encode_image`: record it in CHANGELOG (Changed).
- Red: a 300x187 image gives no black row.
- Control: a 480 short side gives the same output.
- Done when: one implementation remains (`rg -n "256 / min" src` finds nothing).

### 1C-6 GPU-16: sync Triton calls have no `client_timeout` (S)

- Files: `src/clients/triton_client.py::_infer_with_retry`, `src/clients/fast_face_client.py`,
  `src/clients/pe_text_backends.py` (`TritonTextBackend.encode`), `src/services/embedding.py` (two
  `client.infer(` calls), a constant in `src/config/` (for example `TRITON_SYNC_CLIENT_TIMEOUT_S =
  30.0`; add an `OP_*` variable only if an operator needs to tune it, which then needs
  `env.template` and docs).
- Doc: tritonclient `infer(..., client_timeout=None)` waits forever
  (https://github.com/triton-inference-server/client, `grpc/_client.py` docstring). The async pool
  already passes `client_timeout` (`triton_pool_core.py`).
- Fix: one wrapper used by these call sites passes the timeout; map the deadline error to 504 in
  the existing Triton error handling (`src/core/error_handlers.py`); never retry DEADLINE_EXCEEDED
  (1C-3's predicate).
- Red: a stub client that never answers raises within the timeout and the route answers 504.
- Control: a fast stub succeeds.
- Done when: every sync `infer(` in `src/clients` and `src/services/embedding.py` passes a timeout
  (`rg -n "\.infer\(" src` shows none without one, pool calls excepted).

### 1C-7 GPU-17 (with API-12): `model_residency._ready` never learns of an unload (S)

- File: `src/clients/model_residency.py` (`_ready: set[str]`, `_sync_locks: dict[str,
  threading.Lock]`, `_async_locks: dict[str, asyncio.Lock]`).
- Reachable by: `OP_LOAD_PROFILE=ingest` plus a manual unload or a Triton restart.
- Fix: delete `_ready` and both lock dicts. Call `is_model_ready` (cheap) before each ensure and load
  when it is not ready. Triton serialises loads and the code already tolerates "load lost the
  race", so the load is idempotent and needs no lock (Triton model control,
  https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_management.md).
- Red: ready, then unloaded, then a call must issue a load.
- Control: a ready model is not reloaded.
- Done when: no module-level readiness set or lock dict remains in `model_residency.py`
  (`tests/test_no_module_level_async_locks.py` does not see the dict; this task removes it).

### 1C-8 OS-5: unbounded `page` on two list routes (S)

- Files: `src/routers/curation/audit.py` (queue handler), the `classes_crops` listing
  (`src/routers/curation/classes_crops.py`).
- Bug: `from = (page - 1) * page_size` passes `index.max_result_window`; OpenSearch answers 400 and
  the route maps it to 503. `_common.guard_page_depth` exists (used by `review.py`, `regions.py`,
  `crops_list.py`) but not here.
- Doc: OpenSearch `index.max_result_window` (https://docs.opensearch.org/latest/install-and-configure/configuring-opensearch/index-settings/).
- Fix: call `guard_page_depth(page, page_size)` in both handlers. The search routes are fine as
  designed.
- Red: `page=100000` gives 422 (503 today).
- Control: `page=1` gives 200.

### 1C-9 OS-6: legacy visual-search kNN filter is a post-filter (S)

- File: `src/clients/opensearch/search.py` (the body builder that wraps `knn` in
  `bool{must:[knn], filter:[...]}` when `class_filter` or `cluster_ids` is set).
- Reachable by: `/search/objects?class_filter=` and the cluster-routed search.
- Doc: OpenSearch "Filtering in k-NN search", efficient filtering puts the filter inside the `knn`
  clause (https://docs.opensearch.org/latest/vector-search/filter-search-knn/). The indexes use
  faiss HNSW on OpenSearch 3.6.0, which supports it (audit verdict).
- Fix: move the filters into `knn[field]['filter']`; keep `min_score`; correct the docstring's
  `cluster_ids` speed-up claim.
- Red: a unit test on the built body asserts `'filter' in body['query']['knn'][field]` and no
  `bool.must[knn]`.
- Control: a query with no filter builds the same body as today.
- Live (batch sweep): a restrictive class filter returns `k` hits on the live harness; record the
  hit count before and after in the sweep report.

### 1C-10 OS-7: blocking `update_by_query` in `backfill_embedded_state` (S)

- Files: `src/domain/embedding_state.py` (`update_by_query` with `wait_for_completion=True`),
  `src/services/curation/clustering/id_normalize.py::run_update_by_query_polled` and its callers
  (`id_normalize.py`, `clustering/residual_gate.py`).
- Doc: OpenSearch update-by-query `wait_for_completion=false` returns a task; poll the tasks API
  (https://docs.opensearch.org/latest/api-reference/document-apis/update-by-query/).
- Fix: move `run_update_by_query_polled` to `src/clients/opensearch/tasks.py`; update its two
  callers; use it for the backfill. `src/domain/` may import only `config` and `utils`
  (`scripts/codegen/check_layering.py`), so the domain module keeps only the pure rule (the query
  and the script body) and the OpenSearch call moves to its caller (re-find with
  `rg -n "backfill_embedded_state" src`; the audit names `legacy_repair`), which calls the polled
  helper. Nothing else (Appendix D).
- Red: a fake client records `wait_for_completion=False` and the task poll.
- Control: the existing `id_normalize` tests pass.
- Done when: no blocking `update_by_query` remains on the items index.

### 1C-11 OS-9: scroll context not cleared on the error path (S)

- Files: `src/routers/curation/classes_merge.py::_scroll_merge_ids`, `src/clients/opensearch/clusters.py`
  (two `clear_scroll` calls after clean loops).
- Doc: OpenSearch scroll API, "close the search context when you're done"
  (https://docs.opensearch.org/latest/api-reference/search-apis/scroll/).
- Fix: wrap each loop in `try/finally: clear_scroll(...)`. Not PIT (Appendix D).
- Red: a fake scroll that raises on page 2 still sees `clear_scroll` called.
- Control: a clean run calls it once.

### 1C-12 OS-10: unbounded `offset` in the cluster list aggregation (S)

- File: `src/routers/curation/clusters.py` (the handler with `max_clusters: int = Query(200, ge=1,
  le=1000)` and `offset: int = Query(0, ge=0, ...)`, terms `size: offset + max_clusters`).
- Doc: OpenSearch `search.max_buckets` (default 65,535)
  (https://docs.opensearch.org/latest/aggregations/bucket/index/).
- Fix: `le=` on `offset`, sized as `search.max_buckets` minus the `max_clusters` maximum (1000), as a
  named constant with that comment. `make contracts` in the same commit.
- Red: an offset above the bound gives 422.
- Control: `offset=0` gives 200.

### 1C-13 CC-6: PIT slice producers orphaned on error (S)

- Files: `src/services/curation/clustering/pit_pages.py` (`asyncio.gather` over the slice producers
  in `run()`), `src/services/curation/clustering/residual_fetch.py` (`gather` of slice results).
- Reachable by: an OpenSearch error mid-cluster. This is worker and clustering code, not a request
  handler, so a TaskGroup is allowed.
- Doc: Python asyncio Task Groups: on a child failure the remaining tasks are cancelled; errors are
  raised as an `ExceptionGroup` (https://docs.python.org/3/library/asyncio-task.html#task-groups).
- Fix: `async with asyncio.TaskGroup() as tg:` inside `run()`, emit the sentinel after the group
  exits; the same in `residual_fetch`, deleting the PIT after the group.
- Traps: callers that catch a specific exception will stop matching an `ExceptionGroup`. Check every
  caller first (`rg -n "iter_pages|residual_fetch" src`: `cluster_geometry.py`, `item_scores/job.py`,
  `embedding_viz.py`, `retrain_policy.py`, `refine.py`, and the orchestrator). Use `except*` there,
  or re-raise the single child exception at the module boundary.
- Red: a fake client with slice 0 raising leaves 0 pending tasks after the call and issues no search
  after the failure (the audit saw 3 leaked tasks and 10 searches).
- Control: a clean run yields every page once.

### 1C-14 CC-8: detection worker stage tasks have no crash callback (S)

- Files: `scripts/curation/worker/runner.py` (about 12 `asyncio.create_task` stage tasks),
  `scripts/curation/vlm_worker.py::_crash_on_unhandled_exception` (reuse it).
- Fix: `add_done_callback(_crash_on_unhandled_exception)` on each stage task; log the drain gather's
  results. Move the helper to a shared module under `scripts/curation/` only if the layering ratchet
  requires it.
- Red: a stage coroutine that raises outside its per-item try makes the worker exit non-zero.
- Control: a clean drain exits 0.

### 1C-15 Legacy visual search: Triton off the event loop, no box vector against the whole-image index (S)

Run after 1C-9 in the same lane (both touch legacy visual search). Origin: the W4 bug of
`cluster_search_makemodel_plan.md` section 10, planned and never filed.

- Evidence: `src/services/visual_search/search.py` (`SearchMixin`): `search_by_object` builds a sync
  client (`get_sync_triton_client`) and calls `infer_yolo_clip_cpu(image_bytes)` inside `async def`;
  `search_by_image` and `search_by_text` call `self.inference.encode_image` and `encode_text`,
  which are plain `def` in `src/services/inference.py`. The routers in `src/routers/search/` await
  them, so each call blocks the worker's event loop for a full detect-and-embed.
  `src/routers/search/objects.py` also reads the upload with the sync `image.file.read()`. For a
  detected class that is neither a vehicle nor a person, `search_by_object` queries
  `visual_search_global` (whole-image vectors) with a box vector: two different vector spaces.
  `SearchMixin.search_vehicles` and `search_people` have no router caller (check with `rg`; delete
  them if the unused-code ratchet agrees).
- Library or pattern: one offload per request for the blocking inference, with the same helper
  1D-7 (CC-5) uses: `asyncio.to_thread` (Python docs, "Running in Threads", asyncio-task), or the
  repo's CPU executor helper if 1D-7 chose it; `await image.read()` (Starlette `UploadFile` is
  async: starlette.io/requests, "UploadFile"). One offload per request is not a fan-out (PQ-9).
- Fix: as above for the blocking calls. For an unindexed category, fail closed: answer 422 with the
  new error code `object_class_not_indexed` (recommendation) instead of a cross-space query. The
  alternative to settle at kick-off: keep the query and document it as "find images like this crop".
  Contracts (`make contracts`) and the README route table in the same commit.
- Red: (a) a stub sync Triton client that sleeps 0.3 s; two concurrent `/search/object` requests
  through `httpx.AsyncClient` with `ASGITransport` on one loop take at least 0.6 s today and under
  0.45 s after; (b) a detection of a class outside both categories never calls `search_global`
  (fails today). Control: a vehicle query returns the same hits before and after.
- Done when: both red tests pass; no sync Triton call or sync file read remains inside an
  `async def` of `src/services/visual_search/search.py` and `src/routers/search/objects.py`; contracts
  regenerated; CHANGELOG Fixed (event loop) and Changed (422 `object_class_not_indexed`).
- Size: S.

## 4. Gate for this lane

- Targeted red/green per task; `.venv/bin/pre-commit run --files <touched files>`.
- `make contracts` in 1C-12 and 1C-15.
- T0 only ([00_index.md](00_index.md) section 8.1): no full suite, no `--all-files`, no Docker.

## 5. Done when

- Every task's done-when above holds.
- `rg -n "256 / min" src` and `rg -n "\.code\(\)" src/utils/retry.py` find nothing; no module-level
  readiness set or lock dict in `model_residency.py`.

## 6. Docs, contracts, CHANGELOG

- Contracts: 1C-12 (`offset` bound), 1C-15 (the 422 code, if taken).
- Docs: `docs/CURATION.md` or the API reference section that lists the cluster list parameters (the
  `offset` bound); the search docs if they describe class-filter behaviour (`rg -n "class_filter"
  docs README.md`).
- CHANGELOG `[Unreleased]`: Fixed entries for GPU-4, GPU-5, GPU-8, GPU-9, GPU-16 (504 on a hung
  Triton), GPU-17, OS-5 (422), OS-6 (filtered search returns `k` hits), OS-7, OS-9, OS-10 (422),
  CC-6, CC-8, 1C-15 (`/search/object` no longer blocks the loop); Changed for 1C-15 (422
  `object_class_not_indexed`), GPU-10 (resampler on `/embed/image`) and 1C-0 if `/ocr` sizing changes.
  Breaking: none (422 instead of 503 on out-of-range paging is a fix).

## 6.1 What to keep after merge

| Fact or section | Target doc |
|---|---|
| the 422 `object_class_not_indexed` semantics, the `offset` bound, the 422 for out-of-range paging | `docs/design/curation_api_contract.md`, README route table |
| filtered-kNN rule (filter inside the `knn` clause) and the OS-6 hit counts | `docs/opensearch_schema_design.md`, `docs/PERFORMANCE.md` |
| the 1C-0 decision and its evidence (OCR detector sizing) | `docs/ARCHITECTURE.md` (OCR) |
| GPU-10 resampler change | `CHANGELOG.md` |

Done when also: the closing B1 PR deletes this file after the keep-list is moved
([00_index.md](00_index.md) "Plan lifecycle").

## 7. Upgrade impact

None. No mapping, data or engine change. `/embed/image` vectors can differ slightly (GPU-10
resampler): note it in the CHANGELOG; stored vectors are not recomputed.

## 8. Rollback

Each task is one commit; the track reverts alone (`git revert -m 1 <1C merge>`), no coupling.

## 9. Risks

| Risk | Mitigation |
|---|---|
| CC-6's `ExceptionGroup` breaks a caller that catches one exception type | check every caller first; `except*` or re-raise at the boundary; the control test runs the callers' tests |
| GPU-10 changes embedding bytes for `/embed/image` | CHANGELOG; byte-identical control on the CPU path for the sizes that were already exact |
| GPU-16 timeout too short for a legitimate heavy sync call | 30 s is above every measured sync call; the constant is in one place |

## 10. Corrections found while verifying

- `src/core/error_handlers.py` exists but the BA-5 `error_type` hits are only in `src/app/factory.py`
  (relevant to 1D).
- `classes_merge._scroll_merge_ids` already clears the scroll after a clean loop; the error path is
  the bug (as the audit says).

## 11. Lane agent brief (paste as is)

```
You are the implementation agent for B1 track 1C (inference and search hygiene) of OpenProcessor.
Plan: docs/design/program/02c_b1_inference_search.md (this file). Execute it standalone.
Read first: CLAUDE.md, docs/VISION_AND_GOALS.md, docs/design/engineering_standards.md, and
docs/design/codebase_audit_plan.md sections 4.2, 4.3, 4.7 for each task's detail.

Worktree: <repo>-wt-b1-inference, branch b1/inference (created by the coordinator from
int/b1-correctness). Work ONLY there. Never push, never merge, never check out another branch.
Compose project for any Docker command: -p op-b1-inference (you should need none). GPU: none.
Deadline: <coordinator fills in>.

Rules:
- Tasks 1C-0 .. 1C-15 in order (1C-15 after 1C-9, with the same offload helper as 1D-7), one commit each, conventional commits, hooks on.
- Red test first, run on unchanged code, SEE IT FAIL, paste the red output into the commit body;
  then fix; paired control green.
- 1C-0 is evidence first: change code only if the engine profile proves a defect; otherwise record
  the finding and change nothing.
- Targeted tests only and .venv/bin/pre-commit run --files <touched files>. No full suite, no
  --all-files.
- TaskGroup only in clustering/worker code (1C-13); never asyncio fan-out in a request handler.
  No retry library, no new lock, no cache of an error result.
- make contracts in the same commit as 1C-12 and 1C-15.
- Never use pgrep -f, pkill -f, or sleep/until polling loops.

Report at the end: commits (sha + subject), red outputs in one line each, the 1C-0 decision with
its evidence, anything not done and why, and any plan correction.
```
