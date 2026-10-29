# Phase A1 audit: PR #361 (release/0.7.0) and the #357 and #362 fix branches

Independent audit, 2026-10-28. Scope:

- main range `fd64d086..96886e0a` (merge of PR #361);
- `fix/v070-357-work` at `a9c75abf`;
- `fix/v070-362-work` at `04dde264` and `0d350fac`.

The audit was read-only on source. It used offline pytest only, with no Docker and no GPU. Mutations ran in throw-away copies under `/tmp` and were reverted after each run.

## Summary

| Severity | Count | Issues filed |
|---|---|---|
| Critical | 0 | none |
| High | 1 | #373 |
| Medium | 8 | none (not required) |
| Low | 10 | none |

- **Test runs.** The offline suite on `96886e0a` passes: 9,132 passed, 20 skipped (`pytest tests/ -m 'not live'`). The targeted tests of both branches pass: 36 for #357 and 69 for #362. `ruff check` and `ruff format --check` are clean on the files the branches change.
- **Mutation testing.** 33 mutations were applied: 25 on main, 5 on the #357 branch, 3 on the #362 branch. 32 were caught. The survivor is M25: no test notices when the `forget_bootstrapped` call on project delete is removed (finding L4).
- **Mergeability.** Both branches can merge after small doc fixes (details at the end). Keep #357 open: the fix does not cover the case the issue observed. Keep #362 open until the live re-measurement is done.

## 1. Acceptance, issue by issue

| Issue | Criterion (re-derived from the issue body and comments) | Verdict | Evidence |
|---|---|---|---|
| #259 | Embed pass at least 20 img/s **on set A** (the pinned COCO set) | **NOT MET** | Measurement log E1 and F3: COCO 16.2 to 16.7 img/s after the change (15.7 before). The issue was closed against set P (21 MP photos, 51 img/s, log G3 and G6). No comment restates the bar to set P; only the GPU-busy bar was flagged for restatement. |
| #259 | API CPU per image no worse than 0.047 CPU-s | NOT MET on set A, MET on set P | E1: COCO 0.0649 to 0.0660, which is 25 percent above 0.6.0's 0.052 (`offload_always`, `src/utils/cpu_pool.py:111`). Set P: 0.025 (G3, raw `exp1.jsonl` agrees). See M1. |
| #259 | Encoder busy above 80 percent | MET (set P) | G3: 84.5, 88.7 and 92.1 percent, from 500 ms samples (raw `g500_*.log`). |
| #259 | Re-embedded vectors bit-identical to the pre-change pass (median cosine 1.0) | MET within float noise | G4: median 0.9999965 against the old code. That is not bit-identical; the difference is batch-composition noise. |
| #259 | Vector-mismatch decision recorded; if ingest changes, P-pe crop cosine p01 at least 0.995 against the 0.6.0 reference | PARTIAL | The decision is recorded (`docs/PERFORMANCE.md`). `docs/PERFORMANCE.md:3174` itself says "the P-pe parity capture against the 0.6.0 reference ... is still to run". E1 compares ingest with the embed pass, not with the 0.6.0 reference. |
| #358 | Deterministic test: the same content in two concurrent batches stores one image | MET | `tests/curation/test_ingest_concurrent_duplicates.py` (4 tests). Mutations M1 to M4, M19 and M24 were all caught. |
| #358 | Write-time guard (content `_id` plus `create`) | MET | `ingest_writer.py:268,287`, `ingest_index.py:content_image_id`. |
| #358 | Re-run at concurrency 8, 16 and 32, 3 times; counts equal distinct contents | MET (counts not verifiable) | Log E2: 9 runs at 3,000 ingested; img/s matches raw `exp2.jsonl`. The distinct-imohash and orphan counts have no raw file (the projects were deleted). |
| #359 | After a project delete the cache holds none of that project's files, and other projects are untouched | MET | `source_image_cache.py:crop_cache_file`, `delete_paths.py:84`, `tests/curation/test_crop_cache_project_scope.py`. Mutations M5, M6 and M22 caught. |
| #359 | Cache cap and busy-delete documented | MET | `env.template` (crop-cache block), `docs/design/curation_api_contract.md` (`project_busy`). |
| #360 | Region vectors do not depend on the route; parity test; migration documented | MET, with a doc gap | G1: min 0.9835, p01 0.9970, median 0.99927 (raw `reg_*.npz` agrees). Tests: `test_region_vector_no_roundtrip.py` (M8 and M9 caught). The migration doc is inconsistent with the CHANGELOG (M3). |
| #356 | p99 text-search latency below 300 ms at 100k items, recorded in `docs/PERFORMANCE.md` | PARTIAL (issue still open) | The fixes are in: startup bootstrap, route warm-up, text-backend prime. Tests in `test_startup_bootstrap_warm.py` and `test_app_lifespan.py` (M12, M13 and M20 caught). The 100k live p99 re-check is still pending. The "380 to 470 ms cold" figure has no raw data (section 5). |
| #350 | An idle project queues no jobs; growth past the threshold still triggers once | MET | `refresh_schedule.py:131`; `test_cluster_refresh_schedule.py` (M18 caught). |
| #268 (decode instances only) | A shipped default changes only if it beats the spread and passes parity | MET for `full` | E4: 31.8 to 48.0 img/s (raw `exp4.jsonl` agrees). The installer's choice of `full` on 24 GB cards breaks the footprint assumption (H1). The remaining stage-2 items moved to v0.8.0. |
| #341 | One command scores any partition; reproducible from a seed; false merges reported apart from false splits | PARTIAL | The harness and tests are in (`scripts/bench/cluster_eval*.py`; M23 caught). The recorded baseline does not reproduce from the retained data (section 5, M8). |
| #337 | IO pressure sampled in the scale driver | MET for that item (issue open) | `scripts/bench/scale_run.py`, `host_pressure.py`, `tests/bench/test_host_pressure.py`. The other criteria stay open by design. |
| #357 (branch) | A pass over a pool another writer touches cannot report success; a 250k reset completes or fails fast | PARTIAL | See section 8. |
| #362 (branch) | Stage cost at 21 MP no worse than 1.2x v0.6.0; two items of one photo decode it once; activation rejects an unusable detector | PARTIAL | Decode-once and the activation check are tested (B6 to B8 caught). The 1.2x cost bar is not re-measured. |

## 2. Findings

### High

**H1. The installer picks the `full` profile (two `gpu_image_prep` instances) for a 24 GB Triton card.** Filed as #373.

- Where: `setup-openprocessor.sh:821` and `:851` (`free[tri] >= 24` gives `profile=full`); `config_templates/profiles/full.json` (`gpu_image_prep_instances: 2`, `ingest_gpu_decode: true`); `_triton_need_gb` at `setup-openprocessor.sh:719`, which leaves the decode out.
- Scenario: on a host with 2 x 24 GB cards, `recommend_plan` prints `GPU_PROFILE=full`, `OP_INGEST_GPU_DECODE=1` and `TRITON_GPU_ID=0` (also `API_GPU_ID=0`), with no warning. That needs about 17 GB of models plus 2 x 8.5 GB of decode, roughly 34 GB on a 24 GB card. Reproduced offline with the command in #373.
- Why the tests miss it: `tests/test_gpu_image_prep_instances.py` assumes a 48 GB floor for `full`, while the installer's real threshold is 24 GB. Before #268 the overflow was about 1.5 GB; now it is about 10 GB.
- Fix: choose the profile, or at least the instance count and decode flag, from free VRAM including count x 8,704 MiB. Add the decode to the placement arithmetic. Test the planner output, not the profile file alone.

### Medium

**M1. The #259 closure does not meet the issue's own criteria on set A, and the change costs 25 percent more API CPU per image on small photos.**

- Where: `src/services/curation/reprocess_embed.py:345` (`with offload_always()`) and `cpu_pool.py:111-136`.
- Evidence: log E1 and F3, COCO: 0.052 CPU-s per image before, 0.065 after, at 16.2 to 16.6 img/s against a bar of 20.
- Scenario: a COCO-sized (below 3 MP) embed pass now burns more API CPU for 6 percent more throughput. The issue still says "set A".
- Fix: reopen #259 or record an owner restatement to set P. Either drop `offload_always` for photos below the GPU gate, or re-measure COCO with the chunk window (G5 notes this was not done).

**M2. The worker region stage can stop the detection worker's writer on a source image that raises anything other than `OSError`.**

- Where: `scripts/curation/worker/region_embed_stage.py:56-66`, called at `:78` outside any `try`; caller `scripts/curation/worker/flow.py:209`, also outside the `try` that guards `_bulk_update`. `decode_source_rgb` (`src/services/curation/crop_bytes.py:63-83`) catches only `UnidentifiedImageError` and `OSError`.
- Scenario: a large photo is ingested on the GPU route, so Pillow never opens it at ingest. Its malformed EXIF makes `ImageOps.exif_transpose` raise a non-`OSError` (Pillow has raised `TypeError`, `ValueError` or `SyntaxError` on such data). `_flush` has already cleared `pending`, so `out_q.task_done()` is never called for that flush and the writer coroutine dies. A profile swap's `out_q.join()` then waits forever, or the restarted worker re-fetches the same item and dies again (poison pill).
- Before #360 the stage decoded only the system-made crop JPEG. The #362 branch keeps the same exposure: `asyncio.gather` without `return_exceptions`.
- Fix: wrap the per-photo decode and crop in a broad handler that logs and leaves that photo's boxes without a vector, as the docstring promises ("the flush ... must never fail because of it"). Add a test with a decoder stub that raises `ValueError`.

**M3. The upgrade instructions for 0.7.0 are incomplete or wrong in three places.**

- (a) `CHANGELOG.md:16` says to re-embed parts `crop` and `region`. `docs-site/docs/operations/upgrade-and-backup.mdx:51,55` sends `parts: ["region"]` only and says crop vectors are not touched. A user who follows the docs page keeps the zero-padded edge-crop vectors of #259.
- (b) `CHANGELOG.md:19` tells existing installs to regenerate the model config with `./openprocessor profile`. That command refuses outside a checkout (`openprocessor:393-395`, `require_checkout`).
- (c) "Copy the new model files into the Triton model repository" has no command for an installed directory. The installer's seed step copies with `cp -rn` (`scripts/lib/model_setup.sh:490-491`), so an upgrade keeps the old `model.py` and `config.pbtxt`. The GPU embed route then stays off, logging only `reprocess_gpu_embed_model_outdated`, and the main 0.7.0 embed gain never arrives on upgraded installs.
- Fix: give one command sequence for a checkout and one for an installed directory, and make the docs page re-embed `crop` and `region`.

**M4. A 409 on the images-doc `create` is always taken as "another copy won", including when the 409 answers this writer's own earlier attempt.**

- Where: `src/services/curation/ingest_writer.py:268,287` and `:299` (items of a duplicate unit are dropped).
- Scenario: the opensearch-py transport retries a request on connection errors and on 502, 503 and 504 by default (`max_retries=3`, `retry_on_status=(502, 503, 504)`; the clients set no override). If the first bulk was applied but its response was lost, the retry gets 409 for every image of the flush. Each is reported `duplicate`, none of its items are written, and every later ingest of that content also says `duplicate`, leaving an images doc with no items.
- The old blind `index` was idempotent under such a retry; this is new with #358. Found by inspection; not reproduced.
- Fix: on 409, read the stored doc (a realtime GET) and treat it as this unit's own when `ingest_run_id`, `image_path` and `imohash` match, then write the items. Alternatively, set `max_retries=0` for the writer's bulk and handle retries in the writer.

**M5. Startup bootstrap cost grows with workers x projects, and blocks every worker's startup.**

- Where: `src/core/dependencies.py:144-151`, awaited in the lifespan (`src/app/lifespan.py:87`).
- Scenario: each of the 32 uvicorn workers (`docker-compose.yml:263`) runs, for every active project in turn, about 5 exists checks, 22 unconditional `PUT _mapping` calls (for example `ensure_overlay_fields.py:341`) and a class count. That is 32 x P x about 28 requests at once on every API start. Each recycled worker (`--limit-max-requests=10000`) is not ready for about P x 0.4 s. With 20 projects that is about 18,000 cluster-state updates at boot and an 8 s startup per worker, against OpenSearch that may also be serving ingest.
- Fix: skip a `put_mapping` whose fields already exist (one `get_mapping` per index), or run the migrations once per deployment and share the result. Bootstrap only `default` plus recently used projects at startup and leave the rest lazy.

**M6. `docs/PERFORMANCE.md` contradicts itself on the #259 result.** Line 3157 still says "**The throughput gain is not yet measured**", while the section below it reports 51 img/s. Line 3174 says the P-pe parity capture is still to run, which is accurate but means a #259 criterion is open (section 1). Fix: update or delete the stale paragraph.

**M7. Measurement claims without raw data, and one prose count that is wrong** (section 5).

- The #356 `ensure_indexes` "380 to 470 ms cold, 0.01 ms warm" figure is the root-cause number behind the startup bootstrap, and it has no raw file.
- The G2 worker-stage timings (86 against 220 ms per item), which are #362's baseline, have no saved stdout.
- The section D prose says "7 of the first 400 took 445 to 586 ms"; the raw file has 9 requests in that range, plus the one at 1,383 ms.
- The log cites no raw-data paths at all. The data was found under the scratch volume.
- Fix: commit or index the raw-data locations (paths, a manifest with checksums) and correct the count.

**M8. `docs/research/cluster_eval_baseline.md` does not reproduce from the retained scoring data.**

- The doc claims: 134 clusters, pair precision 0.893, recall 0.079, low-res precision 0.846.
- The only retained `score_current.json` for the same 539 labelled crops gives: 128 clusters, precision 0.905, recall 0.0875, low-res precision 0.945, 28 mixed clusters against the doc's 32.
- The `artifacts_local/cluster_eval/` files the doc cites do not exist. #341's "reproducible from a seed" criterion is therefore not demonstrated. A likely cause is that the partition changed between the scoring runs (the cluster-refresh daemon was active).
- Fix: freeze the partition file the baseline was computed from, record its checksum, and re-score.

### Low

- **L1.** `crop_jpeg_from_disk` (`src/services/curation/crop_bytes.py:96-104`) duplicates the rectangle arithmetic of `geometry.crop_rect` without its upper clamp. A box with `x1 == 1.0` gets a one-pixel padded column on a cache miss, so a miss is not byte-equal to the cached crop. `env.template` says a miss gives the "same bytes of result"; that is also untrue for GPU-route crops (cv2 at quality 90 against Pillow). Fix: use `crop_rect` and soften the doc.
- **L2.** `bootstrap_opensearch_indexes` logs `curation_indexes_bootstrapped project=...` (`src/core/dependencies.py:151`) even when `ensure_indexes` failed and returned silently. The log is misleading; check `_INDEXES_BOOTSTRAPPED` before logging success.
- **L3.** `_warm_routes` (`src/app/lifespan.py`) depends on FastAPI 0.143 internals (`_IncludedRouter._effective_candidates`). `requirements.txt` leaves `fastapi` unpinned (only `requirements-test.txt` pins 0.143.0), so the image can ship a version where the warm-up is a no-op or behaves differently. Pin it in the image or make the test fail on a version change.
- **L4.** `forget_bootstrapped(slug)` on delete (`src/services/projects/delete.py:510-512`) is untested (M25 survived) and has no practical effect: deleted slugs are retired (`lifecycle.py:275`) and only the deleting worker forgets. Delete it, or test a delete-then-request path if it is kept.
- **L5.** The combine path inlines the old `sha256(path|imohash)` formula (`src/services/projects/combine/image_copy.py:85`). A combine target therefore does not get the #358 create-guard property: a later concurrent ingest of the same content into the target can store a second image. This is documented in `docs/opensearch_schema_design.md`; noted for completeness.
- **L6.** `gpu_image_prep` stored-box mode puts no upper bound on rows per request: `_embed` allocates `(1 + N) x 3 x 336 x 336` bytes, about 0.34 MB per box. A photo with thousands of region boxes asks for GB-scale memory beyond the CuPy pool. The failure path is safe (TritonError, then `call_failed`, then the CPU path), but a cap in `embed_on_gpu` would avoid wasted calls.
- **L7.** Dataset import duplicate re-attach (`dataset_import/chunk.py:246-247,301-319`) writes a second `pending` ledger row. When the import resumes with a prior row whose `image_created` is true, the final row keeps `image_created: true` for an image another writer created. Undo is protected by `nothing_else_on` (`undo.py:424-434`), so this only affects the report counts.
- **L8.** The #362 branch adds an autouse fixture (`tests/conftest.py`) that stubs `TritonControlService.get_model_config` for every test. This hides any other caller's reliance on the real method. Scope it to the profile-validation tests.
- **L9.** The committed `models/gpu_image_prep/config.pbtxt` carries `count: 2`. A checkout with `GPU_PROFILE=standard` and the decode forced on gets two instances until `./openprocessor profile` is run. This is documented in `docs/PERFORMANCE.md`, but `openprocessor gpu plan` only warns.
- **L10.** The installer's numbered tier menu is not described on `docs-site/docs/getting-started/installer.mdx`. The CHANGELOG mentions it.

## 3. Tests that cannot fail (mutation results)

Each mutation was applied to a clean export of the commit, then the named tests were run with `-x`, then the change was reverted.

| ID | Mutation | Tests run | Result |
|---|---|---|---|
| M1 | Writer uses `index` instead of `create` | `test_ingest_concurrent_duplicates`, `test_ingest_writer` | caught |
| M2 | Writer writes the items of a duplicate unit | same | caught |
| M3 | `index_items` ignores `duplicate` | `test_ingest_concurrent_duplicates` | caught |
| M4 | Image id salted per call | same | caught |
| M5 | Crop-cache file not per project | `test_crop_cache_project_scope` | caught |
| M6 | Delete leaves the crop-cache dir | same | caught |
| M7 | `crop_pil` back to the unclamped rule | `test_crop_is_stored_box` | caught |
| M8 | Reprocess `crop_arrays` through a JPEG round trip | `test_region_vector_no_roundtrip` | caught |
| M9 | Worker stage through a JPEG round trip | same, plus `test_region_embed_stage` | caught |
| M10 | `EMBED_CHUNKS_IN_FLIGHT` = 1 | `test_reprocess_embed_overlap` | caught |
| M11 | No GPU row-count check | `test_reprocess_gpu_embed` | caught |
| M12 | No `ensure_indexes` at startup | `test_startup_bootstrap_warm` | caught |
| M13 | No route warm-up | `test_app_lifespan` | caught |
| M14 | `offload_always` ignored | `test_cpu_pool`, `test_reprocess_embed_overlap` | caught |
| M15 | `full` profile with one prep instance | `test_gpu_image_prep_instances`, `test_gpu_image_prep_config` | caught |
| M16 | GPU `crop_rect` without the x2 clamp | `test_gpu_image_prep_geometry` alone: not caught (36 passed); with `test_crop_is_stored_box`: caught (3 failed) | caught by the parity test |
| M17 | Import ignores `duplicate` | `test_import_concurrent_copy` | caught |
| M18 | #350 fix reverted | `test_cluster_refresh_schedule` | caught |
| M19 | Batch duplicate follows only a `success` representative | concurrent-duplicate, writer and integration tests | caught |
| M20 | `prime_text_backend` made a no-op | `test_startup_bootstrap_warm` | caught |
| M21 | No model-outdated check | `test_reprocess_gpu_embed` | caught |
| M22 | Prune ignores project subdirs | `test_crop_cache_project_scope`, `test_crop_cache_dir_consistency` | caught |
| M23 | False-merge rate uses the wrong numerator | `test_cluster_eval_metrics` | caught |
| M24 | Crop cache written before the write | `test_ingest_concurrent_duplicates` | caught |
| M25 | `forget_bootstrapped` not called on delete | `tests/projects`, `test_startup_bootstrap_warm` (700 tests) | **not caught** (L4) |
| B1 | #357 coverage threshold 0 | `test_cluster_pass_coverage` | caught |
| B2 | #357 `partial` is not a failed stage | same, plus `test_autolabel_job_scope` | caught |
| B3 | #357 pipeline continues after `partial` | `test_autolabel_job_scope` | caught |
| B4 | #357 write failures ignored | `test_cluster_pass_coverage` | caught |
| B5 | #357 reset skips the leftover check | `test_cluster_reset` | caught |
| B6 | #362 decode per task, no grouping | `test_region_embed_stage` | caught |
| B7 | #362 unbounded concurrent decodes | same | caught |
| B8 | #362 output-name check skipped | `test_profile_validation` | caught |

## 4. Docs, comments and upgrade notes

- **Upgrade notes:** see M3. The `POST /curation/projects/{project}/reprocess` body in the docs page is real: `targets.filter.all_images`, `embed.parts` and `dry_run` all exist in `src/services/curation/reprocess_models.py`. With `parts: ["region"]` it re-embeds accepted and false-positive boxes only (`reprocess_embed.embeddable`), as the page says. The page does not fix crop vectors, which the CHANGELOG says it does.
- **`docs/PERFORMANCE.md`:** stale paragraph, see M6.
- **`env.template` crop-cache block:** "same bytes of result" is inaccurate (L1).
- **`config_templates/profiles/full.json`:** the description "48GB+" does not match the installer threshold (H1).
- **Docs-site pages touched** (`installer.mdx`, `quick-start.mdx`, `basic.mdx`, `compose-profiles.mdx`, `ingest-large-folder.mdx`, `monitoring.mdx`, `upgrade-and-backup.mdx`): consistent with the code apart from M3 and L10. The `op_reprocess_decode_total` row matches `ops_metrics.py`.
- **`docs/research/CLAUDE.md`:** does not exist on main, so there was nothing to check against.

## 5. Measurement claims against the raw data

Raw data was found outside the repository, under the scratch volume (`op-data/cluster_ladder`, `op-data/lm`, `lv`, `lw`, `lx`, `cluster-research-data/eval10`). More than 25 numbers were checked, recomputed where needed:

- **Supported:**
  - #259: 27.7 to 51.1 img/s; 28.1 against 12.2 img/s; CPU-s 0.025, 0.029 to 0.032 and 0.497.
  - Encoder busy 84.5 to 92.1 percent.
  - #360 parity: min 0.9835, p01 0.9970, median 0.99927. Before the fix: 0.842, 0.886 and 0.965.
  - #259 stored-box edge min 0.9993; interior min 0.9609 with 24 below 0.999.
  - #268: 31.8 to 48.0 img/s, and +0.6 percent for a third instance.
  - #358: 3,184 to 3,189 ingested in 11 of 24 runs.
  - #337 ladder: 22.0, 80.6, 203.2, 231.2, 203.5, 322.6 and 596.0 s.
  - Fit timings, the k-NN table, the contention run (+7.5 percent) and the E3 restart rows.
  - CHANGELOG `[Unreleased]` figures.
- **Unsupported (no raw file):**
  - The #356 `ensure_indexes` 380 to 470 ms.
  - The G2 worker-stage 86 against 220 ms per item. The "2.7x" there is the CPU ratio (0.212 / 0.077); the wall ratio is 2.57x.
  - The #358 distinct-imohash and orphan counts.
- **Mismatches:**
  - Section D prose count (7 stated, 9 in the raw file).
  - The memory table is headed "median of 3", but peak RSS is the maximum of 3 and the index sizes are repetition 0. The 1M index-after-load figure of 7,899 MB is about 50 percent above the other two repetitions.
  - Sections A and B "GPU util %" and "Triton CPU %" sit 3 to 7 points and 70 to 120 percent off `results.jsonl`, with the method not stated. The conclusions do not change.
  - The cluster-eval baseline (M8).

## 6. Dead code, whitelists, duplication

- **Ratchet baselines:** no whitelist or baseline entries were added in the range. `scripts/codegen/baselines/complexity.txt` only shrank (two lines removed). No stale entries refer to the removed `src/services/detection/region_embed.py` or `image_id_for`.
- **Deleted in the same change, no shim left:** `region_embed.py` and its test; the `decode_failed` counter of the backfill script.
- **Duplication:**
  - L1: crop rectangle in `crop_bytes.py`.
  - L5: combine id formula.
  - Two EXIF-aware decode helpers: `reprocess_embed._decode_image` (from bytes) and `crop_bytes.decode_source_rgb` (from a path, new in the range). Both are small; worth merging when M2 is fixed.
- **Ineffective:** `forget_bootstrapped` (L4).

## 7. Security

- **Crop-cache paths:** slugs match `PROJECT_SLUG_RE` (`src/config/projects.py:30`) and crop ids are sha256 hex. The delete path goes through `_require_project_scoped_path` (`delete_paths.py:26-45`), which refuses the shared root and sibling slugs. An empty `OP_CROP_CACHE_DIR` falls back to the default (`env_str`), so `<cwd>/<slug>` can never be removed. `_drop_cached_crop` only receives ids from server-side sources (stale ids, the ledger, reconcile).
- **Shell:** the installer tier menu indexes its presets only after `case [1-4]`. `cli_gpu.sh` reads the instance count from the model config with `sed` and only multiplies it. No injection surface was found.
- **Secrets and host paths:** none in the new docs (no `/mnt`, `/home`, private IPs or tokens). The installer lock and rename logic is not in this range (it is on `fix/v070-installer-drives`, out of scope).
- **`gpu_image_prep` `boxes` input:** Triton enforces the dtype and the `[-1, 4]` shape. Out-of-range values are clamped by `crop_rect`, and NaN or inf raises inside `_run`, which becomes a TritonError and then the CPU path. The empty `[0, 4]` case is live-verified (log F4). Memory: see L6.

## 8. The unmerged branches

### `fix/v070-357-work` (`a9c75abf`): mergeable after doc corrections; keep #357 open

- **What it fixes:**
  - The pass counts its pool just before the fetch and returns `status: partial` below 99 percent coverage, or on write failures. A partial pass records no last run, and the auto-label pipeline stops and the job fails (`failed_stages`).
  - The bench reset is a polled `update_by_query` task over documents that still hold cluster fields, and it fails if any remain.
  - Tests are real: B1 to B5 were caught. Lint is clean, the targeted tests pass, and the complexity ratchet goes down.
- **Gap (Medium, acceptance PARTIAL):** the case #357 actually observed is the refresh daemon assigning items before the pass starts. That still reports `success` in production, because only the bench passes `expected_pool`. The branch's own test `test_a_pool_that_shrank_before_the_pass_is_not_partial_without_an_expectation` pins this behaviour. The measurement-log edit nevertheless says "(both fixed in #357)" and that the pass "now returns `status: partial`" for that case. Correct the log, or have production callers pass the pool they expect (for example the retrain policy's count at dispatch).
- **Other gaps:**
  - The 250k reset is verified only against an offline stand-in server, not live.
  - No CHANGELOG entry.
  - A persistent shortfall would fail every auto-label job. Fail-closed by design, but the refresh daemon would then re-queue a failing job every cycle; consider a backoff.

### `fix/v070-362-work` (`04dde264`, `0d350fac`): mergeable after the docs row and CHANGELOG; keep #362 open for the re-measurement

- **What it fixes:**
  - The region-embed stage groups a flush's boxes by source path. Each photo is decoded once, with concurrent decodes bounded by `min(cpu/2, OP_INGEST_INFLIGHT_MB / 64)`.
  - Profile validation rejects a READY detector without an `output0` tensor at activation (a warning on save). Unreadable config fails closed. `contracts/openapi/curation.json` is updated.
  - Tests are real: B6 to B8 caught, 69 targeted tests pass, lint is clean.
- **Gaps:**
  - The criterion "stage cost at 21 MP no worse than 1.2x v0.6.0" is not re-measured. The G2 baseline itself has no raw data (M7).
  - `docs-site/docs/guides/region-profiles.mdx` (validation-code table) has no row for `detector_model_unusable`.
  - No CHANGELOG entry.
  - M2 (a non-`OSError` decode exception escaping the stage) is not fixed. With `asyncio.gather` and no `return_exceptions`, one bad photo now fails the region vectors of the whole flush and still escapes to the writer. This branch is the natural place to fix it.
  - The autouse conftest stub (L8).

## Not verified

- Live behaviour: no stacks or GPUs were used. These rest on the measurement log plus the raw-file spot checks:
  - the H1 OOM itself (the planner output was reproduced; the load failure was not);
  - the M4 transport-retry path;
  - the M2 trigger, which depends on Pillow-version-specific EXIF exceptions;
  - the M5 startup load.
- The #358 distinct-content and orphan counts and the #356 bootstrap timing: the raw data is missing or the projects were deleted.
- The `pre-commit` hook set (ratchets, naming scan, contract drift) was not run on the two branches, to avoid auto-fix writes in their worktrees; only `ruff` was run, read-only.
- The research notes added in the range (`cluster_types.md` and the spatial-memory note) were only read for leaks, not for technical content.
