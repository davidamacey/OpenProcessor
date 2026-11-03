# B1 track 1D: API correctness, frontend, test integrity, two security items

| Field | Value |
|---|---|
| Status | plan, not started |
| Batch / release | B1, part of the single v0.7.0 release |
| Depends on | B0 merged; `int/b1-correctness` cut. Independent of 1A, 1C, 1E. 1B rebases over 1D (1B converts `eval_datasets.py`, which BA-4 touches) |
| Decisions applied | D-13; owner rules: maintained libraries first (`cachetools`), no asyncio fan-out in request handlers (one offload per request is not fan-out), thin frontend (backend serves data), fail closed |
| Source IDs | `codebase_audit_plan.md` 4.1 (BA-1, BA-2, BA-3, BA-4, BA-5, BA-8, SEC-3, SEC-6), 4.4 (FE-1, FE-2, FE-3, FE-5, FE-7), 4.5 (T-1, T-2, T-3, T-4, T-7), 4.7 (CC-5 handlers only); the former master plan (rev 4, in git history) 3c (Q-D14), 7.1 1D row (incl. T-8, from ledger section 14); issues N4, N6, N9, N10 (`09_github_actions.md`) |
| Size | M |
| Lane | branch `b1/api-frontend`, worktree `<repo>-wt-b1-api-frontend`, compose project `op-b1-api-frontend` |

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read only, with `rg`).
Consolidated 2026-11-03.

Scope note: 1D owns no dependency bumps. The open dependabot PRs (#433 frontend, #434 backend) are
re-applied by the separate `b1/deps` lane (`02_b1_overview.md` section 4.1), merged before 1D.

Not in this track: BA-6 (71 untyped operations, one router per PR after B4; backlog), BA-7 (done
in B0), CC-5 upload extract and directory ingest (B2 jobs), FE-4 (refuted), FE-6 and T-6 (evidence
first, backlog in `11_risks_cleanup_host.md`), SEC-4 and SEC-5 (B2).

## 1. Goal

Per-process caches stop serving stale or cached-error results across 32 workers, curation bodies
stop carrying undeclared keys, error bodies stop echoing exception class names, thumbnail and
render handlers stop blocking the event loop, the diverse review page makes one batch read, the
export stats refresh, SSE reconnects resync, and tests that cannot fail are fixed so they can.

## 2. Preconditions

- Lane worktree: `git worktree add <repo>-wt-b1-api-frontend -b b1/api-frontend int/b1-correctness`.
- The venv; a real `npm ci` in the worktree's `frontend/` (not a symlinked `node_modules`).
- No Docker, no GPU in the lane. Browser screenshots are taken in the batch sweep on the merged tree
  (`02_b1_overview.md` section 7); the lane runs vitest and `npm run check` only.

## 3. Tasks: backend API

Red test first, seen failing on the base, paired control; one commit each.

### 1D-1 BA-1: class-name cache stale in other workers after a promote (S)

- File: `src/utils/class_names.py` (`get_class_names`, `invalidate_class_names`); callers
  `src/services/training/triton_promote.py` (two `invalidate_class_names` calls); test
  `tests/test_class_names.py` (uses `invalidate_class_names`).
- Fix: key the cache entry by `(model_name, st_mtime_ns, st_size)` of `labels.txt`, one `stat()`
  per lookup (`ClassRegistry.load()` used mtime the same way). Do not cache a missing-file result.
  Delete `invalidate_class_names`, its two calls and its `__all__` entry; adapt the test.
- Doc: `os.stat_result.st_mtime_ns` (https://docs.python.org/3/library/os.html#os.stat_result).
- Traps: promote rewrites `labels.txt` in place: stat the file, not the directory; use
  `st_mtime_ns` plus size (coarse mtime can miss a fast rewrite).
- Red: write labels v1, call `get_class_names`, rewrite as v2 with a new mtime without invalidating,
  assert v2. Fails today.
- Control: unchanged file, a second lookup does not re-read it (count `_read_labels_file` calls).
- Done when: a two-process script (A promotes, B reads) prints v2 in B with no invalidation call.

### 1D-2 BA-2: timing middleware injects keys into `/curation` bodies (S)

- File: `src/app/factory.py` (`rg -n "path in request.url.path" src/app/factory.py`, the
  `is_inference_endpoint` substring list).
- Fix: skip the body rewrite when `request.url.path` starts with `/curation` (curation has its own
  serializer and no `/v1` twin). Keep the keys on core routes (benchmark docs read
  `total_time_ms`). Do not convert middlewares to pure ASGI (Appendix D).
- Traps: the `/v1` aliases keep the keys; `rg total_time_ms frontend/src contracts` shows no curation
  consumer.
- Red: `GET /curation/projects/x/ingest/policy` (OpenSearch stubbed) has no `total_time_ms`.
- Control: `POST /detect` still has it.

### 1D-3 BA-3: per-process caches cache errors and never evict (S-M)

- Files: `src/clients/curation_opensearch/settings_doc.py` (5 s TTL cache that stores `{}` on any
  error), `src/services/curation/diverse_order.py` (`_ORDER_CACHE`), `src/services/curation/clustering/outliers.py`
  (`_CACHE`), `src/services/curation/image_serving.py` (try-imports `cachetools`, with a hand-rolled
  fallback), `requirements.txt`, `requirements-test.txt`, `ATTRIBUTION.md`.
- Library: `cachetools.TTLCache(maxsize=N, ttl=600)` (https://cachetools.readthedocs.io/, MIT).
  Not a declared dependency today: add it to both requirements files in the repo's pin style (check
  PyPI for the current release and pin as the neighbouring pure-Python packages are pinned), add the
  MIT row to ATTRIBUTION.md, and delete the try-import fallback in `image_serving.py` in the same
  commit.
- Fix:
  1. `settings_doc`: on an error, raise (the caller's 503 path) instead of caching `{}`; correct the
     docstring to "5 s per-process TTL; another worker may serve the previous value for up to 5 s".
  2. `diverse_order`, `outliers`: replace the dicts with `TTLCache`.
  3. Leave the capacity cache, `regions_fp` and `field_coverage` alone (verdict).
- Traps: `tests/conftest.py::_PROCESS_CACHES` clears `_ORDER_CACHE`, `_CACHE`, `_settings_cache`
  between tests; `TTLCache` supports `.clear()`, check the clearing helper still works. The
  count-based invalidation stays; no `max(updated_at)` query.
- Red: a settings fetch that raises surfaces as an error and does not return `{}` on the next call
  within 5 s; inserting `maxsize + 1` keys into the order cache drops the oldest.
- Control: a successful settings fetch is cached for the TTL.

### 1D-4 BA-4: `eval_datasets` cache key narrower than its inputs (S)

- File: `src/services/curation/eval_datasets.py` (`_cache_key`, `_CACHE`).
- Fix: resolve the registry names at `to_wire` time, uncached; add the mtimes of `data.yaml` and of
  the `images/test` directory to `_cache_key`. No eviction needed (bounded by the dataset count).
  Track 1B later converts this file's registry read to the document snapshot.
- Red: rename `cat` to `feline` in the registry, add one image, edit `data.yaml`: the listing shows
  `feline` and `n_images == 2` (today `cat`, 1, same object).
- Control: unchanged inputs return the same object.

### 1D-5 BA-5: error bodies echo the exception class name (S)

- Files: `src/app/factory.py::global_exception_handler` (500 body `error_type`),
  `src/core/error_handlers.py` (the Triton 503 body `error_type`); readers
  `tests/test_response_serialization.py` (asserts the key set `{'detail', 'request_id',
  'error_type'}` for 500 and 503), `tests/test_triton_unavailable_503.py` (asserts
  `body['error_type'] == 'RetryExhaustedError'`).
- Doc: FastAPI error handling (do not leak internals): https://fastapi.tiangolo.com/tutorial/handling-errors/ .
  The missing CORS header on a 500 is documented Starlette behaviour (`ServerErrorMiddleware` is
  outermost); leave it (Appendix D).
- Fix: remove `error_type` from response bodies, keep it in the log lines. Run
  `rg -n error_type tests frontend/src contracts` first (today: the two tests above plus
  `tests/curation/occ_fakes.py`, which is an OpenSearch error fake, unrelated) and update every
  reader in the same commit; `make contracts` if a schema lists it (none today).
- Red: a raising route's 500 body has no `error_type`; the Triton 503 body has none.
- Control: both bodies still have `request_id`.
- Done when: `rg -n "'error_type'" src` shows log calls only.
- Breaking (wire): yes, list it in CHANGELOG.

### 1D-6 BA-8: probe concurrency cap is per process (S, docs only)

- File: `src/services/labeling/vlm_endpoint_probe.py` (`MAX_CONCURRENT_PROBES = 2`, the
  `ProbeBusyError` text).
- Fix: say "per API worker process" in the constant's comment and in the user-facing error text.
  No cluster-wide lease (Appendix D).
- Test: update any assertion of the error text (`rg -n "MAX_CONCURRENT_PROBES|ProbeBusyError" tests`).
- Done when: both texts say the cap is per process.

### 1D-7 CC-5 (handlers): blocking PIL work in `async def` handlers (S)

- Files: `src/routers/curation/images.py` (`crop_thumbnail`, `region_thumbnail`: sync
  `THUMBNAIL_CACHE.get_or_compute` with a PIL decode), `src/services/curation/image_serving.py`
  (`render_source_image`: PIL open, transpose, resize, encode inline).
- Library: the project's bounded CPU executor `src/utils/cpu_pool.py::run_cpu` (runs in place under
  `OFFLOAD_MIN_BYTES` unless `offload_always`; `nbytes=None` offloads). Docs: FastAPI "Concurrency
  and async / await" and Python asyncio "Running Blocking Code" (`asyncio.to_thread`)
  (https://fastapi.tiangolo.com/async/ , https://docs.python.org/3/library/asyncio-dev.html#running-blocking-code).
- Fix: wrap the one blocking call in each of the three with `run_cpu` (one offload per request; not
  fan-out). No ruff ASYNC rule (Appendix D).
- Traps: `ThumbnailCache.get_or_compute` is now called from executor threads: check it for shared
  mutable state; a race costs at most one duplicate compute (per-process cache).
- Red: an event-loop lag probe (a coroutine measuring `loop.time()` drift) while a large-image
  thumbnail is computed: drift above 100 ms today, below 20 ms after.
- Control: a cached thumbnail shows no drift either way.
- Done when: the three handlers do no PIL work on the loop.

### 1D-8 SEC-3: harden persisted clustering state (S)

- Neutral. Details are in the maintainers' private notes; read them before starting. Issue N9.
- Red test and control as given in the private notes; keep the public commit message and test names
  neutral (no mechanism, no path, no exploit wording).

### 1D-9 SEC-6: widen the CLI secret redaction key list (S)

- Neutral. Details are in the maintainers' private notes. Issue N10. The URL-userinfo redaction
  extension is cut (Appendix D).
- Red test and control as given in the private notes; neutral commit message.

## 4. Tasks: frontend (Cropwright, `frontend/`)

Frontend rules: Svelte 5 + TypeScript, 2-space indent, `$lib` aliases, `npm` only; the backend
serves data and decisions, the frontend renders. Each task ships with a vitest test; the browser
check at 1440 and 430 px happens in the batch sweep.

### 1D-10 FE-1: `/export` makes a raw `fetch` of `/stats/dataset` that never refreshes (S)

- File: `frontend/src/routes/p/[project]/export/+page.svelte` (an `$effect` with a raw
  `fetch(`${apiBase}${scoped()}/stats/dataset`)`), `getStats` in the API module that builds
  `StatsSummary`.
- Fix: add `by_source` to `StatsSummary` in `getStats` (already in `RawDataset`); delete the effect
  and its imports; `const hddSources = $derived(stats?.by_source ?? [])`; keep the `dataset_error`
  notice. Doc: Svelte `$derived` (https://svelte.dev/docs/svelte/$derived).
- Red: mount with one stubbed GET, click Refresh, the chips update with exactly one request per load.
- Control: a 503 shows the notice.
- Done when: no `fetch(` outside `api*.ts` (`rg -n '\bfetch\(' frontend/src --glob '!*.test.ts'`).

### 1D-11 FE-2: diverse page fans out 30 `GET /crops/{id}` (S)

- Files: `frontend/src/lib/review/reviewDiverseController.svelte.ts`
  (`Promise.all(ids.map((id) => getCrop(id)))`), `frontend/src/lib/api_cropRead.ts`.
- Correction from the verdict: the batch read exists, `GET /curation/projects/{project}/crops?ids=a,b,c`
  (max 500, in order, missing ids dropped); no frontend function uses it.
- Fix: add `getCropsByIds(ids, signal)` in `api_cropRead.ts` over `GET {scoped()}/crops?ids=`; call
  it once per page with the loader's signal. No backend or contract change.
- Traps: check `mapRawCrop` accepts the list-row shape (the review list loader already maps list
  rows).
- Red: one stubbed list response missing 1 of 30 ids gives 29 items and no page error, from exactly
  one request.
- Control: all 30 present.

### 1D-12 FE-3: project binding in `load` with global hover preload (S)

- Files: `frontend/src/app.html` (`data-sveltekit-preload-data="hover"` stays), the three
  cross-project anchor sites `frontend/src/routes/projects/+page.svelte`,
  `frontend/src/routes/projects/combine/[job_id]/+page.svelte`, `ProjectUnavailable.svelte`
  (`rg --files frontend/src | rg ProjectUnavailable`).
- Doc: SvelteKit link options (https://svelte.dev/docs/kit/link-options).
- Fix: `data-sveltekit-preload-data="tap"` on the three sites; a guard test that no `/p/` page
  renders `projectHref(..., otherSlug)` without `tap`. Do not move `select()` out of `load`
  (Appendix D).
- Red: the guard fails on a fixture anchor without `tap`.
- Control: the current tree passes.

### 1D-13 FE-5: client-side sums in `TrainDatasetSection` (conditional, S)

- File: `frontend/src/lib/components/train/TrainDatasetSection.svelte` (two `reduce` sums of
  `validated_count`; path corrected from the audit's shorter form).
- Fix: only if `getStats().validated_crops` is already loaded on `/train`, render it and delete the
  sums. Otherwise leave the sums and note them in `frontend/CLAUDE.md`'s served-facts section as a
  known fallback.
- Red: a stubbed stats value that differs from the per-class sum shows the served value.
- Control: equal values.

### 1D-14 FE-7: SSE reconnect does not resync keymap and config (S)

- Files: `frontend/src/routes/p/[project]/+layout.svelte` (the `config.changed` keymap subscriber),
  the config controllers that subscribe to `config.changed` (`rg -n "config.changed" frontend/src`),
  `openStream` (unchanged; it already supports `onOpen`, as `DatasetStats` uses it).
- Fix: pass `onOpen` from those subscribers and refetch on every open after the first (the backend
  sends no SSE `id:`, so missed events are lost).
- Red: a fake EventSource errors and reopens; the keymap refetch fires once.
- Control: the first open does not refetch.

## 5. Tasks: test integrity

### 1D-15 T-1: assertions only inside a loop over a registry that may be empty (S)

- Sites: `tests/curation/test_crop_scores.py::test_scorer_metadata_writes_match_scorer_class_writes`,
  `tests/curation/test_class_sources.py::test_every_catalog_entry_has_a_short_badge_label`,
  `tests/curation/test_review_sorts.py` (two tests over `get_review_sorts()`).
- Fix: `assert <registry>` before each loop. No ratchet (Appendix D).
- Red: emptying each registry in a scratch copy now fails each test (today they pass).
- Control: the unmutated tree passes.

### 1D-16 T-2: `test_gate_sim` open-set test never runs its asserts (S)

- File: `tests/research_scripts/test_gate_sim.py` (`if out['propagated']:` wraps both asserts; with
  seed 3 `propagated == 0`).
- Fix: choose inputs that propagate and assert exact counts, as the sibling test does (`== 120`);
  remove the `if`.
- Red: mutating `open_propagated` to `10**9` in `gate_sim.py` fails the test (passes today).
- Control: the unmutated script passes.

### 1D-17 T-3: compose render tests skip on a non-zero exit (S)

- Sites: `tests/test_monitoring_config.py`, `tests/compose/test_vlm_service_command.py`,
  `tests/compose/test_images_build_and_overlays.py` (each `if result.returncode != 0:
  pytest.skip(...)`); the correct form is in `tests/compose/test_overlay_render.py`.
- Fix: skip only when the docker CLI or the compose plugin is absent ("unknown command"); otherwise
  `assert returncode == 0, stderr`.
- Traps: the tests run with a stripped environment; a required `${VAR:?}` on a fresh checkout now
  fails instead of skipping. Run the three files once on the CI runner configuration before merging
  (risk in `02_b1_overview.md`).
- Red: append invalid YAML to the compose file in a `git archive` scratch copy: the three tests
  fail (they skip today).
- Control: the clean tree passes.

### 1D-18 T-4: `einops` missing from `requirements-test.txt` (S)

- Files: `requirements-test.txt`; `tests/test_pe_rearrange_shim.py` (8 tests skip today: "could not
  import 'einops'").
- Fix: add a pinned `einops` (MIT; ATTRIBUTION only if the file lists test-only dependencies; check)
  and install it in the venv. Do not add `leidenalg` / `igraph` (Appendix D).
- Red: `pytest -rs tests/test_pe_rearrange_shim.py` shows the skip today.
- Control: after the fix the module runs on the CPU torch CI uses.

### 1D-19 T-7: no test asserts project disk-usage bytes (S)

- File: `src/services/projects/stats.py` (`_dir_bytes`, `_disk_cache`, 300 s); new test.
- Fix: a test with a temporary project directory holding a 10-byte file; clear `_disk_cache` in the
  test or add it to `_PROCESS_CACHES` in `tests/conftest.py`. No registry (Appendix D).
- Red: mutating `_dir_bytes` to return a constant fails the new test (9 tests pass today with it).
- Control: the real code gives 10.

### 1D-20 T-8: test-order pollution (pre-existing on main; ledger section 14) (S)

- Symptom: `tests/curation/cross_project_leak/test_keymap_and_ingest_policy_isolation.py` fails (3
  tests) when run after `tests/projects/test_clone_settings.py` and
  `tests/projects/test_clone_activations.py` in one process. No single file pollutes it alone, and
  xdist hides it in CI.
- Steps: bisect which process-level state survives (start with the module caches not in
  `tests/conftest.py::_PROCESS_CACHES`: keymap and ingest-policy caches, config-store snapshots,
  the project binding); reset it in the shared fixture (add it to `_PROCESS_CACHES`), not in the
  victim test.
- Red: a test that runs `python -m pytest -p no:xdist -q tests/projects/test_clone_settings.py
  tests/projects/test_clone_activations.py
  tests/curation/cross_project_leak/test_keymap_and_ingest_policy_isolation.py` in a subprocess and
  asserts exit 0 (fails today). Keep it as a regression test (it takes seconds).
- Control: each of the three files alone passes.

## 6. Gate for this lane

- Backend: targeted pytest per task, `.venv/bin/pre-commit run --files <touched files>`.
- Frontend: in `frontend/`, `npm ci`, then `npx vitest run <touched test files>` and
  `npm run check` (svelte-check is a static check). Lint the touched files
  (`npx prettier --check <files>`, `npx eslint <files>`). The full set (`npm run lint`,
  `npx knip`, `npm test`, `npm run build`, `npm run test:e2e`) runs once in the batch sweep.
- `make contracts` if 1D-5 touches a schema (none expected).
- No full suite, no `--all-files`, no Docker.

## 7. Done when

- Every task's done-when holds; `rg -n "'error_type'" src` shows log calls only;
  `rg -n "invalidate_class_names" src tests` finds nothing; no `fetch(` outside `api*.ts`; one
  request per diverse page.
- The ordered T-8 run passes; the three compose tests fail on a corrupted compose file.
- The closing B1 PR moves the keep-list below to its targets and deletes this file
  ([00_index.md](00_index.md) "Plan lifecycle").

### 7.1 What to keep after merge

| Fact or section | Target doc |
|---|---|
| per-process cache rules (BA-1, BA-3, BA-4), the probe cap note (BA-8), body-shape changes (BA-2, BA-5) | `docs/ARCHITECTURE.md`, `docs/CURATION.md`, `docs/design/curation_api_contract.md` |
| the Breaking entry (no `error_type`, no `total_time_ms` / `request_id` in bodies) | `CHANGELOG.md`, `frontend/CHANGELOG.md` |
| cachetools licence row | `ATTRIBUTION.md` |

## 8. Docs, contracts, CHANGELOG

- `ATTRIBUTION.md` (cachetools MIT row; einops if test-only deps are listed).
- `docs/CURATION.md` or the API docs where the probe cap is described (per process).
- `frontend/CLAUDE.md` (FE-5 fallback note, if that branch is taken).
- CHANGELOG `[Unreleased]`: Fixed for BA-1, BA-2, BA-3, BA-4, CC-5, T-items are not user-visible
  (no entry, or one "tests" line); **Breaking** for BA-5 ("error bodies no longer carry
  `error_type`; use `request_id` and the server log") and BA-2 ("curation JSON bodies no longer
  carry `total_time_ms` and `request_id`; the `X-Process-Time` and `X-Request-ID` headers remain").
  Security entries for SEC-3 and SEC-6 as neutral one-liners.
- `frontend/CHANGELOG.md`: FE-1, FE-2, FE-3, FE-5, FE-7.

## 9. Upgrade impact

The images rebuild (new `cachetools` and `einops` test dependency). No data or mapping change.
Clients that read `error_type` or curation `total_time_ms` must stop (Breaking list).

## 10. Rollback

Revert the 1D merge commit alone; no coupling (1B is rebased over it and does not depend on its
behaviour).

## 11. Risks

| Risk | Mitigation |
|---|---|
| BA-5 removes `error_type` that a client reads | `rg error_type tests frontend/src contracts` first; Breaking entry |
| T-3 fails on a fresh checkout's required env vars | run the three files on the CI runner config before merging |
| BA-3 adds a dependency | maintained, MIT; ATTRIBUTION row; the try-import fallback deleted in the same commit |
| A thread race in `ThumbnailCache` after CC-5 | per-process cache; worst case one duplicate compute; read it before the change |

## 12. Corrections found while verifying

- BA-5: `error_type` is also in the Triton 503 body (`src/core/error_handlers.py`), and two tests
  assert it (`tests/test_response_serialization.py`, `tests/test_triton_unavailable_503.py`).
- FE-5's component is at `frontend/src/lib/components/train/TrainDatasetSection.svelte`.

## 13. Lane agent brief (paste as is)

```
You are the implementation agent for B1 track 1D (API, frontend, tests, two security items) of
OpenProcessor. Plan: docs/design/program/02d_b1_api_frontend_tests.md (this file). Execute it
standalone. Read first: CLAUDE.md, frontend/CLAUDE.md, docs/VISION_AND_GOALS.md,
docs/design/engineering_standards.md, docs/design/codebase_audit_plan.md sections 4.1, 4.4, 4.5,
4.7 (CC-5). For SEC-3 and SEC-6 read the maintainers' private notes the coordinator points you to;
never copy their details into a commit, test name, comment or doc.

Worktree: <repo>-wt-b1-api-frontend, branch b1/api-frontend (created by the coordinator from
int/b1-correctness). Work ONLY there. Never push, never merge, never check out another branch.
Compose project for any Docker command: -p op-b1-api-frontend (you should need none). GPU: none.
Deadline: <coordinator fills in>.

Rules:
- Tasks 1D-1 .. 1D-20, one commit each, conventional commits, hooks on.
- Red test first, run on unchanged code, SEE IT FAIL, paste the red output into the commit body;
  then fix; paired control green.
- Backend: targeted pytest + .venv/bin/pre-commit run --files <touched files>.
- Frontend: real npm ci in frontend/; npx vitest run <files>; npm run check; prettier/eslint on
  touched files. No full frontend set, no e2e, no full offline suite, no --all-files.
- cachetools is the library for bounded caches (cite its docs); run_cpu for the one offload per
  handler; no asyncio fan-out in handlers; never cache an error; no dead code (delete
  invalidate_class_names, the try-import fallback, the raw fetch).
- Never use pgrep -f, pkill -f, or sleep/until polling loops.

Report at the end: commits (sha + subject), red outputs in one line each, what T-8's shared state
turned out to be, anything not done and why, and any plan correction.
```
