# Codebase audit fix plan (0.7.0)

> **Execution order:** see `docs/design/program/00a_decision_register.md` (the program set replaced the former master plan, rev 4, in git history). The B0 fixes (GPU-1, GPU-3a/13, GPU-2, OS-2, OS-1, CC-1, OPS-1, BA-7) are DONE on `fix/411-r2-round`. The rest are scheduled in B1-B5 there. Security items stay neutral here; details are in the private notes.

Status: plan only, committed 2026-11-03. No code changed while writing it; B0 items are since done (see the header note).

- **Audited tree:** `main` at `5b95591f` (pinned worktree `.claude/worktrees/audit-r2`).
- **Status column checked against:** branch `fix/411-r2-round` at `4103656b`
  (worktree `<repo>-wt-r2round`). Checked with
  `git log main..fix/411-r2-round` and `git diff main...fix/411-r2-round`, by reading
  the diff. No tests were run for this plan.
- **Inputs:** `artifacts_local/audit/` holds `BRIEF.md`, `ADVERSARIAL_BRIEF.md`,
  seven findings files (`api`, `opensearch`, `gpu`, `frontend`, `tests`, `asyncio`,
  `security`), one `*.verdicts.md` per area, and `docs/design/task_queue_plan.md`
  (the Celery plan; its Phase 1 overlaps several findings here).
- **Line numbers** are from `5b95591f`. Some have moved on the branch. Find each site
  again with the `rg` pattern given in its task before you edit.
- **Security items** appear here only as neutral one-line references. The mechanism,
  reproduction and locations are in `artifacts_local/audit/security*.md`, which is
  private and must stay out of git.

A fresh implementer should be able to work from this file plus the private audit
files. Each task gives the verdict-corrected fix, not the audit's original one.

---

## 0. Owner rules every fix in this plan follows

1. **Cross-process correctness** comes from idempotency, OpenSearch optimistic
   concurrency (`if_seq_no`/`if_primary_term`, `op_type='create'`), leases on the job
   record, or the Celery job design in `task_queue_plan.md`.
   - Never add an in-process lock (`asyncio.Lock`, `threading.Lock`) or a file lock
     (`fcntl.flock`) as the correctness mechanism.
   - Existing `flock` helpers (`FileJob._state_lock`, `job_lock.exclusive_start_lock`)
     are not extended by this plan.
2. **No `asyncio` fan-out in request handlers.**
   - Where concurrency is really needed (worker code, or a documented data-plane
     exemption decided by J7 in `task_queue_plan.md`), use `asyncio.TaskGroup` as the
     Python docs describe ("a more modern alternative to `create_task`"; on a child
     failure it cancels the siblings).
   - Heavy work moves to workers.
3. **Read the docs, and use a maintained library before hand-rolling.** Each fix cites
   the doc it follows.
4. **Every test fails first and has a paired control.** Watch the red test fail on
   unchanged code before you apply the fix. The control must pass both before and
   after.
5. **No dead code and no shims.** A superseded helper is deleted in the same change.
   **Class identity is by name.**
6. **Lock rule.** Automated writers never overwrite human- or import-validated labels
   or the frozen holdout.

### How this plan changes `task_queue_plan.md` Phase 1

Phase 1 of the Celery plan prescribes two file locks that rule 1 forbids. This plan
supersedes those two items:

| Celery plan item | What it said | What this plan does instead |
|---|---|---|
| P1.0 | unique temp file **and** `FileJob.write` takes `_state_lock()` | unique temp file only (CC-1). Readers stop writing state (they derive `interrupted` on read); persisting stays with the existing repair path now and the lease reaper (Phase 4) later |
| P1.3 | `exclusive_file_lock(registry.lock)` around registry mutators | the registry moves to an OpenSearch document with OCC (CC-2). This needs the owner decision **J-REG** below |

P1.1, P1.2 and P1.6 are not audit findings. They build on
`exclusive_start_lock`, which is an `flock`, so **the owner must confirm** that the
existing start lock is acceptable there. An `O_EXCL` create and an atomic `rename`
claim are idempotent claims, not locks, so they comply with rule 1.

---

## 1. Index

Areas: **BA** backend-api · **OS** opensearch · **GPU** gpu/inference · **FE** frontend
· **T** tests · **OPS** ops/compose · **CC** cross-cutting concurrency · **SEC**
security (private).

Status values: **open** (open on main and on the branch), **fixed (branch)**,
**partly fixed (branch)**, **needs evidence**, **needs scoping**.

| Plan ID | Source | Title | Class | Severity | Size | Status | Batch |
|---|---|---|---|---|---|---|---|
| BA-1 | API-1 | Class-name cache stale in other workers after a promote | LIVE | medium | S | open | PR-4 (Celery Phase 7 prerequisite) |
| BA-2 | API-3 | Timing middleware injects keys into `/curation` bodies (substring match) | LIVE | low-medium | S | open | PR-4 |
| BA-3 | API-7 | Per-process caches: error cached as "no overrides", unbounded query caches | LIVE | low-medium | S-M | open | PR-4 |
| BA-4 | API-8 | `eval_datasets` cache key narrower than its inputs | LIVE | low | S | open | PR-4 |
| BA-5 | API-9 | 500 body echoes the exception class name | LIVE | low | S | open | PR-4 |
| BA-6 | API-10 | 71 operations without a response model | LIVE | low | L | needs scoping | later, one router per PR |
| BA-7 | API-11 | `review_dismiss` maps every OpenSearch failure to 404 | LIVE | low | S | open | PR-4 (R3 extra) |
| BA-8 | API-13 | Probe concurrency cap is per process; docs imply global | LATENT | low | S | open | PR-4 |
| OS-1 | OS-1 | First create of a config/keymap doc has no `op_type='create'` | LIVE | low-medium | S | open | PR-2 (Celery P1.4) |
| OS-2 | OS-2 | Holdout freeze ignores bulk item errors, then persists the record | LIVE | medium-high | S | open | **PR-1 (R3)** |
| OS-3 | OS-3 | Three more bulk writers drop item errors | LIVE | low | S | open | PR-2 (Celery P1.7) |
| OS-5 | OS-5 | Unbounded `page` reaches OpenSearch on two list routes and becomes a 503 | LIVE | low | S | open | PR-3 |
| OS-6 | OS-6 | Legacy visual-search kNN filter applied as a post-filter | LIVE | medium-low | S | open | PR-3 |
| OS-7 | OS-7 | Blocking `update_by_query` in `backfill_embedded_state` | LATENT | low | S | open | PR-3 |
| OS-8 | OS-8 | `delete_by_query` in `image_admin` and conflicts | UNVERIFIED | ? | S | needs evidence | after evidence |
| OS-9 | OS-9 | Scroll context not cleared on the error path | LATENT | low | S | open | PR-3 |
| OS-10 | OS-10 | Unbounded `offset` makes an aggregation size error | LIVE | low | S | open | PR-3 |
| GPU-1 | GPU-1 | `/ocr` maps boxes with the padded size, so boxes are displaced | LIVE | high | S | open | **PR-1 (R3)** |
| GPU-2 | GPU-2 | pyclipper missing from the Triton image; silent wrong unclip fallback | LIVE | medium | S | open | PR-1 (R3 extra, needs image rebuild) |
| GPU-3a | GPU-3 (part) | `/ocr` runs on the unoriented frame but reports the oriented size | LIVE | high | S | open | **PR-1 (R3)** |
| GPU-3b | GPU-3 (rest) | Five decoders, three EXIF conventions (`/embed`, `/analyze`, legacy ingest) | LIVE | medium | M | open | PR-5 |
| GPU-4 | GPU-4 | `gpu_decode_ready` caches a transient failure forever | LIVE | low-medium | S | open | PR-3 |
| GPU-5 | GPU-5 | Full bitmap allocated before the dimension check | LIVE | low-medium | S | open | PR-3 |
| GPU-6 | GPU-6 | `/detect` never upscales; ingest does | LIVE | ? | M | needs scoping (owner) | after decision |
| GPU-7 | GPU-7 | OCR recognition makes one BLS call per crop | LIVE (perf) | medium | M | open | PR-5 |
| GPU-8 | GPU-8 | Retry classifier branch is dead; the pool retries deterministic errors | LIVE | low | S | open | PR-3 |
| GPU-9 | GPU-9 | Sliver crop makes the CLIP resize allocate up to ~3 GB | LIVE | medium | S | open | PR-3 |
| GPU-10 | GPU-10 | PIL CLIP preprocessors truncate and black-pad one row | LIVE | low | S | open | PR-3 |
| GPU-12 | GPU-12 | Letterbox undo subtracts a half-pixel pad the paste never applied | LIVE | low | S | open | with #407 |
| GPU-13 | GPU-13 | RGBA compositing in OCR uses float64 full-frame temporaries | LIVE | low | S | open | PR-1 (rides on GPU-3a) |
| GPU-14 | GPU-14 | Normalised-or-pixel boxes guessed from `max() > 1.0` | LATENT | medium | S | open | PR-3 |
| GPU-15 | GPU-15 | Region detector guesses the head layout | LATENT | medium | M | **fixed (branch)** | merge branch |
| GPU-16 | GPU-16 | Sync Triton calls have no `client_timeout` | LATENT | low-medium | S | open | PR-3 |
| GPU-17 | GPU-17 + API-12 | `model_residency._ready` never learns of an unload or restart | LATENT | low | S | open | PR-3 |
| GPU-18 | GPU-18 | EfficientNMS plugin deprecation not recorded | LATENT | low | S (docs) | open | PR-3 |
| GPU-19 | GPU-19 | OCR detector normalisation versus upstream | UNVERIFIED | ? | S | needs evidence | after evidence |
| FE-1 | FE-1 | `/export` makes a raw `fetch` of `/stats/dataset`; it never refreshes | LIVE | low | S | open | PR-6 |
| FE-2 | FE-2 | Diverse page fans out 30 `GET /crops/{id}`; one failure fails the page | LIVE | medium | S | open | PR-6 |
| FE-3 | FE-3 | Project binding in `load` with global hover preload | LATENT | low | S | open | PR-6 |
| FE-5 | FE-5 | Client-side sums in `TrainDatasetSection` | LATENT | low | S | open (conditional) | PR-6 |
| FE-6 | FE-6 | Pager buffers grow without bound | UNVERIFIED | ? | L | needs evidence | after measurement |
| FE-7 | FE-7 | SSE reconnect does not resync keymap/config | LATENT | low | S | open | PR-6 |
| T-1 | TESTS-1 | Three tests assert only inside a loop over a registry that may be empty | LATENT | low | S | open | PR-6 |
| T-2 | TESTS-2 | `test_gate_sim` open-set test never runs its asserts | LIVE | low-medium | S | open | PR-6 |
| T-3 | TESTS-3 | Compose render tests skip on the failure they exist to catch | LIVE | low-medium | S | open | PR-6 |
| T-4 | TESTS-4 | `einops` missing from `requirements-test.txt`, so a module is skipped | LIVE | low-medium | S | open | PR-6 |
| T-6 | TESTS-6 | Sleep-ordered FCFS budget test | LATENT | low | S | needs evidence (flake) | only if it flakes |
| T-7 | TESTS-7 | No test asserts project disk-usage bytes | LATENT | low | S | open | PR-6 |
| OPS-1 | API-6 "missed" | Compose lacks `--limit-max-requests-jitter`; the docs say it is set | LIVE | low-medium | S | open | PR-2 (R3 extra) |
| OPS-2 | API-14 | Prometheus multiprocess files of dead workers never cleaned | LATENT | low | S | needs evidence (measure) | Celery Phase 4 |
| CC-1 | API-4 | Fixed temp names: concurrent writers raise `FileNotFoundError` | LIVE | low-medium | S | open | PR-2 (Celery P1.0, revised) |
| CC-2 | API-2 | Class registry read-modify-write loses classes or duplicates ids | LIVE | medium | L | open (lock deleted on branch, race not fixed) | PR-2 / design spike (Celery P1.3, revised) |
| CC-3 | API-5 | Startup reconcile can mark a sibling's just-started job interrupted | LATENT | low | S | open | PR-2 (Celery P1.5) |
| CC-4 | API-6 + asyncio R5 | Request-launched jobs die when their worker recycles | LIVE | medium | L | open | Celery Phase 3 |
| CC-5 | asyncio R1 + "missed" | Blocking PIL and filesystem work inside `async def` handlers | LIVE | medium | S (handlers) / M (upload, dir ingest) | open | PR-4 (handlers); Celery Phase 3 (upload, dir ingest) |
| CC-6 | asyncio R2 + R8 | PIT slice producers orphaned on error (`pit_pages`, `residual_fetch`) | LIVE (error path) | low | S | open | PR-3 |
| CC-7 | asyncio R3 | `ingest_batch._one` never resolves the representative's future on error | LIVE (error path) | low | S | open | Celery Phase 6 (J7) |
| CC-8 | asyncio R4 | Detection worker stage tasks have no crash callback | LATENT | low | S | open | PR-3 |
| CC-9 | asyncio R7 | Spawn-everything gathers in job code | LATENT | low | S | open | Celery Phase 3 (per family) |
| CC-10 | asyncio R6 | VLM chunk fan-out unbounded in a request path | UNVERIFIED | ? | S-M | needs evidence | Celery Phase 6 |
| CC-11 | module-level `asyncio.Lock` sites | Four module-level locks that excluded nothing across processes | LIVE | medium | S | **fixed (branch)**, one documented throttle left | merge branch; last one removed in Celery Phase 3 |
| SEC-1 | security | Harden model upload validation (details in the private notes) | - | - | S | **fixed (branch)** | merge branch (R3) |
| SEC-2 | security | Harden model lifecycle route input handling (details in the private notes) | - | - | S | **fixed (branch)** | merge branch (R3) |
| SEC-3 | security | Harden persisted clustering state (details in the private notes) | - | - | S | open | later |
| SEC-4 | security | Reduce the privileges of an opt-in monitoring service (details in the private notes) | - | - | S | open | later (docs first) |
| SEC-5 | security | Pin an unpinned build input and review per-service container hardening (details in the private notes) | - | - | S | open | later |
| SEC-6 | security | Widen the CLI secret redaction key list (details in the private notes) | - | - | S | open | later |

These items are also fixed on the branch. They were outside the audit's scope (the
brief excluded them), and they are listed so nobody redoes them:
- #437: region head layout (`4a15558c`, `6de660bc`).
- #442: VLM registry refresh on a by-name miss, now coalesced, with a 503 when it
  cannot be read (`04c7f3f5`, `f7277e57`). This is the same class of problem as BA-3:
  a per-process snapshot read across workers.
- #443: cluster retrain held during ingest.
- #445: public COCO sample images.
- #439: indexes are created sequentially again (revert `45f06985`).
- The assist-loop admission and the validator bypass sets.

---

## 2. Severity summary

**Open, non-security tasks: 52.** Of these, 46 are LIVE or LATENT findings and 6 are
LIVE test-only gaps.

| Area | high | medium-high | medium | low-medium | low | total open | needs evidence | needs scoping | fixed (branch) |
|---|---|---|---|---|---|---|---|---|---|
| backend-api (BA) | 0 | 0 | 1 | 2 | 5 | 8 | 0 | 1 (BA-6 also counted open) | 0 |
| opensearch (OS) | 0 | 1 | 0 | 2 | 5 | 8 | 1 (OS-8) | 0 | 0 |
| gpu/inference (GPU) | 2 | 0 | 5 | 3 | 6 | 16 | 1 (GPU-19) | 1 (GPU-6) | 1 (GPU-15) |
| frontend (FE) | 0 | 0 | 1 | 0 | 4 | 5 | 1 (FE-6) | 0 | 0 |
| tests (T) | 0 | 0 | 0 | 3 | 2 | 5 | 1 (T-6) | 0 | 0 |
| ops/compose (OPS) | 0 | 0 | 0 | 1 | 0 | 1 | 1 (OPS-2) | 0 | 0 |
| concurrency (CC) | 0 | 0 | 3 | 1 | 5 | 9 | 1 (CC-10) | 0 | 1 (CC-11) |
| **total** | **2** | **1** | **10** | **12** | **27** | **52** | **6** | **1** | **2** |

- **Security:** 6 items, tracked privately. SEC-1 and SEC-2 are fixed on the branch;
  SEC-3 to SEC-6 are open. Their severities are recorded only in the private notes.
- **Refuted (Appendix C):** 14 claims.
- **Cut as churn (Appendix D):** 37 proposals.

**Severities are the adversarial reviewers' corrected values,** not the audit's
originals. The audit's two "high" API findings (API-1, API-2) were reviewed down to
medium. GPU-9 was reviewed up to medium.

---

## 3. Sequencing

### 3a. Before 0.7.0 release acceptance R3 (LIVE, high or critical, and cheap)

**Required:**
1. **Merge `fix/411-r2-round`.** It carries SEC-1, SEC-2, GPU-15, CC-11, #437 and
   #442. Nothing in this plan should start from `main` while that branch is unmerged.
   Base new work on the integration branch (`git rev-parse --abbrev-ref @{u}`).
2. **GPU-1:** fix the `/ocr` box displacement. High, S.
3. **GPU-3a:** make `/ocr` decode once in the oriented frame. High, S. GPU-13 rides
   along because it touches the same decode function.
4. **OS-2:** the holdout freeze must fail closed on bulk item errors. Medium-high, S,
   and it protects the lock rule's frozen test set.

**Cheap extras, only if the R3 window allows** (all S and LIVE, but below high):
- **GPU-2:** pyclipper. It needs a Triton image rebuild. Pair it with GPU-1 so the
  OCR before/after comparison runs once.
- **CC-1:** unique temp files. It removes reproduced 500s on status polls, and the 500
  path of CC-2.
- **BA-7:** `review_dismiss` returns 503, not 404, on an outage.
- **OS-1:** `op_type='create'` on the first write.
- **OPS-1:** make the compose recycle jitter and `docs/PERFORMANCE.md` agree.

### 3b. PR grouping

The rule followed: few PRs, targeted tests per item, one full offline suite per PR,
and never squash.

| PR | Contents | Depends on | Release |
|---|---|---|---|
| (merge) | `fix/411-r2-round` | owner yes | R3 |
| **PR-1 `fix/ocr-geometry-and-holdout`** | GPU-1, GPU-3a, GPU-13, OS-2 (+ GPU-2 if the image is rebuilt) | merge | **R3** |
| **PR-2 `fix/cross-process-phase1`** = Celery Phase 1 batch | CC-1, CC-3, OS-1, OS-3, OPS-1, CC-2 interim steps 1-2 (CC-2 full fix after J-REG) | merge | after R3 (R3 extras may be cherry-picked) |
| PR-3 `fix/inference-and-search-hygiene` | GPU-4, 5, 8, 9, 10, 14, 16, 17, 18; OS-5, 6, 7, 9, 10; CC-6, CC-8 | merge | 0.7.x |
| PR-4 `fix/api-correctness` | BA-1, 2, 3, 4, 5, 7, 8; CC-5 (thumbnail and render handlers) | merge | 0.7.x |
| PR-5 `perf/ocr-batching-and-decode` | GPU-7, GPU-3b | PR-1 | 0.7.x / 0.8 |
| PR-6 `fix/frontend-and-test-integrity` | FE-1, 2, 3, 5, 7; T-1, 2, 3, 4, 7 | merge | 0.7.x |
| Celery Phase 3 | CC-4, CC-5 (upload extract, directory ingest), CC-9; delete `open_vocab_ingest._PASS_LOCK` (the last CC-11 site) | Celery Phase 2 | later |
| Celery Phase 4 | OPS-2 (after its measurement), delete the startup reconcile hardened in CC-3 | Phase 3 | later |
| Celery Phase 6 | CC-7 (J7), CC-10 (after evidence) | Phase 3 | later |
| Celery Phase 7 | prerequisites: BA-1, BA-3, BA-4, BA-8, GPU-17 | PR-4, PR-3 | later |
| Later / owner | GPU-6 (decision), BA-6 (one router per PR), GPU-12 (with #407), SEC-3..6, needs-evidence items | - | later |

---

## 4. Tasks by area

The template for each task:
- **Class · Severity · Size · Reachable by**
- **Bug**
- **Evidence**: the command and a summary of its output, from the audit or verdict run
  at `5b95591f`. Scratch scripts under `<scratch repro>*` may be gone; the red test recreates
  them.
- **Root cause**
- **Fix**: numbered steps, following the docs.
- **Traps**
- **Tests**: a red test and its paired control.
- **Done when**

### 4.1 backend-api

#### BA-1 (API-1): Class-name cache stale in other workers after a promote
- **Class** LIVE · **Severity** medium · **Size** S
- **Reachable by:** any detection, analyze or ingest response after
  `POST /train/promote`, with 32 uvicorn workers in the default config.
- **Bug.** `src/utils/class_names.py` caches `labels.txt` as `{id: name}` per model
  name for the life of the process. `invalidate_class_names()` runs only in the worker
  that handled the promote, so the other 31 serve the old vocabulary until they
  recycle. If the file was absent at first read, the cached `{}` sticks the same way.
- **Evidence.** `.venv/bin/python -I <scratch repro>` (two real processes
  sharing a model repo). Worker B prints `{0:'cat',1:'dog'}` after worker A promoted
  `{0:'truck',1:'bus'}`. The verdict re-ran it with the same result.
- **Root cause.** A process-local cache keyed only by model name, invalidated by a call
  that reaches only one process.
- **Fix.** Make the cache idempotent; it needs no shared state.
  1. Key the cache entry by `(model_name, st_mtime_ns, st_size)` of `labels.txt`,
     taken with one `stat()` per lookup. `ClassRegistry.load()` already uses mtime the
     same way.
  2. Do not cache a missing-file result.
  3. Delete `invalidate_class_names` and its two calls in
     `services/training/triton_promote.py`, unless a test needs a reset hook. Find the
     sites with `rg -n invalidate_class_names src tests`.
- **Traps.** Promote rewrites `labels.txt` in place. Stat the file, not the directory
  (a directory's mtime does not change on an in-place write). Use `st_mtime_ns` plus
  the size, because coarse mtime resolution can miss a fast rewrite.
- **Tests.**
  - Red: write labels v1, call `get_class_names`, rewrite as v2 with a new mtime
    without calling invalidate, and assert v2. This fails today.
  - Control: with the file unchanged, a second lookup does not re-read it (count the
    `_read_labels_file` calls).
- **Done when:** the two-process script prints v2 in worker B with no invalidation
  call.

#### BA-2 (API-3): Timing middleware injects keys into `/curation` bodies
- **Class** LIVE · **Severity** low-medium · **Size** S
- **Reachable by:** JSON responses of about 41 `/curation/projects/{p}/...` operations
  under `ingest*`, `ingest_profiles*`, `search/*` and `detections/summary`.
- **Bug.** `src/app/factory.py` (`rg -n "path in request.url.path" src/app/factory.py`)
  matches inference paths by substring. It therefore also buffers, re-parses and
  rewrites curation bodies, adding `total_time_ms` and `request_id`, which the curation
  models do not declare.
- **Evidence.** `TestClient(app).get('/curation/projects/x/ingest/nonexistent')`
  returns a body that contains `total_time_ms` and `request_id`. The control
  `/curation/projects/x/crops` does not. The OpenAPI filter counts 41 matching
  operations.
- **Root cause.** A substring test where a prefix test was meant.
- **Fix.**
  1. Skip the body rewrite when `request.url.path` starts with `/curation`. Curation
     has its own serializer and no `/v1` twin.
  2. Keep the keys on the core routes; the benchmark docs reference `total_time_ms`.
  - Do **not** convert the middlewares to pure ASGI in this change (Appendix D).
- **Traps.** The `/v1` aliases of the core routes must keep the keys. `rg
  total_time_ms frontend/src contracts` should show no curation consumer.
- **Tests.**
  - Red: `GET /curation/projects/x/ingest/policy` (OpenSearch stubbed) has no
    `total_time_ms`.
  - Control: `POST /detect` still has it.
- **Done when:** no `/curation/**` response gains an undeclared key.

#### BA-3 (API-7): Per-process caches cache errors and never evict
- **Class** LIVE · **Severity** low-medium · **Size** S-M
- **Reachable by:** settings reads during an OpenSearch error; long-lived workers
  serving many distinct filter queries.
- **Bug.**
  - `clients/curation_opensearch/settings_doc.py` caches `{}` for 5 s on any error,
    which reads as "no overrides" (fail-open). Its docstring promises read-after-write,
    which is false across workers.
  - `services/curation/diverse_order.py` and `clustering/outliers.py` keep caches
    keyed by query JSON with no eviction. The verdict confirmed they only get and set,
    never evict.
- **Evidence.** Code reading at the cited lines. `rg -n "_ORDER_CACHE|_CACHE\b"`
  shows only get and set. The mechanism is the same as BA-1, which was reproduced.
- **Root cause.** Process-local caches carry correctness duties they cannot keep across
  workers.
- **Fix.**
  1. `settings_doc`: on an error, raise (the caller's 503 path) instead of caching
     `{}`. Correct the docstring to "5 s per-process TTL; another worker may serve the
     previous value for up to 5 s".
  2. `diverse_order` and `outliers`: replace the dicts with
     `cachetools.TTLCache(maxsize=N, ttl=600)`. `cachetools` is maintained and MIT
     licensed, but it is **not** a declared dependency today (`image_serving.py`
     try-imports it). Add it to `requirements.txt` deliberately, record it in
     ATTRIBUTION.md, and delete the try-import fallback in the same change.
  3. Leave the capacity cache, `regions_fp` and `field_coverage` alone (verdict).
- **Traps.** The count-based invalidation stays. Do not add a `max(updated_at)` query
  unless staleness is shown to matter.
- **Tests.**
  - Red: a settings fetch that raises must surface as an error and must not return `{}`
    on the next call within 5 s.
  - Red: insert `maxsize+1` distinct keys into the order cache, and the oldest is gone.
  - Control: a successful settings fetch is cached for the TTL.
- **Done when:** no cache stores an error result, and no cache keyed by caller JSON is
  unbounded.

#### BA-4 (API-8): `eval_datasets` cache key narrower than its inputs
- **Class** LIVE · **Severity** low · **Size** S
- **Reachable by:** bake-off dataset listing after a class rename or an edit to
  `data.yaml`.
- **Bug.** `src/services/curation/eval_datasets.py` keys its cache on label-file and
  manifest mtimes, but the cached record also bakes in registry class names, the
  `data.yaml` names and image counts.
- **Evidence.** `<scratch repro>`: after renaming cat to feline in the
  registry, adding one image and editing `data.yaml`, the output is still
  `[('cat',7,'cat')] n_images 1 same object True`. The verdict re-ran it.
- **Fix.**
  1. Resolve the registry names at `to_wire` time, uncached.
  2. Add the mtimes of `data.yaml` and the `images/test` directory to `_cache_key`.
  - Eviction is not needed: the cache is bounded by the number of datasets.
- **Tests.**
  - Red: the script's scenario asserts `feline` and `n_images == 2`.
  - Control: unchanged inputs return the same object.
- **Done when:** the scenario prints feline and 2.

#### BA-5 (API-9): The 500 body echoes the exception class name
- **Class** LIVE · **Severity** low · **Size** S
- **Reachable by:** any unhandled 500.
- **Bug.** `src/app/factory.py` and `src/core/error_handlers.py` put
  `type(exc).__name__` in client bodies (`rg -n error_type src`). FastAPI's
  error-handling docs advise against leaking internals. The missing CORS header on a
  500 is documented Starlette behaviour: `ServerErrorMiddleware` is outermost.
- **Evidence.** A route raising `KeyError` returns
  `{"detail":"Internal server error","request_id":...,"error_type":"KeyError"}` with no
  ACAO header.
- **Fix.**
  1. Remove `error_type` from response bodies. Keep it in the log line.
  2. Before removing it, run `rg -n error_type tests frontend/src contracts`, and
     update any reader in the same commit.
  - Leave the CORS behaviour of the 500 alone (see Appendix D).
- **Tests.**
  - Red: a raising route's 500 body has no `error_type`.
  - Control: the body still has `request_id`.
- **Done when:** `rg -n "'error_type'" src` shows log calls only.

#### BA-6 (API-10): Operations without a response model (needs scoping)
- **Class** LIVE · **Severity** low · **Size** L
- **Reachable by:** contract consumers.
- **Bug.** `app.openapi()` lists 426 operations. 71 of them have an empty or free-form
  2xx JSON schema, and 62 of those are under `/curation`.
- **Evidence.** Running the `.venv/bin/python` OpenAPI scan gives
  `total ops 426, untyped 2xx json 71`. The list was in `<scratch repro>`.
- **Fix.**
  1. Scope the work to routes Cropwright consumes first.
  2. Per router, model today's exact shape in `src/schemas/curation/`, set
     `response_model=`, run `make contracts`, and commit `contracts/` in the same
     commit.
  - One router per PR. No single "baseline to 0" pass.
- **Traps.** Field names are the Cropwright contract. Model the shape as it is first,
  and tighten later.
- **Tests.**
  - Red: a per-router test that its operations have non-empty schemas.
  - Control: an already-typed route.
- **Done when (per PR):** that router's operations are typed and `make contracts-check`
  passes.

#### BA-7 (API-11): `review_dismiss` turns every OpenSearch failure into a 404
- **Class** LIVE · **Severity** low · **Size** S
- **Reachable by:** `POST /curation/projects/{p}/crops/{id}/review_dismiss` while
  OpenSearch is degraded.
- **Bug.** `src/routers/curation/crops_exclude.py` has
  `except Exception: raise HTTPException(404)`. Sibling code in the same file maps
  failures to 503.
- **Fix.**
  1. Catch `opensearchpy.NotFoundError` and return 404.
  2. Let every other error reach the registered 503 handler.
- **Tests.**
  - Red: an update raising `ConnectionError` must give 503.
  - Control: `NotFoundError` gives 404.
- **Done when:** both tests pass.

#### BA-8 (API-13): Probe concurrency cap is per process
- **Class** LATENT · **Severity** low · **Size** S
- **Bug.** `vlm_endpoint_probe.MAX_CONCURRENT_PROBES = 2` is per process, so the real
  limit is 64 across 32 workers. Users see the error as a quota.
- **Fix.** Docs only. State "per API worker process" in the constant's comment and in
  the user-facing error text. A cluster-wide lease is over-engineering (Appendix D).
- **Tests.** Not applicable (text). If the error text is asserted somewhere, update
  that assertion.
- **Done when:** both texts say the cap is per process.

#### SEC-1, SEC-2: fixed on the branch
- See the index. Verify both by the branch's own tests listed in the private notes,
  during the merge's full suite.

#### SEC-3, SEC-6
- One-line tasks only; the index rows cover them. Details are in
  `artifacts_local/audit/security.md` and `security.verdicts.md`.

### 4.2 opensearch

#### OS-1: First create of a config or keymap doc has no `op_type='create'`
- **Class** LIVE (narrow window) · **Severity** low-medium · **Size** S
- **Reachable by:** two concurrent first `save_config` or `activate` calls for an axis,
  or a first keymap save.
- **Bug.** `src/services/config_store/index.py` (`save_config`, `activate`, and the
  immutable `<kind>:<name>@<rev>` copy) and `src/services/curation/keymap.py` leave
  `seq_no=None` when the GET finds no doc, then call `client.index(...)` with neither
  `if_seq_no` nor `op_type`. Two first writers both succeed, and the
  `expected_revision` guard is bypassed. `projects/registry.py`, `bootstrap.py` and
  `policy_doc_store.write_policy` already do this correctly.
- **Docs.** OpenSearch index-document API: `op_type=create` and
  `if_seq_no`/`if_primary_term` give a 409 on conflict.
- **Evidence.** A fake client whose `get` raises `NotFoundError`, then
  `PYTHONPATH=. .venv/bin/python <scratch repro>`, printed the index kwargs
  `{'id': 'activation:vlm'}`, with no OCC kwarg.
- **Fix.**
  1. In each absent-doc branch, set `index_kwargs['op_type'] = 'create'`.
  2. Use `op_type='create'` for the revision copy as well; revision numbers are never
     reused.
  3. The existing `ConflictError` handlers already map to
     `RevisionConflictError`/`ActiveConflictError` and the keymap 409. Confirm each one
     covers the create path.
- **Traps.** Rollback also calls `activate`; a create conflict must map to the same
  409. Do not change the intentional last-write-wins singletons (`vlm_endpoints.py`
  probe record and local desired state, and the runtime heartbeat doc).
- **Tests** (`tests/test_config_store_first_create.py`, as named in Celery P1.4).
  - Red: with a stateful fake that raises `ConflictError` on create of an existing id,
    two sequential `save_config(expected_revision=None)` calls give one success and one
    `RevisionConflictError`. Today both succeed.
  - Control: an update with a stale `seq_no` still conflicts.
- **Done when:** no `client.index(` in `config_store/` or `keymap.py` runs without
  either `if_seq_no` or `op_type`.

#### OS-2: Holdout freeze ignores bulk item errors and persists the record (R3)
- **Class** LIVE · **Severity** medium-high · **Size** S
- **Reachable by:** `POST /curation/projects/{p}/test_holdout/freeze`, when any item
  fails. Examples: a crop deleted between selection and flagging (item 404), or a 429
  rejection.
- **Bug.** `src/routers/curation/holdout_freeze.py`
  (`rg -n "opensearch.bulk" src/routers/curation/holdout_freeze.py`) discards the bulk
  response, then persists the freeze record and its sha, and returns
  `n_frozen=len(chosen_ids)`. An unflagged item that the record claims is frozen can
  then leak into training. That breaks the lock rule.
- **Docs.** The OpenSearch bulk API answers 200 with `errors: true` and a per-item
  `error`/`status`.
- **Evidence.** The rg line above shows the return value unused. The verdict confirmed
  it; `crops_write.py` checks `items[].error` in the same file family. Still present
  on the branch.
- **Root cause.** The bulk call is treated like a single-document write.
- **Fix.**
  1. Promote the existing private `_log_bulk_write_errors`
     (`services/curation/clustering/cluster_write_guard.py`) into a small public
     helper in `src/clients/opensearch/bulk.py`. It returns the failed ids and
     statuses. OS-3 reuses it.
  2. In the freeze route, if any item failed, do **not** persist the record. Return a
     5xx naming the failed ids (`server_error(...)`, the existing helper). A rerun is
     idempotent: the update sets `test_holdout=true`.
  3. Do not skip and shrink the set. Keep `refresh=True`, which is one admin call.
- **Traps.** The sha must cover exactly the ids that are flagged. Failing closed means
  it is never computed over a partial set.
- **Tests.**
  - Red: a fake bulk returning
    `{'errors': True, 'items': [{'update': {'status': 404, 'error': {...}}}]}`. Today
    the route returns 200 and writes the record. After the fix it is a 5xx and there
    is no record.
  - Control: all items 200 persists the record.
- **Done when:** both tests pass and no freeze record exists after a partial failure.

#### OS-3: Three more bulk writers drop item errors
- **Class** LIVE · **Severity** low (derived fields a rerun recomputes) · **Size** S
- **Bug.** `item_scores/job.py` (score fields), `embedding_viz.py` (counts
  `len(chunk_ids)` regardless of outcome) and the early-return flush in
  `probe_predictions.py` discard bulk item errors, so jobs report "written N" when they
  were not.
- **Evidence.** The verdict's AST scan over every `.bulk(` call, looking for
  `errors`/`items` within 15 lines, leaves exactly these three plus holdout_freeze.
- **Fix.** Use the OS-2 helper at the three sites. A job with failed items reports the
  failed count, and ends `failed` when nothing was written.
- **Tests.**
  - Red, per site: a fake bulk with `errors: true` makes the job's reported count or
    status reflect the failure.
  - Control: all-OK bulk.
- **Done when:** the three sites use the helper. A one-off test asserts that no bare
  awaited `bulk(` result is discarded in `src/services/curation`. This is a test, not a
  ratchet.

#### OS-5: Unbounded `page` reaches OpenSearch on two list routes
- **Class** LIVE · **Severity** low · **Size** S
- **Reachable by:** `GET .../audit/queue?page=100000` and the `classes_crops` listing.
- **Bug.** `from=(page-1)*page_size` goes past `index.max_result_window`. OpenSearch
  answers 400, and the route maps every exception to `503 opensearch unavailable`.
  `_common.guard_page_depth` exists but is not called here.
- **Fix.** Call `guard_page_depth` in `audit.py`'s queue handler and in
  `classes_crops`. The search routes are fine as designed: an empty page past `k`.
- **Tests.**
  - Red: `page=100000` gives 422. Today it is 503.
  - Control: `page=1` gives 200.
- **Done when:** both tests pass.

#### OS-6: Legacy visual-search kNN filter is a post-filter
- **Class** LIVE · **Severity** medium-low · **Size** S
- **Reachable by:** `/search/objects?class_filter=` and the cluster-routed search.
- **Bug.** `src/clients/opensearch/search.py` builds
  `bool{must:[knn{k}], filter:[terms ...]}`. Per the docs, that filters after the ANN
  top-k, so a restrictive filter returns fewer than `k` hits. The docstring's
  `cluster_ids` speedup does not happen.
- **Docs.** OpenSearch "Filtering in k-NN search": efficient filtering puts the filter
  inside the `knn` clause.
- **Evidence.** Code reading. The verdict checked that the indexes use `engine: faiss`
  HNSW on OpenSearch 3.6.0, so efficient filtering is supported. No test pins the bool
  shape.
- **Fix.**
  1. Move the filters into `knn[field]['filter']`. Keep `min_score`.
  2. Correct the docstring.
- **Tests.**
  - Red: a unit test on the built body asserts `'filter' in body['query']['knn'][field]`
    and that there is no `bool.must[knn]`.
  - Control: a query with no filter is unchanged.
  - Live, optional: on the live harness, a restrictive class filter returns `k` hits.
- **Done when:** the body test passes.

#### OS-7: Blocking `update_by_query` in `backfill_embedded_state`
- **Class** LATENT · **Severity** low · **Size** S
- **Reachable by:** `legacy_repair` on a large legacy project.
- **Bug.** `src/domain/embedding_state.py` runs `update_by_query` with
  `wait_for_completion=True`. The repo's own `run_update_by_query_polled`
  (`services/curation/clustering/id_normalize.py`) documents that this fails on the
  items index. The domain layer cannot import services.
- **Fix.**
  1. Move `run_update_by_query_polled` to `src/clients/` (for example
     `src/clients/opensearch/tasks.py`).
  2. Update its two existing callers.
  3. Use it here.
  - Nothing else; the extras are optional (Appendix D).
- **Tests.**
  - Red: a fake client records `wait_for_completion=False` and the task poll.
  - Control: the existing `id_normalize` tests still pass.
- **Done when:** no blocking `update_by_query` remains on the items index.

#### OS-8: `delete_by_query` in `image_admin` (needs evidence)
- **Class** UNVERIFIED.
- **Background.** The reviewer could not reproduce "the route reports success". With
  the default `conflicts=abort`, opensearch-py raises `ConflictError` on a 409, so a
  conflict is not silent. The only silent case would be a non-empty `failures` array
  on a 200.
- **Evidence task.**
  1. On the live harness (`-p` name, GPUs 0/2), run the legacy image `DELETE` while a
     concurrent update touches a matching doc.
  2. Record the status and the response's `failures` and `version_conflicts`.
- **Act only if:** a 200 with non-empty `failures` is observed. In that case, check
  `failures` and raise.

#### OS-9: Scroll context not cleared on the error path
- **Class** LATENT · **Severity** low · **Size** S
- **Bug.** `classes_merge._scroll_merge_ids` and `clients/opensearch/clusters.py` call
  `clear_scroll` only after a clean loop.
- **Docs.** OpenSearch scroll API: "Close the search context when you're done".
- **Fix.** Wrap the loop in `try/finally: clear_scroll(...)` at both sites.
- **Tests.**
  - Red: a fake scroll that raises on page 2 must still see `clear_scroll` called.
  - Control: a clean run calls it once.
- **Done when:** both tests pass.

#### OS-10: Unbounded `offset` in the cluster list aggregation
- **Class** LIVE · **Severity** low · **Size** S
- **Bug.** `routers/curation/clusters.py` sets terms `size: offset + max_clusters` with
  `offset` declared `ge=0` and no upper bound. A large offset exceeds
  `search.max_buckets`.
- **Fix.** Add `le=` to `offset`, sized from `search.max_buckets` minus the
  `max_clusters` maximum. Run `make contracts`.
- **Tests.**
  - Red: an offset above the bound gives 422.
  - Control: `offset=0` gives 200.
- **Done when:** both tests pass and the contracts are regenerated.

### 4.3 gpu/inference

#### GPU-1: `/ocr` maps boxes with the padded size (R3)
- **Class** LIVE · **Severity** high · **Size** S
- **Reachable by:** `POST /ocr*`, `/analyze` with OCR, and generic ingest OCR. Any
  image whose resized side is not a multiple of 32 is affected, which is most images.
- **Bug.** `src/clients/triton_ocr.py` (`rg -n "pad_h|pad_w" src/clients/triton_ocr.py`)
  zero-pads the detector input to a multiple of 32. The BLS
  (`models/ocr_pipeline/1/ocr_postprocess.py`) then maps back with
  `src / pred.shape`, the **padded** shape, so boxes shift up and left and the
  recognition crops are cut from the wrong place. The cascade caller
  (`cascade_detect/ocr_recognizer.py`) stretches each axis to a multiple of 32 and is
  correct.
- **Docs.** PaddleOCR `DetResizeForTest` resizes without padding.
  `db_postprocess.boxes_from_bitmap` scales by `dest / bitmap` size.
- **Evidence.** With `DBPostProcess` on a 448x640 map for a 427x640 image, the text
  rows 400-420 come out at y 376.2-406.3 (centre 391 against the true 410). The
  verdict reproduced it exactly.
- **Root cause.** The map is in the padded frame; the scale assumes the content frame.
- **Fix.**
  1. In `infer_ocr`, remove the pad block. Resize each axis to its multiple-of-32
     target, as `ocr_det_input_size` and PaddleOCR do.
  2. Make the client and the cascade share one `ocr_det_input_size`; delete the
     duplicate.
  3. Do not change the BLS.
- **Traps.**
  - `/ocr` boxes and the recognised text will change (intended).
  - Compare once together with GPU-2 and GPU-3a on public COCO text images. The
    existing tests `tests/test_ocr_uint8_inputs.py` and
    `tests/test_ocr_pipeline_rec_width.py` do not assert positions.
- **Tests.**
  - Red: the `DBPostProcess` snippet, fed the map shape that `infer_ocr` produces,
    asserts the box centre is within 1 px of 410. It fails today.
  - Control: a 640x480 image (no padding) is exact both before and after.
- **Done when:** the centre error is at most 1 px for padded and unpadded sizes, and
  the two OCR test files are green.

#### GPU-2: pyclipper missing from the Triton image; silent fallback
- **Class** LIVE · **Severity** medium · **Size** S, plus an image rebuild
- **Reachable by:** every OCR request.
- **Bug.** `ocr_postprocess.py` imports `pyclipper` and on `ImportError` falls back to
  a centre scale. That is not DB's unclip, which offsets each side by
  `d = area * ratio / perimeter`. Nothing installs pyclipper.
- **Evidence.**
  - `docker run --rm --entrypoint python3 openprocessor-local/openprocessor-triton:0.7.0-rc5 -c "import importlib.util as u; print(bool(u.find_spec('pyclipper')))"`
    printed `False`. The verdict reproduced it.
  - On a 200x20 line at ratio 1.5, DB gives 227x47 and the fallback gives 300x30.
- **Fix.**
  1. Add a pinned `pyclipper` to `Dockerfile.triton` and to the hardened Triton
     Dockerfile. Check the licence against CONTRIBUTING's dependency rules (Boost
     licence for the C++ library) and record it in ATTRIBUTION.md.
  2. Make the import top-level and delete the fallback, so model `initialize` fails
     loudly.
  3. Add `pyclipper` to `requirements-test.txt` so the offline tests import the BLS
     module.
- **Traps.**
  - `tests/test_ocr_pipeline_rec_width.py` pops and re-imports `ocr_postprocess`, so it
    breaks without pyclipper in the venv. Step 3 covers this; check it before landing.
  - The image rebuild must pass the image-vs-git audit.
- **Tests.**
  - Red: `_unclip` on a 200x20 box returns about 227x47.
  - Control: a square box, where both methods agree.
- **Done when:** `find_spec('pyclipper')` is True in the image and no fallback code
  remains.

#### GPU-3a: `/ocr` frame mismatch (R3)
- **Class** LIVE · **Severity** high · **Size** S
- **Reachable by:** phone photos with EXIF orientation 5-8 sent to `/ocr`.
- **Bug.** `src/services/ocr_service.py` decodes with `IMREAD_COLOR` (oriented) and
  reports that `image_size`. `triton_ocr` then decodes the same bytes with
  `IMREAD_UNCHANGED`, which ignores EXIF. OCR runs on the sideways frame, and the
  normalised boxes are in a different frame from the reported size.
- **Docs.** OpenCV `imgcodecs.hpp`: "IMREAD_UNCHANGED ... Ignore EXIF orientation".
- **Evidence.** For a 200x100 JPEG with EXIF tag 6, PIL reports (200,100), COLOR gives
  (100,200) and UNCHANGED gives (200,100). The verdict reproduced it, including a
  4-channel PNG.
- **Fix** (the verdict's smaller one):
  1. Have `ocr_service` pass its already-decoded, oriented array to `infer_ocr`, so
     there is one decode. Keep the bytes path only for callers that do not pre-decode.
  2. For alpha images, decode with Pillow, apply `ImageOps.exif_transpose`, then
     composite onto white with `Image.alpha_composite`. This also fixes GPU-13.
- **Traps.** `IMREAD_COLOR` honours EXIF but drops alpha, so transparent pixels would
  come out black. Do not just swap the imread flag.
- **Tests.**
  - Red: an orientation-6 JPEG through `ocr_service` must give boxes in the same (w,h)
    frame as the reported `image_size`.
  - Red: an RGBA PNG with transparent text background composites white.
  - Control: orientation 1 is identical before and after.
- **Done when:** both red tests pass and the control is unchanged.

#### GPU-3b: Unify the remaining decoders on one oriented decode
- **Class** LIVE · **Severity** medium · **Size** M
- **Reachable by:** rotated photos on `/embed*`, `/analyze` and legacy `visual_search`
  ingest.
- **Bug.**
  - `analyze_pipeline`, `infer_yolo_clip_cpu`, `preprocess_for_mobileclip`,
    `services/embedding.py` and `routers/embed.py` use PIL without `exif_transpose`.
  - Face detection uses the oriented `IMREAD_COLOR`. `/analyze` can therefore return
    object boxes and face boxes in different frames.
  - `/embed` vectors of a rotated photo differ from its ingested vectors.
- **Fix.**
  1. Move `services/curation/image_decode.decode_image` (proven equal to
     `exif_transpose` for all 8 tags) to `src/utils/`.
  2. Adapt the callers to its `DecodedImage` and `ImageDecodeError`.
  3. Have `get_jpeg_dimensions_fast` return the oriented size, or document that it
     does not.
- **Traps.** This is a behaviour change for rotated inputs (no shims, by owner rule).
  Record it in CHANGELOG.
- **Tests.**
  - Red: for EXIF tags 1-8, `/embed` preprocessing, `/analyze` objects and faces all
    agree on (w,h).
  - Control: tag 1 output is unchanged.
- **Done when:** one decode helper serves all these paths.

#### GPU-4: `gpu_decode_ready` caches a transient failure forever
- **Class** LIVE · **Severity** low-medium · **Size** S
- **Bug.** `src/services/curation/ingest_gpu_decode.py`
  (`rg -n "_READY\[key\]" src/services/curation/ingest_gpu_decode.py`) stores `False`
  after any exception. One failed Triton call during a restart leaves that worker on
  the CPU path until it recycles.
- **Fix.** Write `_READY[key]` only when a verdict is reached inside the `try`. On an
  exception, return False without caching. That is all (the extras are in Appendix D).
- **Tests.**
  - Red: the first call raises and the second succeeds; the second must return True.
  - Control: a success stays cached (one probe).
- **Done when:** both tests pass.

#### GPU-5: Full bitmap allocated before the dimension check
- **Class** LIVE · **Severity** low-medium · **Size** S
- **Reachable by:** `/detect`, `/ocr` and the model test route, all through
  `utils/image_processing.decode_image`. Uploads are capped at 50 MB.
- **Bug.** `cv2.imdecode` runs before `validate_image` checks the 16384 px limit. A
  6 MB JPEG of 20000x20000 costs about 2.3 GB RSS before it is rejected.
- **Docs.** Pillow `Image.open` is lazy, so `.size` reads only the header.
- **Fix.** Read the header size first (`Image.open(BytesIO(b)).size`, or the existing
  `megapixels()` helper) and reject anything over `MAX_IMAGE_DIMENSION` before
  `imdecode`. Keep `validate_image` as the post-check.
- **Tests.**
  - Red: the 20000x20000 JPEG raises before the decode. Assert that `cv2.imdecode` is
    not called, by monkeypatching it.
  - Control: 4000x3000 decodes.
- **Done when:** both tests pass.

#### GPU-6: `/detect` letterbox never upscales (needs scoping)
- **Class** LIVE divergence; severity is the owner's call.
- **Background.** `/detect` uses `scaleup=False`, as Ultralytics validation does.
  Ingest upscales. The same 320x240 object is 160x120 in network space on `/detect`
  and 318x238 in ingest.
- **Decision needed:**
  - Is `/detect` meant to match predict (`scaleup=True`)?
  - If yes, measure recall before and after on a COCO val subset first, and sequence
    the change after #40/#407.
  - Do not touch the documented PIL versus cv2 interpolation split.

#### GPU-7: OCR recognition makes one BLS call per crop
- **Class** LIVE (performance) · **Severity** medium · **Size** M
- **Bug.** `models/ocr_pipeline/1/model.py:_call_recognition` issues one blocking
  `exec()` per crop, up to `MAX_TEXTS=128`. `paddleocr_rec_trt` accepts batch 64.
- **Docs.** The python_backend README: `async_exec` allows "multiple inflight
  inference requests", and the dynamic batcher then batches them.
- **Fix.**
  1. Prefer `async_exec` for all crops, which needs `async def execute`. The Triton
     dynamic batcher does the batching, and each crop keeps its exact width.
  2. If the pinned python backend does not support it, bucket the crops by target
     width and send one request per bucket.
  3. Keep the per-crop `-1.0` failure sentinel.
  - This is TaskGroup-free by design: the concurrency is inside Triton, not in an API
    handler.
- **Traps.** Bucketing with padding changes the CTC logits slightly. Compare the text
  on the pinned public image set before and after.
- **Tests.**
  - Red: count the `exec` calls for 40 crops (40 today, at most the bucket count
    after).
  - Control: a single crop gives the same output as before.
- **Done when:** the call count equals the bucket count (or one async gather) and the
  text matches on the pinned set.

#### GPU-8: Retry classifier branch is dead; the pool retries deterministic errors
- **Class** LIVE · **Severity** low · **Size** S
- **Bug.**
  - `InferenceServerException` has `status()`, not `code()`, so the gRPC-code branch
    of `utils/retry.is_retryable_error` never runs, and RESOURCE_EXHAUSTED and ABORTED
    are never retried.
  - `AsyncTritonPool.infer` retries every exception, including INVALID_ARGUMENT.
- **Evidence.** Today the classifier gives UNAVAILABLE True, RESOURCE_EXHAUSTED False,
  ABORTED False, INVALID_ARGUMENT False. A fake pool made 3 calls for one
  INVALID_ARGUMENT.
- **Fix.**
  1. One predicate on `exc.status()`, used by both `retry.py` and the pool.
  2. Delete the `.code()` branch and the string patterns.
  - No retry library (Appendix D).
- **Tests.**
  - Red: the status table above, after the fix, gives True for UNAVAILABLE,
    RESOURCE_EXHAUSTED and ABORTED.
  - Red: the pool makes 1 call on INVALID_ARGUMENT.
  - Control: the pool makes 3 calls on UNAVAILABLE.
- **Done when:** all three tests pass.

#### GPU-9: A sliver crop makes the CLIP resize allocate up to ~3 GB
- **Class** LIVE · **Severity** medium · **Size** S
- **Reachable by:** `POST /embed/boxes` (up to 100 caller-chosen boxes) and
  `analyze_pipeline.crop_batch_tensor`.
- **Bug.** `utils/cpu_preprocess.center_crop_cpu` scales the short side to 256 before
  centre-cropping. A 1x16384 box becomes about 256 x 4.2M x 3, roughly 3.2 GB.
- **Evidence.** The verdict measured peak RSS of 623 MB for a (3000,1) crop against
  124 MB for (3000,8).
- **Fix.** Crop the long side first, to the window that survives the centre crop, then
  resize only that window. This is exact for a centre crop and changes no valid
  output.
- **Tests.**
  - Red: `tracemalloc` peak for a (3000,1) crop is under 20 MB.
  - Control: a 256x256 and a 480x640 crop give byte-identical output to the current
    function.
- **Done when:** no resize intermediate exceeds about 4x the output.

#### GPU-10: PIL CLIP preprocessors truncate and black-pad one row
- **Class** LIVE · **Severity** low · **Size** S
- **Reachable by:** `/embed/image` and `TritonClient.encode_image`.
- **Bug.** `int(w * 256 / min(w,h))` undershoots 256 for about 11 % of sizes, so one
  black row is cropped in. `center_crop_cpu` already fixes this with `round()` and
  `max()`.
- **Fix.** Route both PIL copies (`triton_preprocess.py`, `services/embedding.py`)
  through `center_crop_cpu` on the array, and delete the copies. This changes the
  resampler on those paths; record that in CHANGELOG.
- **Tests.**
  - Red: a 300x187 image gives no black row.
  - Control: a 480 short side.
- **Done when:** one implementation remains.

#### GPU-12: Letterbox undo subtracts the half-pixel pad
- **Class** LIVE · **Severity** low (sub-pixel) · **Size** S
- **Fix** with #407. That work owns the byte-identical GPU parity gate. Use
  `float(int(pad))` consistently in the CPU letterbox, the GPU geometry and the client
  copy, and update `tests/models/test_gpu_image_prep_geometry.py` together.
- **Tests.**
  - Red: round-trip error on an odd-margin size (0.5 px today).
  - Control: an even margin.
- **Done when:** the round-trip error is below 0.01 px.

#### GPU-13: RGBA compositing in OCR uses float64 temporaries
- **Class** LIVE · **Severity** low · **Size** S
- **Fix.** Covered by GPU-3a step 2 (Pillow `alpha_composite`). Delete the numpy
  float64 path in `triton_ocr._uint8_bgr`.
- **Tests.**
  - Red: peak allocation for a 20 MP RGBA image is under 3x the image size.
  - Control: pixels within 1 level of the old output.

#### GPU-14: Box frame guessed from `boxes.max() > 1.0`
- **Class** LATENT · **Severity** medium · **Size** S
- **Reachable by:** `/detect`, `/analyze`, `infer_yolo_clip_cpu` (`utils/affine.py`).
  It becomes live the first time a normalised end2end box overshoots 1.0.
- **Bug.** One box with `x2 = 641/640` makes every box in the image be treated as
  pixels, and they collapse to about 0.
- **Evidence.** `inverse_letterbox_coords` on `[[.1,.2,.3,.4],[.5,.5,641/640,.9]]`
  destroys the first, valid box. The verdict reproduced it.
- **Fix.**
  1. Interim, in this PR: clip normalised end2end outputs to [0,1] in the adapter
     before `inverse_letterbox_coords`, so the heuristic cannot misfire.
  2. Final (follow-up): carry the box format as a Triton model parameter set at export
     (`normalize_boxes`), and delete the guess. The branch's region-detector fix
     (GPU-15) now decides per tensor; align with it.
- **Tests.**
  - Red: the case above returns the first box unchanged.
  - Control: all boxes at or below 1 give the same result as today.
- **Done when:** both tests pass.

#### GPU-15: Region detector head layout (fixed on the branch)
- **Fixed by:** `6de660bc` and `4a15558c`.
  - The layout now comes from the model's declared `output0` shape, through
    `AsyncTritonPool.get_model_output_shape`.
  - An unknown declaration yields no candidates and is logged, so a 3-class
    `[B,7,N]` head fails closed.
  - Pixel versus normalised is decided per tensor, not per row.
- **Left over, optional:** refuse such a model at profile validation instead of at
  decode time. Not scheduled.

#### GPU-16: Sync Triton calls have no `client_timeout`
- **Class** LATENT · **Severity** low-medium · **Size** S
- **Reachable by:** a live-but-hung Triton. A dead connection is already caught by the
  30 s gRPC keepalive.
- **Bug.** `triton_client._infer_with_retry`, `fast_face_client`,
  `pe_text_backends.TritonTextBackend.encode` and `services/embedding.py` call `infer`
  without `client_timeout`. The Triton client docs: `None` waits forever.
- **Fix.**
  1. Add one settings-driven `client_timeout` (for example 30 s, in `src/config`)
     through a single wrapper used by these call sites.
  2. Map the deadline error to 504.
  3. Do not retry on DEADLINE_EXCEEDED; GPU-8's predicate excludes it for heavy
     calls.
- **Tests.**
  - Red: a stub client that never answers raises within the timeout.
  - Control: a fast stub succeeds.
- **Done when:** `rg -n "\.infer\(" src` shows no call without a timeout.

#### GPU-17 (with API-12): `model_residency._ready` never learns of an unload
- **Class** LATENT · **Severity** low · **Size** S
- **Reachable by:** only under `OP_LOAD_PROFILE=ingest` plus a manual unload or a
  Triton restart.
- **Bug.** `src/clients/model_residency.py` keeps a process-global `_ready` set and
  per-model `asyncio.Lock` dicts. After an unload, every worker skips the readiness
  check and fails. The branch's module-level-lock guard does not catch the dict of
  locks.
- **Fix.**
  1. Delete `_ready` and the per-model locks.
  2. Call `is_model_ready` (cheap) before each ensure, and load when it is not ready.
     Triton serialises loads, and the code already tolerates "load lost the race".
     That load is idempotent, so no lock is needed.
- **Tests.**
  - Red: ready, then unloaded, then a call must issue a load.
  - Control: a ready model is not reloaded.
- **Done when:** no module-level readiness set or lock dict remains in
  `model_residency.py`.

#### GPU-18: EfficientNMS plugin deprecation not recorded
- **Class** LATENT · **Severity** low · **Size** S (docs only)
- **Fix.** Add an entry to `docs/MIGRATION_TRITON_26.md`:
  - EfficientNMS_TRT has been deprecated since TensorRT 10.12, and its ONNX variant
    was removed in 10.16. The replacement is `INMSLayer`.
  - YOLO26 NMS-free (`FusedDetAdapter`) is the forward path, and YOLO11 end2end is
    legacy.
  - No CI check (Appendix D).
- **Done when:** the entry exists and `scripts/docs/check_docs_vs_code.py` passes.

#### GPU-19: OCR detector normalisation (needs evidence)
- **Background.** The BLS uses `x/127.5-1` with a BGR to RGB flip. Upstream PP-OCRv5
  uses ImageNet mean/std, and the converted ONNX's true contract is unknown.
- **Evidence task.** Run 50 public COCO-Text images through the Paddle reference (or
  onnxruntime with each normalisation) and through the BLS, and compare boxes and
  recall.
- **Act only on:** a measured gap.

### 4.4 frontend

#### FE-1: `/export` makes a raw `fetch` of `/stats/dataset` that never refreshes
- **Class** LIVE · **Severity** low · **Size** S
- **Bug.** `frontend/src/routes/p/[project]/export/+page.svelte` runs an `$effect`
  with a raw `fetch` of the same URL that `getStats()` already reads. It is a double
  request, it bypasses `apiFetch`, and its chips stay stale after a Refresh.
- **Evidence.** `rg -n '\bfetch\(' frontend/src --glob '!*.test.ts'` finds this site
  and `api.ts` only. The verdict confirmed it.
- **Fix.**
  1. Add `by_source` to `StatsSummary` in `getStats` (it is already in `RawDataset`).
  2. Delete the effect and its imports.
  3. Add `const hddSources = $derived(stats?.by_source ?? [])`.
  4. Keep the `dataset_error` notice.
- **Tests.**
  - Red: mount with one stubbed GET, click Refresh, and assert the chips update with
    exactly one request per load.
  - Control: a 503 shows the notice.
- **Done when:** there is no `fetch(` outside `api*.ts`. Open the page in a browser
  (desktop and 430 px) before reporting done.

#### FE-2: Diverse page fans out 30 `GET /crops/{id}`; one failure fails the page
- **Class** LIVE · **Severity** medium · **Size** S
- **Reachable by:** `/review` with the diverse overlay.
- **Bug.** `frontend/src/lib/review/reviewDiverseController.svelte.ts` runs
  `Promise.all(ids.map(getCrop))`: 30 requests, no AbortSignal, and the whole page is
  rejected when one stored id is stale.
- **Correction from the verdict.** A batch read already exists:
  `GET /curation/projects/{project}/crops?ids=a,b,c` (max 500, in order, missing ids
  dropped). No frontend function uses it.
- **Fix.**
  1. Add `getCropsByIds(ids, signal)` in `api_cropRead.ts` over
     `GET {scoped()}/crops?ids=`.
  2. Call it once per page, with the loader's signal.
  - No backend change and no contract change.
- **Traps.** Check that `mapRawCrop` accepts the list-row shape; the review list loader
  already maps list rows.
- **Tests.**
  - Red: one stubbed list response missing 1 of 30 ids gives 29 items and no page
    error, from exactly one request.
  - Control: all 30 present.
- **Done when:** one request per page. Browser check.

#### FE-3: Project binding in `load` with global hover preload
- **Class** LATENT · **Severity** low · **Size** S
- **Bug.** `+layout.ts` `load` calls `projectsStore.select()`, and `app.html` sets
  `data-sveltekit-preload-data="hover"`. The verdict confirmed that no cross-project
  anchor lives inside a `/p/<slug>` page today.
- **Fix.**
  1. Add `data-sveltekit-preload-data="tap"` to the three cross-project anchor sites
     (`routes/projects/+page.svelte`, `projects/combine/[job_id]/+page.svelte`,
     `ProjectUnavailable.svelte`).
  2. Add a guard test that no `/p/` page renders `projectHref(..., otherSlug)` without
     `tap`.
  - Do not move `select()` out of `load` (Appendix D).
- **Tests.**
  - Red: the guard test fails when an anchor without `tap` is added in a fixture.
  - Control: the current tree passes.
- **Done when:** both tests pass.

#### FE-5: Client-side sums in `TrainDatasetSection` (conditional)
- **Class** LATENT · **Severity** low · **Size** S
- **Fix.** Only if `getStats().validated_crops` is already loaded on `/train`: render
  that value and delete the two `reduce` sums. Otherwise leave it, and note it in
  `frontend/CLAUDE.md`'s served-facts section as a known fallback.
- **Tests.**
  - Red: a stubbed stats value differs from the per-class sum, and the served value is
    shown.
  - Control: equal values.
- **Done when:** the served value is shown, or the fallback is documented.

#### FE-6: Pager buffers grow without bound (needs evidence)
- **Evidence task.** In a browser on the live stack, with a public COCO project of
  more than 10k crops, scroll 500 pages of `/clusters` and review. Record the DOM node
  count and the heap.
- **Act only if:** it degrades. Windowing is Size L with high regression risk.

#### FE-7: SSE reconnect does not resync keymap and config
- **Class** LATENT · **Severity** low · **Size** S
- **Bug.** The keymap `config.changed` subscriber in `p/[project]/+layout.svelte` and
  the config controllers do not refetch on reconnect. The backend sends no SSE `id:`,
  so missed events are lost. `openStream` already supports `onOpen`, and
  `DatasetStats` uses it.
- **Fix.** Pass `onOpen` from those subscribers, refetch on every open after the
  first, and leave `openStream` unchanged.
- **Tests.**
  - Red: a fake EventSource errors and reopens, and the keymap refetch fires once.
  - Control: the first open does not refetch.
- **Done when:** both tests pass.

### 4.5 tests

#### T-1: Assertions only inside a loop over a registry that may be empty
- **Class** LATENT · **Severity** low · **Size** S
- **Sites:**
  - `tests/curation/test_crop_scores.py` (`test_scorer_metadata_writes_match_scorer_class_writes`)
  - `tests/curation/test_class_sources.py` (`test_every_catalog_entry_has_a_short_badge_label`)
  - `tests/curation/test_review_sorts.py` (two tests over `get_review_sorts()`)
- **Evidence.** The mutation harness (`<scratch repro>`) emptied each registry, and
  the tests still passed. The control mutation (`["bogus"]`) failed.
- **Fix.** Add one `assert <registry>` before each loop. No ratchet (Appendix D).
- **Tests.**
  - Red: the emptying mutation, applied in a scratch copy, now fails each test.
  - Control: the unmutated tree passes.
- **Done when:** the three mutations fail their tests.

#### T-2: `test_gate_sim` open-set test never runs its asserts
- **Class** LIVE · **Severity** low-medium · **Size** S
- **Bug.** `tests/research_scripts/test_gate_sim.py` wraps both asserts in
  `if out['propagated']:`, and with seed 3 `propagated == 0`. The asserts are also
  invariants that hold by construction.
- **Evidence.** `simulate(np.zeros(220,int), ['goat']*200+['sheep']*20, seed=3, open_set={'goat'})`
  returns `propagated: 0`.
- **Fix.** Choose inputs that propagate and assert exact counts, as the sibling test
  does (`== 120`). Remove the `if`.
- **Tests.**
  - Red: mutating `open_propagated` to `10**9` in `gate_sim.py` must fail the test. It
    passes today.
  - Control: the unmutated script passes.
- **Done when:** the mutation fails the test.

#### T-3: Compose render tests skip on a non-zero exit
- **Class** LIVE · **Severity** low-medium · **Size** S
- **Sites:**
  - `tests/test_monitoring_config.py`
  - `tests/compose/test_vlm_service_command.py`
  - `tests/compose/test_images_build_and_overlays.py`
- **Bug.** Each one does `if returncode != 0: pytest.skip(...)`.
  `tests/compose/test_overlay_render.py` has the correct form.
- **Fix.** Skip only when the docker CLI or the compose plugin is absent ("unknown
  command"). Otherwise `assert returncode == 0, stderr`.
- **Traps.** The tests run with a stripped environment. A required `${VAR:?}` on a
  fresh checkout would now fail rather than skip, so run the three files once on the
  CI runner config before merging.
- **Tests.**
  - Red: append invalid YAML to the compose file in a `git archive` scratch copy; the
    three tests must fail (today they skip).
  - Control: the clean tree passes.
- **Done when:** a corrupted compose file fails all three.

#### T-4: `einops` missing from `requirements-test.txt`
- **Class** LIVE · **Severity** low-medium · **Size** S
- **Bug.** `tests/test_pe_rearrange_shim.py` (8 tests) skips: "could not import
  'einops'". `onnxconverter-common` is already listed; its absence is local venv drift
  only.
- **Fix.** Add a pinned `einops` to `requirements-test.txt` and install it in the
  venv. Do not add `leidenalg`/`igraph` (Appendix D).
- **Tests.**
  - Red: `pytest -rs tests/test_pe_rearrange_shim.py` shows the skip today.
  - Control: after the fix, the module runs on the CPU torch that CI uses.
- **Done when:** the module runs in CI with no skip.

#### T-6: Sleep-ordered FCFS test (needs evidence)
- **Background.** `tests/models/test_gpu_image_prep_budget.py` orders threads with
  `time.sleep(0.05)`. It failed only under an injected 150 ms delay; it passed at load
  of about 23.
- **Act only if:** it flakes in CI or locally. Then poll `thread.is_alive()` with a
  bounded wait. Do not add a production counter for tests.

#### T-7: No test asserts project disk-usage bytes
- **Class** LATENT · **Severity** low · **Size** S
- **Bug.** `services/projects/stats.py` `_dir_bytes` and its cache `_disk_cache` (300 s)
  have no test that asserts `exports_bytes`/`uploads_bytes`/`runs_bytes`. Mutating
  `_dir_bytes` to return a constant left 9 tests passing.
- **Fix.** Add a test with a temporary project directory holding a 10-byte file. Clear
  `_disk_cache` in the test (or add it to `_PROCESS_CACHES` in `tests/conftest.py`).
  No registry (Appendix D).
- **Tests.**
  - Red: the constant mutation of `_dir_bytes` fails the new test.
  - Control: the real code gives 10.
- **Done when:** both tests pass.

### 4.6 ops/compose

#### OPS-1: Recycle jitter documented but not set
- **Class** LIVE · **Severity** low-medium · **Size** S
- **Reachable by:** the default compose. All 32 workers recycle after similar request
  counts, and in-flight jobs die together (CC-4).
- **Bug.** `docs/PERFORMANCE.md` documents `--limit-max-requests-jitter=1000`, but
  `docker-compose.yml` has no such flag. `rg -n jitter docker-compose*.yml` returns 0
  hits; it was checked on the branch.
- **Docs.** `uvicorn --help` (0.54.0): `--limit-max-requests-jitter`.
- **Fix.** Add `--limit-max-requests-jitter=1000` to the `api` command in
  `docker-compose.yml`, next to `--limit-max-requests`, and check the Dockerfile `CMD`
  for the same drift. If the owner prefers no jitter, correct `PERFORMANCE.md`
  instead. One of the two, never both out of step.
- **Tests.**
  - Red: an offline compose test (`tests/compose/`) asserts that the `api` command
    contains the jitter flag whenever it contains `--limit-max-requests`.
  - Control: the overlay render still passes.
- **Done when:** the docs and compose agree.

#### OPS-2 (API-14): Prometheus multiprocess files of dead workers (needs evidence)
- **Background.** `mark_process_dead` is never called. Each recycled worker leaves its
  `.db` files in the tmpfs. The scrape-cost impact is **unmeasured**; with
  `livemostrecent`, correctness is not affected.
- **Evidence task.** Simulate 1,000 recycles into a temporary
  `PROMETHEUS_MULTIPROC_DIR` and time `/metrics` generation.
- **Act only if:** the scrape time grows materially. Then call
  `prometheus_client.multiprocess.mark_process_dead(os.getpid())` in the lifespan
  shutdown (uvicorn has no parent hook). Fold this into Celery Phase 4; Celery workers
  get their own directory (task_queue_plan E19).

#### SEC-4, SEC-5
- One-line tasks; see the index. Details are private.

### 4.7 cross-cutting concurrency

#### CC-1 (API-4): Fixed temp names make concurrent writers raise `FileNotFoundError`
- **Class** LIVE · **Severity** low-medium · **Size** S
- **Reachable by:** a status poll of a file-backed job while its heartbeat is stale
  (`item_scores` `get_state` writes), job progress writes, and resume or claim paths.
- **Bug.** `src/services/curation/file_job.py` writes `state.tmp` then `replace()`.
  Two processes share that one temp path, and the second `replace()` finds it gone.
  The same fixed-name pattern appears in:
  - `holdout.py` (`current.json.tmp`)
  - `routers/curation/bakeoff.py`
  - `dataset_import/store.py`
  - `label_ops/store.py`
  - `clients/curation_opensearch/registry.py` (the CC-2 500 path)
  - Find them all with `rg -n "with_suffix\(.*\.tmp|\.tmp'\)" src`.
- **Docs.** Python `os.replace` is atomic for one source file; each writer needs its
  own source. `tempfile.NamedTemporaryFile(dir=..., delete=False)` gives a unique name
  in the same directory.
- **Evidence.** `<scratch repro>`, 4 spawned processes x 3000 `FileJob.write`
  calls, gave 1585-2053 `FileNotFoundError` per process.
- **Root cause.** A shared temp name. With unique names, last-writer-wins on whole
  documents is the only remaining behaviour, and writes are atomic.
- **Fix** (no lock, per rule 1):
  1. Add `src/utils/atomic_write.py` with `atomic_write_json(path, obj)` and
     `atomic_write_bytes(path, data)`: `NamedTemporaryFile(dir=path.parent,
     delete=False)`, write, `flush`, `os.fsync`, `os.replace`, and unlink the temp file
     on failure.
  2. Use it at every site above, and delete each hand-rolled tmp+replace.
  3. Readers must not persist state. Convert the `get_state` functions that write a
     repaired state (for example `item_scores/job.py`, `embedding_viz.py`; find them
     with `rg -n "def get_state" -A25 src/services/curation | rg "write\("`) so they
     **return** the derived `interrupted` view without writing it. Persisting stays
     with `FileJob.repair_if_stale` and the startup reconcile today, and with the lease
     reaper in Celery Phase 4.
  - Do not add `_state_lock` to `write()` (it is a file lock, against rule 1, and it
    self-deadlocks with `update()`).
- **Traps.** `os.fsync` on tmpfs is cheap. Keep the snapshot file in `registry.py`.
- **Tests** (`tests/test_atomic_write_concurrent.py`, the renamed Celery P1.0 test).
  - Red: 4 spawned processes x 3000 writes to one job give 0 errors. It is red today
    through `FileJob.write`.
  - Red: a status poll on a stale job does not change the `state.json` mtime.
  - Control: a single writer round-trips the content.
- **Done when:** the 4-process run gives 0 errors in 10 consecutive runs.

#### CC-2 (API-2): Class registry read-modify-write loses classes or duplicates ids
- **Class** LIVE · **Severity** medium · **Size** L
- **Reachable by:** `POST /curation/projects/{p}/classes`, `PUT /classes/{id}`, merge,
  dataset-import prepare and project combine. Two users, or a user plus an import.
- **Status.** The branch deleted the unused module-level `_registry_lock`
  (`b48a8f3a`). The race itself is **not** fixed.
- **Bug.** `ClassRegistry.add_class/rename_class/merge_class/set_deprecated` each do
  `load()` (cached by mtime), then mutate, then `_atomic_write()` on
  `class_registry.json`, with no version check. `new_id = max+1` comes from a stale
  snapshot. There are two failure modes:
  - a lost class with a duplicated id, while the API still answers 201;
  - a 500 from the shared `.tmp` name.
- **Evidence.** `<scratch repro>` (16 spawned `add_class` calls on one file):
  1 of 3 runs persisted 15 of 16 with id 6 handed out twice. The verdict's re-run gave
  the `FileNotFoundError` mode.
- **Root cause.** A shared document mutated with neither a version check nor
  idempotent creation.
- **Decision J-REG (owner):** the registry's authority moves to OpenSearch with OCC,
  because file locks are ruled out. Recommended design:
  - One registry document per project (in the project's config or classes index),
    holding `classes[]` and `next_class_id`. This is the same pattern as
    `config_store`'s revisioned docs.
  - Each mutator does: GET (with `seq_no`/`primary_term`), apply the mutation in
    memory, then `index(..., if_seq_no, if_primary_term)`. On a 409, re-read and
    re-apply, with bounded retries. Use the repo's OCC helpers in `src/clients/occ*.py`
    rather than a new loop.
  - `add_class` is idempotent by name: if a non-deprecated class with that name
    exists after the re-read, return its id. A retry therefore never duplicates.
  - `class_registry.json` becomes a **derived export**, written for the file readers
    (74 references, including `docker/trainer/` and the export writers;
    `rg -n "class_registry_path|class_registry\.json" src scripts docker`). The
    design spike must decide how the file is kept consistent. Options:
    - (a) write it at export and training-dataset time only, from the doc;
    - (b) write it after each successful OCC write, carrying the doc `_seq_no`, so a
      reader can detect an older file.
    - Do not use (b) where a stale file can be read as truth.
- **Steps that ship before J-REG** (in PR-2):
  1. CC-1 gives the registry a unique temp name, which removes the 500 mode.
  2. Fold the follow-up hotkey write in `create_registry_class` and the rename-then-write
     in `update_class` (`routers/curation/classes_write.py`) into the same single
     mutation. This is valid whichever store is chosen.
- **Traps.**
  - Class identity is by name, so the migration must preserve ids exactly. Seed the
    doc from the current file with `op_type='create'`; that is idempotent.
  - `combine` and `import` call the registry from async code. OpenSearch calls are
    async, so no `to_thread` is needed once the authority moves.
  - `sync_to_opensearch` (the per-class mirror into the `classes` index) stays in step
    with the doc.
- **Tests** (`tests/test_class_registry_cross_process.py`, Celery P1.3).
  - Red: 8 spawned processes `add_class(f'c{i}')` at a `multiprocessing.Barrier`
    against a stateful fake OpenSearch that enforces `if_seq_no`. Assert 8 classes, 8
    unique ids, and each id maps to its own name. Against today's file-backed registry
    this test is red intermittently, so repeat it 20x.
  - Control: sequential adds give ids 0..N-1.
  - Live: the same scenario on the live harness with two API workers.
- **Done when:** 100 runs of the race give N persisted classes and N distinct ids, and
  no file lock was added.

#### CC-3 (API-5): Startup reconcile can mark a sibling's just-started job interrupted
- **Class** LATENT · **Severity** low · **Size** S
- **Bug.** `job_reconcile.reconcile_stale_running` treats a missing heartbeat as
  stale. With 32 workers, one worker's startup can flip a job that another worker
  started microseconds earlier. `FileJob.is_live()` documents the opposite rule.
- **Fix** (idempotent, no lock): treat a missing heartbeat as stale only when the
  job's `started_at`, or else the `state.json` mtime, is older than
  `HEARTBEAT_STALE_S`. Keep the sweep (Appendix D); Celery Phase 4 deletes it in favour
  of the lease reaper.
- **Tests** (`tests/test_reconcile_sibling_job.py`).
  - Red: `running`, no heartbeat, `started_at` = now; the job is left alone.
  - Control: `running` with a heartbeat 60 s old becomes `interrupted`.
- **Done when:** both tests pass.

#### CC-4 (API-6 + asyncio R5): Request-launched jobs die when their worker recycles
- **Class** LIVE · **Severity** medium · **Size** L
- **Reachable by:** the item-score, selection, embedding-viz, probe, reprocess,
  dataset-import, autolabel and promote jobs. All are `asyncio.create_task` calls
  inside an API worker, which recycles after 10,000 requests.
- **Evidence.** A minimal uvicorn app at `--limit-max-requests 6` shows the job
  printing ticks 0-5, then "Maximum request limit of 6 exceeded. Terminating process",
  and DONE never printed.
- **Fix.** This is `task_queue_plan.md` **Phase 3**: move each family into a Celery
  task, one family per commit. Interim: OPS-1 (jitter).
  - Do **not** add a `spawn_job` helper or a `create_task` ratchet (Appendix D); they
    would cement jobs in API workers.
  - Shutdown waits are bounded by `--timeout-graceful-shutdown=30` and cannot cover
    multi-minute jobs, so they are not a fix.
- **Tests.** Per family, as in task_queue_plan Phase 3. For example, a live test that
  kills `jobs-worker` mid-job and sees the reaper mark the job `interrupted`.
- **Done when:** no job family runs as a task inside an API worker.

#### CC-5 (asyncio R1 and the "missed" thumbnails): Blocking work in `async def` handlers
- **Class** LIVE · **Severity** medium · **Size** S (handlers) / M (upload, directory
  ingest)
- **Reachable by:** every grid view (thumbnails, on a cache miss), the source-image
  render, dataset upload and legacy directory ingest.
- **Bug.**
  - `routers/curation/images.py` `crop_thumbnail` and `region_thumbnail` call the sync
    `THUMBNAIL_CACHE.get_or_compute` (PIL decode) inside `async def`.
  - `services/curation/image_serving.render_source_image` runs PIL
    open/transpose/resize/encode inline.
  - `dataset_import/upload.py` extracts, `rmtree`s and hashes on the loop.
  - `routers/ingest/directory.py` runs `rglob` and `read_bytes` over a tree.
- **Docs.** FastAPI "Concurrency and async/await": blocking work belongs in a plain
  `def` path operation (threadpool) or behind `run_in_threadpool`. The Python asyncio
  dev guide, "Running Blocking Code", says to use `asyncio.to_thread`.
- **Fix.**
  1. PR-4: wrap the one blocking call in each of the three image handlers with the
     project's existing `run_cpu` helper or `asyncio.to_thread`. Use `run_cpu` if it is
     the bounded shared executor; check its docstring. This is not fan-out: one
     offload per request.
  2. Celery Phase 3: the upload extraction becomes the `dataset_upload_extract` job,
     and directory ingest becomes the `directory_ingest` job (202 plus a job id, with
     the contracts regenerated). No interim `to_thread` for these two; they are rare
     operator actions.
  - No ruff ASYNC rule (Appendix D: 22 hits, about 0 real, and PIL is not detected).
- **Traps.** The thumbnail cache must stay safe when called from threads. Check
  `get_or_compute` for shared mutable state; it is a per-process cache, so a thread
  race costs at most a duplicate compute.
- **Tests.**
  - Red: an event-loop lag probe (a coroutine measuring `loop.time()` drift) while a
    large-image thumbnail is computed. Drift is above 100 ms today and below 20 ms
    after.
  - Control: a cached thumbnail shows no drift either way.
- **Done when:** the three handlers do no PIL work on the loop.

#### CC-6 (asyncio R2 + R8): PIT slice producers orphaned on error
- **Class** LIVE (error path) · **Severity** low · **Size** S
- **Reachable by:** an OpenSearch error mid-cluster, in `pit_pages.iter_pages` and
  `residual_fetch.py`. This is worker and clustering code, not a request handler.
- **Bug.** `gather` over the slice producers does not cancel the siblings when one
  raises. They keep searching a PIT that is being deleted, and block on a full queue.
- **Evidence.** `<scratch repro>` (fake client, slice 0 raises) showed "leaked tasks
  after return: 3" and 10 searches issued after the failure.
- **Docs.** Python asyncio "Task Groups": on a child failure, the remaining tasks are
  cancelled. Exceptions are raised as an `ExceptionGroup`.
- **Fix.**
  1. Replace the `gather` with `async with asyncio.TaskGroup()` inside `run()`. Emit
     the sentinel after the group exits.
  2. Do the same in `residual_fetch`, deleting the PIT after the group.
- **Traps.** Callers that catch a specific exception
  (`clustering/orchestrator.py`, `cluster_geometry.py`) will stop matching an
  `ExceptionGroup`. Use `except*` there, or re-raise the single child exception at the
  boundary. Check every caller first with `rg -n "iter_pages|residual_fetch" src`.
- **Tests.**
  - Red: the fake client with slice 0 raising leaves 0 pending tasks after the call
    returns, and no search after the failure.
  - Control: a clean run yields every page once.
- **Done when:** both tests pass.

#### CC-7 (asyncio R3): `ingest_batch._one` representative future never resolved on error
- **Class** LIVE (error path) · **Severity** low · **Size** S
- **Bug.** In `curation/ingest_batch.py`, when `ingest_one` raises, the rep future is
  never set. Followers stay pending until GC, which logs "Task was destroyed but it is
  pending".
- **Fix.** Covered by the J7 decision in task_queue_plan Phase 6. If the ingest data
  plane keeps in-request concurrency as a documented exemption, restructure it as an
  `asyncio.TaskGroup`, so a failure cancels the siblings and nothing is left pending.
  If it moves to a worker, apply the same TaskGroup there.
- **Tests.**
  - Red: a stubbed `ingest_one` raising for the representative leaves no pending
    follower tasks after the request fails.
  - Control: an all-success batch is unchanged.
- **Done when:** both tests pass.

#### CC-8 (asyncio R4): Detection worker stage tasks have no crash callback
- **Class** LATENT (reachability unverified) · **Severity** low · **Size** S
- **Bug.** `scripts/curation/worker/runner.py` starts about 12 stage tasks with no
  done-callback. A dead consumer would not show in the heartbeat, which covers only the
  producer and writer.
- **Fix.**
  1. Reuse `vlm_worker`'s existing `_crash_on_unhandled_exception` with
     `add_done_callback` on each stage task. Moving it to a shared module is allowed if
     the layering ratchet requires it.
  2. Log the drain gather's results.
- **Tests.**
  - Red: a stage coroutine that raises outside its per-item try makes the worker exit
    non-zero.
  - Control: a clean drain exits 0.
- **Done when:** both tests pass.

#### CC-9 (asyncio R7): Spawn-everything gathers in job code
- **Class** LATENT · **Severity** low · **Size** S per site
- **Sites:** `reprocess_open_vocab.py`, `assist/sample_eval.py`,
  `autolabel/vlm_stage.py`.
- **Fix.** When each family moves to a worker (Celery Phase 3), rewrite its fan-out as
  `asyncio.TaskGroup` with an `asyncio.Semaphore` bound. Only where the list is large
  and uncapped.
- **Tests.**
  - Red: with N=1000 inputs, at most the bound is in flight (count a fake).
  - Control: N=3.
- **Done when:** each site has a bound and uses TaskGroup.

#### CC-10 (asyncio R6): VLM chunk fan-out in a request path (needs evidence)
- **Background.** `vlm_labeler_classify` (behind `POST /curation/vlm/label_batch`)
  gathers chunk requests with no in-flight bound beyond the token bucket. The
  reviewers did not re-measure it.
- **Evidence task.** With a stub VLM endpoint, count the peak in-flight requests for a
  maximum-size batch.
- **Fix if real:** move it to a worker (task_queue_plan Phase 6, J7), with a
  `Semaphore` plus TaskGroup there. Rule 2 forbids keeping the fan-out in the handler.

#### CC-11: Module-level `asyncio.Lock` sites (fixed on the branch)
- **Fixed by:** `b48a8f3a` and `d453f856`.
  - `index_bootstrap._ensure_indexes_lock` is deleted. The bootstrap is now idempotent
    (create-if-absent, additive mapping puts, marker last).
  - `pipeline_events._stats_cache_lock` is deleted, replaced by per-project in-flight
    coalescing. That is still a per-process cache, documented as such.
  - `registry._registry_lock` is deleted (unused). See CC-2 for the race.
- **Left:** `open_vocab_ingest._PASS_LOCK` remains, allowlisted in
  `tests/test_no_module_level_async_locks.py` as a documented local throttle. Celery
  Phase 3, item 9, deletes it.
- **Note:** the new guard test checks only module-level `asyncio` primitives. It does
  not see the lock dict in `model_residency` (GPU-17).

---

## 5. Verification steps

Run these per PR. Do not run them all at once; the host is shared.

1. **Red first.** Run each task's red test on the unchanged base and record the
   failure output in the PR description. Then apply the fix and show the red test and
   its control both green.
2. **Targeted runs:** `.venv/bin/python -m pytest <the task's test files> -q --no-cov`.
   No `-n`.
3. **One full offline suite per PR:**
   `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'`.
4. **Hooks:** `.venv/bin/pre-commit run --all-files`. Fix any hook failure; never
   bypass one.
5. **Contracts:** `make contracts` in the same commit as any route or model change
   (OS-10, BA-6, CC-4/CC-5 Phase 3). Then `make contracts-check`.
6. **Docs:** `.venv/bin/python scripts/docs/check_docs_vs_code.py` (GPU-18, OPS-1,
   BA-8).
7. **Frontend (PR-6):** in `frontend/`, run `npm run check` and `npm run build`. Open
   `/p/<slug>/export` and `/review` (diverse) in a browser with `DISPLAY=:11`, at
   desktop and 430 px, and view the full-page screenshots.
8. **OCR (PR-1):** on the live harness (`docker compose -p op-ocr-fix ...`, GPUs 0/2,
   healthchecks, no sleep loops):
   - Run 50 public COCO val2017 text images through `/ocr` before and after.
   - Record the box geometry and the text.
   - Include rotated copies (EXIF 6 and 8) and RGBA PNGs.
   - Expect boxes to align with the text. The text may change.
9. **Images (GPU-2):**
   - Rebuild the Triton image under a local-only tag.
   - `find_spec('pyclipper')` must be True.
   - Run the image-vs-git audit and Trivy before any push.
10. **Cross-process (PR-2):** run the spawn-based races (CC-1, CC-2) 10x locally. Then
    run the CC-2 live scenario on the live harness with two API workers under its own
    `-p` name.
11. **Before push:** run every CI gate locally (changelog rules, frontend full set,
    offline suite). Verify the push by `git ls-remote` and the CI run, not by a piped
    exit code.

---

## Appendix A: Method

1. **Audit.** Seven area audits on the pinned tree at `5b95591f`, run read-only to
   `BRIEF.md`.
   - Every claim had to carry a doc citation, a reproduction command, executed proof,
     a root-cause fix and its blast radius.
   - Each claim was classified LIVE, LATENT, NOT A DEFECT or UNVERIFIED.
2. **Adversarial review.** Fresh sessions, to `ADVERSARIAL_BRIEF.md`.
   - The default stance was "wrong until reproduced".
   - Reviewers re-ran the repro scripts, hunted for upstream guards, checked each fix
     against the build, the tests, production and the owner rules, and value-tested any
     tooling.
   - Verdicts: CONFIRMED, RIGHT-SIZED, NEEDS SCOPING, NOT A DEFECT or UNVERIFIED.
3. **This plan.**
   - It adopts the verdict severity and the right-sized fix.
   - It merges duplicates: API-12 with GPU-17, API-6 with R5, R2 with R8.
   - It splits GPU-3 into an R3-sized part and a later part.
   - It rewrites every fix that used a file or in-process lock to use OCC, idempotency,
     or the Celery job design.
4. **Branch status.** `git log main..fix/411-r2-round` (44 commits), and
   `git diff main...fix/411-r2-round` read per file for every finding's site. A
   spot-check `rg` in the branch worktree confirmed that the open items are still
   present there: the factory substring match, the unchecked holdout bulk, the OCR pad
   block, the optional pyclipper import, the fixed `.tmp` names, the `_READY` write,
   `except Exception` in `crops_exclude`, `error_type` bodies, no jitter in compose,
   `SAM3_SHA=main`, no `einops`, and the class-name cache.
5. **Constraints.** No tests, stacks, engines, npm or browsers were run for this plan.
   It relies on the audit and verdict runs, which were single files and scratch scripts
   under `/tmp`.

## Appendix B: Negative results (checked and clean, do not re-litigate)

- **API.**
  - No Pydantic v1 APIs.
  - Pydantic mutable defaults are deep-copied per instance.
  - `os.environ` reads are confined to `src/config/env.py`.
  - The lifespan is correct (asynccontextmanager, isolated shutdown steps).
  - These multi-process designs are correct:
    - the project registry (revision counter plus `ensure_fresh`);
    - the config store;
    - `FileJob` start, `repair_if_stale` and the heartbeat;
    - the ingest dispatcher;
    - the event hub;
    - the ops-metrics `_claim_turn`;
    - `delete._FINISH_IN_PROGRESS` (backed by OCC);
    - `crop_orders._BACKFILLING` (idempotent).
  - Clients are app-scoped and closed in the lifespan.
- **OpenSearch.**
  - One client per process through the factory.
  - The OCC helpers and the project, policy, bootstrap and config writes are correct.
  - Every `update_by_query` passes `conflicts='proceed'`.
  - `refresh=True` appears only on admin paths.
  - Deep pagination elsewhere uses `search_after`.
  - The curation kNN uses efficient filtering.
  - Every client sets a timeout.
- **GPU.**
  - BGR/RGB order is correct on every path.
  - `ensemble_nms` uses `torchvision.ops.nms`.
  - SCRFD NMS and Umeyama match their references.
  - No per-request Triton clients.
  - No float64 reaches a tensor.
  - Fused or end2end selection is by metadata.
  - Dockerfile pins are present.
  - GPU-20..25: EXIF `orient_array` equals `exif_transpose` (8/8); `set_data_from_numpy`
    copies; the interpolation split is documented; the keepalive settings are matched;
    the `preferred_batch_size` trade-off is documented; the GPU images are digest-pinned
    and run as non-root.
- **asyncio.**
  - No bare `create_task` with a dropped reference.
  - `shield` sites keep their references.
  - Most `gather` sites are correct.
  - Exemplary: `reprocess_embed.py`, `assignment_writer.py`, `assist_loop/driver.py`,
    `promote_job.run_detached`.
- **Frontend.**
  - No `{@html}`.
  - 65 AbortController uses; search, filter and pagination are sequenced.
  - Typed inputs are debounced.
  - Timers and listeners are cleaned up.
  - No Svelte 4 legacy syntax.
  - URLs are built only in `api*.ts` (apart from FE-1).
  - The modal matches the APG pattern.
  - No localStorage for multi-user state.
- **Tests.**
  - 0 try/except blocks swallowing an assert, 0 asserts on constants, 0 xfails.
  - The self-comparisons and call-count asserts are meaningful.
  - The early returns are in `collect_ignore`d scripts or parametrized special cases.
  - vitest has no `.skip`, `.only` or un-awaited `resolves`.
- **Security.** The private notes list the areas checked and found clean.

## Appendix C: Refuted claims (NOT A DEFECT), with the reason

| Claim | Source | Why refuted |
|---|---|---|
| Combine holdout recompute overwrites concurrent edits | OS-4 | The combine target has status `building`, and `_project_deps._resolve_and_bind` returns 409 for every request to it, so no concurrent writer exists. Its missing item-error check is covered by OS-3's helper only if combine is touched; it is not scheduled. |
| ArcFace std should be 127.5, not 128 | GPU-11 | A 0.4 % input scale with no measured effect. Changing it forces a re-embed of all stored faces. Document the constant; do not change it without a measured gain. |
| Periodic loops die silently on a `for_each_project` error | asyncio R9 | `for_each_project` iterates an in-memory snapshot (no I/O) and cannot raise in practice. Per-project work is already inside `try`. The kNN warmup is a finite one-shot. |
| `BulkStore.append_done` lines interleave | asyncio R10 | `_iter_jsonl` skips torn lines and the pipeline is idempotent; an interleave only redoes images. |
| `health.py` readiness catches only `TimeoutError` | asyncio minor | Any other exception gives a non-200 `/ready`, which is fail-closed; only the message differs. |
| `get_event_loop()` inside coroutines | asyncio minor | Inside a running coroutine it returns the running loop with no deprecation warning. Style only. |
| `event_hub.append_nowait` drops tasks | asyncio R6 (part) | It keeps strong references in `_pending_writes` and prunes them via a done-callback. |
| 24 state-writing `$effect`s are a defect | FE-4 | Resetting on a prop change is the sanctioned Svelte idiom when no event handler owns the change. No failure was found. |
| "No batch read-by-ids route exists" | FE-2 premise | `GET /curation/projects/{project}/crops?ids=` exists, with exactly the semantics needed. |
| Missing `xfail_strict` | TESTS-5 | There are zero xfails in the tree, so the setting catches nothing. |
| `/detect` 503 test asserts only `!= 500` | TESTS-8 | A sibling test already asserts 503 on the same route and failure. Redundant, not a coverage gap. |
| Script-style bool-returning `test_*.py` | TESTS-9 | Excluded via `collect_ignore`, and CLAUDE.md names the files, so renaming is churn. |
| `/ocr` and `/curation` "wire contract violated" by extra keys | API-3 wording | Extra keys do not break typed consumers. The real costs are body copies and undeclared keys, so it is kept at low-medium (BA-2). |
| Process-cache leak between tests | TESTS-7 (part) | No test observes a leak. Per-process caches are legitimate under the multi-process rule. Only the missing bytes assertion was real (T-7). |

## Appendix D: Cut as churn, with the reason

| Cut proposal | From | Reason |
|---|---|---|
| Enable ruff `ASYNC` rules with a baseline | asyncio | 22 hits on this tree (11 ASYNC240, 10 ASYNC109, 1 ASYNC110), about 0 real defects. It does not detect PIL or `rmtree`, the actual defect class. The five sites are fixed by hand (CC-5). |
| `check_test_integrity.py` ratchet | TESTS-1, TESTS-7 | First-order catch count is 3 low-value hits; most of the 68 "assert under loop" hits loop over literals. |
| `xfail_strict = true` | TESTS-5 | Zero xfails exist. |
| `register_process_cache` registry plus an AST rule | TESTS-7 | It rewrites about 70 module globals for a leak no test observed. |
| Expected-skip list CI gate; adding `leidenalg`/`igraph` | TESTS-4 | New machinery; the research dependencies gate parked research. |
| Renaming script-style tests | TESTS-9 | Docs reference the names. |
| Rewriting sleeps outside the one FCFS test | TESTS-6 | No organic failure. |
| Replacing the 7 frontend source-text tests | tests (frontend) | Latent; unverified by mutation. |
| URL-userinfo and whitespace redaction | SEC-6 | Catch count 0 today. |
| Several larger security hardening options (signing, proxies, blanket capability drops, a lock file) | SEC-3..5 | Reasons in the private notes. Each was new surface or regression-prone for a latent issue. |
| Converting the three HTTP middlewares to pure ASGI | API-3 fix 3 | Large churn. The contextvar hazard is not shown to be live. |
| A pure-ASGI catch-all inside CORS for 500s | API-9 | New middleware for a dev-origin cosmetic issue. |
| `fcntl.flock` in the class registry and in `FileJob.write` | API-2, API-4, Celery P1.0/P1.3 | Owner rule 1 (no file locks). It also self-deadlocks with `update()`. |
| Dropping the startup reconcile now | API-5 | It is the crash-recovery path until the Celery Phase 4 reaper exists. |
| API-10 "baseline to 0" in one pass | API-10 | Size L. One router per PR instead. |
| Capacity cache change | API-7 | A rare create-time 409; harmless. |
| Cluster-wide lease for the probe cap | API-13 | Over-engineering; document it instead. |
| A `spawn(coro, registry)` helper plus a `create_task` ratchet | asyncio R5 | It would cement jobs in API workers; they move to Celery instead. |
| A ratchet on `gather` without `return_exceptions` | asyncio | Many sites want fail-fast; it would need a large allowlist. |
| Bounded-queue writer for `event_hub` | asyncio R6 | No reordering was observed, and cross-process SSE order is not guaranteed anyway. |
| `search_after` cursor for `audit/queue` | OS-5 | Churn for a client error; `guard_page_depth` suffices. |
| Composite aggregations | OS-10 | The legacy-index site is bounded by the cluster count; `le=` suffices. |
| Replacing scroll with PIT/`search_after` | OS-9 | `try/finally` suffices. |
| Rewriting `combine/holdout.py` | OS-4 | Not a defect (Appendix C). |
| `.tasks` doc deletion and cancel-on-timeout in the polled helper | OS-7 | Optional extras; the move alone fixes the defect. |
| TTL re-validation and `/health` reporting for GPU decode | GPU-4 | Model parameters change only on an explicit reload. |
| CI check that TensorRT still ships EfficientNMS | GPU-18 | The image and engine builds already fail on a TRT bump. |
| Removing the unit guessing in the region decoder before a box-format parameter exists | GPU-15 | Contrived trigger; the branch now decides per tensor. |
| GPU-12 as a standalone change | GPU-12 | It belongs with #407's parity gate. |
| GPU-6 change before an owner decision | GPU-6 | Behaviour change on every small image. |
| A retry library (tenacity) for the Triton retry | GPU-8 | Churn for 40 lines. |
| `OPENCV_IO_MAX_IMAGE_PIXELS` as the GPU-5 fix | GPU-5 | Optional belt only; the header check is the fix. |
| Changing the ArcFace constant | GPU-11 | See Appendix C. |
| Refactoring the 24 `$effect`s; an effect lint | FE-4 | No failure; regression risk. |
| Pager windowing | FE-6 | Unmeasured; Size L. |
| Moving `select()` out of the `[project]` `load` | FE-3 | Risks the documented prefix-before-loaders ordering; `tap` on three anchors suffices. |
| New `POST crops/get_batch` route; `allSettled` interim | FE-2 | The `?ids=` route already exists. |

## Appendix E: Not verified (be honest about these)

- **For this plan:** no test, stack, engine, npm build or browser run. Branch status
  comes from reading the diff plus one `rg` spot-check pass in the branch worktree.
  "Fixed (branch)" means the change is present, **not** that its tests were seen to
  fail first or pass here.
- **No running 32-worker compose stack was used** by the audit or the review.
  - The cross-process items (BA-1, BA-3, CC-1, CC-2) were proven with spawned
    processes and fakes.
  - CC-3's race window was argued, not reproduced.
  - CC-4 was reproduced with a minimal uvicorn app, not with the repo's job runners.
- **No Triton or TensorRT engine was run.** Engine-level claims are argued:
  GPU-14's trigger, GPU-11's drift, GPU-19's normalisation, and the GPU-7 speedup.
  GPU-2 was the only check run inside the shipped image.
- **No live OpenSearch:** OS-6's live hit count, OS-8 and `search.max_buckets`
  behaviour.
- **The frontend was static analysis only.** svelte-check, vitest and Playwright were
  not run. FE-6 is an argument.
- **Tests.** TESTS-3's mutation result is the auditor's own; the reviewer did not run
  docker. Full-suite order dependence (random order) was not checked.
- **Unmeasured:** OPS-2 (API-14) scrape cost and CC-10 (R6) in-flight peak.
- **Security:**
  - The SEC-1 positive control (a legitimate model validates) was not run by the
    reviewer; the branch added a test for it, which this plan did not run.
  - Further SEC-1/SEC-2 verification gaps are recorded in the private notes.
- **Scratch evidence** under `<scratch repro>*`, `<scratch repro>*` and `<scratch repro>` may no
  longer exist. Each red test above re-creates its scenario.
