# B1 overview: correctness foundation (five tracks, one PR, one sweep)

| Field | Value |
|---|---|
| Status | plan, not started |
| Batch / release | B1, the second of the batches B0-B4 that make up the single **v0.7.0** release (milestone `v0.7.0`). No tag, no VERSION bump, no release images and no GitHub release after B1 |
| Depends on | B0 merged to main (`01_b0_close_out.md`); the owner's approval of this plan set (owner decision 1); D-8, D-9, D-10, D-11, D-12, D-13, D-15 (approved 2026-11-03) |
| Followed by | B2 (`03_*` B2 file), then B3, B4, R-final (`06_release_acceptance.md`) |
| Decisions applied | owner decisions 1-8 of 2026-11-03: plan only now; no intermediate release; R-B1 dropped (its upgrade and migration checks move to R-final); D-16 replaced by one tag decision after R-final; breaking changes accumulate in CHANGELOG `[Unreleased]`; supported upgrade 0.6.0 -> 0.7.0 directly; local-only image tags; the branch and lane model below |
| Source IDs | the former master plan (rev 4, in git history) sections 3 (3a, 3b, 3c, 3d), 6 (6.1, 6.2, 6.3), 7 common rules, 7.1, and the B1 rows of 8, 9, 10, 11, 12, 15, 16 and Appendix A; where each went: [00_index.md](00_index.md) section 12 |
| Track files | `02a_b1_job_leases.md` (1A), `02b_b1_class_registry.md` (1B), `02c_b1_inference_search.md` (1C), `02d_b1_api_frontend_tests.md` (1D), `02e_b1_ultralytics_live_bugs.md` (1E); the deps lane (section 4.1) |
| Decisions, budget | the decision register is [00_index.md](00_index.md) section 6 (D-37 approved: B5 is inside v0.7.0); tests and GPU time follow [00_index.md](00_index.md) section 8.1 (Test and GPU budget) |

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read only, with `rg`; nothing was
run).
Consolidated 2026-11-03.

Path legend: `<repo>` is the checkout, `<repo>-wt-<name>` a sibling worktree, `<scratch>` the
host scratch directory, `artifacts_local/` the gitignored evidence directory on the owner's machine.

## 1. Scope and mission

OpenProcessor must process images fast and correctly for many users and many processes at once.
B1 fixes the correctness gaps that are live under concurrency before any job moves to Celery (B2):

| Track | What it fixes | Size |
|---|---|---|
| **1A leases** | two POSTs can no longer start one singleton twice (Q-D2); a reconcile never flips a sibling's live job (Q-D4, CC-3 cut); a blocked loop no longer costs a job its liveness (Q-D9, thread heartbeat); region clustering's 1800 s stale window ends (Q-D12); trainer and training-job claims are create-only (Q-D7 trainer part, D-15); bulk item errors stop being dropped (OS-3, Q-D10); a shrink-only ratchet stops new file locks | L |
| **1B registry** | duplicate class ids and lost classes end (CC-2, Q-D3); the authority moves to an OpenSearch document with OCC; an upgrade migration that fails closed; rollback keeps the classes | L |
| **1C inference and search** | sliver crops and huge JPEGs stop allocating GBs (GPU-9, GPU-5); deterministic errors are not retried (GPU-8); sync Triton calls time out (GPU-16); residency learns of an unload (GPU-17); filtered kNN returns `k` hits (OS-6); paging and offsets are bounded (OS-5, OS-10); scroll and PIT work stops leaking (OS-9, CC-6); OS-7, GPU-4, GPU-10, CC-8; the B0 open choice on a shared OCR detector-size helper (1C-0); legacy `/search/object` stops blocking the loop and stops querying the whole-image index with a box vector (1C-15) | M |
| **1D API, frontend, tests** | per-process caches stop serving stale vocabularies and cached errors (BA-1, BA-3, BA-4); undeclared body keys and leaked class names go (BA-2, BA-5); BA-8 documented; thumbnail and render handlers stop blocking the loop (CC-5 handlers); FE-1, FE-2, FE-3, FE-5, FE-7; T-1, T-2, T-3, T-4, T-7, T-8; SEC-3 and SEC-6 (neutral; details in the maintainers' private notes) | M |
| **1E Ultralytics live bugs (minimal, D-10)** | #447 (YOLO26 export `nms=False` + head check, promote refuses a wrong head); U-C2 upload routing by head kind; U-C10 secondary detector class by name; 1E-4b primary detector class by name; the YOLOv5 objectness probe input from the ONNX (D-11 option b); CI job `families` | S-M |
| **deps lane** | the open dependabot PRs re-applied by hand, one commit each (#434, #433, #423) and the dependabot `ignore` fixes (ENV-12); a separate lane so no track carries dependency churn | S |

Not in B1: Celery and everything that moves jobs (B2); the Ultralytics version move, the trainer's
`nms`, the export module, the full U-6.1 matrix and GPU-14 (B3); the API worker knob, bulk OCC, the
event log and the last file locks (B4); speed (B5, inside the same v0.7.0 release). Evidence-first items (OS-8, GPU-19,
FE-6, T-6) and BA-6 stay in the backlog (`11_risks_cleanup_host.md`).

## 2. What the 2026-11-03 decisions change against the former master plan (rev 4, in git history) 7.1

| Master plan said | Now |
|---|---|
| B1 is released as v0.8.0 "early" after R-B1 | B1 is not a release. v0.7.0 = B0-B5 + R-final (D-37, [00_index.md](00_index.md) section 6); v0.8.0 = platform breadth |
| R-B1 acceptance subset (installer_upgrade from 0.6.0 and 0.7.0, class/holdout/label_ops/ingest/cluster/export/train/promote phases, `tests/live`, `npm run test:live`, image audit, Trivy, screenshots) | dropped. Its checks move to R-final (`06_release_acceptance.md`), including `installer_upgrade` from a 0.6.0 install with real class registries. B1 only changes the acceptance phase code (1B-12) |
| D-16 "tag v0.8.0 after R-B1" | replaced by one tag decision after R-final (`06_release_acceptance.md`) |
| "Track 1B may slip into its own follow-up PR before the 0.8.0 tag if D-9 is late" | obsolete: D-9 is approved; 1B ships in the B1 PR |
| "0.6.0 -> 0.8.0 and 0.7.0 -> 0.8.0 are supported upgrades" | the supported upgrade is 0.6.0 -> 0.7.0 directly (0.7.0 never shipped); the registry migration handles a 0.6.0 install (same layout, verified with `git show v0.6.0:src/config/projects.py`) |
| CHANGELOG "Breaking" section for the 0.8.0 release | breaking entries accumulate under `[Unreleased]` (section 12 lists them); the release commit in R-final turns `[Unreleased]` into `[0.7.0]` |
| "Commit the source plans as the first commit of `int/b1-correctness`" | done before B1 (D-14) on `fix/411-r2-round`, which B0 merges to main |
| R-B1's image hygiene (image-vs-git audit, Trivy) | B1 publishes no image. The audit and Trivy run in R-final, and before any image push |
| Live harness "on GPU 0" | the live harness (`docker/test/compose.yml`) has no Triton and needs no GPU; B1 has no GPU gate at all |

## 3. Preconditions

1. The owner approved this plan set (owner decision 1). No B1 code before that.
2. B0 is merged to main with green CI (`01_b0_close_out.md`). B1 is cut from that main.
3. The host is free for heavy runs: check `uptime`, check peer sessions before the sweep
   (`11_risks_cleanup_host.md`), and announce heavy runs.
4. Every lane has its own worktree and venv; the coordinator has a worktree for the integration
   branch.

## 4. Branch and worktree model (owner decision 7)

- Integration branch: `int/b1-correctness`, cut from main after the B0 merge:
  `git worktree add <repo>-wt-b1-int -b int/b1-correctness main`.
- One lane per track plus the deps lane, one agent per lane, each in its own worktree on its own
  branch, all cut from `int/b1-correctness` at the same commit:

| Track | Branch | Worktree | Compose project (if any) |
|---|---|---|---|
| deps | `b1/deps` | `<repo>-wt-b1-deps` | none (no Docker) |
| 1A | `b1/leases` | `<repo>-wt-b1-leases` | `op-b1-leases` |
| 1B | `b1/registry` | `<repo>-wt-b1-registry` | `op-b1-registry` |
| 1C | `b1/inference` | `<repo>-wt-b1-inference` | `op-b1-inference` |
| 1D | `b1/api-frontend` | `<repo>-wt-b1-api-frontend` | `op-b1-api-frontend` |
| 1E | `b1/ultralytics` | `<repo>-wt-b1-ultralytics` | `op-b1-ultralytics` |

- Lanes run only their red/green tests plus static checks on touched files
  (`.venv/bin/pre-commit run --files ...`). No full suite, no `--all-files`, no push, no merge, no
  other checkout. One commit per task, hooks on, never `--no-verify`.
- Opus planned this; Sonnet implements each lane and Sonnet reviews each lane's diff before the
  coordinator merges it (a review agent gets the diff and the track file, not the coordinator's
  conclusion).
- The coordinator merges lanes into `int/b1-correctness` with `git merge --no-ff`, one merge commit
  per track, in the order of section 5, and runs exactly one full sweep for the batch (section 7).
  Test tiers follow [00_index.md](00_index.md) section 8.1: T0 per task, one offline sweep per
  wave, no per-lane suites.

### 4.1 Lane b1/deps (dependabot PRs, separate from every track)

| Field | Value |
|---|---|
| Branch / worktree | `b1/deps`, `git worktree add <repo>-wt-b1-deps -b b1/deps int/b1-correctness` |
| Size | S |
| Rules owned elsewhere | package content: [08_environment_and_setup.md](08_environment_and_setup.md) section 7.2 and ENV-12; sequencing, close comments and the ready-to-paste brief: [09_github_actions.md](09_github_actions.md) section 8 |

The PRs, in commit order (one commit each, subject `build(deps): <what>`, body `Refs #<n>`):
1. ENV-12: `.github/dependabot.yml` `ignore` entries (including `onnxruntime-gpu`, whose cap widening
   in #434 is rejected) and the corrected `ultralytics` reason.
2. #434 pieces: ruff together with the `ruff-pre-commit` hook rev; mypy; pydantic; pytest
   (`frontend/e2e`); then fastapi alone with `make contracts` (`test_the_image_pins_fastapi_as_the_tests_do`
   keeps image and tests equal).
3. #433 (frontend dev deps: adapter-static, kit, `@types/node`, knip).
4. #423 (`node` and `nginx-unprivileged` digests).
#432 and #422 are closed, not applied (D-38 in the register).

Rules: T0 only. After each commit run the targeted checks the 09 brief lists (the requirements
contract tests, `make contracts-check` for fastapi, the ruff and mypy hooks only, the frontend set
for #433). No full suite, no Docker, no GPU. Never touch the `ultralytics` pin, the TensorRT pins or
the Triton and Python base tags. New ruff or mypy findings are fixed in code in the same commit;
baselines never grow. #433 and #434 do **not** go into the 1D lane: 1D owns no dependency changes.

Merge: first (section 5), so every track rebases onto a stable toolchain.

Done when: the four steps above are committed; `git diff int/b1-correctness...b1/deps --stat` shows
only manifests, `.github/dependabot.yml`, the lockfile, regenerated `contracts/` (fastapi commit) and
the code fixes that new lint findings forced; the 09 close comments can name the commit SHAs.
- Every agent brief (the brief at the end of each track file) forbids `pgrep -f`, `pkill -f` and
  sleep/until loops, and names its compose project, its GPU (none in B1), its worktree and its
  deadline.

## 5. Waves, parallelism and merge order

**One wave, six lanes in parallel (the deps lane and five tracks).** The deps lane, 1E, 1C, 1D and 1A are independent. **1B also starts at once,
in parallel, and rebases onto `int/b1-correctness` after 1A is merged.** Why parallel rather than
"1B on top of merged 1A":
- 1B's bulk (the store, about 48 reader files, the migration script, the CLI and installer hooks)
  does not use any lease code.
- The only shared lines are the two object keys in `_configs_body()` / `_settings_body()` and the
  bootstrap migration. 1A owns both keys (`class_registry` included, task 1A-1); 1B never edits
  those lines, so its rebase has no conflict there.
- On folded projects (`configs` index, `dynamic: false`) the document works before the mapping key
  exists, so 1B's offline tests do not wait for 1A.
- The other overlap is small: 1A changes the claim lines of `combine/service.py`,
  `combine/execute.py` (fencing) and `dataset_import/runner.py`; 1B changes the registry calls in
  `combine/execute.py`, `combine/plan.py` and `dataset_import/prepare.py`. A rebase resolves it.
- Serial 1A then 1B would add about the length of 1B (L) to the batch.

**Merge order into `int/b1-correctness`: deps, 1E, 1C, 1D, 1A, 1B** (the same order as the W1 row in [00_index.md](00_index.md) section 4).
- deps first: the toolchain bumps (ruff, mypy, fastapi) change what every other lane's hooks say.
- 1E, 1C and 1D are small and independent; merging them first gives 1A and 1B a stable base.
- 1A before 1B (mapping keys; `combine` and `dataset_import` overlap).
- 1D before 1B (BA-4 edits `eval_datasets.py`, which 1B converts; 1E-4 edits `ingest_detect.py`,
  which 1B converts).
- After each merge the coordinator runs that track's targeted tests on the integration branch
  (minutes) and nothing else; the full sweep runs once after the last merge.

**Revert coupling:** 1B depends on 1A (mapping key, bootstrap migration). 1C, 1D and 1E revert
independently. To drop 1A, revert 1B first.

**Shared files the coordinator resolves at merge time:**
- `src/schemas/curation/config_common.py::ErrorCode` (1A may add busy codes, 1B adds
  `registry_not_migrated`, `registry_full`, `registry_conflict`): union the members, then
  `make contracts` and commit the regenerated `contracts/` in the merge commit.
- `CHANGELOG.md` `[Unreleased]` and `frontend/CHANGELOG.md`: keep every entry.
- `tests/conftest.py::_PROCESS_CACHES` (1D adds entries; 1B may add the registry snapshot cache).
- `CLAUDE.md` (1A lock rule, 1B CLI subcommand, 1E offline-suite marker).

## 6. Coordinator procedure

1. Create `int/b1-correctness` and the six lane worktrees (section 4).
2. Start the six lanes with the briefs at the end of each track file and the deps brief in
   [09_github_actions.md](09_github_actions.md) section 8 (fill in the deadline).
3. When a lane reports done: a Sonnet reviewer reads the lane diff against its track file; fix
   findings in the lane (same agent or a follow-up lane agent); then merge in the section 5 order.
4. Before merging 1B: tell the 1B lane that 1A is merged; it rebases and re-runs its targeted tests.
5. After the last merge: the sweep (section 7). Failures are fixed on the lane branch that owns the
   code (or a short `b1/sweep-fixes` lane), merged `--no-ff`, and only the failed steps re-run.
6. Push `int/b1-correctness` only after the sweep is green and the owner says yes; verify the push
   with `git ls-remote` and the CI run (never a piped exit code); open the PR against main; merge
   commit, never squash, on the owner's yes.
7. After the merge: remove the lane and integration worktrees and branches (cleanup rule).

## 7. The single B1 sweep (on `int/b1-correctness` after all five merges)

Announce it to peer sessions first; one heavy run at a time; check `uptime` and cap parallelism.

| Step | Command (from `<repo>-wt-b1-int`) | Pass |
|---|---|---|
| 1 | `.venv/bin/pre-commit run --all-files` (ruff, mypy, bandit, ratchets incl. the new `check_file_locks.py`, naming-leak scan, contract drift) | green |
| 2 | hook stages beyond pre-commit: `.pre-commit-config.yaml` has a `commit-msg` hook (exercised by every commit) and, checked on `af62c947`, no `pre-push` stage; if one exists by then, run `.venv/bin/pre-commit run --hook-stage pre-push --all-files` | green |
| 3 | `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live and not families' -n 8` (heavy; capped at 8 on the shared host) | green |
| 4 | spawn races 10x: `-p no:xdist` run of `tests/test_job_lease_cross_process.py`, `tests/test_class_registry_cross_process.py`, `tests/test_atomic_write_concurrent.py`, `tests/curation/trainer_protocol/test_job_claim_generations.py` (each parametrised over 10 repetitions) | green every time |
| 5 | the T-8 ordered run (part of step 3 via its subprocess test; also run once by hand with `-p no:xdist`) | green |
| 6 | `make contracts-check`; `.venv/bin/python scripts/docs/check_docs_vs_code.py`; the changelog rules as CI runs them (root and frontend) | green |
| 7 | frontend full set in `frontend/`: real `npm ci`, `npm run check`, `npm run lint`, `npx knip`, `npm test`, `npm run build`, then `npm run test:e2e` (heavy; never in parallel with step 3 or 9) | green |
| 8 | browser check at 1440 and 430 px, full-page screenshots actually viewed: `/p/<slug>/export` (FE-1 chips refresh), `/review` with the diverse overlay (FE-2, one request), the classes page (1B: create, rename, hotkey, merge on the document), a project in the `registry_not_migrated` state (the 409 message is readable), the `/train` dataset section (FE-5); headed runs use the desktop session's display | screenshots saved under `artifacts_local/b1/screens/` and viewed |
| 9 | live harness: a **short smoke check, cap 30 minutes** (the budget in [00_index.md](00_index.md) section 8.1 replaces the old 90-minute run; the broad live coverage of B1 comes with B2's single live run, which covers B1 and B2 together): `docker compose -p op-b1-live -f docker/test/compose.yml up -d --wait` with the API image built from this branch under a **local-only** tag (never a public tag: watchtower), then only the lease and registry live tests of 7.1, then `down -v --remove-orphans` | see 7.1 |
| 10 | the `families` tests once locally in the export image built from this branch (local-only tag): `docker run --rm --cpus 8 -v <repo>-wt-b1-int:/src -v ~/.cache/op-test-weights:/root/.cache/op-test-weights -w /src <export image> python -m pytest tests -q -m families --no-cov` | green, under 15 minutes |
| 11 | CI on the pushed branch, including the new `families` job (after the owner's yes to push) | green |

No GPU window in B1. GPU: none. The step list covers the lanes' tasks including 1C-15 (the
`/search/object` fix is exercised by its offline tests in step 3) and 1B-12b (step 6 of the sweep
adds `acceptance_run.sh --list` naming `dataset_import` and `combine`; their live run is R-final
RF-70 and RF-71).

### 7.1 Live harness contents (step 9, smoke check only, cap 30 minutes)

- Not run here: the full `tests/live` set. It runs once at the end of B2 for B1 and B2 together
  (`03_b2_overview.md`).
- `tests/live/test_job_lease_live.py`: the 8-process lease race against the harness OpenSearch,
  reduced to 5 repetitions to fit the cap; record p50 and p99 acquire latency in the sweep report.
- `tests/live/test_class_registry_live.py`: the store-level 8-process race (5 repetitions) and one
  HTTP sequence (the harness API runs `--workers=1`).
- OS-6 filtered-kNN hit count: build the visual-search body with the 1C code and run it against a
  `verify_` index seeded with fixed vectors and a restrictive class filter; record hits before
  (bool post-filter body) and after (efficient filter): after returns `k`. This is a filter
  mechanics check, not a quality claim.
- Blocked-loop check: start an API-hosted job (an assist run against the harness fake VLM), then
  post a dataset upload large enough to block the event loop for at least 30 s during extract;
  the job's lease document's `heartbeat_at` keeps advancing every 15 s and the job keeps its lease.
- Snapshot `docker ps` and the volumes before and after; nothing outside `op-b1-live` is touched.

## 8. Done when (batch)

- Every task's done-when in the five track files holds.
- `rg -n "exclusive_(file|start)_lock\(" src | rg -v job_lock.py` lists only
  `bulk_ingest/dispatcher.py` and `open_vocab_sweeper.py`; `rg -n "fcntl" src` lists only
  `job_lock.py`, `source_image_cache.py` and `event_hub.py`; the file-lock baseline lists only the
  B2/B4 rows and `job_lock.py`.
- No `class_registry.json` read outside export snapshots, the trainer, the training pin, the
  migration and the `registry_not_migrated` check; `registry_not_migrated` is tested.
- The sweep (section 7) is green, CI is green, the PR is merged with a merge commit on the owner's
  yes.
- The B1 plan files are retired per [00_index.md](00_index.md) "Plan lifecycle": the closing PR
  moves the keep-list (section 8.1) to its target docs, then deletes `program/02_*.md` through
  `02e` and removes the B1 rows of `00_index.md` and `docs/design/README.md`.

### 8.1 What to keep after merge (mechanical move before the plan files are deleted)

| Fact or section | Target doc |
|---|---|
| lease design as built: document shape, TTL 120 s and self-fence at 90 s, thread heartbeat, fencing generation, kinds and scopes (02a sections 5-6) | `docs/ARCHITECTURE.md` |
| class registry as an OpenSearch document: OCC, `registry_full` bound, migration and reverse step, `registry_not_migrated` (02b sections 5-6) | `docs/ARCHITECTURE.md`, `docs/opensearch_schema_design.md` |
| new error codes and 409/422 semantics (`registry_*`, busy codes, `object_class_not_indexed`) | `docs/design/curation_api_contract.md` |
| lease p50/p99 acquire latency and the OS-6 hit counts | `docs/PERFORMANCE.md` |
| upgrade 0.6.0 -> 0.7.0 steps (section 11) and the breaking list (section 12) | `INSTALLATION.md`, `CHANGELOG.md` `[Unreleased]` |
| file-lock ratchet rule and the `families` CI job | `CONTRIBUTING.md`, `CLAUDE.md` |

## 9. CI job `families` (owned by 1E, `02e_b1_ultralytics_live_bugs.md` task 1E-6)

- Its own job in `.github/workflows/ci.yml`, with network, CPU, `timeout-minutes: 15`.
- `actions/cache` keyed on the Ultralytics pin (hash of `requirements-test.txt`) and the fixture
  list; the original-YOLOv5 ONNX produced once by the official `ultralytics/yolov5` `export.py` at a
  pinned commit, in its own venv.
- Runs `pytest -m families`; the main `pytest` job selects `-m 'not live and not families'`.
- Skips with an explicit reason only when offline locally; never skips in CI (`CI` is set).
- B3 extends this job to the full U-6.1 matrix; it is not rewritten.

## 10. Rollback

- Before the PR merges: revert a track's merge commit on `int/b1-correctness`
  (`git revert -m 1 <merge>`); 1B before 1A.
- After the PR merges (before the v0.7.0 tag): revert the PR's merge commit on main, or one track's
  merge inside it.
- 1B after users migrated: the installer rollback runs the reverse registry step (Q-1B-1); a manual
  downgrade follows the note in `02b_b1_class_registry.md` 5.5 item 4. The document stays.

## 11. Upgrade impact (0.6.0 -> 0.7.0, the supported path)

- The registry migration runs from `openprocessor upgrade` (and `repair`) after the stack is healthy:
  dry-run saved under `.install/registry_migration/`, then the real run; until it succeeds, class
  routes of an unmigrated project answer 409 `registry_not_migrated` (fail closed) and inference
  keeps working.
- The two mapping keys (`lease`, `class_registry`) are applied additively by the bootstrap on first
  start (the schema fingerprint changes, so each project re-runs its migrations once).
- Lease documents appear on first start of each job kind. In-flight in-API jobs die with the
  container recreate, as on any upgrade; resumable ones show `interrupted` (lease mid-upgrade rule).
- Queued export jobs without `head_kind` fail closed ("re-upload the model").
- The trainer claims jobs create-only (`g1` for a job file written by 0.6.0).
- R-final (`06_release_acceptance.md`) proves this path from a real 0.6.0 install with classes.

## 12. Breaking-change list for CHANGELOG `[Unreleased]`

Add each as a `**Breaking: ...**` bullet under `[Unreleased]`, in the subsection layout the release
file fixes (`06_release_acceptance.md`; create `### Breaking` if the owner's layout needs it and it
does not exist yet):

| Track | Breaking change |
|---|---|
| 1A | OpenSearch must be reachable to start a curation job (503 otherwise); a crashed job blocks a new start of the same kind for up to 120 s (was 30 s); a label operation whose writer died is shown `interrupted` once its lease expires (up to 120 s) instead of at once |
| 1B | the class registry moves from `class_registry.json` to an OpenSearch document per project; `openprocessor upgrade` migrates it; until then class routes answer 409 `registry_not_migrated`; the file is kept as `class_registry.json.migrated`; per-write registry file backups end; rollback writes the registry back to the file; new 409 codes `registry_full`, `registry_conflict` |
| 1C | none breaking; `/embed/image` resampler change (Changed); 422 instead of 503 on out-of-range `page` / `offset` (Fixed) |
| 1D | 500 and Triton 503 bodies no longer carry `error_type`; curation JSON bodies no longer carry `total_time_ms` / `request_id` (the `X-Process-Time` and `X-Request-ID` headers remain) |
| 1E | uploads route by head kind (YOLOv10 now uses the NMS-free path); an export or promote whose head is not `[batch, 300, 6]` is refused; export jobs queued before the upgrade must be re-uploaded |

## 13. Docs touched (each track file lists its own; `check_docs_vs_code.py` runs in the sweep)

| Doc | Tracks |
|---|---|
| `docs/ARCHITECTURE.md` | 1A (leases, heartbeat thread), 1B (registry document) |
| `docs/CURATION.md` | 1A (busy window, codes, fencing residual), 1B (`registry_not_migrated`, migrate command, `registry_full`, `registry_conflict`), 1C (cluster list `offset` bound), 1D (probe cap per process), 1E (probe input) |
| `docs/opensearch_schema_design.md` | 1A, 1B (lease and registry docs, the two mapping keys) |
| `INSTALLATION.md` | 1A (multi-host clock note), 1B (upgrade runs the migration, rollback reverse, manual downgrade note, CLI subcommand) |
| `docs/design/curation_api_contract.md`, `contracts/` | 1A (busy codes, region-cluster job shape if changed), 1B (error codes), 1C (OS-10 `offset`), 1D (BA-5 if any schema lists `error_type`; none today) |
| `CLAUDE.md` | 1A (lock rule names the lease and the ratchet), 1B (`curation migrate-registry`), 1E (offline command `-m 'not live and not families'`) |
| `CONTRIBUTING.md` | 1A (file-lock ratchet), 1E (offline command) |
| `ATTRIBUTION.md` | 1D (cachetools MIT; einops if test deps are listed) |
| `scripts/README.md`, `docs-site/docs/operations/curation-workflow.mdx`, `docs-site/docs/guides/use-your-own-domain.mdx` | 1B (seed and migration scripts) |
| `docs-site/docs/configuration/advanced.mdx`, `export/README.md` | 1E |
| `frontend/CLAUDE.md` | 1D (FE-5 fallback, if taken) |
| `CHANGELOG.md`, `frontend/CHANGELOG.md` | all tracks |

## 14. Risks (B1 rows of the former master plan section 15, updated; the full register is [11_risks_cleanup_host.md](11_risks_cleanup_host.md))

| # | Risk | Mitigation |
|---|---|---|
| M4 | a lease holder pauses longer than the TTL | thread heartbeat, TTL 120 s, self-fence at 90 s, fenced terminal writes, `lease_generation` |
| M5 | OpenSearch outage blocks every job start | fail closed with 503; `/health` reports OpenSearch |
| M6 | lease doc lost with a project delete while a job runs | next renew is `Lost`; the job stops |
| M7 | registry migration of the owner's real projects | dry-run first, idempotent, ids preserved, rename not delete, fail-closed 409, reverse on rollback; R-final runs it from a real 0.6.0 install |
| M8 | a reader still reads `class_registry.json` | the 1B-7 grep in done-when; dead-code and layering ratchets |
| M19 | contract drift between tracks | `make contracts` per commit; `ErrorCode` union and regeneration at merge; contracts-check once in the sweep |
| M20 | source-plan line numbers moved | every site is re-found by symbol or `rg`; verified on `af62c947` |
| M21 | one track fails the sweep | one merge commit per track; revert that track (1B before 1A) and ship the rest |
| M22 | a project with more than 10,000 classes | 409 `registry_full`; per-class docs would be a separate design |
| M24 | Cropwright changes in B1, B2 and B4 | they all ship once in v0.7.0 now; `frontend/CHANGELOG.md` accumulates |
| M25 | the 0.6.0 -> 0.7.0 upgrade misses a step | `installer_upgrade` from a real 0.6.0 install with classes in R-final |
| new | the single XL batch takes long with five parallel lanes on a shared host | lanes run no full suites; one sweep; announce heavy runs |
| new | T-3 fails on a fresh checkout's required env vars | run the three files on the CI runner config before merging (1D-17) |
| new | BA-5 removes `error_type` that a client reads | `rg error_type tests frontend/src contracts` first; Breaking entry |
| new | rollback after migration serves an empty registry on the older version | reverse step (Q-1B-1) |

Honest unknowns: the lease cost per tick and behaviour under OpenSearch load are argued, not
measured; the live test measures them. Nothing in this plan was run.

## 15. Issues (GitHub actions are in `09_github_actions.md`)

| Issue | Track | Milestone (after the renumbering) |
|---|---|---|
| #447 YOLO26 raw head (re-scoped: also U-C2 routing) | 1E | v0.7.0 |
| N2 class registry document with OCC | 1B | v0.7.0 |
| N3 job leases replace start file locks | 1A (ratchet reaches zero in B4) | v0.7.0 |
| N4 per-process caches, bodies, blocking image work | 1D | v0.7.0 |
| N5 inference and search hygiene | 1C (OS-3 in 1A) | v0.7.0 |
| N6 frontend batch read, refresh, SSE resync, tests that cannot fail | 1D | v0.7.0 |
| N7 secondary class by name, original-YOLOv5 ONNX for probe and secondary | 1E | v0.7.0 |
| N9 (SEC-3), N10 (SEC-6), neutral | 1D | v0.7.0 |

The B1 PR references these issues; they close at the v0.7.0 tag (not at the B1 merge), per
`09_github_actions.md`.

## 16. Where each item of the former master plan (rev 4) landed in B1

| Master plan item | Landed in |
|---|---|
| 3a Q-P1.0 (`_state_lock` deleted in B1) | 1A-5 |
| 3a Q-P1.1, Q-P1.2 (lease claims; four unlocked starts; region clustering to `FileJob`) | 1A-3, 1A-6, 1A-7 |
| 3a Q-P1.3 (registry) | 1B |
| 3a Q-P1.5 (cut; reconcile by takeover) | 1A-5 |
| 3a Q-P1.6 (trainer and API create-only claims; export worker stale-only; `claim_trigger` rewrite cut) | 1A-10, 1A-11 (the cut stays cut: B2 Phase 5) |
| 3a Q-P1.7 (OS-3) | 1A-12 |
| 3b B1 rows (14 start sites, label_ops, promote_json, `_state_lock`) and the ratchet | 1A-6, 1A-8, 1A-9, 1A-5, 1A-13 (counts corrected: 55 hits, 19 files) |
| 3b B2/B4 rows (dispatcher, sweeper, cache prune, event hub, `job_lock.py`) | stay in the ratchet baseline; B2/B4 files |
| 3c Q-D2, D3, D4, D7 (trainer), D9, D10 (OS-3), D12, D14 (BA-1, BA-3, BA-4, BA-8) | 1A, 1B, 1A, 1A-10, 1A-4, 1A-12, 1A-7, 1D |
| 3d superseded tests (`test_job_claim_cross_process.py`, `test_reconcile_sibling_job.py`, file-lock `test_class_registry_cross_process.py`, trainer claim test) | replaced by `test_job_lease_cross_process.py` + `tests/live/test_job_lease_live.py` (1A-3, 1A-14), `test_job_lease_takeover.py` (1A-3), the OCC rewrite of `test_class_registry_cross_process.py` (1B-1), `tests/curation/trainer_protocol/test_job_claim_generations.py` (1A-10, path corrected) |
| 6.1 J-LEASE (where, shape, code, operations, kind/scope, rules, traps, tests) | 1A sections 5 and 6 |
| 6.2 J-REG (authority, size bound, mutators, readers, file, upgrade path, mirror, async, tests, #365 interplay) | 1B sections 5 and 6 |
| 6.3 trainer create-only generations | 1A-10 |
| 7 common rules (branch model, agents, red/control, contracts, line numbers, baselines only shrink, docs gate) | sections 4-7 here and every track file |
| 7 common rule "commit the source plans first" | done (D-14); section 2 |
| 7.1 dependencies, tracks, 1E tests and CI, merge order, revert coupling, test cycle, done-when, size, rollback, upgrade impact, docs, risks | sections 3-14 here and the track files |
| 7.1 "Release: R-B1, then v0.8.0" | dropped; checks moved to R-final (section 2) |
| 8 D-8..D-12, D-15 (approved), D-16 (replaced) | header and section 2 |
| 9.1 N2-N7, N9, N10; 9.2 #447 re-scope | section 15; created by `09_github_actions.md` |
| 10 issue map B1 rows (#447) | section 15 |
| 11 task list rows B1-0, B1-1A..1E, R-B1 | B1-0 done (D-14); tracks here; R-B1 dropped |
| 12 T2 B1 row (live 90 min), CI `families` row, image hygiene for R-B1 | section 7, section 9, section 2 (image hygiene in R-final) |
| 15 M4-M8, M19-M25 | section 14 |
| 16 B1 docs row | section 13 |
| B0 open choices: shared OCR det-size helper; `autolabel/job.get_state` `_fail_unclaimed`; pyclipper in the hardened Triton image | 1C-0; left for B2 (1A-7 note); B2 (not B1) |
| Appendix A items for B1: W-3 (lease index), W-4 (O_EXCL "added"), W-5 (done-when), W-9 (flock count), C-2 (one PR, merge per track), C-4 (TTL and thread heartbeat), C-5 (stale reclaim race), C-7 (trainer change in B3), R4 items 8-16 and 18, R5 items 2-4, R6 fencing and `nms`-only tests, R7 edits 4-8 and 15 | 1A (W-3, W-4, W-9, C-4, C-5, R4 8-11, 18, R6 fencing), 1B (R4 12-14), 1D (R4 15), 1E (C-7, R4 16, R6 `nms`), this file (W-5, C-2, R5 2-4). R5 item 3 (twelve start paths edited twice, B1 claim lines and B2 launch lines) stays an accepted cost |

## 17. Corrections to the former master plan's references (found while verifying)

- 3b: 55 hits in 19 files (not 51 in 18).
- 6.1 kind table: `combine` is scoped per combine job id and lives in the target project's index;
  `promote_json` needs its own short lease because the sharing route is a second writer.
- 6.2: `keymap.py` writes the registry directly twice; `projects/clone.py` copies the registry
  file; `training/jobs.py` pins a registry snapshot; four readers read another project's file;
  `add_class` cannot be idempotent by name without breaking `POST /classes`'s 409 (split into
  `add_class` and `ensure_class`); rollback leaves an older version without its registry file
  unless a reverse step runs.
- 6.3: trainer tests live in `tests/curation/trainer_protocol/`, not `tests/trainer/`; the export
  worker has no per-task heartbeat yet.
- 7.1 1E: the `export` pytest marker is not registered (`--strict-markers`); one marker `families`
  is used. The ingest primary `assigns_class` path also maps by model index (fixed by 1E-4b).
- 7.1 test cycle: the live harness needs no GPU and runs the API with one worker, so the registry
  "two-worker" live scenario runs at the store level.
- 1D: BA-5's `error_type` is also in the Triton 503 body and asserted by two tests; FE-5's file is
  `frontend/src/lib/components/train/TrainDatasetSection.svelte`.
- The `ultralytics==8.4.197` pin (D-2) is already in `requirements.txt` and `requirements-test.txt`
  on `fix/411-r2-round`.
- `setup-openprocessor.sh` is at its file-size baseline (3112 lines); the 1B hooks must make room by
  moving a cohesive function into `scripts/lib/`.

## 18. Open questions

| ID | Question | Recommendation | File |
|---|---|---|---|
| Q-1B-1 | rollback after the registry migration: automatic reverse step, or manual note only | automatic reverse step | `02b_b1_class_registry.md` section 14 |
| Q-1E-1 | CLOSED: the primary `assigns_class` detector path maps by name too (task 1E-4b) | decided, not a question | `02e_b1_ultralytics_live_bugs.md` |

Owner decisions still open elsewhere (see [00_index.md](00_index.md) section 6) do not block B1.

## 19. Coordinator brief (paste as is)

```
You coordinate B1 (correctness foundation) of OpenProcessor.
Plan: docs/design/program/02_b1_overview.md (this file) and the five track files 02a-02e, plus
the deps lane (section 4.1; brief in 09_github_actions.md section 8).
Read first: CLAUDE.md, docs/VISION_AND_GOALS.md, docs/design/program/00_index.md,
docs/design/program/11_risks_cleanup_host.md (host, GPUs, peers).

Preconditions: the owner approved the plan set; B0 is merged to main with green CI. Do not start
otherwise.

1. git worktree add <repo>-wt-b1-int -b int/b1-correctness main; then create the six lane
   worktrees and branches (deps + five tracks) from int/b1-correctness (section 4 table).
2. Launch six Sonnet lane agents in parallel with the brief at the end of each track file and the
   deps brief (fill in the deadline). They never push, merge or check out another branch.
3. As lanes finish: a Sonnet reviewer reads each lane's diff against its track file (give it the
   diff and the file, not your conclusion); fixes go back to the lane.
4. Merge with git merge --no-ff in the order deps, 1E, 1C, 1D, 1A, 1B; after 1A, tell the 1B lane to
   rebase onto int/b1-correctness and re-run its targeted tests. Resolve ErrorCode, CHANGELOG,
   conftest and CLAUDE.md overlaps at merge; run make contracts and commit contracts/ in the merge
   commit when ErrorCode changed. After each merge run only that track's targeted tests.
5. Run the sweep once (section 7), announced to peer sessions, capped (-n 8), one heavy step at a
   time, live smoke check (30 min cap) under -p op-b1-live with local-only image tags, no GPU. Save the sweep
   report and screenshots under artifacts_local/b1/.
6. Fix sweep failures in the owning lane (or b1/sweep-fixes), merge --no-ff, re-run only the
   failed steps.
7. Report to the owner and wait. Push only on the owner's yes; verify with git ls-remote and the
   CI run (including the families job); PR against main; merge commit on the owner's yes.
   No tag, no VERSION bump, no release images.
8. After the merge remove the B1 worktrees and branches. In the closing PR, move the section 8.1
   keep-list to its target docs and delete the B1 plan files (00_index.md "Plan lifecycle").

Never use pgrep -f, pkill -f, or sleep/until loops; wait with the tool's background
notification or healthchecks. Always pass -p <name> to docker compose; never touch a peer
session's containers, volumes or networks.
```
