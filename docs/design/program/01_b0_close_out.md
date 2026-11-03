# B0: close out the R2 round and merge it to main

Status: plan (the code is done; commit of this plan set, the CPU gate cycle, push, PR and merge
remain).
Depends on: owner approval of this plan set; the host all-clear from the OpenTranscribe release
session (section 4).
Decisions applied: D-1 (B0 scope, minus release prep), D-2 (exact Ultralytics pin, done), D-3
(gates, push, CI, merge commit; R3 deferred into R-final), D-5 (#42 never blocks), D-6 dropped,
owner decisions 1-3 and 7-9 of 2026-11-03 ([00_index.md](00_index.md), decision register).
Source IDs: GPU-1, GPU-3a, GPU-13, GPU-2, OS-2, OS-1, OPS-1, CC-1, BA-7 (`codebase_audit_plan.md`
section 4); U-J2, U-C4 (`model_family_versions_plan.md`); Q-D1, Q-D10, Q-D11, Q-D13
(`task_queue_plan.md`); ledger sections 3, 14, 16-18 (private).

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03, plus the pin commit `d1c6bab0`
that landed on the branch the same day, and the docs fixes `ad553ecd` and `0cc2b3ac`.
Consolidated 2026-11-03.

## 1. Goal

Get everything on `fix/411-r2-round` onto `main` in one push, behind one CPU gate cycle (no GPU
run: [00_index.md](00_index.md) section 8.1), with a merge commit. **No VERSION bump, no tag, no
release images, no GitHub release** (owner decisions 2 and 3, D-37: the tag comes only after
every plan is implemented and tested). B0 ends when `main` holds the branch and the worktree is
cleaned up. The next step is B1 ([02_b1_overview.md](02_b1_overview.md)).

Done on the branch (not repeated in the task list): `d1c6bab0` (exact pin `ultralytics==8.4.197`
in `requirements.txt`, `requirements-test.txt` and `pyproject.toml`, plus a test in
`tests/test_requirements_contract.py`); `ad553ecd` (the stale "patch breaks on 8.4" comments
corrected in the `Dockerfile`, `requirements-export-y11.txt` and `.github/dependabot.yml`);
`0cc2b3ac` (the docs checker no longer applies its route and env checks to `docs/design/`; it
was failing on the committed plans).

Remaining: commit this plan set (with removal wave R-1, B0-1b), the CPU gate cycle once the host
is free, push, PR (closes #437, #440, #442, #443, #445 and #366, which PR #438 already did),
CI, merge commit, fast-forward `main`. No tag.

What moved out of B0 compared with section 5 of the former master plan (rev 4, in git history):

| Item in the old B0 | Where it is now |
|---|---|
| OCR before/after comparison on GPU 0 (B0-3) | GPU window 1, the B3 gate ([04_b3_ultralytics_one_version.md](04_b3_ultralytics_one_version.md)); the B0 merge needs CPU gates only |
| Release prep (VERSION 0.7.0, CHANGELOG known-issue lines for #447, CC-2, CC-4) | VERSION and CHANGELOG consolidation: release prep in [06_release_acceptance.md](06_release_acceptance.md). The known-issue lines are dropped (D-6): no release ships before B1-B4 fix them |
| R3 live acceptance and its extra checks | merged into R-final, [06_release_acceptance.md](06_release_acceptance.md) |
| Tag decision point | one tag decision after R-final, which runs after B5 (D-16, D-37), [06_release_acceptance.md](06_release_acceptance.md) |
| B0 rows of the issue close table | [09_github_actions.md](09_github_actions.md) |
| Dependabot PRs | not in B0; the separate `b1/deps` lane in B1 ([02_b1_overview.md](02_b1_overview.md) "Lane b1/deps"; per-PR data in [09_github_actions.md](09_github_actions.md) section 8) |

## 2. Preconditions

- The owner approved this plan set (owner decision 1).
- The owner's yes to merge B0 to `main` is recorded (given 2026-11-03: "B0 merges to main now").
  Pushing still happens only after the gate cycle is green.
- The host all-clear (section 4).
- Nothing else writes in `<repo>-wt-r2round` while the gate runs (one writer per worktree).

## 3. Branch, worktree and what is already on it

- Branch `fix/411-r2-round`, worktree `<repo>-wt-r2round`, based on `main` at `5b95591f`, not
  pushed, no upstream configured (`git rev-parse --abbrev-ref @{u}` fails), so the PR goes
  against `main`.
- The worktree has its own `.venv` and a real `frontend/node_modules` directory. Call the venv
  binaries directly (`.venv/bin/python`, `.venv/bin/pre-commit`); never `source` it.
- Read the head with `git -C <repo>-wt-r2round log -1 --oneline`; never hard-code it.

Commits on the branch (`git log --oneline --no-merges 5b95591f..HEAD`), grouped:

| Group | Commits (by subject) |
|---|---|
| R1/R2 installer | adopt engines already on disk instead of rebuilding them; word the engine-load log lines for non-engine models too; keep the explicit segmenter GPU on re-runs, `--skip-models` exits 0; accept `--yes` on models install and repair; stop on an unreachable Triton, pin adopted ONNX hashes, order `--skip-models` startup; changelog and installation notes for the R2 installer fixes |
| Assist loop | admit nodes only when the crop budget can finish them; admit nodes on their full claim and every budget limit; serve a sentence for each node reason; say why loop clusters were not visited (frontend) |
| Issues #437, #442, #443, #445 | decode the end2end `[300,6]` head in the region detector; pick the region head from the declared output, fail closed (GPU-15); refresh the VLM registry on a by-name miss before `not_found`; coalesce by-name miss refresh and fail closed when unreadable; hold the cluster retrain while images are still being ingested; probe recent ingest for existence and require the input; replace the Ultralytics bus/zidane sample images with public COCO images |
| #440 Cropwright | list and activate ingest profiles on the ingest policy page; harden the ingest-profile picker; make the projects card count labels real text; stack the projects table rows at phone width |
| Validators and tests | stamp an issue's bypassable flag from its own validator's gate; require each validator's bypass set, including shared checks; fail loudly when `test_full_system` cannot fetch its input images |
| #439 | create and stamp a new project's indexes concurrently, then the revert (the issue stays open, B4-4D) |
| In-process locks | keep the index bootstrap lock per event loop; drop in-process locks that excluded nothing across processes; drop the superseded per-loop lock changelog entry |
| Security (neutral) | validate the model name and confine lifecycle routes to the repository; load uploaded weights in restricted mode during validation; refuse upload validation when restricted loading is unsupported; require ultralytics 8.4.174 or newer for restricted weight loading |
| B0 OCR fixes (`b0/ocr-fixes`, merged) | GPU-1 stretch the detector input to 32 multiples instead of padding it (`a9ac8ed8`); GPU-3a/13 decode once, EXIF-oriented, and composite alpha onto white with Pillow (`a6571415`); GPU-2 require pyclipper in the Triton image and drop the silent unclip fallback (`0b722ddc`) |
| B0 data/ops fixes (`b0/data-ops-fixes`, merged) | OS-2 fail the holdout freeze closed on bulk item errors (`16c3924f`); OS-1 make the first create of config, activation and keymap docs create-only (`b76f744f`); OPS-1 set the worker recycle jitter that PERFORMANCE.md documents (`3b5a3ddd`); BA-7 answer 503 on OpenSearch failure in `review_dismiss` (`1ff98a77`); CC-1 write state files through unique temp names, and stop status polls writing state (`55219ec5`); require uvicorn 0.41.0 or newer for the recycle jitter flag (`6a53673c`); return `hits.total` from the worker-leak fake search (`713d50e9`) |
| D-2 pin (done) | pin ultralytics to exactly 8.4.197 everywhere (`d1c6bab0`): `requirements.txt`, `requirements-test.txt` and `pyproject.toml` (which carried a third range, `>=8.4.82,<8.5`); the false "patch requires <8.4" comment in `requirements.txt` is corrected (U-C4); `test_ultralytics_is_one_exact_version_everywhere` added to `tests/test_requirements_contract.py`, seen red then green; CHANGELOG `[Unreleased]` "Changed" entry |
| Docs fixes (done) | `ad553ecd` correct the stale "patch breaks on 8.4" comments in the `Dockerfile`, `requirements-export-y11.txt` and `.github/dependabot.yml`; `0cc2b3ac` the docs checker skips its route and env checks under `docs/design/` |
| Plans and git hygiene | ignore Claude Code agent worktrees; add the master plan and its three source plans; path legend; point each source plan at the master plan; mark the plans committed and D-14 done |

Red tests of the B0 fixes, already on the branch: `tests/test_ocr_client_frame.py` (GPU-1),
`tests/test_ocr_orientation.py` (GPU-3a/13), `tests/test_ocr_unclip_pyclipper.py` (GPU-2),
`tests/curation/test_test_holdout_freeze.py` (OS-2), `tests/curation/test_config_store_first_create.py`
(OS-1), `tests/compose/test_worker_recycle_jitter.py` (OPS-1), `tests/curation/test_crop_discard.py`
(BA-7), `tests/test_atomic_write_concurrent.py` (CC-1, including the read-only status-poll
tests), `tests/test_requirements_contract.py` (uvicorn floor, Ultralytics pin). A targeted run of
the B0 test files on the merged branch gave 86 passed (ledger section 14), before the pin commit.

CHANGELOG entries already present in `[Unreleased]` (do not duplicate): pyclipper unclip,
`/ocr` reads the frame it reports, `review_dismiss` 503, recycle jitter with the uvicorn floor,
create-only first writes, holdout freeze fails closed, the Ultralytics pin, the R2 installer and
issue entries. `frontend/CHANGELOG.md` has the #440 entry. Check with
`rg -n "jitter|holdout freeze|create-only|pyclipper|EXIF|8\.4\.197" CHANGELOG.md` before adding
anything.

## 4. Host hold and coordination

- The OpenTranscribe 0.6.1 release session asked this repo to hold heavy runs until its release
  gate finishes. It will message when it is done, and again when GPU 1 frees. Find it with
  ListAgents (its name changes between sessions); message every OpenTranscribe session before a
  heavy run or GPU use, and wait for the reply ([11_risks_cleanup_host.md](11_risks_cleanup_host.md)
  section 1).
- Before the sweep: `uptime` (load average) and `nvidia-smi` (who holds GPU 0). Another
  project's Triton was seen on GPU 0 (about 5 GB); that is not ours, do not touch it.
- One heavy run at a time from this repo. The offline suite and Playwright e2e never run in
  parallel with each other.

## 5. Tasks

| ID | Task | Size | Status |
|---|---|---|---|
| B0-1 | Commit this plan set and retire the master plan | S | files edited in the consolidation pass; the commit remains |
| B0-1b | Retire the finished plans (removal wave R-1) | S | done in the working tree in the consolidation pass; part of the plan-set commit |
| B0-2 | CPU gate cycle (one sweep) | M (wall time) | to do, after the host all-clear |
| B0-3 | OCR before/after comparison | S (GPU, about 2 h) | moved to GPU window 1 (the B3 gate); not part of the B0 merge |
| B0-4 | Push, PR, CI, merge commit, fast-forward main | S | to do |
| B0-5 | Cleanup of the B0 worktree and branches | XS | to do, after the merge |

### B0-1 Commit the plan set and retire the master plan

Files: `docs/design/program/*.md` (new), the master-plan file in `docs/design/` (deleted),
`docs/design/task_queue_plan.md`, `docs/design/codebase_audit_plan.md`,
`docs/design/model_family_versions_plan.md` (each had a header pointer to the master plan;
repointed to `docs/design/program/00_index.md`), `docs/design/README.md` (a "Program plan
(current)" section with one row per `program/` file, rows for the three source plans and the
other missing rows, the statuses of file 12 section 5).

Steps:
1. Check that every section of the former master plan is placed: walk the mapping table in
   [00_index.md](00_index.md) section 12 and open each target file.
2. `git rm` the master-plan file; rewrite every reference (the three source-plan headers,
   `docs/design/program/*.md`) to `program/00_index.md`, the specific batch file, or, where the
   history is meant, "the former master plan (rev 4, in git history)".
3. A search of `docs` for the old file name (master, underscore, plan) must find nothing.
4. Run the naming-leak scan and the docs check on the changed files:
   `.venv/bin/python scripts/codegen/check_naming_leaks.py <files>` and
   `.venv/bin/python scripts/docs/check_docs_vs_code.py --only <files>`. Fix any hit; the public
   tree carries no machine paths, no private names and no security detail.
5. One commit: `docs(design): split the master plan into the program plan set`.

Done when: the commit exists, the stem search above is empty (only the wording "the former
master plan" remains), both checks pass. This resolves conflict C14 of
[12_prior_plans_reconciliation.md](12_prior_plans_reconciliation.md).

### B0-1b Retire the finished plans (removal wave R-1; docs only, one commit)

After B0-1, in the same worktree: apply the removal wave R-1 of
[12_prior_plans_reconciliation.md](12_prior_plans_reconciliation.md) section 6 (deletes, moves to
`docs/research/`, the content moves and every reference fix listed there), then the README rows
of its section 5. In this consolidation pass the edits are already in the working tree; they
ship in the plan-set commit.
- Commit: `docs(design): retire finished plans and move measured notes to research`.
- Checks: `.venv/bin/python scripts/docs/check_docs_vs_code.py`,
  `.venv/bin/python scripts/codegen/check_naming_leaks.py`, `.venv/bin/pre-commit run --all-files`
  (the cited-paths guard checks comments in `.py`, `.ts`, `.svelte`, `.js`, `.mjs` and `.sh`
  files), `.venv/bin/python -m pytest tests/test_doc_links.py tests/codegen/test_check_cited_paths.py -q --no-cov`.
- Done when: for every removed file, `rg -l '<file name>' --glob '!CHANGELOG.md'` lists only
  historical review files that cite it in plain text (file 12 section 6 says which may stay),
  and both checks pass.
- If B0 is already past its gate, this commit is the second commit of B1 instead.

### B0-2 Gate cycle (one sweep, in `<repo>-wt-r2round`)

Run in this order and keep every log under `artifacts_local/b0/gate/`:

1. `.venv/bin/pre-commit run --all-files`. No pre-push stage is configured today
   (`rg -n "pre-push" .pre-commit-config.yaml` finds nothing), so this is the whole hook set.
2. Offline suite, once, capped: `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live' -n 8`.
   Announce it to peer sessions first. R2's entry gate on `main` was 11,718 passed, 26 skipped.
   - **T-8 is known and pre-existing**: `tests/curation/cross_project_leak/test_keymap_and_ingest_policy_isolation.py`
     fails (3 tests) when it runs after `tests/projects/test_clone_settings.py` and
     `tests/projects/test_clone_activations.py` in one process. xdist usually hides it. If it
     shows, it is not a B0 regression: record it and confirm it also fails on `main` with
     `.venv/bin/python -m pytest -p no:xdist tests/projects/test_clone_settings.py tests/projects/test_clone_activations.py tests/curation/cross_project_leak/test_keymap_and_ingest_policy_isolation.py -q --no-cov`.
     The fix is B1-1D ([02d_b1_api_frontend_tests.md](02d_b1_api_frontend_tests.md)). Any other
     failure is root-caused and fixed on this branch before the push.
3. `make contracts-check`. The B0 commits add no wire change; the `reason_message` contracts are
   already committed.
4. `.venv/bin/python scripts/docs/check_docs_vs_code.py`.
5. Frontend full set, from `frontend/` (the branch touches `frontend/src` for #440): `npm ci`
   (real install, not a symlinked `node_modules`), then `npm run check`, `npm run lint`,
   `npx knip`, `npm test`, `npm run build`, then `npm run test:e2e` (Playwright, heavy: never in
   parallel with step 2).
6. Changelog rules as CI runs them: the root rule (changes under `src/`, `scripts/`, `docker/`,
   `export/`, `models/` need a `CHANGELOG.md` entry; `.github/workflows/ci.yml` job `changelog`)
   and the frontend rule (`frontend/src/` or `frontend/scripts/` need `frontend/CHANGELOG.md`;
   `.github/workflows/frontend.yml` job `changelog`). Reproduce both locally against `main`.
7. Static check of the plan commit: `git diff --stat main...HEAD -- docs/design` lists only the
   design docs expected.

There is no GPU step: the OCR comparison (B0-3) moved to GPU window 1.

Done when: every step is green, or each failure is fixed with a red/green test on this branch.

### B0-3 OCR before/after comparison (moved to GPU window 1, the B3 gate)

This task is not part of the B0 merge. It runs in the B3 GPU window
([04_b3_ultralytics_one_version.md](04_b3_ultralytics_one_version.md);
[00_index.md](00_index.md) section 8.1), against `main` after B0 (the "before" tree stays the
`5b95591f` worktree). The procedure is kept here so B3's gate agent can run it unchanged.

Why: GPU-1, GPU-3a and GPU-2 change `/ocr` output (box geometry, orientation, unclip). The
offline red tests prove each fix; this records the behaviour change on real images.

The live harness (`docker/test/compose.yml`) has no Triton and no GPU, so it cannot serve
`/ocr`. Use a real stack instead:

1. Images, local tags only (the watchtower hazard: never a public tag or `latest`):
   - "before": `git worktree add --detach <repo>-wt-b0-ocr-base 5b95591f`, then from it
     `scripts/release/build_and_publish.sh --dry-run --allow-dirty --only api,triton --local-tag-suffix -b0base`;
   - "after": from `<repo>-wt-r2round`, the same command with `--local-tag-suffix -b0head`.
   Check the resulting tags with `docker image ls 'openprocessor-local/*'` and use the exact names
   it prints.
2. Stack, compose project `op-ocr-b0`, every GPU id on GPU 0, core services only, ports no peer
   uses (R2 used port base 5200; check `ss -ltn` first). Two documented ways; pick one:
   - installer into `<scratch>/b0-ocr` from a local bundle, as R2 did (`--release-dir <bundle>
     --image-tag <local tag> --port-base <base> --project op-ocr-b0`; see
     `docs/design/release_acceptance_plan.md` for the bundle and the acceptance `--installer-args`);
   - or `docker compose -p op-ocr-b0 -f docker-compose.yml up -d --wait triton-server api opensearch`
     from the worktree with an env file that sets `TRITON_HTTP_PORT`, `API_PORT` and the other
     published ports, the GPU ids, and `OP_API_IMAGE`/`OP_TRITON_IMAGE`.
   Engines: build the OCR engines once (or adopt engines from an earlier scratch install with its
   `.install/precision.tsv`); B0 does not change any engine, so "before" and "after" share them.
3. The OCR pipeline Python files are model-repository files, not image files. For "before",
   point `OP_API_IMAGE`/`OP_TRITON_IMAGE` at the `-b0base` tags **and** copy
   `models/ocr_pipeline/1/*.py` from the `5b95591f` worktree into the install's model directory;
   for "after", the `-b0head` tags and the branch's files. Restart Triton and the API with the
   same `-p op-ocr-b0` (or `./openprocessor restart` inside an installer dir).
4. Image set (public data rule): the first 50 image ids, ascending, from the public COCO-Text v2
   annotations that have at least 3 legible machine-printed words (COCO-Text annotates COCO
   train2014). Add EXIF orientation 6 and 8 copies of 10 of them and 10 RGBA PNG conversions.
   Record the ids in `artifacts_local/b0/ocr/ids.txt`. If the COCO-Text download or licence is a
   problem on the day, use the first 50 COCO val2017 ids whose captions contain one of `sign`,
   `text`, `written`, `menu`, `banner`, and record that rule instead.
5. Call `/ocr` for every image on "before" and on "after"; store boxes, text and `image_size` as
   JSON in `artifacts_local/b0/ocr/{before,after}/`, plus a short summary (boxes per image, how
   many moved, rotated frames now matching `image_size`, transparent images composited on white).
6. Check `docker compose -p op-ocr-b0 exec triton-server python3 -c "import importlib.util as u; print(u.find_spec('pyclipper') is not None)"`
   prints `True` on "after". Record the pinned pyclipper version (1.4.0 in `Dockerfile.triton`)
   in the PR description.
7. Tear down: `docker compose -p op-ocr-b0 down -v --remove-orphans` (or
   `./openprocessor uninstall --yes --purge-volumes` in an installer dir), then remove the two
   local tag sets and the `<repo>-wt-b0-ocr-base` worktree.

Pass: after-boxes overlay the text on a visual spot check of 10 images (open them), rotated
copies match `image_size`, no image errors that did not error before. Time budget: 2 hours of
GPU 0 inside the window. If the B3 window gives no room, it moves into the R-final window
(nothing ships before R-final); record that in the B3 PR. A rejected OCR change is reverted by
the OCR-only path in section 9.

### B0-4 Push, PR, CI, merge

1. `git -C <repo>-wt-r2round push -u origin fix/411-r2-round`. Verify with
   `git ls-remote origin refs/heads/fix/411-r2-round` (the SHA equals the local head), never a
   piped exit code.
2. `gh pr create --base main --head fix/411-r2-round --title "fix(release): R2 round, B0 small fixes and the program plan" --body-file <file>`.
   The body has: "Closes #437", "Closes #440", "Closes #442", "Closes #443", "Closes #445" on
   their own lines; "#366 was done by PR #438" (close it with a comment if still open); "Refs
   #411" (it stays open until the tag); the red/green output per B0 task; the OCR behaviour
   change and a note that its before/after evidence comes in GPU window 1; the pyclipper
   version; the T-8 note if it showed.
3. Wait for CI with `gh pr checks <n> --watch` (no sleep loops). On a CI-only failure: reproduce
   locally, fix on the branch, push again.
4. Merge with a merge commit: `gh pr merge <n> --merge` (never squash, never rebase-merge).
5. Fast-forward the main checkout: `git -C <repo> fetch origin` then
   `git -C <repo> merge --ff-only origin/main`.
6. Verify: `git ls-remote origin refs/heads/main` equals `git -C <repo> rev-parse main`; each of
   #437, #440, #442, #443, #445 and #366 shows `CLOSED` (`gh issue view <n> --json state`).
   No tag is created.

### B0-5 Cleanup after the merge

See [11_risks_cleanup_host.md](11_risks_cleanup_host.md) section 5 for the full table. For B0:
`git worktree remove <repo>-wt-r2round`, then `git branch -d fix/411-r2-round` and
`git push origin --delete fix/411-r2-round`. Keep `perf/220-width-buckets` (B5c evidence).

## 6. Batch gate (B0 done when)

- B0-1, B0-1b, B0-2 and B0-4 done; CI green on the PR; the PR merged with a merge commit; `main`
  fast-forwarded locally and on the remote; the six issues closed. No GPU run was needed.
- No VERSION change (`cat VERSION` is `0.6.0`), no tag (`git ls-remote --tags origin | rg v0.7`
  is empty).
- Gate evidence under `artifacts_local/b0/`.
- Delete this batch's plan file (`program/01_b0_close_out.md`) and its rows in `00_index.md` and
  `docs/design/README.md` after moving the keep-list below ([00_index.md](00_index.md)
  section 8.2). Do this in the B0 closing PR, or in the first B1 docs commit if B0 already
  merged.

### What to keep after merge

| Fact or section | Moves to |
|---|---|
| the OCR behaviour changes (stretch to 32 multiples, EXIF-oriented single decode, alpha on white, pyclipper hard import, real DB unclip) and the pyclipper 1.4.0 pin | `docs/ARCHITECTURE.md` (OCR pipeline) and `CHANGELOG.md` (present) |
| the exact Ultralytics pin rule (`==8.4.197` in three files, one test) | `docs/ARCHITECTURE.md` (supported model families) until B3 supersedes it |
| the recycle jitter and the uvicorn `>=0.41.0` floor | `docs/PERFORMANCE.md` (present) |
| the B0-3 OCR before/after procedure and result | carried by B3's plan until GPU window 1; the result goes to `docs/PERFORMANCE.md` |
| T-8 order-pollution note | closed by B1-1D; nothing to keep |

## 7. Docs, contracts, CHANGELOG

- Docs: the plan set and `docs/design/README.md` (B0-1). `docs/PERFORMANCE.md` already describes
  the jitter; `ATTRIBUTION.md` already has the pyclipper row.
- Contracts: no change in B0.
- CHANGELOG: entries already present (section 3). The OCR geometry and orientation changes are
  behaviour changes; they are written as such under `[Unreleased]` and are gathered into the
  release's upgrade notes at release prep ([06_release_acceptance.md](06_release_acceptance.md)).

## 8. Upgrade impact

None for users: nothing is released. For a developer on `main`: rebuild the Triton image
(pyclipper is now a hard import, so `ocr_pipeline` fails to load without it) and the API image
(uvicorn floor). The Ultralytics pin equals what the current images resolve.

## 9. Rollback

- Before the merge: drop or revert the failing commit on the branch.
- After the merge: revert the merge commit (`git revert -m 1 <merge>`) on a short branch with its
  own PR.
- OCR only: if the OCR behaviour change is rejected, revert the three OCR commits (`a9ac8ed8`,
  `a6571415`, `0b722ddc`) on a short branch; the rest of B0 stands.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Host contention delays the gate (M1) | wait for the all-clear; announce heavy runs; one at a time |
| The large branch hides a regression | the full offline suite and e2e run once on the merged tree; R-final later covers it live |
| T-8 order pollution looks like a B0 failure | known; reproduce on `main` as above; fixed in B1-1D |
| OCR behaviour change surprises a consumer (M3) | before/after evidence in GPU window 1; CHANGELOG says so; OCR-only revert path |
| Watchtower re-pulls public tags over local builds | local-only tags (`openprocessor-local/*:<suffix>`) for every build in B0-3 |

## 11. Ready-to-paste brief: B0 gate agent (coordinator may run it itself)

```
You run the B0 CPU gate cycle for OpenProcessor. Plan: docs/design/program/01_b0_close_out.md,
sections 4 and 5 (B0-2; no GPU step). Worktree: <repo>-wt-r2round, branch fix/411-r2-round. You are the
only writer in it. Do not push, merge or tag; report back.
- Before any heavy step: ListAgents, message every OpenTranscribe session, wait for "go"; run
  uptime and record it.
- Use .venv/bin/... directly; never source the venv.
- Offline suite with -n 8, never -n auto. Playwright e2e never in parallel with it.
- No GPU run, no Docker stack: the OCR comparison runs later in GPU window 1.
- When the gate is green, delete nothing; the coordinator moves the keep-list and deletes this
  plan file (section 6).
- Never use pgrep -f, pkill -f, or sleep/until loops. Wait on healthchecks, `gh pr checks --watch`
  or the Monitor tool. Never touch another project's containers, volumes or networks.
- Any failure other than the known T-8 order pollution: root-cause it, fix it with a red/green
  test in one conventional commit (hooks on, never --no-verify).
- Deadline: report within one working session of the host "go", with every log path.
```

## 12. Corrections found while splitting

- Section 5.1 of the former master plan lists `tests/test_job_state_read_only.py` as the CC-1 red-test file. It
  does not exist: the read-only status-poll tests landed in `tests/test_atomic_write_concurrent.py`
  (`test_status_poll_of_stale_job_does_not_write_state`, parametrized over item scores, embedding
  viz and selection). The derived view there reads `failed`, not `interrupted` (none of the three
  families is resumable). The planned "dead holder does not block `selection.start_job`" and
  "stale dataset import still resumes" tests were not added; the resume path is covered by the
  existing `tests/curation/dataset_import/test_stale_and_undo_gate.py`. B1-1A replaces this path
  with leases and its tests cover the dead-holder start ([02a_b1_job_leases.md](02a_b1_job_leases.md)),
  so no B0 test is added.
- The former master plan (5.2 step 7) ran the OCR comparison on the live harness; the harness
  has no Triton (B0-3 above uses a real stack, in GPU window 1).
- The former master plan (D-2, 5.1) names two requirements files and a new
  `tests/test_ultralytics_pin.py`; the pin landed in three files (`pyproject.toml` too) and the
  test went into `tests/test_requirements_contract.py`.
