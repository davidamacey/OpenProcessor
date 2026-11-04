# Risks, cleanup, and host, GPU and peer rules

Status: plan, not started (the rules in section 1 apply to every batch from now on).
Depends on: nothing; every other file in this set links here.
Decisions applied: owner decisions 2, 3, 7 and 8 of 2026-11-03; D-5 (#42 never blocks); the
standing owner rules (GPUs 0 and 2 for this repo, GPU 1 only when the OpenTranscribe session frees
it; never touch another session's containers; always `docker compose -p <name>`).
Source IDs: the former master plan (rev 4, in git history) sections 13 (host), 14 (cleanup), 15
(risks and the unverified list), 7.6 (the #42 side track) and Appendix B (refuted and cut), which
[00_index.md](00_index.md) section 12 maps; ledger sections 8, 15, 17, 18 (private). Decisions
are in the register at [00_index.md](00_index.md) section 6 and are not restated here.

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (`git worktree list`, `git branch
-a`, `git branch -r --merged origin/main`; Docker state is taken from the ledger and must be
re-checked before any removal).
Consolidated 2026-11-03. Updated 2026-11-04: risks M34-M46 for the PostgreSQL control plane
([02_control_plane_database.md](02_control_plane_database.md)); M4, M5, M9, M11, M16 and M25-M27 revised.

What to keep after merge: the host rules (section 1) go to `docs/CONTRIBUTING.md`-level
maintainer notes (the repo's `CONTRIBUTING.md` "Working on a shared host"), the risks that stay
live go to the GitHub issues that own them (N14 for evidence items), and the "do not redo" list
to `docs/design/engineering_standards.md` section 5. The files go as described in
[00_index.md](00_index.md) "Plan lifecycle".

## 1. Host, GPUs and peer sessions

| Resource | Rule |
|---|---|
| GPU 0 (A6000) | this repo's live and GPU runs. Another project's Triton has been seen on it (about 5 GB); it is not ours, leave it |
| GPU 2 (A6000) | this repo's by the standing owner decision; check `nvidia-smi` and ask the peer sessions (ListAgents) before using it, because another session has asked for it |
| GPU 1 (3080 Ti, 12 GB) | only when the OpenTranscribe session says it is free; used only for the #42 side track (section 2) |
| Heavy runs | full offline suite, Playwright e2e, image builds, live harness, acceptance, GPU gates, benchmarks |

Rules for every session and every agent brief:
1. **Before any heavy run:** ListAgents; message every peer session that shares the host (each
   OpenTranscribe session included) and wait for the reply. Run `uptime` and record the load
   average; if the host is loaded, wait or cap the run. Never stop or restart another session's
   stack yourself.
2. **One heavy run at a time from this repo, and only the budgeted ones.** The
   [Test and GPU budget](00_index.md) allows three live-harness runs (a 30-minute B1 smoke, one at
   the end of B2 covering B1 and B2, one at B4) and exactly three GPU windows (B3 gate plus the B0
   OCR before/after; the B5 window, which now also holds the short W5e gate; R-final). The research
   prototype windows of rule 9 are separate and use only idle GPU time. No long or GPU run per branch or per commit. The
   windows never overlap each other or another live run. The offline suite runs once per wave on
   the merged branch with `-n 8`, never `-n auto`, unless the owner says otherwise.
3. **Pause at once** when a peer session asks.
4. **Never touch another session's containers, volumes, networks or images.** Snapshot
   `docker ps -a`, `docker volume ls`, `docker network ls` and `docker image ls` before and after
   every acceptance run (the R2 method) and diff them.
5. **Always `docker compose -p <name>`.** The default project name (`openprocessor`) is the
   owner's live stack. From a worktree, always pass `-p`. Every lane brief names its project.
6. **Local-only image tags** (`openprocessor-local/*:<suffix>`, or a suffix through
   `build_and_publish.sh --local-tag-suffix`) for every build that a gate or a harness uses. A
   watchtower on the host re-pulls public tags (`latest`, release tags) over local builds.
7. **Waiting:** healthchecks (`up -d --wait`), `gh pr checks --watch`, or the Monitor tool. Never
   `pgrep -f`, `pkill -f` or sleep/until loops (a `pgrep -f "<text>"` from a shell whose own
   command line holds that text matches itself and never ends). If a lane agent goes idle,
   check its worktree's shells by PID.
8. **Secrets:** pass the HF token and other keys from `.env` without printing or reading them
   into a log; the token-leak grep runs over every evidence directory before a report.

9. **Research prototype windows** (approved by the owner 2026-11-04 as short runs only, RD-2; [00e](00e_research_track.md) section 2.1): the venv prototypes
   run only on idle GPU time (GPU 0, GPU 2 when agreed), never inside a program GPU window or a
   live-harness run, one GPU job at a time on the host under `flock "$SCRATCH/locks/gpu<N>.lock"`
   (one lock file per card, shared by every topic), `CUDA_DEVICE_ORDER=PCI_BUS_ID` always (without
   it `CUDA_VISIBLE_DEVICES=2` selects the 12 GB card), `nice -n 10` or 19 for CPU work with at most
   8 threads, `uptime` at start and end (a start above load 16 is noisy and repeated), each run
   inside its row's tier-1 budget and the whole track inside about 10 GPU hours before the tag (the
   coordinator's ledger; more needs the owner's OK), and a pause the moment the program or a peer asks. No compose stack
   unless a prototype names its own `-p`.
10. **Set P** (the owner's private high-resolution photos, approved RD-8): **one** shared seeded
   sample for every topic is copied to local NVMe by one copy script whose one manifest lives
   outside the repository (review 3.7: not one copy per topic); the copy is removed after PW-3 and
   the removal date recorded in the coordinator's ledger; every dataset root
   is a command-line argument or an environment variable, never a path in a tracked file; results
   stay in `artifacts_local/`; public text says only "met / not met on set P" unless the owner says
   yes to an aggregate per note; no set-P image, name, path, location value or plate string in any
   tracked file, commit, issue or hosted model call (RD-9).

Current holds (re-check before acting):
- The OpenTranscribe 0.6.1 release session asked this repo to hold heavy runs until its release
  gate finishes; it will message when done, and again when GPU 1 frees. B0's gate waits for it.
- The owner's own 0.6.0 dev install is stopped, with its data volumes and images kept. It is the
  owner's; do not start, upgrade or remove it without the owner.

## 2. Side track: #42 small-GPU acceptance (GPU 1)

The procedure, pass list and brief are owned by [07_b5_overview.md](07_b5_overview.md) section 11.
The host rule here: GPU 1 is used only after the OpenTranscribe session that holds it confirms in
writing that it is free; the run never blocks a batch (D-5), and if it has not run by the v0.7.0
tag the release notes say the `small` profile is not yet verified on a 12 GB card
([06_release_acceptance.md](06_release_acceptance.md) RP-4).

## 3. Risk register

`Owner file` is where the detailed mitigation lives; this table is the single list.

| # | Risk or edge case | Mitigation | Owner file |
|---|---|---|---|
| M1 | Host load from peer sessions delays the gates and the acceptance run | wait for the all-clear; announce heavy runs; one at a time (section 1) | this file |
| M2 | The acceptance run finds a regression in a large merged tree | `fix/rfinal-findings`, targeted tests, re-run only the failed rows (was `fix/r3-findings`) | [06](06_release_acceptance.md) |
| M3 | OCR changes (GPU-1, GPU-2, GPU-3a) alter downstream text consumers | before/after evidence on public images; CHANGELOG behaviour change; revert the three OCR commits alone if rejected | [01](01_b0_close_out.md) |
| M4 | A job holder pauses longer than the TTL and the reaper takes over | thread heartbeat with its own connection; TTL 120 s; every job-state write fenced by `fencing_token` in SQL (no residual window for job state); `may_write()` self-fence for OpenSearch writes | [02b](02b_jobs_locking_events.md) section 3 |
| M5 | A Postgres outage blocks every job start and every control write; an OpenSearch outage blocks the data plane | fail closed with 503; `/health` reports `database` and OpenSearch; live step 9 of the B2 run | [02](02_control_plane_database.md) section 2 rule 7 |
| M6 | A lease doc is deleted with its project while a job runs | the next renew is `Lost` and the job stops; delete already waits for or cancels jobs | B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| M7 | Registry migration of real projects | idempotent CLI, dry run saved, file renamed not deleted, ids preserved, fail-closed 409 until migrated, downgrade note | B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| M8 | A reader still reads `class_registry.json` after B1 | the B1 done-when greps for it outside export snapshots and the trainer; dead-code and layering ratchets | B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| M9 | Celery redelivery storms, cold-shutdown ack loss, a queue nothing drains, wedged pools | the claim `UPDATE`, attempt cap 3, the reaper from the `jobs` rows, soft shutdown, the compose consumer test, the pool healthcheck | [02b](02b_jobs_locking_events.md), [03](03_b2_overview.md) |
| M10 | The `queued` status breaks Cropwright | contracts, lane 4E (B2 stage 4), a browser check | [03b](03b_b2_w2_job_families.md) |
| M11 | An upgraded install lacks a Postgres, Valkey or Flower secret | `upgrade` and `repair` create missing ones, never rotate; R-final RF-56 | [03a](03a_b2_w1_queue_infra.md) lane 1C, [06](06_release_acceptance.md) |
| M12 | Ultralytics ships a breaking patch between the CPU matrix and the image build | exact pin (done in B0); the pin test in CI and in each image; the D-28 freeze rule | [04](04_b3_ultralytics_one_version.md) |
| M13 | Stage B misses throughput (unrolled NMS at batch 64) | ship Stage A; the owner sees the measurement | [04](04_b3_ultralytics_one_version.md) |
| M14 | The deployed default detector engine is incompatible after B3 | the adapter fails closed; the engine stamp forces a rebuild on `repair`/`upgrade`; CHANGELOG "Breaking" | [04](04_b3_ultralytics_one_version.md) |
| M15 | The `API_WORKERS` default lowers throughput | benchmark against the 32-worker baseline retaken on the B3 engine | [05](05_b4_stateless_api.md) |
| M16 | The Postgres event feed changes SSE ordering or resume | the `(xid, id)` cursor and its out-of-order commit test; live SSE step; browser check of every live page | [02b](02b_jobs_locking_events.md) section 6 |
| M17 | A security detail leaks into a public file or issue | neutral titles and bodies; the naming-leak scan; a manual read of every diff and issue that touches SEC | [09](09_github_actions.md) |
| M18 | An image ships untracked files | image-vs-git audit before every image push and in every acceptance | [06](06_release_acceptance.md) |
| M19 | Contract drift between the lanes of one batch | `make contracts` in each commit that changes a wire model; `make contracts-check` once in the batch sweep; wire-changing lanes merge last in their wave | [00](00_index.md) |
| M20 | Line numbers in the source plans have moved | re-find every site by symbol or the given `rg` pattern | every file |
| M21 | One track of a large batch PR fails the sweep | one merge commit per track; revert that track's merge (`git revert -m 1`); 1B is coupled to 1A | B1 (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| M22 | A project with more than 10,000 classes | 409 `registry_full`; per-class documents would be a separate design (not planned) | B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| M23 | Users stay on 0.6.0 for the whole of B0-B5 and R-final (no intermediate release, owner decision 2; D-37 = yes: nothing is tagged until every plan is implemented and tested); the registry race and the audit fixes reach them only at v0.7.0 | accepted by the owner; keep batches moving; merged work sits on `main` untagged | [00](00_index.md) |
| M24 | Cropwright is touched in B1, B2 and B4 | all ship in one release; one frontend release note; R-final browser checks of every changed page | [06](06_release_acceptance.md) |
| M25 | The one supported upgrade (0.6.0 -> 0.7.0) carries every change at once: the control-data move to Postgres, Postgres and Valkey and the new services, an engine rebuild, `API_WORKERS` | R-final install B upgrades a real 0.6.0 install with real class registries and an in-flight job, then rolls back and re-upgrades (RF-40 to RF-63) | [06](06_release_acceptance.md) |
| M26 | Rollback to 0.6.0 after the control-data move finds its OpenSearch control indexes and job files gone | `rollback_to_060.py` recreates them from the upgrade bundle and renames `class_registry.json.migrated-0.7.0` back; RF-62 | [02c](02c_migration_ops_tests.md) section 2.3, [06](06_release_acceptance.md) |
| M27 | `openprocessor upgrade` run from a 0.6.0 install uses the 0.6.0 CLI, so the new migration hook might not run | the hand-off to the new installer is tested (`tests/installer/test_upgrade_060_hook.py`); RF-40 proves it live | [03d](03d_b2_s2_control_data.md) lane 2E |
| M28 | Integration branches drift from `main`, or a dependabot merge moves `main` under them | no direct dependabot merges during the program; the separate `b1/deps` lane (B1 (merged; see [CHANGELOG](../../../CHANGELOG.md)) section 4.1) takes the open PRs; each batch cuts its integration branch from `main` after the previous batch merged | [09](09_github_actions.md) |
| M29 | A base-image bump (Triton 26.09, Python 3.14) breaks the TensorRT pin and every engine | not taken as-is; D-38 places it in B3 behind the GPU gate, or after the tag | [09](09_github_actions.md), [04](04_b3_ultralytics_one_version.md) |
| M30 | `CHANGELOG.md` `[Unreleased]` grows over five batches with repeated subsections and scattered "Breaking" lines | each batch adds entries under the existing headings; release prep consolidates once (RP-2) | [06](06_release_acceptance.md) |
| M31 | The pre-existing T-8 test-order pollution is taken for a regression in a sweep | documented in B0's gate; fixed in B1-1D with an ordered no-xdist red test | [01](01_b0_close_out.md), B1 1D (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| M32 | The plan set drifts from the code as batches land | every file carries its "Verified against" line; each batch's first task re-verifies its own file's references with `rg` and adds a Corrections entry | every file |
| M33 | Several lanes write to one worktree, or a lane runs a full suite and loads the host | one agent per worktree and branch; lanes run T0 targeted tests only; the coordinator runs one sweep per wave on the merged branch (the [Test and GPU budget](00_index.md)) | [00](00_index.md) |
| M34 | The control-data move loses or corrupts a project's classes, configs or job history | bundle first, one-transaction load, verify (counts, ids, checksums) before any delete, rollback from the bundle, the fixture tests, R-final on a real install | [02c](02c_migration_ops_tests.md) section 2 |
| M35 | A query reads or writes another project's control rows | row-level security with `FORCE`, `project_tx` only, the RLS test over every table, the scope-helper ratchet; the OpenSearch guard unchanged for the data plane | [02a](02a_postgres_schema.md) section 1.4 |
| M36 | Connection exhaustion (32 API processes, Celery children, listeners) | the budget table and `tests/config/test_db_connection_budget.py`; `OP_PG_MAX_CONNECTIONS`; the PgBouncer overlay for more replicas | [02b](02b_jobs_locking_events.md) section 8 |
| M37 | A long write transaction stalls the event feed or holds row locks during a network call | rule 4 (no network call inside a transaction), batched inserts, trap T-13/T-14 ratchets | [02c](02c_migration_ops_tests.md) section 8 |
| M38 | Celery, kombu and redis-py versions drift (kombu 5.7.0 appeared on 2026-11-03 under Celery 5.6.3's open range) | exact pins in file 13, `pip check` per image, spike S1 on the pins | [02b](02b_jobs_locking_events.md) section 7.1 |
| M39 | The Postgres 18 image's new data path or a secret ownership mismatch breaks the first start or the upgrade | spike S0; compose render tests; installer tests | [03a](03a_b2_w1_queue_infra.md) lane 1C |
| M40 | Two beats on two hosts double every periodic side effect | the session advisory lock at `beat_init`; transaction locks in each periodic body; live step 10 | [02b](02b_jobs_locking_events.md) section 4.1 |
| M41 | The class-name projection on items drifts from the registry | one writer path (`resolve`), the outbox row in the same transaction, the sync job, the verifier (D-47) | [02](02_control_plane_database.md) section 3.3 |
| M42 | Two detection or VLM worker replicas process one project | project claims in `stream_worker_claims` (D-53); live step 13 | [02b](02b_jobs_locking_events.md) section 5 |
| M43 | A backup of Postgres and the OpenSearch data taken at different times disagree after a restore | `openprocessor db verify` reports the drift; INSTALLATION says to restore both from one time | [02c](02c_migration_ops_tests.md) section 6 |
| M44 | B2 is large (six stages) and blocks B4 | stage merges with the `db` tier each; stage 3 codes alongside stage 2, stages 4 and 5 in parallel; B3 codes throughout | [03](03_b2_overview.md) section 3 |
| M45 | B3 and B2 lane 5A both edit `docker/trainer/` | disjoint file lists; B3 rebases on `main` after B2 merges | [03f](03f_b2_s5_gpu_queues.md) |
| M46 | The GPU arbiter stays single-host (it drives the local Docker socket) while everything else scales out | documented limitation; `gpu_reservations` keyed by host; a multi-host arbiter is out of scope | [03c](03c_b2_w3_beat_reaper_workers.md) lane 6A |

### 3.1 Unverified (be honest)

- Nothing in this plan set was run (no tests, stacks, engines, npm or browsers) beyond reading
  files, git and GitHub; the B0 targeted run (86 passed) and the pin commit's red/green predate it.
- The lease and registry designs are designs. Their cost per heartbeat and behaviour under
  OpenSearch load are argued, not measured; the B1 live tests measure them.
- R2's loop-accounting oddity is believed fixed by "admit nodes on their full claim and every
  budget limit"; R-final RF-30 verifies it.
- R2 fix (e), cleanup cancels jobs, was verified by reading code only; R-final RF-29 verifies it.
- The PostgreSQL control plane is a design: the event cursor, the claim statements, row-level
  security under load and the connection budget are argued from the PostgreSQL docs, not measured;
  the `db` tests and the B2 live run measure them. Spikes S0 (Postgres image) and S1 (Celery on
  Valkey with the exact pins) run first in stage 1.
- U-6.2 was never run: TensorRT builds, Triton serving, embedded-NMS throughput at batch 64, and
  checkpoints from the 8.4.48 trainer.
- The YOLO26 `end2end=True` discrepancy is explained only by the maintainer's "`end2end`
  deprecated, mapped to `nms`" comment; checking the head after every export stays the safeguard.
- #439's cause is not attributed.
- GPU 2's allocation is a standing decision with an open request from another session.
- The OpenTranscribe patterns were re-read from its public repository on 2026-11-04
  ([02](02_control_plane_database.md) section 8); the Valkey image's runtime user and whether
  `valkey-cli` honours `REDISCLI_AUTH` are not confirmed by the docs (spike S1 checks).
- COCO-Text v2's current download location and licence were not re-checked (B0-3 has a fallback
  rule).
- 8.4.197 is what the earlier rc images resolved; the exact pin makes later builds resolve the
  same release by construction.

## 4. Do not redo: refuted and cut audit items

These stay refuted or cut. A lane that finds itself implementing one stops and asks.

**Refuted** (audit Appendix C): OS-4 combine holdout recompute; GPU-11 ArcFace std; asyncio R9
"periodic loops die"; asyncio R10 `append_done` interleave; the `health.py` readiness
`TimeoutError` claim; `get_event_loop()` in coroutines; asyncio R6 (part) "`append_nowait` drops
tasks"; FE-4 state-writing `$effect`s; the FE-2 premise "no batch read-by-ids route"; TESTS-5
`xfail_strict`; TESTS-8 `/detect` 503 `!= 500`; TESTS-9 script-style tests; the API-3 "wire
contract violated" wording; TESTS-7 (part) process-cache leak.

**Cut as churn** (audit Appendix D):

| Area | Cut proposals |
|---|---|
| asyncio | ruff `ASYNC` rules; a `spawn` helper and `create_task` ratchet (asyncio R5); a `gather` ratchet; the bounded `event_hub` writer |
| tests | `check_test_integrity.py`; `xfail_strict`; `register_process_cache`; the expected-skip CI gate and research deps; renaming script tests; rewriting other sleeps; replacing the 7 frontend source-text tests |
| security | URL-userinfo redaction (SEC-6 extension); the larger SEC-3..5 options (reasons in the private notes) |
| API | pure-ASGI middlewares; a pure-ASGI 500 catch-all; dropping the startup reconcile early (B2 deletes it); API-10 to zero in one pass; the capacity-cache change; a cluster-wide probe-cap lease |
| locks | `flock` in the registry and in `FileJob.write` |
| OpenSearch | `search_after` for `audit/queue`; composite aggregations; PIT for scroll; rewriting `combine/holdout.py`; OS-7 extras |
| GPU/inference | GPU-4 TTL re-validation; an EfficientNMS CI check; region unit-guess removal before a box-format parameter (GPU-15); GPU-12 standalone; GPU-6 before the decision; tenacity for GPU-8; `OPENCV_IO_MAX_IMAGE_PIXELS` as the GPU-5 fix; changing the ArcFace constant |
| frontend | refactoring the `$effect`s and an effect lint; pager windowing; moving `select()` out of `load`; a new `crops/get_batch` route |

Also cut by the program (no churn): CC-3's mtime heuristic (the lease fixes it in B1); P1.6's
auto-label and assist-loop `claim_trigger` rewrite (B2 stage W3 deletes those files); GPU-14's
interim clip (B3 deletes the heuristic).

## 5. Cleanup

Verify contents before any removal; ask the owner before deleting anything not marked "safe".
Never delete the archived repositories (listed in the owner's private repo map) or bake-off
outputs (some back the paper).

| Item | State (2026-11-03) | Action | When | Safe? |
|---|---|---|---|---|
| `<repo>-wt-r2round`, branch `fix/411-r2-round` | integration branch, unpushed | `git worktree remove`; delete the local and remote branch after the PR merges | after the B0 merge | safe once merged |
| `b0/ocr-fixes`, `b0/data-ops-fixes`, `review/*`, `fix/model-route-hardening` and their agent worktrees | ledger 17: already removed; `git worktree list` shows only the main checkout and `-wt-r2round` | re-list; nothing to do if gone | now | n/a |
| Unregistered `.claude/worktrees/agent-*` directories (three seen) | not in `git worktree list`; may belong to running agents | check ListAgents and the directories' age and content; remove only when no agent uses them | after this planning round | ask |
| `.claude/worktrees/audit-r2` | removed (ledger 17) | recreate with `git worktree add --detach .claude/worktrees/audit-r2 5b95591f` only if a lane needs the pinned line numbers; remove at the end of B1 | B1 | safe |
| Branch `perf/220-width-buckets` (1 commit, unmerged) | #220 step-2 evidence | deleted 2026-11-04 with the owner's OK; the numbers are in the #220 comment of 2026-11-02 and the maintainers keep the patch | - | done |
| Merged remote branches: `build/41-slim-images`, `chore/doc-example-addresses`, `chore/v080-batch2`, `-batch3`, `-batch4`, `-batch7`, `chore/v080-quick-wins`, `dependabot/npm_and_yarn/frontend/frontend-6b036a9217`, `dependabot/pip/backend-93d8807eb1`, `fix/286-loop-vlm-outage`, `fix/adopt-unstamped-engines`, `fix/promoted-model-input-dtype`, `fix/release-clean-build-context`, `perf/42-small-gpu`, `security/labels-path-confine` | all in `git branch -r --merged origin/main` | `git push origin --delete <branch>` for each, after re-running the merged check | owner yes | safe once re-checked |
| Unmerged dependabot branches (#422, #423, #432, #433, #434) | open PRs | handled by [09_github_actions.md](09_github_actions.md) section 8; GitHub deletes them when the PRs close | per section 8 | n/a |
| `<scratch>/b8/deploy` (old deploy worktree) | unregistered; a root-owned folder remains | owner removes with sudo | owner | ask |
| Root-owned empty dirs (`secrets/vlm` remnants in the main checkout, `<scratch>/op-install-2`) | need sudo | ask the owner to run the removal | owner | ask |
| `<scratch>/fixa`, `<scratch>/r1`, `<scratch>/r2` (incl. `r2/install`, `r2/release`) | evidence copied to `artifacts_local/r1`, `r2`; the owner must run the delete (agent deletes were blocked) | list contents, confirm the copy, owner removes; keep `r2/release` until R-final has its own bundle | after R-final | ask |
| `/tmp` evidence copies of earlier rounds | owner must run the delete | confirm the copies in `artifacts_local/`, owner removes | now | ask |
| `artifacts_local/model_versions` (Ultralytics matrix) | the evidence the model-family plan cites | keep; verify it holds the matrix, logs and probe scripts | keep | keep |
| Docker: `op-r1_triton_net` | stuck until a Docker daemon restart | owner restarts the daemon at a quiet time, then `docker network rm` | owner | ask |
| Docker: `op-r1_*`, `op-b8` volumes | ledger 17: the owner must run the removal; ledger 18: "our test volumes are removed" | re-check with `docker volume ls`; owner removes any left | owner | ask |
| Docker: volumes of an older test project of unknown origin | unknown origin | leave; ask the owner | owner | ask |
| Docker: the `openprocessor` default-project volumes | the owner's live data | never touch | never | no |
| The owner's 0.6.0 dev install (stopped; data volumes and images kept) | the owner's | leave; the owner decides when to upgrade it | owner | no |
| Images `openprocessor-local/*:0.7.0-rc3`, `-rc5`, `latest-rc*`, `:b8` | ledger 18: "old image tags are removed" | re-check with `docker image ls 'openprocessor-local/*'`; remove leftovers | now | safe (local tags only) |
| Untracked files in the main checkout: two research notes under `docs/` and `images.lock.sha256` (0.6.0 cut) | pre-existing, not from this round | done 2026-11-04: the notes were older copies of files already in `docs/research/` (moved to `artifacts_local/`); `images.lock.sha256` is gitignored (a local release artifact) | owner | done |
| Budgeted run projects (`op-b1-live` smoke, `op-b2-live` for B1+B2, `op-b3-gpu` with the B0 OCR comparison in the same window, `op-b4-live`, the B5 window's project, `op-rfinal*`) | future | `docker compose -p <name> down -v --remove-orphans` at the end of each run; remove their local tags | after each run | safe |
| R-final scratch (`<scratch>/rfinal`) and `op-rfinal*` projects | future | uninstall with `--purge-volumes`; keep the report and evidence in `artifacts_local/rfinal/` | after the tag decision | safe |
| Bake-off outputs | some back the paper | never bulk-delete | never | no |
