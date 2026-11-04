# R-final: release prep, one release acceptance run, and the v0.7.0 tag

Status: plan, not started.
Depends on: B5d merged to `main` ([06_b5_overview.md](06_b5_overview.md); D-37 = yes, so this
file runs last); a host window agreed with peer sessions (GPU window 3,
[00a_decision_register.md](00a_decision_register.md) section 8.1).
Decisions applied: owner decisions 2 and 3 of 2026-11-03 (one release acceptance, one release
v0.7.0, no intermediate tags, supported upgrade 0.6.0 -> 0.7.0 directly); D-37 (B5 inside the
release, R-final after it); D-3 and D-4 (R3 checks and the segmenter tier move here); D-5 (#42
never blocks); D-16 (one tag decision, replacing the old D-16 and D-34); D-6 dropped.
Source IDs: the old R3 (sections 5.3 and 5.4 of the former master plan, rev 4, in git history),
R-B1 (7.1 "Release"), R-0.9 (7.4); task-plan upgrade notes (Q-Phase 2, Q-E14, Q-E25); U-6.2,
U-R3; `docs/design/release_acceptance_plan.md` (#54; removed at RP-9);
`docs/releases/RELEASE_CHECKLIST.md`; R2 report (private notes).

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (scripts, phases, flags and
checklist steps named here exist on that tree; the B1-B5 features they test do not exist yet).
Consolidated 2026-11-03.

## 1. Goal

Prove, once, on the merged `main` after B5, that a new user and an upgrading 0.6.0 user get a
working, correct, fast product, then tag v0.7.0 on the owner's word. The run is the union of the
old R3 (after B0), R-B1 (after B1) and R-0.9 (after B4) checks, plus the B5 speed and parity
gates of [06_b5_overview.md](06_b5_overview.md) (recorded rows, section 4.6), so nothing they
would have caught is lost. It also verifies the production-ready definition in
[00a_decision_register.md](00a_decision_register.md) section 5. R-final is GPU window 3.

## 2. Preconditions

- B0-B5 merged to `main` with green CI; every batch gate passed and is recorded
  ([00a_decision_register.md](00a_decision_register.md) section 4).
- Release prep (section 3: VERSION and CHANGELOG first) merged, so `/health` and the images
  report 0.7.0.
- `rg -n "fcntl" src` is empty on `main` (B4 done-when) and the file-lock ratchet baseline is at
  zero.
- The 1-vs-32 worker diff of B4 is recorded (not re-run here).
- Host: ListAgents, message every peer session that may use the GPUs, wait for "go"; `uptime`,
  `nvidia-smi` recorded ([11_risks_cleanup_host.md](11_risks_cleanup_host.md) section 1).

## 3. Release prep (branch `chore/release-0.7.0-prep`, worktree `<repo>-wt-release-prep`)

This is the only VERSION bump of the program. It lands after B5 and before the run (RP-1 and
RP-2 come first), so the run tests the version that ships and `/health` and the images report
0.7.0 (R2 found the rc images reporting 0.6.0 because VERSION was not bumped). Size of the whole
section: M.

| ID | Task | Files | Done when |
|---|---|---|---|
| RP-1 | VERSION bump `0.6.0` -> `0.7.0` and every version site (first prep step, so the run reports 0.7.0) | `VERSION`, `pyproject.toml`, `frontend/package.json` and its lock, `docs-site/package.json` and its lock, the `OP_IMAGE_TAG` defaults in `docker-compose.yml`, `SCRIPT_VERSION` in `setup-openprocessor.sh`, the CLI default, the roadmap data (the list is in `RELEASE_CHECKLIST.md` step 1) | `.venv/bin/python -m pytest tests/test_version_consistency.py -q --no-cov` passes |
| RP-2 | Consolidate `CHANGELOG.md` `[Unreleased]` | `CHANGELOG.md` | one heading per type (today `[Unreleased]` repeats `Fixed`, `Added` and `Changed` several times); the "Breaking" entries of B1-B4 (registry migration, removed wire fields, new services, engine rebuild, `API_WORKERS`) gathered first under "Upgrade notes / behaviour changes", each starting "**Breaking:**" (the convention of `[0.6.0]`); the OCR behaviour changes of B0 listed there; a line saying that commits and PRs titled "v0.8.0 batch N" on `main` ship in 0.7.0. The heading `[Unreleased]` becomes `[0.7.0] - <date>` here (a new empty `[Unreleased]` above it), so the release commit only confirms the date |
| RP-3 | Consolidate `frontend/CHANGELOG.md` `[Unreleased]` | `frontend/CHANGELOG.md` | same rules |
| RP-4 | Release notes | `docs/releases/v0.7.0.md` (new, from `v0.6.0.md`'s layout) | highlights; the 0.6.0 -> 0.7.0 upgrade (the control-data move into Postgres and its report, the new services and secrets, the default detector engine rebuild, `API_WORKERS`, the downgrade note); the `small` profile note if #42 has not run on a 12 GB card (D-5); the verification summary is filled after the run |
| RP-5 | Release checklist for 0.7.0 | `docs/releases/RELEASE_CHECKLIST.md` | retitled to v0.7.0; the image list and the `images.lock` line count match the 0.7.0 image set (the export image, Postgres and Valkey are new since the 0.6.0 text: re-count from `scripts/lib/image_keys.sh` and the compose files) |
| RP-6 | Plan statuses | `docs/design/README.md`, `docs/design/program/00a_decision_register.md` (status lines) | statuses match what merged |
| RP-7 | Stale "known issue" notes (XS) | `docs-site/docs/operations/curation-workflow.mdx` (caution "a gated promote stops the retrain", #280, fixed by #302), `docs-site/docs/guides/pipeline-modes.mdx` section 4 (same), `docs/research/installer_drive_log.md` (the SIGINT row; fixed by `_op_on_signal` in `setup-openprocessor.sh`, which prints the resume hint). Pattern: the docs check lists issue links; no library | every "Known issue" note and every "(issue #N)" in `docs-site/docs`, `docs/` and `README.md` names an open issue (`gh issue list --state open`) or is removed. Red: a `rg` for the three strings finds them today and nothing after |
| RP-8 | Assist-loop live numbers (XS) | `docs/PERFORMANCE.md` (loop section) | the #286 closing evidence (runs, budget never exceeded, rollback 138/138, reproducibility, lock rule, yield per crop) is copied with its stack facts (`gh issue view 286`); docs only, no new measurement. Closes the S11 row of the auto-label plan |
| RP-9 | Retire `release_acceptance_plan.md` (removal wave R-2; XS; done after the R-final run PASSES, before the release commit, because the run still follows its method) | `docs/design/release_acceptance_plan.md`, `docs-site/docs/operations/release-acceptance.mdx`, `scripts/release/acceptance_run.sh` line 6 (a comment citing the plan; change the comment text only), `docs/reviews/rc_summary.md` line 15, `docs/design/README.md`, `docs/design/program/00a_decision_register.md` and this file (citations) | the run rules (one run per release) and the go/no-go list (old plan sections 1 and 5) live in `release-acceptance.mdx`; its phases D, E, H, I, K live in RF-70..RF-76 and 1B-12b; `git rm` the plan; `rg -l release_acceptance_plan.md --glob '!CHANGELOG.md'` lists only plain-text history; the docs check and the cited-paths hook pass. Pattern: `git rm` after the move ([12_prior_plans_reconciliation.md](12_prior_plans_reconciliation.md) section 6) |

Gate: `.venv/bin/pre-commit run --all-files`, `tests/test_version_consistency.py`, the changelog
rules, `.venv/bin/python scripts/docs/check_docs_vs_code.py`, `make contracts-check`; CI green;
merge commit into `main`. No full offline suite (no code changes); if RP-1 touches code beyond
version strings, run it.

## 4. The checks (union of R3, R-B1 and R-0.9)

Every row is recorded as PASS, FAIL, PASS-WITH-FINDING or BLOCKED (with the reason), with wall
time and an evidence path. A failing row becomes a finding; never relax a criterion.

### 4.1 Procedure checks (the R2 procedure, kept in full)

| ID | Check | How | Pass |
|---|---|---|---|
| RF-01 | Fresh install (Install A) | installer from the local bundle into `<scratch>/rfinal/a`, compose project `op-rfinal`, tiers core, curation, vlm, trainer, cropwright, segmenter | exit 0; `/health` ready; every required model `READY`; VLM healthy |
| RF-02 | Acceptance phases | `scripts/release/acceptance_run.sh` with every phase (`--list` prints them: health, resource_links, docs_selfhosted, project_create, ingest, detect_embed, cluster, vlm_label, confirm_labels, holdout, label_ops, loop, export, train, bakeoff, promote, infer, model_delete, metrics, log_noise, vlm_switch, installer_rerun, installer_repair, installer_upgrade, installer_uninstall, teardown) and `--vlm-switch-id`, `--upgrade-version`, `--installer-args "--release-dir <bundle> --image-tag <tag>"` | every required phase PASS; optional phases PASS (no skip except as stated); `installer_uninstall` run last with `--only installer_uninstall` |
| RF-03 | Live API suite | `docker compose -p op-rfinal-live -f docker/test/compose.yml up -d --wait`, `.venv/bin/python -m pytest tests/live -q --no-cov -m live`, then `down -v --remove-orphans` | 0 failed, 0 unexplained skips |
| RF-04 | Frontend live | from `frontend/`, real `npm ci`; `OP_CROPWRIGHT_LIVE_URL=<install A cropwright>` `npm run test:live` on a project with region data from the segmenter | 0 failed; the 8 region tests that R2 skipped now run |
| RF-05 | Full system | `.venv/bin/python tests/test_full_system.py` against install A, from a scratch copy whose `test_images/` is writable (R2 found the repo's `test_images/` root-owned) | 0 failed; skips only the two by-design ones |
| RF-06 | Visual results | `.venv/bin/python tests/validate_visual_results.py`, then open every image in `test_results/` | all pass; images viewed and plausible |
| RF-07 | Screenshots | full-page captures at 1440 and 430 px of every project page and the projects list, each opened and looked at | no page-level overflow, no page errors; the pages changed in B1-B4 are opened individually (export stats, diverse review, the `queued` job status, the conflicts display of bulk review, the `building` state of a project being created, every live (SSE) page after a reconnect) |
| RF-08 | Image audit | `scripts/release/audit_image_files.py` for every built image (run by the build, then once more independently) | exit 0 for every image |
| RF-09 | Trivy | the build's gate (`--report-dir`), CRITICAL fixable outside `scripts/release/trivy-allowlist.txt` | 0 for every image; HIGH counts recorded |
| RF-10 | Token-leak grep | `rg -n -i "hf_[a-z0-9]{10}|ghp_|sk-[a-z0-9]{10}|Bearer [a-z0-9]|AKIA|(password|secret|token)=\S" <evidence dirs> <install logs> <acceptance workdir>` | no secret value (doc comments and bundled JS strings are noted, not failures) |
| RF-11 | Peer isolation | `docker ps -a`, `docker volume ls`, `docker network ls`, `docker image ls`, `nvidia-smi` before and after, saved as `snap_before_*`/`snap_after_*` | no change to any container, volume, network or image that is not `op-rfinal*`; differences in other repos' stacks are explained as their own restarts |
| RF-12 | Uninstall | `./openprocessor uninstall --yes --purge-volumes` (A and B) | no `op-rfinal*` container or volume remains |

### 4.2 Old R3 checks (B0 and the R2 round)

| ID | Check | How | Pass |
|---|---|---|---|
| RF-20 | Ultralytics pin | `python -c "import importlib.metadata as m; print(m.version('ultralytics'))"` in the API, export and trainer images | all three print the B3 pin X (D-28); `/opt/venv-y11` does not exist in the export image (B3 deletes it in Stage A and Stage B) |
| RF-21 | uvicorn floor and jitter | the same for `uvicorn`; `docker inspect` of the api command | version >= 0.41.0; `--limit-max-requests-jitter` present; `--workers` absent and `WEB_CONCURRENCY` set from `API_WORKERS` (B4) |
| RF-22 | Segmenter tier (D-4) | segmenter tier installed; HF token from the stack's `.env`, passed without printing it; segmenter on GPU 0 (GPU 2 only if free and agreed) | segmenter healthy; a region profile that uses it runs on a COCO subset; the region-dependent acceptance phases and RF-04 region tests pass |
| RF-23 | `--skip-models` live | fresh install with `--skip-models` in `<scratch>/rfinal/skip` | exit 0; OpenSearch healthy; the message says models were skipped; then `models install --yes` and `start` reach ready |
| RF-24 | Engine adoption | a second fresh install dir seeded with install A's engines and `.install/precision.tsv` | every group adopted or "up to date, skipped"; nothing rebuilt (R2 rebuilt all 8 in about 30 min) |
| RF-25 | Triton down vs engine invalid | stop Triton during `models install` | a clear "Triton unreachable" stop; no rebuild |
| RF-26 | VERSION | `GET /health` | reports `0.7.0` |
| RF-27 | OCR fixes | `/ocr` on the B0 OCR image set (ids in the private evidence, rule in [13_b0_close_out.md](13_b0_close_out.md) B0-3) | boxes on the text; rotated frames match `image_size`; RGBA composited on white |
| RF-28 | Holdout freeze | acceptance `holdout` phase | PASS (the fail-closed path is covered offline) |
| RF-29 | Cleanup cancels jobs (R2 fix (e), never verified live) | start a job that outlives the cleanup window (dataset import of the 2k COCO set, or a train with more epochs), then run the acceptance cleanup | the job ends `cancelled` in its worker and the project deletes |
| RF-30 | Assist-loop admission (R2 oddity: budget spent, nothing visited) | UI loop with `min_root_size 10`, budget 120 | every spent crop belongs to a node with a decision; unvisited nodes carry `reason_message` |
| RF-31 | #437 | promote phase yields a YOLO26 model; point a region profile at it | region candidates decode (`[B,300,6]`) |
| RF-32 | #442, #443 | activate a VLM endpoint right after creating it; ingest while cluster refresh runs | 200 on activate; the log shows the `recent_ingest` hold |
| RF-33 | #440 | browser at 1440 and 430 px | the ingest-profile list and activate work, the refusal report shows, cards readable at 430 |
| RF-34 | SEC-1, SEC-2 | the branch tests named in the maintainers' private notes, plus the positive control (a legitimate model validates and exports) | pass |
| RF-35 | Status polls under load (CC-1) | poll a running job's status from a script while ingest runs | no 500; a stale job's `state.json` mtime unchanged by polls |
| RF-36 | BA-7 | stop OpenSearch, call `review_dismiss` | 503, not 404 |

### 4.3 Old R-B1 checks (B1)

| ID | Check | How | Pass |
|---|---|---|---|
| RF-40 | Upgrade from 0.6.0 with real class registries, configs and an in-flight job | Install B (section 5.2) | the control-data step ran from `openprocessor upgrade` ([02c](02c_migration_ops_tests.md) section 2): the bundle and its report are under `.install/`; every project's class ids and names, config revisions and activations are equal before and after; the OpenSearch control indexes are gone; `class_registry.json.migrated-0.7.0` exists; the in-flight job reads `interrupted` |
| RF-41 | Upgrade stops on a failed verify | Install B, a second project whose source data is made inconsistent before the upgrade (the offline fixture's forced-mismatch case) | the upgrade stops at verify with the report path; nothing was deleted from OpenSearch; `--rollback` returns to a working 0.6.0 |
| RF-42 | Job busy and takeover (rewritten after review R-23: jobs now run in `jobs-worker`, so killing the API no longer kills a job) | two concurrent POSTs that start one singleton job (for example item scores) on install A; then `docker kill` the `jobs-worker` replica running it | exactly one start, the other 409 `job_busy`; the job is `interrupted` within 150 s (`LEASE_TTL_S` 120 s plus one 30 s reaper tick, [02b](02b_jobs_locking_events.md) section 4) and then resumes or can be restarted per its kind; a new start is accepted once no job of the group is active |
| RF-43 | Registry concurrency | 8 parallel `add_class` calls with distinct names, then 8 with one shared name | 8 distinct ids; the shared name yields one id |
| RF-44 | Filtered kNN (OS-6) | a curation search with a `class_name` filter and `k=50` on a project with at least 50 matches | 50 hits returned |
| RF-45 | B1-1D pages | export page stats refresh; diverse review loads with one batch read (browser network panel) | stats change after a label; one `?ids=` request instead of 30 |
| RF-46 | `families` CI job | the CI run of the release-prep merge | green |

### 4.4 Old R-0.9 checks (B2, B3, B4)

| ID | Check | How | Pass |
|---|---|---|---|
| RF-50 | Broker hardening | `docker port` of `redis`; mode of the generated password file; `CONFIG GET` of `appendonly`, `appendfsync`, `maxmemory-policy` through `docker compose -p op-rfinal exec redis` | no published port; mode 600; AOF on, `everysec`, `noeviction` |
| RF-51 | Worker services | `docker compose -p op-rfinal ps` | `jobs-worker`, `open-vocab-worker`, `periodic-worker`, `jobs-beat` healthy; one `jobs-beat` only; Flower only under the monitoring profile |
| RF-52 | Jobs survive an API recycle (CC-4) | start a dataset import; `docker compose -p op-rfinal restart api` | the job completes; its record shows one run |
| RF-53 | Worker kill mid-job | `docker compose -p op-rfinal kill jobs-worker` during a resumable job, then start it again | the reaper re-enqueues it; the record shows `attempt=2`; it completes |
| RF-54 | `queued` in Cropwright (Q-E14) | start a job while the worker is busy | the UI shows `queued`, then `running` |
| RF-55 | One periodic run | `jobs-beat` log over 10 minutes | each periodic task fires once per interval, not once per API worker |
| RF-56 | Upgrade adds the control plane (M11) | Install B after the upgrade | the Postgres, Valkey and Flower secret files exist with mode 600 (created by `upgrade`); `postgres`, `valkey`, the workers and beat are healthy; `db-migrate` exited 0; the env keys are present; the installer printed the interrupted jobs; a `pg_dump` was taken before the step |
| RF-57 | Engine contract (B3) | Triton model metadata of the default detector; `/detect` on a COCO image; a region profile still pointing at a raw-head engine on install B | `output0 [-1,300,6]`; boxes returned; the raw-head profile fails closed with the documented re-export command |
| RF-58 | Engine rebuild on upgrade (U-R3) | Install B upgrade log | the default detector engine rebuilt once (stamp changed); other groups skipped |
| RF-59 | `API_WORKERS` (B4) | `.env` of install B after the upgrade; install A's `.env` | the key is present with the D-33 default |
| RF-60 | Events (B4-4C) | a live page in the browser; restart the API; watch the page | the page reconnects and resumes from `Last-Event-ID` with no lost or duplicated event |
| RF-61 | Queue outage | stop `redis` for 30 s | `/ready` answers 503 naming `queue`; the UI shows the outage; job starts answer 503; recovery after restart (D-69) |
| RF-62 | Rollback to 0.6.0 after the control-data move | `setup-openprocessor.sh --dir <B> --rollback --yes` on install B after RF-40 | 0.6.0 API healthy; no 0.7.0-only container (Postgres, Valkey, workers, beat) left running; the control indexes and job files recreated from the bundle; the projects' classes and configs readable without a manual step; then a re-upgrade succeeds |
| RF-63 | Re-upgrade (reworded after review R-9: rollback abandons the Postgres data, so this is a fresh upgrade) | upgrade install B again after RF-62 | the same counts and ids as the first upgrade; `upgrade_state` completed; no `upgrade_incomplete` refusal |

### 4.5 Checks carried from `release_acceptance_plan.md`

Phases D, E, H, I, K and the notes of the old plan that no scripted phase covers
([12_prior_plans_reconciliation.md](12_prior_plans_reconciliation.md) section 2.20).

| ID | Check | How | Pass |
|---|---|---|---|
| RF-70 | Dataset import | `acceptance_run.sh --only dataset_import` (script task 1B-12b in B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md))) on install A | PASS |
| RF-71 | Combine | `acceptance_run.sh --only combine` (1B-12b) | PASS; both sources unchanged |
| RF-72 | Project isolation, live | two projects with data on install A; follow the event stream of one while ingesting into the other; `docker compose -p op-rfinal logs api` and the metrics endpoint filtered by the first project's slug | nothing of the second project appears in the first's stream, its log lines or its labelled metrics |
| RF-73 | Day-2 operations | a reprocess over items with a human label; archive the protected default project; delete a project while a job runs; with the monitoring tier on, open a dashboard, a log search and the experiment-tracking run of the `train` phase | reprocess reports the skipped locked items and changes none; the guards refuse; the three UIs show this run's data |
| RF-74 | Train mode frees the GPU | `./openprocessor train-mode on` before the `train` phase, `nvidia-smi` before and after; `train-mode off` after | the segmenter and VLM stop and at least 16 GB is free on the training GPU; both come back healthy |
| RF-75 | Monitoring credential | install with the monitoring tier | if the installer does not ask for a new admin password, record a neutral security finding (details in private notes) |
| RF-76 | Docs site | `cd docs-site && npm ci && npm run build`; serve it; open getting-started, quick-start, ingest-large-folder, pipeline-modes, release-acceptance and one page per guide group at 1440 and 430 px | build passes; pages render; screenshots viewed |

Run RF-70 and RF-71 with the install A phases (step 3 of section 5.2), RF-72 to RF-75 as extra
live checks on A (step 4), RF-76 with the suites (step 5).

### 4.6 B5 speed and parity rows

D-37 puts B5 inside this release, so the run records, without re-measuring the whole baseline,
the rows that guard it: the parity gates P-box, P-vlm, P-sam, P-region, P-ocr and P-pe of
[06_b5_overview.md](06_b5_overview.md) on the candidate images (the sub-batch records are the
reference; a gate that regressed is a FAIL), and the ingest and search figures of
[06a_b5_baseline.md](06a_b5_baseline.md) re-taken once on install A at the shipped
`API_WORKERS` default (no regression against the B5 result). These rows use GPU window 3 and no
new tooling.

### 4.7 Research-seam rows (approved by the owner 2026-11-04, RD-1 of [00d](00d_owner_questions.md))

Public data only (COCO, Open Images, constructed copy sets). Each row reads the record in
PostgreSQL, not a log.

| ID | Check | How | Pass |
|---|---|---|---|
| RF-80 | The assist loop as a controllable assistant | on install A, a loop on a public COCO project: pause, resume three times, lower the budget, cancel; a second project's loop on the one consumer | each pause lands within one request timeout plus 1 s; the fourth resume runs; the budget stop is `paused` with reason `budget`; project B starts within one slice; `vlm_usage_hourly` holds the crops |
| RF-81 | Clustering partitions | a `cluster_run`, its activation and projection; the 0.6.0 upgrade's `upgrade` partition on install B | the served view equals the upgraded assignment; ids survive a re-run on permuted input |
| RF-82 | Near-duplicates and the holdout | a full build on a constructed public copy set, then a freeze | every leak-group member of a held-out item is held out and counted in `holdout_near_dup_moves`; a freeze with pending rows answers 409 `near_dup_stale`; on install B, after the `source_backfill` job, a re-ingest of an upgraded image is refused as an exact duplicate (review RR-10) |
| RF-83 | Embedding registry | a new project has three primary ready spaces; an upgraded project records the live `knn_method`; a search with a wrong `space` refuses | as stated |
| RF-84 | Capture metadata (W4) | ingest the EXIF fixture set (orientations 1-8, offsets, a GPS fixture) | `capture.*` correct; `has_location` true; under the default `location_policy = store` the image `capture.location` holds the fixture's coordinates and no item, manifest, training file or log holds one; after a switch to `drop` the purge leaves no coordinate in the images index; an exported copy of the GPS fixture keeps its GPS block under `store`, as today (owner 2026-11-04, RD-51 reversed, RD-54), and has none under `drop` (CM-5, review RN-4); a resized export of an orientation-6 fixture lands its boxes on the object (W5b) |
| RF-85 | Registration-plate text (W5e) | read a public plate sample, run consensus on a constructed public burst (re-encoded and cropped copies of one Open Images plate; UFPR-ALPR is skipped, RD-10), search exact, fuzzy and fragment, switch to keyed mode, purge | readings and tiers written; keyed mode shows no text; the purge leaves 0 values in every sink |
| RF-86 | Serving profile and engine contracts | `openprocessor gpu plan` and the active serving profile on install A | every ready engine row has a contract and a measured device memory; `serving_reports` lists every loaded model |

Run RF-80 to RF-86 with the install A phases (RF-81's upgrade half on install B), in GPU window 3.

## 5. Procedure

Evidence root: `artifacts_local/rfinal/`. Report: `artifacts_local/rfinal/REPORT.md`, written
as the R2 report was (entry SHA, stack, per-step results, findings, hygiene, recommendation).

### 5.1 Environment

| Item | Value |
|---|---|
| Entry SHA | `git -C <repo> rev-parse origin/main` after the release-prep merge |
| Compose projects | `op-rfinal` (A), `op-rfinal-b` (B), `op-rfinal-skip`, `op-rfinal-adopt`, `op-rfinal-live` |
| Install dirs | `<scratch>/rfinal/{a,b,skip,adopt}` |
| Ports | one port base per install, none used by a peer (check `ss -ltn`; R2 used 5200) |
| GPUs | GPU 0 for every service, `CUDA_DEVICE_ORDER=PCI_BUS_ID`; GPU 2 only for the segmenter, only if free and agreed |
| Images | `scripts/release/build_and_publish.sh --dry-run --version v0.7.0 --local-tag-suffix -rc<N> --report-dir artifacts_local/rfinal/trivy`; local tags only; never a public tag (watchtower) |
| Bundle | the local release bundle as R2 built it, plus a copy named with the next rc suffix for the upgrade/rollback phase (`docs/design/release_acceptance_plan.md`) |
| Data | public COCO via `scripts/datasets/fetch_coco_subset.py`; no private data, names or projects anywhere |
| Budget | about one working day; stop and report if a step exceeds twice its R2 time |

### 5.2 Steps

1. **Entry.** Record the entry SHA, `uptime`, `nvidia-smi`. Peer snapshots before (RF-11).
2. **Build.** Images and bundle (5.1). The Trivy gate and the image audit run inside the build;
   keep their output (RF-08, RF-09).
3. **Install A** (RF-01), then the acceptance phases (RF-02) except the installer lifecycle and
   uninstall, plus `--only dataset_import,combine` (RF-70, RF-71).
4. **Extra live checks on A:** RF-20 to RF-36, RF-42 to RF-45, RF-50 to RF-55, RF-57, RF-59 to
   RF-61, RF-72 to RF-75, and the section 4.6 rows. RF-27 is the place where the B0 OCR
   comparison evidence of GPU window 1 is checked again; if window 1 could not run it, run the
   B0-3 procedure here.
5. **Suites on A:** RF-03 (the live harness, once, as part of the R-final run itself; the three
   batch runs of the budget are B1 smoke, the B1+B2 run and B4), RF-04, RF-05, RF-06, RF-07, RF-76.
6. **Install B: the supported upgrade.**
   - Install the published v0.6.0 with its own published installer and images (pulling the
     public 0.6.0 images is correct here; nothing local is tagged with a public name) into
     `<scratch>/rfinal/b`, compose project `op-rfinal-b`.
   - Seed real class registries through the 0.6.0 API: two projects with public COCO images,
     classes created, renamed and one merged (so ids have a gap), labels confirmed, one export.
     Record each project's class list with ids. Start one resumable job so it is in flight at
     the upgrade.
   - Upgrade to the candidate: `./openprocessor upgrade --yes --release-dir <bundle> --image-tag <tag>`
     (the flags R2 used for its rc upgrade).
   - Run RF-40, RF-41, RF-56, RF-57 (raw-head profile), RF-58, RF-59, then the acceptance phases
     on a fresh project in B with `--only ingest,cluster,vlm_label,confirm_labels,holdout,label_ops,export,train,promote`
     (there is no `class` phase; classes are exercised inside these).
   - RF-62 (rollback to 0.6.0), then RF-63 (re-upgrade).
7. **Installer lifecycle on A:** `--only installer_rerun,installer_repair,installer_upgrade`,
   then `--only installer_uninstall` last (RF-02, RF-12).
8. **Side installs:** RF-23 (`--skip-models`), RF-24 (engine adoption), RF-25 (Triton down).
9. **Hygiene:** RF-10 token grep, RF-11 snapshots after and the diff, RF-12 uninstall of every
   `op-rfinal*` install with `--purge-volumes`, removal of the local tags (keep the newest rc set
   until the tag decision).
10. **Report** to the owner: PASS/FAIL per row, findings, the recommendation, the evidence path.

## 6. Fix loop

- A failure is fixed on `fix/rfinal-findings` (worktree `<repo>-wt-rfinal-fix`, from `main`):
  red/green test per fix, targeted tests and static checks only, one PR, merge commit.
- If a fix changes an image, rebuild with the next `-rc<N>` suffix and re-run the image audit and
  Trivy for the rebuilt images.
- Re-run only the failed rows and the rows that depend on the changed code; record both runs in
  the report.
- Nothing else merges to `main` while R-final runs.

## 7. Tag and publish (only on the owner's explicit word, D-16)

Follow `docs/releases/RELEASE_CHECKLIST.md` (updated in RP-5). The order:

1. **Release commit** on `release/v0.7.0` from `main` (RP-9 done first): the `[0.7.0]` headings
   made in RP-2 and RP-3 get the final date; the CHANGELOG header line "This release is
   `[0.6.0]`" -> `[0.7.0]` (if RP-2 has not done it); the verification summary in
   `docs/releases/v0.7.0.md` filled from the report.
2. **Full gate on the release commit** (checklist step 1): pre-commit `--all-files`, the offline
   suite with `-n 8`, `generate_contracts.py --check`, frontend `npm ci && npm run check && npm test && npm run build`
   plus `npm run test:e2e`, `docs-site` `npm ci && npm run build`.
3. `make release-verify` (with `OP_PROJECT=op-relverify` on this host).
4. `make release-dry-run`: builds every image, Trivy gate, image audit. **The image-vs-git audit
   is never skipped before a push** (the 0.6.0 image shipped untracked files).
5. `make release`: pushes `davidamacey/<image>:0.7.0` (plus `latest`) and writes `images.lock`.
   Check `rg 'dev[0-9a-f]$' images.lock` is empty and the line count matches RP-5.
6. Commit the lock: `chore(release): pin v0.7.0 image digests`.
7. `scripts/release/build_deploy_bundle.sh v0.7.0`; `sha256sum --check --ignore-missing SHA256SUMS`
   in `dist/release-v0.7.0/`.
8. Merge `release/v0.7.0` into `main` with a merge commit (`--no-ff`) and push.
9. Annotated tag: `git tag -a v0.7.0 -m "OpenProcessor v0.7.0"`, `git push origin v0.7.0`.
10. Verify the push by the remote, never a piped exit code: `git ls-remote --tags origin v0.7.0`
    lists the tag object and `git ls-remote origin 'refs/tags/v0.7.0^{}'` equals the release
    commit.
11. Publish immediately (not a draft):
    `gh release create v0.7.0 --title "OpenProcessor v0.7.0" --notes-file docs/releases/v0.7.0.md dist/release-v0.7.0/openprocessor-deploy-v0.7.0.tar.gz dist/release-v0.7.0/SHA256SUMS dist/release-v0.7.0/setup-openprocessor.sh dist/release-v0.7.0/release-manifest.txt`.
12. Post-release verification (checklist section): `gh release view v0.7.0` (four assets, not a
    draft); fetch the published installer and `SHA256SUMS`, check them, `--dry-run` install, then
    a real install in a scratch dir with its own `-p`; `docker pull davidamacey/openprocessor:0.7.0`
    digest equals the `api` line in `images.lock`.
13. GitHub: close #411 with the release link and the report location; close milestone v0.7.0 once
    it has no open issue ([09_github_actions.md](09_github_actions.md) section 6).

The organisation transfer (checklist step 11) is not part of this release unless the owner asks.

## 8. After the tag

There is no second acceptance for B5 (D-37). Milestone v0.8.0 (platform breadth) starts after the
tag; its issues are listed in [09_github_actions.md](09_github_actions.md) section 7. Side
task #42 (12 GB card) runs when GPU 1 frees and never blocks the tag.

### What to keep after the release

Per [00a_decision_register.md](00a_decision_register.md) section 8.2, after the tag the whole `program/` folder is removed
(`git rm -r docs/design/program`) and its rows leave `docs/design/README.md`, after this keep-list
is moved:

| Fact or section | Moves to |
|---|---|
| the RF rows that became standing checks (RF-01..RF-12 procedure, RF-70..RF-76) | `docs-site/docs/operations/release-acceptance.mdx` |
| the release procedure (section 7) | `docs/releases/RELEASE_CHECKLIST.md` |
| the final report summary and gate numbers | `docs/releases/v0.7.0.md` and `docs/PERFORMANCE.md` |
| the upgrade and downgrade notes (the control-data move, Postgres, Valkey, `API_WORKERS`) | `INSTALLATION.md` and the release notes |
| the library register and standards of file 10 | `docs/design/engineering_standards.md` |
| the host and GPU etiquette of file 11 | `docs/design/engineering_standards.md` (a short section) |

## 9. Rollback

- Before the tag: nothing is published; fix forward with `fix/rfinal-findings` or revert the
  offending merge on `main`.
- After publishing: never move a published tag. A defect becomes a 0.7.1 with its own short
  acceptance (the failed rows plus RF-01, RF-02, RF-08, RF-09). Users roll back with
  `setup-openprocessor.sh --rollback` and the downgrade note in the release notes.

## 10. Risks

| Risk | Mitigation |
|---|---|
| The single upgrade carries everything at once (the control-data move, Postgres, Valkey, new services, engine rebuild, `API_WORKERS`) | Install B tests exactly that path from a real 0.6.0 install with real registries; RF-62 tests the way back |
| Rollback after the control-data move loses classes in 0.6.0 | RF-62; the downgrade note in the release notes; B2 lane 2E owns the rollback ([03b](03b_stage2_control_data.md) 2E-4, [02c](02c_migration_ops_tests.md) section 2.3; re-pointed from B1-1B, review R-23) |
| a 0.6.0 API started before the rollback re-index creates the control indexes empty | the rollback order of 02c section 2.3 (OpenSearch only, recreate, verify, then 0.6.0 services); RF-62 checks it; if it fails, 2E fixes the step and RF-62 re-runs |
| `./openprocessor upgrade` in a 0.6.0 install runs the 0.6.0 CLI, which must hand off to the 0.7.0 installer for the upgrade step to run | 2E-5 tests the hand-off from a 0.6.0 install offline (the v0.6.0 bootstrap downloads the newest installer, review section 3.1); RF-40 proves it live (the upgrade report exists under `.install/`) |
| Host load skews the run or delays it (M1) | one heavy run at a time; peers told first; budget per step |
| A local tag is clobbered by a public pull (watchtower) | local-only names for every candidate image |
| A secret appears in evidence or the report | RF-10; HF token passed without printing it |

## 11. Ready-to-paste brief: R-final runner (Sonnet)

```
You run R-final for OpenProcessor. Plan: docs/design/program/07_release_acceptance.md, sections 4
and 5. Entry: origin/main after the release-prep merge (record the SHA). You do not tag, push
images or publish anything; you report.
- Compose projects op-rfinal, op-rfinal-b, op-rfinal-skip, op-rfinal-adopt, op-rfinal-live; always
  `docker compose -p <name>`. Install dirs under <scratch>/rfinal/. GPU 0 only (GPU 2 for the
  segmenter only if free and agreed with the peer sessions).
- Before starting and before every heavy step: ListAgents, message every peer session that uses
  the GPUs, wait for "go"; record uptime and nvidia-smi.
- Local-only image tags (-rc<N>); never a public tag. Public COCO data only.
- Snapshot docker state before and after; never touch a container, volume, network or image that
  is not op-rfinal*.
- Never use pgrep -f, pkill -f or sleep/until loops; wait on healthchecks or the Monitor tool.
- Record every RF row as PASS, FAIL, PASS-WITH-FINDING or BLOCKED with time and evidence path in
  artifacts_local/rfinal/REPORT.md. Never relax a criterion. Pass the HF token without printing it.
- Deadline: the report within two working days of the "go".
```
