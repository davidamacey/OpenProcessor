# Program plan: every open issue and research item, through v0.9.0 and the white paper

Status: plan (2026-10-28, written against `main` at `96886e0a`). Owner: program coordinator. This
file is the single ordered program for completing **every open GitHub issue** (34 at the time of
writing) and **every research item** (Triton and serving, clustering, the optional embedding model,
spatial memory and photogrammetry), ending in a public white paper. A fresh agent with no memory
can take any phase below and execute it: each task names its tier, its verify command, its expected
number and where the number is recorded.

The winter goal that orders everything: ingest a large share of the owner's private high-resolution
photo collection (**set P**: about 21 MP JPEGs, about 9.5 MB per file, read from the NAS), cluster
it, label it with the auto-label loop, review, export, and train a better vehicle detector. Phase A
delivers that; later phases proceed in parallel branches and never block it.

Read first, in this order:

1. `CLAUDE.md` and [`docs/VISION_AND_GOALS.md`](../VISION_AND_GOALS.md) (standards).
2. [`v070_wave_plan.md`](v070_wave_plan.md): ground rules (section 0), test tiers (2.0), winter
   rehearsal budgets (section 3). This plan reuses them and does not restate them.
3. `docs/design/auto_label_loop_plan.md` (on branch `feat/v070-assist-loop`, worktree
   `OpenProcessor-wt-loop`; merges with the loop): build order S0 to S11, test plan, Q1 to Q9.
4. [`triton_pipeline_optimization_plan.md`](triton_pipeline_optimization_plan.md) waves 6 to 9,
   [`v060_serving_backend_trade_study.md`](v060_serving_backend_trade_study.md), and the roadmap
   part (section 29 to 33) of
   [`triton_deep_research_2026-09.md`](../research/triton_deep_research_2026-09.md).
5. [`cluster_types.md`](../research/cluster_types.md), [`clustering_methods.md`](../research/clustering_methods.md),
   [`cluster_eval_baseline.md`](../research/cluster_eval_baseline.md).
6. `docs/research/embedding_model_eval_plan.md` (branch `research/v080-347-embedding-eval-plan`).
7. [`OpenProcessor_Spatial_Memory_Photogrammetry_Research_README.md`](../research/OpenProcessor_Spatial_Memory_Photogrammetry_Research_README.md)
   sections 9, 11, 13, 16 and 18.
8. [`v070_measurement_log.md`](../research/v070_measurement_log.md) (the running evidence log) and
   `docs/research/CLAUDE.md` plus `docs/research/README.md` (branch `docs/research-index`, merged
   through `chore/v070-followups`): rules for every research note.
9. [`docs/market/09-gap-analysis-and-positioning.md`](../market/09-gap-analysis-and-positioning.md)
   and [`06-feature-comparison-matrix.md`](../market/06-feature-comparison-matrix.md) (paper
   positioning).

## 0. Binding rules (every phase, every agent)

| Rule | Content |
|---|---|
| Isolation | Never `docker compose` without `-p <name>` and own ports; `--container-prefix <name>` for `scripts/bench/scale_run.py`. The default project `openprocessor` and the live test stack are someone else's. Preflight: `nvidia-smi`, other containers, timed `curl` of `/health`. |
| GPUs | GPU 0 and GPU 2 (A6000) only. **GPU 1 is never used** (other users). GPU 2 holds a live test stack: a GPU 2 run or a two-card run needs an owner window. **One GPU job at a time**: every GPU 0 job runs as `flock -w 3600 "$SCRATCH/locks/gpu0.lock" <cmd>` (the shared lock directory every agent agrees on), per job, never per session. |
| Decode | One decode per image per request or pass. Any new model path (embedding arms, spatial descriptors, local features) consumes the decoded frame from `gpu_image_prep` or the pass's single decode; a test proves it (decode counter equals images). |
| Fail closed | An unbound project, a stale registry, an ambiguous state or a partial pass refuses or reports non-success. |
| Integrity | Lock rule (`src/clients/occ_locks.py`): automated writers never overwrite human-validated, import-validated, holdout or excluded items. Class identity is the name at every boundary. Writes through the OCC helpers. |
| Code | No dead code, no shims, superseded paths deleted in the same change. Route or wire change: `make contracts` and `contracts/` in the same commit. Retired env vars go into `src/config/retired_env.py`. |
| Data | Committed text uses public data only (COCO, Open Images, Commons set B, the public multi-view sets of section 7). Set P (and the second owner library, **set L**, if the owner allows its use) is described by numbers only: no paths, folder or file names, thumbnails, places or people. Raw private results stay in `artifacts_local/` (gitignored). NAS data is **read-only, never deleted, never modified**; copies go to NVMe working directories. `scripts/codegen/check_naming_leaks.py` must pass. |
| Commits | Small logical conventional commits (`feat(scope): ...`), one branch per task bundle, merged with `--no-ff` (never squash). `.venv/bin/pre-commit run --all-files` before every commit; never `--no-verify`. Red-first: a bug-fix test is seen failing before the fix. No push, PR, tag or release without the owner's yes. |
| Venv | New worktree: create `.venv` as CONTRIBUTING.md shows; never install `requirements.txt`. Call `.venv/bin/...` directly. |
| Measurement | `docs/research/CLAUDE.md` rules: never invent a number, 3 repetitions on the headline row, one factor at a time, load average recorded, raw JSON kept (`artifacts_local/bench/<phase>/<task>/`; small public result files that back a published number go to `docs/benchmarks/`), reproduce command in the note. |

Test tiers are those of `v070_wave_plan.md` 2.0, referred to below as **T0** (unit + import +
pre-commit), **T1** (live write-path harness `docker/test/compose.yml`, no GPU), **T2**
(rehearsal-lite, one A6000, 40 min), **T3** (headline benchmark, 3 reps, owner window) and **Full**
(`.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'`, once per bundle). Shorthands used in
verify columns:

```bash
T0="pytest <touched tests> -q --no-cov && .venv/bin/python -c 'import src.main' && .venv/bin/pre-commit run --all-files"
T1="docker compose -p op-live-verify -f docker/test/compose.yml up -d --wait && .venv/bin/python -m pytest tests/live -q --no-cov -m live; docker compose -p op-live-verify -f docker/test/compose.yml down -v --remove-orphans"
DOCS=".venv/bin/python scripts/docs/check_docs_vs_code.py && make docs-build"
CONTRACTS="make contracts && make contracts-check"
```

### 0.1 Definition of done (DoD), applies to every issue and research item

An item is done only when all eight hold; the auditor ticks them in the completion matrix (section 3):

1. **Code** merged to the release branch (or `main`) with `--no-ff`, small conventional commits.
2. **Unit** tests: offline, red-first for fixes, at least one test that would fail if the feature were
   removed (mutation check: revert the core line, the test fails).
3. **Live** test: T1 (or a GPU live test for GPU paths) covering the behavior end to end.
4. **Acceptance** numbers of the issue met, measured on the stated data.
5. **Measurements** recorded in a `docs/research/` note (setup, method, results, findings, reproduce)
   with raw JSON kept; public headline rows also in `docs/PERFORMANCE.md`.
6. **Docs**: docs-site page(s) where user or operator behavior changes, design doc status updated in
   [`README.md`](README.md), `DOCS` passes.
7. **CHANGELOG** `[Unreleased]` entry (upgrade note if behavior changes).
8. **Closed** on GitHub with a comment linking the merge commit, the note section and the raw JSON
   location, after the phase audit passed.

### 0.2 Model tiers

| Tag | Model | Used for |
|---|---|---|
| `[S]` | Sonnet | implementation, tests, runs, docs, ordinary reviews |
| `[H]` | Haiku | lookups, contract and link checks, size and naming scans, collecting JSON, running fixed commands |
| `[O-design]` | Opus | design review **before** a risky stage is built (section 2.1) |
| `[O-review]` | Opus | code review of concurrency, data integrity, isolation, security, migrations; math review |
| `[O-audit]` | Opus, read-only | the audit gate at the end of every phase and before every release (section 2.2) |

## 1. State on 2026-10-28

Merged on `main` (release/0.7.0, under `[Unreleased]`): cluster evaluation harness and baseline
(#341 code), IO-pressure sampling (#337), embed pass on the GPU decode and chunk overlap (#259,
closed), clamped boxes (#259), region vectors from decoded pixels (#360), content-hash ids (#358),
per-project crop cache (#359), two `gpu_image_prep` instances on `full` (#268 part), worker-start
bootstrap and warm-up (#356 part), `auto_label` idle fix (#350).

In progress in sibling worktrees (do not edit them from another task):

| Branch | Worktree | State |
|---|---|---|
| `feat/v070-assist-loop` | `OpenProcessor-wt-loop` | #286 S0 to S6a done (stats, tree, propagated kind, shared class writer, `ask_sample`, partition and splitter, job store and budgets); S6b to S11 open |
| `fix/v070-362-work` | `OpenProcessor-wt-362` | one decode per photo per flush and the `output0` activation check committed; re-measure and merge open |
| `fix/v070-357-work` | `OpenProcessor-wt-357` | partial pass reported as partial committed; the 250k reset failure open |
| `chore/v070-followups` | `OpenProcessor-wt-fu` | installer fixes, research index, loop design and embedding plan merged in; to merge to main |
| `research/v080-342-leiden-prototype` | `OpenProcessor-wt-342` | prototype scripts staged under `scripts/research/` (graph lib, Leiden, baselines, planted partitions, GPU methods); results note not written |
| `docs/v070-335-screenshots`, `fix/v070-341-work` | | empty (no commits beyond main) |

## 2. Governance: Opus design reviews and audit gates

### 2.1 Design review (DR) before a risky stage is built

A DR is an `[O-design]` agent that reads the plan section, the code it touches and the tests
proposed, and writes its verdict as a section `Design review` appended to the relevant design doc
(go, go-with-changes listing each change, or no-go). Implementation starts only after go. DRs in
this program:

| DR | Before | Focus |
|---|---|---|
| DR-A1 | loop S6b/S6c (driver, propagate, audit, rollback) | lock rule, OCC, budget write-ahead, drift, resume idempotency, rollback exactness |
| DR-A2 | #357 whole-index reset fix | task API, slices, fail-fast, no partial silent success |
| DR-A3 | winter real run on set P | read-only mount, disk budget, lock-rule audit queries, stop/resume |
| DR-B1 | engine stamps for 4 groups (#268) and resident-frame BLS (#40 successor) | READY refusal, rollback switches, parity gates |
| DR-B2 | #67 request plans | plan wire model, skip guarantees, contract stability |
| DR-C1 | #343 versioned partitions | storage model, mapping migration, rollback exactness, stable ids, lock rule |
| DR-C2 | #344 continuous refinement | concurrency with ingest, loop and refresh worker; cost bound |
| DR-C3 | #354 person ids, #353 index persistence | destructive re-run, stable ids by overlap, volume mounts |
| DR-D1 | optional embedding field (D2) | per-project field, one decode, migration-free opt-in, query path |
| DR-E1 | spatial data model, EXIF fields, scene graph storage | isolation, privacy (GPS), mapping migration, versioned scenes |
| DR-E3 | durable reconstruction worker | job protocol, idempotency, cancellation, GPU admission, artifact integrity |
| DR-F1 | #351 basic auth, #101 pinned transport | secrets handling, nginx config, TLS SNI, DNS rebinding test |
| DR-F2 | #8 video architecture after spikes S1 to S7 | data layer, idempotent writer, GPU placement without GPU 1 |
| DR-G | the white paper's statistics (loop guarantees, bootstrap CIs, ladder fits) | math correctness, claims supported by raw JSON |

### 2.2 Audit gate (AG) at the end of every phase and before every release

Tier `[O-audit]`, read-only (it may run tests and scripts, never edits code). Token budget: **at most
400k tokens per phase audit, 150k per re-audit, 600k for a release audit** (A, B+C+D+F, E); the
auditor stops and reports coverage if the budget runs out rather than skimming. Procedure:

1. **Re-derive acceptance.** For every issue in the phase, read the issue body and comments
   (`gh issue view N`) and the plan section; write the acceptance list in its own words before
   reading the implementation.
2. **Check against evidence.** Code (`git log --first-parent`, `git diff <base>..<merge>`), tests
   (exist, assert the acceptance, can fail), measurements (note section plus raw JSON plus reproduce
   command), docs (docs-site page, design-doc status, CHANGELOG). Every DoD item gets pass or fail
   with a pointer.
3. **Hunt defects builders and their tests would miss.** Checklist (reuses the `codebase-audit`,
   `fail-open-audit` and `dead-code-audit` skill methods):
   - fail-open handlers: broad `except` returning None, False, `[]`, 0 or success; partial results
     reported as success; `.get(..., default)` on required config;
   - race conditions: read-modify-write outside OCC, check-then-act on files or indexes, worker
     restarts mid-write, two writers (ingest, refresh, VLM worker, loop, human) on one item;
   - lock-rule and class-identity violations: any writer path that can touch a locked, validated,
     holdout or excluded item; any raw class index crossing import, combine, export, train, promote;
   - isolation: an OpenSearch client not built through the factory, an index name from a constant,
     a cross-project read;
   - tests that cannot fail: assertions on mocks only, tautologies, skipped or xfail without issue,
     tests passing with the feature line reverted (run the mutation on a sample);
   - stale docs and comments: routes, env names, file paths, numbers that no longer match code or the
     log (`DOCS`, `scripts/codegen/check_cited_paths.py`);
   - dead code and duplicates: `scripts/codegen/check_vulture.py`, superseded routes or fields still
     present, duplicated logic beside a shared helper;
   - migration gaps: new mapping fields without a migration in `src/clients/curation_opensearch/`,
     existing projects not covered, env changes not in `env.template` and `tests/test_env_surface.py`;
   - measurement claims: every number in the phase's notes and CHANGELOG traced to raw JSON; spread
     and load recorded; private data only as aggregates.
4. **Re-run a sample** of the builders' fail-before/pass-after checks itself (at least 3 per phase or
   20 % of fixes, whichever is more): check out the parent commit, run the test, see it fail; check
   out the merge, see it pass. GPU re-runs only under the lock and only the cheapest row.
5. **Write** `docs/research/audits/<phase>_audit.md` (`phase_a_audit.md`, ..., `release_v070_audit.md`):
   scope, acceptance lists, DoD table per issue, findings ranked **high / medium / low** with file
   and line, reproduce, and suggested fix. File each finding as a GitHub issue in the phase's
   milestone (label `audit`), linked from the report (filing happens after the owner's yes for
   GitHub writes in this session, as for any push).
6. **Block.** The phase exits, and a release is tagged, only when every high finding is fixed with a
   red-first test and a re-audit (150k budget) marks it closed. Mediums are fixed in the phase or
   carried with an owner-approved issue; lows may be carried.

## 3. Completion matrix (auditor ticks; every column is a DoD item of 0.1)

Columns: C code, U unit, L live, A acceptance, M measurements, D docs, G CHANGELOG, X closed.
Release is where it ships. "Record" is the note section that holds the numbers.

| Item | Phase | Depends on | Acceptance (short) | Record | Rel | C | U | L | A | M | D | G | X |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| #286 auto-label loop | A1 | S0-S6a done | lock rule per item class; same seed same samples; propagate == apply; budget never exceeded; precision >= 0.95, similar-class false propagation <= 3 %, coverage >= 60 % at <= 1 VLM crop per 20 labeled (10k) | `v070_measurement_log.md` H (new) | 0.7.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #362 region-embed one decode | A2 | branch | <= 1.2x v0.6.0 per-item cost at 21 MP; one decode per photo test; activation rejects unusable detector | log G2 rerun | 0.7.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #357 partial pass, 250k reset | A2 | branch | partial cannot report success; 250k+ reset completes or fails fast | log I (new) | 0.7.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #341 eval harness | A2 | merged code | one command scores any partition; seeded baseline; merges vs splits separate; planted synthetic partitions at large N | `cluster_eval_baseline.md` | 0.7.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #335 screenshots | A3 | none | all 17 states from an empty isolated stack; reruns identical file set; leak scan clean | screenshots page | 0.7.0 | [ ] | [ ] | [ ] | [ ] | n/a | [ ] | [ ] | [ ] |
| #355 clustering docs | A3 | none | `DOCS` clean; wrong routes and fields gone; `backbone_embedding` removed or justified | n/a | 0.7.0 | [ ] | [ ] | n/a | [ ] | n/a | [ ] | [ ] | [ ] |
| #356 text search outlier | A2 | part merged | p99 < 300 ms at 100k; stage breakdown; latency check in scale driver at 100k and 1M (real + synthetic) | log D/E3 + J (new) | 0.7.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #349 unattended bind | A3 | owner decision | default stated in help and docs; test of the unattended path | installer page | 0.7.0 | [ ] | [ ] | [ ] | [ ] | n/a | [ ] | [ ] | [ ] |
| #363 installer edges | A3 | none | each item fixed or documented, tests where testable | `installer_drive_log.md` | 0.7.0 | [ ] | [ ] | [ ] | [ ] | n/a | [ ] | [ ] | [ ] |
| #337 cluster pass | A4 | #357 | 0.6.0 2,042 s explained (stage named); <= 2.5 ms/crop at 100k hi-res incl. geometry and assign-only | log C + K (new) | 0.7.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #219 work crops at model size | A4 | set B pin | set B crop-cache bytes/image -50 %; VLM agreement >= 0.97; off = byte-identical | `PERFORMANCE.md` | 0.7.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #84 wheel loop | A5 | loop, #45a | wheel loop from clean project; IoU, recall, img/s vs SAM 3; `wheel` by name | `PERFORMANCE.md` + wheel guide | 0.7.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #45 ladders | A5, B6 | #337 | pinned manifests; full COCO within 10 % of 29.0 img/s, 0 failures; 20 MP 100k with mean GPU; 1M per-quarter throughput, index/img, cluster time | `PERFORMANCE.md`, log L (new) | 0.7.0 / 0.8.0 (1M) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #40 epic | A4, B3 | children | set A >= 29 img/s one card, >= 55 two; 20 MP >= 29 img/s, GPU > 75 %; one decode across detect/embed/faces/OCR; parity gates | `PERFORMANCE.md` | 0.8.0 (closes) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #41 slim images | A7, B9 | final deps | runtime image <= 50 % of 21.4 GB; no image grows; suites pass on slim; release-dry-run clean | `image_size_audit.md` | each release | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #268 stage-2 follow-ups | B1 | set B | sweeps recorded; default changes only beyond spread + parity; set B vs 35.0 img/s; stamp mismatch never READY (5 groups) | `PERFORMANCE.md`, log M (new) | 0.8.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #42 12 GB engines | B2 | #268 | READY on 12 GB with peak < 11 GB; live suite; parity p01 >= 0.995 | gpu-sizing page | 0.8.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #220 faces/OCR resident | B4 | B3 | OCR bytes -4x; P-ocr, ArcFace p01 >= 0.995, P-face 1.0; before/after | `PERFORMANCE.md` | 0.8.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #67 request plans | B5 | B3, #220 | test per stage proves skip; decode once per image; faces-only cheaper (img/s, bytes); old requests unchanged | `PERFORMANCE.md` | 0.8.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #231 ortloom head-to-head | B7 | B1, B3 | same host/ONNX/TRT; decision rule applied (1.10x PE or 1.3x set B) | log N (new) | 0.8.0 | [ ] | n/a | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #342 Leiden prototype | C1 | #341 | report per method (false merge, split recovery, unassigned, stability); go/no-go with graph and resolution | `leiden_refinement_prototype.md` | 0.8.0 | [ ] | [ ] | n/a | [ ] | [ ] | [ ] | [ ] | [ ] |
| #343 stable ids, versions | C3 | #342 go, DR-C1 | locked items never move; rollback exact; ids stable above overlap threshold | design doc + note | 0.8.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #344 continuous refinement | C4 | #343, DR-C2 | refresh cost scales with changed neighborhoods at 100k; drift report | note | 0.8.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #345 GPU and experimental | C2 | #342 | runtime, memory, quality per method on eval set and ladder; recommendation | `gpu_clustering.md` | 0.8.0 | [ ] | [ ] | n/a | [ ] | [ ] | [ ] | [ ] | [ ] |
| #346 epic | C5 | children | all children closed; epic acceptance summary | note | 0.8.0 | [ ] | n/a | n/a | [ ] | [ ] | [ ] | [ ] | [ ] |
| #353 FAISS reload | C0 | DR-C3 | restart and recreate keep assignment; missing tests added | `cluster_types.md` | 0.8.0 | [ ] | [ ] | [ ] | [ ] | n/a | [ ] | [ ] | [ ] |
| #354 persons cluster | C0 | DR-C3 | singleton, re-run stability, memory bound at synthetic 100k faces | `cluster_types.md` | 0.8.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #347 embedding eval | D1 | #341 | numbers per arm (latency, VRAM, dims, quality); single-placement recommendation | `embedding_model_eval_plan.md` 10 | 0.8.0 | [ ] | [ ] | n/a | [ ] | [ ] | [ ] | [ ] | [ ] |
| #351 basic auth | F1 | DR-F1 | LAN request 401 without credentials, works with; `--local-only` unchanged; no secret in logs | security page | 0.8.0 | [ ] | [ ] | [ ] | [ ] | n/a | [ ] | [ ] | [ ] |
| #101 VLM IP pin | F1 | DR-F1 | rebinding test refused or pinned; TLS uses hostname | `SECURITY.md` | 0.8.0 | [ ] | [ ] | [ ] | [ ] | n/a | [ ] | [ ] | [ ] |
| #154 MLflow aliases | F1 | none | alias set, stage call never made; no deprecation warning | promotion docs | 0.8.0 | [ ] | [ ] | [ ] | [ ] | n/a | [ ] | [ ] | [ ] |
| #65 deferred features | F2 | owner triage | each box: design doc + own issue + tests + docs, or dropped by owner | per item | 0.8.0 / 0.9.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #8 video subsystem | F3 | spikes, DR-F2 | issue acceptance 1 to 8 (fake cameras, golden tracks, stitch, search, replay, >400 FPS, Trivy, Kafka) | video note | 0.9.0 (or later, owner) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| #348 spatial feasibility | E0-E1 | 0.7.0 released | measured go/no-go report before product code | `spatial_phase0_benchmark.md` | 0.9.0 | [ ] | [ ] | n/a | [ ] | [ ] | [ ] | [ ] | [ ] |
| R1 resident frame (#40 successor, to file) | B3 | B1 | frame decoded once, crops resident across models, rollback switch, parity | `PERFORMANCE.md` | 0.8.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| R2 DCT-scaled decode for detector input | B3 | B1 | detector P-box >= 0.98 vs full decode; decode ms and img/s on set P sample | log M | 0.8.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| R3 synthetic limits table (#45/#337 owner direction) | A5 | #356 | k-NN, text, filtered query p50/p95/p99 and cluster time at 100k, 500k, 1M+, storage per 100k | `PERFORMANCE.md` limits | 0.7.0 | [ ] | [ ] | n/a | [ ] | [ ] | [ ] | [ ] | [ ] |
| R4 optional embedding model shipped (to file) | D2 | D1 go | off by default; one decode; per-project field; search path; TensorRT where it pays | design doc + note | 0.8.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| R5 spatial MVP A (to file) | E2 | E1 go | scene suggestions with evidence readiness, routes, Svelte panel | spatial note | 0.9.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| R6 spatial MVP B (to file) | E3-E4 | E2 | durable COLMAP + splat job survives restart; viewer opens a scene | spatial note | 0.9.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| R7 spatial MVP C (to file) | E5 | E4 | historic photos localized with confidence; time scrub | spatial note | 0.9.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| R8 winter run on set P (to file, ops) | A6 | A1-A5 | rehearsal budgets of `v070_wave_plan.md` 3.1; better detector (O4: +3 mAP50-95 on frozen holdout, no class 1 point worse) | log W (new), aggregates only | 0.7.0 ops | [ ] | n/a | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| R9 methods/test/code index | A0, every phase | none | every method has note, design doc, code path, tests, figure | `docs/research/methods_index.md` | each | [ ] | n/a | n/a | [ ] | n/a | [ ] | n/a | n/a |
| R10 white paper (to file) | G | all | builds from the repo; every claim in the claims table traced | `docs/research/whitepaper/` | after 0.9.0 | [ ] | n/a | n/a | [ ] | [ ] | [ ] | [ ] | [ ] |

New issues marked "to file" are filed by the coordinator after the owner's yes, in the milestone
shown, each linking this plan.

## 4. Phases, order and parallelism

```text
A (0.7.0, winter pipeline) ──> tag v0.7.0 ──> A6 winter run (Dec-Jan, ops)
   └─ parallel from week 2: B (Triton), C (clustering), D (embedding eval), F1 (auth, small fixes)
B + C + D + F1 ──> tag v0.8.0
E0 (data and benchmarks) may start once v0.7.0 is tagged; E1..E6 after C's partition model ──> tag v0.9.0
G (paper) collects from week 1, drafts after v0.8.0, final after v0.9.0
```

Concurrency: up to 5 or 6 agents; **one GPU job at a time** on GPU 0 under the lock; CPU-only work
(code, tests, docs, scoring from stored vectors) runs freely. Each phase lists its lanes.

### Phase A: finish the 0.7.0 pipeline (weeks 1 to 4)

Goal: the winter run can start. Lanes: L1 loop (one Sonnet, Opus reviews), L2 bugs and passes
(#362, #357, #356, #341 close-out), L3 docs and installer (#335, #355, #349, #363), L4 high-res and
cluster pass (#219, #337, #40 hi-res), L5 public rehearsal and ladders (#45a, #84, R3). Slim images
(#41) last.

| ID | Tier | Work | Verify | Expected / recorded in |
|---|---|---|---|---|
| A0 | [S] | Merge `chore/v070-followups` to main (research index, installer fixes, plans); create `docs/research/methods_index.md` (section 8) and `docs/research/audits/` | T0; `DOCS` | index exists, links resolve |
| A1.1 | [O-design] | DR-A1 on loop sections 5 to 8 | verdict appended to `auto_label_loop_plan.md` | go |
| A1.2 | [S]+[O-review] | Loop S6b `evaluate.py`, `propagate.py`, `driver.py` | `pytest tests/curation/test_assist_loop_lock_rule.py tests/curation/test_assist_loop_drift.py tests/curation/test_assist_loop_veto_dissent.py tests/curation/test_assist_loop_resume.py tests/curation/test_assist_loop_reproducible.py -q --no-cov` | all red-first then green |
| A1.3 | [S]+[O-review] | S6c `audit.py`, `rollback.py` | `pytest tests/curation/test_assist_loop_audit.py tests/curation/test_assist_loop_rollback.py -q --no-cov` | rollback restores pre-loop state exactly |
| A1.4 | [S] | S7 worker service (profile `curation`), refresh hold, `auto_label/start` refusal, env in `env.template` + `tests/test_env_surface.py` | `pytest tests/curation/test_cluster_refresh_loop_hold.py tests/test_env_surface.py -q --no-cov` | 409 `assist_loop_running` |
| A1.5 | [S]+[H] | S8 routes, schemas, cross-project leak rows, contracts, docs-site guide `guides/auto-label-loop.mdx` | `pytest tests/curation/test_assist_loop_routes.py tests/curation/cross_project_leak -q --no-cov`; `CONTRACTS`; `DOCS` | contracts committed |
| A1.6 | [S] | S9 Cropwright panel | `cd frontend && npm run check && npm run build`; stubbed e2e; `DISPLAY=:11` screenshots desktop and 430 px viewed | panel renders backend decisions only |
| A1.7 | [S]+[H] | S10 fake VLM `class_cycle`, `tests/live/test_live_assist_loop.py`, `loop` stage in rehearsal driver | T1 | budget respected, locked unchanged, rollback |
| A1.8 | [S] runs, [O-review] go/no-go | S0 VLM rate (if not done) and S11 T2 runs: #341 eval project, 10k COCO rehearsal-lite (stub and real VLM), set P sample (aggregates) | loop plan section 12 T2 | precision >= 0.95, false propagation <= 3 %, coverage >= 60 % at <= 1/20; log section H, raw `artifacts_local/bench/v070/w3/` |
| A2.1 | [S] | #362: re-measure `embed_written_regions` with the G2 harness (500 photos, 3 runs); merge `fix/v070-362-work` | `pytest tests/curation -k "region_embed or activation" -q --no-cov`; G2 command from the log | <= 1.2x v0.6.0 (86 ms, 0.077 CPU-s/item); log G2 rerun |
| A2.2 | [O-design] then [S] | #357 part 2: reproduce the 250k `update_by_query` failure (synthetic vectors, isolated OpenSearch), fix with sliced `update_by_query` via the task API and a bounded header set, or fail fast | red-first test on the fake; live repro at 250k rows | completes or fails fast with a clear error; log section I |
| A2.3 | [S] | #356 remainder: per-stage timing in the search response log, semantic search default decision (owner O-A1), latency check in `scale_run.py` (p50/p95/p99 at 100k and 1M, real and synthetic) | T0; run at 100k (91k set P project + synthetic) | p99 < 300 ms at 100k; `PERFORMANCE.md` + log J |
| A2.4 | [S]+[O-review] | #341 close-out: confirm hard-case strata in `cluster_eval_set.py`, add planted synthetic partitions at large N (reuse `scripts/research/planted.py` once #342 merges, else a small generator in `scripts/bench/`), docs-site page `operations/cluster-quality.mdx` | `pytest tests -k cluster_eval -q --no-cov`; score 1M planted partition | JSON with merges and splits separate; `cluster_eval_baseline.md` updated |
| A3.1 | [S]+[H] | #335 self-seeding capture on own `-p`; regenerate 17 states | run twice; `git status` shows same file set; leak scan | all states regenerated |
| A3.2 | [H]+[S] | #355 doc and comment fixes; remove `backbone_embedding` if no reader (mapping migration rule) | `rg -n backbone_embedding src`; `DOCS` | no wrong routes or fields |
| A3.3 | [S]+[O-review] | #349 unattended default to 127.0.0.1 unless `--bind`/LAN flag (fail closed; owner O-A2); #363 items | `pytest tests/installer -q --no-cov` | help, docs, `SECURITY.md` agree |
| A4.1 | [S] | #219 `crop_max_side` (off first, 1024 recommended) and VLM crop cap from the VLM's input size; set B gates | parity capture twice, compare: `scripts/bench/parity.py capture|compare` | bytes/image -50 %, agreement >= 0.97; `PERFORMANCE.md` |
| A4.2 | [S]+[O-review] | #337: cluster pass during sustained ingest at 100k+ crops with merge pressure; time geometry and assign-only stages; schedule `_forcemerge` between legs | `scale_run.py` with IO pressure and merges; `cluster_ladder.py` | stage that grows named; <= 2.5 ms/crop hi-res at 100k; log K |
| A4.3 | [S] | #40 hi-res acceptance row on a set P 20k NVMe sample (aggregates) after #219 and the two `gpu_image_prep` instances | T3 (3 reps) | >= 29 img/s, GPU > 75 %; `PERFORMANCE.md` |
| A5.1 | [S]+[H] | #45a Open Images category pins (vehicles, people, animals; per-image license in manifest) | `scripts/datasets/fetch_openimages_bench.py --verify` on the manifest | seeded manifests with sha256 |
| A5.2 | [S]+[O-review] | #84 wheel loop at about 3,000 COCO cars: export single class, train, bake-off, promote, trained detector leg vs SAM 3 | `scripts/examples/wheel_example_live.py` end to end on an isolated stack | IoU, recall, img/s table; wheel guide updated |
| A5.3 | [S] | #45 rows: full COCO (about 164k, sources on the NAS bench share read-only or NVMe), 20 MP 100k repeat from the NAS read-only, R3 synthetic limits table | `scale_run.py --manifest ... --container-prefix <p>` | within 10 % of 29.0 img/s, 0 failures; 20 MP row with mean GPU (0.6.0: 21.0 img/s, 51 %) |
| A5.4 | [S] runs, [O-review] go/no-go | Full public rehearsal (`v070_wave_plan.md` 3.1, P1 + B parts) | `scripts/bench/rehearsal_run.py` | stage budgets met or overrun named; log section W-public |
| A7 | [S]+[H] | #41 slim images on the final 0.7.0 dependency set; Trivy; installer test on the slim images | `make release-dry-run`; T1 on slim images; offline Full | <= 50 % of 21.4 GB; `image_size_audit.md` |
| A8 | [O-audit] | Phase A audit, then release audit v0.7.0 | section 2.2 | `phase_a_audit.md`, `release_v070_audit.md` |
| A6 | [O-design] DR-A3, [S] runs | Winter real run R8 on set P (after tag; ops): ingest P-300k read-only from the NAS, cluster, loop, review (owner), holdout, export, train, bake-off, promote | rehearsal driver stages on the real project | budgets of 3.1; detector gate O4; aggregates in log W |

Exit criteria A: matrix rows of release 0.7.0 all ticked; Full green once on the release branch;
T1 green; rehearsal budgets met or explained; audit high findings zero; owner yes to tag.
Risks: loop precision below 0.95 (mitigation: theta 0.9, validated veto, audit pause; tune with T2
before the real run); cluster pass misses 2.5 ms/crop (agreement-gated GPU fit, O6 of the v0.7.0
plan); set P disk (section 6: ingest from the NAS read-only, never a full copy).

### Phase B: Triton and serving optimization (weeks 3 to 10, v0.8.0)

Lanes: B-sweeps (GPU, serialized), B-code (stamps, CUDA graphs config, plan model; CPU), B-docs.

| ID | Tier | Work | Verify | Expected / recorded in |
|---|---|---|---|---|
| B0 | [S]+[H] | Re-pin set B (Commons 12 to 24 MP; throttle downloads to avoid HTTP 429; manifest with sha256 and licenses) | manifest verify | 2,000 photos pinned |
| B1.1 | [O-design] DR-B1, [S] | #268 engine stamps for yolo, mobileclip, faces, ocr groups; mismatch never READY | `pytest tests -k stamp -q --no-cov`; live: corrupt a stamp, model not READY | 5 groups enforced |
| B1.2 | [S] | #268 sweeps one factor at a time: set B one and two cards, two-card instance/delay/worker sweep (owner window), `gpu_image_prep` parity at 2 instances, nvJPEG throughput, `minimal`/`standard` profiles | `scale_run.py`, `perf_analyzer` via `triton-sdk` profile | default changes only beyond spread + parity; log M |
| B1.3 | [S] | Detector CUDA graphs in Triton (TensorRT `cuda_graph` optimization in `config.pbtxt` per batch size) and `priority_levels` (interactive over bulk) | parity P-box; `perf_analyzer` p50/p99 at batch 1/8/16 | p99 change recorded; adopt only with gain beyond spread |
| B2 | [S] | #42 small-workspace engines; instance guidance. GPU 1 is never used, so verify on GPU 0 with device memory capped at 12 GB (CUDA MPS `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=12G`) unless the owner grants a GPU 1 window (O-B1) | live suite on the capped card; parity p01 | READY, peak < 11 GB; gpu-sizing page |
| B3.1 | [O-design] then [S] | R1 resident frame: `ingest_pipeline` BLS keeps the decoded frame on the GPU, crops resized per model on the GPU, rollback switch (plan waves 6 to 8) | parity chain (`scripts/bench/parity_gates.json`); decode counter == images | set A >= 29 / 55 img/s; 20 MP >= 29 img/s, GPU > 75 % |
| B3.2 | [S] | R2 DCT-scaled (reduced-size) nvJPEG decode for the detector input on large photos | P-box >= 0.98 vs full decode; decode ms | log M |
| B3.3 | [S] | Python/GIL profile per stage (`py-spy` on the API workers during set B ingest); native path only if it dominates | profile flame graphs in `artifacts_local/` | finding recorded in `triton_pipeline_optimization_plan.md` 11 |
| B4 | [S]+[O-review] | #220 plan doc `docs/design/faces_ocr_resident_crops_plan.md`, uint8 OCR input, width buckets, argmax in graph, face alignment on resident frames | parity P-ocr, ArcFace p01 >= 0.995 | bytes -4x; `PERFORMANCE.md` |
| B5 | [O-design] DR-B2, [S] | #67 plan doc, typed plan wire model, Triton ensemble skip question (C1) answered, per-stage metrics | test per stage proves no Triton call and no host work; `CONTRACTS` | faces-only img/s and bytes vs full chain |
| B6 | [S] | #45 1M Open Images row (sources read from the NAS bench share, index on NVMe about 95 GB) | `scale_run.py` | per-quarter img/s, KB/img, cluster time; log L |
| B7 | [S] | #231 ortloom-serve arms (a), (b), (c) after B1 and B3 (arm c needs the passthrough in the ortloom repository) | decision rule of the issue | log N |
| B8 | [S] | Model Analyzer run on the workload mix (deep research phase 10); record recommended instance mix | `model-analyzer profile` in `triton-sdk` | log M |
| B9 | [O-audit] | Phase B audit | 2.2 | `phase_b_audit.md` |

Exit B: rows #268, #42, #220, #67, #231, #40, R1, R2 ticked; parity green; audit clean.
Risks: BLS resident path regresses parity (rollback switch, gate per wave); two-card windows scarce
(batch all two-card rows into two owner windows); ortloom arm (c) depends on another repository
(record "blocked" with date if not available, apply decision rule on (a)/(b)).

### Phase C: clustering (weeks 3 to 12, v0.8.0)

Lanes: C-bugs (#353, #354), C-research (#342 results, #345 GPU), C-build (#343 then #344).

| ID | Tier | Work | Verify | Expected / recorded in |
|---|---|---|---|---|
| C0.1 | [O-design] DR-C3, [S] | #353 load persisted FAISS indexes lazily in `is_trained`, mount a volume for `faiss_indexes/`, add the missing faces IVF, `auto_cluster_faces`, album and `check_balance` tests | train, restart, ingest assigns without a manual call (live) | `cluster_types.md` updated |
| C0.2 | [O-design] DR-C3, [S] | #354 singleton rule, stable `person_id` by membership overlap, blocked or ANN neighbors instead of the dense matrix | tests: singleton, re-run stability, memory bound at 100k synthetic faces (RSS sampled) | memory O(n k), not n^2 |
| C1 | [S] runs, [O-review] math | #342: finish on `research/v080-342-leiden-prototype` (owned by its agent): CPU Leiden (CPM, several resolutions, initialized from the current partition), HDBSCAN leaf, FINCH, consensus Leiden vs IVF+AHC baseline on the #341 set, planted synthetic partitions, the 91k set P vectors (aggregates) | `scripts/research/bench_driver.py` then `scripts/bench/cluster_eval.py score` per partition | `docs/research/leiden_refinement_prototype.md`: false merges, split recovery, unassigned, stability, runtime, memory, items moved; go/no-go with graph and resolution range |
| C2 | [S] | #345 GPU: cuGraph Leiden and ECG, cuML HDBSCAN, FAISS GPU IVF fit, on GPU 0 under the lock (`docker-compose.gpu-clustering.yml`); FINCH levels; HIT-Leiden implementations; embedding adapter only if graph methods plateau (research note only) | same harness; ladder 100k, 500k, 1M (synthetic + real) | runtime, VRAM, quality per method; `gpu_clustering.md` |
| C3 | [O-design] DR-C1, [S], [O-review] | #343 cluster records (`cluster_uuid`, `parent_uuid`, level, run id, confidence, model version, lineage, provenance), versioned partitions with rollback, constrained split/merge (CHAMELEON validation, cannot-link from trusted labels), mapping migration, wire via `wire.py`, dry-run-first Cropwright view; the loop's partition reader switched to versions | tests per locked class; rollback exact; id stability above threshold; `CONTRACTS`; T1 | design doc `docs/design/cluster_partitions_plan.md` |
| C4 | [O-design] DR-C2, [S] | #344 tiered refresh: confidence-gated kNN assignment on ingest, neighborhood re-evaluation, scheduled community recompute, periodic full audit with drift; microcluster summaries | cost vs changed neighborhoods at 100k (scale ladder); drift report | note section; `PERFORMANCE.md` |
| C5 | [S] | Kinds 1 and 3 (whole-image MobileCLIP, faces ArcFace): score with the harness on public sets; record whether the graph method transfers (no mixing of spaces) | harness per kind | `cluster_types.md` section 6 |
| C6 | [O-audit] | Phase C audit | 2.2 | `phase_c_audit.md` |

Exit C: #342 to #346, #353, #354 ticked. Risk: Leiden no-go (then #343 still ships stable ids and
versioned partitions for the current method; #344 uses kNN assignment only; recorded honestly).

### Phase D: optional embedding model (weeks 4 to 10, v0.8.0)

| ID | Tier | Work | Verify | Expected / recorded in |
|---|---|---|---|---|
| D1.1 | [S] | Merge the eval plan; build the throwaway env outside the repo; data D1 to D4 (public) | manifest with sha256 | ledger in `artifacts_local/eg2/` |
| D1.2 | [S] | Scenarios S1 to S6 (S7 optional) for PE-Core, MobileCLIP, EG2-T, EG2-TV; reference arms Qwen3-VL-Embedding 2B/8B and NVIDIA arms (omni-embed measurement-only, Nemotron 3.5 Embed 1B) **served through vLLM or transformers**, not converted to TensorRT | plan 3.5 commands under the GPU 0 lock, telemetry per plan section 6 | metrics JSON per run (plan section 7); bootstrap CIs |
| D1.3 | [S] | TensorRT track for the encoder-style arms (EG2 text and vision towers): ONNX, parity, build, speed, memory (plan 5.4, 5.5), CPU ONNX Runtime text variant | parity mean cos >= 0.999, min >= 0.99; >= 1.5x over ORT or half VRAM | speed table; shipped runtime line |
| D1.4 | [O-review] | Go/no-go per placement (A OCR text, B captions, C cross-modal, D clustering view) against rubric 3.6 | | recommendation in plan section 10 |
| D2 | [O-design] DR-D1, [S] | R4 (only on go): optional model off by default, never replacing PE-Core; text encoder on CPU or GPU (selectable); vision tower fed from `gpu_image_prep` (plan section 9, one decode); per-project opt-in field and query path; export script in `export/`; Triton config; docs-site page | decode counter test; `CONTRACTS`; T1; recall on D1 reproduces D1.2 within CI | design doc + `PERFORMANCE.md` row |
| D3 | [O-audit] | Phase D audit (with B/C release audit) | 2.2 | `phase_d_audit.md` |

Risks: license gates (record and stop that arm); TensorRT build fails (one-day box, then ONNX
Runtime); sm_86 rules out FP8/NVFP4 cells.

### Phase E: spatial memory and photogrammetry (v0.9.0; E0 after the v0.7.0 tag)

Product phases follow section 16 of the spatial note; every phase has data, tests, metrics, budgets.
GPU budget assumes GPU 0 under the lock; heavy jobs never run during a Triton benchmark.

| ID | Tier | Work | Data | Metrics and gate | GPU / storage budget |
|---|---|---|---|---|---|
| E0.1 | [S]+[H] | License review of COLMAP, pycolmap, ALIKED, LightGlue (not SuperPoint weights), SALAD and MegaLoc, gsplat, PlayCanvas engine, hloc, against AGPL; record in the note | n/a | each component: license, version, ship or measure-only | none |
| E0.2 | [S] | Fetch public sets (section 7) with manifests; `scripts/research/nas_spatial_scan.py` (section 6.3) read-only inventory of set P and set L | section 7 | manifests with sha256; scan JSON in `artifacts_local/spatial/` | <= 80 GB NVMe |
| E0.3 | [S]+[O-review] | Build the labeled mixed library (spatial note 13.1): 500 to 2,000 images mixing an IMC 2025 scene set, IMC PhotoTourism scenes and distractors from COCO; ground truth `same_place`, `geometric_overlap`, `same_scene_component`; set P same-place subset labeled privately (aggregates only) | public + set P sample | label manifest committed for public part | none |
| E0.4 | [S] | COLMAP baseline on known single-scene sets (Mip-NeRF 360, Tanks and Temples training, ETH3D high-res) | | registered %, median reprojection error, time, peak RAM/VRAM | <= 2 h GPU 0 |
| E1.1 | [O-design] DR-E1, [S] | EXIF capture at ingest (timestamp, precision, GPS with accuracy, camera, focal) as project fields with a mapping migration; GPS treated as sensitive (not served unless the project enables it) | ingest public sets | test: fields stored, absent GPS stays absent | none |
| E1.2 | [S] | VPR descriptor arm (SALAD or MegaLoc, license permitting) on the decoded frame (one decode), stored in its own field; never mixed with PE or MobileCLIP | public + set P sample | same-place recall@K vs MobileCLIP and PE (note 18 Q1) | <= 4 GiB VRAM |
| E1.3 | [S] | Pair verification: ALIKED (or SIFT) + LightGlue + MAGSAC; edge classes verified / likely / unrelated | | edge precision/recall, inlier stats, ms per pair | <= 6 h GPU 0 total |
| E1.4 | [S]+[O-review] | Scene graph: robust components, bridge pruning, stable `scene_id` with versions (reuses #343 lineage ideas) | | purity, fragmentation, false merges, id stability (note 13.2) |
| E1.5 | [O-review] | #348 go/no-go: continue only if same-place groups are found reliably without false geometry merges | | report `docs/research/spatial_phase0_benchmark.md` | |
| E2 | [O-design], [S] | MVP A: readiness score (detailed report, labels of note 9.5), routes (proposed: `curation/projects/{project}/scenes`, `.../scenes/{scene_id}`, `.../readiness`, `.../overlap-graph`; none exists yet), wire models, Cropwright scene-suggestions panel (Svelte, thin) | | route tests, cross-project leak rows, `CONTRACTS`, browser check | none |
| E3 | [O-design] DR-E3, [S] | Durable reconstruction worker (profile `reconstruction`, file job protocol like the trainer): COLMAP SfM, idempotency key, checkpoints, cancellation, VRAM/disk admission, restart survival | | job survives restart; poses produced; failure reasons stored | one scene job at a time; <= 30 GB per scene artifacts |
| E4 | [S] | gsplat training and web export (SOG/SPZ), PlayCanvas viewer embedded in Cropwright (Svelte), photo/3D toggle | Mip-NeRF 360 + one discovered group | held-out PSNR/SSIM/LPIPS, browser fps, size | <= 2 h GPU per scene |
| E5 | [S] | MVP C: localize historic photos into a reference scene (hloc + LightGlue), time slider, confidence markers, manual anchors | set P cross-year same-place sessions (private), Aachen v1.1 day/night as public proxy | fraction localized, anchor drift | |
| E6 | [S]+[O-review] | Hardening: quotas, delete of derived features and artifacts, privacy of GPS and trajectories, docs-site section `guides/spatial-*.mdx` | | audit checklist | |
| E7 | [O-audit] | Phase E audit, release audit v0.9.0 | | `phase_e_audit.md`, `release_v090_audit.md` | |

Exit E: #348 and R5 to R7 ticked; go/no-go honored (a no-go at E1.5 ends E with the report; MVP
work is then re-scoped by the owner). Risks: false geometry merges on repetitive architecture
(bridge pruning, labeled negatives); license blocks (measure-only arms); GPU contention with Triton
(separate allocation, lock); storage growth (artifact presets, per-scene cap).

### Phase F: auth, small fixes, deferred backlog, video

| ID | Tier | Work | Verify |
|---|---|---|---|
| F1.1 | [O-design] DR-F1, [S] | #351 basic auth in Cropwright's nginx, credentials from the installer into `secrets/` (mode 600, shown once), default on beyond loopback; revisit #349 default once auth exists | installer tests; curl 401/200 on an isolated stack; no secret in logs (`docker logs` grep) |
| F1.2 | [S]+[O-review] | #101 connect to the validated IP with original Host and SNI | rebinding test (public at check, private at connect) refused |
| F1.3 | [S] | #154 MLflow aliases | trainer test asserts alias, no stage call |
| F2 | [O-design] per item, [S] | #65 triage with the owner: embedding-model swap (dual-field migration, feeds from D), `ingest_profile` config kind, second segmenter, training family registry, single-class exporter, API auth beyond #351, native importers; each taken item gets a design doc and its own issue, the rest dropped by the owner | per item |
| F3 | [S] spikes, [O-design] DR-F2 | #8 video: spikes S1 to S7 first (S1 go/no-go), then PR1 to PR10 of the issue; GPU placement on GPU 0/2 only (the issue's guide assigns GPU 1; replace) | issue acceptance 1 to 8 |

### Phase G: the white paper (section 9)

## 5. Release cadence

| Release | Contents | Gate |
|---|---|---|
| v0.7.0 | Phase A (winter pipeline) | matrix rows ticked, release audit clean, #41 on the final deps, Trivy, installer test on the slim images, owner yes |
| v0.8.0 | Phases B, C, D, F1 (+ #45 1M, #65 items the owner keeps) | same, plus parity chain green, contracts in sync |
| v0.9.0 | Phase E (+ #8 if the owner keeps it here) | same, plus spatial go/no-go report and MVPs |

Each release: release branch `release/0.X.0` cut from main, later phases continue on their own
branches and merge into the next release branch only. Slim images (#41) are rebuilt once per release
on that release's final dependency set. Annotated tag, CHANGELOG section, GitHub release (no draft)
only after the owner's yes.

## 6. Data inventory, storage budget and NAS scan

### 6.1 Collections (numbers only for private data)

| Set | What | Size (measured 2026-10-28, read-only listing) | Use | Where it lives |
|---|---|---|---|---|
| A | public COCO 2,000/4,000 pins | about 1 GB | headline throughput | NVMe |
| COCO full | about 164k images | 25 to 28 GB | #45 full row | NAS bench share (read-only) or NVMe |
| OI 100k | Open Images v7 pin | 100,000 files, 30 GB | ladders, 1M base | NAS bench share (exists) |
| OI vehicles P1 | 200k public | about 60 GB | rehearsal | NAS bench share |
| OI 1M | public | about 300 GB | #45 1M | NAS bench share |
| B | Commons 12 to 24 MP, 2,000 | about 15 GB | hi-res gates | NVMe |
| P | owner photos, about 21 MP | about 1.93M image files on three volumes, about 9.5 MB each (about 18 TB), date-sessioned shoots 2019 to 2024 at recurring roadside locations | winter run; same-place subset for spatial E5 | NAS, read-only |
| L | second owner library (personal) | about 218k image files in about 4,600 dated event folders since 2004 | spatial candidates only if the owner allows (O-E1) | NAS, read-only |
| Spatial public | section 7 | about 70 GB | E0 to E5 | NVMe |
| Embedding eval | COCO val, OI val sample, TextOCR/TextCaps, model weights | about 60 GB | D | NVMe (HF cache) |

### 6.2 NVMe budget (about 420 GB usable; `df` showed 708 GB free on 2026-10-28, the owner's 420 GB cap governs)

| Use | Peak GB | When | Released after |
|---|---|---|---|
| OpenSearch data, winter project (P-300k + P1 200k, about 95 KB per image index) | 50 | A6 | stays (live project) |
| Set P NVMe samples: P-5k bench (48 GB), P-20k hi-res row (190 GB) | 190 at most, one at a time | A4, B | row recorded (copies only; NAS untouched; owner yes before removal) |
| Exports and training runs (1280 px) | 60 | A6 | stays until promoted model verified |
| 1M row index | 95 | B6 | row recorded (owner yes) |
| Public spatial sets + work dirs + reconstructions | 110 | E | stays for the paper |
| Embedding eval data and weights | 60 | D | stays until D closes |
| Image builds (slim, transient) | 50 | A7/B9 | after push |

Rule: never two of {P-20k sample, 1M index, spatial reconstructions} at once. The winter run reads
set P from the NAS through a read-only bind mount (measured: NAS reads at 291 MB/s and ingest at
about 26 img/s from the NAS, not the bottleneck); no full copy.

### 6.3 NAS scan for same-place candidates (read-only, proposal)

Script `scripts/research/nas_spatial_scan.py` (E0.2, `[S]`, reviewed `[O-review]` for read-only
safety): `--root <path>` (passed at run time, never committed), `--out artifacts_local/spatial/nas_scan/`.
Level 1 (approved now): `os.scandir` only, no file opened; counts per folder, extension histogram,
dated session folders per year, folder-name recurrence (same location label across years). Level 2
(owner approval O-E2): EXIF header reads on a seeded sample of at most 2 % (timestamp, GPS presence,
camera, focal), read-only file handles; GPS clustered into same-place candidates at 50 m radius and
cross-year recurrence. Output: aggregate JSON (counts, sessions, candidate groups by id), never paths
in any committed file. A test runs it on a temp tree and asserts no write outside `--out` and no
`open` in level 1 (monkeypatched).

## 7. Public multi-view datasets for spatial work (licenses verified 2026-10-28 by web search)

| Dataset | Content | Size | License (as found) | Role |
|---|---|---|---|---|
| IMC 2025 (Kaggle, "IMC 2025 On-going" continues it) | unsorted image sets with several scenes plus outliers; task is to cluster into scenes and pose | not stated on the page; check at download | Kaggle metadata lists Apache 2.0; per-image terms to check | **primary discovery benchmark** (scene clustering + outliers matches E1) |
| IMC PhotoTourism (2020/2021, UBC host) | landmark collections from YFCC Flickr CC, COLMAP pseudo-GT, co-visibility | validation tarballs small; 2019 train 127 GB (use validation + a few train scenes) | images carry their own Flickr CC terms; no single dataset license | pair verification, scene graph |
| Mip-NeRF 360 | 9 object-centric scenes (+2 extra) | about 12.5 GB + about 4.5 GB (mirror sizes) | no license stated on the project page; treat as research-only, do not redistribute | splat quality (PSNR/SSIM/LPIPS) |
| Tanks and Temples | large-scale scenes, laser GT | per-scene downloads | official page says CC BY 4.0 but adds a non-commercial research grant; treat as research-only | SfM and mesh accuracy |
| ETH3D high-res multi-view | 25 scenes, 6221 x 4146 DSLR, 13 with public GT | per-scene archives | CC BY-NC-SA 4.0 (mirrors; confirm on eth3d.net) | high-res SfM with laser GT |
| Aachen Day-Night v1.1 | 6,697 reference + 1,015 queries (day/night) | a few GB | not confirmed (third party lists CC BY-SA 4.0; check the README) | historic/condition-change localization proxy (E5) |
| Cambridge Landmarks | 5 scenes, 231 to 3,015 images each, VisualSFM poses | King's College 5.6 GB | CC BY-NC-SA 2.0 UK | localization sanity |
| Google Landmarks v2 (subset) | 4.1M train, 762k index; Wikimedia Commons | full 593 GB; use a 20k subset | annotations CC BY 4.0, images per-image licenses | same-place retrieval distractors at scale |
| MegaDepth (subset) | 196 landmarks, Flickr, SfM depth | 199 GB full; subset only | not stated; images keep Flickr terms | matcher training reference only (no training planned) |

Rule: committed results name the dataset, version and license; no images are redistributed;
research-only sets are never shipped in images or examples. Ingest needed beyond these: the COCO
distractors for E0.3 and the set P same-place sample (private, aggregates only).

## 8. Documentation plan

- **As built, every method gets three documents in the same branch**: a `docs/research/` note
  (evidence; dated; reproduce), a `docs/design/` doc (decision, design review verdict, status), and a
  docs-site page where users or operators see a change (guides, operations, deployment, API
  reference). Design doc status rows in [`README.md`](README.md) are updated in the same merge.
- **Methods/test/code index** `docs/research/methods_index.md` (A0, then every merge): one row per
  method: name, issue, design doc, research note section, code entry points, tests, raw JSON
  location, paper figure or table id. The auditor checks the index against the matrix.
- **Measurement logs**: `v070_measurement_log.md` gains sections H (loop), I (#357), J (search
  latency), K (#337), L (ladders), W (winter run, aggregates); v0.8.0 opens `v080_measurement_log.md`
  (M Triton, N ortloom, clustering and embedding summaries), v0.9.0 opens `v090_measurement_log.md`.
- **Audits** live in `docs/research/audits/` (section 2.2).
- New docs-site pages planned: `guides/auto-label-loop.mdx`, `operations/cluster-quality.mdx`,
  `guides/cluster-refinement.mdx`, `guides/optional-embeddings.mdx`, `guides/spatial-scenes.mdx`,
  `deployment/basic-auth.mdx` (or a section in `deployment/security.mdx`), video pages with #8.

## 9. The white paper

### 9.1 Thesis and positioning

Thesis: *a self-hosted, single-stack data engine turns raw private images into a served, better
detector with measured throughput and statistically bounded automatic labeling, while never
overwriting trusted labels.* Positioning (from the market survey of 2026-10-04): the pieces exist
separately (auto-label in Roboflow and Autodistill, embedding curation in FiftyOne and Lightly,
annotation in CVAT); none found combines detector, embedding clusters, VLM suggestion, human confirm
by cluster, train, bake-off and promote, with class identity by name and per-project isolation, in
one open self-hosted product. The paper says what it is not (not an annotation tool; boxes first)
and compares honestly.

### 9.2 Differentiators to substantiate (each needs measured evidence)

1. GPU decode-once pipeline on Triton and its throughput ladder (2k to 1M, 0.7 MP to 21 MP).
2. Curation backend with isolated projects, fail-closed guard and the lock rule.
3. Cluster-stratified data engine (IVF + refine, evaluation harness, cost ladder).
4. Agreement-gated auto-labeling loop with statistical guarantees, audit and rollback.
5. Graph-based global cluster refinement with stable ids and versioned partitions.
6. Optional multimodal embeddings without double decode.
7. Spatial memory: same-place discovery with geometric verification inside a photo library.

### 9.3 Claims-to-evidence table (filled as evidence lands; "pending" blocks the claim)

| Claim | Evidence (log section / script / raw JSON) | Status |
|---|---|---|
| 0.6.0 ingest 28.81 to 29.07 img/s one A6000, 55.08 two (public COCO) | `docs/PERFORMANCE.md` stage 2; `docs/benchmarks/v060_wave0_headline_set_a.json` | measured |
| A second `gpu_image_prep` instance raises 20 MP ingest 31.8 to 48.0 img/s | log E4; `scale_run.py`; `artifacts_local/bench/v070/` | measured (parity pending, B1.2) |
| Embed pass 51 vs 28 img/s at 21 MP with GPU decode and chunk overlap | log G3/G5 | measured |
| Cluster pass linear after the IVF training cap; 1M crops in 596 s | log "Cluster pass wall time"; `scripts/bench/cluster_ladder.py` | measured |
| Text search p99 under 300 ms at 100k | log J; `scale_run.py` latency check | pending A2.3 |
| Hi-res ingest >= 29 img/s, GPU > 75 % | `PERFORMANCE.md`; A4.3 JSON | pending |
| One decode per image across all models | decode counters in stage timers; B3.1 tests | pending |
| Loop precision >= 0.95 with budget guarantee and zero lock violations | log H; loop tests; S11 JSON | pending |
| Loop sampling bound (theta, alpha) holds | `src/domain/loop_stats.py` tests; DR-G review | pending |
| Trained detector beats incumbent by >= 3 mAP50-95 (public wheel twin and set P aggregates) | `PERFORMANCE.md` #84 table; log W | pending |
| Leiden refinement lowers false merges at equal split recovery | `leiden_refinement_prototype.md`; harness JSON | pending |
| Versioned partitions roll back exactly | #343 tests | pending |
| Optional embedding adds recall where measured (or no-go reported) | eval plan section 10; ledger | pending |
| Same-place discovery precision/recall without false geometry merges | `spatial_phase0_benchmark.md` | pending |
| Storage: about 95 KB index per input image | `storage_sizing_and_ingest_baselines.md`; `docs/benchmarks/v050_storage.json` | measured (1M recheck B6) |

### 9.4 Outline, figures and tables

| Section | Content | Figures / tables (source) |
|---|---|---|
| 1 Introduction | problem, thesis, contributions | Fig 1 system diagram (archify from `docs/ARCHITECTURE.md`) |
| 2 Related work | annotation, curation, auto-label, serving, SfM | Tab 1 feature matrix (`docs/market/06`) |
| 3 System | projects, guard, lock rule, wire contract, workers | Fig 2 data flow; Tab 2 integrity rules and their tests |
| 4 Serving pipeline | decode once, uint8, FP16 PE, GPU decode, resident frame | Fig 3 stage time breakdown (`v060_wave0_stages_set_*.json`, B3 JSON); Fig 4 throughput ladder (log L); Tab 3 parity gates (`docs/benchmarks/parity/`) |
| 5 Data engine | IVF + refine, cost ladder, eval harness | Fig 5 cluster pass ms/crop vs N (log Results); Tab 4 baseline metrics (`cluster_eval_baseline.md`) |
| 6 Auto-label loop | sampling, gate, statistics, audit, rollback | Fig 6 node tree example; Tab 5 precision, coverage, crops per label (log H); Fig 7 budget curve |
| 7 Refinement | kNN graph, Leiden, versioned partitions | Tab 6 per-method metrics (C1/C2 JSON); Fig 8 runtime vs N CPU/GPU |
| 8 Embeddings | optional model study | Tab 7 recall and latency per arm (D ledger) |
| 9 Spatial memory | discovery, verification, readiness | Fig 9 recall@K per descriptor; Tab 8 edge precision/recall (E1 JSON) |
| 10 End to end | wheel twin, winter run (aggregates) | Tab 9 stage budgets vs measured (log W); Tab 10 detector mAP by class name |
| 11 Limitations | boxes-first, no RBAC beyond basic auth, licenses | |
| Appendix | reproducibility | Tab A1 commands, manifests, versions, hardware |

### 9.5 Reproducibility appendix and deliverable

`docs/research/whitepaper/`: `main.tex`, `sections/*.tex`, `refs.bib`, `figures/make_*.py` (each
reads only committed JSON under `docs/benchmarks/whitepaper/` and writes `figures/*.pdf`),
`Makefile` and a `README.md` with the build: `make -C docs/research/whitepaper figures pdf`, which
runs the figure scripts with `.venv/bin/python` and `latexmk -pdf main.tex` inside a pinned TeX Live
container (`docker run --rm -v "$PWD":/w -w /w/docs/research/whitepaper texlive/texlive:<pinned tag>
latexmk -pdf main.tex`). Every public figure's data is copied (public data, small JSON) into
`docs/benchmarks/whitepaper/`; set P appears only as aggregate JSON. The appendix lists for each
table: manifest (seed, count, sha256), commit, command, hardware, driver, Triton and TensorRT
versions, load average. Tier: `[S]` drafts, `[O-review]` DR-G for math and claims, `[H]` checks every
citation and number against the claims table.

Phase G tasks: G1 skeleton and figure scripts (week 2, from existing JSON); G2 sections 3 to 6 after
v0.7.0; G3 sections 7 and 8 after v0.8.0; G4 section 9 and final after v0.9.0; G5 audit (claims
table fully traced, `[O-audit]`), owner review, then publication only on the owner's yes.

## 10. GPU, host and disk schedule (weeks from 2026-10-29; estimates)

| Week | GPU 0 lane (serialized, lock) | Owner windows (GPU 2 / two cards) | CPU lanes | Disk peak |
|---|---|---|---|---|
| 1 | #362 rerun, #357 250k repro, loop S0 | none | loop S6b-S8, docs A3, A0 | 60 GB |
| 2 | #219 parity, #356 latency, loop T2 (VLM on GPU 0) | none | loop S9-S10, #341, G1 | 110 GB |
| 3 | #337 under ingest, A4.3 hi-res row (P-20k) | two-card sweep window 1 (3 h) | #84 prep, B1.1 stamps, C0, C1 scoring | 250 GB |
| 4 | #84 train, full COCO, 20 MP 100k (NAS), rehearsal | rehearsal window (3 h) | #41, audit A | 200 GB |
| 5 | tag v0.7.0 (owner); winter ingest (Triton GPU 0) | GPU 2 if freed for training | B, C, D code | 260 GB |
| 6-8 | winter VLM loop, then training (24 h), B1.2/B1.3 sweeps between | two-card window 2 | C3 design and build, D1 CPU scoring | 300 GB |
| 6-10 | D1 encodes and TensorRT builds; C2 GPU clustering; B3 resident frame; B6 1M | | F1, #65 triage | 380 GB (1M index, then released) |
| 10-12 | B7 ortloom, B8 Model Analyzer, C4 ladder | | audits B, C, D; tag v0.8.0 | 250 GB |
| 11-20 | E0 COLMAP baselines, E1 descriptors and matching, E3/E4 jobs | | E2 UI, E6, G3 | 360 GB |
| 20-22 | E5, final ladder rechecks | | G4, G5 audit; tag v0.9.0 | 360 GB |

Host: the CPU is shared (load 8 to 33 observed); record load per row; a 1-minute load above 16 marks
a run noisy (repeat before using it for a decision).

## 11. Risks

| Risk | Mitigation |
|---|---|
| Wrong labels at scale from the loop | theta/alpha gate, validated veto, audit pause, propagated kind excluded from export until reviewed, one-call rollback, A1.8 precision gate before the winter run |
| GPU contention with other users | lock, GPU 1 never used, short tiered runs, owner windows for GPU 2 |
| Disk exhaustion on NVMe | section 6.2 rule; NAS read-only ingest; copies removed only with owner yes |
| Set P or set L leak | numbers only, `artifacts_local/`, naming scan, no screenshots of private projects, GPS never served by default |
| Scope too large for one winter | phases B to E run behind A and never block the winter run; each release independently gated |
| Third-party licenses (SAM 3, VGGT, research-only datasets, NVIDIA omni) | measure-only arms, never shipped; license table in each note |
| Research no-go (Leiden, EG2, spatial) | honest report closes the issue with the numbers; product work re-scoped by the owner |
| Audit finds deep defects late | DRs before risky stages; audits per phase, not only per release |
| #8 video size (10 PRs) | spikes first; owner decides v0.9.0 or later after S1 |

## 12. Open questions for the owner

| ID | Question | Recommendation |
|---|---|---|
| O-A1 | Semantic text search on by default in the installer (#356)? | yes, with the warm-up fix |
| O-A2 | Unattended Cropwright bind default (#349) before basic auth lands? | 127.0.0.1 (fail closed); LAN default returns with #351 |
| O-A3 | Winter set P size and composition (300k photos read from the NAS; which years)? | 300k seeded across all sessions, P1 public twin alongside |
| O-A4 | Close #40 at v0.8.0 (resident frame inside it) or split R1 into a new issue and close #40 at v0.7.0 with the hi-res acceptance? | split (v0.7.0 plan O8) |
| O-A5 | Loop questions Q1 to Q9 of the loop plan and v0.7.0 O3/O4 (export states, detector gate) confirmed as recommended? | yes |
| O-B1 | #42 verification: CUDA MPS 12 GB cap on GPU 0, or a one-off GPU 1 window? | MPS cap on GPU 0 |
| O-B2 | Two owner windows with both A6000s (3 h each) in weeks 3 and 6? | yes |
| O-D1 | Approve the embedding eval arms and one-day time box per arm and tier? | yes |
| O-E1 | May set L (personal library) be used privately for spatial evaluation (aggregates only)? | yes for E0/E1 metrics only, never screenshots |
| O-E2 | Allow level 2 of the NAS scan (EXIF headers on a 2 % sample, read-only)? | yes |
| O-E3 | Spatial storage: OpenSearch fields plus project artifact directories (no new database), or add Postgres as the note suggests? | OpenSearch + files; decide in DR-E1 |
| O-F1 | #8 video: keep in v0.9.0 or move after the paper? | decide after spike S1 |
| O-F2 | #65 items to keep vs drop | review in F2 |
| O-G1 | Paper publication venue or channel (repository PDF only?) and author list | repository PDF; owner decides further |
| O-X1 | NVMe cap: 420 GB as stated (708 GB showed free on 2026-10-28)? | keep 420 GB |
| O-X2 | Public bulk data (OI 1M, P1, COCO full) stored on the NAS bench share (new files only, nothing deleted)? | yes |
