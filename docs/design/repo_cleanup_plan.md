# Repo cleanup plan: effective guards, dead code, naming, fail-open, duplicates, splits

Status: plan (nothing implemented). Milestone 0.6.0. All work lands under `## [Unreleased]` in
`CHANGELOG.md` (and `frontend/CHANGELOG.md` for frontend changes); no version bump, no release.
Evidence: the three review reports in `artifacts_local/review/` (`python-code.md` = PC-xx,
`frontend-scripts-env.md` = F-x/FD-x, `org-and-tooling.md` = T-x/TD-x), re-verified for this plan
against `origin/main` at `64ce2eff`. File:line references drift; re-find by symbol.

Owner scope (binding): do **all** of the review's fixes; this is a cleanup, not a rewrite. No
behavior change except the fail-open fixes and the decisions marked below. Trimming a file to fit
the 700-line ceiling is not acceptable: every near-limit (640-700) and oversize file is split by
concern on the schedule in this plan. Allowlists exist only for the transition and shrink to the
cohesive exemptions named in section 11.

A fresh implementer can execute any task from this file alone. Task ids (A1, B3, ...) and review
ids (PC-11, TD-4) are for this plan and PR descriptions only: **never write them into code,
comments, test names or file names** (guard C7 rejects them).

## 0. Ground rules (every PR)

1. **One branch and one worktree per PR**, off fresh `origin/main` after the previous PR merged:
   ```bash
   MAIN=<absolute path of the main checkout>; cd "$MAIN" && git fetch origin
   WT="$MAIN-wt/<slug>"                           # e.g. cleanup-guards
   git worktree add "$WT" -b <branch> origin/main  # branch names in section 2
   ln -s "$MAIN/.venv" "$WT/.venv"   # .venv is gitignored; hooks need it
   (cd "$WT/frontend" && npm ci)                  # only when the PR touches frontend/
   ```
   Git hooks are shared by all worktrees (install once in the main checkout:
   `.venv/bin/pre-commit install --hook-type pre-commit --hook-type commit-msg`).
   After merge: `git worktree remove "$WT" && git branch -d <branch> && git fetch origin`.
2. **Python:** call `.venv/bin/python` / `.venv/bin/pre-commit` directly; never `source`. No
   Docker and no GPU are needed for any task here except the F5 image builds (use `-p cleanup-<x>`,
   never the default compose project, GPUs 0/2 only).
3. **Commits:** conventional (`<type>(<scope>): <summary>`), one coherent group per commit,
   `.venv/bin/pre-commit run --all-files` green before each commit, never `--no-verify`.
   Merge with `gh pr merge <n> --merge` (never squash) after CI is green; push and merge only
   with the owner's yes (standing rule). `CHANGELOG.md` conflicts: keep both sides.
4. **Tests:** targeted tests while working (`.venv/bin/python -m pytest <paths> -q --no-cov`); the
   full offline suite **once per PR** before pushing:
   `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'`. Frontend PRs also run
   `cd frontend && npm run check && npm run lint && npx vitest run && npm run build`, and open the
   changed pages in a browser (`DISPLAY=:11`) at desktop and 430 px width.
5. **Real tests:** every behavior fix (section E) and every new guard first gets a test that
   fails on the current code; paste the failing output in the PR description, then fix.
6. **Contracts:** any route or wire-model change: `make contracts` and commit `contracts/` in the
   same commit. **Docs:** `.venv/bin/python scripts/docs/check_docs_vs_code.py` (only existing
   routes, `OP_*` names read by code, resolving links); the naming-leak scan must pass.
7. **No shims:** a moved symbol is imported from its new home everywhere in the same commit; no
   re-export from the old module, no alias names, no "kept for compatibility" code.

## 1. Decisions

OC = owner confirms before the PR that implements it starts; others are the plan's call. OC by
PR: PR1 D1, D2, D5, D12; PR2 D20, D22, D23; PR4 D8, D24; PR5 D25, D26, D29, D31, D32, D33;
PR7 the `setup-openprocessor.sh` keep; PR9 D3.

| # | Question | Recommendation and reason | OC |
|---|---|---|---|
| D1 | Bump hook revs (pre-commit-hooks 5->6, ruff 0.14->latest, gitleaks, bandit, shellcheck-py, hadolint; CI pins hadolint and golangci-lint too) | Yes, in PR1 as two commits: (a) all non-ruff revs, (b) ruff alone with any new findings and format churn fixed in that commit. Rename hook id `ruff` to `ruff-check`. Bumping later would re-touch every file the cleanup touched. Pin the same ruff version in the venv | OC |
| D2 | mypy version and mode | Drop `mirrors-mypy`; run mypy as a local hook from the project interpreter so numpy/fastapi/pydantic types are real (TD-1). Pin the latest stable mypy in `requirements-test.txt`, enable `plugins = ["pydantic.mypy"]` (removes the `ItemCrop` call-arg false positives), add `types-PyYAML`. Fix every error; a coded `# type: ignore[code]  # reason` only where a third-party stub is wrong | OC (pin choice) |
| D3 | mypy strictness path (TD-3) | PR1: `warn_unused_ignores = true` (drops ~25 stale ignores). PR9: `disallow_incomplete_defs` and `warn_return_any` globally, `disallow_untyped_defs` globally except `tests`; delete the nine-code relax override as modules are fixed (most of its modules are touched in B and G anyway) | OC |
| D4 | mypy and bandit scope (TD-2) | Bandit: `src`, `scripts`, `export`, `docker` in PR1 (fix the 2 High / ~18 Medium; coded `# nosec Bxxx  reason` only for true false positives). mypy: `src`, `scripts`, `docker`, `export` in PR1 if the new error count is <= 150, else `src`, `scripts/curation`, `scripts/codegen`, `docker` in PR1 and the rest in PR9. `models/` (Triton python backend, unimportable stubs) stays out of mypy, in ruff | |
| D5 | Size gate scope | All first-party code: `src scripts export docker models tests frontend/src frontend/e2e frontend/scripts` plus root shell programs; `.py .ts .svelte .js .mjs .sh`. One cap: 700. One baseline file, transition only; each entry names the PR that splits it | OC (tests included at 700) |
| D6 | Frontend size allowlist | Same script and baseline as the backend (no ESLint `max-lines` second mechanism). Entries drop as PR8 splits them; empty at the end | |
| D7 | Layering guard tool | A repo AST checker `scripts/codegen/check_layering.py` (no new dependency; sees function-level imports, which hide the cycles because `PLC0415` is ignored; works with the namespace package `src`). Contracts in section 3.A6. Baseline of today's findings (23 files / 48 imports into `src.routers`, 11 client files / 22 imports into `src.services`, 7 files importing fastapi), emptied in PR4 except D8 | |
| D8 | The guarded OpenSearch client factory imports `services.projects.guard/registry` from `src/clients/opensearch/client.py` | Keep as the one permanent, reasoned contract exception: isolation must wrap every client at construction. Everything else moves | OC |
| D9 | Complexity | ruff `C901` at 15 through a per-function ratchet (`check_ruff_ratchet.py`, 67 functions today). The top 15 (CC >= 38) are decomposed in the split PRs; others may only go down | |
| D10 | Broad excepts | Enable `S110`, `S112` outright (0 hits). `BLE001` (545) through a per-file count ratchet; PR3 triages every return-empty handler | |
| D11 | Dead-code guard | `vulture` pre-commit hook at `--min-confidence 60` with `--ignore-decorators` for route/validator decorators and a reviewed `scripts/codegen/vulture_whitelist.py` (each entry with a reason). Lands in PR2 after the deletions | |
| D12 | Coverage floor (`--cov-fail-under=51`, never enforced: CI passes `--no-cov`) | Remove `--cov*` from default `addopts` (fast local runs); CI pytest job runs `--cov=src --cov=scripts --cov-fail-under=<measured - 1>`; raise the floor whenever a PR raises coverage. Costs CI minutes | OC |
| D13 | Timeouts | Add `pytest-timeout`, `timeout = 300` in pytest config (set after a `--durations=30` run; slowest test x3) | |
| D14 | CHANGELOG check | CI step in `ci.yml` mirroring `frontend.yml`: changes under `src/ scripts/ docker/ export/ models/` require `CHANGELOG.md`, escape `[skip-changelog]` | |
| D15 | Conventional commits | Enable `conventional-pre-commit` at `commit-msg` stage, without `--force-scope`; `default_install_hook_types: [pre-commit, commit-msg]`. Prove merge commits still pass | |
| D16 | Naming-leak `examples/` exemption (TD-6) | Directory-prefix entries apply only to scans B/C; scan A has no directory exemptions | |
| D17 | `check-no-literal-region-fields` hook (TD-7) | Delete hook, script and its allowlist line; the naming scan and `tests/test_no_legacy_region_scalars.py` cover it | |
| D18 | Version guard (TD-8) | Lookahead `(?!\w|\.\d)`; add `SECURITY.md`, `CONTRIBUTING.md`, `docs-site/**/*.md`. Do not add `docs/design`, `docs/research`, `docs/releases`, `frontend/docs/releases`, `CHANGELOG.md`: they are plans and history that name other versions by design | |
| D19 | `models/` lint scope (TD-9) | Into ruff and the size gate; global pre-commit exclude narrows to binary engine files; `export/` blanket `F401` becomes two inline coded noqa | |
| D20 | `scripts/bench/wave0_*` names | Rename the modules and tests by behavior (names in section 11); keep `docs/benchmarks/v060_wave0_*.json` data names (they back published numbers) | OC |
| D21 | Stats "legacy keys" (PC-07) | Verified: `total_crops`, `validated`, `test_holdout`, `by_source` are live wire fields read by Cropwright; keep, delete the misleading comment. `regions.by_human` is a true alias of `by_human_drew`: delete it, repoint the frontend, `make contracts` | |
| D22 | `status_legacy` (PC-12) | Verified live: requeue stashes the prior status there (`region_requeue.py:416`). Not a shim; the name lies. Rename to `status_before_requeue` (`region_status_before_requeue`) with the mapping change; existing stashed values on dev indexes may be dropped (indexes are resettable) | OC |
| D23 | Status aliases `'pending'`, `'pending_verify'` (5 sites) | Delete. Writers emit only `RegionStatus` values; no back-compat by design. Pre-check on the dev stack: count docs with those values; if > 0, reset the test indexes | OC |
| D24 | Env reads (PC-15: 173 raw reads in `src`, 6 private helpers with different malformed-value behavior) | One typed module `src/config/env.py` (`env_str/env_int/env_float/env_bool/env_path`, malformed or out-of-range value raises naming the variable); all `src` reads go through it or `settings.py`; ruff `TID251` bans `os.environ`/`os.getenv` in `src/**` except `src/config/**`. Behavior change: a malformed value now refuses startup instead of silently defaulting | OC |
| D25 | `CROPWRIGHT_*` env names (F-8) | Rename to `OP_CROPWRIGHT_*`; `openprocessor` and `setup-openprocessor.sh` refuse a `.env` that still sets an old name and print the new one (the `src/config/retired_env.py` pattern). Breaking by design | OC |
| D26 | Frontend standalone leftovers (F-7, FD-7) | Verified: the Cropwright image ships through `scripts/release/build_and_publish.sh`; `docs/releases/RELEASE_CHECKLIST.md` says do not run `frontend/scripts/release/*`; the root dev overlay builds `cropwright`. Delete `frontend/scripts/release.sh`, `release/`, `release-smoke.sh`, `frontend/docker-compose.yml`, `docker-compose.build.yml`, `frontend/.pre-commit-config.yaml`, the duplicate diagram and gif scripts; point `frontend/README.md`, `SECURITY.md`, `CONTRIBUTING.md` at the root ones (delete the latter two). Keep `frontend/docs/releases/` as history | OC |
| D27 | `frontend/CLAUDE.md` 3104 lines (FD-3) | Cut to <= 300 lines of current rules; history moves to `frontend/CHANGELOG.md` (already has it) or is deleted; add a line cap check (400) for every `CLAUDE.md` to `check_file_size.py` | |
| D28 | OCC loops (PC-16) | Move the paged retry loop in `routers/curation/regions_edit.py` into `src/clients/occ_bulk.py` as a callback-driven function both callers use; the single-doc loop in `occ.py` stays. Concurrency surface: independent review before merge | |
| D29 | Frontend duplicate formatters (F-5) | One `lib/format/duration.ts`; output style `1h 05m` / `5m 03s` / `42s`, fallback string is a parameter. Visible text changes on four panels | OC (style) |
| D30 | Modal, SSE, polling (FD-4, FD-5) | Shared `lib/components/Modal.svelte` (backdrop, Escape, focus trap, aria) replacing the 46 hand-rolled dialogs; one `openStream()` in `lib/sse.ts`; `lib/poll.svelte.ts` for interval polling. Browser check every migrated dialog | |
| D31 | Dockerfile bases (FD-9) | Pin `FROM` by digest in all Dockerfiles; bump procedure documented in `docs/releases/RELEASE_CHECKLIST.md`; `docker/segmenter` and `docker/evaluator` become multi-stage; `Dockerfile.fake-vlm` gets a `USER` | OC |
| D32 | Build context and image content (FD-8) | Add `artifacts_local/ docs-site/ frontend/ .claude/ .install/ export_*/` to `.dockerignore`. Narrow `Dockerfile` `COPY scripts/` to the runtime set (`scripts/curation`, `scripts/__init__.py` and what they import), verified by building and running the live harness | OC |
| D33 | Root layout and names (TD-10, TD-11) | Do: delete `requirements-dev.txt` if no consumer remains (only docs mention it), move `src/routers/curation_train/` to `src/routers/curation/train/`, `curation_images.py` and `curation_umap.py` into `src/routers/curation/`, legacy `src/services/clustering.py` into `src/services/visual_search/`, domain code out of `src/utils/` (`scrfd_decode.py` to `src/services/face_identity/`, `pytorch_utils.py` to `src/services/`). Do not move compose overlays or rename `Dockerfile`: installed users, the installer, release scripts and docs reference them; cost exceeds benefit | OC |
| D34 | `noqa` reasons (T-4) | Pygrep hook: every `# noqa: CODES` in `src scripts docker export` must be followed by `  # <reason>` or a trailing reason; T-4's pydantic `runtime-evaluated-base-classes` removes ~57 `TC001` noqas first | |
| D35 | Ruff ignore list (T-3) | Remove dead entries (`DTZ005`, `TRY003`, tests `S101`, `[tool.ruff.lint.pydocstyle]`, dali blocks); enable `PTH123` and fix 54 sites; keep `EM101/EM102/SIM108/PLR2004/PLC0415` ignored with honest one-line reasons | |
| D36 | `src/utils/__init__.py` re-exports (T-5) | Empty it; repoint the one doc example (`src/utils/README.md:190`) | |
| D37 | Process-id comments | Drop the id, keep the why; replace with a public issue `#NNN` where one exists. Guard scope: comments and docstrings of code files, and file names under `tests/`, `frontend/src`, `frontend/e2e`, `scripts/` | |

## 2. PR sequence

Sequential (each starts from `main` after the previous merge). Est. CI 30 min each.

| PR | Branch | Content | Depends |
|---|---|---|---|
| 1 | `chore/cleanup-guards` | A: config fixes, every guard, transition baselines | - |
| 2 | `chore/cleanup-dead-code` | B + C: dead code, shims, process ids, history-named tests, vulture/id/filename guards | 1 |
| 3 | `fix/cleanup-fail-open` | E: fail-open fixes with failing-first tests, BLE001 triage | 2 |
| 4 | `refactor/cleanup-layering` | L: layering contract to zero, fastapi out of services, router logic to services, env module | 3 |
| 5 | `refactor/cleanup-duplicates` | K + F: backend and frontend duplicates, frontend/scripts/env/docker hygiene | 4 |
| 6 | `refactor/split-backend` | G1: every `src/` and `scripts/` file >= 640 lines split by concern, top-CC functions | 5 |
| 7 | `refactor/split-tooling-tests` | G2: `export/ docker/ models/` and shell programs, oversize tests | 6 |
| 8 | `refactor/split-frontend` | G3: `api.ts` family, pages, types, oversize frontend tests | 7 |
| 9 | `chore/cleanup-finish` | H: typing ladder, baselines at their final state, `docs/design/engineering_standards.md` | 8 |

PR 8 can run in parallel with PR 7 (disjoint trees: `frontend/` vs the rest) if both rebase
cleanly on `CHANGELOG.md`; the baseline file `scripts/codegen/baselines/file_size.txt` is the one
shared file: each PR only deletes its own lines, so conflicts resolve by keeping both deletions.

Within a PR, tasks are grouped into **phases**; tasks in one phase own disjoint files and can go
to parallel implementers; phases run in order.

## 3. PR 1 `chore/cleanup-guards` (section A)

First commit: `docs(design): add the repo cleanup plan` with this file (it was written untracked
in the main checkout; copy it into the worktree) and its row in `docs/design/README.md` (status
in progress). New directory `scripts/codegen/baselines/`; every baseline line is
`key<TAB>value<TAB>reason`.
Every checker gets unit tests under `tests/codegen/` that feed a planted violation and assert it
fails, a clean input passes, and a stale baseline entry fails.

**Phase 1 (parallel)**

- **A1 pyproject cleanup** (`pyproject.toml`, `.gitignore`). Remove `dali/**/*.py` include, the
  dali per-file block, `nvidia.dali.*` override (T-1); `src.utils.face_alignment` (T-2); dead
  ignores per D35; add `[tool.ruff.lint.flake8-type-checking] runtime-evaluated-base-classes =
  ["pydantic.BaseModel", "pydantic_settings.BaseSettings"]` then
  `.venv/bin/ruff check --fix --select RUF100 src scripts docker export` to drop stale noqas (T-4);
  enable `PTH123`, `S110`, `S112`, `TID251` scaffold (empty ban list); remove `models` from ruff
  `exclude`, add `models/**/*.py` to include, fix `models/ocr_pipeline/1/model.py:44` F401;
  narrow export `F401` to inline coded noqa; `.gitignore` adds `.install/`. bandit: remove the
  no-op `B601` skip; remove the global `B311` skip if <= 10 sites need a coded nosec (T-7).
  New test `tests/test_tooling_config.py`: every pyproject include glob and per-file-ignores key
  matches >= 1 tracked file; every mypy override module resolves to a module or package; every
  pre-commit `exclude`/`files` literal path prefix exists. Accept: test fails on today's
  pyproject (dali, face_alignment), passes after.
- **A2 mypy hook** (`.pre-commit-config.yaml` mypy block, `requirements-test.txt`, `[tool.mypy]`,
  files with errors). Per D2/D4: local hook
  `entry: bash -c 'py=python3; [ -x .venv/bin/python ] && py=.venv/bin/python; exec "$py" -m mypy'`,
  `language: system`, `pass_filenames: false`, `files: ^(src|scripts|docker|export)/.*\.py$`;
  targets in `[tool.mypy] files`, `exclude` for `src/ultralytics_patches`. Add `mypy==<pin>`,
  `types-PyYAML`, `types-requests`, `types-tabulate`, `types-tqdm` to `requirements-test.txt`.
  `warn_unused_ignores = true`. Fix all errors: Pillow constants to `Image.Resampling.*`,
  `ImageFile` vs `Image` annotations, `None` narrowing, `var-annotated`. Reference output:
  `artifacts_local/review/mypy_project_env.txt` (53 errors in 35 files on 1.13). Accept:
  `.venv/bin/python -m mypy` exits 0; warm run time recorded in the PR (target < 60 s).
- **A3 bandit scope** (bandit hook `exclude`, findings in `scripts/ export/`). Exclude only
  `^(src/ultralytics_patches/|tests/|frontend/)`. Fix: `hashlib.sha1(..., usedforsecurity=False)`
  in `scripts/datasets/build_import_fixture.py`; `B104` in `scripts/curation/worker/runner.py:66`
  (bind from config, default loopback unless the compose service sets it); `B310` urlopen: assert
  `https?` scheme or coded nosec with reason; `B608` bench SQL: parameterize or nosec with reason.
- **A4 hooks file** (`.pre-commit-config.yaml` except the mypy block and `max-file-size`). D1
  bumps (commit a); header comment `# Pre-commit configuration for OpenProcessor`; delete
  `check-no-literal-region-fields` plus `scripts/codegen/check_no_literal_region_fields.py` and
  its allowlist line (D17); conventional-commit hook (D15); `default_install_hook_types`; global
  `exclude` keeps `models/` binaries only (`models/.*\.(plan|onnx|engine|bin|pt)$`); noqa-reason
  pygrep hook (D34). Bump hadolint and golangci-lint pins in `.github/workflows/ci.yml` to match.
- **A5 size gate** (`scripts/codegen/check_file_size.py`, its hook, baseline
  `baselines/file_size.txt`). Extend the script: enumerate `git ls-files` for the D5 scopes
  (`pass_filenames: false`, `always_run: true`), `--max 700`, `--warn 640` (prints the danger
  band, never fails), `--baseline`; fail when a non-baselined file exceeds 700, a baselined file
  grows past its recorded count, or a baseline entry is stale (file now <= 700 or gone). Add the
  `CLAUDE.md` cap (400) per D27 as a second rule in the same script. Seed the baseline with
  today's oversize files (about 53: 8 export/docker/models, 19 tests, 19 frontend, 7 shell),
  reason column `split in PR <n>` or `cohesive: <reason>` (section 11).
- **A6 layering guard** (`scripts/codegen/check_layering.py`, hook, `baselines/layering.txt`).
  AST walk of every `Import`/`ImportFrom` (module level and function level). Contracts:
  (1) `src/{services,clients,core,config,utils,schemas}` and `scripts/` (except
  `scripts/codegen/`) must not import `src.routers`; (2) `src/clients` must not import
  `src.services`; (3) `src/{services,clients,config,utils,schemas}` must not import `fastapi` or
  `starlette`; (4) `src/{config,utils,schemas}` must not import `src.services` or `src.clients`
  (measure; baseline whatever exists). Baseline key `path -> module`; stale entries fail.
- **A7 complexity and broad-except ratchet** (`scripts/codegen/check_ruff_ratchet.py`, hook,
  `baselines/complexity.txt`, `baselines/broad_except.txt`). Runs
  `ruff check --isolated --select C901 --config lint.mccabe.max-complexity=15
  --output-format json` and `--select BLE001` over `src scripts docker export models`.
  Complexity key `path::function` value = CC (from the message); new function > 15 fails, a
  baselined one may not grow; BLE001 key `path` value = count, may not grow. Stale entries fail.
- **A8 docs and CI guards** (`tests/test_version_consistency.py`, `.github/workflows/ci.yml`,
  `pyproject.toml` pytest block, `scripts/codegen/check_naming_leaks.py`,
  `scripts/docs/check_docs_vs_code.py`). D18 regex and roots (fix any mention it newly flags);
  D14 changelog step; D12 coverage; D13 timeout (`pytest-timeout` in `requirements-test.txt`);
  D16 scan-A semantics with a unit test that puts a scan-A term under `examples/` and expects a
  hit; extend the env check to `${VAR}` names in compose files (F-8: any compose variable must be
  in `env.template`, commented or not).

**Phase 2:** **A9 env.template** fill the F-8 gaps found by A8 (advanced, commented section):
`CROPWRIGHT_INGEST_MAX_REQUEST_MB`, `CROPWRIGHT_DATASET_UPLOAD_MAX_MB`, `CROPWRIGHT_IMAGE`,
`PUBLIC_API_PREFIX`, `PUBLIC_TRITON_API_URL`, every `*_IMAGE` pin, `API_UPSTREAM`,
`DOCS_UPSTREAM`, `OP_DOCS_IMAGE`, `OP_VLM_WAIT_S`, `OP_VLM_POLL_S` (renames per D25 happen in PR5).

**Guard proof procedure** (run in the worktree, record output in the PR; remove the plant):

| Guard | Plant | Expect |
|---|---|---|
| mypy | `src/_probe.py`: `import numpy as np` / `def f() -> str: return np.zeros(3)` | mypy error (was MISSED) |
| bandit | `scripts/_probe.py`: `subprocess.run('ls', shell=True)` | B602 |
| size | 701-line `export/_probe.py`, `frontend/src/lib/_probe.ts`, `tests/_probe.py` | fail each; baselined file +1 line fails |
| layering | `src/services/_probe.py`: `from src.routers.curation import _common` inside a function | fail |
| complexity | 16-branch function in `src/_probe.py` | fail |
| BLE001 | add an `except Exception:` to a baselined file | fail |
| naming leak | scan-A term in `examples/_probe.yaml` (staged) | fail (was MISSED) |
| version | `docs/_probe.md`: `shipped in v0.9.9.` | fail (was MISSED) |
| noqa reason | `x = 1  # noqa: E741` in `src/` | fail |
| commit-msg | `git commit --allow-empty -m "bad"` fails; `-m "chore: ok"` and a merge commit pass | as stated |
| config paths | add `"nope/**/*.py"` to ruff include | test fails |

Acceptance PR1: all hooks green on `--all-files`; every proof row recorded; full offline suite
green; CI green including the changelog step. CHANGELOG `### Changed`: tooling bullets.

## 4. PR 2 `chore/cleanup-dead-code` (sections B and C)

**Phase 1 B (one implementer, one commit per group; re-run `git grep -w <name>` per symbol
first, delete a now-unused import with it, update the vulture/size/complexity baselines).**

- **B1** delete `src/services/cluster_maintenance.py` (PC-01) and the mention in
  `docs/design/ingest_surfaces_plan.md:73`.
- **B2** delete the PC-02 list (verified 1 reference each): clients `opensearch/ocr.py`
  `get_ocr_for_image`, `delete_ocr_for_image`; `opensearch/search.py` `get_faces_by_person_id`;
  `triton_client.py` `infer_arcface_batch`; `triton_pool.py` `infer_batch`, `reset_stats`,
  `get_async_client_pool_stats`; `core/exceptions.py` `PreprocessingError`; services
  `cpu_preprocess.py` `preprocess_for_triton`, `preprocess_batch_for_triton`; `embedding.py`
  `encode_image_batch`, `get_embedding_norm`; `image.py` (whole unused class: `resize_if_needed`,
  `get_dimensions`, `validate_size`, `convert_to_jpeg`); `face_identity/face_records.py`
  `get_face_by_id`, `delete_face`; `face_identity/identification.py` `ingest_face`,
  `assign_face_to_person`; `clustering.py` `search_similar_clusters`, `get_cluster_centroid`;
  `duplicate_detection/__init__.py` `remove_from_group`, `merge_groups`;
  `visual_search/lifecycle.py` `setup_index`, `ping_opensearch`; `detection/region_lean.py`
  `deskew`, `recover_lean`; `bulk_ingest/sources.py` `folder_source_id`; `ivf_store.py`
  `assign_one`; `dataset_import/issues.py` `has_unbypassable_blocking`; `dataset_import/store.py`
  `drop_ledger`; `dataset_import/runner.py` `stable_hash`; `keymap.py` `actions_in_context`;
  `projects/script_binding.py` `bind_script_project_from_env`; schemas in
  `schemas/visual_search.py` (`DetectOnlyResponse`, `FaceFullResponse`, `PersonFacesResponse`,
  `FaceIngestRequest/Response`, `FaceIdentityIdentify*`, `PersonFacesListResponse`,
  `FaceAssignRequest/Response`), `schemas/models.py` `ModelUploadRequest`, `routers/models.py`
  `TestInferenceRequest`, `routers/clusters.py` `AlbumInfo`. If a schema appears in
  `contracts/`, run `make contracts`.
- **B3** test-only production code (PC-03): delete with their tests `crop_blur`/`is_blurry`,
  `get_curation_index_settings`, `visual_search/lifecycle.delete_index`, `clear_request_id`,
  `event_hub.publish_region_verified` (the worker emits `crop.region_verified` over HTTP; fix the
  comment in `frontend/src/lib/sse.ts:111`); **wire in** `schedule_due`: `dispatcher.py:99-106`
  re-implements it, call it instead. PC-04 unused names (`export/preflight.py:24`,
  `vlm_catalog.py:117`, `stage_timing.py:81`).
- **B4** shims (PC-07, F-2, T-5): drop the `_common.py:340+` re-export block and repoint all
  importers of those models to `_common_models` (109 import lines; use
  `rg -l 'from src.routers.curation._common import'` and split each import by origin); inline or
  rename `class_label.human_label_update` honestly; D21 (`regions.by_human` + `_stats_models.py:37`
  + frontend + contracts); delete `scripts/curation/region_worker_main.py`, repoint
  `tests/integration/wheel_worker.py:18`, `tests/curation/test_worker_vlm_hot_reload.py:469,499`,
  `tests/curation/test_region_cascade_integrity.py:29` to `scripts.curation.worker` submodules,
  fix docs `scripts/README.md:140`, `docs/ARCHITECTURE.md:421`, docstrings in
  `scripts/curation/worker/*`, `src/routers/curation/events.py:89`; D36; T-6 stale docstrings
  (`triton_pool.py`, `core/dependencies.py`, `.hadolint.yaml`, `.dockerignore` old-name headers);
  `triton_pool.py:5` "legacy" claim reconciled.
- **B5** D22 rename and D23 alias deletion (if confirmed): `region_fields.py`, `ensure_fields.py`,
  `bodies_core.py`, `wire.py`, `region_requeue.py`, `scripts/curation/requeue_regions.py`,
  `preflight_checks.py`, `routers/curation/ingest.py`, `scripts/curation/worker/{cascade,state}.py`,
  `region_eval.py`, `ops_metrics_refresh.py`; `make contracts`.
- **B6** vulture hook (D11) with whitelist; proof: plant an unused function, see it caught.

**Phase 2 C (parallel by tree; mechanical, reviewed diffs).** A one-off rewrite helper lives
in `artifacts_local/cleanup/strip_process_ids.py` (not committed). It strips the safe shapes
`\s*\((?:W|R|F-)[\w.-]+\)`, a leading `W\d+(\.\d+)*[a-z]?:\s*`, `round-\d+ ` tags, and lists
every residual hit for a hand rewrite that keeps the why. Pattern set (also the guard's):
`\bW\d+(\.\d+)*[a-z]?\b`, `\bR\d+-\d+\b`, `\bWP-[A-Z]?\d[\w.]*`, `\bround-\d+\b`, `\bF-\d+\b`,
`\bChunk \d+\b`, `\bWave \d+\b`, `\bdqM\d+`, `\bPC-\d+`. Today: ~270 hits in Python, 61 in
`frontend/src`. Review with `git diff --word-diff`.

- **C1** `src/routers/**`; **C2** `src/services/**`; **C3** `src/{clients,config,core,utils,schemas}`
  + `docker/ export/ models/`; **C4** `scripts/**`, Makefile, shell; **C5** `tests/**` content +
  renames (section 11 renames; class names inside too, e.g. `TestR7*`); **C6** `frontend/src`,
  `frontend/e2e` content + renames (section 11).
- **C7** guard `scripts/codegen/check_process_ids.py` (hook, unit tests): pattern set in comments
  and docstrings of `.py .ts .svelte .js .mjs .sh` and Dockerfiles (Python via `tokenize`,
  others by comment-prefix lines), plus file names under `tests/ scripts/ frontend/src
  frontend/e2e` against `(^|[_/])(r\d+|w\d+[a-z]?|wave\d+|round\d+)[_.]|[0-9a-f]{7}\.|dqM\d|
  logicMoves|interactivePass`. No baseline: zero hits after C1-C6. Proof: plant a comment
  `# W8.2: x` and a file `tests/test_r9_x.py`.

Acceptance PR2: `git grep -w` count 1 for none of the deleted names; vulture green; id guard
green; contracts in sync; full suite + frontend checks green; size/complexity/BLE001 baselines
updated (stale entries removed). CHANGELOG `### Removed`.

## 5. PR 3 `fix/cleanup-fail-open` (section E)

Rule for each fix: write the test that simulates the outage (client raising) and asserts the
fail-closed outcome; watch it fail; fix; keep the user-visible distinction between "unknown"
and "zero". Wire changes (`int` to `int | None`) require `make contracts` and a frontend render
of "unknown" (browser check).

- **E1** `services/projects/stats.py` `_term_count`, `index_count` (and their callers in
  `routers/curation/projects.py:101-102`): return `None` and log, as `_query_count` already does;
  serve `null`; Cropwright shows "unavailable".
- **E2** `training/preflight_checks.py:148` `count_pending_ingest`: on failure emit a preflight
  warning "pending ingest could not be counted" instead of 0; `:295` `read_export_manifest`:
  distinguish missing file (`{}`) from unreadable (raise).
- **E3** `config_store/vlm_gate.py:102` `_registry_class_names`: let the error propagate
  (activation refuses); in PR4 (L2) the three copies merge.
- **E4** `routers/curation/_common.py` `_ensure_indexes_locked`: replace the ~22 copied try
  blocks with an ordered list of `(name, migration)`; collect failures; do not add the slug to
  `_INDEXES_BOOTSTRAPPED` if any failed (the next request retries); log each failure once.
- **E5** legacy visual-search client `clients/opensearch/clusters.py:200,259,303`, `ocr.py`,
  `search.py`, `ingest.py:115,173,214`: raise a typed `UpstreamError` the routers map to 503
  (`src/utils/upstream_errors.py` exists); callers no longer see `0`/`[]` for an outage.
- **E6** `export_readiness.py:90`: verify every caller treats `None` as "unknown"; add the test.
- **E7** `background_jobs.stop_background_jobs`: stop suppressing `Exception` (log the crash);
  `bulk_ingest/dispatcher.run_tick` wraps `run_project_tick` in `asyncio.to_thread` (PC-17).
- **E8** issue #161 `file_job.FileJob.read`: an unreadable or non-dict state makes the busy
  check refuse to start (409 with reason), per consumer test.
- **E9** triage of the remaining BLE001 return-empty and log-only handlers. List:
  `.venv/bin/ruff check --isolated --select BLE001 src scripts docker export --output-format
  concise`. For each: (a) best-effort and correct (probe, poller tick, per-project isolation,
  warm-up): narrow the exception type where possible and add a one-line why comment; (b) hides an
  outage behind an empty answer: fix as above with a test. Record the triage table (path:line,
  verdict) in the PR description. Shrink `baselines/broad_except.txt` accordingly.

Acceptance: each fix has a test seen failing; BLE001 baseline lower; CHANGELOG `### Fixed`.

## 6. PR 4 `refactor/cleanup-layering` (section L)

Goal: `baselines/layering.txt` holds only the D8 exception.

- **L1 models out of routers.** Move `src/routers/curation/_config_common_models.py`
  (`ValidationIssue`, `ValidationReport`, `api_error`'s payload model), `_models_segmenter.py`,
  `_models_class_mapping.py`, `_project_models.py` to `src/schemas/curation/` (same module names
  without the leading underscore); `_bakeoff_models.py` (663 lines) goes straight to
  `src/schemas/bakeoff/` split as in section 11; routers and services import from there.
  `api_error` itself (it builds an `HTTPException`) stays in routers; services raise domain errors.
- **L2 helpers out of routers.** `_registry_class_names` (3 copies) to
  `src/services/curation/class_registry_names.py` (raises on failure); `_resolve_profile`,
  `bound_project_slug`, `_ensure_indexes` bootstrap to `src/services/curation/index_bootstrap.py`
  (also used by `scripts/curation/seed_live_harness.py:455` and `core/dependencies.py:145`);
  `routers/curation/select.py` logic used by `crop_orders.py` and `ingest._get_ingest_service`
  used by `open_vocab_sweeper.py` move to services.
- **L3 fastapi out of services** (7 files: `ocr_service.py`, `image_serving.py`, `holdout.py`,
  `projects/lifecycle.py`, `review_queries.py`, `clone_activation_gate.py`,
  `projects/combine/plan.py`): domain exceptions in `src/core/exceptions.py` mapped in
  `src/core/error_handlers.py`; `FileResponse` built in the router from a path the service
  returns. Each: a router test asserting the same status code and body as before.
- **L4 clients off services.** Leaf constants and pure predicates move down:
  `item_text`, `open_vocab_fields`, `vlm_class_attempt` field constants, `audit_math`,
  `embedding_state` clause, `region_boxes` readers and `ingest_class_sources` predicates used by
  `occ_locks.py` go to `src/config/` (field names) or a new `src/domain/` package (pure functions,
  no I/O, imports only `src/config`); `history`/`metrics` side effects in `occ_upsert.py` are
  passed in by the service caller; `clustering` constants for `clients/opensearch/*` move to
  `src/config`; `pe_preprocess`/`cpu_preprocess` calls in clients move to the service callers.
  Contract (4) then also covers `src/domain`.
- **L5 router business logic to services:** `routers/curation_train/preflight.run_preflight`
  (CC 64), `promote.promote_run` (38), `routers/curation/pipeline._run_auto_label` (49) become
  `src/services/training/preflight_run.py`, `src/services/training/promote_run.py` (models to
  `src/schemas/train_promote.py`), `src/services/curation/auto_label_run.py`, decomposed under
  CC 15 per section 11; routers parse, call, map errors.
- **L6 env module (D24):** `src/config/env.py` + tests (malformed and range cases); repoint the
  173 reads (`rg -n 'os\.(environ|getenv)' src`), largest first (`dataset_import/limits.py`,
  `bulk_ingest/object_store.py`, `vlm_client.py`, `strategy_registry.py`, `routers/curation/select.py`);
  delete the private helpers; enable the `TID251` ban. `tests/test_env_surface.py` stays green.

Phases: L1 alone first (wide import churn), then L2+L3 (disjoint files), then L4, L5, L6 in
parallel (disjoint). Acceptance: layering baseline = D8 line; ruff `TID251` green; routes'
status codes unchanged (existing router tests); contracts unchanged (`make contracts-check`).

## 7. PR 5 `refactor/cleanup-duplicates` (sections K and F)

Backend (phase 1, parallel):
- **K1** `src/utils/timeutil.py` `now_iso()` = `datetime.now(UTC).isoformat()` (full precision;
  the two `timespec='seconds'` sites change precision: stored strings stay ISO-8601 and sort
  correctly; check no test compares exact strings); delete the 14 copies.
- **K2** OCC per D28 (`occ_bulk.py`, `routers/curation/regions_edit.py`); other hand-rolled
  retries (`assignment_writer.py:83`, `region_box_embeddings.py:164`, `triton_pool.py:276`) use
  `src/utils/retry.py`; one retry library (keep tenacity only if `vlm_client.py` needs its
  features, else `retry.py`).
- **K3** `_registry_class_names` already merged in L2; rename the `get_triton_client` in
  `clients/triton_client.py` (it collides with `triton_pool.get_triton_client`) to
  `get_sync_triton_client`; `fast_face_client`/`ocr_service` stay on the sync client (live).
Frontend (phase 1, parallel; `frontend/CHANGELOG.md` entry):
- **K4** formatters per D29: `lib/format/duration.ts` (+ tests), reuse `combineText.formatBytes`
  (move to `lib/format/bytes.ts`), `formatCount`, `formatLatency`, one `formatTime`; a scan test
  `lib/format/format.scan.test.ts` fails on `function (formatDuration|fmtBytes|formatTime|fmtEta)`
  outside `lib/format/`.
- **K5** `Modal.svelte` per D30; migrate the 46 dialogs; a scan test forbids
  `svelte-ignore a11y_click_events_have_key_events` outside `Modal.svelte`.
- **K6** `openStream()` in `lib/sse.ts` (3 call sites); `lib/poll.svelte.ts` for the
  `setInterval` pollers in controllers (tests with fake timers for cancellation).
- **K7** FD-6: replace the 39 `prefer-svelte-reactivity` suppressions with `SvelteDate`/
  `SvelteMap`/`SvelteSet`, or where non-reactive by intent, use a plain local and drop the
  suppression; the rule stays on.
Hygiene (phase 2, parallel):
- **F1** dead frontend exports (F-3): delete `staleProjectError`, `probeCombine`,
  `formFromStatus`, `isStringList`, `toastWarnings`, `selectedModelRefs`,
  `findApiPrefixTemplates`, `keymapConflictText`, `toastKeymapConflicts`, the unused fixtures,
  `frontend/scripts/lib/previewServer.d.mts`, `tslib`, the duplicate `builtinSlots|
  registeredSlots` export; drop `export` on file-local helpers. Add `knip` (`frontend/knip.json`,
  route entries, `ignoreExportsUsedInFile: true`, types not reported) as a **blocking** step in
  `.github/workflows/frontend.yml`.
- **F2** #155 / F-4: move `$components`, `$stores`, `$contracts` to `package.json` `imports`;
  CI build fails on the deprecation text (grep the build log).
- **F3** D26 deletions; D27 `frontend/CLAUDE.md`; D25 renames (`docker-compose*.yml`,
  `frontend/nginx*.conf` templates, `env.template`, `openprocessor`, `setup-openprocessor.sh`,
  docs).
- **F4** scripts: `--help` for `scripts/generate-architecture-diagrams.sh`,
  `scripts/docs/create-workflow-gif.sh`, `scripts/release/verify_release_assets.sh`, argparse in
  `scripts/create_ocr_test_images.py`, `scripts/download_ocr_test_images.py`; new test
  `tests/test_scripts_help.py` runs every `scripts/**/*.py` with a `__main__` guard with `--help`
  (subprocess, 10 s each) to catch import breakage.
- **F5** D31 Dockerfiles; D32 `.dockerignore` and runtime `COPY` (build with
  `docker compose -p cleanup-img build api`, then the live harness per `docker/test/README.md`).
- **F6** T-9 tracked export artifacts: `pytorch_models/yolo11s-trt.txt`,
  `pytorch_models/yolo11_face/yolov11s-face-trt.txt`: `git grep` for readers; untrack and
  gitignore if only written by export, else move next to the code that reads them.
- **F7** D33 moves (router package moves, `services/clustering.py`, `utils` domain modules,
  `requirements-dev.txt`).

Acceptance: scan tests fail before migration, pass after; knip blocking green; browser check of
every page with a migrated dialog or formatter (screenshots in the PR); live harness green after
F5.

## 8. PRs 6-8 split by concern (section G)

Per file: create the new modules, move code verbatim (no logic edits in the same commit), repoint
every importer, delete the moved code, then (separate commit) decompose functions over CC 15 in
the moved code. One commit per original file. Accept per file: every resulting file <= 450
lines target, hard <= 700; its baseline line deleted; targeted tests of the area green; no
re-exports. The per-file designs are in section 11 (split tables).

- **PR 6 `refactor/split-backend`**: every `src/` and `scripts/` file >= 640 lines (31 files) and
  `src/services/duplicate_detection/__init__.py` (logic out of `__init__`). Phases by package:
  (a) `src/routers` + `src/main.py`, (b) `src/clients`, (c) `src/services/curation`,
  (d) other `src/services` + `src/config`, (e) `scripts/`. Top-CC functions not removed by L5:
  `dataset_import/chunk._import_one` (57), `region_eval.evaluate` (53),
  `visual_search/ingest_batch.ingest_batch` (48), `projects/guard.check_request` (46),
  `export.export_dataset` (40), `occ_upsert.occ_upsert_bulk` (40),
  `keymap_validator.validate_keymap` (38), `routers/analyze.analyze_batch` (38),
  `scripts/datasets/build_import_fixture.build_variants` (68), `fetch_coco_subset.run` (42),
  `scripts/bench/parity_assemble.build_arrays` (50). `guard.check_request` is isolation code:
  extract pure helpers only, independent review.
- **PR 7 `refactor/split-tooling-tests`**: `export/` (5), `docker/trainer/` (2),
  `models/ocr_pipeline/1/model.py`, the shell programs, and the 19 oversize test files.
- **PR 8 `refactor/split-frontend`**: `api.ts` into the `api_<domain>.ts` family (FD-1),
  the pages (FD-2), `types.ts`, `parseSlotConfig.ts`, controllers, oversize tests, e2e
  `conftest.py`, `frontend/scripts/capture_docs_screenshots.py`. Browser check every split page.

## 9. PR 9 `chore/cleanup-finish` (section H)

- **H1** typing ladder D3 (and the deferred mypy scope from D4, if any): enable the flags, fix (expect ~135 untyped defs, ~85 incomplete,
  ~66 return-any on today's numbers; fewer after deletions); delete the relax override.
- **H2** baselines: `file_size.txt` holds only section-11 cohesive entries; `layering.txt` only
  D8; complexity and broad-except baselines show the post-cleanup state; add a unit test that
  fails if `file_size.txt` gains a line without the `cohesive:` reason prefix.
- **H3** write `docs/design/engineering_standards.md` (section 12) from the real results: the
  measured numbers before and after, each guard's proof output, what went wrong.
- **H4** update `docs/design/README.md` (this plan's status, the new doc), `CONTRIBUTING.md`
  (hooks, baselines, how to read a ratchet failure; a file entering the 640-700 band is split by
  concern in the same PR, collapsing lines to fit is rejected in review), root `CLAUDE.md`
  "Pre-commit and checks".

## 10. Risks, and what to do when a guard finds too much

| Risk | Mitigation |
|---|---|
| A guard lands with hundreds of findings | Never mass-ignore (no blanket noqa, no excluding a directory, no raised threshold). Record today's findings in a baseline with a reason column; the guard fails on growth and on stale entries, so the baseline only shrinks; schedule its emptying in this plan |
| Vacuous guard (passes because it sees nothing: missing deps, wrong `files:` regex, excluded dir) | Every guard has a planted-violation unit test in `tests/codegen/` and a proof row; CI runs pre-commit `--all-files` |
| mypy hook slow or env-dependent | `pass_filenames: false` + incremental cache; falls back to `python3` (CI has deps); record warm time |
| Ruff bump floods findings or reformats many files | Separate commit in PR1 before any other edits; if > 300 files reformat, ask the owner whether to defer the bump to PR9 |
| Moving router models (L1) breaks the OpenAPI schema names | `make contracts-check` after L1; Pydantic schema names are class names, not module paths; if any changes, regenerate and check Cropwright types |
| Fail-open fixes change UI (null counts) | Contracts + frontend "unknown" rendering in the same PR; browser check |
| Splits break `unittest.mock.patch('module.symbol')` targets in tests | After each move: `rg "patch\(['\"]<old.module>"` and repoint; full suite once per PR |
| Concurrency/isolation code touched (D28, guard.py, OCC) | Behavior-identical moves, existing tests unchanged, independent review (Sonnet reviewer) before merge |
| Long PR queue blocks other work | Each PR is green and mergeable alone; other branches rebase on `main`; `CHANGELOG.md` conflicts keep both |
| Env fail-closed (D24) refuses startup on an existing `.env` with a malformed value | Error names the variable; CHANGELOG `### Changed` calls it out |

## 11. Split designs

Designs from a per-file read on `64ce2eff` (line ranges approximate; re-read before moving).
"keep" = what stays in the original file. Cross-cutting rules: repoint `mock.patch` /
`monkeypatch` targets and source-scan tests that read a moved file; keep route registration
order when splitting routers (catch-all paths such as `/review/{tab}`, `/crops/{crop_id}`,
`/{model_name}/...` after the literal ones); `export/` and `docker/trainer/` use flat sibling
imports, `scripts/` uses package imports; new shipped `scripts/lib/*.sh` must be added to
`release-manifest.txt`. Do L1 (models out of routers) before any router split.

**PR 6, `src/` and `scripts/` (phase order: pure moves, then route/service extractions, then
`main.py`, then `src/config/curation.py` last via a codemod: 241 importers)**

| File (lines) | New modules / what moves | Notes |
|---|---|---|
| `routers/models.py` (700) | `routers/models_lifecycle.py` (status/load/unload/delete/list); `routers/models_export.py` (upload/export/exports, size limits); `schemas/model_admin.py` (request/response models); keep test-inference | one prefix, include order preserved |
| `clients/triton_pool.py` (700) | `clients/triton_pool_core.py` (options, `PoolStats`, `uint8_images_input`, `AsyncTritonPool`); `clients/triton_client_manager.py` (`TritonClientManager`, module functions) | 21 importers; one singleton home; `get_triton_client` collision handled in K3 |
| `routers/curation/pipeline.py` (694) | `services/curation/auto_label_run.py` (L5, stage functions <= CC 15); router thin | characterize pack pin and `ensure_fresh` order first |
| `main.py` (694) | `src/app/resources.py`, `src/app/lifespan.py` (startup/shutdown steps, arbiter loop), `src/app/factory.py` (`create_app`); `main.py` keeps `app = create_app()` | 67 files patch `src.main.*`; `uvicorn src.main:app` stays |
| `curation/clustering/embedding_reduce.py` (692) | `clustering/residual_fetch.py`, `clustering/umap_state.py`; keep reducer fit/rebuild | |
| `curation/strategy_registry.py` (689) | `strategy_flags.py`, `strategy_builders.py`, `field_coverage.py`; keep `get_registry` | |
| `routers/curation/crops.py` (688) | `crops_list.py`, `crops_write.py`, `crops_exclude.py`; query building to `services/curation/crop_listing.py`, OCC bulk helpers to `services/curation/crop_bulk.py` | lock rule unchanged |
| `curation/export_single_class.py` (687) + `curation/export.py` (665) | shared `export_models.py` (profiles, results, artifact names) and `export_writers.py` (labels, yaml, registry, stats, image copy); each service keeps orchestration | dedupes the two exporters; `export_dataset` decomposed |
| `routers/analyze.py` (683) | `schemas/analyze.py`; `services/analyze_pipeline.py` (shared single/batch orchestration); thin sync handlers | |
| `projects/delete.py` (678) | `projects/delete_paths.py`, `projects/delete_models.py`; keep dry-run/delete/finish | safety checks unchanged |
| `training/gpu_arbiter.py` (677) | `gpu_arbiter_policy.py` (parse, lock file), `gpu_arbiter_docker.py`; keep claim/release/reconcile | fail-closed preserved |
| `clients/pe_encoder.py` (674) | `clients/pe_text_backends.py`; keep `PEEncoder` and image path | lazy torch/onnx imports stay lazy |
| `config_store/store.py` (673) | `config_store/snapshot.py`, `config_store/store_poll.py`; keep `ConfigStore` | |
| `duplicate_detection/__init__.py` (668) | `duplicate_detection/service.py`, `duplicate_detection/queries.py`; `__init__.py` empty | |
| `training/jobs.py` (664) | `training/job_manifest.py`, `training/job_campaigns.py` | imported `as train_jobs`: repoint per symbol |
| `projects/clone.py` (664) | `projects/clone_validate.py`, `projects/clone_activation.py` | |
| `curation/ingest.py` (664) | `ingest_helpers.py`, `ingest_items.py`, `ingest_dedup.py`; keep service | `_crop_id` format byte-identical |
| `curation/bakeoff_jobs.py` (663) | `bakeoff_resolve.py`, `bakeoff_mapping.py` | |
| `routers/curation/_bakeoff_models.py` (663) | `src/schemas/bakeoff/{datasets,models,run,results,job_spec}.py` (done in PR4 L1) | OpenAPI identical |
| `curation/review_queries.py` (662) | `review_tabs.py`, `review_reasons.py`; `build_tab_query` split per tab | golden query bodies identical |
| `routers/curation/review.py` (660) | `routers/curation/review_terms.py` (+ `services/curation/review_terms.py`), `routers/curation/holdout_freeze.py` (freeze + stats; not `test_*`, pytest would collect it) | route order |
| `config/curation.py` (657) | `config/curation_roles.py`, `config/curation_accessors.py`; keep `CurationConfig` | last; scripted repoint |
| `routers/query.py` (653) | `schemas/query.py`, `routers/query_duplicates.py`, `services/image_admin.py` | |
| `projects/guard.py` (651) | `projects/guard_parse.py` (url/json/ndjson/bulk parsing), `projects/guard_policy.py` (tables, `check_request`); keep transport, exceptions, factories | move only; independent review |
| `curation/region_eval.py` (641) | `region_eval_geometry.py`, `region_eval_data.py`; keep `evaluate`, `run_eval` | |
| `services/clustering.py` (641) | `services/visual_search/clustering.py` (D33) + `clustering_models.py`, `clustering_persist.py`, `clustering_stats.py` | |
| `clients/triton_client.py` (641) | `clients/triton_preprocess.py`, `clients/triton_ocr.py`; class delegates | |
| `routers/curation_train/preflight.py` (634), `promote.py` (551) | L5: `services/training/preflight_run.py` (ordered check functions), `services/training/promote_run.py`, `schemas/train_promote.py` | same check ids, order, messages |
| `scripts/curation/vlm_worker.py` (700) | `scripts/curation/vlm_pending.py` (pending query/fetch), `scripts/curation/vlm_cli.py` (argparse); keep `label_batch`, `run`, `main` | ~8 tests patch `vlm_worker.*` |
| `scripts/datasets/fetch_coco_subset.py` (667) | `coco_selection.py`, `coco_fetch.py`; keep CLI | CLI path unchanged |
| `scripts/bench/baseline_suite.py` (665) | `suite_ingest.py`, `suite_phases.py`; keep parser/main | |

**PR 7, `export/ docker/ models/`, shell, tests**

| File (lines) | New modules / what moves | Notes |
|---|---|---|
| `export/export_models.py` (1569) | `yolo_trt_build.py`, `yolo_model_config.py`, `yolo_triton_config.py`, `yolo_formats.py`; keep constants, orchestration, summary, main | ultralytics patch preamble stays first |
| `export/export_pe_text_encoder.py` (888) + `export_pe_image_encoder.py` (802) | shared `pe_onnx_common.py` (`OnnxReport`, `preferred_batch_sizes`, `_is_dynamic`); `pe_text_triton_config.py`, `pe_text_validate.py`, `pe_text_parity.py`; `pe_image_triton_config.py`, `pe_image_validate.py` | rendered `config.pbtxt` byte-identical (`test_committed_model_configs`); update naming-leak allowlist paths |
| `export/export_detector_dual_head.py` (814) | `dual_head_model.py`, `dual_head_triton.py` | test loads by file path: add `export/` to `sys.path` |
| `export/export_scrfd.py` (738) | `scrfd_onnx.py` (rename `test_onnx_inference` to `check_onnx_inference`), `scrfd_triton.py` | |
| `docker/trainer/trainer.py` (1011) | `gpu_select.py`, `trainer_eval.py`, `trainer_kwargs.py`; keep run_job/watch/main | module name and `pgrep -f trainer.py` healthcheck stay; Dockerfile copies `*.py` |
| `docker/trainer/job_protocol.py` (804) | `job_spec.py`, `job_status.py`, `job_manifest.py`; keep pending/reject | repoint 4 tests and 4 trainer files |
| `models/ocr_pipeline/1/model.py` (704) | `models/ocr_pipeline/1/ocr_postprocess.py` (`DBPostProcess`, `CTCLabelDecode`) | precedent `models/gpu_image_prep/1/`; tests add the dir to `sys.path` |
| `openprocessor` (1097) | `scripts/lib/cli_core.sh`, `cli_gpu.sh`, `cli_dev.sh` (lazy-sourced) | add to `release-manifest.txt`; tests that `sed`-extract functions |
| `scripts/setup.sh` (967) | `scripts/lib/setup_prereq.sh`, `scripts/lib/setup_services.sh` | checkout-only |
| `scripts/lib/config.sh` (823) | `scripts/lib/config_models.sh` (per-model generators; split in two if > 450) | heredocs byte-identical; manifest |
| `scripts/release/acceptance_run.sh` (758) | `acceptance_lib.sh`, `acceptance_phases.sh` (split installer phases if > 450) | |
| `scripts/lib/export.sh` (702) | `scripts/lib/export_steps.sh` (per-model exports) | manifest |
| `scripts/export_paddleocr.sh` (702) | source `scripts/lib/{colors,export}.sh` instead of redefining (dedupe), `scripts/lib/paddleocr_config.sh` deduped against `config.sh` generators | test extracts `dc`/`check_container` by `sed` |
| `setup-openprocessor.sh` (3214) | **keep, cohesive**: one `__op_define` function so a truncated `curl | bash` download is a syntax error; pre-install code cannot source libs (trust model) | OC: optional later generation from fragments with a drift test |
| `tests/curation/test_cross_project_leak.py` (2516) | `tests/curation/cross_project_leak/` : `conftest.py` (`leak_env`), `_route_bodies.py`, `_seeders.py`, `_transport.py`, `test_every_scoped_route_sweep.py`, `test_registry_lifecycle_and_events.py`, `test_keymap_and_ingest_policy_isolation.py`, `test_clone_and_import_dir_isolation.py` | isolation suite: same test count |
| `tests/curation/test_train_router.py` (1865) | `tests/curation/train_router/` : conftest + `test_preflight_trainer_and_gpus`, `_disk_and_mount`, `_class_and_label_checks`, `_export_selection`, `test_start_and_campaign`, `test_status_runs_cancel_artifacts`, `test_promote_gate`, `test_promote_force_and_reload` | |
| `tests/curation/test_trainer_protocol.py` (1522) | `tests/curation/trainer_protocol/` : conftest + `test_job_and_status_file_contract`, `test_class_remap_and_subset_rewrite`, `test_gpu_order_and_capabilities`, `test_campaign_and_post_train_hooks`, `test_eval_summary_and_comparison`, `test_run_job_lifecycle` | |
| `tests/curation/test_bakeoff_router.py` (1380) | `tests/curation/bakeoff_router/` : conftest, `_helpers.py`, 5 test files by area (datasets/models, run enqueue, results/matrix, profiles/baselines, contract + evaluator e2e) | |
| `tests/curation/test_ingest_service.py` (1235) | `tests/curation/ingest_fakes.py` (fakes, `_make_service`; ~10 importers repointed); `test_ingest_batched_inference_and_events.py`, `test_ingest_region_status_seeding.py`, `test_ingest_backbone_embedding.py`; keep core | |
| `tests/test_compose_contract.py` (1178) | `tests/compose/` : `_helpers.py` + `test_service_profiles_and_healthchecks`, `test_ports_names_and_project_name`, `test_images_build_and_overlays`, `test_mounts_and_env_wiring`, `test_vlm_service_command` | |
| `tests/curation/test_region_worker.py` (1085) + `test_region_cascade_integrity.py` (793) | `tests/curation/region_cascade_helpers.py` (fakes, `_drive_worker`; 9+ importers); `test_region_worker_routing`, `_verdicts_and_provenance`, `_bulk_write`, `_shutdown_signals`; `test_vlm_combined_reply_parsing.py`; keep integrity core | |
| `tests/curation/test_bakeoff_harness.py` (1043) | `tests/curation/bakeoff_harness/` : conftest, `_helpers.py`, 6 test files by area | |
| `tests/curation/test_train_jobs.py` (1020) | `tests/curation/train_jobs/` : conftest + 5 files (write/lineage, gpu allowlist, read/status/campaign, mlflow urls, run dir/artifacts) | |
| `tests/curation/test_triton_promote.py` (939) | `tests/curation/triton_promote/` : conftest (autouse warm-up stub) + 4 files | |
| `tests/installer/test_setup_dry_run.py` (881) | `_setup_dry_run_helpers.py` + 4 files (plan/collisions, env/flags, bind/cropwright, release pinning) | |
| `tests/curation/test_worker_hot_reload.py` (803) | + `test_worker_runtime_build_and_swap.py`, `test_worker_bulk_write_stamps.py`; shared autouse reset in a conftest | |
| `tests/curation/test_export_service.py` (797) + `test_export_single_class.py` (748) | shared `tests/curation/export_fakes.py` (diff the two fakes first); `test_export_sampling`, `test_export_splits_and_boxes`, `test_export_manifest_and_dataset_sha`, `test_export_single_class_dataset`, `test_export_region_mode` | |
| `tests/curation/test_cascade_detect.py` (762) | `cascade_detect_helpers.py`; `test_region_detector_decoding`, `test_region_detector_inference_and_crop_mapping`, `test_paddle_ocr_text_recognizer` | |
| `tests/curation/test_segmenter_service.py` (745) | `tests/curation/segmenter_service/` : conftest, `_helpers.py`, 3 files | |
| `tests/test_validate_models.py` (736) | not a pytest module (no tests; live CLI): move to `scripts/validation/` as `validate_models.py`, `validate_models_lib.py`, `modality_checks.py` | update `tests/conftest.py` and docs references |

**PR 8, frontend** (repoint `vi.mock('$lib/api')` and the 73 source-scan tests; allow
`lib/api_errors.ts` in `errorText.scan.test.ts`; do `api.ts` and `types.ts` with a symbol-to-module
rewrite script in `artifacts_local/cleanup/`, then `npm run check` and the full vitest run)

| File (lines) | New files / what moves | Notes |
|---|---|---|
| `lib/api.ts` (5775) | keep HTTP core (prefix, scope, `apiFetch`, `qs`, ~450); new `api_errors.ts`, `api_projects.ts`, `api_settings.ts`, `api_viz.ts`, `api_regions.ts`, `api_regionBoxes.ts`, `api_models.ts`, `api_stats.ts`, `api_cropMap.ts` (leaf), `api_clusters.ts`, `api_cropRead.ts`, `api_cropWrites.ts` (split label/discard if > 450), `api_review.ts`, `api_select.ts`, `api_classes.ts`, `api_export.ts`, `api_train.ts`, `api_autoLabel.ts`, `api_ingest.ts`, `api_bakeoff.ts`, `api_datasetImport.ts`, `api_reprocess.ts`, `api_packs.ts`, `api_profiles.ts` | 280 importers; domain modules import core, never the reverse |
| `lib/api.test.ts` (2757) | `api.core`, `api.settings`, `api.review`, `api.cluster`, `api.clusters`, `api.viz`, `api.select`, `api.autoLabel` `.test.ts` | shared mock helper only if >= 3 users |
| `lib/types.ts` (1431) | keep core entities; `types_classes.ts`, `types_export.ts`, `types_review.ts`, `types_health.ts`, `types_ingest.ts`; model types into `types_models.ts` | 143 importers |
| `review/+page.svelte` (3262) | controllers in `lib/review/`: `reviewTabsController`, `reviewDiverseController`, `reviewFilterController`, `reviewQueueLoader`, `reviewClassPicker`, `reviewDismissedController`, `reviewSlotEditController`, `reviewSlotActions`, `reviewKeyboard` (`.svelte.ts`); components in `lib/components/review/`: `ReviewTabsBar`, `ReviewStrategyRow`, `ReviewFilterBar`, `ReviewEmptyQueue`, `ReviewCropPane`, `ReviewItemMeta`, `ReviewSlotPanel`, `ReviewClassButtons`, `ReviewContextPanel`, `ReviewStatusBar`, `DismissedPanel`, `RejectReasonPrompt`, `ClassPickerModal`; page ~300 | shared `queue`/`cursor`/`handledIds` by reference; ordered effects stay in one controller |
| `train/+page.svelte` (1642) + `components/TrainForm.svelte` (708) | `lib/train/`: `trainDatasetController`, `trainSingleClassExport`, `trainRunsController`, `trainActionsController`, `trainCohortsController` (`.svelte.ts`), `trainDisplay.ts`, `trainSpec.ts`; `lib/components/train/`: `TrainDatasetSection`, `PastRunsTable`, `TrainingCohortsPanel`, `TrainHyperparams`, `TrainPreflightPanel` | poll/abort disposed on unmount |
| `lib/annotations/config/parseSlotConfig.ts` (1530) + its test (717) | `parseCommon.ts`, `parseSlotAppearance.ts`, `parseSlotRegion.ts`, `parseSlotLifecycle.ts`, `parseSlotQueue.ts`, `parseSlotCohorts.ts`; keep the two orchestrators; tests split along the same lines | pure functions |
| `clusters/+page.svelte` (1488) | `lib/clusters/clusterListController.svelte.ts`, `clusterSearchModes.svelte.ts`, `slotInventory.svelte.ts`; `lib/components/clusters/ClustersToolbar`, `ClustersGrid`, `ClusterGridCard`, `ClustersStatusBar` | |
| `clusters/[id]/+page.svelte` (1473) | `lib/clusters/clusterDetailController`, `clusterDetailActions`, `clusterDetailLive` (`.svelte.ts`); components `ClusterDetailToolbar`, `ClusterDetailFilters`, `SubClusterTabs`, `ClusterDetailGrid`, `MoveClusterPicker` | keep move-picker state together (race test) |
| `clusters/slotGalleryController.svelte.ts` (868) | move to `lib/slots/`: `slotGalleryController.svelte.ts`, `slotGalleryJobs.ts`, `slotGalleryWrites.ts` | removes a lib -> routes import |
| `classes/+page.svelte` (1072) | `lib/classes/newClassProposalsController.svelte.ts`, `mergeClassesController.svelte.ts`; components `ClassRegistryTable`, `DeprecatedClassesTable`, `NewClassProposalsPanel`, `MergeClassesModal` | |
| `export/+page.svelte` (1047) | `lib/export/exportRunController.svelte.ts`, `freezeHoldoutController.svelte.ts`; components `HoldoutStatusCard`, `ExportClassTable`, `ExportControls`, `ExportRunModal`, `FreezeHoldoutModal` | polling teardown test moves with it |
| `lib/clusters/clusterController.test.ts` (1343), `lib/strategies.test.ts` (1128) | by behavior: `clusterController.{assign,vlm,discardUndo,ignore,move}.test.ts` + `clusterControllerTestKit.ts`; `strategies.{parse,availability,axes}.test.ts` + fixtures | same `it` count before/after |
| `components/CropMetaPanel.svelte` (753), `settings/KeymapCard.svelte` (752) | `components/cropMeta/{CropMetaSlots,CropItemText,CropHistoryBlock,CropSourceContext}.svelte` + `lib/cropMeta/cropMetaLoader.svelte.ts`; `settings/keymap/{keymapCardController.svelte.ts,keymapModel.ts,KeymapGroupsSection,KeymapContextsSection,KeymapConfirmModals}` | |
| `frontend/e2e/conftest.py` (826) | `e2e/harness/{server,stub,page_log,assets,constants}.py` via `pytest_plugins`; 79 `from conftest import` sites repointed | run the e2e suite |
| `frontend/scripts/capture_docs_screenshots.py` (1008) | `frontend/scripts/docs_screenshots/{guard,ctx,states_core,states_v040,states_ingest,registry}.py`; keep `main` | rename `states_v040` by subject (e.g. `states_slots`); diff the `STATES` listing before/after |

**Renames done in PR 2 (C5, C6).** `tests/curation/test_r3_activation_probes.py`,
`test_r4_probes.py`, `test_r5_probes.py` (split in five), `test_r6_probes.py`, `test_r7_fixes.py`,
`test_w34_round2_activation_regression.py` -> new dir `tests/curation/activation/` (conftest with
`app_client`, `ready_state`, `registry`, `_reset_caches`; constants module) as
`test_activation_pinned_body_and_stamp.py`, `test_settings_gate_and_reactivation.py`,
`test_settings_two_axis_pairing_gate.py`, `test_activation_rollback.py`,
`test_settings_after_deactivate.py`, `test_start_job_resolves_requested_activation.py`,
`test_every_activation_writer_rejects_gate_failing_pair.py`,
`test_pack_profile_pairing_and_rollback_deleted.py`,
`test_start_job_omitted_pack_and_ivf_path.py`; `test_w34_regression_probes.py` ->
`tests/curation/test_config_name_rules_and_template_activation.py`. Projects:
`test_r3_clone_probes.py`, `test_r4_clone_probes.py`, `test_w34_clone_axes_probe.py`,
`test_w34_round2_clone_regression.py` -> `tests/projects/test_clone_settings_real_flow.py`;
`test_r5_clone_probes.py` -> merge into `test_clone_activations.py`; `test_r6_clone_cause.py` ->
`test_clone_shared_model_ownership.py` (delete if it only reproduces without asserting);
`test_r7_clone_pack_mirror.py` -> `test_clone_pack_validation_for_env_default_pack.py`. Bench (D20):
`scripts/bench/wave0_{lib,probes,report,suite}.py` -> `ingest_sweep_*.py`, tests
`tests/bench/test_ingest_sweep_{lib,report}.py` (update `baseline_suite.py`,
`tests/test_env_surface.py`, `tests/test_triton_instance_budget.py`, `docker/evaluator/Dockerfile`,
docs). Frontend (`frontend/src/routes/p/[project]/`): `train/interactivePassFixes` ->
`cohortPreviewInline`; `train/logicMovesW6TrainingCohorts` -> `servedTrainingCohorts`;
`export/dqM9ExportGate` -> `exportEmptyErrorDetail`; `export/dqM9CountLabels` ->
`exportTotalColumnLabel`; `classes/dqM9CountLabels` -> `classesTotalColumnLabel`;
`dashboard/dqM11ClassBalanceSlice` + legend part of `dashboard/interactivePassFixes` ->
`classBalanceChart`; rest of it -> `dashboardYoloExportConfirm`; `dashboard/logicMovesW3` ->
`dashboardVlmClusterLabelJob`; `clusters/[id]/logicMovesW3W6` -> `vlmClusterLabelJob` +
`clusterDetailServedFields`; `clusters/[id]/dqM3CutLineVisibility` -> `clusterCutLineVisibility`;
`clusters/dqM2PurityDisplay` + `clusters/logicMovesW6` -> `clusterPurityDisplay`;
`clusters/dqM4DisplayOrderRepresentatives` -> `clusterRepresentativesWindow`;
`review/adoption3f1a11e` -> `reviewServedEnumFilters` + `reviewRejectionDisplay`;
`review/dqM8DetectorScoreLabel` -> `reviewDetectorScoreLabel`; `review/dqM7Locate` + deep-link
part of `review/logicMovesW5` -> `reviewDeepLinkLocate`; rest of `logicMovesW5` ->
`reviewAcceptModelClass`; `review/logicMovesW1` -> `reviewDismissedPanel`; `review/logicMovesW2`
-> `reviewServerRenderedSlotWrites`; `review/dqM6RejectReasonPrompt` -> `reviewRejectReasonPrompt`;
`review/m6RegionUndo` -> `reviewRegionUndo`; `review/interactivePassFixes` -> its parts into
`reviewDismissedPanel`, `reviewImageSizing`, `reviewConfirmProvenance`,
`reviewNameOnlyProposalStyle`, `reviewRejectReasonPrompt`; `clusters/interactivePass2026_09_24` ->
by subject. e2e: `test_export_freeze_split_counts_df01309.py` ->
`test_export_split_counts_and_holdout_freeze.py`; `test_review_region_status_filter_3f1a11e.py` ->
`test_review_region_status_filter.py`. Also rename test functions and classes that carry ids
(e.g. `test_the_defaults_are_exactly_the_pre_w9_command`, `TestR7*`).

## 12. Outline: `docs/design/engineering_standards.md` (written last, PR 9)

Portable standards and lessons for a sister Python/FastAPI + Svelte repo (OpenTranscribe-style).
The sister repo is not touched by this plan.

1. Principles: no dead code or shims; fail closed; class identity by name; review depth by risk;
   split by concern, never trim to fit.
2. Repo layout and layering: routers (HTTP only) > services (no fastapi) > clients > domain /
   config / schemas / utils; the contract table and the one allowed exception pattern.
3. Config snippets: `pyproject.toml` ruff (select, justified ignores, pydantic
   `runtime-evaluated-base-classes`, `TID251` env ban), mypy (project-interpreter hook, pydantic
   plugin, `warn_unused_ignores`, strictness ladder), bandit scope, pytest (markers, timeout,
   coverage in CI only).
4. Hooks: the final `.pre-commit-config.yaml` with one line per hook on what it catches.
5. Guards and ratchets: size, complexity, broad-except, layering, process ids, vulture,
   naming-leak, version, changelog, config-path existence; the baseline file format and stale
   rule; how to read a failure.
6. Ordered port checklist for the sister repo: measure (vulture, radon/ruff C901, mypy with real
   deps, bandit, knip), fix config, add guards with baselines, delete dead code, fix fail-open,
   layering, duplicates, splits, typing ladder.
7. Pitfalls seen here: mypy in an isolated pre-commit env makes third-party types `Any`
   (53 errors invisible); exclude lists that silently drop production code (`scripts/`,
   `export/`); directory exemptions that disable zero-tolerance scans; stale config naming
   deleted paths; trim-to-fit at a size ceiling; aliases kept "for older clients" in a monorepo;
   coverage floors that CI never runs; regexes that miss sentence-end versions.
8. Before/after numbers table (from PR descriptions) and per-guard proof outputs.
9. Frontend: knip, scan tests, shared Modal/format/SSE/poll modules, size gate shared with backend.

## 13. Issues to file

None for the review findings: the owner wants every fix in this series. File an issue only for
an OC decision the owner declines (title `chore(cleanup): <decision> deferred`, body: the
decision row, evidence, why deferred) and for anything new an implementer finds that is outside
the review (body: evidence, proposed fix, acceptance). This series closes or advances: #155
(PR5 F2), #161 (PR3 E8), #289 (PR1 A4 header, PR5 F3 frontend compose), #287 (PR1 A8 regex and
roots part; its remaining items stay open).
