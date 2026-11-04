# 08: environment, setup, configuration and CI parity (cross-cutting)

Status: plan, not started. Nothing in this file has been implemented.
Depends on: [00_index.md](00_index.md) (decision register, batch order) and the batch files:
B0 [01_b0_close_out.md](01_b0_close_out.md); B1 tracks 1A B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)),
1B B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)), 1C B1 1C (merged; see [CHANGELOG](../../../CHANGELOG.md)),
1D B1 1D (merged; see [CHANGELOG](../../../CHANGELOG.md)), 1E B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md));
B2 [03_b2_overview.md](03_b2_overview.md) with W1 [03a_b2_w1_queue_infra.md](03a_b2_w1_queue_infra.md),
W2 [03b_b2_w2_job_families.md](03b_b2_w2_job_families.md), W3 [03c_b2_w3_beat_reaper_workers.md](03c_b2_w3_beat_reaper_workers.md);
B3 [04_b3_ultralytics_one_version.md](04_b3_ultralytics_one_version.md); B4
[05_b4_stateless_api.md](05_b4_stateless_api.md); R-final [06_release_acceptance.md](06_release_acceptance.md);
B5 [07_b5_overview.md](07_b5_overview.md); [09_github_actions.md](09_github_actions.md),
[11_risks_cleanup_host.md](11_risks_cleanup_host.md). Each finding's batch below names the track
(for example "B1-1D" is the 02d file).
Companion: [10_standards_and_libraries.md](10_standards_and_libraries.md).
Source IDs: Q-Phase 2, Q-Phase 4, Q-Phase 5, Q-Phase 7, Q-J3, Q-J9, Q-J14 (`task_queue_plan.md`);
CC-1, BA-3, GPU-16, OS-7, T-3, T-4, T-8, OPS-1 (`codebase_audit_plan.md`); U-Step 3, U-Step 7,
U-J8 (`model_family_versions_plan.md`); the decisions it applies are in the register at
[00_index.md](00_index.md) section 6 (D-2, D-9, D-18..D-26, D-30, D-33 came from the former
master plan, rev 4, in git history).

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03. The branch head moved to
`d1c6bab0` (the B0 Ultralytics pin) while this file was written; rows that the pin commit
changed say so.
Consolidated 2026-11-03. Updated 2026-11-04 for the PostgreSQL control plane (owner decisions of
2026-11-04, [02_control_plane_database.md](02_control_plane_database.md)): the B2 rows of sections
2.2, 4, 4.1, 5 and 7.3 are rewritten; package versions live only in
`13_package_pins.md` (written on its own branch).

**Owner decisions that override the former master plan here (2026-11-03).** No intermediate
releases: B0..B5 land on `main`, then one release acceptance (R-final, last) and v0.7.0 (D-37).
The only supported upgrade is **0.6.0 -> 0.7.0, directly**. Every "v0.8.0", "R-B1", "R-0.9" and
"upgrade from 0.7.0 or 0.8.0" in the former master plan is replaced by that one path (section 5).
The test and GPU runs this file mentions follow the [Test and GPU budget](00_index.md): three
live-harness runs and three GPU windows, nothing per branch or per commit.

What to keep after merge: the per-batch config delta (section 4) becomes `env.template`,
`docs-site/docs/operations/` configuration pages and the CHANGELOG entries as each batch lands;
the CI parity table (section 6) moves to `CONTRIBUTING.md`; the exact-pin table (section 7.1) is
enforced by `tests/test_requirements_contract.py`. The files go as described in
[00_index.md](00_index.md) "Plan lifecycle".

## 0. How to use this file

- Each finding has an ID (`ENV-n`), the evidence (file and symbol, or the `rg` pattern that finds
  it), the fix, the documented pattern or library with its URL, the target batch, a size and the
  test that proves it. Re-find code by symbol; line numbers move.
- A finding that an existing task already covers names that task instead of adding work.
- Public-tree rules: no machine paths, no secrets, no private names; security items are neutral
  one-liners.
- Read-only audit. No test, build, container or script was run for this file. "Argued" marks a
  claim from reading code, not from a run; its red test is the first step of the fix.
- This file names planned `OP_*` variables that no code reads yet (for example the B2 queue
  settings). `scripts/docs/check_docs_vs_code.py::check_env` scans every tracked markdown file,
  `docs/design/` included, so run `.venv/bin/python scripts/docs/check_docs_vs_code.py --only docs/design/program/08_environment_and_setup.md`
  before committing this file and resolve any hit the way the committed source plans were
  resolved.

## 1. Summary

The repo already guards most of its config surface: `tests/test_env_surface.py` checks every
`OP_*` read in `src scripts docker export` against `env.template` in both directions,
`scripts/docs/check_docs_vs_code.py` checks every compose `${VAR}` and every template setting,
`src/config/retired_env.py` and `scripts/lib/retired_env.sh` refuse retired names, and
`tests/test_version_consistency.py` ties the version strings together. The gaps are at the edges
of those guards:

| Rank | Gap | ID | Batch |
|---|---|---|---|
| 1 | The checkout setup (`scripts/setup.sh`) writes Triton configs with the chosen host GPU index while compose maps only `TRITON_GPU_ID` (default 0) into the container (argued) | ENV-3 | B1 |
| 2 | Two `.env` generators that disagree (installer copies `env.template`, mode 600, merges new keys on upgrade; checkout setup writes a 20-line heredoc) | ENV-4 | B2 stage 1 |
| 3 | `Settings` (pydantic-settings, no prefix) reads 18 variables outside `src/config/env.py`; 9 of its fields are read by nothing | ENV-1 | B1 |
| 4 | The retired-name guard is not called by 4 process entry points | ENV-6 | B1-1E, B2 stage 1 |
| 5 | Two names for one setting (Triton URL, OpenSearch hosts) | ENV-2 | B1-1C, B2 |
| 6 | Config conflicts between the source plans (the control-plane secret shape, `API_WORKERS` default, Flower port, retired event vars) | ENV-14..ENV-18 | B2, B4 |
| 7 | Dependabot rules let an interpreter move, a Triton/TensorRT move and an onnxruntime cap widening through | ENV-12 | B1 deps lane |

## 2. Developer setup, end to end

### 2.1 Steps as documented today, checked against the code

| Step | Documented in | Command (as written there) | Code agrees? |
|---|---|---|---|
| venv | CONTRIBUTING "Dev setup" | `python3 -m venv .venv`; `.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision`; `.venv/bin/pip install -r requirements-test.txt pre-commit pytest-xdist` | yes. CI installs the same (`.github/workflows/ci.yml` job `pytest`), without `pytest-xdist` |
| never install `requirements.txt` in the venv | CONTRIBUTING, CLAUDE.md | (rule) | yes: `faiss-gpu-cu12` in `requirements.txt` breaks `faiss-cpu` |
| hooks | CONTRIBUTING | `.venv/bin/pre-commit install` | yes: `default_install_hook_types: [pre-commit, commit-msg]` in `.pre-commit-config.yaml`. There are no pre-push hooks |
| worktree lanes | `docs/design/engineering_standards.md` 4.4 step 2 only | `git worktree add ...`; `ln -s <main>/.venv <worktree>/.venv`; one `npm ci` per worktree | yes (this worktree's `.venv` is such a symlink). Not in CONTRIBUTING (ENV-9) |
| Node for `frontend/` | `frontend/.nvmrc` (26), CI `actions/setup-node` with `node-version-file` | `cd frontend && npm ci` | yes. One `npm ci` per worktree: a shared `node_modules` symlink breaks `svelte-kit sync` (engineering_standards 5.12) |
| GPU placement | CLAUDE.md "Services and compose" | `TRITON_GPU_ID`, `API_GPU_ID`, `SEGMENTER_GPU_ID`, `EVALUATOR_GPU_ID`, `VLM_GPU_ID` | partly: compose also reads `OP_TRAIN_GPU_ORDER` (trainer) and `OP_CLUSTER_GPU_DEVICE` (`docker-compose.gpu-clustering.yml`); `GPU_ID` is a fourth, legacy name (ENV-3) |
| `-p` per worktree | CLAUDE.md, CONTRIBUTING | `docker compose -p <name> ...` | the Makefile `COMPOSE` variable passes no `-p`; it relies on `COMPOSE_PROJECT_NAME` from `.env` (ENV-9) |
| live harness | CONTRIBUTING, `docker/test/README.md` | `docker compose -p op-live-verify -f docker/test/compose.yml up -d --wait`; `.venv/bin/python -m pytest tests/live -q --no-cov -m live`; `... down -v --remove-orphans` | yes. Harness images are local (`${VERIFY_CONTAINER_PREFIX:-op-verify}-api:latest`, built from the checkout). Run only the three budgeted times ([Test and GPU budget](00_index.md)) |
| local-only image tags | the former master plan 5.3, memory of the image-tag clobber | `build_and_publish.sh --dry-run --version <v> --local-tag-suffix -rcN` | yes; the compose defaults use `${OP_IMAGE_TAG:-0.6.0}`, so a local build must set `OP_IMAGE_REPO`/`OP_IMAGE_TAG` (`env.template` "Image repo / tag") |
| restart after code or `.env` change | CONTRIBUTING "After pulling or merging code" | `make dev-restart` (code), `make dev-up` (`.env`) | yes |

### 2.2 The core flows a fresh setup must reach

The acceptance phases in `scripts/release/acceptance_phases.sh` are the executable form of "works
end to end". R-final ([06_release_acceptance.md](06_release_acceptance.md)) owns them. This file only
adds the configuration each flow needs, so a fresh agent can tell a config gap from a code bug:

| Flow | Services (profile) | Config that must be present |
|---|---|---|
| Inference routes (`/detect`, `/faces`, `/embed`, `/search`, `/ingest`, `/ocr`, `/analyze`) | `triton-server`, `api`, `opensearch` | engines built (`openprocessor models install`), `OP_REQUIRED_MODELS` written by the installer, `TRITON_GPU_ID`/`API_GPU_ID` |
| Curation project, ingest, regions | + `curation-detection-worker` (`curation`), `segmenter` (`segmenter`) for segmenter profiles | `OP_SOURCE_ROOT_HOST`, a region profile (`OP_REGION_PROFILE_PATH` or a stored profile), the segmenter's HF token in `.env` (never printed), `SEGMENTER_GPU_ID` |
| VLM labeling, review | + `curation-vlm-worker`, `vlm` (`vlm`) or a remote endpoint | `openprocessor vlm use <id>` (writes `VLM_CATALOG_ID` and the endpoint), `VLM_GPU_ID`, `secrets/vlm/` for remote keys |
| Import, combine, export | `jobs-worker` (from B2) | the storage roots mounted; Postgres and Valkey up |
| Train, promote | + `curation-trainer`, `curation-mlflow` (`training`) | `OP_TRAIN_GPU_ORDER` (must equal the trainer's `device_ids`), shared runs path |
| Bake-off | `curation-evaluator` (`evaluator`) | `EVALUATOR_GPU_ID` |
| Model upload and export | `model-export-worker` (`export`), a `gpu_export` consumer from B2 | the models root; Postgres and Valkey up |
| Monitoring | `monitoring` profile | ports 4604-4606, 4608, 4610; `monitoring/prometheus.yml` scrape targets |
| After B2 | + `postgres`, `valkey`, `db-migrate` (one-shot), `jobs-worker`, `periodic-worker`, `open-vocab-worker`, `jobs-beat` (core), Flower (`monitoring`) | the secrets of ENV-14 (Postgres owner and app passwords, Valkey password and config, Flower auth), the `OP_DB_*` and `OP_QUEUE_*` keys |

## 3. Environment and config audit (read-only)

### 3.1 Method

1. Code readers: the reader regex of `tests/test_env_surface.py` (`_DIRECT_ENV_RE`), widened from
   `OP_*` to any upper-case name and to `os.environ[...]`, run with `rg -o ... -r '$1' src scripts docker export --type py | sort -u`
   (200 names), plus the `Settings` fields of `src/config/settings.py` (pydantic-settings maps
   each field name to an upper-case variable) and the `CurationConfig.from_env` keys (already in
   `tests/test_env_surface.py`).
2. Template: `rg -o "^\s*#?\s*([A-Z][A-Z0-9_]+)=" -r '$1' env.template | sort -u` (308 names;
   20 active lines, the rest commented with their default).
3. Compose: `rg -o '\$\{([A-Z][A-Z0-9_]*)' -r '$1' docker-compose*.yml | sort -u` (96 names).
4. `comm` between the three lists; every hit read by hand.

What the existing guards already prove, so it is not repeated here: every `OP_*` read is in the
template and vice versa (`tests/test_env_surface.py`); every compose `${VAR}` has a template
assignment line and every template setting is read by something (`check_docs_vs_code.py`, run by
`tests/test_docs_vs_code.py`). The compose-only names (`OP_IMAGE_TAG`, `OP_BIND_ADDRESS`,
`OP_CROPWRIGHT_*` ...) are allowlisted there with a reason. The unprefixed `CROPWRIGHT_*`
pass-throughs in the `cropwright` service exist only so its entrypoint can refuse them (retired
names, `scripts/lib/retired_env.sh`); that is correct.

### 3.2 Findings

| ID | Finding (evidence verified now) | Fix | Doc / pattern | Batch | Size | Proof |
|---|---|---|---|---|---|---|
| ENV-1 | `src/config/settings.py::Settings` is a `pydantic_settings.BaseSettings` with `env_prefix=''`, so 18 fields read the environment outside `src/config/env.py` (the "one place" its docstring and CONTRIBUTING's `TID251` row promise; ruff's banned-api only sees `os.environ`). Nine fields are read by nothing: `triton_grpc_timeout`, `opensearch_index`, `default_max_resize`, `min_model_size`, `embedding_cache_size`, `embedding_cache_ttl`, `affine_cache_size`, `api_title`, `api_description` (`rg -n "\.<field>\b" src scripts export` is empty for each), so for example `TRITON_GRPC_TIMEOUT=5` silently does nothing. `api_version` can be overridden by `API_VERSION`, which would change the version `/health` reports. The live fields `LOG_LEVEL`, `JSON_LOGS`, `OPENSEARCH_TIMEOUT`, `SLOW_REQUEST_THRESHOLD_MS` are in neither `env.template` nor the docs. Vulture misses the dead fields because class-body fields are dropped from its report (engineering_standards 3.2) | Delete the nine dead fields. Make `api_version` a plain property over `_read_version()` (not a field). Document the live fields in the template's advanced tier. Extend `tests/test_env_surface.py` with a third direction: every `Settings.model_fields` name is in `env.template` (upper-cased) or allowlisted with a reason. **GPU-16 interplay:** GPU-16 adds "one settings-driven `client_timeout`"; it should be this one setting (revive `triton_grpc_timeout` with a reader, or delete it and add an `env.py` reader), never two | pydantic-settings field-to-variable mapping: https://docs.pydantic.dev/latest/concepts/pydantic_settings/ | B1-1D (GPU-16 in B1-1C uses the same name) | S | red: the new `test_env_surface` direction fails today on the dead fields and on `LOG_LEVEL`; control: an `OP_*` var still passes |
| ENV-2 | Two names for one setting. (a) Triton URL: `src/services/curation/autolabel/embed_stage.py::triton_url` reads `OP_TRITON_URL`, else `TRITON_URL`; everything else reads `TRITON_URL` (`Settings.triton_url`, `scripts/curation/worker/state.py`). (b) OpenSearch: `Settings.opensearch_url` and the scripts read `OPENSEARCH_URL` (set in compose); `src/clients/opensearch/client.py` falls back to `OPENSEARCH_HOSTS` and then `http://localhost:9200` when no hosts are passed, and `OPENSEARCH_HOSTS` is also the Dashboards container's own variable (`docker/test/compose.yml` comment says so) | (a) Read only `TRITON_URL`; add `'OP_TRITON_URL': 'TRITON_URL'` to `RETIRED_ENV`; drop its template line. (b) The client default comes from the caller (`settings.opensearch_url`); delete the `OPENSEARCH_HOSTS` read so one name configures the client. No alias is read (no-shim rule) | `src/config/retired_env.py` pattern (engineering_standards 1.5) | (a) B2-2E (the embed stage moves with auto-label, so it is edited once); (b) B1-1C | S | (a) `tests/config/` retired-env test: `OP_TRITON_URL` set refuses startup naming `TRITON_URL`; (b) a client built with no hosts uses `settings.opensearch_url` (monkeypatched), and `OPENSEARCH_HOSTS` has no effect |
| ENV-3 | Checkout GPU choice (argued). `scripts/setup.sh --gpu=N` exports `GPU_ID=N` (`scripts/lib/setup_prereq.sh`), passes it to `generate_all_configs` (`scripts/lib/config.sh`), whose awk rewrites `gpus: [ N ]` in each model config, and writes `GPU_ID=N` into `.env`. Compose never reads `GPU_ID`; it maps `device_ids: ['${TRITON_GPU_ID:-0}']`, and inside the container the one visible card is index 0. So `--gpu=1` yields configs asking for a device the container lacks, or (with `TRITON_GPU_ID` unset) Triton on card 0. The installer path is right: `scripts/lib/model_setup.sh` calls `generate_all_configs "$1" 0`. `env.template` still ships an active `GPU_ID=0` ("Which GPU to use") | `scripts/setup.sh`: write `TRITON_GPU_ID=N` and `API_GPU_ID=N` and generate configs with GPU 0 (container-relative), as the installer does. Keep `GPU_ID` only as the host-side index for the `nvidia-smi` checks in `scripts/lib/gpu.sh`, renamed to a local variable; drop the active `GPU_ID=0` line from `env.template`; add `GPU_ID` to `scripts/lib/retired_env.sh` (bash-only name) | NVIDIA Container Toolkit: the container sees only the listed devices, renumbered from 0 (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html); compose GPU `device_ids` (https://docs.docker.com/compose/how-tos/gpu-support/) | B1-1D | S | red: a `tests/installer/` test runs `generate_env_file` and `generate_all_configs` with GPU 1 in a temp dir and asserts every `gpus:` is `[ 0 ]` and `.env` has `TRITON_GPU_ID=1` (fails today); control: GPU 0 unchanged |
| ENV-4 | Two `.env` generators. The one-line installer (`setup-openprocessor.sh::env_create_or_merge`) copies `env.template` with mode 600 and, on a re-run or upgrade, adds every **active** (uncommented) template key the `.env` lacks. The checkout setup (`scripts/lib/config.sh::generate_env_file`) writes a 20-line heredoc with `cat >` (default umask, so not 600), no GPU placement keys and no merge; with an existing `.env` it does nothing without `--force` | One implementation in `scripts/lib/` used by both: copy the template (`umask 077`, `install -m 600`), then upsert the computed keys (profile, heap, GPU decode, placement). Both B2 (the secrets, ENV-14) and B4 (`API_WORKERS`, ENV-15) must then write through it, so do it as the first commit of B2 stage 1 (lane 1C) | the installer's own `env_create_or_merge` (in-repo pattern) | B2 stage 1 | S-M | red: a `tests/installer/` test runs the checkout generator and asserts mode 600 and that every active template key is present (fails today); control: an existing value is kept |
| ENV-5 | Non-`OP_` names read by code and not documented (the env-surface test checks only `OP_*`): `OCC_BULK_PAGE_SIZE` (`src/clients/occ.py`, `env_int`), `GPU_IMAGE_PREP_MODEL` (`TritonModelConfig`; missing from the template's "model-name overrides" block, where its nine siblings are), `TRITON_METRICS_URL` (`src/routers/curation/models.py`), plus the ENV-1 `Settings` names | Rename `OCC_BULK_PAGE_SIZE` to `OP_OCC_BULK_PAGE_SIZE` (retired-env entry for the old name); document the rest in the advanced tier; the ENV-1 test direction covers `Settings`; add an explicit allowlist of non-`OP_` names that `env_*` readers may use (model names, `TRITON_URL`, `OPENSEARCH_URL`, `HF_*`, `PROMETHEUS_MULTIPROC_DIR`, `CUDA_*`, the harness `FAKE_*`) so a new unprefixed read fails | existing `tests/test_env_surface.py` pattern | B1-1D | S | red: extended test lists the three names today; control: `OP_*` unchanged |
| ENV-6 | `reject_retired_env()` is called by `src/app/lifespan.py`, `scripts/curation/worker/__main__.py`, `scripts/curation/vlm_worker.py` and `bakeoff_runner.py` only. `scripts/curation/auto_label_worker.py`, `scripts/curation/assist_loop_worker.py`, `scripts/curation/cluster_refresh_daemon.py` and `export/export_worker.py` do not call it (`rg -ln "reject_retired_env\(\)" src scripts docker export`). Two of them build VLM labelers, where a stale `VLM_URL` would silently do nothing. Mitigated today because every service loads the same `.env` and the API refuses first, but a worker started alone runs on | B2 deletes the first three entry points: call `reject_retired_env()` from the Celery app's `worker_init` and `beat_init` signal handlers in `src/tasks/app.py` (no I/O, so the "no I/O in `worker_process_init`" rule is not touched). From B2 the export worker, the trainer and the evaluator are Celery consumers too, so the same `worker_init` handler (in `src/tasks/runtime/gpu_app.py`) covers them | Celery signals `worker_init`, `beat_init`: https://docs.celeryq.dev/en/stable/userguide/signals.html | B2 stage 1 (Celery), stage 5 (GPU consumers) | S | red: `tests/test_retired_env_entrypoints.py` parses every compose service `command` that runs Python from this repo, resolves the entry module and asserts (AST) that it calls `reject_retired_env` or is the Celery app; fails today for the export worker |
| ENV-7 | Version strings outside `tests/test_version_consistency.py`: `frontend/package.json` (and its lock) `"version": "0.6.0"`, and the CLI fallback `${OP_CLI_VERSION:-0.6.0}` in `openprocessor`. The test covers VERSION, `pyproject.toml`, `docs-site/package.json`, every compose image default and `SCRIPT_VERSION` | Add both to the test; bump them in the R-final release-prep commit with the rest | existing test | R-final prep (test now, in B1-1D) | XS | red: the extended test passes today only because all say 0.6.0; plant a mismatch and see it fail |
| ENV-8 | Tiering. `env.template` is 1,154 lines: 20 active keys and about 290 commented ones. Tiers are marked by habit (an "Advanced:" comment prefix on some keys; a "Curation quick-config -- START HERE" block at line 241, after the ports). Nothing checks a tier. The docs site already splits `docs-site/docs/configuration/basic.mdx` and `advanced.mdx` | Restructure once, after the last batch that adds keys: a `BASIC` section first (what an installer or a user sets: profile, GPU placement, ports, heap, bind address, VLM choice, `API_WORKERS`, the queue secret pointer) and an `ADVANCED` section (tuning, model-name overrides, every `OP_*` with a default). A test asserts every assignment line sits under exactly one tier header and that every key the installer writes is `BASIC`. Keep the docs-site pages in step | owner rule ".env settings tiered (basic vs advanced)" | B4 (last config batch; before R-final) | M | the tier test, seen failing on today's template |
| ENV-9 | Docs vs code for dev setup: (a) CLAUDE.md's services table lacks `curation-assist-loop-worker` (profile `curation`) and `cropwright` (profile `cropwright`); (b) its GPU list lacks `OP_TRAIN_GPU_ORDER` and `OP_CLUSTER_GPU_DEVICE`; (c) `make dev-up` (`DEV_DEV_COMPOSE` in the Makefile) always adds `docker-compose.gpu-arbiter.yml`, which CLAUDE.md calls opt-in: say that `dev-up` opts in; (d) the worktree `.venv` symlink and per-worktree `npm ci` are only in engineering_standards 4.4: add them to CONTRIBUTING; (e) the Makefile passes no `-p`: document "set `COMPOSE_PROJECT_NAME` in a worktree's `.env` or call `docker compose -p` directly", or add `-p $(P)` with `P ?=` defaulting to `COMPOSE_PROJECT_NAME`; (f) CONTRIBUTING's intro and `.github/PULL_REQUEST_TEMPLATE.md` name a region-field ratchet (`check_no_literal_region_fields.py`, `PORTED_PATHS`) that does not exist (`ls scripts/codegen`); delete both mentions (see STD-7) | doc edits; the Makefile change is optional | `docker compose -p`: https://docs.docker.com/reference/cli/docker/compose/ | B1-1D (docs commit) | S | `check_docs_vs_code.py` and `tests/test_doc_links.py` green; a reader check of the services table against `docker compose config --services --profile '*'` |
| ENV-10 | Interpreter drift: CI, mypy (`python_version = "3.12"`), ruff (`py312`) and host venvs run 3.12; the `api`, `export` and `evaluator` images run 3.13 (`Dockerfile` `FROM python:3.13-slim-trixie@sha256:...`); the trainer image 3.12. The offline suite never runs on the shipped interpreter (the live harness does, because it builds the `api` image) | Owner decision (question Q-1): either one CI `pytest` job on 3.13 (`actions/setup-python` matrix) or keep 3.12 CI and rely on the live harness, stated in CONTRIBUTING. Interpreter bumps stay manual (ENV-12) | `actions/setup-python` matrix: https://github.com/actions/setup-python | B1 (with [09_github_actions.md](09_github_actions.md)) | S | the CI job runs green on 3.13 once |
| ENV-11 | Manifest drift beyond Ultralytics. The B0 pin commit (`d1c6bab0`) made `ultralytics==8.4.197` identical in `requirements.txt`, `requirements-test.txt` and `pyproject.toml` (`test_ultralytics_is_one_exact_version_everywhere`). Other exact pins still drift: `fastapi==0.143.3` in both requirements files but bare `fastapi` in `pyproject.toml` (`tests/test_dependency_manifest.py` compares names only). Pillow is imported directly (`ImageOps.exif_transpose` in `src/utils/image_processing.py`, `pe_preprocess.py`, `cascade_detect/preprocess.py`) but declared nowhere; the manifest test allowlists it as "transitive dependency of ultralytics" | Generalise the pin test: every `==` in `requirements.txt` has the same specifier in `pyproject.toml` and `requirements-test.txt` where the package appears. Declare `Pillow` in all three (floor, not exact), since the EXIF handling is behaviour the tests rely on; drop the allowlist entry | PEP 508 specifiers: https://packaging.python.org/en/latest/specifications/dependency-specifiers/ | B1-1D | XS | red: the generalised test fails on `fastapi` today |
| ENV-12 | Dependabot rules (`.github/dependabot.yml`). `ultralytics` is ignored for every file (right for an exact pin; its comment still describes the 8.4 range and the y11 venv, both gone after B0/B3). `onnxruntime-gpu` is not ignored, so PR #434 widens its cap to `<1.32` against the stated `nvidia-modelopt ~=1.24.2` constraint. The `docker` ecosystem has no rule for interpreter or Triton moves, so #422 and #432 propose Python 3.14/3.15 bases and Triton 26.09 (whose TensorRT must equal `tensorrt-cu13==11.1.0.106`, asserted in `Dockerfile.triton`) | Add `ignore` entries: `onnxruntime-gpu` `versions: [">=1.25"]` with the modelopt reason; for the `docker` ecosystem, `python` and `nvcr.io/nvidia/tritonserver` with `update-types: ["version-update:semver-major", "version-update:semver-minor"]` (digest-only updates still arrive for the same tag). Keep `ultralytics` ignored after B3 (D-28: any bump reruns U-6.1 and U-6.2); rewrite its comment | Dependabot `ignore` options: https://docs.github.com/en/code-security/dependabot/working-with-dependabot/dependabot-options-reference | B1 deps lane `b1/deps`, first commit (B1 (merged; see [CHANGELOG](../../../CHANGELOG.md)) section 4.1) | XS | `check-yaml` passes; the next weekly dependabot run opens no interpreter-tag, Triton-tag or onnxruntime-cap PR (check with `gh pr list --label dependencies`) |
| ENV-13 | Unpinned runtime packages. Most of `requirements.txt` is unpinned or floor-only (`torch`, `numpy`, `opencv-python`, `tritonclient[all]`, `httpx`, `opensearch-py>=2.3.0`), so two image builds on different days can differ. The shipped bits are fixed by the published image digest; the risk is an unreproducible rebuild | Cheap now: R-final records `pip freeze` (and `importlib.metadata` for the key packages) of every built image in the release evidence. A lock file (pip-tools `pip-compile --generate-hashes`, or `uv pip compile`) is an owner decision (Q-2), backlog | pip-tools: https://pip-tools.readthedocs.io/ ; pip `--require-hashes`: https://pip.pypa.io/en/stable/topics/secure-installs/ | R-final (freeze record); backlog (lock) | XS / M | the freeze files exist in the release bundle |

Checked and clean (do not redo): `os.environ` reads in `src` go through `env.py` except the
documented `worker_liveness.py` healthcheck (`# noqa: TID251` with a reason); the auto-label
worker's `OP_AUTO_LABEL_WORKER_METRICS_PORT=4611`, the region worker's `4609` and the VLM worker's
`4610` reuse numbers that are host ports elsewhere, but they are container-internal (not published),
so there is no clash; `OP_IVF_RETRAIN_CHECK_S` is read with a raw `float(os.getenv(...))` in a
script (scripts are exempt from `TID251`), it is documented, and B2 moves it to beat anyway.

## 4. Config delta per batch

Each row: what the batch adds, renames or removes; where it must be documented; which installer or
CLI step writes it; the tier. "Template" means `env.template` (commented with its default unless the
installer must write it, in which case active). Every new `OP_*` read gets its template line in
the same commit (the env-surface test enforces it). Every removed `OP_*` gets a `RETIRED_ENV`
entry (`removed: <what to do>`) in the same commit.

| Batch | Change | Kind | Documented in | Written by | Tier |
|---|---|---|---|---|---|
| B0 | `ultralytics==8.4.197` in three manifests (done, `d1c6bab0`); `uvicorn[standard]>=0.41.0` (done); `--limit-max-requests-jitter=1000` in the api command (done); `pyclipper==1.4.0` in the Triton images (done) | deps, compose | CHANGELOG, ATTRIBUTION (pyclipper, done) | image build | n/a |
| B1-1A | lease and class-registry objects added to the configs and settings mappings (additive put at bootstrap); `LEASE_TTL_S=120`, `HEARTBEAT_TICK_S=15` as **module constants, not env** (do not add knobs); `_JOB_STALE_S` deleted; new baseline `scripts/codegen/baselines/file_locks.txt` | mapping, constants, ratchet | `docs/opensearch_schema_design.md`, `docs/CURATION.md` (busy window) | API bootstrap | n/a |
| B1-1B | `scripts/curation/migrate_class_registry.py`; CLI `./openprocessor curation migrate-registry [--project <slug>\|--all] [--dry-run]` (`openprocessor`, `scripts/lib/cli_*.sh`); upgrade and repair hook writes the dry-run report under `.install/`; new error codes `registry_not_migrated`, `registry_full`, `registry_conflict`; `MAX_REGISTRY_CLASSES` constant | CLI, installer step | CLAUDE.md command list, INSTALLATION (CLI table, upgrade, downgrade note), `docs/CURATION.md`, docs-site | `setup-openprocessor.sh` upgrade/repair (section 5 step 11) | n/a |
| B1-1C | GPU-16 Triton `client_timeout`: one setting (ENV-1); ENV-2(b) `OPENSEARCH_HOSTS` read deleted | settings | template advanced tier | none | ADVANCED |
| B1-1D | `cachetools` added to `requirements.txt`, `requirements-test.txt`, `pyproject.toml` (BA-3); ENV-1, ENV-3, ENV-5, ENV-7, ENV-9, ENV-11; `OCC_BULK_PAGE_SIZE` -> `OP_OCC_BULK_PAGE_SIZE` (retired entry) | deps, env | ATTRIBUTION (cachetools, MIT), template | checkout setup (ENV-3) | ADVANCED |
| B1 deps lane | ENV-12 (`.github/dependabot.yml` ignore entries) and the re-applied dependabot bumps of section 7.2; no `OP_*` change | deps | none | none | none |
| B1-1E | CI job `families` (network, `actions/cache` on the pin and fixture list); fixture cache `~/.cache/op-test-weights`; ENV-6 export worker | CI | [09_github_actions.md](09_github_actions.md), CONTRIBUTING (how to run it locally) | n/a | n/a |
| B2 stage 1 | services `postgres` (the image and digest of [13](13_package_pins.md) section 4, MQ-2, volume `openprocessor-pg` at `/var/lib/postgresql`, `max_connections`), `valkey` (the image and digest of 13 section 4, volume `openprocessor-queue`, AOF everysec, `noeviction`, no `ports:`), `db-migrate` (one-shot), `jobs-worker`, `periodic-worker`, `open-vocab-worker`, `jobs-beat`, `flower` (`monitoring`); env `OP_DB_HOST`, `OP_DB_PORT`, `OP_DB_NAME`, `OP_DB_APP_USER`, `OP_DB_APP_PASSWORD_FILE`, `OP_DB_OWNER_USER`, `OP_DB_OWNER_PASSWORD_FILE`, `OP_DB_DIRECT_HOST`, `OP_DB_SSLMODE`, `OP_DB_POOLER`, `OP_DB_API_CONNECTIONS_TOTAL`, `OP_PG_MAX_CONNECTIONS` and `OP_PG_SHARED_BUFFERS` (compose-only), `OP_QUEUE_URL` (default `redis://valkey:6379/0`, no password in it), `OP_QUEUE_PASSWORD_FILE`, `OP_JOBS_CONCURRENCY` (default 4, D-23), `OP_STOP_GRACE_JOBS` (compose-only, default 60), `OP_DB_BACKUP_KEEP` (7), `OP_DB_BACKUP_INTERVAL_H` (unset); added after the review [00c](00c_plan_review.md): `OP_DB_SUPER_PASSWORD_FILE` (R-13), `OP_API_REPLICAS` (default 1, R-6), `OP_VALKEY_MAXMEMORY` (1gb), `OP_JOBS_MAX_MEMORY_PER_CHILD_KB`, and the compose-only memory limits `OP_PG_MEM_LIMIT`, `OP_JOBS_MEM_LIMIT`, `OP_PERIODIC_MEM_LIMIT`, `OP_BEAT_MEM_LIMIT` (R-19); every Python service, `api` included, waits on `db-migrate` and on healthy `postgres`, `valkey` and `opensearch` (D-69, which replaced D-64); secrets (ENV-14); CLI `openprocessor db backup/restore`; the shared env generator first (ENV-4); deps named in [03](03_b2_overview.md), pinned in file 13 | services, env, secrets, deps | ARCHITECTURE, CLAUDE.md, INSTALLATION, SECURITY (neutral), CONTRIBUTING (test tiers), docs-site, ATTRIBUTION (section 7.3) | installer `do_install` (fresh, re-run, upgrade, repair) and the shared env generator | BASIC: none (installer-managed); ADVANCED: the rest |
| B2 stage 2 | `op_projects`, `op_global_configs` and four per-project index roles deleted; retired `OP_PROJECTS_INDEX`, `OP_GLOBAL_CONFIGS_INDEX`; the 0.6.0 control-data upgrade step; `openprocessor db verify`; `curation migrate-registry` removed from the CLI; `OP_CONFIG_POLL_S` kept (reads the epoch) | env, CLI, installer step | INSTALLATION (upgrade, downgrade), CLAUDE.md command list | `setup-openprocessor.sh` upgrade (section 5 steps 9-12) | ADVANCED |
| B2 stage 3 | `OP_EVENT_BUS`, `OP_EVENT_LOG_MAX_BYTES` retired with the JSONL log (ENV-16 settled: events in Postgres, D-48) | env | ARCHITECTURE (events) | none | - |
| B2 stages 4-5 | `OP_TRAIN_JOBS_DIR`, `OP_COMBINE_JOBS_DIR`, `OP_AUTO_LABEL_STATE_DIR`, `OP_BAKEOFF_JOBS_DIR`, `OP_GPU_ARBITER_TRAINER_CONTAINER`, the `/jobs/model_export` setting and every other env var read only by deleted job-file code retired (`removed:` entries); ENV-2(a) `OP_TRITON_URL` retired (4D) | env | template, CHANGELOG | none | - |
| B2 stage 6 | `curation-cluster-refresh` deleted; the curation, trainer, export and evaluator services run Celery consumers; `openprocessor-jobs` volume deleted; no `container_name` on any service (D-54); GPU-arbiter overlay mounts the socket into `gpu-arbiter-worker` only (RC-6); opt-in `docker-compose.pgbouncer.yml` (D-50); `OP_INGEST_DISPATCH_S`, `OP_OPEN_VOCAB_SWEEP_S`, `OP_CLUSTER_REFRESH_INTERVAL_S`, `OP_IVF_RETRAIN_CHECK_S` keep their names as beat intervals; `monitoring/prometheus.yml` targets updated | services, env | as stage 1; `docs-site` operations pages | `up -d --remove-orphans` removes the deleted service | ADVANCED |
| B3 | `ultralytics==<X>` everywhere (D-28); `requirements-export-y11.txt`, `/opt/venv-y11`, `src/ultralytics_patches/` deleted; `YOLO11_EXPORT_PYTHON` and `_Y11_REEXEC` (read in `export/export_models.py`, internal, not in the template) deleted; trainer `ULTRALYTICS_SHA` = tag `v<X>` (U-J8); engine stamp changes so `repair`/`upgrade` rebuild the default detector engine (U-R3); export format names renamed (contracts); Triton model name `yolov11_small_trt_end2end` kept (U-J6, D-27) | deps, image, engines | README, `export/README.md`, `docs/MIGRATION_TRITON_26.md`, `docs/image_size_audit.md`, INSTALLATION (engine rebuild on upgrade), ATTRIBUTION (vendored fork removed) | installer model step (stamp) | n/a |
| B4 | `--workers` dropped from compose and the `Dockerfile` CMD; `api` gets `WEB_CONCURRENCY=${API_WORKERS:-<fallback>}` (ENV-15); `API_WORKERS` written by the installer (`min(nproc, 32)`, D-33, confirmed by the B4 benchmark); executor, Triton channel and database pools sized from totals divided by `API_WORKERS` (names ENV-17, plus B2's `OP_DB_API_CONNECTIONS_TOTAL`); `OP_BATCH_SYNC_MAX` (new, 4A); `project_create` job (4D) | env, compose | INSTALLATION (`API_WORKERS`, orchestration note), ARCHITECTURE (events, workers), PERFORMANCE, engineering_standards (J7 exemption), template | installer `env_set_default` (never overwrites a user value) and the shared generator | BASIC: `API_WORKERS`; ADVANCED: the rest |
| B2 research seams (pending RD-1; [00b](00b_traceability.md) section 5.2) | stage 1: service `cluster-worker` (`curation` profile, CPU), secret `secrets/identity_text/master` (32 bytes, mode 600, 1C-9), queue `cluster`; stage 2: the `serving_profile` kind (the installer's `config.pbtxt` instance edits are written from the active profile; `config_templates/profiles/*.json` become the upgrade's seed), the embedding catalog `examples/embedding/catalog.tsv`; stage 4: `OP_CROP_DUP_THRESHOLD` retired (4A-6, the near-duplicate plan's ND-9, `removed:` naming the `near_dup_profile` kind; pending RD-49), the cluster-refresh and IVF-retrain values `OP_CLUSTER_REFRESH_GROWTH_THRESHOLD`, `OP_CLUSTER_REFRESH_MAX_DEFERRAL_S`, `OP_CLUSTER_REFRESH_RETRY_BACKOFF_S`, `OP_CLUSTER_REFRESH_AUTO_LABEL`, `OP_IVF_RETRAIN_GROWTH`, `OP_IVF_RETRAIN_MIN_INTERVAL_S`, `OP_IVF_RETRAIN_CHECK_S`, `OP_MAX_REFINE_MEMBERS`, `OP_RESIDUAL_EMBEDDING_FIELD` retired into `cluster_profile` (4F-7; pending RD-29; `OP_CLUSTER_REFRESH_INTERVAL_S` stays as the beat interval); stage 6: the DCGM profiling fields in the `monitoring` profile (6B-8) | services, env, secrets, config kinds | INSTALLATION, CURATION, ARCHITECTURE, template, CHANGELOG | installer and the shared env generator | ADVANCED |
| B4 research lanes | `OP_PE_TEXT_BACKEND` and `OP_PE_TEXT_ONNX_PATH` retired (`removed:`, ES-E in 4B-7; pending RD-31); `OP_IDENTITY_TEXT_MAX_MODE` (`off`, `keyed`, `raw`, default `raw`, read live, malformed refuses startup; pending RD-61) and `OP_IDENTITY_TEXT_SECRETS_DIR` (default `/run/secrets/op_identity_text`), both ADVANCED; ExifTool in the jobs image for the export GPS strip (CM-4; pending RD-54) | env, image | SECURITY (privacy sections), template, ATTRIBUTION (ExifTool) | none | ADVANCED |
| W5e (plate text) | service `text-read-worker` (`curation`, no GPU) and its secret mount; queue `text_read`; the `region_profile` identity-text fields | services, config | CURATION, INSTALLATION | `up -d` | ADVANCED |
| R-final | VERSION and every version string (ENV-7) to 0.7.0; `${OP_IMAGE_TAG:-0.7.0}` in compose; `SCRIPT_VERSION`; template tiering already done in B4 (ENV-8); image `pip freeze` records (ENV-13) | release | CHANGELOG, release notes | release-prep commit | n/a |

### 4.1 Conflicts between the plans (flagged; each needs the noted resolution before its batch)

| ID | Conflict | Recommendation | Decide in |
|---|---|---|---|
| ENV-14 | Secret shape for the control plane (was the Redis password question). Valkey needs a `requirepass` config and the clients a raw password; Postgres needs an owner password (for `db-migrate` and backups only) and an app password; a password must never reach argv, compose `environment:` or a log | Decided for B2 stage 1 (lane 1C): one random value per secret, written with `umask 077` before compose runs (so Docker never creates a root-owned directory, the trap the installer already handles for `secrets/vlm`): `secrets/valkey/valkey.conf` (`requirepass ...`, passed as the config-file argument to `valkey-server`, AOF and eviction flags on the command line) and `secrets/valkey/password` (`OP_QUEUE_PASSWORD_FILE`); `secrets/postgres/owner_password` (`POSTGRES_PASSWORD_FILE`, `OP_DB_OWNER_PASSWORD_FILE`; mounted only into `postgres`, `db-migrate`, the backup command) and `secrets/postgres/app_password` (`OP_DB_APP_PASSWORD_FILE`); `secrets/flower/auth`. The URLs are built at runtime from host settings plus the files. Valkey healthcheck with `REDISCLI_AUTH` from the file, never `-a`. `upgrade` and `repair` create missing files and never rotate. The file ownership rule (the Postgres and Valkey images drop to their own users; the Python images run as uid 1000) comes from spikes S0 and S1 | B2 stage 1 |
| ENV-15 | `API_WORKERS` default: task plan Phase 7 writes `WEB_CONCURRENCY=${API_WORKERS:-8}`; D-33 says the installer default is `min(nproc, 32)`; today compose hard-codes `--workers=32` and the Dockerfile CMD `--workers 16` | Compose fallback 8 (a checkout without the key starts small); both generators (ENV-4) write the computed `min(nproc, 32)`; the B4 benchmark may change the formula. Document that uvicorn reads `WEB_CONCURRENCY` only when `--workers` is absent (uvicorn settings, `--workers` "Defaults to the `$WEB_CONCURRENCY` environment variable if available, or 1") | B4-4B |
| ENV-16 | Event-log settings: B2 stage 3 deletes the JSONL log (events move to Postgres, D-48; this was B4-4C), which `OP_EVENT_BUS` (`file` or `process`) and `OP_EVENT_LOG_MAX_BYTES` configure. No plan lists their retirement, and `OP_EVENT_BUS=process` is a compatibility mode ("restores the pre-S-3 in-process-only" behaviour, `event_hub.py` docstring) | Retire both with `removed:` entries; the event retention (24 h) is a module constant (no speculative knobs) | B2 stage 3 |
| ENV-17 | Pool sizing: task plan Phase 7 says "size the shared executor and the Triton pool from the totals divided by `API_WORKERS` (new config in `src/config/`)" with no names | Name them at B4 kick-off (proposal: `OP_CPU_POOL_THREADS_TOTAL`, `OP_TRITON_CHANNELS_TOTAL`), read through `env.py`, ADVANCED tier; if the current per-process values are already env-driven, rename with retired entries instead of adding a second name (`rg -n "env_int\('OP_.*(POOL|THREAD|CHANNEL)" src`) | B4-4B |
| ENV-18 | Flower port and exposure. "Flower under `monitoring`, internal only" has no port; 4600-4612 are taken | Decided (owner, Q-3, 2026-11-04): both, port 4613 published on `${OP_UI_BIND_ADDRESS:-${OP_BIND_ADDRESS:-127.0.0.1}}` like the other monitoring UIs and the gateway subpath (`OP_GATEWAY_SUBPATHS`), and only when the `monitoring` profile runs; the SECURITY.md line is neutral and Flower stays loopback by default | B2 stage 1 |
| ENV-19 | Release-train text. The former master plan (rev 4) sections 4, 5.1, 7.1-7.4 and its CHANGELOG known-issue lines ("until the fix in 0.8.0") assume v0.8.0/v0.9.0 and upgrades from 0.7.0/0.8.0 | Owner decision 2 replaces them: one v0.7.0; the B0 known-issue lines are dropped (01 says so); `installer_upgrade` runs from 0.6.0 only (section 5) | [00_index.md](00_index.md), [06_release_acceptance.md](06_release_acceptance.md) |
| ENV-20 | Worker liveness after B2 (lane 4D for the two curation consumers, 5A-5C for the GPU consumers): Phase 5 says "keep the `worker_liveness` heartbeats or use `celery inspect ping`; pick one"; Phase 2 ports a pool healthcheck because "a ping cannot see a wedged prefork pool" | Celery services use the ported pool healthcheck only; `worker_liveness.py` stays for the detection and VLM workers (not migrated). Delete the liveness entries of the converted workers in the same commit | B2 stages 4-6 |

## 5. Upgrade path 0.6.0 -> 0.7.0 (config side; input for R-final)

`openprocessor upgrade` runs `setup-openprocessor.sh` (`cmd_upgrade() { _run_setup "$@"; }` in
`scripts/lib/cli_core.sh`). What it must do, in order. Steps marked **new** do not exist today.

| # | Step | Today | Batch that adds it | Failure behaviour |
|---|---|---|---|---|
| 1 | Refuse retired names in `.env` (bash list in `scripts/lib/retired_env.sh`; Python list at every entry point, ENV-6) | bash list covers the Cropwright names only | B1 (`GPU_ID`, ENV-3), B2 (`OP_TRITON_URL`, `OP_EVENT_*`, the job-directory settings, `OP_PROJECTS_INDEX`, `OP_GLOBAL_CONFIGS_INDEX`), each batch's own retirements | exit 2 naming the new variable, before anything changes |
| 2 | Back up the install (`backup_install`); from 0.7.0 on also `openprocessor db backup` | exists (files) | B2 stage 1 (database dump) | stop |
| 3 | Merge new **active** template keys into `.env` (`env_create_or_merge`); never overwrite a user value | exists | ENV-4 makes the checkout path do the same | log the added keys |
| 4 | **new** Write computed defaults with `env_set_default`: `API_WORKERS=min(nproc, 32)` | `env_set_default` exists | B4 | n/a |
| 5 | **new** Create the Postgres, Valkey and Flower secrets when missing (ENV-14); never rotate | no | B2 stage 1 | stop with the path; nothing is printed |
| 6 | Pull and verify images by digest (`images.lock`), including the Postgres and Valkey digests pinned in compose | exists for own images | B2 stage 1 | stop |
| 7 | Stop the 0.6.0 API and workers with their grace periods | compose stop | n/a | n/a |
| 8 | **new** `up -d postgres valkey opensearch`, then `run --rm db-migrate` (schema to head, the app role) | no | B2 stage 1 | stop; `--rollback` available |
| 9 | **new** From 0.6.0 only: the control-data step (bundle, load, verify, delete sources; [02c](02c_migration_ops_tests.md) section 2); in-flight 0.6.0 jobs recorded `interrupted` (D-51) | no | B2 stage 2 | stop with the report path before anything is deleted; `--rollback` recreates from the bundle |
| 10 | Engine stamp check rebuilds the default detector engine once (U-R3) | stamp logic exists (`scripts/lib/model_stamps.sh`) | B3 | stop with "Triton unreachable" vs "engine invalid" distinguished (R3 check carried into R-final) |
| 11 | `up -d --remove-orphans` (removes `curation-cluster-refresh` and the `/jobs` mounts, adds the B2 services); the B1 `migrate-registry` hook is gone (B1 never shipped; the class registry moves in step 9) | exists | B2 stages 2 and 6 | stop; `--rollback` available |
| 12 | **new** Print the control-data report, the interrupted jobs, the bundle path and the new services (and that Flower is opt-in) | no | B2 stage 2 | n/a |

**Rollback to 0.6.0** (`--rollback` restores the backup): `scripts/db/rollback_to_060.py` recreates
the deleted control indexes and files from the upgrade bundle and renames
`class_registry.json.migrated-0.7.0` back; the Postgres and Valkey volumes are left unused. Work done
after the upgrade exists only in Postgres and is not carried back (export first). INSTALLATION's
downgrade note and the CHANGELOG "Breaking" section say so ([02c](02c_migration_ops_tests.md)
section 2.3).

**Proof (R-final):** `acceptance_run.sh --only installer_upgrade` from a 0.6.0 install whose
projects have classes, configs and an in-flight job: ids, names and revisions equal after the
upgrade; the OpenSearch control indexes gone; the bundle present; the secrets present with mode 600;
`.env` holds `API_WORKERS` and no retired name; `curation-cluster-refresh` gone; `postgres`,
`valkey`, the workers and beat healthy; then rollback and re-upgrade. Offline: the `tests/upgrade/`
set of [02c](02c_migration_ops_tests.md) section 2.4 and `tests/installer/test_control_plane_secrets.py`.

## 6. CI parity: every job and its local command

Run these in the integration worktree before a push (the user's rule: never discover a failure in
CI). Lanes run only their targeted tests; the sweep runs this table once per wave.

| Workflow / job | Trigger | Local command (from the repo root unless noted) | Gap |
|---|---|---|---|
| `CI` / `pytest` | PR, push to main | `.venv/bin/python -c "import src.main"` then `.venv/bin/python -m pytest tests/ -q -m 'not live' --cov=src --cov=scripts --cov-report=term --cov-fail-under=80` | CI runs **serially** on 3.12 with only `requirements-test.txt`; local sweeps use `-n 6`, which hides test-order pollution (T-8). Run the sweep once without `-n` or accept that T-8's ordered run is the only order check. ENV-10 for 3.13 |
| `CI` / `pre-commit` | PR, push | `.venv/bin/pre-commit run --all-files --show-diff-on-failure` | needs `hadolint` 2.15.1 and `golangci-lint` 1.64.8 on `PATH` and Go >= 1.22 for the gitleaks build (CI installs them); a different local binary can pass what CI rejects (engineering_standards 5.4) |
| `CI` / `changelog` | PR, push | `git diff --name-only "$(git merge-base origin/main HEAD)"...HEAD \| rg '^(src\|scripts\|docker\|export\|models)/'` non-empty requires `CHANGELOG.md` in the same list, unless the last commit message has `[skip-changelog]` (`tests/test_ci_changelog_guard.py` executes the workflow block) | no Make target; the command above is the equivalent |
| `Frontend` / `check` | PR or push touching `frontend/**`, `contracts/**` | in `frontend/`: `npm ci && npm run check && npm run lint && npx knip && npm test && npm run build 2>&1 \| tee build.log && ! grep -qi deprecated build.log && python3 -m py_compile scripts/*.py e2e/*.py e2e/stubbed/*.py e2e/fixtures/*.py e2e/live/*.py` | real `npm ci` per worktree (no symlinked `node_modules`) |
| `Frontend` / `e2e-stubbed` | same | in `frontend/`: `python3 -m venv e2e/.venv && e2e/.venv/bin/pip install -r e2e/requirements.txt && e2e/.venv/bin/playwright install chromium && npm run test:e2e` | heavy; CI also installs system deps (`--with-deps`, needs root); schedule with other heavy runs; headed runs need the active X display |
| `Frontend` / `changelog` | same | as the backend check, for `^frontend/(src\|scripts)/` and `frontend/CHANGELOG.md` | none |
| `Deploy Documentation` / `build` | PR or push touching `docs-site/**`, `docs/**` | in `docs-site/`: `npm ci && npm run typecheck && npm run build` (`make docs-build` runs `npm ci` and the build; add `npm run typecheck` to match CI) | `make docs-build` runs `npm ci && npm run build` only (verified): it skips CI's `npm run typecheck`. Add the step to the target (XS, B1 with 09) or run it by hand |
| `Security Scan` / `fs-scan` | PR to main, push to main, weekly | `trivy fs --severity CRITICAL,HIGH --ignore-unfixed --exit-code 1 --ignorefile scripts/release/trivy-allowlist.txt .` | none (Trivy is installed on the host) |
| `Security Scan` / `image-scan` (7 images: api, export, evaluator, trainer, segmenter, cropwright, triton) | same | per image: `docker build [--target T] -t openprocessor-scan:<img> -f <file> <ctx> && trivy image --severity CRITICAL,HIGH --ignore-unfixed --exit-code 1 --ignorefile scripts/release/trivy-allowlist.txt --timeout 30m openprocessor-scan:<img>` | `make scan` covers only two images; run the full set only when a Dockerfile or requirements file changed, and always in R-final. Heavy (the Triton image is about 32 GB) |
| `Docs External Links` | weekly, manual | `lychee` is not installed; `docker run --rm -v "$PWD":/w -w /w lycheeverse/lychee:<pinned> docs` if wanted | **no local equivalent installed**; not a PR gate |
| `Frontend mutation testing` | weekly | in `frontend/`: `npm run test:mutation` | not a PR gate; slow |
| `families` (new in B1-1E) | PR touching the pin, `export/`, adapters, `model_export.py` | the master plan's local run in the export image (CPU, `--cpus 8`, its own `-p`), or `.venv/bin/python -m pytest tests/export -m export` with the weight cache populated | needs network for the fixture download; must never skip in CI |

Not separate jobs, already inside the above: contracts drift (`api-contracts-drift` and
`region-status-ts-drift` pre-commit hooks; `make contracts-check` is the same), the docs check
(`tests/test_docs_vs_code.py` in the offline suite), the ratchets (pre-commit hooks).

## 7. Dependency hygiene

### 7.1 Exact-pin status of the packages the plans depend on

| Package | `requirements.txt` | `requirements-test.txt` | `pyproject.toml` | Image / other | Plan |
|---|---|---|---|---|---|
| ultralytics | `==8.4.197` | `==8.4.197` | `==8.4.197` | `/opt/venv-y11` `==8.3.253`; trainer git SHA `8f5d355c` (`docker/trainer/Dockerfile` `ULTRALYTICS_SHA`) | one `==<X>` and the trainer at tag `v<X>` in B3 (D-28, U-J8) |
| fastapi | `==0.143.3` | `==0.143.3` | unpinned | `tests/test_requirements_contract.py::test_the_image_pins_fastapi_as_the_tests_do` | ENV-11; bump only with `make contracts` (#434) |
| pydantic | via fastapi | `==2.14.1` | via fastapi | | keep |
| uvicorn | `>=0.41.0` | `>=0.41.0` | `>=0.41.0` | images resolve 0.54.0 (master plan) | floor is enough (flag added in 0.41.0) |
| tensorrt-cu13 | export: `==11.1.0.106` | | `[export]` `==11.1.0.106` | asserted in `Dockerfile.triton` | moves only with the Triton base (ENV-12) |
| onnx / onnxruntime-gpu | `>=1.12,<1.22` / `>=1.24.2,<1.25` | onnx, onnxruntime (CPU) unpinned | same as runtime | modelopt constraint | reject #434's cap widening |
| pyclipper | (Triton image) | `==1.4.0` | | `Dockerfile.triton` `==1.4.0` | done (B0, GPU-2) |
| tenacity | `>=8.2` | `>=8.2` | `>=8.2` | | keep (used by `vlm_client.py`) |
| opensearch-py | `>=2.3.0` | `opensearch-py[async]>=2.3.0` | `>=2.3.0` | | keep; `aiohttp` stays a dependency because the async client needs it |
| cachetools | absent (try-import in `image_serving.py`) | absent | absent | | add in B1-1D (BA-3), floor pin, MIT |
| SQLAlchemy, Alembic, psycopg, Celery, kombu, billiard, redis-py, Flower, testcontainers; images `postgres`, `valkey` | absent | absent | absent | | exact pins and digests in `13_package_pins.md` (written on its own branch); the traps (kombu 5.7 under Celery 5.6, the redis-py range kombu accepts, billiard >= 4.3.1) in [02b](02b_jobs_locking_events.md) section 7.1; spike S1 runs on the pins |

Rule for exact pins: use `==` where the package's behaviour is part of a tested contract or a
gate (Ultralytics export head, FastAPI OpenAPI output, TensorRT engine format, Celery delivery
semantics). Elsewhere a floor plus dependabot's `increase-if-necessary` is the documented policy
(`.github/dependabot.yml` comment).

### 7.2 The five open dependabot PRs

All five target `main`, which does not move until B0 merges. Ask dependabot to rebase after the B0
merge (`@dependabot rebase` as a PR comment), then act. Never let one of them move the Ultralytics
pin: dependabot ignores `ultralytics` already, and `test_ultralytics_is_one_exact_version_everywhere`
fails any manual drift.

The PRs below are taken by the **separate B1 lane `b1/deps`**, not by the 1D lane; its brief,
branch, checks and merge position are in B1 (merged; see [CHANGELOG](../../../CHANGELOG.md)) section 4.1, and
the GitHub side (close comments, sequencing) is in [09_github_actions.md](09_github_actions.md)
section 8. This table owns the package content only. The "Batch" column below reads "deps lane".

| PR | Content (read with `gh pr diff`) | Disposition | Batch |
|---|---|---|---|
| #422 | docker group: Python bases 3.13 and 3.12 -> **3.15**, Triton `26.06` -> `26.09`, `alpine/git` digest; also edits `docker/evaluator/Dockerfile`, which no longer exists on `main` (stale) | close with a comment (interpreter and Triton moves are owner decisions, ENV-10, ENV-12); take the `alpine/git` digest by hand only if Trivy asks | deps lane (after ENV-12 lands) |
| #432 | docker group: Python bases -> **3.14**, Triton `26.09` | close; a Triton move changes TensorRT, so it needs the `tensorrt-cu13` pin, the `Dockerfile.triton` assertion and an engine rebuild together. If wanted, evaluate it inside B3, which rebuilds engines anyway (owner question Q-4) | B3 or later |
| #423 | web-docker: `node:26-alpine` and `nginx-unprivileged:1.31-alpine` **digest-only** updates | take after rebase: frontend `check` + `e2e-stubbed`, Trivy of the cropwright image, merge commit | deps lane |
| #433 | frontend group: `@sveltejs/adapter-static` 4.0.1, `@sveltejs/kit` 3.1.1, `@types/node` 26.6.6, `knip` 6.44.0 (patch/minor) | close and re-apply by hand in the deps lane; the frontend set runs once in the B1 sweep (frontend changed), plus the lane's own targeted `npm run check`/`build` | deps lane |
| #434 | backend group: `onnxruntime-gpu` cap `<1.25` -> `<1.32`; `fastapi` 0.143.3 -> 0.162.0; `pydantic` 2.14.2; `ruff` 0.17.3; `mypy` 2.4.1; `pytest` 9.2.0 (`frontend/e2e`) | close and re-apply by hand in the deps lane, one commit each: ruff with the `ruff-pre-commit` hook rev in the same commit (engineering_standards 2.1: venv and hook must match), mypy, pydantic, pytest (e2e); **fastapi as its own commit** with `make contracts` and the contract tests (19 minor versions); **reject** the onnxruntime-gpu cap (modelopt `~=1.24.2`); add the ENV-12 ignore so it does not return | deps lane |

### 7.3 Licences for libraries the plans add (ATTRIBUTION.md rows)

`ATTRIBUTION.md` today attributes models and vendored code, plus a pyclipper bullet. The plans add
a row per new runtime dependency in a shipped image. Check each licence in the installed wheel's
metadata at pin time (`python -c "import importlib.metadata as m; print(m.metadata('<pkg>')['License'])"`
in the image) and with Trivy's licence scanner (`trivy image --scanners license <image>`,
https://trivy.dev/latest/docs/scanner/license/), which also flags a transitive copyleft or unknown
licence.

| Library | Licence (to confirm at pin time) | Added in | Note |
|---|---|---|---|
| cachetools | MIT | B1-1D | BA-3; the try-import fallback in `image_serving.py` is deleted in the same commit |
| celery, kombu, billiard, vine | BSD-3-Clause | B2 stage 1 | transitive `click` (BSD-3) and friends come with celery |
| redis (redis-py) | MIT | B2 stage 1 | the client for Valkey through kombu's `redis://` transport |
| flower | BSD-3-Clause | B2 stage 1 | monitoring profile only |
| SQLAlchemy, Alembic | MIT | B2 stage 1 | |
| psycopg (3) | LGPL-3.0 | B2 stage 1 | used as a library, unmodified |
| testcontainers | Apache-2.0 | B2 stage 1 | test only, not in any image |
| Valkey server 9 (image) | BSD-3-Clause (https://github.com/valkey-io/valkey/blob/unstable/COPYING) | B2 stage 1 | D-18 approved 2026-11-04 |
| PostgreSQL 18 (image) | PostgreSQL Licence (permissive) | B2 stage 1 | |
| pyclipper | MIT (Clipper: Boost 1.0) | done | B0, GPU-2 |
| Ultralytics vendored patch | AGPL-3.0 | removed in B3 | delete its ATTRIBUTION section with the code |

## 8. Owner questions raised here

| Q | Question | Recommendation |
|---|---|---|
| Q-1 | Which interpreter does CI test (ENV-10)? | add one 3.13 `pytest` job; keep host venvs and mypy on 3.12 until the host has 3.13 |
| Q-2 | Lock files for image builds (ENV-13)? | record `pip freeze` per image in R-final now; lock files later |
| Q-3 | Flower: port 4613 or gateway subpath only (ENV-18; B2 stage 1)? | decided 2026-11-04 (owner): gateway subpath plus loopback port 4613, both off unless `monitoring` runs |
| Q-4 | Evaluate Triton 26.09 inside B3's engine rebuild (#432)? | only if B3's GPU window allows; otherwise after v0.7.0 |
