# Contributing to OpenProcessor

Thanks for contributing. This document covers dev setup, how to run the
test suites, the pre-commit guards and baselines, and the commit/merge
conventions this repo uses.

## Dev setup

```bash
git clone https://github.com/davidamacey/OpenProcessor.git
cd OpenProcessor
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.1
.venv/bin/pip install -r requirements-test.txt pre-commit
.venv/bin/pre-commit install
```

`requirements-test.txt` is the CPU-installable set CI uses. Do not also install
`requirements.txt` into this venv: it is the container image's list and carries
`faiss-gpu-cu12`, which overwrites `faiss-cpu` and breaks the faiss tests.

**Always call venv binaries directly — never `source .venv/bin/activate`:**

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/pre-commit run --all-files
.venv/bin/pre-commit run ruff --all-files
.venv/bin/pre-commit run mypy --all-files
```

`source .venv/bin/activate && ...` works, but it's not this project's
convention and some CI/agent environments flag it — prefer the direct
binary path.

## Running the offline test suite

```bash
# The Docker-free tier: everything except the `live`, `families` and `db` markers (no Docker, no GPU)
.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live and not families and not db'

# With coverage (also run in CI)
.venv/bin/python -m pytest tests/ -q -m 'not live and not families and not db' --cov=src --cov=scripts \
    --cov-report=term-missing:skip-covered

# Import sanity — the loudest, cheapest check; run this first when debugging
.venv/bin/python -c "import src.main"
```

`pyproject.toml`'s `addopts` deselects the `live`, `families` and `db` markers by default, so
a plain `pytest tests/` never touches Docker or the network.

### Test tiers

| Tier | Marker | Needs | Run |
|---|---|---|---|
| Docker-free (the default) | none | nothing | the commands above |
| `db` | `db` | a PostgreSQL: Docker (testcontainers) or `OP_TEST_DATABASE_URL`; no GPU | `make test-db` |
| `live` | `live` | the `docker/test/` harness | `-m live`, see below |

A test under `tests/db/`, or one that uses the `pg` fixture, must carry `pytest.mark.db`
(`pytestmark = pytest.mark.db`); collection stops with an error naming any that does not
(`tests/fixtures/db_marker_policy.py`). `make test-db` runs `-m db` with pytest-xdist: the
fixture starts one PostgreSQL 18 container (the image and digest of `docker-compose.yml`) and
every worker migrates its own template database on it. To use a server you already have, set
`OP_TEST_DATABASE_URL` to a URL whose user is the superuser `op_super`
(`postgresql://op_super:<password>@host:5432/postgres`); CI does this with a service container.
The fixture raises when no database is reachable and never skips, so a green `-m db` run proves
tests ran.

Migrations keep **one Alembic head**. When two branches each add a revision, the second to merge
rebases its migration onto the new head (it changes its `down_revision`) instead of adding a
merge revision; `alembic heads` must print one line. The `db` tier fails on more than one.

The `families` tests run against the official Ultralytics weights and the original-YOLOv5 ONNX
exports. Their files live in `~/.cache/op-test-weights` (`$XDG_CACHE_HOME` when set), are never
committed, and a missing file skips the test locally and fails it under `CI`. The CI job
`families` fills the cache (`python tests/_ultralytics_weights.py fetch`, sha256-checked) and
builds the YOLOv5 ONNX files with the official `ultralytics/yolov5` `export.py` in a separate venv
at the commit pinned in `.github/workflows/ci.yml`. To run them locally, fetch the weights with
that command, export the ONNX once
(`python export.py --weights yolov5n.pt --include onnx --imgsz 640 --device cpu --simplify`, and
with `--dynamic`) as `yolov5n_640.onnx` and `yolov5n_dynamic.onnx` in the cache, then
`.venv/bin/python -m pytest tests/ -q --no-cov -m families`. The weights and the YOLOv5 repository
are AGPL-3.0 and used for tests only.

`tests/test_full_system.py` and `tests/validate_visual_results.py` hit the running stack over HTTP and read
their ports from the environment (`API_PORT`, `TRITON_HTTP_PORT`, `OPENSEARCH_PORT`; defaults 4603/4600/4607):

```bash
mkdir -p test_results
.venv/bin/python tests/test_full_system.py 2>&1 | tee test_results/test_results.txt
.venv/bin/python tests/validate_visual_results.py 2>&1 | tee test_results/visual_validation.txt
```

**Docker-only path.** The production `api` image installs only `requirements.txt` (no `pytest`), so a bare
`docker compose exec api pytest ...` fails. To run the suite in the container, install the test deps first
(not persisted across a recreate):

```bash
docker compose -p openprocessor cp requirements-test.txt api:/tmp/requirements-test.txt
docker compose -p openprocessor exec api pip install -r /tmp/requirements-test.txt
docker compose -p openprocessor exec api python -m pytest tests/ -q --ignore=tests/live
```

## The local gate

`make gate` (or `scripts/dev/gate.sh [py|fe|all]`) runs the checks CI runs, in one command. It
installs first and alone (`pip install -r requirements-test.txt`, `pip check`, `npm ci` in
`frontend/` and `docs-site/`), then runs the Python track (pre-commit on all files, the offline
suite with xdist using the `-m` expression from `pyproject.toml`, contracts, docs check) and the
frontend track (check, lint, knip, unit tests, build, e2e, docs-site build) in parallel. Each step
logs to `artifacts_local/gate/<utc>/<step>.log` and one line to `summary_<track>.txt`; a failed
step is recorded and its track goes on, and the exit code is non-zero if any step failed. On a
shared host, steps run under `nice`/`ionice` and wait while the 1-minute load is above
`GATE_MAX_LOAD` (default 35); set `GATE_NICE=0` on a quiet machine, `GATE_XDIST` for the worker count.

## Running the live write-path verification harness

Roughly half the curation subsystem's write endpoints are only
meaningfully exercised against a real OpenSearch instance and a real
file-based job protocol — mocked-client tests can't catch a mis-wired
env var or a mapping that silently rejects a real document. See
[`docker/test/README.md`](docker/test/README.md) for the full harness;
short version:

```bash
mkdir -p docker/test/verify-data/jobs

docker compose -p op-live-verify -f docker/test/compose.yml up -d --wait
.venv/bin/python -m pytest tests/live -q --no-cov -m live
docker compose -p op-live-verify -f docker/test/compose.yml down -v --remove-orphans
```

Use a **distinct Compose project name** (`-p op-live-verify` above) and
never run this against the repo-root `docker-compose.yml` from a
worktree that might collide with a real deployment's container names or
ports (4600–4610).

## Parallel lanes and worktrees

When several agents or people work in git worktrees of one repository, each uses its own
worktree on its own branch, and no two writers share a worktree. Never use `git stash` there:
the stash is shared by every worktree of a repository, so one lane can pop another's work.
Commit work in progress to your own branch instead and amend or rework it before the merge.

## Contracts

`contracts/` holds generated files: the curation OpenAPI schema, the item wire
types and the keymap action table. After any change to a route, request or
response model, or an item field, regenerate and commit the output in the same
commit:

```bash
make contracts          # regenerate
make contracts-check    # fail if anything is stale
```

Pre-commit rejects a commit that leaves a contract stale. In a parallel lane, regenerate in the
commit that changes the route or model and never hand-edit a file under `contracts/`; when two
lanes both changed a contract, the coordinator regenerates once on the merged branch rather than
resolving the generated files by hand. See
[`contracts/README.md`](contracts/README.md).

## After pulling or merging code

The api and the workers bind-mount the source. Python modules already imported by a
running container go stale, which shows up as transient 500 `ImportError`s. After a pull
or merge run `make dev-restart` (restarts `api` and the detection, VLM, auto-label,
assist-loop workers without recreating them). Restart does not re-read `.env`:
after editing it run `make dev-up` instead (plain `docker restart` or `make dev-restart`
keep the old values).

## Building and scanning images

Two production Dockerfiles at the repo root: `Dockerfile` (the FastAPI `api`
image, with `export` and `evaluator` targets; base `python:3.13-slim-trixie`) and
`Dockerfile.triton` (Triton; base `nvcr.io/nvidia/tritonserver:26.06-py3`).
`scripts/docker-build-push.sh` builds, scans and pushes them:

```bash
./scripts/docker-build-push.sh all      # build and push to Docker Hub (also: api, triton)
./scripts/docker-build-push.sh local    # build locally without pushing
./scripts/docker-build-push.sh scan     # build locally and scan, no push
VERSION=v1.0.0 ./scripts/docker-build-push.sh all
```

Scanning (`scripts/security-scan.sh`) uses free tools: Hadolint (Dockerfile lint),
Dockle (CIS Docker Benchmark), Trivy and Grype (vulnerabilities) and Syft (SBOM).
`./scripts/security-scan.sh install` installs them, `all` or `api` scans, and the
reports land in `security-reports/` as `*-hadolint.txt`, `*-dockle.json`,
`*-sbom.json`, `*-trivy.json/txt` and `*-grype.json/txt`.

In CI, fail the build on findings, or skip the scan for a faster build:

```bash
FAIL_ON_SECURITY_ISSUES=true FAIL_ON_CRITICAL=true ./scripts/docker-build-push.sh all
SKIP_SECURITY_SCAN=true ./scripts/docker-build-push.sh all
```

## Documentation checks

Docs are checked against the code. A route written in a doc must exist, an
`OP_*` variable must be read by code, and relative links and anchors must
resolve:

```bash
.venv/bin/python scripts/docs/check_docs_vs_code.py                  # everything
.venv/bin/python scripts/docs/check_docs_vs_code.py --only README.md # some files
.venv/bin/python -m pytest tests/test_doc_links.py -q --no-cov
```

The docs site builds with `make docs-build` (npm only; `onBrokenLinks` and
`onBrokenAnchors` are set to `throw`, and CI runs the same build on every PR touching
`docs-site/`).

Write each route in full on its own mention (`GET /curation/projects/{project}/stats`,
not a brace-expanded shorthand). Public data only in docs and fixtures (COCO,
Open Images); no real hostnames, secrets or private product names
(`scripts/codegen/check_naming_leaks.py` enforces the last one).

## Guards and baselines

`.venv/bin/pre-commit run --all-files` runs the guards below; each is a script in
`scripts/codegen/` with unit tests in `tests/codegen/`. The ratchets keep a baseline of
today's findings in `scripts/codegen/baselines/` (`key<TAB>value<TAB>reason`, one reason
per line). A baseline only shrinks: a new finding, a finding that grew, a recorded value
higher than reality and an entry with no finding left all fail. After you fix something,
run the guard with `--tighten` to lower or drop the lines it names, and commit the result.
Never add a line to hide a new finding; fix it.

| Guard | Rule |
|---|---|
| `check_file_size.py` | 700 lines per first-party file (`CLAUDE.md` 400) is a guide, not a target. A file in the 640-700 band is split by concern when you touch it, never by collapsing lines and never to dodge the number. A file that is one cohesive concern may exceed it with a `cohesive: <reason>` entry in `scripts/codegen/baselines/file_size.txt`, reviewed at merge; existing baselines otherwise only shrink |
| `check_layering.py` | routers import services and clients, never the reverse; clients never import services; `src/domain`, `config`, `utils` and `schemas` import neither; only routers import `fastapi`. No baseline: the one permanent exception (the guarded OpenSearch client factory) is listed with its reason in the script |
| `check_ruff_ratchet.py` | no function above complexity 15, no new `except Exception` |
| `check_file_locks.py` | no `fcntl.flock`, `exclusive_*_lock` or `filelock` guarding cross-process state: claim it with an OpenSearch job lease (`src/clients/job_lease.py`) |
| `check_process_ids.py` | no wave, round or work-package ids, and no history names, in comments, docstrings or file names |
| `check_cited_paths.py` | a comment or docstring that cites a `docs/`, `scripts/` or similar file cites one that exists |
| `check_vulture.py` | no unused code; a reviewed false positive goes in `vulture_whitelist.py` with a reason |
| `check_naming_leaks.py` | no private product or domain names in the tracked tree |
| ruff `TID251` | `src/` reads the environment only through `src/config/env.py` (`env_str`, `env_int`, `env_float`, `env_bool`, `env_path`); a malformed value raises naming the variable |
| `noqa-needs-reason` | every `# noqa: CODE` ends with `  # <why>` |

The config that makes these work, each guard's baseline and proof recipe, and the pitfalls hit while
landing them are in [docs/design/engineering_standards.md](docs/design/engineering_standards.md).

## Commit conventions

**Conventional commits**, imperative mood: `<type>(<scope>): <summary>`.
Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `build`,
`chore`, `security`, `perf`, `ci`. The `commit-msg` hook checks the format.

Examples from this repo's history:
- `feat(curation): add generic YOLO label import`
- `fix(config): unify feature-flag and crop-cache env vars across router and service layers`
- `docs(curation): add the curation user guide`

**Merge commits, never squash.** This repo preserves history with
`git merge --no-ff` (or rebase-then-merge) rather than squash-merging —
each commit is treated as a documentation artifact of what changed and
why. Please don't ask for a squash merge on your PR.

**Never bypass hooks or signing** — no `--no-verify`, no
`--no-gpg-sign`. If a pre-commit hook fails, fix the underlying issue
rather than skipping it.

## Pull requests

- Keep PRs scoped to one logical change; large mechanical renames
  (e.g. a service rename) should be their own PR, separate from
  functional changes.
- CI (`.github/workflows/ci.yml`) runs the offline pytest suite and
  `pre-commit run --all-files` on every PR — both must be green. Changes under
  `docs-site/` also build the documentation site (`docs.yml`).
- A change that moves the API or wire format includes the regenerated
  `contracts/`.
- No compatibility shims: when a route, field or setting is replaced, delete
  the old one in the same change and update the docs that named it.
- If your change touches `docs/`, run
  `.venv/bin/python -m pytest tests/test_doc_links.py -q --no-cov`
  locally first — it catches dangling relative markdown links.

## Leak discipline (if you're porting from a private reference)

If you're contributing code that started life in a private/internal
codebase, genericize it before it lands: no internal product names,
internal namespace prefixes, internal host paths, or licensed-corpus
names in anything committed. See
`docs/design/curation_design_rationale.md` for the pattern this repo
already follows (the `RegionFields` indirection, the frozen wire
contract) when porting domain-specific code into a generic subsystem.
