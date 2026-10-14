# Contributing to OpenProcessor

Thanks for contributing. This document covers dev setup, how to run the
test suites, what the region-field ratchet is, and the commit/merge
conventions this repo uses.

## Dev setup

```bash
git clone https://github.com/davidamacey/OpenProcessor.git
cd OpenProcessor
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv/bin/pip install -r requirements-test.txt pre-commit pytest-xdist
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
# Everything except the `live` marker (needs no Docker, no GPU)
.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'

# With coverage (also run in CI)
.venv/bin/python -m pytest tests/ -q -m 'not live' --cov=src --cov=scripts \
    --cov-report=term-missing:skip-covered

# Import sanity — the loudest, cheapest check; run this first when debugging
.venv/bin/python -c "import src.main"
```

`pyproject.toml`'s `addopts` deselects the `live` marker by default, so
a plain `pytest tests/` never touches Docker.

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

## Contracts

`contracts/` holds generated files: the curation OpenAPI schema, the item wire
types and the keymap action table. After any change to a route, request or
response model, or an item field, regenerate and commit the output in the same
commit:

```bash
make contracts          # regenerate
make contracts-check    # fail if anything is stale
```

Pre-commit rejects a commit that leaves a contract stale. See
[`contracts/README.md`](contracts/README.md).

## After pulling or merging code

The api and the workers bind-mount the source. Python modules already imported by a
running container go stale, which shows up as transient 500 `ImportError`s. After a pull
or merge run `make dev-restart` (restarts `api` and the detection, VLM, auto-label
and cluster-refresh workers without recreating them). Restart does not re-read `.env`:
after editing it run `make dev-up` instead (plain `docker restart` or `make dev-restart`
keep the old values).

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
| `check_file_size.py` | 700 lines per first-party file (`CLAUDE.md` 400); a file in the 640-700 band is split by concern when you touch it, never by collapsing lines |
| `check_layering.py` | routers import services and clients, never the reverse; clients never import services; `src/domain`, `config`, `utils` and `schemas` import neither; only routers import `fastapi`. No baseline: the one permanent exception (the guarded OpenSearch client factory) is listed with its reason in the script |
| `check_ruff_ratchet.py` | no function above complexity 15, no new `except Exception` |
| `check_process_ids.py` | no wave, round or work-package ids, and no history names, in comments, docstrings or file names |
| `check_vulture.py` | no unused code; a reviewed false positive goes in `vulture_whitelist.py` with a reason |
| `check_naming_leaks.py` | no private product or domain names in the tracked tree |
| ruff `TID251` | `src/` reads the environment only through `src/config/env.py` (`env_str`, `env_int`, `env_float`, `env_bool`, `env_path`); a malformed value raises naming the variable |
| `noqa-needs-reason` | every `# noqa: CODE` ends with `  # <why>` |

Commit messages follow Conventional Commits (`type(scope): summary`); the `commit-msg` hook
checks them.

## Commit conventions

**Conventional commits**, imperative mood: `<type>(<scope>): <summary>`.
Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `build`,
`chore`, `security`, `perf`, `ci`.

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
