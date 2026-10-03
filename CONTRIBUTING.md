# Contributing to OpenProcessor

Thanks for contributing. This document covers dev setup, how to run the
test suites, what the region-field ratchet is, and the commit/merge
conventions this repo uses.

## Dev setup

```bash
git clone https://github.com/davidamacey/OpenProcessor.git
cd OpenProcessor
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-test.txt
.venv/bin/pre-commit install
```

**Always call venv binaries directly — never `source .venv/bin/activate`:**

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/pre-commit run --all-files
.venv/bin/python -m ruff check src/
.venv/bin/python -m mypy src/
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
or merge run `make dev-restart` (restarts `yolo-api` and the detection, VLM, auto-label
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

## The `check-no-literal-region-fields` ratchet

`scripts/codegen/check_no_literal_region_fields.py` is a pre-commit hook
that rejects new `'plate_...'`/`"plate_..."` string literals in changed
`src/`, `scripts/`, and `tests/` files. `RegionFields`
(`src/config/region_fields.py`) is the single source of truth for
OpenSearch region field names — a hardcoded `plate_*` literal almost
always means code was copied in without being routed through
`RegionFields`, silently reintroducing a domain-specific name into
otherwise-generic code.

It only checks files already listed in the script's `PORTED_PATHS`
allowlist — new files aren't retroactively checked until you add them.
**If you port or rewrite a file that touches region fields, add its
path to `PORTED_PATHS` in the same commit.** This makes the guard a
ratchet: once a module is clean, it can never regress.

`src/config/region_fields.py`'s docstrings and
`tests/curation/test_region_fields.py`'s overridability fixture use
`roi_*` as their illustrative "existing deployment" example name, so
neither needs an exemption from the `plate_*` guard. See the script's
module docstring for the frozen-wire-contract exemptions
(`region_status` etc. are legitimate on the HTTP JSON contract; see
`docs/design/curation_api_contract.md`).

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
