# Engineering standards and lessons learned

Status: reference. Every number, snippet and failure below comes from the repo cleanup series
(nine PRs plus follow-up fixes, planned in [repo_cleanup_plan.md](repo_cleanup_plan.md)). This file
is written to be applied to a **sister repo** (a Python/FastAPI + Svelte application of similar
shape). The sister repo is not touched by this series; section 4 is the ordered port checklist.

Notation: `$MAIN` is the main checkout of a repo, `$MAIN-wt/<slug>` a worktree. Commands use the
project venv binaries directly (`.venv/bin/python`, `.venv/bin/pre-commit`), never `source`.

Contents: 1 principles, 2 config that worked, 3 guards, 4 port checklist, 5 pitfalls, 6 results,
7 what is not portable.

## 1. Principles

1. **A guard has a baseline that only shrinks; never a mass ignore.** When a new rule finds
   hundreds of violations, record today's findings in a baseline (`key<TAB>value<TAB>reason`) and
   make the guard fail on growth, on a recorded value higher than reality, and on an entry with no
   finding left. Never add a blanket `noqa`, exclude a directory, or raise a threshold. Each later
   change deletes its own lines (`--tighten`), so the debt is visible and ends at zero.
2. **Split by concern; never trim to fit.** A size ceiling (700 lines) is enforced, but the answer
   to a long file is a split along its responsibilities, with the code moved verbatim first, then
   importers repointed. Collapsing lines, deleting comments or minifying to get under the cap is
   rejected. A cohesive file may keep a reasoned baseline entry (one remains: a single-function
   installer whose truncated download must be a syntax error).
3. **Fail closed.** An outage must not look like an empty answer. A handler that catches a broad
   exception and returns `None`, `False`, `[]` or `0` is a bug unless the fallback is the feature.
   Counts become nullable and render as unknown; an unreadable job-state file counts as "busy"; a
   malformed environment value refuses startup naming the variable; an unbound project refuses
   the operation.
4. **Comments say non-obvious why, never what, never history.** No work-package, wave, round or
   review-finding ids in comments, docstrings or file names; keep the reason in words or link a
   public issue number. Tests are named by behavior, not by the round that found the bug.
5. **No shims, no aliases, no compatibility code in a monorepo.** A moved symbol is imported from
   its new home everywhere in the same commit. A re-export "for older callers" hides dead code from
   every other tool. Retire a removed config name with a startup refusal that names the new one.
6. **Identity by name, not index.** Never carry a raw class index across a boundary.
7. **Evidence first.** Every behavior fix gets a test that is watched failing on the old code; every
   new guard gets a plant-a-violation proof (section 3). A test or an audit that was never seen
   failing is not evidence. Agent reports are not evidence either: re-check with the tool.
8. **Review depth follows risk.** Behavior-identical moves of ordinary code need the guards and the
   existing tests; isolation, locking and concurrency code gets characterization tests first and an
   independent review.

## 2. Configuration that worked

Each snippet is the working config with the reason it is shaped that way. Paths are this repo's
(`src`, `scripts`, `docker`, `export`); substitute the sister repo's trees.

### 2.1 Ruff (`pyproject.toml`)

```toml
[tool.ruff]
target-version = "py312"
line-length = 100
include = ["src/**/*.py", "scripts/**/*.py", "tests/**/*.py", "export/**/*.py",
           "models/**/*.py", "docker/**/*.py"]
force-exclude = true        # apply exclude even when pre-commit passes explicit paths

[tool.ruff.lint]
select = ["E","W","F","I","N","UP","B","A","C4","DTZ","T10","EM","ISC","ICN","PIE","PT",
          "Q","RSE","RET","SIM","TID","TCH","ARG","PTH","ERA","PL","PERF","RUF",
          "S110","S112"]       # a swallowed error: zero hits at adoption, enforced outright
ignore = ["E501","PLC0415","PLR0913","PLR0917","PLR0912","PLR0915","PLR2004",
          "N803","N806","EM101","EM102","SIM108","RUF012"]
fixable = ["ALL"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py"   = ["ARG001","PLR2004","PT028","RUF001","TID251"]
"scripts/**/*.py" = ["T201","PLR2004","PT028","TID251"]
"docker/**/*.py"  = ["TID251"]        # side-car images do not import src
"src/config/env.py" = ["TID251"]      # the one module that reads the environment

[tool.ruff.lint.flake8-tidy-imports.banned-api]
"os.environ".msg = "Read env through src.config.env (env_str/env_int/env_float/env_bool/env_path)."
"os.getenv".msg  = "Read env through src.config.env (env_str/env_int/env_float/env_bool/env_path)."

[tool.ruff.lint.flake8-type-checking]
runtime-evaluated-base-classes = ["pydantic.BaseModel", "pydantic_settings.BaseSettings"]
exempt-modules = ["typing", "src.routers.curation._common_models"]
```

Why:

- **Every ignore carries a reason.** `PLC0415` stays ignored because lazy imports of heavy
  dependencies are deliberate (this is also why the layering guard walks function bodies, 3.2).
  `PLR0912/0915` are ignored because branch and statement counts are replaced by the per-function
  complexity ratchet (3.3), which is a single number that can only go down.
- **`S110`/`S112` selected on their own**, not the whole `S` family, because they had zero hits and
  catch the swallowed-error shape; the rest of the bandit-like rules run in bandit (2.4).
- **`TID251` env ban.** One typed module (`env_str`, `env_int`, `env_float` (finite only),
  `env_bool`, `env_path`; unset or blank returns the default, set-but-malformed raises an
  `EnvError` naming the variable). 170 raw reads in 70 files were migrated. Exemptions are
  per-file-ignores for trees that are separate programs (`tests`, `scripts`, `docker`, `export`),
  not `src`. The one `src` exception (a stdlib-only healthcheck run by file path) is a single
  `# noqa: TID251  # <reason>`.
- **`runtime-evaluated-base-classes`** removed about 57 stale `TC001` noqa comments: pydantic
  resolves annotations at runtime, so those imports are not type-checking-only. FastAPI resolves
  route annotations at call time too, hence the `exempt-modules` entry for the module that holds
  the request and response models. Without it `TCH` rewrites the import into `TYPE_CHECKING` and
  the route breaks.
- **Hook id `ruff-check`** (renamed from `ruff` by the 0.17 hook). Pin the same ruff version in
  the venv (`requirements-test.txt`) and the hook rev; the ratchets (3.3) run ruff from the venv,
  so a version drift would make a local pass and a CI fail.
- **`PTH123` and `models/`** were added late; both found real sites (54 `open()` calls) and a
  directory that had silently been outside lint. Re-read the `include` list when adopting.
- Never use `ruff check --fix --select RUF100` to remove unused noqa comments: `--select`
  replaces the configured rules, so every other rule's noqa looks unused and is deleted. Use
  `ruff check --fix --fixable RUF100`.

### 2.2 Coded noqa must say why

```yaml
- repo: local
  hooks:
    - id: noqa-needs-reason
      name: noqa carries a reason
      language: pygrep
      entry: '# noqa: ?[A-Z][A-Z0-9, ]*$'
      types: [python]
      files: ^(src|scripts|docker|export)/
```

A `# noqa: CODE` that ends the line is rejected; write `# noqa: CODE  # <why>`. Plus the stock
`python-check-blanket-noqa` and `python-check-blanket-type-ignore` hooks. Adoption cost here: 87
reasons added in one commit.

### 2.3 mypy (project interpreter, pydantic plugin)

```toml
[tool.mypy]
python_version = "3.12"
files = ["src", "scripts", "docker", "export"]
exclude = ["^src/ultralytics_patches/"]       # vendored third-party patches
plugins = ["pydantic.mypy"]
check_untyped_defs = true
no_implicit_optional = true
warn_unused_configs = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
strict_equality = true
ignore_missing_imports = true
explicit_package_bases = true
mypy_path = "."
namespace_packages = true
```

```yaml
- id: mypy
  name: mypy
  entry: bash -c 'py=python3; [ -x .venv/bin/python ] && py=.venv/bin/python; exec "$py" -m mypy'
  language: system
  pass_filenames: false
  files: ^(src|scripts|docker|export)/.*\.py$|^(pyproject\.toml|requirements-test\.txt)$
```

Why:

- **Run mypy from the project interpreter (`language: system`), not the `mirrors-mypy` hook.** The
  isolated hook environment has no numpy, fastapi or pydantic, so every third-party type becomes
  `Any` and the hook passes vacuously. Switching found 175 errors over four trees (77 once stale
  ignores were dropped) that had been invisible. The fallback to `python3` is for CI, where the
  dependencies are installed.
- **`pass_filenames: false`** plus the incremental cache: cold run 1 min 39 s at load 22 (6 min 53 s
  under heavier load), warm 3 s. CI caches `.mypy_cache`, keyed on `requirements-test.txt` and
  `pyproject.toml` plus the SHA with a restore prefix.
- **`plugins = ["pydantic.mypy"]`** removed false call-arg errors on model constructors.
- **`warn_unused_ignores = true`** deleted about 25 stale ignores immediately; keep it on so an
  ignore is always live. A remaining `# type: ignore[code]  # reason` is for a wrong third-party
  stub only.
- Pin mypy and the `types-*` stubs in the test requirements; add the hook's `files:` trigger on
  `pyproject.toml` and the requirements so a dependency bump re-runs it.
- **Typing ladder (final step of the port, section 4 step 11).** After everything else: set
  `warn_return_any = true`, `disallow_incomplete_defs = true`, `disallow_untyped_defs = true`
  globally with a `tests.*` override that sets `disallow_untyped_defs = false`, and delete the
  per-module `disable_error_code` relax override as its modules get fixed (most are touched by the
  split and layering steps anyway). Enabling it last avoids typing code that was about to be
  deleted or moved. Final error and override counts: see the placeholder row in section 6.

### 2.4 Bandit

```toml
[tool.bandit]
exclude_dirs = ["tests", ".venv", "venv", "cache"]
skips = ["B101", "B311", "B404"]
```

```yaml
- repo: https://github.com/PyCQA/bandit
  rev: 1.9.4
  hooks:
    - id: bandit
      args: [-c, pyproject.toml]
      additional_dependencies: ["bandit[toml]"]
      exclude: ^(src/ultralytics_patches/|tests/|models/|frontend/)
```

Scope was `src` only; it became `src scripts export docker` (the exclude list had silently dropped
production code). Findings: 2 High and 18 Medium fixed or given a coded `# nosec Bxxx  <reason>`, plus
19 Low call-site findings with a `nosec` each. `B311` (random, 12 sites of non-cryptographic jitter
and sampling) and `B404` (importing subprocess is advisory; each call is checked by B602/B603/B607)
stay skipped with a stated reason. A bind-all address in a metrics server that Prometheus scrapes
over the compose network is a reasoned `nosec`, not a config change, because loopback would break
monitoring.

### 2.5 pytest, timeout, coverage

```toml
[tool.pytest.ini_options]
addopts = ["-v", "--strict-markers", "--tb=short", "-m", "not live"]
timeout = 400      # about 3x the slowest offline test (124 s measured under 6 workers)
markers = ["integration: ...", "live: needs the verification harness", "unbound: ..."]

[tool.coverage.run]
source = ["src", "scripts"]
omit = ["*/tests/*", "*/test_*.py", "*/__pycache__/*", "*/site-packages/*"]
```

- **`pytest-timeout`**: set the value from `--durations=30`, not from a guess.
- **Coverage floor policy.** `--cov*` is removed from default `addopts` (fast local runs). The CI
  job passes `--cov=src --cov=scripts --cov-fail-under=<floor>`. The old floor of 51 had never been
  enforced because CI ran `--no-cov`, so measure before choosing. Measured 84.95 percent in the
  full local environment; the CI floor is **80**, not "measured minus one", because the runner
  installs fewer packages and skips more tests. Raise it after the first CI run reports, and
  whenever a change raises coverage.
- Live-stack tests are deselected by default with a marker; the offline suite is the merge gate.
- Run the full offline suite once per PR with `-n 6` (pytest-xdist). It grew from 8,179 to 8,678
  tests across the series and takes 10 to 11 minutes.

### 2.6 Hooks, revs and tool notes

Hook revs bumped in one commit before any other edit (so later PRs do not re-touch files), with ruff
alone in a second commit (its findings and format churn: 118 files, below the 300 that would have
justified deferring): pre-commit-hooks 6.0.0, ruff v0.17.0, gitleaks v8.30.1, bandit 1.9.4,
shellcheck-py v0.11.0.1, hadolint v2.15.1 (CI binary too), conventional-pre-commit v4.4.0,
golangci-lint 1.64.8 (a v1 bump; v2 needs a config migration).

- **hadolint**: the hook is `language: system` and uses the binary on `PATH`. 2.14 passed what 2.15.1
  rejects (`DL3025`: a shell-form `HEALTHCHECK CMD`; fix by using the JSON form with the same
  `sh -c` command). CI downloads the pinned binary; put the pinned one first on `PATH` locally.
  Dockerfile `FROM` lines are pinned by multi-arch index digest (`image:tag@sha256:...`, tag kept so
  Dependabot still updates it); a test recomputes the closure of what the image `COPY`s.
- **shellcheck** at `--severity=warning`. Shell libraries that shellcheck cannot follow across files
  get a file-level `disable=SC2034,SC2154,SC2164` with a stated reason. Do not dedupe shell
  functions that only look alike (here `log_success` printed a different tag and `dc()` differed).
- **gitleaks** builds from source in CI, which needs a Go new enough for its `go.mod`.
- **conventional-pre-commit** at the `commit-msg` stage, scope optional, with
  `default_install_hook_types: [pre-commit, commit-msg]`. A merge commit still passes (proven with a
  real `git merge --no-ff` in a clone). The global `exclude` must not match `.git/`: pre-commit
  applies it to the message file `.git/COMMIT_EDITMSG` and silently drops the file argument.
- **Global exclude** narrowed to caches and binary engine files, so `models/`, `scripts/` and
  `export/` are linted.

### 2.7 CI jobs that mirror the hooks

- `pytest`: install CPU torch first (`pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision`),
  then `requirements-test.txt`; `python -c "import src.main"` as an import sanity step; the coverage
  floor above.
- `pre-commit`: `pre-commit run --all-files --show-diff-on-failure`, with caches for the
  pre-commit environments (keyed on the config hash) and `.mypy_cache`.
- `changelog`: a change under `src/ scripts/ docker/ export/ models/` requires `CHANGELOG.md`;
  escape hatch `[skip-changelog]` in the last commit message. The workflow's `run:` block is
  executed by a test against throwaway git repos (changes with and without the entry, with the
  escape, outside those trees).
- Frontend: `npm run check`, `npm run lint` (prettier and eslint), `npx knip` (blocking), `npm test`,
  a build step that fails if the output contains the word "deprecated", `py_compile` of the e2e
  harness, and the stubbed Playwright suite. A separate scheduled mutation workflow for the frontend.

## 3. The guards

All guards live in `scripts/codegen/`, share one baseline module, run as pre-commit hooks with
`pass_filenames: false` (they scan the whole tracked tree through `git ls-files`) and have unit
tests in `tests/codegen/`. Every guard's tests cover: a planted violation fails, clean input passes,
a stale baseline entry fails, a loosened entry fails and `--tighten` repairs it, a syntax error in a
scanned file fails closed, and the real repo passes.

### 3.1 Baseline format and the `--tighten` workflow

`scripts/codegen/baselines/<guard>.txt`, one entry per line:

```
key<TAB>value<TAB>reason
```

`value` is an integer (line count, complexity score, handler count, hit count). Blank lines and `#`
comments are ignored; a line with no reason, a duplicate key or a non-integer value is a parse
error. The comparison rule:

| Situation | Result |
|---|---|
| finding not in the baseline | fail: new violation |
| value grew past the recorded value | fail |
| value is lower than recorded | fail: loosened, run `--tighten` |
| entry with no finding left | fail: stale, run `--tighten` or delete the line |

Workflow: fix something, run the guard with `--tighten`, commit the baseline change with the code.
`--tighten` only lowers values and drops lines; it never adds a line and never raises a value.
After every merge that touched a baseline, re-run it (conflict rule in 4.4).

```bash
.venv/bin/python scripts/codegen/check_file_size.py --tighten
.venv/bin/python scripts/codegen/check_ruff_ratchet.py --tighten    # complexity and broad-except
.venv/bin/python scripts/codegen/check_vulture.py --tighten
.venv/bin/python scripts/codegen/check_process_ids.py --tighten
```

**Audit with an empty baseline.** A baseline can hide hits. To see the true state of a guard, point it
at an empty file: `check_process_ids.py --baseline /tmp/empty_baseline.txt`. A first round of
agents reported "clean" while the committed baseline hid 48 files; a second round driven by the
empty-baseline run finished the job.

### 3.2 Each guard

For every guard: what it checks, the baseline, and the plant-a-violation proof (plant, observe the
failure through the real hook, remove). The plants were run with `pre-commit run <id> --all-files`.

| Guard (script) | Checks | Baseline | Plant and observed result |
|---|---|---|---|
| File size (`check_file_size.py`) | 700 lines per first-party file (`.py .ts .svelte .js .mjs .sh`) in `src scripts export docker models tests frontend/src frontend/e2e frontend/scripts` plus root shell programs; every `CLAUDE.md` capped at 400; the 640 to 700 band is printed as a warning only; vendored bundles out of scope | `file_size.txt`: `path`, current lines, reason naming the split | 701 lines in `export/_probe.py`, `frontend/src/lib/_probe.ts`, `tests/_probe.py`: failed for each (none was in scope before). One extra line added to a baselined 739-line file: failed |
| Layering (`check_layering.py`) | AST walk of every `import` and `from ... import`, module level and inside functions (lazy imports hide the cycles). Routers may import services and clients; services, clients, `core`, `config`, `domain`, `utils`, `schemas` and `scripts` must not import routers; clients must not import services; everything below routers must not import `fastapi`/`starlette`; `config`, `domain`, `utils`, `schemas` must not import services or clients | none: starts with a baseline, ended at zero; the guard then **deleted its baseline file**. Permanent exceptions are a dict in the script with a reason; an exception whose import is gone fails as stale | function-level `from src.routers.curation import _common` in a service: `src/services/_probe.py -> src.routers.curation: 1 import(s) (new violation, not in the baseline)`; later four plants in one run (client importing a service, domain importing a service, service importing `fastapi` and a router) all reported |
| Complexity (`check_ruff_ratchet.py`) | `C901` at 15, ruff run with `--isolated` so a per-file ignore cannot hide a finding. Key `path::function` (`#2`, `#3` for repeats), value = McCabe score | `complexity.txt` | 16-branch function: `src/_probe.py::f: 17 CC (new violation, not in the baseline)` |
| Broad except (same script) | `BLE001` count per file, ruff isolated. A new handler, or a handler moved to a new file without moving its baseline line, fails | `broad_except.txt`: key `path`, value = count, reason `ratchet: may only go down` | extra `except Exception:` in a baselined file: `src/utils/cpu_pool.py: 2 handler(s) grew past the baseline 1` |
| Process ids (`check_process_ids.py`) | wave, round, work-package and review-finding ids and history names in comments and docstrings (Python via `tokenize` and `ast`; string data ignored; other languages by comment prefix) of code files and Dockerfiles; and in file names under `tests scripts frontend/src frontend/e2e` | `process_ids.txt` (ended **empty**); `path (file name)` keys for name hits | comment `# W8.2: ...` in `src/_probe.py` and a file `tests/test_r9_x.py`: `src/_probe.py: 1 hit(s)` and `tests/test_r9_x.py (file name): 1 hit(s)` |
| Dead code (`check_vulture.py`) | vulture at `--min-confidence 60` over `src scripts export docker` (tests are not scanned, so test-only code is reported: no production caller is dead). `--ignore-decorators` for route, validator, fixture, setter and app-factory decorators (`@application.*` was needed because the factory names its app `application`); class-body fields (pydantic, dataclass, enum members) dropped from the report. A reviewed false positive goes in `vulture_whitelist.py` with a reason per line | `vulture.txt` (ended empty, 0 bytes) | unused function: `src/_probe.py::function never_called: 1 finding(s) (new violation, not in the baseline)` |
| Naming leaks (`check_naming_leaks.py`) | three `git grep -P` scans of the tracked tree: A zero tolerance for private names and paths (pattern stored ROT13 so neither tree nor history spells them), B domain vocabulary generic code must not hardcode, C private class-registry names. Hits are filtered through `naming_leak_allowlist.txt` (reason per line) | allowlist; scan A accepts no directory-prefix exemption | a scan-A term in `examples/_probe.yaml`: `Scan A: 1 non-allowlisted hit(s)` (it had passed before because a directory entry covered scan A) |
| Version consistency (test) | docs name no release other than the current one; a regex lookahead that rejects only a following word character or dot-digit, so a version at a sentence end is caught; scope includes `SECURITY.md`, `CONTRIBUTING.md`, `docs-site`; plans and history (`docs/design`, `docs/research`, release notes, `CHANGELOG.md`) are out of scope because they name other versions by design | none | `Shipped in v0.9.9.` in a doc: `test_docs_name_only_the_current_release` fails |
| Config-path existence (`tests/test_tooling_config.py`) | every ruff `include` glob and per-file-ignore key matches a tracked file; every mypy override module resolves; every pre-commit path prefix exists. Found: a ruff include for a deleted tree, an override for a deleted module, hook prefixes for two directories that no longer existed (10 failures on the first run) | none | `"nope/**/*.py"` added to ruff `include`: `test_ruff_include_globs_match_a_tracked_file[nope/**/*.py]` fails |
| Docs vs code (`scripts/docs/check_docs_vs_code.py`) | docs may name only routes that exist in the contracts or app, environment variables read by code, links that resolve; a compose check (every `${VAR}` has an `env.template` assignment line; prose mentions do not count) and a template check (every template setting is read by something) | none | `${PLANTED_VAR}` in a compose file: `docker-compose.yml: ${PLANTED_VAR} is not in env.template`; `# PLANTED_SETTING=1` in the template: `env.template: PLANTED_SETTING is set by no compose file, script or code` |
| Noqa reason (hook) | see 2.2 | none | `x = 1  # noqa: E741`: `noqa carries a reason ... Failed` |
| Dockerfile hygiene (test) | every `FROM` pinned by digest; final stage sets `USER`; no `pip install` in a runtime stage; `.dockerignore` lists non-runtime trees; the `COPY scripts/` set equals the AST import closure of what the image runs | none | `FROM python:3.13-slim-trixie AS builder`: `is not pinned by digest`; deleting a needed `COPY scripts/datasets/`: `Dockerfile does not COPY: [...]` |
| Contract drift (hooks) | generated OpenAPI and TypeScript/JSON contracts are in sync with the code; regenerate with `make contracts`; the generated files are committed in the same commit | none | pre-commit hook rejects stale contracts |
| Frontend knip (`npx knip`) | unused files, exports and dependencies; `knip.json` with value exports only (types not reported), `ignoreExportsUsedInFile` off | none | `export const plantedUnused = 1;` appended: `Unused exports (1) plantedUnused ...`, exit 1 |
| Frontend scan tests (vitest) | source-scan tests that fail on a regression pattern: no local duration/bytes/time formatter copies (`format.scan`), no hand-rolled dialog (`modal.scan`), API prefix and endpoint catalog scans, no per-page thumbnail URL building. Each has a negative-control test that the pattern matches each spelling | none | failing outputs on the old code were the proof: `expected [ ...(9) ] to deeply equal []` (9 local copies), `(87)` hand-rolled dialog lines in 28 files |
| Pytest timeout | `--timeout` kills a hung test | n/a | a test sleeping 5 s with `--timeout=1`: `Failed: Timeout (>1.0s) from pytest-timeout.` |
| Changelog CI step | see 2.7 | n/a | change under `src/` without `CHANGELOG.md` exits 1; with it, with the escape, or outside those trees exits 0 |
| Commit-msg | see 2.6 | n/a | `git commit -m "bad"` rejected; `chore: ok` accepted; a `--no-ff` merge commit accepted |

How to read a failure: the message is `key: value unit (reason)`. "new violation, not in the
baseline" means fix the code (never add the line); "grew past the baseline" means you made it
worse; "is below the baseline ... --tighten" means good news, lower the file; "stale baseline
entry" means delete the line or run `--tighten`.

## 4. Port checklist for the sister repo

Order matters: it is the order that avoided merge conflicts here. Guards first so everything after
is held to them; mechanical and deletion work before anything that moves code; typing last. Each
step is one PR from fresh `main` after the previous one merged, except step 8, which fans out.

Effort figures are the cost observed here (about 800 Python files, 450 frontend files, one
supervising agent plus implementer agents). Scale them to the sister repo's size; CI per PR was
about 30 minutes.

### 4.1 Measure first (no code changes; half a day)

```bash
.venv/bin/pip install ruff mypy vulture bandit radon
.venv/bin/ruff check --isolated --select C901 --config 'lint.mccabe.max-complexity=15' src | tail
.venv/bin/ruff check --isolated --select BLE001,S110,S112 src | tail
.venv/bin/vulture src scripts --min-confidence 60 | wc -l
.venv/bin/mypy src            # from the project venv with real dependencies, never an isolated env
.venv/bin/bandit -r src scripts -c pyproject.toml
git ls-files '*.py' '*.ts' '*.svelte' | xargs wc -l | sort -nr | head -60      # files over 700
cd frontend && npx knip && cd ..
```

Record the counts; they are the "before" column. Check which directories are outside each tool's
`include`/`files`/`exclude` (a directory that no tool sees is the first finding).

### 4.2 Step 1: guards and config (the biggest PR; 2 to 3 days)

1. Bump hook revs in one commit (all but ruff), then ruff alone in a second commit with its fixes and
   formatting churn. If more than 300 files reformat, defer the bump and ask first.
2. Config fixes: apply section 2 (ruff, mypy local hook, bandit scope, pytest timeout, coverage in CI
   only, noqa reason hook, conventional commits, narrower global exclude). Add the config-path test
   so stale entries cannot return.
3. Copy `scripts/codegen/_baseline.py` and the guards you want (3.2), their `tests/codegen/` tests, and
   generate the baselines from today's findings with a reason per line. Verify each guard with a
   plant (section 3 recipes) and paste the observed output in the PR description.
4. Fix mypy errors found by the real-dependency run, and bandit High/Medium findings.
5. Test-first: run the new tests against the old tree and record the failures.

Done when `.venv/bin/pre-commit run --all-files` is green and every baseline exists.

### 4.3 Steps 2 to 11 in order

| # | PR | Content | Effort here | Expected result |
|---|---|---|---|---|
| 2 | dead code and comments | vulture findings fixed-point (delete what a deletion exposes), shims removed, mechanical process-id strip, hand rewrite by tree in parallel, tests renamed by behavior | largest diff (835 files, +5.7k/-11.4k); 2 to 3 days | vulture baseline to near zero, process-id baseline to zero |
| 3 | fail-open | one failing-first test per site; nullable counts plus frontend "unknown" rendering; typed upstream error mapped to 503; broad-except triage by category | 1 to 2 days | broad-except baseline down (544 to 443 here), the rest categorized and reasoned |
| 4 | layering | routers thin, models to `schemas`, pure logic to `domain`, services stop raising `HTTPException` (a generic rejection exception handled centrally), clients stop importing services | 1 to 2 days | layering baseline file deleted |
| 5 | duplicates and hygiene | one env module plus the `TID251` ban, one timestamp helper, one OCC/retry loop, frontend shared Modal/format/SSE/poll modules, Dockerfile digest pins, `.dockerignore`, compose anchors | 2 days (backend, frontend and infra as parallel sub-PRs) | duplicates gone, env reads typed |
| 6 | splits (backend) | every backend file near the cap, then the top-complexity functions | 2 to 3 days with parallel agents | no source file over 600 lines; complexity entries halved |
| 7 | splits (tooling, shell, oversize tests) | export tools, side-car images, shell programs, test files | 2 days, parallel with step 8 | no file in scope over 700 |
| 8 | splits (frontend) | API client by domain, pages into controller plus components, big tests by capability | 2 days, parallel with step 7 (disjoint trees) | frontend baseline lines all deleted |
| 9 | docs | this document adapted | half a day | |
| 10 | (optional) perf and docs closeout | as the repo needs | | |
| 11 | typing ladder | section 2.3 settings, last | 2 to 4 days (see placeholder in 6) | strictness on globally |

Rules inside every step: conventional commits, one coherent group per commit, pre-commit green
before each commit, never `--no-verify`, `CHANGELOG.md` entry under Unreleased (comment-only
changes need one too, because the changelog CI step looks at the paths), run `--tighten` after
deleting code and commit the baseline with it, full offline suite once at the end.

Per-step order inside a split PR (this avoided red intermediate commits): characterization tests
first (golden digests, branch tests, mutation-checked), then code moved verbatim in one commit per
original file, importers/mock targets/docs repointed in the same commit, then decomposition of the
high-complexity functions, then `--tighten`.

Splitting rules that worked:

- Verify "moved verbatim" mechanically (compare line multisets, or an unchanged golden).
- Patch targets: after each move run `rg "patch\(['\"]<old.module>"` and repoint.
- Constants shared by the new modules and the script entry point go to a leaf module (a module cannot
  import from the script it runs as; `__main__` would load twice).
- Do not merge two helpers because they look alike; compare behavior first (two report classes had
  different fields and two batch ladders ended at different sizes).
- A shared fixture directory becomes a package with `conftest.py` and `_helpers.py`; an autouse
  fixture belongs in the subdirectory it applies to, not in the parent `conftest.py`.
- Keep test counts identical before and after and compare sorted test ids.
- A split file's logger name changes; docs that cite old line numbers are history, leave them.

### 4.4 Using agents token-efficiently

1. **Plan on disk, not in the conversation.** One Opus planning pass writes
   `docs/design/<topic>_plan.md` with file paths, exact renames, sequencing and verification
   commands. Every implementer loads the plan from disk; nobody re-derives or re-pastes it. The
   plan gets sections: ground rules, decisions with owner-confirm flags, PR sequence, one section per
   PR with task ids, risks, split designs.
2. **One agent per PR, each in its own worktree and branch** (never the main checkout):
   ```bash
   MAIN=<absolute path of the main checkout>; cd "$MAIN" && git fetch origin
   git worktree add "$MAIN-wt/<slug>" -b <branch> origin/main
   ln -s "$MAIN/.venv" "$MAIN-wt/<slug>/.venv"        # .venv is gitignored; hooks need it
   (cd "$MAIN-wt/<slug>/frontend" && npm ci)           # one npm ci per worktree
   ```
   Hooks are shared by all worktrees but must be installed once in the main checkout
   (`.venv/bin/pre-commit install --hook-type pre-commit --hook-type commit-msg`); several PRs here
   were committed without hooks and only had the full hook set run on the final tree.
3. **Brief template** (agent_brief): worktree path and "work only here"; the exact check command that
   decides done (an empty-baseline guard run, not the committed one); the rewrite rules with two or
   three examples; commands the agent may and may not run (for parallel agents in one tree:
   no commit, no stash, no checkout, `git mv` for renames, only the assigned paths); the targeted test
   command (never the full suite); a final report under 15 lines (paths done, hits before/after,
   renames, unresolved items with path:line). State the PR deadline and the sizes expected.
4. **Sonnet implements; Opus plans; Sonnet reviews.** Parallel sub-agents work on disjoint trees in
   a phase; phases run in order. Mechanical shapes go to a script first (the strip helper handled 559
   lines), judgment shapes to agents.
5. **Verify agent claims yourself**: re-run the guard with the empty baseline, `rg`, and the suite.
   "Pre-existing failure" and "cleaned up" are claims.
6. **Merge-when-green helper.** Waits for checks, refuses on any failure, merges with a merge commit
   only when the state is clean. The minimum-total guard matters (pitfall in section 5):
   ```bash
   #!/bin/bash
   # usage: merge_when_green.sh <pr>
   pr=$1
   for i in $(seq 1 120); do
     sleep 60
     out=$(gh pr checks "$pr" 2>&1)
     total=$(echo "$out" | wc -l)
     pend=$(echo "$out" | awk -F'\t' '$2=="pending"' | wc -l)
     fail=$(echo "$out" | awk -F'\t' '$2=="fail"' | wc -l)
     [ "$fail" -gt 0 ] && { echo "FAILED: $(echo "$out" | awk -F'\t' '$2=="fail"{print $1}' | tr '\n' ' ')"; exit 1; }
     [ "$pend" = 0 ] && [ "$total" -ge 5 ] && break      # 5 = this repo's number of checks
   done
   for k in 1 2 3 4 5 6; do
     st=$(gh pr view "$pr" --json mergeStateStatus -q .mergeStateStatus)
     [ "$st" = CLEAN ] && break; sleep 30
   done
   [ "$st" = CLEAN ] || { echo "NOT CLEAN: $st"; exit 2; }
   gh pr merge "$pr" --merge && echo "MERGED $pr"
   ```
   Set the `-ge` threshold to the repo's real check count; a watcher that sees 2 of 5 checks
   registered and none pending merges early.
7. **Conflict rules when stacked or parallel PRs meet:**
   - `CHANGELOG.md`: keep both sides.
   - A baseline file: keep both deletions, then re-run `--tighten` (never hand-edit values up).
   - A file that one side split and the other edited: take the split structure, then re-apply the
     other side's change in the new files (typically import changes; an AST codemod rewrites them).
   - A route table or import-order dependent file: see the route-order pitfall.
   - After resolving: pre-commit, targeted tests, `--tighten`.
8. After merge: `git worktree remove "$MAIN-wt/<slug>" && git branch -d <branch> && git fetch origin`.
   Merge with `--no-ff` merge commits (never squash) so each commit stays a documentation artifact.

## 5. Pitfalls actually hit

1. **Vacuous hooks.** mypy in the isolated `mirrors-mypy` environment treated numpy, fastapi and
   pydantic as `Any` and passed while 53+ errors existed. Rule: any hook that needs the app's
   dependencies runs from the project interpreter; prove it with a planted type error that only a real
   dependency can reveal. The same holds for ratchets that run ruff: run the pinned venv ruff.
2. **The global `exclude` ate the commit message.** pre-commit applies the global exclude to every
   file argument including `.git/COMMIT_EDITMSG`; a pattern matching `.git/` silently removed the
   file from the conventional-commit hook. Keep `.git/` out of the exclude.
3. **`ruff check --fix --select RUF100` deletes unrelated noqa comments** (see 2.1). Use `--fixable`.
4. **hadolint 2.15 rejects shell-form `HEALTHCHECK CMD`** (`DL3025`) that 2.14 accepted. Hooks run
   whatever binary is on `PATH`; pin and put the pinned one first. Converted to the JSON form.
5. **Hooks not installed in worktrees.** Hooks live in the shared git dir and must be installed once
   in the main checkout; PRs were committed unguarded until the full `--all-files` run at the end.
   Install first, check `ls .git/hooks` (or `git rev-parse --git-path hooks`).
6. **Baselines hide hits.** Agents reported clean while the committed baseline hid 48 files. Audit
   with an empty baseline file; treat agent output as a claim.
7. **Route order depended on import chains.** The route table order came from incidental imports
   (one router imported another, so the second registered first). Moving helpers during layering
   reordered 15 routes; the cross-project leak test caught it. Fix: pin the import positions with
   `# isort: skip` in the package `__init__`, and prove the table is identical by dumping every route
   (method, path, name, order) before and after (239 and later 400 routes identical).
8. **Vulture whitelist matches by name.** A whitelisted name hides a same-named dead symbol; run
   `git grep -w` before adding an entry. A same-named function in two modules also fooled vulture
   in the other direction (one was dead, the other live); delete after reading the call sites.
9. **Plan line numbers go stale.** Re-find by symbol; the plan's `_common` repoint was "109 import lines"
   and was 30 statements; a shim thought to have 3 users had 7. Plans name symbols, not lines.
10. **Comment-only changes still need changelog entries** because the CI step is path-based; a mechanical
    comment strip across 835 files needs one entry.
11. **A guard scope that quietly excluded production code** (`scripts/`, `export/`, `models/`). Also
    directory exemptions that disabled the zero-tolerance scan, config globs naming deleted trees,
    coverage floors CI never ran, and a version regex that missed a version at a sentence end.
12. **Frontend**: `knip` flags exports used only inside their own file (make them file-local; do not
    delete); a deprecated `kit.alias` needs `resolve.alias` plus `tsconfig` paths; prettier and
    eslint in CI differ from a quick local run; stubbed e2e tests can time out under CPU load (a
    45 s timeout passed alone and on re-run: prove a real race with a deterministic variant, as the
    per-box write queue fix did, not by rerunning 30 times); `svelte-kit sync` writes through a
    shared `node_modules` symlink, so use one `npm ci` per worktree or `check` reports false
    `./$types` errors; moving code into `.svelte.ts` triggers `prefer-svelte-reactivity` on `new Set`.
13. **GitHub "Closes" keyword closes only the first issue in a list.** `Closes #1, #2` closes `#1`.
    Write one `Closes #N` per line.
14. **CI watcher minimum check count.** A watcher that merges when "nothing is pending" merges early if
    checks have not registered yet. Require a minimum total (section 4.4).
15. **A moved logger or module name** breaks `unittest.mock.patch('module.symbol')` targets and
    source-scan tests that read file text; repoint them to the file that holds the fact, do not
    weaken them.
16. **Two-sided `Optional`**: making a count nullable needs the contract regenerated and the frontend
    type updated in the same commit, or the type check fails on one side.
17. **Fakes that lack an attribute** made a narrowed handler stop catching; narrowing there needed the
    fakes fixed first (a handler was left broad and noted).
18. **Compose cannot refuse a set variable natively** (`${A:+${B:?}}` evaluates the inner expression
    eagerly), so the retired-name refusal lives in the CLI, the installer and the container
    entrypoint, not in compose.

## 6. Results

Before is the first guards commit (`d4e44556`, the day the baselines were generated from the old
tree) unless noted; after is the tip of the merged series.

| Metric | Before | After |
|---|---|---|
| Files over 700 lines (first-party scope) | 54 (33 export/docker/shell/tests, 19 frontend, 1 reasoned, 1 trim) | 0; one reasoned baseline entry remains (a single-function installer, 3,214 lines) |
| Largest `src/` or `scripts/` file | 698 (the largest frontend files were 5,766 and 3,243 lines) | 595 |
| Files in the 640-700 band (src, scripts, frontend src) | n/a | 10 (all frontend and tooling files noted for the next touch) |
| `file_size.txt` lines | 54 | 1 |
| Layering baseline | 64 keys, 81 imports (23 router imports from lower layers, 22 client-to-service imports, 7 files importing `fastapi`) | baseline file deleted, 0 findings, 1 permanent reasoned exception |
| Complexity (`C901` above 15) entries | 65 functions | 33 (32 removed: 23 in the backend split, 1 in fail-open, 4 in layering, 1 in duplicates, 3 in tooling; the 33 left are ratchets) |
| Broad `except` handlers (`BLE001`) | 544 in 222 files | 438 in 230 files (more files because splits spread handlers; no handler was added) |
| Dead-code (vulture) findings | 214 | 0 (empty baseline; 73 reviewed whitelist names with a reason each when it landed) |
| Process ids / history names | 692 keys, 1,734 hits | 0 (empty baseline) |
| mypy errors (four trees, real dependencies) | 175 (77 after stale ignores dropped); invisible before | 0 |
| Raw environment reads in `src` | 173 reads (170 migrated across 70 files), 6 private parsers | 0 (one typed module; one reasoned `noqa`) |
| Private timestamp helpers | 21 copies | 1 |
| Bandit scope / findings | `src` only; 2 High, 18 Medium in newly scanned trees | `src scripts export docker`; 0 unreasoned |
| Coverage floor | 51, never enforced | 80 enforced in CI (measured 84.95) |
| Offline backend tests | 8,179 passed (guards PR) | 8,678 passed, 19 skipped |
| Frontend vitest tests | 4,575 (4,604 after hygiene) | 4,750 |
| Image size, `evaluator` | 14.82 GB | 14.35 GB (multi-stage, -470 MB) |
| Dockerfile `FROM` pinned by digest | 0 of 9 | all |
| Typing strictness (final step) | `disallow_untyped_defs` false, `warn_return_any` false, nine-code relax override on 13 module patterns | **PLACEHOLDER, filled by the typing PR: final mypy error count, remaining ignores, override list** |

Per-guard proof outputs are in the table in 3.2. Cost notes: pre-commit full run with mypy cold was
minutes and warm seconds; the full offline suite 10 to 11 minutes at `-n 6`; the guards themselves
add seconds because they use `git ls-files` and a single ruff pass.

## 7. What is not portable, and what to decide per repo

Decide before starting; each is an owner decision here (marked OC in the plan).

- **Layer names and the permanent exception.** The routers, services, clients, domain/config/schemas/utils
  split and the one reasoned exception (the OpenSearch client factory wrapping every client in
  project isolation) are this repo's. Write the sister repo's contract table first.
- **Size ceiling.** 700 lines for code and tests, 400 for `CLAUDE.md`, 640 warn band. Pick numbers the
  team accepts; the principle is one cap, one baseline, split by concern.
- **Complexity cap 15 and the broad-except ratchet.** Calibrate to the baseline; do not let the first
  number be an aspiration that forces a mass ignore.
- **Quote style** (single quotes), line length 100: this repo's formatter choices, not requirements.
- **Env module and `TID251`.** Fail-closed environment parsing is a behavior change (a malformed value now
  refuses startup instead of defaulting); note it in the changelog.
- **Coverage floor** depends on what the CI runner can install; measure on CI.
- **Retired names.** A rename of public environment variables is breaking; this repo chose a startup
  refusal that names the new variable. Others may prefer a deprecation window.
- **Docs gate and naming-leak scan** assume a public repo with private origins; drop scan A if there are
  no private names to protect. The ROT13 trick keeps the pattern itself out of the tree.
- **Frontend**: this repo is a single SvelteKit app with a shared size gate; the Modal/format/SSE/poll
  modules are app-specific. Keep the idea (one home for each concern, enforced by a scan test and knip),
  not the file names.
- **Docker digest pins and multi-stage images** need a bump procedure documented in the release
  checklist, and a check that each digest is a multi-arch index.
- **GPU, Triton and TensorRT specifics** (engine builds, GPU placement) are irrelevant to a repo
  without them.
- **Agents.** The Opus-plans/Sonnet-implements split and parallel worktrees assume cheap parallel CI;
  a repo with slow or scarce CI should batch more per PR and merge less often.

Related: [CONTRIBUTING.md](../../CONTRIBUTING.md) (day-to-day rules and the guard table),
[repo_cleanup_plan.md](repo_cleanup_plan.md) (the plan this series executed).
