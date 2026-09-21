# OpenProcessor `main` completion plan — from "ported" to "shippable"

Status: **DRAFT — awaiting owner decisions D1–D9 (§7). Waves 0, 1, 5 and 7
are unblocked and can start immediately; Waves 2, 3, 4 and 6 each depend on
at least one decision.**

Repo: public `origin/main` (`github.com/davidamacey/OpenProcessor`).
Working tree for this plan: `/mnt/nvm/repos/wt-oss-hardening` on branch
`feat/oss-hardening` (identical to local `main`, verified `git diff main
HEAD` is empty).

> ## ⚠️ READ THIS FIRST — what this plan is and is not
>
> **This is not the genericization plan.** That work is *done*: a private,
> domain-specific (vehicle / licence-plate) curation backend was ported into a
> generic `curation` subsystem across 28 commits, merged locally as
> `1079933`. 217 files, 57,436 insertions, 811 tests green, pre-commit fully
> green. The design rationale for that port lives in a plan document on the
> private reference line, reachable as
> `/mnt/nvm/repos/wt-kb-readonly/docs/design/oss_genericization_phase2_plan.md`
> — referred to below as **the port plan**.
>
> **This plan covers what is LEFT** to make `main` a complete, trustworthy,
> genuinely generic OpenProcessor backend that a third party can clone,
> install, ingest into, infer with, curate with, and train from.
>
> **The single most important verified fact:** `main` is **29 commits ahead of
> `origin/main` and has never been pushed**. Nothing in the port is public
> yet. That is a feature, not a problem — it means every defect this plan
> lists can be fixed *before* first publication, and there is no deprecation
> burden on any rename.
>
> **Working-document notice.** This file contains absolute local paths
> (`/mnt/nvm/repos/...`) so a fresh implementation agent can execute it
> standalone. Those paths are listed in §A.3 and must be stripped or moved to
> an untracked location before the repo is published. See D1.

Revision history:
- **2026-09-20 (rev 1, this doc)** — written from four parallel read-only
  audits (port completeness, test coverage, live-stack verification,
  documentation/OSS readiness) plus direct verification by the planning
  agent. Every claim marked ✅ was re-checked first-hand against the working
  tree at `1079933` on 2026-09-20; claims marked ⚠️ are audit findings the
  planner could not independently confirm and that the implementing agent
  must re-verify before acting.

---

## 0. Verified state of `main` today

Everything in this section was checked first-hand on 2026-09-20 against
`/mnt/nvm/repos/wt-oss-hardening` at commit `1079933`. **Do not re-derive
it; do re-run the commands in §0.9 as your baseline before touching
anything.**

### 0.1 — What is green

| Gate | Result | How verified |
|---|---|---|
| `pre-commit run --all-files` | **27/27 hooks Passed, exit 0** ✅ | full run, 2026-09-20 |
| `python -c "import src.main"` | **silent, exit 0** ✅ | direct |
| `pytest tests/ -q --no-cov` | **811 passed, 4 skipped** ✅ | direct |
| Leak scan `rg -ni '<four proprietary tokens>' src/ scripts/ tests/ docs/ *.md *.yml *.toml` | **zero hits** ✅ | direct |
| `git ls-files \| grep -i '<reference namespace>'` | **zero files** ✅ | direct |
| `mypy`, `bandit`, `ruff`, `shellcheck`, `gitleaks`, `hadolint` | all Passed ✅ | in the pre-commit run |

The port's own leak discipline held completely **at commit `1079933`**.

> ### 🚨 REGRESSION DETECTED 2026-09-20, AFTER THIS PLAN WAS DRAFTED
>
> While this document was being written, concurrent work in the same
> worktree introduced **five new leak-scan hits**, reversing the clean
> result above. Verified ✅ at the time of writing:
>
> | Path | Hits | Tracked? |
> |---|---:|---|
> | `docs/design/oss_genericization_phase2_plan.md` | **341** | untracked |
> | `docs/design/cropwright_backend_integration_plan.md` | 10 | untracked |
> | `docs/design/curation_api_contract.md` | 2 | untracked |
> | `tests/integration/test_labeler_route_parity.py:65` | 1 | untracked |
> | `src/services/curation/strategy_registry.py:373` | 1 | **modified, will be committed** |
>
> The first of these is a **2,195-line near-verbatim copy of the private
> port plan**, placed in the public tree. The last two name a private
> module inside a source comment and a test string literal.
>
> **Do not commit or push any of these as they stand.** This is the exact
> failure mode the port plan's §7 R6 named as the only remaining exposure:
> a hand-copied file carrying an identifier that should have been
> genericized. Resolve it as the very first act of Wave 0 — it is also
> decision D1's subject matter, and the recommended answer there (author a
> *genericized* rationale doc rather than copying the private one) is
> precisely what avoids it.
>
> **Consequence for this plan:** §0 describes the tree at `1079933`. The
> working tree has since moved — ⚠️ 14 tracked files modified, 5 files
> added. **Re-run the §0.9 baseline and the §5 leak scan before starting
> Wave 0**, and reconcile any finding below that the concurrent work has
> already addressed (it appears to have touched `env.template`,
> `docker-compose.yml`, `docs/design/labeler_api_contract.md`,
> `src/config/region_fields.py` and `src/services/curation/strategy_registry.py`,
> which overlap Wave 0 and Wave 1). Note that as of this writing CFG-1 was
> **not** yet fixed — `strategy_registry.py:85,97` still read the `KB_*`
> flag names ✅.

### 0.2 — Scale of what landed

| Area | LOC | Note |
|---|---:|---|
| `src/{config,services/curation,services/detection,services/labeling,services/training,routers/curation}` | 27,136 ✅ | |
| `src/routers/curation_*.py` + `src/clients/{curation_opensearch,occ,pe_encoder}.py` | 4,152 ✅ | |
| `scripts/curation/` | 8,690 ✅ | **not measured by coverage — see §0.7** |
| `tests/curation/` (68 files) | 15,797 ✅ | |
| Total port diff vs `origin/main` | 217 files / **57,436 insertions** ✅ | `git diff --stat origin/main main` |

Route surface ✅ (`python -c "import src.main"` + route walk):

| Metric | Count |
|---|---:|
| Total routes on the app | **219** |
| Routes under `/curation` | **103** |
| **Write** routes under `/curation` (POST/PUT/PATCH/DELETE) | **47** |
| Write routes ever exercised against a live stack | **0** |

### 0.3 — The port's declared scope gaps, re-confirmed

The port plan's §1 declared a "Bucket B" of proprietary modules it would not
port. Re-reading each on the reference line confirms **four of six
classifications, and overturns two**:

| Reference module | LOC | Declared | Verdict on re-read |
|---|---:|---|---|
| domain ingest service | 1990 | proprietary | **~55% generic.** See §3.1. |
| domain YOLO exporter | 1143 | proprietary | **~350 LOC generic** (stratified split, dense class remap, image resize, atomic manifest). See §3.2. |
| domain single-class exporter | 893 | proprietary | **Correct** — two small generic helpers only (§3.2). |
| domain label importer | 444 | proprietary | **OVERTURNED — this is generic code with a domain name.** YOLO `.txt` parse, IoU match, bulk index, validated-flag flip. Its only domain coupling is two hardcoded index-name constants. ⚠️ |
| `src/config/plate_state.py` | 75 | proprietary | **OVERTURNED — a bare `str, Enum` state machine with zero domain logic.** Its omission left **52 hardcoded status-string literals** across ≥10 files on `main`, and two dangling doc references to a file that does not exist here (`src/config/region_fields.py:21-22`, `docs/design/labeler_api_contract.md:151`). ⚠️ |
| trained weights + the licensed image corpus | — | proprietary | **Correct, no action.** |

Nothing under `src/routers/curation*`, `src/services/curation/`,
`src/services/detection/`, `src/services/labeling/` or
`src/services/training/` was found truncated or stubbed relative to its
reference counterpart. `rg -n 'TODO|FIXME|XXX|HACK'` over the whole ported
surface returns **zero hits** ⚠️. The port is materially complete *for what
it claimed to port*.

### 0.4 — What was never ported and was never on any list: the `docker/` runtime companions ✅

This is the largest structural gap and it is not mentioned anywhere in the
port plan. The reference line carries a `docker/` tree of runtime companion
services. `main` has only `docker/hardened/`:

| Reference asset | LOC | What depends on it on `main` | Present on `main`? |
|---|---:|---|---|
| `docker/trainer/Dockerfile` + `kb_trainer.py` (2351) + `augment.py` (491) + `subset_dataset.py` (288) + `mlflow_callbacks.py` (357) | 3624 | **The entire `/curation/train/*` surface** — `src/routers/curation_train.py` (1466), `src/services/training/jobs.py` (925), `triton_promote.py` (801). The API's whole protocol with the trainer is writing `<job_id>.job.json` into a shared volume that nothing reads. | **NO** |
| `docker/sam3/Dockerfile` + `main.py` (754) | 816 | `scripts/curation/worker/client.py` posts to `/sam3/segment_plate`; `src/services/detection/cascade_detect.py:63` names `segmenter_name='sam3'`. The cascade's segmenter leg has no server. | **NO** |
| `docker/evaluator/Dockerfile` | 61 | `scripts/curation/bakeoff/` (~4,000 LOC, ported) is invoked as a container ENTRYPOINT that does not exist here. | **NO** |

Compounding this: ✅ `docker-compose.yml` on `main` declares **11 services**
(`triton-server`, `yolo-api`, `triton-sdk`, `node-exporter`,
`dcgm-exporter`, `prometheus`, `grafana`, `loki`, `alloy`, `opensearch`,
`opensearch-dashboards`) and **not one** of them runs any of the five ported
long-lived workers under `scripts/curation/`, nor a trainer, nor a
segmenter, nor a VLM. `rg -n 'curation|sam_worker|vlm_worker|auto_label|trainer'
docker-compose.yml Dockerfile Makefile` → **zero matches** ✅.

**Consequence:** roughly 12,000 LOC of ported asynchronous machinery
(workers + bakeoff + the training router) is unreachable in the shipped
artifact.

### 0.5 — Dependency defects that make the shipped image non-functional ✅

Four Python packages are imported by curation code and appear in **no**
dependency manifest — not `requirements.txt`, not `pyproject.toml`
`[project.dependencies]`, not the `Dockerfile` (which installs
`requirements.txt` only):

| Package | Import sites (verified) |
|---|---|
| `scikit-learn` | `src/services/curation/clustering/orchestrator.py:299,1113,1478`; `methods/ahc.py:72,120`; `clustering/backend.py:67`; `scripts/curation/bakeoff/sample.py:87` |
| `umap-learn` | `src/services/curation/clustering/embedding_reduce.py:499`; `clustering/backend.py:73`; `src/services/curation/embedding_viz.py:379` |
| `hdbscan` | `src/services/curation/clustering/methods/hdbscan.py:165` |
| `joblib` | `src/services/curation/clustering/embedding_reduce.py:417,425`; `src/services/curation/embedding_viz.py:391` |

Proof they are not transitive ✅: in the dev venv,
`pip show umap-learn` reports `Required-by:` **empty** — i.e. it was
hand-installed, not pulled by any declared dependency; `scikit-learn` is
`Required-by: pynndescent, umap-learn`, both themselves undeclared.
`hdbscan` is **not installed at all**, so the HDBSCAN clustering backend is
both undeployable *and* untested.

Because the imports are all *lazy* (inside functions), nothing fails at
import time — `import src.main` passes, the test suite passes in a venv that
happens to have them, and the shipped container would `ImportError` the
first time anyone calls `POST /curation/clusters/refine/{id}`.

Separately, `pyproject.toml` `[project.dependencies]` has drifted from
`requirements.txt` ✅: missing `structlog`, `tenacity`, `faiss-gpu-cu12`,
`imohash`, `open-clip-torch`, `ftfy`, `httpx`; and carries
`nvidia-dali-cuda120`, which `requirements.txt` does not.

### 0.6 — Config-layer defects ✅

| # | Defect | Evidence | Effect |
|---|---|---|---|
| **CFG-1** | Feature-flag env-var name drift | Router reads `OP_SEMANTIC_SEARCH_ENABLED` (`src/routers/curation/search.py:30`); the capability probe that feeds `GET /curation/methods` reads `KB_SEMANTIC_SEARCH_ENABLED` (`src/services/curation/strategy_registry.py:85`). Same split for viz: `src/routers/curation/viz.py:49` vs `strategy_registry.py:97`. | The UI is told a capability is disabled while the endpoint is enabled (or vice versa). **Both existing tests pass** — `tests/curation/test_search_router.py:45` sets `OP_*`, `tests/curation/test_methods_router.py:53-54,201` sets `KB_*`; neither crosses the seam. |
| **CFG-2** | Crop-cache path split | Writer/reader path comes from `CurationConfig.crop_cache_dir` (default `/dev/shm/openprocessor_crops`, env `OP_CROP_CACHE_DIR`) at `scripts/curation/worker/state.py:159`; but `src/routers/curation/vlm.py:188` reads `os.environ.get('GEMMA_CROP_CACHE_DIR', '/dev/shm/curation_crops')`. | Different var, different default → **100% cache miss out of the box**. The reference had one var and one default across all three sites. |
| **CFG-3** | 23 `KB_*` env vars survive on the public branch | `KB_BAKEOFF_{CONCURRENCY,GPUS,OUT_DIR}`, `KB_COREML_HOST`, `KB_CROP_DUP_THRESHOLD`, `KB_FIELD_COVERAGE_TTL_S`, `KB_PREFLIGHT_SCAN_CAP`, `KB_SCORES_{ENABLED,KNN_K,NPROBE,SHADOW,STATE_DIR}`, `KB_SELECT_{DIVERSE_ENABLED,JOB_MAX_N,JOBS_DIR,MAX_N,SYNC_MAX_OPS,CACHE_TTL_S}`, `KB_SEMANTIC_SEARCH_ENABLED`, `KB_TRAIN_{JOBS_DIR,RUNS_ROOT,STAGING}`, `KB_VIZ_PROJECTION_ENABLED` ✅ | The prefix is the private product's initials. Alongside 23 `OP_*` vars, the config surface has two conflicting conventions. |
| **CFG-4** | `env.template` documents **zero** curation vars | `rg 'KB_\|OP_\|curation' env.template` → no matches ✅; the file declares 15 vars, all infra ports/sizes. | |
| **CFG-5** | ⚠️ `.env` is wired to nothing | Audit reports `docker-compose.yml` has no `env_file:` and no `${VAR}` interpolation, and `Settings.Config` (`src/config/settings.py:156-159`) sets no `env_file`. **Re-verify before acting.** | If true, every operator-facing var is unreachable via `.env`. |
| **CFG-6** | Owner-private absolute paths shipped as defaults | ⚠️ `src/routers/curation/bakeoff.py:62-73` defaults to `/mnt/nvm/curation_train_data/...` and `/mnt/nvm/datasets/plates`; `scripts/curation/bakeoff/baselines.json` carries six `/mnt/nvm/...` weight paths; `scripts/curation/bakeoff/run.py:307`. One of these directories is the location of a proprietary licensed image corpus. **Re-verify, then fix regardless.** | |
| **CFG-7** | `src/config/detection_profile.py` ships domain-tuned values as *generic* defaults | ⚠️ `:36-37` aspect 1.2–8.0, `:38-41` text length 4–10, `:44` charset pattern, `:47` `segmenter_name='sam3'`, `:57` `ocr_rec_model='paddleocr_rec'` — **and no model directory of that name exists** (the real one is `paddleocr_rec_trt`), so `GET /curation/models/status` permanently reports a missing model. `DetectionProfile` has **no `from_env`**. | A new domain must edit source. |
| **CFG-8** | A domain-named field is baked into the *generic* index mapping | ⚠️ `src/clients/curation_opensearch.py:216` maps `'gemma_plate_visible': {'type': 'boolean'}`; `src/routers/curation/pipeline.py:499-500` writes it. Not covered by `RegionFields` (no matching attribute), and still carries the VLM vendor's name after the `gemma`→`vlm` rename. | |
| **CFG-9** | 25 `kb_*` Prometheus metric names exposed from the public build | `src/services/curation/metrics.py` — 25 metric-name string literals, incl. `kb_gemma_call_combined_count`, `kb_sam3_request_retries_total` ✅ | Externally visible naming; the port plan deferred this to "Phase 3", which is now. |

**Correction to an audit claim.** One audit reported that
`RegionFields.from_env()` is never called in production, making all 36
`OP_REGION_FIELD_*` vars dead. **That is false.**
`src/config/region_fields.py:133-136` builds the module singleton via
`RegionFields.from_env()`, and the planner proved it live ✅:

```
$ OP_REGION_FIELD_STATUS=zzz_custom python -c \
    "from src.config.region_fields import get_region_fields; print(get_region_fields().status)"
zzz_custom
```

Do not "fix" this. (The confusion is that `curation.py` had exactly this bug
and it was already fixed in `07d20fa`.)

### 0.7 — Test-suite defects ✅ / ⚠️

Baseline ⚠️ (from the coverage audit; re-run per §0.9): **811 passed, 4
skipped**; overall coverage **49.55%** with the default `--cov=src`.

Config-level, verified by the planner ✅:
- `[tool.coverage.run] source = ["src"]` → **`scripts/` is never
  measured**. That silently excludes `scripts/curation/worker/` (the entire
  detection worker) and `scripts/curation/bakeoff/` (~4,000 LOC, which no
  test imports at all). Measured explicitly, `worker/runner.py` is ⚠️ 37.7%.
- **No `fail_under`** anywhere — coverage can regress to zero without
  failing anything.
- `tests/conftest.py` has no shared fixtures at all: no OpenSearch double, no
  `TestClient` fixture. Only a 4-entry `collect_ignore` ✅.

Substantive findings ⚠️ (re-verify each before acting; the planner
independently confirmed the two marked ✅):

| # | Finding |
|---|---|
| **T-1** ✅ | **Three stale skips.** `tests/curation/test_select_router.py:232,243,258` all say `reason='GET /crops (src/routers/curation/crops.py) not ported yet -- Chunk 9'`. `src/routers/curation/crops.py` **exists** (27,592 bytes) and is at ⚠️ 29.20%. Free coverage on the worst-covered large router. |
| **T-2** | **Five `test_holdout` write-guard tests were silently deleted** from `test_write_guards.py` (reference 479 LOC/29 tests → target 376/22) with no commit note, while the `must_not` clauses they pinned survive in `src/services/curation/clustering/auto_promote.py:186`, `src/routers/curation/classes.py:256`, `src/services/curation/probe_predictions.py:389` **and `:547`** (the latter in uncovered code), and `src/routers/curation/vlm.py:342`. Deleting a guard today yields a green suite. |
| **T-3** | The dropped invariants suite was **not unfakeable**. `tests/curation/occ_fakes.py:96 FakeOccOpenSearch` and `tests/integration/test_ingest_occ.py:48 FakeUpsertOpenSearch` both implement real `_seq_no`/`_primary_term` conflict semantics. Per-test verdict: **4 of 7 fakeable today, 0 need a live stack, 3 obsolete.** Highest value: the class⊥region orthogonality invariant and the history-on-relabel OCC round trip. |
| **T-4** | **139 statements of ported detection code at exactly 0.00%**: `src/services/detection/pe_preprocess.py` (33), `ensemble_nms.py` (37), `region_lean.py` (69). All pure-numpy leaves. Plus `src/services/curation/clustering/outliers.py` (61) and `src/services/curation/autolabel/cli.py` (117) at 0.00%. |
| **T-5** | `scripts/codegen/check_no_literal_region_fields.py` accumulated subtle skip rules across 8 commits and **has zero behavioural tests**. `tests/curation/test_precommit_paths.py` only asserts hook registration and path existence. |
| **T-6** | Two collateral drops with live subjects: a `get_clustering_service` singleton test (subject `src/services/clustering.py:633`, ⚠️ 30.53%, **zero tests**, includes a fail-open error-swallow case) and a default-model-pointer grep guard (subject `src/config/settings.py:33`). Neither imports anything from the reference namespace. |
| **T-7** | Worst-covered ported surfaces: `src/routers/curation/pipeline.py` **5.08%** (224/236 missed, the largest curation router), `regions.py` **12.62%** + `regions_fp.py` **18.18%** (270 missed — the **entire human region-labelling write path**, knowingly ported with no tests because the reference had none), `src/services/curation/clustering/orchestrator.py` **31.92%** (465 missed, 1,769 LOC). |
| **T-8** | **Tautology pattern**: self-referential constant assertions — importing a constant from the module under test and asserting the module emitted it — at `tests/curation/test_clustering_orchestrator.py:131-133`, `test_methods_router.py:223-225`, `test_plate_sanity.py:101-116`, `test_curation_opensearch.py:228`, `test_review_router.py:95-114`. Also two assertion-free tests at `tests/curation/test_metrics.py:16,32`. |
| **T-9** | **Zero integration test crosses a router→service→OpenSearch→router round trip.** All 10 `tests/integration/*.py` are single-axis. This is precisely where the port's four post-hoc bugs hid (`8e485f1` dropped CORS, `07d20fa` silently ignored every `OP_*` override, `48e594a` hardcoded reference index names, `1e9ebff` ignored `api_prefix` in thumbnail URLs) — **all four passed the 811-test suite** and were found only by standing the thing up. |

### 0.8 — Documentation / OSS-readiness defects ✅

| # | Finding | Evidence |
|---|---|---|
| **D-1** ✅ | **42 references across 36 files point at a design doc that does not exist on `main`.** `docs/design/` contains only `labeler_api_contract.md`. Three are *live markdown links* that render broken: `docs/README.md:48`, `docs/ARCHITECTURE.md:87`, `CLAUDE.md:229`. Nineteen are in `src/` docstrings, twelve in `tests/`, one in `scripts/`, and **four are in `.pre-commit-config.yaml` itself** (`:123,147,167,183`), where they justify the ratchet exemptions. One is in a **pre-commit failure message** (`scripts/codegen/check_no_literal_region_fields.py:314`) that sends a failing developer to a nonexistent file. |
| **D-2** ✅ | `README.md` (526 lines) mentions curation **zero** times. So do `INSTALLATION.md` (465), `CHANGELOG.md`, `src/README.md`, `scripts/README.md`, `docs/opensearch_schema_design.md`. The only coverage is `docs/ARCHITECTURE.md:80-137` (a genuinely good component map) and a 28-line table at `CLAUDE.md:221-248`. **47% of the API is undocumented for a third party.** |
| **D-3** ✅ | **No CI runs tests or lint.** `.github/workflows/` contains exactly one file, `security-scan.yml`, with two Trivy jobs. 811 tests and 27 pre-commit hooks have zero enforcement. |
| **D-4** ✅ | **`make` has no curation targets** (`rg -i curation Makefile` → 0) and **no target runs pytest**. Worse, five documented test targets reference files that do not exist and never did on `origin/main`: `tests/test_inference.sh`, `scripts/test_integration.py`, `tests/test_end2end_patch.py`, `tests/test_onnx_end2end.py`, `tests/test_shared_vs_per_request.sh`. |
| **D-5** ✅ | **Version identity is four-way inconsistent**: `VERSION` = `0.2.1`, `pyproject.toml:6` = `0.1.0`, `README.md:40` example output = `0.2.0`, `CLAUDE.md:133` example output = `0.1.0`. |
| **D-6** ✅ | `CHANGELOG.md` has **no `[Unreleased]` section**; newest entry is `[0.2.1] - 2026-07-04`. The largest feature in the tree is unlogged. |
| **D-7** | ⚠️ `docs/design/labeler_api_contract.md` is titled for the private `/kb` prefix, and states at `:10-11` *"This repo does not (yet) ship an implementation of these routes"* — **false in this tree**, corrected only by an appendix 163 lines later. It also contains live cross-team coordination notes (`:159-169`) and points readers at a `/kb/*` table in `CLAUDE.md` that does not exist here. |
| **D-8** | ⚠️ `docs/security/*` documents a DeepStream investigation pinned to this machine's GPU slot 2. DeepStream is not part of the product. |
| **D-9** | ⚠️ Missing standard OSS furniture: `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`, `CODEOWNERS`, `dependabot.yml`. |
| **D-10** | ⚠️ **The API has no authentication of any kind** while exposing `DELETE /query/image/{id}`, `DELETE /curation/models/{model_name}` and `POST /ingest/directory` (arbitrary server-side path read). Grafana ships `admin/admin`; OpenSearch ships `DISABLE_SECURITY_PLUGIN=true`. Not currently stated anywhere. |
| **D-11** | ⚠️ `CLAUDE.md:120-121` asserts *"All API endpoints are available at both the root path and under the `/v1` prefix"*. 51 paths have a `/v1` twin; the 103 `/curation` paths do not. |
| **D-12** | ⚠️ `LICENSE:23-58` flags the vendored Ultralytics fork as AGPL-3.0 in an MIT-badged repo. |

### 0.9 — Baseline commands (run these first, record the output)

```bash
cd /mnt/nvm/repos/wt-oss-hardening
V=/mnt/nvm/repos/triton-api/.venv/bin     # there is no .venv in this worktree

$V/python -c "import src.main"                          # expect: silent
$V/python -m pytest tests/ -q --no-cov 2>&1 | tail -5   # expect: 811 passed, 4 skipped
$V/pre-commit run --all-files 2>&1 | tail -5            # expect: exit 0
git rev-list --left-right --count origin/main...main    # expect: 0  29
```

⚠️ `pyproject.toml` `addopts` forces `--cov=src --cov-report=html`; use
`--no-cov` for the inner loop and run coverage on once per wave.

---

## 1. Scope

### 1.1 In scope

Making `origin/main` a complete, trustworthy, generic OpenProcessor
backend — the owner's words: *"fully generic and set up for use by us and
other people to ingest, store, inference, and use as a backend… get all the
latest code merged to the main branch, working and tested, and if there are
missing tests to create them."*

Concretely, eight waves (§4): provenance and baseline hygiene; dependency
and config correctness; the generic ingest path; the export/training
artifact chain; runtime companion services; test restoration and real
coverage; a live write-path verification harness; documentation, OSS
furniture and CI. Then push (§4.9).

### 1.2 Explicitly OUT of scope

- **The ortloom inference engine.** Future work, separately planned in
  `/mnt/nvm/repos/ortloom/docs/design/architecture_plan.md` and in the
  labeler frontend repo's own `docs/design/ortloom_backend_integration_plan.md`,
  already correctly scoped by another team. **Do not design around it, do
  not add seams for it, do not mention it in shipped docs.**
- **Migrating the private production deployment onto this generic code.**
  That is the port plan's Appendix A — a separate future project with its own
  risk budget and its own production-change window. It may never happen.
  Nothing in this plan touches `/mnt/nvm/repos/triton-api`, the live
  checkout, or any running container named `yolo-api`, `triton-server`,
  `triton-opensearch`, or with the private product's prefix.
- **The labeler frontend's own genericization.** Separate project, separate
  repo. This plan's only interaction with it is: keep the HTTP wire contract
  working (§7 D3) and, in Wave 6, point the existing frontend at the test
  harness read-only.
- **Renaming Prometheus metric *names*** is in scope only as decision D4 —
  if the owner declines, it stays deferred and this plan records that.

### 1.3 Things this plan never does

- ❌ never runs `docker compose up/down/restart` against `docker-compose.yml`
  at repo root — its container names and host ports 4600–4610 collide with
  the live stack (verified: those containers are up right now).
- ❌ never mounts `/var/run/docker.sock` into any harness container.
- ❌ never writes to an OpenSearch index whose name lacks a harness-specific
  prefix.
- ❌ never commits to, merges into, or pushes the private reference branch.
- ❌ never force-pushes, squash-merges, or bypasses hooks/signing.

---

## 2. Delivery model

```
origin/main ──●(c287cd5)────────────────────────────────────────────●── origin/main
               \                                                   /
                └─ (already local) main ●…●─1079933                /
                                          \                       /
                                           └─ feat/oss-hardening ●─●─●─…  W0…W7
                                                                  (one branch)
```

**One worktree** (`/mnt/nvm/repos/wt-oss-hardening`), **one branch**
(`feat/oss-hardening`, already cut and already at `main`'s tip), waves
landing as sequential conventional commits. At the end, one `--no-ff` merge
into `main` and **one push of `main` to `origin`** — which is also the first
publication of the curation port (§4.9).

Per-wave procedure:

1. Implement the wave's file targets exactly as listed.
2. Write the wave's tests **in the same wave**. A wave is not complete
   without them.
3. Run the §5 gate.
4. Commit with the wave's stated conventional subject. Do not squash later —
   each wave commit is a documentation artifact.

### 2.1 Reading the reference line without writing its names into this file

**Read-only reference material** lives at `/mnt/nvm/repos/wt-kb-readonly`
(detached HEAD). Read from it freely; never `git add`, `git commit`,
`git mv` or `docker compose` there. Copy files **out** with `cp`/`git show`,
never `git mv`.

This document deliberately does **not** spell the reference line's private
product name or namespace directory, because the §5 gate greps `docs/` for
exactly those tokens and this file lives under `docs/`. Resolve them at the
shell instead — the reference namespace is the single non-generic package
under `src/services/`:

```bash
REF=/mnt/nvm/repos/wt-kb-readonly
ls "$REF/src/services/"          # one dir is the private namespace; that is $NS
NS=<that directory name>
ls "$REF/src/services/$NS/"      # the Wave 2/3 sources, identified below by LOC
ls "$REF/tests/$NS/"             # the Wave 5 test sources
ls "$REF/docker/"                # the Wave 3/4 runtime companions
```

Throughout Waves 2–5 the reference files are named by **role and verified
LOC** rather than filename (e.g. "the reference ingest service (1990 LOC)").
`wc -l "$REF/src/services/$NS"/*.py | sort -rn` disambiguates every one of
them unambiguously.

**Every file you copy out must be genericized before it is committed.** The
port recipe still applies: a ported file that still contains the private
product name, the private namespace prefix, a private NAS mount path, the
licensed corpus's name, or a raw `'plate_…'` OpenSearch-field literal is not
done.
`scripts/codegen/check_no_literal_region_fields.py` enforces the last one —
**add each newly-ported path to its `PORTED_PATHS` allowlist in the same
commit that ports it.**

---

## 3. The four substantive capability gaps

### 3.1 The ingest gap is structural, not cosmetic

The port plan's §7 R5 framed this as *"the public curation stack ships with a
thinner ingest path than the reference by design."* That framing is too
soft. Verified ✅: `src/routers/curation/ingest.py` (169 LOC) exposes only
`GET /ingest/status`, `GET /ingest/sam_drain` and `POST /ingest/path_lookup`
— three pure read queries. Its own docstring (`:1-16`) admits the four write
endpoints were dropped. **No route among the 103 can create an item**, and
⚠️ no service on `main` creates an item either: every `bulk()` call under
`src/services/curation/`, `src/services/detection/` and `scripts/curation/`
is an *update* path.

Three consequences prove this is structural:

1. **Four ported modules are dead on arrival** ⚠️ — each had exactly one
   production caller on the reference line, the ingest service:
   `src/services/detection/ensemble_nms.py` (112 LOC, **zero** importers,
   not even a test), `src/services/detection/pe_preprocess.py` (78 LOC,
   zero importers), `src/services/detection/crop_quality.py` (183 LOC,
   test-only), `src/clients/occ.py::occ_upsert_bulk` (~194 LOC, test-only).
2. **Fields declared in the index mapping and filtered on by routers are
   never written** ⚠️: `crop_area_norm`, `crop_rank_in_image`,
   `blur_lap_var`, `blur_lap_ratio` (mapped at
   `src/clients/curation_opensearch.py:245-248`; queried at
   `src/routers/curation/crops.py:99,107`, `clusters.py:101`,
   `regions.py:157`, `regions_fp.py:132`, `search.py:69`), plus
   `pe_embedding` and `coco_proposal_name`.
3. **The tmpfs crop cache is read and never written** — compounding CFG-2.

Mechanical decomposition of the reference ingest service (1990 LOC) ⚠️:
**≈1,100 LOC generic**, **≈700 LOC domain-specific**, ≈190 scaffolding. Its
*only* dependency on unported code is a single import of the domain label
importer — which §0.3 shows is itself generic. **The file is one import away
from being portable.**

### 3.2 The export/training artifact chain is broken end to end

Verified by the planner ✅: `src/services/curation/export.py`'s
`ARTIFACT_FILENAMES` (`:43-46`) declares `'class_registry': 'class_registry.json'`,
and `GET /curation/export/registry/{artifact}` serves it — but
`export_dataset` writes only `data.yaml` (`:259`), `label_stats.json`
(`:274`) and `manifest.json` (`:289`). **It never writes
`class_registry.json` at all.**

That in turn means **`export_id_map` has readers and no producer** ✅.
`rg -n 'export_id_map' src/ scripts/ tests/` shows readers at
`src/services/training/preflight_scan.py:94,168-178` and
`src/routers/curation_train.py:315-318` — and *every* writer is a
hand-written test fixture (`tests/curation/test_train_router.py:293,320,381`,
`test_train_preflight_scan.py:25`). Both readers **fail open** on a missing
map, so on any export this codebase actually produces, the registry-id →
dense-export-id translation silently no-ops and `include_classes` filtering
is **inert**. The existing tests hide this by supplying the fixture the
product never creates.

This is the single most consequential live bug found, and it is exactly what
the dropped `test_subset_dataset.py` existed to catch.

Also missing from the generic exporter versus the reference ⚠️: stratified
split (the generic one is an unstratified hash split, so rare classes can
land entirely in one split), even per-stratum sampling, **image copy/resize
entirely** (it writes labels but never copies pixels, so its `images/` dirs
are empty), atomic manifest write + atomic `current` symlink flip, code-SHA
lineage, and a `dedup_threshold` parameter that is accepted and
**silently ignored** (`src/services/curation/export.py:196`, with a `# noqa:
ARG002` admitting it) even though
`src/services/detection/frame_dedup.py::dedup_rows_by_embedding` is ported
and available.

### 3.3 The asynchronous half of the product cannot run

Per §0.4. Five long-lived workers, a trainer, a segmenter and an evaluator
exist as code with no container, no compose service, and no documentation.

### 3.4 Nothing has ever been verified against a live stack in a write path

47 write endpoints; zero exercised. The live-verification audit established
that this is far more tractable than assumed ⚠️:

| Cohort | Endpoints | Minimum infrastructure |
|---|---:|---|
| **(a)** pure OpenSearch | **32** | one OpenSearch container + seeded items + a writable registry + a writable state dir. **No Triton, no VLM, no GPU.** Includes `clusters/refine` and `clusters/auto_promote`, which run sklearn AHC / purity maths over *stored* embeddings and call no model. |
| **(b)** needs Triton | 3 | `pipeline/auto_label`(+`/start`), `DELETE models/{name}` |
| **(c)** needs a VLM | 4 | **any OpenAI-compatible `/v1/chat/completions`** — `src/services/labeling/vlm_client.py:43-45` reads `OPENWEBUI_BASE_URL`. A ~60-line fake suffices. |
| **(d)** needs the trainer | 7 | **a pure shared-volume file protocol** — `src/services/training/jobs.py:7-12`: API writes `<id>.job.json`, trainer writes `<id>.status.json`, cancel is a `<id>.cancel` sentinel. A ~40-line fake trainer covers 6 of 7. |
| **(e)** needs export dirs | 1 | a writable `OP_EXPORT_ROOT`. **Zero image files needed** — the exporter writes labels only (which is itself defect §3.2). |

**41 of 47 write endpoints are verifiable with no GPU.** The only true
GPU-only writes are `train/promote/{job_id}`, `DELETE models/{name}`,
`pipeline/auto_label`(+`/start`) and the evaluation half of `bakeoff/run`.

Two safety facts that make this safe to run on this host ⚠️:
- `AsyncTritonPool.initialize()` (`src/clients/triton_pool.py:170-199`) only
  constructs gRPC client objects; it never dials. A bogus `TRITON_URL` is
  harmless until an actual `infer()`.
- The GPU arbiter is a **no-op on `main`**: `GpuArbiterConfig`
  (`src/config/gpu_arbiter.py:33-44`) defaults `containers=()` and
  `allowed_gpu_ids=frozenset()`, `get_gpu_arbiter_config()` (`:56-66`)
  constructs a bare instance with **no env reading at all**, and
  `stop_gpu_services` (`src/services/training/gpu_arbiter.py:332-334`)
  returns `noop` before touching the docker SDK. `POST /curation/train/start`
  **cannot stop a container** even if the socket were mounted.

---

## 4. Waves

Each wave is one or more commits on `feat/oss-hardening`. Run the §5 gate
after every wave.

---

### Wave 0 — Provenance, baseline hygiene, free wins

**Blocked by:** D1. **Everything else in W0 is unblocked.**

**Goal:** stop `main` from pointing at things that do not exist, and collect
the zero-risk fixes.

**Files:**

| Action | Target | What |
|---|---|---|
| **DECISION D1** | `docs/design/curation_design_rationale.md` (new) **or** a scrub | Resolve the 42 dangling references (§0.8 D-1). Recommended: author a **new, genericized** rationale doc on `main` covering the three config dataclasses, the frozen-wire-contract split, the `RegionFields` indirection, the ratchet exemptions and the known gaps — then rewrite all 42 references to point at it. It must survive the §5 gate's leak scan, so it must not name the private product, its namespace prefix, its NAS mount path, or the licensed corpus. Do **not** copy the port plan verbatim. |
| EDIT | `docs/README.md:48`, `docs/ARCHITECTURE.md:87`, `CLAUDE.md:229` | repoint the three broken markdown links |
| EDIT | `.pre-commit-config.yaml:123,147,167,183` | repoint the four ratchet-justification comments |
| EDIT | `scripts/codegen/check_no_literal_region_fields.py:314` | repoint the failure message so a failing developer lands on a real file |
| EDIT | 19 `src/` + 12 `tests/` + 1 `scripts/` docstrings | mechanical repoint: `rg -l oss_genericization_phase2_plan` then sed |
| EDIT | `pyproject.toml:6` | `version = "0.1.0"` → `"0.2.1"` (match `VERSION`) |
| EDIT | `README.md:40`, `CLAUDE.md:133` | correct the stale example version strings |
| EDIT | `src/main.py:255`, `src/config/settings.py:112` | unify the API title to `OpenProcessor` (currently `'Visual AI API'` and `'Visual Search API'` — three names for one product) |
| DELETE | `src/main.py:271` | ⚠️ a comment naming the internal repo directory; verify then delete |
| EDIT | `CHANGELOG.md` | add `## [Unreleased]` with an `### Added` entry for the curation subsystem |
| EDIT | `tests/curation/test_select_router.py:232,243,258` | **remove the three stale `@pytest.mark.skip`** (T-1) — `src/routers/curation/crops.py` exists. If a test then fails, fix the test, not the skip. |
| EDIT | `Makefile` | fix or delete the five broken test targets (D-4): `test-inference`, `test-integration`, `test-patch`, `test-onnx`, `test-shared-client`. Add `test:` → `$(V)/python -m pytest tests/ -q`. |

**Tests:** re-running the suite must show **814 passed, 1 skipped** (the
three un-skipped tests now run). Add
`tests/test_doc_links.py::test_no_broken_relative_markdown_links` — walk
every `*.md` under the repo root and `docs/`, extract `](…​.md)` targets,
assert each resolves. This converts D-1's whole class into a test failure.

**Commits:**
- `docs(design): add the curation design rationale and repoint 42 dangling references`
- `chore(identity): reconcile version strings and API title`
- `test(curation): un-skip three stale diverse-order router tests`
- `build(make): repair broken test targets and add a pytest entrypoint`

---

### Wave 1 — Make the shipped artifact actually work

**Blocked by:** D2 (env-var rename scope). The dependency fixes are
unblocked and should land first as their own commit.

**Goal:** a `docker compose up` from a clean clone produces a container in
which the curation endpoints do not `ImportError`, and in which one config
name means one thing.

**W1.a — Dependencies (unblocked, do this first)**

| Action | Target | What |
|---|---|---|
| EDIT | `requirements.txt` | add `scikit-learn`, `umap-learn`, `hdbscan`, `joblib` with a comment naming the lazy import sites from §0.5 |
| EDIT | `pyproject.toml` `[project.dependencies]` | add the same four, plus the seven drifted-out packages (`structlog`, `tenacity`, `faiss-gpu-cu12`, `imohash`, `open-clip-torch`, `ftfy`, `httpx`); remove `nvidia-dali-cuda120` unless it is genuinely used (grep first) |
| NEW | `requirements-test.txt` | a CPU-installable subset sufficient to run `pytest tests/` on GitHub Actions — needed by Wave 7's CI job. `import src.main` pulls `torch` (`src/routers/health.py:19`), `cv2`, `ultralytics` (`src/clients/triton_client.py:35`) and `transformers` (`src/utils/cache.py:32`); `faiss`, `tensorrt` and `onnx` are lazy and can be omitted. Pin the CPU torch wheel index. |
| NEW | `tests/test_dependency_manifest.py` | assert `pyproject.toml` `[project.dependencies]` and `requirements.txt` agree (modulo a small documented allowlist), **and** that every top-level third-party module imported anywhere under `src/` appears in one of them. This is the test that would have caught §0.5. |

**W1.b — Config correctness**

| # | Action | Target | What |
|---|---|---|---|
| CFG-1 | EDIT | `src/services/curation/strategy_registry.py:85,97` | `KB_SEMANTIC_SEARCH_ENABLED` → `OP_SEMANTIC_SEARCH_ENABLED`, `KB_VIZ_PROJECTION_ENABLED` → `OP_VIZ_PROJECTION_ENABLED`, matching `src/routers/curation/search.py:30` and `viz.py:49`. Update `tests/curation/test_methods_router.py:53-54,201`. |
| CFG-1 | NEW test | `tests/curation/test_feature_flag_consistency.py` | For each gated capability, set the env var **once** and assert *both* the router's gate and `GET /curation/methods` agree. Parameterise over every flag so a future drift fails. |
| CFG-2 | EDIT | `src/routers/curation/vlm.py:188` | replace `os.environ.get('GEMMA_CROP_CACHE_DIR', '/dev/shm/curation_crops')` with `get_curation_config().crop_cache_dir`, matching `scripts/curation/worker/state.py:159`. Add a test asserting the router and the worker resolve to the **same** path under a given `OP_CROP_CACHE_DIR`. |
| CFG-3 | EDIT | **DECISION D2** | Rename the 23 `KB_*` vars to `OP_*`. Since nothing is published, recommend a **hard rename with no back-compat shim** (per the owner's standing preference against shims for code removed in the same session). If D2 says otherwise, implement a dual-read with a `DeprecationWarning`. |
| CFG-6 | EDIT | `src/routers/curation/bakeoff.py:62-73`, `scripts/curation/bakeoff/baselines.json`, `scripts/curation/bakeoff/run.py:307` | ⚠️ verify, then replace every `/mnt/nvm/...` default with a config-derived path under `CurationConfig`. **One of these paths is the location of a licensed proprietary corpus and must not ship.** Make `baselines.json` carry relative paths or a documented placeholder. |
| CFG-7 | EDIT | `src/config/detection_profile.py:57` | ⚠️ `ocr_rec_model='paddleocr_rec'` → `'paddleocr_rec_trt'` (verify the real model dir name first). |
| CFG-7 | NEW | `src/config/detection_profile.py` | add a `from_env(prefix='OP_DETECTION_')` classmethod mirroring `CurationConfig.from_env` / `RegionFields.from_env`, so a new domain configures rather than forks. Add `tests/curation/test_detection_profile.py::test_from_env_overrides_every_field`. |
| CFG-8 | EDIT | ⚠️ `src/clients/curation_opensearch.py:216`, `src/routers/curation/pipeline.py:499-500` | rename the domain-named boolean in the *generic* mapping. Add it as a proper `RegionFields` attribute (e.g. `visible`) so the name is configurable like every other region field, and add it to `tests/curation/test_region_fields_mapping_coverage.py`. |
| CFG-4/5 | EDIT | `env.template`, `docker-compose.yml`, `src/config/settings.py` | ⚠️ **verify CFG-5 first.** If `.env` really reaches nothing, add `env_file: .env` to the `yolo-api` service and `env_file` to `Settings.Config`. Then rewrite `env.template`: delete the entries no code reads, add a curation tier (`OP_*`, `OPENWEBUI_*`, segmenter URL, worker flags). |
| CFG-4 | NEW test | `tests/test_env_surface.py` | assert every `os.environ.get('OP_…')` / `getenv('OP_…')` literal under `src/` and `scripts/` appears in `env.template`, and vice versa. Allowlist genuinely internal vars explicitly. |
| — | EDIT | `docker-compose.yml` | add a `./data` bind mount and a state-dir volume to `yolo-api` — `CurationConfig` defaults to `./data/{images,exports}`, `./data/class_registry.json` and `/var/lib/openprocessor`, and ⚠️ `data/` does not exist in the tree, `/var/lib/openprocessor` is root-owned while `Dockerfile:105` runs `USER appuser`. Without this, **all curation state dies with the container.** |
| — | NEW | `data/class_registry.example.json` + `config_templates/` entry | a worked, **non-vehicle** example registry (see D7) so a new user has something to copy. |

**Commits:**
- `build(deps): declare scikit-learn, umap-learn, hdbscan and joblib; reconcile pyproject with requirements`
- `fix(config): unify feature-flag and crop-cache env vars across router and service layers`
- `refactor(config): rename KB_* environment variables to OP_*`
- `fix(config): remove private absolute-path defaults from the bakeoff harness`
- `feat(config): make DetectionProfile env-configurable; document the curation env surface`

---

### Wave 2 — Close the ingest gap

**Blocked by:** D5 (does the cascade's segmenter leg ship?) only for the
optional cascade integration; the core ingest path is unblocked.

**Goal:** `POST /curation/ingest/image` and `/ingest/batch` exist, create
items, and in doing so revive the four dead modules from §3.1.

**Read from:** `$REF/src/services/$NS/` — the reference ingest service (1990
LOC) and the reference label importer (444 LOC). See §2.1 for how to resolve
`$REF` and `$NS` without writing the private names into this file.
Genericize as you copy. **Do not port the
COCO-vehicle class allowlist, the dual-head vehicle detector runner, the
region-status assignment policy, or the mismatch-report sink** — those are
the ≈700 domain-specific LOC and belong in a future overlay.

**New files:**

| Target | Source (reference) | Contents |
|---|---|---|
| `src/services/curation/label_import.py` | domain label importer (444 LOC) | near-verbatim port. `_parse_yolo_txt`, `_iou`, `_label_id`, `_crop_id_for`, `_lookup_image`, `_lookup_existing_crops`, `import_yolo_labels(..., label_source: str)`, `import_labels_batch`. Index names from `CurationConfig`; history via the already-ported `src/services/curation/history.py`. **`_crop_id_for` must produce byte-identical ids to the ingest service's `_crop_id`** — assert that in a test. |
| `src/services/detection/geometry.py` | ingest service `:396-547` | `bbox_norm`, `crop_id`, `iou`, `letterbox_to_square`, `undo_letterbox`, `crop_to_jpeg`, `roi_pool_sppf`. Shared with `cascade_detect.py`; refactor its private copies to import from here. |
| `src/services/curation/ingest.py` | ingest service, generic half | `CurationIngestService`, parameterized by `CurationConfig` + `DetectionProfile`. Ports `_check_duplicate`, `_check_duplicates_msearch` (batched msearch dedup with per-image fallback), `_decode_image`, `_ingest_passes_gate`, `_write_crop_cache`, `_bulk_index` (**calls the already-ported `occ_upsert_bulk` with human-field guards**), `ingest_one`, `ingest_batch` (batched decode → batched detector calls → semaphore-bounded gather, with a clean per-image fallback). Must compute and write `crop_area_norm`, `crop_rank_in_image`, `blur_lap_var`, `blur_lap_ratio` via the already-ported `src/services/detection/crop_quality.py`, and `pe_embedding` via the extended encoder below. **This file will exceed 700 LOC; split it before committing, or it needs a ratchet entry — prefer splitting `ingest_one`'s doc-building into a `src/services/curation/item_doc.py`.** |
| `src/services/curation/clustering/ivf_ingest.py` | ingest service `:165-229` | mtime-keyed in-process IVF centroid cache that hot-reloads when the worker retrains, plus the ingest quality gate. |
| `scripts/curation/ingest_walker.py` + `scripts/curation/_fast_walk.py` | reference one-off scripts (647 + 115 LOC) | parallel `os.scandir` walker → reader threads → bounded queue → concurrent `POST /curation/ingest/batch`, with a resume progress file. This is the bulk front door a new user needs. |

**Edited files:**

| Target | Change |
|---|---|
| `src/clients/pe_encoder.py` | add `embed_crops(...)` (max-batch-aware chunking + L2 normalize) and `embed_whole_frame(...)`. The latter finally gives `src/services/detection/pe_preprocess.py::whole_frame_chw` a production caller (currently 0 importers). |
| `src/services/curation/source_image_cache.py` | add the crop-cache **write** side. Currently the worker and `vlm.py` read a cache nothing populates. |
| `src/routers/curation/ingest.py` | add `POST /ingest/image`, `POST /ingest/batch`, `POST /import_labels`, `POST /import_labels/batch`. Reuse the wire models already present in `src/routers/curation/_common.py` (`IngestImageRequest`, `IngestImageResponse`, `BatchIngestResponse`, `ImportLabelsRequest`, `ImportLabelsBatchRequest`) — they were ported and are currently unused by any route. Replace the module docstring's "intentionally NOT ported" paragraph. |
| `src/services/detection/ensemble_nms.py` | no change, but `ingest.py` must **call** `apply_ensemble_nms` when a profile declares two detectors — that is what takes it off 0 importers. |
| `scripts/codegen/check_no_literal_region_fields.py` | append every new path to `PORTED_PATHS` in the same commit |
| `docs/design/labeler_api_contract.md` | delete the "no equivalent under `/curation`" claim for the four ingest routes |

**Tests (all new, under `tests/curation/`):**

- `test_label_import.py` — YOLO `.txt` parse incl. out-of-range and deprecated class rejection; IoU best-match against existing items; new-item creation on no match; `label_validated` flip; **`_crop_id_for` == ingest `crop_id` for the same (path, bbox)**.
- `test_ingest_service.py` — duplicate detection (single and batched msearch, plus the msearch-failure fallback); the quality gate; `occ_upsert_bulk` invoked **with** human-field guards (extend `tests/curation/occ_fakes.py`); crop-cache write; `crop_area_norm`/`crop_rank_in_image`/`blur_*` actually present on the produced doc.
- `test_geometry.py` — letterbox/undo-letterbox round trip to sub-pixel tolerance; `crop_id` determinism; `iou` edge cases.
- `test_pe_preprocess.py` — **restore the three dropped preprocessing cases** (T-4): output shape `(3,336,336)` and dtype `float32`; ImageNet-normalized mean ≈ 0; **zero-size crop → all zeros**.
- `test_ensemble_nms.py` — suppression at the IoU boundary, class-awareness, empty input.
- `tests/integration/test_ingest_roundtrip.py` (`@pytest.mark.integration`) — ingest two images through the router against a fake OpenSearch, then `GET /curation/crops` and assert the items are listed with the quality fields populated, and `GET /curation/ingest/status` reflects them.

**Commits:**
- `feat(curation): add generic YOLO label import`
- `refactor(detection): extract shared crop/bbox geometry helpers`
- `feat(curation): add the generic curation ingest service`
- `feat(curation): expose ingest and label-import endpoints`
- `feat(curation): add a bulk directory ingest walker`

---

### Wave 3 — Repair the export → training artifact chain

**Blocked by:** D6 (does the trainer container ship?) for the second half.
**The first half fixes a live bug and is unblocked.**

**W3.a — Export producer (unblocked, highest priority in this plan)**

| Target | Change |
|---|---|
| `src/services/curation/export.py` | **Write `class_registry.json` into the export dir**, including an `export_id_map` built by a new `_build_export_id_map` (registry id → contiguous dense id, skipping deprecated classes) and applied by `_remap_rows_to_export_ids` before labels are written. This closes §3.2. |
| same | Replace the unstratified `hash_split` with a **stratified** split (per class, and per an optional `group_key` so near-duplicate bursts sharing a `cluster_id` cannot straddle train/test — a general data-leakage guard). Keep the existing deterministic-seed contract. |
| same | Add **image copy/resize**: a `ProcessPoolExecutor` stage with `resize_mode: Literal['letterbox','aspect']`. Today `images/` dirs are written empty. |
| same | Wire the accepted-and-ignored `dedup_threshold` (`:196`) to `src/services/detection/frame_dedup.py::dedup_rows_by_embedding`, **or** raise `NotImplementedError` — silently ignoring it is the worst option. |
| same | Atomic manifest write + atomic `current` symlink flip; record a code SHA and the frozen-holdout checksum in the manifest. |

**Tests:**
- `tests/curation/test_export_service.py` (extend) — restore seven cases from the dropped reference export suite: `data.yaml` `nc` matches; frozen holdout rows land in the `test` split (an **advertised** behaviour at `export.py:21-23` with no test); stratification distributes classes; region-bbox conversion round trip; manifest structure + deterministic SHA; `current` symlink resolution happy path; **re-derivability from the manifest's recorded seed** (`export.py:93-107` literally claims this guarantee and nothing tests it).
- `tests/curation/test_export_id_map.py` (new) — dense-id map is contiguous from zero and skips deprecated classes; label `.txt` files use dense ids; the manifest records the remap and it is invertible.
- `tests/integration/test_export_preflight_roundtrip.py` (new, `@pytest.mark.integration`) — **the single highest-value test in this plan.** Run `GenericYoloExportService.export_dataset`, then feed the *produced* directory to `src/services/training/preflight_scan.py::scan_export` with `include_classes`. Assert the filter actually applies. **Write this test first and watch it fail** — it fails today.

**W3.b — Trainer container (blocked by D6)**

If D6 says ship: port `docker/trainer/` genericized from the reference
(3,624 LOC), stripping the vehicle taxonomy, the hardcoded licence-plate
class lookups (`_resolve_license_plate_registry_id`,
`_resolve_license_plate_export_id`), the private model-name preference list
and the `/mnt/nvm/...` run roots.

| Target | Source |
|---|---|
| `docker/trainer/Dockerfile` | reference `docker/trainer/Dockerfile` (137) |
| `docker/trainer/trainer.py` | reference `kb_trainer.py` (2351) — the watcher/runner loop over the `job.json` protocol |
| `docker/trainer/subset_dataset.py` | reference (288) — **generic already**; it is what `test_subset_dataset.py` existed to test |
| `docker/trainer/augment.py` | reference (491) — strip the class-conditional HFlip rule that keys on a licence-plate class id; parameterise it as "classes whose content is text-bearing" on the profile |
| `docker/trainer/mlflow_callbacks.py` | reference (357) — parameterise the experiment name |
| `docker-compose.yml` | add a `trainer` service **behind a compose profile** (see Wave 4) |

**Tests:** restore `test_subset_dataset.py` against
`docker/trainer/subset_dataset.py` (the reference splices `docker/trainer/`
onto `sys.path`; do the same), and restore the producer-side class-remap
propagation tests from the reference trainer-manifest suite — TARGET tests
the *consumer* (`tests/curation/test_triton_promote.py:295`) while nothing
produces or validates the artifact.

**Commits:**
- `fix(export): emit class_registry.json with a dense export_id_map`
- `feat(export): stratified, group-aware splits and image resize`
- `feat(trainer): add the generic training container`

---

### Wave 4 — Runtime companions and deployment

**Blocked by:** D5, D6, D7.

**Goal:** everything that exists as code has a documented way to run, and the
default `docker compose up` experience is unchanged for someone who only
wants the core detection API.

**Design principle — use compose profiles.** Add the curation services under
`profiles: [curation]` so `docker compose up -d` keeps starting the 11
services it starts today, and `docker compose --profile curation up -d`
starts the rest. This is how the curation subsystem becomes opt-in (D7)
without a code flag.

| Target | What |
|---|---|
| `docker-compose.yml` | add, all under `profiles: [curation]`: `curation-detection-worker` (`python -m scripts.curation.worker`), `curation-vlm-worker` (`scripts/curation/vlm_worker.py`), `curation-auto-label-worker`, `curation-cluster-refresh` (`scripts/curation/cluster_refresh_daemon.py`), plus `trainer` (D6) and `segmenter` (D5). Bind-mount `./src` and `./scripts` read-only, matching the existing `yolo-api` pattern. Give each a healthcheck. |
| `docker/segmenter/` | **D5.** If shipping: genericize the reference `docker/sam3/` (Dockerfile 62 + `main.py` 754) into a model-agnostic segmentation service with a `POST /segment` route, and rename `DetectionProfile.segmenter_name`'s default off a specific model. If not shipping: make the segmenter leg of `src/services/detection/cascade_detect.py` **explicitly optional** — a profile with `segmenter_url=None` must degrade cleanly, with a test proving it — and document that users supply their own. |
| `docker/evaluator/Dockerfile` | port the reference's (61 LOC), repointing the ENTRYPOINT to `scripts.curation.bakeoff.bakeoff_runner`. Without it, ~4,000 LOC of ported bakeoff harness has no runner. |
| `Makefile` | add `curation-up`, `curation-down`, `curation-logs`, `curation-seed`, `curation-status` (D-4) |
| `scripts/openprocessor.sh` | add the curation subcommands; also fix `cmd_test`'s `source .venv/bin/activate 2>/dev/null \|\| true` + bare `python`, which hides its own failure and contradicts this repo's stated venv rule |

**Tests:** `tests/test_compose_contract.py` — parse `docker-compose.yml` and
assert (a) every service `command:` that invokes a repo path points at a file
that exists, (b) every curation service carries `profiles: [curation]`, (c)
no two services share a `container_name` or host port. This is the compose
analogue of `test_precommit_paths.py` and converts a whole class of
silent-drift into a test failure.

**Commits:**
- `feat(deploy): add curation worker services behind a compose profile`
- `feat(deploy): add the segmentation and evaluator containers`
- `test(deploy): pin the compose service contract`

---

### Wave 5 — Test restoration and real coverage

**Blocked by:** nothing. Can run in parallel with Waves 2–4 if a second
agent is available, but the gate must be re-run after merging.

**Goal:** the suite would actually fail if the code were wrong.

**W5.a — Restore what was wrongly dropped**

| Target | What | Source |
|---|---|---|
| `tests/curation/test_write_guards.py` | **restore the five `test_holdout` write-guard tests** (T-2). Their subjects are all live: `src/services/curation/clustering/auto_promote.py:186`, `src/routers/curation/classes.py:256`, `src/services/curation/probe_predictions.py:389` **and `:547`**, `src/routers/curation/vlm.py:342`. **Watch each fail** by temporarily deleting its `must_not` clause before restoring it. | `$REF/tests/$NS/test_write_guards.py` (479 LOC / 29 tests) |
| `tests/integration/test_curation_invariants.py` (new) | restore **four of seven** invariants (T-3) using the existing doubles: (1) no silent overwrite via OCC — two concurrent `occ_update_one` with `max_retries=0` → exactly one success + one `OCCFinalConflictError`, which also covers the currently-uncovered `src/clients/occ.py:150-173`; (3) **class ⊥ region orthogonality** — a class-side write leaves region fields intact and vice versa (this is the invariant the whole `RegionFields` split depends on); (4) history preserved on relabel — `record_class_history` + two relabels, both prior class ids survive with `at` + `writer`; and a *rewritten* invariant 2 asserting no merger sets `class_validated=True` from a lone VLM signal. **Do not port invariants 5 and 6** — one targets a never-ported module, the other tests OpenSearch's own partial-update semantics. | the 636-LOC / 7-test invariants file under `$REF/tests/integration/`; doubles already on `main` at `tests/curation/occ_fakes.py:96` and `tests/integration/test_ingest_occ.py:48` |
| `tests/test_clustering_singleton.py` (new) | restore the collateral-dropped singleton test (T-6). Subject `src/services/clustering.py:633 get_clustering_service` / `:547 load_all_indexes` — ⚠️ 30.53%, **zero tests**, and its error-swallowing branch is a fail-open guard. This file imports nothing from the reference namespace; it was pure collateral damage. | reference `tests/test_kb_clustering_singleton.py` |
| `tests/test_default_model_pointer.py` (new) | restore the grep guard over the default detector model name; subject `src/config/settings.py:33`. Adjust the expected-count constant for the call sites that left with the unported modules. | reference `tests/test_active_model_pointer.py` |
| `tests/curation/test_history.py` (extend) | add the never-picked-up `merge_class` history-writer case — `src/routers/curation/classes.py` landed without it despite an explicit commit-body promise. | `$REF/tests/$NS/test_history_writers.py` (359 LOC / 4 tests; 2 already folded in) |

**W5.b — Zero-coverage leaves**

| Module | Stmts | Test to write |
|---|---:|---|
| `src/services/detection/region_lean.py` | 69 | synthetic text quads at 0°, ±15°, ±45°, plus degenerate/collinear input; assert the returned angle. Pure geometry. |
| `src/services/curation/clustering/outliers.py` | 61 | synthetic cluster with planted far-from-centroid members; assert returned ids/distances, and behaviour on 1-member and empty clusters. Feeds `/curation/review/outliers`. |
| `src/services/curation/autolabel/cli.py` | 117 | invoke `main()` through `argparse` with a faked job runner; assert arg→param mapping, exit codes, dry-run path. |
| `src/services/detection/pe_preprocess.py` (33), `ensemble_nms.py` (37) | | covered by Wave 2. |

**W5.c — The worst-covered routers**

| Target | Current | Test to write |
|---|---:|---|
| `src/routers/curation/regions.py` + `regions_fp.py` | 12.62% / 18.18% | `tests/curation/test_regions_router.py` + `test_regions_fp_router.py`. **These carry the entire human region-labelling write path and have never had a test on either line.** Drive `PUT /crops/{id}/plate`, `PATCH /crops/{id}/plate_meta`, `POST /plates/batch_status` against a fake OpenSearch; assert the detector-chain field **appends rather than replaces**, verifier fields are stamped, and a value outside the human-settable status set is rejected (`src/routers/curation/_common.py:332`). |
| `src/routers/curation/pipeline.py` | **5.08%** (224/236 missed) | `tests/curation/test_pipeline_router.py`. Drive `POST /curation/pipeline/auto_label` with and without the broadened residual pool; assert the pool differs and candidate cluster ids land in the residual band. |
| `src/services/curation/clustering/orchestrator.py` | 31.92% (465 missed) | targeted tests for `refine_region_cluster`, `cluster_residuals`, and the cluster-write payload shape. |

**W5.d — Structural test-quality fixes**

| Target | Change |
|---|---|
| `pyproject.toml` `[tool.coverage.run]` | `source = ["src", "scripts"]` — currently `scripts/` (8,690 LOC incl. the whole detection worker and the bakeoff harness) is **invisible**. Expect the headline number to move; that is the point. |
| `pyproject.toml` `[tool.pytest.ini_options]` | add `--cov-fail-under=<baseline>` set to the measured post-Wave-5 number minus 1. Without it, coverage can regress to zero silently. |
| `tests/curation/test_metrics.py:16,32` | two tests with **no assertions**. Replace with assertions on the registry: metric names, label sets, and that `/metrics` exposition contains them. |
| `tests/curation/test_clustering_orchestrator.py:131-133`, `test_methods_router.py:223-225`, `test_plate_sanity.py:101-116`, `test_curation_opensearch.py:228`, `test_review_router.py:95-114` | **self-referential constant assertions** (T-8) — each imports a constant from the module under test and asserts the module emitted it. Replace each with a literal expected value, so changing the constant fails the test. |
| `tests/curation/test_semantic_search.py:208` | the test name promises the encode call is offloaded to an executor; the only assertion is `assert_called_once_with`. Either assert the offload (capture the loop's executor) or rename the test to what it checks. |
| `tests/curation/test_precommit_paths.py` (extend) → new `tests/test_region_field_guard.py` | **behavioural tests for `scripts/codegen/check_no_literal_region_fields.py`** (T-5). Subprocess the guard against fixture file contents covering each accumulated skip rule: comment lines, Pydantic attribute declarations, wire-response dict keys routed through `RegionFields`, `model_fields_set` membership checks, and a genuine violation that **must** be caught. |

**W5.e — The state enum (§0.3 overturned classification)**

| Target | What |
|---|---|
| `src/config/region_state.py` (new) | port the 75-LOC status `str, Enum` + the terminal/pending frozensets from the reference `src/config/plate_state.py`, **keeping the on-disk string values byte-identical** (same no-migration reasoning as `RegionFields`). Name it `RegionStatus`. |
| `src/config/__init__.py` | add to `__all__` (a missed `__all__` entry breaks `from src.config import …`) |
| ⚠️ ≥10 files, 52 sites | replace the hardcoded status-string literals with enum members. Start with `scripts/curation/worker/runner.py:436,627,641,671,797`, `worker/combined.py:91`, `worker/state.py:67-69`, `src/routers/curation/regions.py:392`, `_common.py:333`, `src/services/curation/review_queries.py:149`. |
| `tests/curation/test_region_state.py` (new) | assert every enum value round-trips, that terminal ∪ pending covers the enum, and — a real guard — that no `.py` under `src/`/`scripts/` still contains a bare status literal outside the enum module. |

**Commits:** one per sub-wave (`test(curation): restore the frozen-holdout
write guards`, `test(curation): restore four OCC invariants against the
existing fakes`, `test(curation): cover the zero-coverage detection and
clustering leaves`, `test(curation): add region-router write-path tests`,
`test(build): measure scripts/ and enforce a coverage floor`,
`refactor(config): introduce RegionStatus and retire 52 status literals`).

---

### Wave 6 — Live write-path verification harness

**Blocked by:** Waves 1 and 2 landing (the harness should exercise the real
ingest path), plus D8 if the service rename happens first.

**Goal:** exercise 41 of 47 write endpoints against a real OpenSearch, in an
isolated compose project that cannot touch the live stack.

**Why `docker/test/` and not the repo root** ✅: `.gitignore:118` is
`docker-compose.*.yml` with a `!docker-compose.yml` exception, so **any**
`docker-compose.<x>.yml` at the root is ignored and cannot be checked in.
`docker/test/compose.yml` is not ignored, and `docker/hardened/test/` is an
existing precedent.

**New files:**

```
docker/test/
  compose.yml                 project name: op-live-verify
  Dockerfile.fake-vlm
  fake_vlm.py                 OpenAI-compatible /v1/chat/completions, fixed replies
  fake_trainer.sh             polls the jobs dir; queued -> running -> finished; honours .cancel
  README.md
scripts/curation/seed_live_harness.py
tests/live/
  __init__.py  conftest.py
  test_live_labels.py  test_live_clusters.py  test_live_classes.py
  test_live_regions.py  test_live_vlm.py  test_live_train.py  test_live_export.py
```

**Compose design (exact, to avoid collisions):**

- Project name `op-live-verify`; container names `op-verify-{opensearch,api,fake-vlm,trainer}`.
- Host ports **147xx** — outside 4600–4610 (live stack, currently occupied),
  outside the 5xxx band (labeler / mlflow), and outside 14603/14607 (an older
  ad-hoc harness) so both can coexist.
- **Never mount `/var/run/docker.sock`.**
- `api-verify` runs `--workers=1` — several cancel endpoints and the SSE hub
  keep **in-process** state, so with 2 workers a cancel POST can land on a
  different worker than the start POST.
- `api-verify` gets a healthcheck; `depends_on: {opensearch: {condition: service_healthy}}`.

**Required env on `api-verify`** — each entry fixes a concrete defect found
in the previous ad-hoc harness:

| Var | Value | Why |
|---|---|---|
| `OPENSEARCH_URL` | `http://os-verify:9200` | ⚠️ the previous harness set `OPENSEARCH_HOSTS`, which is the *dashboards* variable; the app reads `opensearch_url` (`src/config/settings.py:68-70`, `env_prefix=''`). It only worked by accident because the service was named `opensearch`. |
| `TRITON_URL` | `triton-unavailable:8001` | the pool never dials (§3.4) |
| `OP_API_PREFIX` | `/kb` | ⚠️ makes the harness serve byte-identical paths to what the existing labeler frontend already calls, so it can be pointed at the harness with only `PUBLIC_TRITON_API_URL=http://localhost:147xx` — **no frontend rebuild, no `API_PREFIX` refactor.** See D3. |
| `OP_ITEMS_INDEX` etc. | `verify_items`, `verify_images`, `verify_labels_confirmed`, `verify_classes` | belt-and-braces: even a mis-wired host cannot collide with real data |
| `OP_REGISTRY_PATH`, `OP_SOURCE_ROOT`, `OP_EXPORT_ROOT`, `OP_STATE_DIR`, `OP_CROP_CACHE_DIR` | mounted, writable paths | the previous harness mounted none of these; every registry write and export vanished into the container layer |
| `OP_TRAIN_JOBS_DIR` | a named volume shared with the fake trainer | |
| `OPENWEBUI_BASE_URL` | `http://fake-vlm:8000/v1` | `src/services/labeling/vlm_client.py:43` |
| `OP_SCORES_ENABLED`, `OP_SELECT_DIVERSE_ENABLED`, `OP_VIZ_PROJECTION_ENABLED` | `1` | ⚠️ **three write endpoints are disabled by default** and returned 400 throughout the previous pass regardless of what was "exercised". |

**Seed script** (`scripts/curation/seed_live_harness.py`) — must call the
repo's own helpers, not hand-roll mappings:

1. `src.routers.curation._common._ensure_indexes(client)` — this chains
   `create_curation_indexes` plus the twelve `ensure_*` migrations **in the
   right order with the one intentionally-unwired migration correctly
   skipped**. ⚠️ Do **not** call `ensure_items_class_name_keyword`; it always
   fails on a generic index (documented at `_common.py:115-121`).
2. `get_class_registry().add_class(...)` ×8 — **before any label write**, since
   the label endpoint 400s on an unknown or deprecated class.
3. Bulk-index **≥400 item docs** and ≥50 image docs, with `_id == crop_id`
   (every OCC path addresses by doc id). Cover all three cluster-kind bands:
   negative (noise / excluded), the class band, and the residual/candidate
   band. Include at least one cluster with ≥4 members at ≥0.85 dominant-class
   share (so auto-promote promotes) and one below threshold (so you can
   assert it does not). Seed the stored probe-prediction fields so review
   sorting and mistakenness scoring need no model.
4. **Embeddings must be non-degenerate**: per intended sub-cluster, draw a
   random unit centroid and generate members as `centroid + N(0, 0.15)`,
   L2-normalized, under a fixed seed. ⚠️ Identical or all-zero vectors
   collapse AHC to one sub-cluster and make cosine similarity undefined — the
   refine assertions would pass vacuously.
5. Generate ~50 tiny solid-colour JPEGs into `OP_SOURCE_ROOT` — only needed
   for the image/thumbnail routes, but cheap and it lets the frontend be
   eyeballed against the harness.

**Driver:** pytest. Add a `live` marker to `pyproject.toml` `markers`
(`--strict-markers` is on, so an undeclared marker errors) and **deselect it
by default** (`addopts += ["-m", "not live"]`). Run with `--no-cov`. Use a
synchronous `requests`/`httpx.Client` driver — there is no
`asyncio_mode = auto` configured.

**Safety guard in `tests/live/conftest.py`:** refuse to run if
`$OPENSEARCH_URL` resolves to port 4607, or if any configured index name
lacks the `verify_` prefix. Make this the first fixture.

**Scenario list** — implement in this order; later scenarios depend on
earlier state. Full ordered list of 42 scenarios covering: registry
create/rename/sync; single + batch label with **`class_id_history` growth**
asserted; unknown-class rejection with `_version` unchanged; unlabel;
review-dismiss; exclude/unexclude round trip; move; flag-new-class; region
bbox set/clear; region metadata patch; bulk region status; **cluster refine
asserting ≥2 distinct sub-ids via a terms aggregation**; refine idempotence;
refine member-count floor; **auto-promote dry-run asserting `_seq_no`
unchanged** then apply; region clustering; FP centroid build; holdout freeze;
**class-merge refusal when frozen holdout rows would be relabelled** (a
negative test, and it must run *before* the successful merge); merge success;
score compute; diverse select asserting **no** mutation; viz rebuild; cancel
semantics; VLM label batch asserting the **exact** fake reply string; VLM
verify; **VLM visibility batch with the fake returning an explicit negative**
(⚠️ that endpoint fails open to `True`, so a fake returning garbage is
indistinguishable from success); export YOLO with label-count and symlink
assertions; export determinism across two tags; registry-artifact serve;
train preflight asserting **no** file appears in the jobs dir; train start
asserting the job file appears **and `docker ps` is byte-identical before and
after** (proving the arbiter no-op); fake-trainer lifecycle; cancel;
campaign + cancel-campaign; **two parallel labels of the same item** (one
wins, the other retries or raises the final-conflict error, and the history
array has no torn entry); and a **restart-durability** check that recreates
only the API container and re-reads everything.

**Teardown:** `docker compose -p op-live-verify -f docker/test/compose.yml
down -v --remove-orphans`.

**Document as explicitly unverifiable without a GPU**, in
`docker/test/README.md`: `POST /curation/train/promote/{job_id}` (needs a
real ONNX + live Triton), `DELETE /curation/models/{name}`,
`POST /curation/pipeline/auto_label`(+`/start`), and the evaluation half of
`POST /curation/bakeoff/run`. Also state plainly that the VLM fake proves
**wiring and persistence, not label quality**.

**Commits:**
- `test(live): add an isolated write-path verification harness`
- `test(live): seed script and cohort-a label/cluster/class scenarios`
- `test(live): fake VLM and fake trainer scenarios`

---

### Wave 7 — Documentation, OSS furniture, CI

**Blocked by:** D7 (experimental gating) and D9 (licence question) for
wording only; the rest is unblocked.

| # | Target | Must contain | How to verify |
|---|---|---|---|
| **7.1** | `docs/CURATION.md` (new) | The missing user-facing doc. What the subsystem is, **domain-neutrally**; the three (four, after D-1's `RegionStatus`) config dataclasses; the full `OP_*` env table; the class-registry schema with a **worked non-vehicle example**; which models are required and how to obtain them (or an explicit "you must supply these"); the workers and how to start them (`--profile curation`); the seed path; and the known gaps stated up front. | Follow it verbatim on a clean box; `GET /curation/classes` returns the example registry's classes. |
| **7.2** | `README.md` | Add a `### Curation & Active Learning` section covering the 23 route groups / 103 routes, linking 7.1. Fix `:216` (`/health/models` does not exist). Fix `:467-468` (tells the reader to `source .venv/bin/activate`, and names only a standalone smoke script — never `pytest`, never the 68-file curation suite). | `rg -c curation README.md` > 0; assert every path in the README tables exists in `app.routes` (make that a test). |
| **7.3** | `docs/design/labeler_api_contract.md` | Rewrite (D-7): retitle to `/curation`, **delete the false `:10-11` claim** that no implementation ships, fold the appendix into the body, remove the cross-team coordination notes at `:159-169`, fix `:70`'s pointer to a table that does not exist here. | A reader who has never seen the private repo can implement a client from it. |
| **7.4** | `docs/ARCHITECTURE.md` | Strip internal vocabulary — `:85-87` "private, domain-specific reference implementation", `:140` "Bucket B". Describe the gaps in product terms. Add the runtime-companion topology from Wave 4. | No reader needs the private repo to parse it. |
| **7.5** | `.github/workflows/ci.yml` (new) | Two jobs: `pytest` (using `requirements-test.txt` from Wave 1) and `pre-commit run --all-files`. **811+ tests and 27 hooks currently have zero enforcement.** Trigger on PR and push to `main`. | A PR that removes a `sklearn` import goes red. |
| **7.6** | `SECURITY.md` (new) | Disclosure contact **plus an explicit statement that the API has no authentication** and must not be internet-exposed — it serves `DELETE /query/image/{id}`, `DELETE /curation/models/{name}` and `POST /ingest/directory` (arbitrary server-side path read). Note Grafana ships `admin/admin` and OpenSearch ships with its security plugin disabled. | GitHub shows the Security policy tab. |
| **7.7** | `CONTRIBUTING.md` (new) | Dev setup (venv, `pre-commit install`), how to run the suite, **how to run the Wave 6 live harness**, what the `check_no_literal_region_fields` ratchet is and why, conventional-commit + merge-commit conventions. | A new contributor gets a green pre-commit on first try. |
| **7.8** | `CHANGELOG.md` | Flesh out `[Unreleased]` from Wave 0 with every wave's user-visible change. | `rg -i curation CHANGELOG.md` > 0 |
| **7.9** | `CLAUDE.md` | Fix `:120-121` (the `/v1`-twin claim is false for all 103 curation paths). Fix `:5-23` — it instructs `source .venv/bin/activate`, contradicting this project's own stated rule of calling venv binaries directly. | |
| **7.10** | `scripts/README.md`, `docs/README.md`, `export/README.md` | ⚠️ `scripts/README.md` documents a `check_services.sh` that does not exist and a `make test` that did not; it omits `setup.sh`, `openprocessor.sh` and the entire 28-file `scripts/curation/` tree. `docs/README.md` claims "Last Updated 2026-01-27". Refresh all three. | the Wave 0 broken-link test stays green |
| **7.11** | `docs/security/` | ⚠️ Move out of the public tree (D-8) — internal DeepStream notes pinned to this host's GPU slot 2, for a component not in the product. | |
| **7.12** | `.github/` | `ISSUE_TEMPLATE/`, `PULL_REQUEST_TEMPLATE.md`, `dependabot.yml`, `CODEOWNERS`, `CODE_OF_CONDUCT.md` | |

**Commits:** `docs(curation): add the curation user guide`, `docs(readme):
document the curation surface and fix stale instructions`, `ci: run the test
suite and pre-commit on every PR`, `docs(oss): add SECURITY, CONTRIBUTING and
issue templates`.

---

### 4.9 — Wave 8: merge, tag, push

```bash
cd /mnt/nvm/repos/wt-oss-hardening
# full §5 gate one last time
git checkout main
git merge --no-ff feat/oss-hardening      # merge commit, NEVER squash
git tag -a v0.3.0 -m 'v0.3.0 — generic curation subsystem'   # annotated, never lightweight
git push origin main --follow-tags
```

⚠️ This is the **first publication** of all 29 local commits. Before it:

1. Re-run the leak scan over the **whole** tree, not just `src/scripts/tests/docs`:
   `git ls-files | xargs rg -ni '<the four proprietary tokens>'`.
2. Re-run a host-path sweep over every tracked file — any absolute
   `/mnt/...` path, and any occurrence of the host user's name. CFG-6 and
   D-8 must both be resolved, and §A.3 must have been applied to this file.
3. Confirm `git log origin/main..main --format='%an %ae'` contains no
   unexpected author.
4. The repo's `.git/hooks/pre-push` blocks pushes matching the private
   product's name. Do **not** disable it; if it fires, you have a leak.
5. Never `--no-verify`, never `--no-gpg-sign`.

---

## 5. Verification gate — run after EVERY wave

```bash
cd /mnt/nvm/repos/wt-oss-hardening
V=/mnt/nvm/repos/triton-api/.venv/bin

# 1. Import integrity — loudest, cheapest.
$V/python -c "import src.main"                       # must be silent

# 2. Suites. `integration` means "several components, no live stack".
$V/python -m pytest tests/ -q --no-cov -m 'not live'

# 3. Coverage, once per wave (from Wave 5 on this also enforces the floor).
$V/python -m pytest tests/ -q -m 'not live' --cov=src --cov=scripts \
    --cov-report=term-missing:skip-covered

# 4. Hooks, exactly as CI will see them.
$V/pre-commit run --all-files

# 5. Route surface only ever grows.
$V/python - <<'PY'
from src.main import app
r = sorted(x.path for x in app.routes if x.path.startswith('/curation'))
print(len(r))                      # was 103 at 1079933; must not shrink
PY

# 6. Leak scan — every wave, not just before the push.
git ls-files | xargs rg -ni '<the four proprietary tokens>' && echo LEAK || echo clean
rg -n "'plate_[a-z_]+'" src/ scripts/ tests/     # only the two documented exemptions

# 7. Dangling-doc scan (new in Wave 0).
$V/python -m pytest tests/test_doc_links.py -q --no-cov
```

**Wave 6 only**, additionally:

```bash
docker compose -p op-live-verify -f docker/test/compose.yml up -d --wait
$V/python -m pytest tests/live -q --no-cov -m live
docker compose -p op-live-verify -f docker/test/compose.yml down -v --remove-orphans
```

⚠️ Before any of that: `docker ps --format '{{.Names}} {{.Ports}}'` and
confirm nothing you are about to start collides. The live stack is running.

---

## 6. Risks

**R1 — The dependency gap (§0.5) means the product has never actually run
its own clustering code in its own container.** The test suite passes only
because the dev venv was hand-augmented. Wave 1's `requirements-test.txt` +
Wave 7's CI job together close this permanently; until both land, a green
suite is not evidence the image works. **Mitigation:** make Wave 1 the first
substantive wave and do not defer the CI job.

**R2 — Wave 2 is the largest new-build in this plan (~1,500 LOC of service
code) and its source is a 1,990-LOC file on the reference line.** The risk is
re-importing domain logic by accident. **Mitigation:** the §5 gate's leak
scan and the region-field guard both run every wave; and the explicit
"do not port" list in Wave 2 names the four domain-specific blocks. Port the
generic half only, and let the test for a *second*, non-vehicle
`DetectionProfile` (`tests/curation/test_detection_profile_second_profile.py`,
already present) be the acceptance test that no domain assumption crept in.

**R3 — Fixing §3.2 will change export output for anyone already consuming
it.** Nobody is: `main` is unpushed. **This is the cheapest moment in the
project's life to fix it.** After the push it becomes a breaking change.

**R4 — The 8 grandfathered oversize files.** `.pre-commit-config.yaml`
excludes eight curation-port files from the 700-LOC ratchet (the port plan
pre-approved five; three more — `src/services/training/jobs.py`,
`src/services/detection/cascade_detect.py`,
`src/services/training/triton_promote.py` — were added during the port with
in-file justifications) ✅. Wave 2 adds `src/services/curation/ingest.py`,
which **must not** become the ninth: split it during the port (the plan
names the split point). Every other new file must land under 700 LOC.

**R5 — Wave 6 runs containers on a host with a live production stack.** The
mitigations are structural, not procedural: a distinct compose project name,
container names in a distinct namespace, host ports in a distinct band,
index names with a mandatory `verify_` prefix enforced by a conftest guard,
and no docker socket. ⚠️ Independently confirmed that the GPU arbiter is a
no-op on `main` (§3.4), so the training endpoints cannot stop a container
even if misconfigured. **Never run `docker compose` against the repo-root
compose file from this worktree.**

**R6 — Parallelism.** Wave 5 is independent of Waves 2–4 and can run
concurrently in a second agent, but Wave 5's coverage floor
(`--cov-fail-under`) must be set **after** Waves 2–4 merge, or it will be
computed against a smaller codebase and immediately fail. Sequence: land
2–4, then set the floor.

**R7 — Audit claims marked ⚠️.** Roughly a third of the findings in §0 come
from subagent audits the planner could not independently re-verify in the
time available. One such claim was checked and found **wrong** (the
`RegionFields` env override; §0.6). **Treat every ⚠️ as "verify first, then
act"** — a wasted verification is cheap; acting on a false finding is not.

---

## 7. Decisions needed from the owner

| # | Question | Options | Planner's recommendation |
|---|---|---|---|
| **D1** | **Design-doc provenance.** 42 references point at a plan document that lives only on the private line. | (a) author a new, genericized rationale doc on `main` and repoint; (b) scrub all 42 references and ship no rationale; (c) publish the port plan (leaks the private product name — **not viable**). | **(a).** The references exist because the code genuinely needs the rationale (why the ratchet has eight exemptions, why `RegionFields` exists, why the wire contract is frozen). Scrubbing loses that. Budget ~400 lines. |
| **D2** | **`KB_*` → `OP_*` env rename** (23 vars). | (a) hard rename, no shim; (b) dual-read with a deprecation warning. | **(a).** Nothing is published; there is no deployment to break. A shim for a name that never shipped is pure debt. |
| **D3** | **The `/curation/plates/*` URL paths** (10 public routes) name a specific domain in a generic product's URL. The port deliberately froze them because an existing frontend calls them. | (a) leave frozen; (b) rename to `/curation/regions/*` and break the frontend; (c) mount **both**, with the domain-named set documented as a deprecated alias. | **(c).** It costs a few lines of router aliasing, keeps the existing frontend working unmodified (which Wave 6 depends on), and lets the generic name be the documented one. Revisit (b) at v1.0. **SUPERSEDED — work item B2 landed (b)**: the ten routes are now `/curation/regions/*` with no deprecated alias, agreed with the frontend consumer first. See the B2 section in `docs/design/curation_api_contract.md`. |
| **D4** | **25 `kb_*` Prometheus metric names** exposed from the public build. | (a) rename to `op_*` now; (b) keep deferred. | **(a), in Wave 1.** Same reasoning as D2 — unpublished, and `kb_gemma_*` additionally leaks a model vendor. `monitoring/` references none of them, so the blast radius is zero. |
| **D5** | **The segmentation service.** The detection cascade's segmenter leg calls an HTTP service whose container was never ported. | (a) port and genericize it (~820 LOC); (b) make the segmenter leg optional and document that users supply their own; (c) drop the leg. | **(b) for this release, (a) later.** (b) is a day of work and makes the cascade honest; (a) is a meaningful project and the reference implementation is tied to a specific model family. |
| **D6** | **The trainer container** (3,624 LOC). Without it `/curation/train/*` — 13 routes, ~3,200 LOC of API code — drives nothing. | (a) port and genericize; (b) ship the API and document that users supply a trainer implementing the documented `job.json` file protocol; (c) hide the training routes. | **(a) if there is appetite, else (b).** The file protocol is genuinely simple and well documented (`src/services/training/jobs.py:7-12`); (b) plus a reference `fake_trainer` from Wave 6 is a defensible interim. **(c) is worse than both** — it hides 3,200 LOC rather than explaining it. |
| **D7** | **Should `/curation` ship as "experimental"** for the first public release? | (a) yes — behind the Wave 4 compose profile, labelled experimental in the README; (b) no — ship it as a first-class feature. | **(a) for v0.3.0, (b) at v0.4.0 once Wave 6 has actually run green.** This converts a credibility liability into a stated roadmap item, and the compose profile makes it free. |
| **D8** | **Rename the `yolo-api` service** (a legacy name for a general visual-AI API) to `openprocessor-api`? ⚠️ ~127 references across 25 files. | (a) do it as one mechanical sweep; (b) defer. | **(b), defer to a dedicated commit after this plan.** It is purely cosmetic, it is the widest-blast-radius change on the list, and mixing it into functional waves would make every diff unreviewable. |
| **D9** | ⚠️ `LICENSE:23-58` flags the vendored Ultralytics fork as **AGPL-3.0** inside an **MIT**-badged repo. | Legal question, not an engineering one. | **Get an answer before the Wave 8 push.** Options: isolate the vendored fork behind an optional extra; replace it; or re-badge. This is the only item that could require unwinding work after publication. |

---

## 8. Critical files for the implementing agent

| File | Why |
|---|---|
| `src/main.py` | Curation routers are registered at `:472-478` with no extra prefix — each router bakes in `CurationConfig.api_prefix` itself. `import src.main` is the gate's loudest check. |
| `src/config/__init__.py` | Explicit `__all__` ✅ — every new config module (`region_state.py` in Wave 5) must be added or `from src.config import …` fails. |
| `src/config/curation.py` | `CurationConfig`, `IndexRole`, `index_name()`, `from_env()`, `get_curation_config()`. `:146-163` contains the fixed version of the exact bug an audit wrongly attributed to `region_fields.py` — read the comment there before touching either. |
| `src/config/region_fields.py` | `get_region_fields()` at `:119-136` **does** call `from_env()`. Verified live. Do not "fix" it. |
| `src/routers/curation/_common.py` | Owns the shared `APIRouter` (`:56`), the index constants, the Pydantic wire models (**frozen**, including the five ingest models currently unused by any route — Wave 2 uses them), `_ensure_indexes` (`:92-145`, with one intentionally-unwired migration documented at `:115-121`), and the human-settable status frozenset at `:332-334`. |
| `src/clients/curation_opensearch.py` | 1,537 LOC, ratchet-exempt. Index bodies, `ClassRegistry` (atomic write + snapshot), 13 `ensure_*` migrations, `create_curation_indexes` (`:1251`). The seed script must call these, not hand-roll mappings. |
| `src/services/curation/export.py` | 323 LOC. `ARTIFACT_FILENAMES` at `:43-46` declares an artifact the code never writes (§3.2). Wave 3's centre of gravity. |
| `src/services/training/jobs.py` | `:7-12` documents the entire trainer protocol; `_resolve_jobs_dir()` at `:58-64` re-reads its env var on every call *specifically so tests can override it* — that is the Wave 6 injection point. |
| `tests/curation/occ_fakes.py` | `FakeOccOpenSearch` at `:96` with real `_seq_no`/`_primary_term` semantics, confirmed against a real OpenSearch 3.6.0 cluster. **Extend this; do not invent a second double.** Wave 5's invariants restoration depends on it. |
| `tests/integration/test_ingest_occ.py` | `FakeUpsertOpenSearch` at `:48` with genuine conflict detection at `:124-127`. The second existing double. |
| `.pre-commit-config.yaml` | One `max-file-size` ratchet, **8** curation exemptions (`:128-145`), four dangling doc references (`:123,147,167,183`), and the `check-no-literal-region-fields` hook (`:188-195`). |
| `scripts/codegen/check_no_literal_region_fields.py` | `PORTED_PATHS` must gain every newly-ported path in the same commit. Zero behavioural tests today (Wave 5). |
| `pyproject.toml` | `testpaths=["tests"]`, `addopts` forces `--cov`, one `markers` entry, `--strict-markers` on, `[tool.coverage.run] source=["src"]` (excludes `scripts/`), **no `fail_under`**. |
| `docker-compose.yml` | **Do not run it from this worktree.** Its container names and ports 4600–4610 are in use by the live stack right now. |
| `/mnt/nvm/repos/wt-kb-readonly` | Read-only reference (detached HEAD). Sources for Waves 2, 3, 4 and 5 live here. Never commit there. |

---

## Appendix A — Out of scope, recorded so it is not lost

### A.1 The ortloom inference engine

Explicitly future work, planned elsewhere
(`/mnt/nvm/repos/ortloom/docs/design/architecture_plan.md`, and the labeler
frontend repo's own `docs/design/ortloom_backend_integration_plan.md`).
Do not design seams for it here.

### A.2 The private-deployment cutover

The port plan's Appendix A. A separate project with its own risk budget and
its own production-change window; it is the only work that would ever touch
live containers. It may never happen, and this plan does not presuppose it.

### A.3 Lines in this document that must be stripped before publication

This file contains absolute local paths so a fresh agent can execute it —
the working worktree, the read-only reference worktree, the venv, and the
ortloom repo, all under `/mnt/nvm/repos/`. Before the Wave 8 push, either
parameterise them or move this document to an untracked location. Resolve
together with D1 and the CFG-6 / D-8 path cleanups — one `rg -n '/mnt/nvm'`
sweep over tracked files covers all three.

Note this file is deliberately written to pass the §5 leak scan **as it
stands** (no private product name, namespace prefix, NAS path, or corpus
name — §2.1 explains the `$REF`/`$NS` convention that makes that possible).
Keep it that way when editing: re-run step 6 of the §5 gate after any change
to this document.

### A.4 Deferred beyond this plan

- The `yolo-api` → `openprocessor-api` service rename (D8).
- Splitting the eight grandfathered oversize files — especially
  `src/services/labeling/vlm_labeler.py` (1,870), `orchestrator.py` (1,769),
  `curation_train.py` (1,466) and `scripts/curation/worker/runner.py`
  (1,303). File these as issues at merge time so the ratchet has a stated
  path back down.
- A generic *single-class* dataset exporter (the reference's is genuinely
  domain-specific; only two helpers — aspect-preserving resize without
  padding, and cluster-aware split grouping — are generic, and Wave 3 absorbs
  both).
- Authentication (D-10). Wave 7 documents its absence; implementing it is a
  separate project.

---

## Appendix B — Corrections to the audits that produced this plan

Recorded so a later reader does not reintroduce them.

1. **`RegionFields.from_env()` is NOT dead.** An audit reported that
   `src/config/region_fields.py` builds a bare `RegionFields()` and silently
   ignores all 36 `OP_REGION_FIELD_*` vars. False: `:133-136` calls
   `from_env()`, and the planner proved the override works at runtime. The
   confusion is that `src/config/curation.py` had exactly this bug, and it
   was fixed in commit `07d20fa` — the docstring at `curation.py:150-153`
   describes it verbatim.
2. **The port's file mapping is incomplete, not wrong.** Thirteen files were
   ported without appearing in any mapping: nine `src/services/curation/*`
   modules (`embedding_viz`, `event_hub`, `history`, `image_serving`,
   `metrics`, `probe_predictions`, `review_queries`, `review_sorts`,
   `semantic_search`, `source_image_cache`, `strategy_registry`),
   `src/clients/occ.py`, `src/clients/pe_encoder.py`,
   `src/config/gpu_arbiter.py` and `scripts/curation/backfill_scores.py`. ⚠️
3. **`src/services/curation/export.py` is not a port** of the reference
   exporter — it is a 323-LOC from-scratch generic re-implementation
   covering roughly a quarter of it. Its own docstring (`:9-15`) says so.
   The mapping line claiming otherwise is wrong.
4. **`src/clients/{occ,pe_encoder}.py`, `src/config/{curation,region_fields,
   detection_profile,gpu_arbiter}.py` all arrived with the port**, while
   `src/clients/fast_face_client.py`, `src/services/cluster_maintenance.py`,
   `src/services/cpu_preprocess.py` and `src/routers/persons.py` predate it
   (verified by `git log --follow`). ⚠️
