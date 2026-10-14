# Documentation overhaul: audit, task guides, pipeline modes, screenshots

Status: plan only. Nothing outside this file was changed. Written 2026-10-13 against `main` at
`bbd1c9ac`, VERSION `0.5.0`. A fresh implementation agent should be able to run this plan from this
file alone:
- section 1 is the audit;
- sections 2 to 4 give the content specs;
- section 5 lists the rules every edit must obey;
- section 6 breaks the work into tasks with owned files and acceptance checks;
- section 7 covers the integration and the PR;
- section 8 lists the code follow-ups to file as issues, not to fix here.

Line references drift. Re-find them by symbol before editing.

**Owner goal.** A new or existing user can operate the app end to end from the docs alone, with
copy-paste commands, real captured output and screenshots. The docs must state only what the code
does today. Planned work is labelled as planned. Code gaps are filed as issues (section 8), not
papered over in prose.

## 0. Decisions

1. **One source of truth for user and operator docs: `docs-site/docs/`.** It is the published,
   build-checked site, with `onBrokenLinks: 'throw'` (`docs-site/docusaurus.config.ts:38-39`).
   - `docs/` keeps maintainer material: architecture, schema, design plans, research,
     `PERFORMANCE.md` as the measurement log, releases and security.
   - Nothing generates docs-site pages from `docs/`. There is no sync script; every page is hand
     written, so duplication is manual and has already drifted.
2. **`docs/CURATION.md` (1539 lines) becomes a thin map** of the curation topics, one short
   paragraph and a link to the site page per topic.
   - Every heading that code, compose, tests or scripts cite keeps its exact text, so their
     anchors still resolve. Cited headings: "Mounting your image source"
     (`docker-compose.yml:177,411`, `env.template:318`), "Importing other formats"
     (`src/routers/curation/datasets.py:106`), "Workers and the curation compose profile",
     "VLM", "Wiring up Cropwright", "Models you must supply" (`scripts/setup.sh:846-891`,
     `scripts/lib/export.sh:43`).
   - The file stays in `tests/test_generic_default_docs.py:14`.
   - Body text unique to CURATION.md moves into the matching site page first (task T4 owns the
     diff).
3. **Core guide versus Cropwright page split.**
   - `docs-site/docs/guides/*` explain the concept and the API/CLI path.
   - `docs-site/docs/cropwright/user-guide/*` explain the UI steps and link to the core guide for
     the concept.
   - Pages that pair up this way: projects, prompt-packs, region-profiles, multi-box-regions,
     dataset-import, combine-projects, open-vocabulary, ingest-policy, vlm-selection/vlm-models.
     Remove the duplicated concept prose from the Cropwright side.
4. **`docs/VISION_AND_GOALS.md` stays canonical** (CLAUDE.md links it).
   `docs-site/docs/about/vision-and-goals.mdx` becomes a short summary that links to the GitHub
   file instead of a drifting mirror (it mirrors the file per its lines 8-9 and differs at lines
   3-4).
5. **Branch `docs/overhaul`, one PR, merge commit, conventional commits per task.**
   The `CHANGELOG.md` `[Unreleased]` entry is under `### Changed` (docs).

## 1. Audit findings

All guards pass today:
- `.venv/bin/python scripts/docs/check_docs_vs_code.py` reports routes 0, env 0, links 0;
- `scripts/codegen/check_naming_leaks.py` exits 0;
- the doc pytest files (`tests/test_version_consistency.py`, `test_doc_links.py`,
  `test_docs_vs_code.py`, `test_generic_default_docs.py`, `test_makefile_routes.py`,
  `test_readme_*`) pass.

So every finding below is something the guards miss.

### 1.1 Inventory (wc -l)

Root: `README.md` 945, `INSTALLATION.md` 850, `CONTRIBUTING.md` 188, `SECURITY.md` 148,
`CHANGELOG.md` 4993. `docs/`: `CURATION.md` 1539, `PERFORMANCE.md` 3688, `ARCHITECTURE.md` 535,
`OCR.md` 643, `opensearch_schema_design.md` 281, `image_size_audit.md` 299 (not indexed),
`FACE_RECOGNITION_IMPLEMENTATION.md` 191, `MIGRATION_TRITON_26.md` 111, `README.md` 150,
`VISION_AND_GOALS.md` 129, plus `design/` (17), `research/` (10), `releases/` (4), `market/` (11),
`benchmarks/`, `security/`, `Technical/`. `docs-site/docs`: 40 core `.mdx`, 41 Cropwright `.md`
plus 1 `.mdx`, 2 developer-guide pages. `frontend/`: `README.md` 327, `CLAUDE.md` 3101,
`docs/FEATURES.md` 138, `docs/annotation-slots-contract-draft.md` 911, `docs/releases/*`. Other
READMEs: `src/` 476, `scripts/` 211, `export/` 468, `monitoring/` 471, `benchmarks/` 135,
`docker/*/`. The untracked, gitignored `docs/OpenProcessor_Triton_Deep_Research_2026-09-26.md`
nearly duplicates `docs/research/triton_deep_research_2026-09.md`; leave it alone.

### 1.2 Stale or wrong statements (fix in the owning task)

| # | Where | Problem | Truth in code | Task |
|---|---|---|---|---|
| S1 | `docs/CURATION.md:721` | bulk defaults "8 and 16" | batch 16, concurrency 16 (`src/services/curation/bulk_ingest/limits.py:15-16`) | T4 |
| S2 | `docs/CURATION.md:700-705`; `scripts/curation/bulk_ingest.py:23-24` docstring | `docker compose exec api python scripts/curation/bulk_ingest.py` with no `--api-base` | inside `api` the API is on 8000 (`docker-compose.yml:257`); the default `localhost:4603` fails, so pass `--api-base http://localhost:8000/curation` (README:281-283 does) | T2, T4 |
| S3 | `docs-site/docs/getting-started/quick-start.mdx:159,191-192` | says `--root` | `bulk_ingest.py` uses `--folder` (`:70-74`); `--root` belongs to `ingest_upload.py` | T1 |
| S4 | `docs/CURATION.md:1408-1418`, `configuration/basic.mdx:142`, `env.template:315-318` | only api and the detection worker mount the source root | three services mount it read-only at `/data/source`: `api` `docker-compose.yml:178`, `curation-detection-worker` `:412`, `curation-auto-label-worker` `:525` | T1, T2, T4 |
| S5 | `README.md:278`, `quick-start.mdx:146` | recreate only `api`, no `-p` | after changing `OP_SOURCE_ROOT_HOST` all three services must be recreated: `./openprocessor start` (`openprocessor:197-205`, which always passes `-p`) or `docker compose -p <project> up -d` | T1 |
| S6 | `basic.mdx:142`, `advanced.mdx:87` | host default `./data/source` only | installer writes `OP_SOURCE_ROOT_HOST=./data` (`setup-openprocessor.sh:2979`); a checkout gets the compose default `./data/source` | T1 |
| S7 | `INSTALLATION.md:104` | `CW_GH_REPO` | no code reads it; the installer reads `OP_GH_REPO` (`setup-openprocessor.sh:22,72`) | T1 |
| S8 | `INSTALLATION.md:473-521`, `deployment/gpu-sizing.mdx:18-27` | minimal is "core only / no OCR"; recommends `openprocessor.sh profile` | a profile tunes only instance counts, batch, workers, shm and GPU decode (`scripts/lib/gpu.sh:137-162`). The Triton load list is fixed (`docker-compose.yml:88-117`) and the profile JSON `models` list is never read. `profile` is broken (issue F4) | T3 |
| S9 | `operations/workers.mdx:35` | cluster refresh at 1000 items, "optional" auto-label | default growth threshold 200, auto-label on by default (`scripts/curation/cluster_refresh_daemon.py:89-133`), interval 120 s | T3 |
| S10 | `compose-profiles.mdx`, `workers.mdx:36` | evaluator long-lived | on demand, `run --rm` (`docker-compose.yml:597-598`, `docker/evaluator/Dockerfile:28`); note `--profile curation up -d` still starts it | T3 |
| S11 | `docs/CURATION.md:1364-1376` | VLM worker "verifies"; "none needs Triton or a GPU" | the VLM worker labels classes and the detection worker verifies regions; the detection worker needs `triton-server`; the evaluator reserves a GPU | T3 (site), T4 (map) |
| S12 | `docs/CURATION.md:801-805`; `src/domain/embedding_state.py:21` comment | `deferred` only from combine | `lazy` ingest also writes `deferred` (`src/services/curation/ingest_policy.py:196-198`). The code comment is a follow-up (F12), not a docs edit | T4 |
| S13 | `env.template:405-408` | compose mounts `./examples` | only `docker-compose.dev.yml:58,76,97`; release images bake them in | T1 |
| S14 | `openprocessor:572-573` comment | `start` never starts curation workers | it does when `.env` has `COMPOSE_PROFILES=curation` (written by `setup-openprocessor.sh:2975`); code comment, F12 | none |
| S15 | `api-reference/curation.mdx:14` "237 routes"; `README.md:403` "more than 230" | wrong count | the contract has 272 method routes. Do not print a count; say "every route is in `contracts/openapi/curation.json`" | T7, T1 |
| S16 | `src/README.md:1,408,410`; `docs/PERFORMANCE.md:128,3524` | title "Visual AI API"; `OPENSEARCH_URL` default `:4607`; `SLOW_REQUEST_THRESHOLD_MS` | default `http://opensearch:9200` (`src/config/settings.py:76-78`); no code reads `SLOW_REQUEST_THRESHOLD_MS` | T4 |
| S17 | `docs/design/gpu_decode_plan.md:3,554,558,679` | "nothing in src changed", default `cpu` | built; on by default above 3 MP (section 13 of that file; `src/config/curation.py:257-264,424-427`) | T4 |
| S18 | `docs/design/storage_sizing_and_ingest_baselines.md:14-21,66-75` | per-vector faiss figures used as the sizing rule | new projects use Lucene, about 4.5 KiB per item against 8.6 KiB on faiss (`docs/PERFORMANCE.md:3229-3238`); add a dated note | T4 |
| S19 | `docs/design/v060_wave_plan.md:86-89` | daemon posts the synchronous route | changed by #212; add a status note | T4 |
| S20 | `docs/design/README.md` | missing `label_confirmation_plan.md`, `v060_execution_path.md`, and this file | add rows | T4 |
| S21 | `docs/README.md` | does not index `design/`, `research/`, `releases/`, `benchmarks/`, `image_size_audit.md` | rewrite as a maintainer index pointing users to the site | T4 |
| S22 | `docs/market/README.md:7,67`, `10-sources.md:6`, `07-*.md:37` | name an older release | reword as "the release current at the snapshot date (2026-10-04)" without a version string | T4 |
| S23 | `benchmarks/README.md:3`; `.pre-commit-config.yaml:1` | old product name "Triton API" | "OpenProcessor" | T4 (README only; the `.pre-commit-config.yaml` comment is F12) |
| S24 | `Makefile:1235` | cites the heading "Create classes from zero", which does not exist | point at the quick-start seeding step | T6 |
| S25 | `docs-site/docusaurus.config.ts:12` | cites the missing `docs-site/TEMPLATE.md` | remove the reference | T8 |
| S26 | `cropwright/developer-guide/screenshots.md:35,159,161`; `configuration/annotation-profiles.md:19` | "(repo root)" for the script; names an older release; "in the Cropwright repo" | the script is `frontend/scripts/...`; monorepo wording | T6, T5 |
| S27 | `frontend/README.md:32,57-60,310-316`; `frontend/docs/FEATURES.md:8`; `frontend/CLAUDE.md:30,797,2414-2441,2765` | standalone docs site, `cropwright-docs` image, "screenshots not published", missing design doc | root `docs-site/docs/cropwright/`, `docs-site/static/img/screenshots/cropwright/` | T5 |
| S28 | `docs-site/docs/guides/ingest-policy.mdx` | not in `docs-site/sidebars.ts` (orphan page) | T2 expands it; T8 adds it to the sidebar | T2, T8 |
| S29 | `docs-site/src/data/screenshot_routes.json` | stale 5-entry copy nothing reads; `ScreenshotShowcase/index.tsx:22` and `frontend/CLAUDE.md:2436` still name it | the script reads `frontend/scripts/screenshot_routes.json` (`capture_docs_screenshots.py:61`) | T6 |
| S30 | docs-site top-level PNGs `bakeoff/clusters/review/review-regions/train-1600.png` | byte-identical copies of the `cropwright/` ones | dedupe after all pages land | T8 |
| S31 | ingest docs everywhere | silent on these behaviours of the walker (`src/services/curation/bulk_ingest/walk.py:158,201-209`) | it does not follow symlinks and silently skips unreadable directories | T2 |
| S32 | VLM docs | never say the VLM is on by default for every project once `OP_VLM_URL` is set | `active_vlm_endpoint()` returns the env built-in when a project never activated one (`src/services/labeling/vlm_endpoints.py:446-447`) | T3 |
| S33 | ingest-policy docs | never say `lazy` still computes the whole-frame embedding | always computed (`src/services/curation/ingest_index.py:216-226`, `ingest_gpu_decode.py:168-174`) | T2, T3 |

### 1.3 What the guards do not catch

`scripts/docs/check_docs_vs_code.py` (309 lines) checks three things:
- routes against the contracts plus the app (code fences included);
- that `OP_*` names appear somewhere in the code corpus;
- that relative links and anchors resolve.

It does not catch:
- non-`OP_` variables (S7, S16);
- retired `OP_*` variables (`src/config/retired_env.py` is in its corpus);
- defaults, ports and numbers (S1, S15);
- CLI flags, `./openprocessor` subcommands and `make` targets (only Makefile routes are tested,
  in `tests/test_makefile_routes.py`);
- file paths and headings in prose (S24, S25);
- screenshot existence;
- `docs/` against `docs-site` drift;
- external URLs.

It is not a pre-commit hook; it runs only through pytest.

`tests/test_version_consistency.py:16-23,53-60` scans only these:
- `README.md`
- `INSTALLATION.md`
- `CLAUDE.md`
- top-level `docs/*.md`
- `docs-site/docs/**/*.mdx`

The regex `(?<![\w.])v(0\.\d+\.\d+)` needs the `v`. Cropwright `.md` pages, `frontend/**` and
`docs/` subfolders are not scanned (S22, S26).

`.github/workflows/docs.yml:15-30` builds the site only when `docs-site/**` changes. There is no
link checker of any kind; that gap is F10.

## 2. Spec: "Ingest a large folder from the NAS" (T2)

New page `docs-site/docs/guides/ingest-large-folder.mdx` (title "Ingest a large folder"). Example
paths use `/srv/nas/photos`, never a real host path. Sections, in order:

1. **Preflight checklist.** A short checklist; the same list goes in the page header admonition:
   - the stack is healthy (`./openprocessor health`);
   - a detector is configured; ingest returns 503 without `OP_INGEST_PRIMARY_DETECTOR_MODEL`
     (`src/routers/curation/ingest.py:98-111`);
   - the share is mounted on the host and readable by the container user;
   - `OP_SOURCE_ROOT_HOST` is set and the services are recreated;
   - `GET /curation/projects/{project}/ingest/bulk/config` lists `/data/source`;
   - a fresh project exists and its ingest policy is set before ingest (policy affects future
     ingests only, `src/routers/curation/ingest_policy.py:102`);
   - the 200-500 image trial passed;
   - free disk was checked (step 11);
   - the GPU decode state is known (`./openprocessor gpu plan`, `openprocessor:858-883`).
2. **Mount the share.**
   - Host mount (NFS/SMB) example as plain shell.
   - Set `OP_SOURCE_ROOT_HOST=/srv/nas/photos` in `.env`.
   - It is one root, mounted read-only at `/data/source` in `api`, `curation-detection-worker` and
     `curation-auto-label-worker`. Without the worker mounts, thumbnails 404 and the VLM stage
     predicts 0 (`docker-compose.yml:520-525`).
   - Apply with `./openprocessor start` (installed or checkout). Or, in a checkout, use
     `docker compose -p <project> -f docker-compose.yml -f docker-compose.dev.yml up -d api curation-detection-worker curation-auto-label-worker`.
   - Say that `restart` does not re-read `.env` (`openprocessor:214-229`, `Makefile:137-174`).
   - Container paths are `/data/source/...`.
3. **More than one share: `OP_SOURCE_PATH_ALIASES`.**
   - Format is JSON object or `alias=/path,alias2=/path` (`src/config/curation.py:431-472`).
   - The alias path must also be bind-mounted into the same three services with a user compose
     override file. Show the snippet, and state that compose does not do this for you (F6).
   - Images are served at `GET /curation/projects/{project}/images/root/{alias}`.
4. **Create a fresh project.**
   - `curl -X POST localhost:4603/curation/projects` with `slug`, `display_name`, and optional
     `clone_settings_from`.
   - Slug rules (`src/config/projects.py:25-60`): `^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$`, 2-32
     characters, reserved words, retired slugs.
   - Or use the Cropwright Projects page.
   - Then set the ingest policy (link `guides/ingest-policy`).
5. **Trial run on 200-500 images.**
   - There is no dry-run or count preflight (verified: no flag in `bulk_ingest.py:67-101`, no
     field in `BulkIngestRequest`, which has `extra='forbid'`; F1). So make a trial manifest:
     `find /srv/nas/photos -type f -iname '*.jpg' | head -n 300 | sed 's#^/srv/nas/photos#/data/source#' > data/trial.txt`.
   - The manifest must be readable inside the container: put it under the source root or a
     mounted dir. Verify where.
   - Run `--manifest`, then check review/thumbnails.
   - Ingest the trial into the real project (re-ingest is deduplicated by content hash), or into a
     scratch project you delete later.
6. **Full run, CLI.**
   - `docker compose -p <project> exec api python scripts/curation/bulk_ingest.py --project <slug> --folder /data/source --api-base http://localhost:8000/curation`.
   - Flags table from `scripts/curation/bulk_ingest.py:67-101`: `--folder`, `--manifest`,
     `--status RUN_ID`, `--cancel RUN_ID`, `--api-base`, `--source`, `--extensions`,
     `--batch-size` (16, max 64), `--concurrency` (16, max 32), `--poll-interval`, `--no-wait`,
     `--show-failures`, `--project`.
   - Exit codes 0/1/2/130 (`:57`). Paste real output captured from a public run.
   - Mention `scripts/curation/ingest_upload.py` for shares the server cannot mount (it copies
     bytes into the upload root).
7. **Full run, Cropwright.**
   - Ingest page, "Bulk ingest (folder or manifest)" panel
     (`frontend/src/lib/components/ingest/BulkIngestPanel.svelte`).
   - Screenshots `cropwright/ingest-bulk-panel-1600.png` and `cropwright/ingest-bulk-run-1600.png`
     (section 4).
8. **Progress, cancel, resume, failures.**
   - `GET /curation/projects/{project}/ingest/bulk/{run_id}` (fields `discovered`,
     `discovery_complete`, stage table).
   - `POST /curation/projects/{project}/ingest/bulk/{run_id}/cancel`; Ctrl-C once cancels, twice
     detaches.
   - Resume: re-POST the same source. There is no resume route.
     `src/services/curation/bulk_ingest/runner.py:99-140` matches by kind, path and extensions
     for runs that were interrupted, failed, cancelled or completed_with_errors. A completed
     source starts a new run.
   - One run per project; 409 `bulk_ingest_busy`.
   - Failures: `GET /curation/projects/{project}/ingest/bulk/{run_id}/failures` with
     `offset`/`limit` up to 1000.
   - The run stops after 5 consecutive fully failed batches (`limits.py:22`).
   - Walker caveats: symlinks are not followed and unreadable directories are skipped silently
     (S31).
9. **Throughput and time estimate.**
   - Quote only measured rows, with hardware and a link to the `docs/PERFORMANCE.md` section:
     - bulk runner, COCO-size frames, one RTX A6000: about 21 img/s
       (`PERFORMANCE.md:1706-1736`, "Bulk ingest runner");
     - stage 2 end to end, set A: 28.81-29.07 img/s on one card, 55.08 on two
       (`:3100-3109`);
     - 12-24 MP photos, `/ingest/batch` 8 x 16: 27.0 img/s on the CPU path, 35.0 with GPU decode,
       44.0 detect only (`:3121-3148`);
     - host CPU 1.12 against 0.34 s per photo.
   - Formula: `hours = images / (img_per_s x 3600)`. Tell the user to read their own rate from the
     trial (images / wall time) rather than trust the table.
   - Say that these are A6000 numbers and that a 12 GB card and other profiles were not measured
     (`:3112-3114`).
10. **GPU decode for large photos.**
    - `OP_INGEST_GPU_DECODE` has code default on, and the installer writes 1 for `full` and 0 for
      minimal and standard (`setup-openprocessor.sh:1074`, `scripts/lib/config.sh:708-726`).
    - Size gate `OP_INGEST_GPU_MIN_MEGAPIXELS=3.0`. It costs about 8.5 GB extra VRAM on the Triton
      card.
    - Non-baseline JPEGs fall back to the CPU (`op_ingest_decode_total`).
    - It applies to the `/ingest/batch` path, which bulk uses.
11. **Disk estimate.**
    - Folder ingest should not copy source images (the mount is read-only) while upload ingest stages
      copies under the project upload root; confirm both in `src/services/curation/` before writing.
    - Index size: per-1,000-image figures from `PERFORMANCE.md:329-366` (22.4 MB narrow detector
      all embedded; 79.8 MB full vocabulary all embedded; 21.0 MB none embedded).
    - Lucene item cost from `:3229-3238`. Extrapolation labelled as such (`:3264-3270`).
    - Use `POST /curation/projects/{project}/ingest/policy/preview` (`estimated_vector_mb`) on the
      trial project.
    - Add OpenSearch heap and HNSW RAM from `storage_sizing_and_ingest_baselines.md:80-84`.

Also in T2:
- update `docs-site/docs/cropwright/user-guide/ingest.md` (UI steps only, link to the new page);
- expand `docs-site/docs/guides/ingest-policy.mdx` (fields of `detect`, `embedding.mode`
  `all|selected|lazy`, `detector`; S33; preview);
- fix the `bulk_ingest.py` module docstring usage lines (S2), docstring only.

## 3. Spec: pipeline modes (T3)

New page `docs-site/docs/guides/pipeline-modes.mdx`. Open with a table: mode, what runs, how to
select it, what you cannot turn off today. Every mode gives the exact commands.

**Knobs that exist today** (verified):
- GPU profile `minimal|standard|full` (`config_templates/profiles/*.json`). It tunes Triton
  instances, batch, API workers, shm and GPU decode. It does not choose models.
- Compose profiles: `curation`, `vlm`, `segmenter`, `training`, `monitoring`.
- Per-project ingest policy (`GET/PUT /curation/projects/{project}/ingest/policy`).
- Per-project VLM activation: `POST /curation/projects/{project}/vlm/endpoints/{name}/activate`
  and `POST /curation/projects/{project}/vlm/endpoints/deactivate`, which sets the explicit `off`
  state (`src/routers/curation/vlm_activation.py:106`).
- Per-project VLM policy: `GET/PUT /curation/projects/{project}/vlm/policy` with
  `scope: all|uncertain|representatives|off`, `per_cluster`, `max_crops_per_day`, `sample_frac`
  (`src/services/curation/vlm_policy.py:29-43`).
- Region profile and prompt pack activation (config store).
- `OP_IVF_RETRAIN_CHECK_S=0`, which disables the auto-label worker's idle retrain
  (`scripts/curation/auto_label_worker.py:126`).
- `OP_REGION_EMBED_ENABLED=false`.
- The VLM CLI `./openprocessor vlm list|status|use|apply|probe|key set`
  (`openprocessor:907-953`, `scripts/lib/vlm_switch.sh`), the catalog
  `examples/vlm/catalog.tsv`, and `OP_LOCAL_VLM_ENDPOINT`.

**The four modes.**

**(a) Detection only.**
- Core stack only (`./openprocessor start` with no curation profile), plus the core `/detect`
  route for stateless use.
- For curation storage without embeddings, use ingest policy `embedding.mode=lazy`.
- **Must state the limits.** The whole-frame embedding is still computed and stored (S33). The
  curation stack keeps clustering via two triggers: the `curation-cluster-refresh` daemon and the
  auto-label worker idle retrain.
- To stop clustering: do not start `curation-cluster-refresh`; set `OP_IVF_RETRAIN_CHECK_S=0`;
  deactivate the VLM per project.
- Start a subset of workers with
  `docker compose -p <project> --profile curation up -d curation-detection-worker`. There is no
  CLI for a subset (F8).
- Gap issue F2.

**(b) Detection + embedding + clustering.**
- This is the default: policy `embedding.mode=all`, `curation` profile up.
- VLM off per project: deactivate, or VLM policy `scope=off`.
- Cluster refresh every 120 s after 200 new items. Its knobs are CLI flags only
  (`cluster_refresh_daemon.py:89-133`), changed by a compose override of `command`.
- Mention the 409 `audit_required` behaviour (F3) as a known issue with its issue number once
  filed.

**(c) VLM-assisted labeling of a large batch.**
1. `--profile vlm` or a remote endpoint.
2. `./openprocessor vlm use <id>`.
3. Activate per project.
4. Set the VLM policy scope and budget.
5. The continuous `curation-vlm-worker` labels embedded items, or a one-shot
   `POST /curation/projects/{project}/pipeline/auto_label/start` with `run_vlm=true` and
   `embed_missing=true` if the policy was `lazy`.

The VLM needs embeddings (`scripts/curation/vlm_worker.py:199`). A deactivated project returns 409
`vlm_not_configured`. State S32 (the VLM is on by default when `OP_VLM_URL` is set). Link
`guides/vlm-selection` for endpoint details.

**(d) Iterative cluster-sampled auto-labeling: PLANNED, not shipped.** Use an admonition
`:::info Planned`.
- What exists today: the `representatives` scope labels the `per_cluster` crops nearest each
  centroid once (`src/services/curation/vlm_scope.py`, `vlm_rep_claims.py`);
  `POST /curation/projects/{project}/clusters/auto_promote` with audit gate; AHC sub-cluster ids
  via `clusters/refine/{cluster_id}`.
- What is planned: an iterative loop (sample, label, measure agreement, propagate or split into
  sub-clusters, repeat). Link issue F9 once filed, plus #97 and #218. No design doc exists; do not
  describe an interface.

Also in T3:
- `docs-site/docs/operations/workers.mdx` (S9, S10, S11);
- `getting-started/compose-profiles.mdx` (S10);
- `deployment/gpu-sizing.mdx` (S8, mark `profile` as broken until F4 is fixed and give the manual
  `.env` edit instead);
- `guides/vlm-selection.mdx` (S32, the explicit off state);
- `operations/curation-workflow.mdx` (mode table link, F3 note);
- `INSTALLATION.md` "GPU Profiles" section `:473-521` (S8). T3 owns only that section; T1 owns
  the rest of INSTALLATION.md, and T3 edits only between those headings. This is the one shared
  file; T3 merges after T1.

## 4. Spec: screenshots (T6)

**Tools.**
- `frontend/scripts/capture_docs_screenshots.py` (Cropwright).
  - Header `:1-45`; read-only browser guard `:84-154`.
  - Route list `frontend/scripts/screenshot_routes.json`.
  - Output `docs-site/static/img/screenshots/cropwright/`.
  - Run with `frontend/e2e/.venv/bin/python` after `npm run test:e2e` has created that venv once.
- `scripts/docs/capture_backend_screens.py` (Swagger, Grafana, MLflow, OpenSearch). It refuses
  ports 4600-4799.
- Pages embed with `<Screenshot name="cropwright/<file>.png">`
  (`docs-site/src/components/Screenshot/index.tsx:41`).

**Data rule (hard).** Public data only. Use project `coco-bench-2k` (COCO) or `openimages-plates`
(Open Images).
- The running local stack (compose project `opmono`, API host port 4953, Cropwright 5284; verify
  with `docker ps`) **also holds non-public projects**.
- So on that stack, capture only project-scoped pages with `--project coco-bench-2k`.
- Never capture these on that stack, because they list every project slug: `projects`,
  `projects/combine`, the project switcher, `clone-from-project`, `models-sharing*`, and the
  resources menu if it lists projects.
- Global-page shots may be refreshed only from an isolated stack (`-p opdocs`, GPUs 0/2) seeded
  with `make sample-coco` and `make sample-plates`. Otherwise keep the existing global shots,
  which commit history records as public captures (`fb77fd75`, `c09afedd`).
- The script does not enforce any of this (F7). Every PNG must be opened and viewed before commit,
  looking for slugs, hostnames, paths and class names. Writes for fixtures use only `cwlife-*`
  projects (`frontend/scripts/docs_screenshot_fixtures.py`).

**New captures** (names fixed here so that pages can embed them in parallel). For each, add a
route or `STATES` entry in the script. Any state that needs a write gets its data set up by curl
outside the browser, into a `cwlife-*` project.

| File (under `cropwright/`) | Page/state | Embedded in |
|---|---|---|
| `ingest-bulk-panel-1600.png` | Ingest page, bulk panel with roots listed | `guides/ingest-large-folder.mdx`, `cropwright/user-guide/ingest.md` |
| `ingest-bulk-run-1600.png` | bulk run in progress with stage table (start the run by curl on a `cwlife-*` project from public COCO) | same |
| `ingest-bulk-failures-1600.png` | failures list (manifest with 2 missing paths) | `ingest-large-folder.mdx` |
| `ingest-policy-1600.png` | `settings/ingest-policy` full page | `guides/ingest-policy.mdx`, `cropwright/user-guide/ingest-policy.md` |
| `vlm-endpoint-off-1600.png` | `settings/models` for a deactivated project | `guides/pipeline-modes.mdx`, `guides/vlm-selection.mdx` |
| `audit-1600.png` | `p/[project]/audit` | new `cropwright/user-guide/audit.md` |
| `cluster-detail-1600.png` | `clusters/[id]` | `cropwright/user-guide/clusters.md`, `pipeline-modes.mdx` |
| `dataset-imports-1600.png` | `datasets/imports` list | `cropwright/user-guide/dataset-import.md` |
| `prompt-packs-list-1600.png`, `region-profiles-list-1600.png`, `open-vocab-list-1600.png` | settings list pages | the matching Cropwright pages |

**Currency of existing shots.** Re-capture the 15 routes for `coco-bench-2k` into a scratch dir
(`--out artifacts_local/screens_check/`, gitignored), compare visually, and replace only those
whose UI changed. Do not replace global pages unless they come from the isolated stack.

**Regeneration.**
- Add `make docs-screenshots` (Makefile, T6). It runs
  `cd frontend && e2e/.venv/bin/python scripts/capture_docs_screenshots.py --base-url $(CROPWRIGHT_URL) --project $(DOCS_PROJECT)`.
- `CROPWRIGHT_URL` is required with no default. `DOCS_PROJECT` defaults to `coco-bench-2k`.
- Document it in `docs-site/docs/developer-guide/screenshots.mdx`, the single screenshot guide.
  `cropwright/developer-guide/screenshots.md` becomes a short UI-specific state table that links
  to it.
- Script changes need a `frontend/CHANGELOG.md` entry (`.github/workflows/frontend.yml:93-127`).

**Text "screenshots".** For CLI steps paste real captured terminal output from a public project
(bulk_ingest progress, `./openprocessor vlm list`, `gpu plan`) in code blocks; do not fabricate
output.

## 5. Rules every edit obeys

1. Routes named in docs exist in `contracts/openapi/curation.json` or the app. `OP_*` names are
   read by code. Relative links resolve: `.venv/bin/python scripts/docs/check_docs_vs_code.py`.
   Name no planned route or `OP_*` variable; describe planned work in words.
2. Docs name no release version except VERSION `0.5.0` with a `v` prefix
   (`tests/test_version_consistency.py:53-60`). Apply this to Cropwright `.md` pages too, even
   though the test does not scan them.
3. Naming-leak scan: `.venv/bin/python scripts/codegen/check_naming_leaks.py`. It forbids private
   product, company and domain names, private host paths and some model and domain terms (patterns
   in the script; its allowlist is `scripts/codegen/naming_leak_allowlist.txt`, entries require a
   `# reason`). Use `/srv/nas/...` style example paths. Never use a private project slug.
4. No private data in screenshots or pasted output (section 4).
5. Docs site builds: `cd docs-site && npm ci && npm run build` (or `make docs-build`). Broken links
   and anchors fail the build. Also run `npm run typecheck` when touching `docs-site/src`. npm only.
6. Keep `tests/test_generic_default_docs.py` true: no `OP_INGEST_PRIMARY_CLASS_IDS=<digits>` and no
   `2,3,5,7`; quick-start keeps "all 80 COCO classes" and "Switching detectors". Keep the README
   tests (`tests/test_readme_*`) passing.
7. Plain, task-first writing: numbered steps, one copy-paste block per step, the expected result
   after each block, troubleshooting at the end. State limits plainly.

## 6. Work breakdown

Branch `docs/overhaul` from `main`. Each task is one Sonnet implementer working in its own worktree
off `docs/overhaul`, committing with `docs(<scope>): ...`. File ownership does not overlap, except
INSTALLATION.md, which T1 and T3 split by section (T3 merges after T1).

**Wave 1, in parallel:** T1, T2, T4, T5, T6, T7.
**Wave 2:** T3 (after T1), then T8.

Every task runs before handing back:

```bash
.venv/bin/python scripts/docs/check_docs_vs_code.py
.venv/bin/python scripts/codegen/check_naming_leaks.py
.venv/bin/python -m pytest tests/test_version_consistency.py tests/test_doc_links.py tests/test_docs_vs_code.py tests/test_generic_default_docs.py tests/test_readme_sample_walker_and_docker_test_path.py tests/test_readme_search_and_landmarks_docs.py -q --no-cov
cd docs-site && npm run build
```

**T1: Getting started and configuration (new and existing users).**
- Owns: `README.md`; `INSTALLATION.md` (except "GPU Profiles"); `env.template` (comments only: S4, S13, S6 note); `docs-site/docs/getting-started/{introduction,quick-start,installer,model-export,second-stack}.mdx`; `docs-site/docs/configuration/{basic,advanced}.mdx`; new `docs-site/docs/getting-started/choose-your-path.mdx`.
- `choose-your-path.mdx` has three paths: inference API only; curation with Cropwright; existing user upgrading: `./openprocessor upgrade`/`update`, `restart-after-update`, retired variables rejected at startup (`src/config/retired_env.py`).
- Fixes S3, S5, S6, S7, S15 (README).
- README is shortened to the overview, the one-line install, a 10-minute first result and links
  into the site. Keep all content the README tests assert.
- Accept: the guards pass; every command in quick-start runs on a fresh checkout (run it); no
  `--root` remains in the bulk_ingest context.

**T2: NAS ingest guide** (section 2).
- Owns: new `docs-site/docs/guides/ingest-large-folder.mdx`; `docs-site/docs/guides/ingest-policy.mdx`; `docs-site/docs/cropwright/user-guide/{ingest,ingest-policy}.md`; the `scripts/curation/bulk_ingest.py` docstring.
- Accept: every flag, route and env var in the page is grep-verified; the trial steps run against `coco-bench-2k`, or a `cwlife-*` project with public COCO, on the local stack, and the pasted output comes from that run; no number appears without a `PERFORMANCE.md` citation.

**T3: Pipeline modes** (section 3).
- Owns: new `docs-site/docs/guides/pipeline-modes.mdx`; `docs-site/docs/guides/vlm-selection.mdx`; `docs-site/docs/getting-started/compose-profiles.mdx`; `docs-site/docs/operations/{workers,curation-workflow}.mdx`; `docs-site/docs/deployment/gpu-sizing.mdx`; `INSTALLATION.md` "GPU Profiles" section; `docs-site/docs/cropwright/user-guide/vlm-models.md`.
- Accept: each mode lists its commands, and each "cannot" links a section 8 issue (use placeholder text `(issue to be filed)` until T8 fills the numbers); mode (d) is inside a Planned admonition.

**T4: Repository docs and the CURATION.md map.**
- Owns: `docs/CURATION.md`; `docs/README.md`; `docs/design/README.md`; status notes in `docs/design/{gpu_decode_plan,storage_sizing_and_ingest_baselines,v060_wave_plan}.md`; `docs/PERFORMANCE.md` (S16 only); `docs/market/*` (S22); `src/README.md` (docs only); `benchmarks/README.md`; `docs-site/docs/about/vision-and-goals.mdx`.
- First produce a section-to-page mapping table: CURATION.md heading, site page, content missing
  there. Content missing from a site page goes to the owning task as a note in the PR description;
  do not edit pages another task owns. Then cut CURATION.md down to the map, keeping the cited
  headings (decision 2).
- Fix S1, S2 (map), S12, S16-S22.
- Accept: `rg -n 'CURATION.md' src scripts docker-compose.yml env.template Makefile` citations all still resolve to existing headings; the doc tests pass.

**T5: Cropwright docs.**
- Owns: `docs-site/docs/cropwright/**`, except the files T2, T3 and T6 own; `frontend/README.md`; `frontend/docs/README.md`, `frontend/docs/FEATURES.md`; the doc sections of `frontend/CLAUDE.md` (`:30`, `:797`, `:2414-2441`, `:2765`, and the sections that name an older release).
- Apply decision 3: remove the concept prose duplicated from `guides/*` and link instead.
- New `cropwright/user-guide/audit.md` (route `p/[project]/audit`; API `GET /curation/projects/{project}/audit/report`
  and `/audit/queue`).
- Fix S26 (annotation-profiles), S27.
- Accept: the guards pass; there are no "Cropwright repo" or standalone docs-site references (an
  rg check for `cropwright-docs` returns nothing in docs).

**T6: Screenshots and screenshot tooling** (section 4).
- Owns: `frontend/scripts/capture_docs_screenshots.py`; `frontend/scripts/screenshot_routes.json`; `frontend/scripts/docs_screenshot_fixtures.py`; `docs-site/static/img/screenshots/cropwright/*` (new files); delete `docs-site/src/data/screenshot_routes.json`, fix `docs-site/src/components/ScreenshotShowcase/index.tsx:22`; `docs-site/docs/developer-guide/screenshots.mdx`; `docs-site/docs/cropwright/developer-guide/screenshots.md`; `Makefile` (`docs-screenshots` target, S24 echo); `frontend/CHANGELOG.md`.
- Accept: all section 4 files exist and were viewed; `make help` lists `docs-screenshots`; `npm run typecheck` and the build pass; `cd frontend && npm run check` passes if anything there changed.

**T7: Operations, deployment and API reference.**
- Owns: `docs-site/docs/api-reference/*.mdx` (S15); `docs-site/docs/operations/{health-and-stalls,exports-and-retention,training-and-promote,monitoring,release-acceptance}.mdx`; `docs-site/docs/deployment/{security,sizing-and-storage}.mdx`: sizing uses Lucene per-item figures from `PERFORMANCE.md:3229-3238` and links the T2 disk section; new `docs-site/docs/operations/upgrade-and-backup.mdx`: `./openprocessor upgrade|update|repair`; what lives in `data/` and OpenSearch; verify each subcommand in `openprocessor` before writing; `monitoring/README.md`.
- Accept: guards pass; every CLI subcommand named is found in `openprocessor`.

**T8: Integration and verification** (runs last, on `docs/overhaul` after T1-T7 merge into it with
`--no-ff`).
- Owns: `docs-site/sidebars.ts`; `docs-site/docusaurus.config.ts` (S25); `docs-site/src/data/screenshots.json`; the five duplicate top-level PNGs (S30): repoint the gallery JSON and any page at `cropwright/` and delete the copies only after `rg -n "review-1600|clusters-1600|train-1600|bakeoff-1600|review-regions-1600" docs-site` shows no top-level users; `CHANGELOG.md`.
- Sidebar: Getting Started (introduction, choose-your-path, quick-start, installer, model-export,
  compose-profiles, second-stack); Guides grouped as:
  - "Bring in data": ingest-large-folder, ingest-policy, dataset-import
  - "Choose a pipeline": pipeline-modes, vlm-selection, region-profiles, prompt-packs,
    open-vocabulary, multi-box-regions
  - "Organize and review": projects, settings-and-config, keymaps, reprocess, test-on-crop,
    combine-projects
  - "Examples": wheel-example, use-your-own-domain

  Operations gains upgrade-and-backup. The Cropwright User Guide gains audit.
- File the section 8 issues (`gh issue create`) and replace the `(issue to be filed)` placeholders
  with the numbers.
- Run section 7.

## 7. Integration checks and PR

```bash
.venv/bin/python scripts/docs/check_docs_vs_code.py
.venv/bin/python scripts/codegen/check_naming_leaks.py
.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'
cd docs-site && npm ci && npm run typecheck && npm run build && cd ..
rg -o 'name="[^"]+\.png"' docs-site/docs | sed 's/.*name="//;s/"$//' | sort -u | while read f; do test -f "docs-site/static/img/screenshots/$f" || echo "MISSING $f"; done
rg -n '\bv0\.[0-9]+\.[0-9]+' docs-site/docs frontend/README.md frontend/docs | rg -v 'v0\.5\.0'   # expect none
rg -n 'cropwright-docs|--root .*bulk_ingest|CW_GH_REPO|SLOW_REQUEST_THRESHOLD_MS' -g '*.md' -g '*.mdx' .   # expect none
.venv/bin/pre-commit run --all-files
```

Then open the built site (`cd docs-site && npm run serve`, then `DISPLAY=:11` browser, or a
Playwright full-page capture at 1600 and 430 px) and view at least:
- choose-your-path;
- ingest-large-folder;
- pipeline-modes;
- one Cropwright page with a new screenshot.

**PR.**
- `CHANGELOG.md` `[Unreleased]` → `### Changed` gets one entry: documentation overhaul; a single
  source of truth in docs-site; new guides for ingesting a large folder, pipeline modes,
  choose-your-path, upgrade and backup, and audit; refreshed screenshots; `make docs-screenshots`.
- Open the PR against `git rev-parse --abbrev-ref @{u}` (main) with `gh pr create`. Merge with a
  merge commit, never squash.
- Push nothing without the owner's yes.

## 8. Follow-up issues (code gaps; file them, do not fix them in this PR)

- **F1. feat(ingest): preflight for bulk ingest (file count, bytes, estimated time and disk).** Today there is no dry-run or count option: no flag in `scripts/curation/bulk_ingest.py:67-101`, `BulkIngestRequest` is `extra='forbid'`, and there is no preflight route. A run's `discovered` grows while it ingests. Add a read-only preflight that walks the folder or manifest with the same walker (`bulk_ingest/walk.py`). It should return: the count; total bytes and a megapixel histogram (the GPU decode gate); skipped and unreadable directories and symlinks; an estimate from the policy preview and measured rates. Expose it as a CLI flag and in the Cropwright panel.
- **F2. feat(curation): a true detection-only mode.** Under `embedding.mode=lazy` the whole-frame PE embedding is still computed and stored (`ingest_index.py:216-226`, `ingest_gpu_decode.py:168-174`), and ingest-time IVF placement has no toggle (`:265-303`). Clustering runs from two triggers with no per-project off switch. Proposal: a policy field to skip the whole-frame vector and a per-project clustering switch that both the daemon and the idle retrain honour. Relates to #67, #65.
- **F3. fix(cluster-refresh): an audit-gate 409 stops the retrain.** `scripts/curation/cluster_refresh_daemon.py:167-181` posts `/clusters/auto_promote` with `raise_for_status`. The route raises `AuditGateError` → 409 when a promotable class lacks an audit (`src/services/curation/clustering/auto_promote.py:455-478`, `src/routers/curation/clusters.py:495`). The daemon then returns before queuing the auto-label retrain (`:288-293`), so clustering never refreshes for that project. Treat a 409 as "nothing promoted" and continue. Add a test.
- **F4. fix(cli): `openprocessor profile` does not switch profiles.** `generate_env_file` is called without force, so an existing `.env` keeps its `GPU_PROFILE` (`openprocessor:505-516`, `scripts/lib/config.sh:680-683`). The `docker-compose.override.yml` it writes is never loaded (`dc()` passes explicit `-f`), and it would pin `device_ids: ['0']`. The restart hint names `./scripts/openprocessor.sh`. The profile JSON `models` list is never read, and PE has no profile tuning (`scripts/lib/config.sh:639-672`).
- **F5. feat(vlm): make VLM use explicit per project.** With `OP_VLM_URL` set, every project that never activated an endpoint labels with the env endpoint (`src/services/labeling/vlm_endpoints.py:446-447`). Decide on an opt-in or a visible "inherited from env" state in `GET /curation/projects/{project}/vlm/endpoints/active` and Cropwright.
- **F6. feat(compose): more than one source root.** `OP_SOURCE_ROOT_HOST` is one read-only root. `OP_SOURCE_PATH_ALIASES` roots are accepted by the API but need hand-written bind mounts on three services. Provide a supported mechanism (generated override or documented include) and a startup check that every alias path exists in each container.
- **F7. feat(docs-tooling): enforce the public-data guard in `capture_docs_screenshots.py`.** Only a printed warning exists (`:831`, `:873`). Add a required project allowlist, and refuse global pages when `GET /curation/projects` returns any slug outside it.
- **F8. feat(cli): start a subset of curation workers.** `./openprocessor curation up` and `make curation-up` start all five services, including the GPU-reserving evaluator (`openprocessor:578-616`, `Makefile:1174`). Accept service names, and stop starting the on-demand evaluator with `up -d`.
- **F9. feat(curation): iterative cluster-sampled VLM auto-labeling (planned).** Loop: sample per cluster and sub-cluster, label with the VLM, measure agreement, propagate when agreement passes the gate, split and recurse otherwise. Builds on the `representatives` scope, `clusters/refine`, and the agreement-gated auto-promote. No design doc or issue exists. Needs a plan doc in `docs/design/`. Relates to #97, #218.
- **F10. test(docs): close the docs-guard gaps.** Covers section 1.3: scan Cropwright `.md` and the `frontend/` docs in the version test; check screenshot names resolve; check `./openprocessor` subcommands, `make` targets and non-`OP_` variables named in docs; trigger `docs.yml` on `docs/**`; add an external link check on a schedule.
- **F11. fix(bulk_ingest): a default `--api-base` that works inside the container.** The default targets host port 4603 (`bulk_ingest.py`) while every documented invocation runs inside `api` on port 8000.
- **F12. chore: stale code comments.** One small PR for: `openprocessor:572-573`; `embedding_state.py:21`; `.pre-commit-config.yaml:1`; `frontend/docker-compose.yml:75-80` (the `docs` service using a retired image); `docs-site/.nvmrc` (20) against `docs-site/Dockerfile:2` (node 26).
