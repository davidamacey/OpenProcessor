# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The docs screenshot capture (`scripts/capture_docs_screenshots.py`) now requires `--public-projects` and
  refuses any project outside that list, and refuses pages that list every project while the backend holds
  another project, so a capture can never show private data.

### Added

- The ingest-policy page offers the `none` embedding mode (detection only: no vector of any kind) and a
  "Cluster automatically" checkbox (`clustering.automatic`), both written to the policy the backend serves.

- The Bulk ingest panel has a **Preview** button: it asks `POST /ingest/bulk/preview` what the folder or
  manifest holds and shows the served file count, bytes, extensions, image-size sample, skipped entries
  and the estimated time and index size with the measured figures behind them. A capped scan is marked
  "partial" (every figure a lower bound); pressing the button again while it scans cancels it.
- `/ingest` has a "Bulk ingest (folder or manifest)" panel, shown with the server-path ingest it sits beside: start a
  run over a folder or a manifest the backend has mounted, follow served progress (counts, speed, time left,
  seconds per stage), cancel, resume, and read the failed images with their reasons. Everything shown and every
  allowed action (`can_cancel`, `can_resume`) comes from the backend's `/ingest/bulk*` routes.

- `/ingest` has a "Sources (S3 buckets and watched folders)" panel: list, create, edit and remove sources (S3
  bucket and prefix or a server folder; poll interval, cron schedule, webhook), **Run now** and **Retry** for
  objects that gave up. Counts, trigger times, the webhook path, the last run, failures, `can_run` and the
  `blocked_reason`/`blocked_message` are served by `/ingest/sources*`; the panel renders them and refreshes after the
  served `poll_after_s`. No credential is entered or shown.

### Changed

- **BREAKING**: the container reads `OP_CROPWRIGHT_INGEST_MAX_REQUEST_MB` and
  `OP_CROPWRIGHT_DATASET_UPLOAD_MAX_MB` (were `CROPWRIGHT_*`), and `docker-entrypoint.sh` exits 2 at
  start while a retired `CROPWRIGHT_*` name is still set. The browser-side `window.__CROPWRIGHT_*__`
  globals are unchanged. `.env.example` and the README drop the container-name variables only the
  removed standalone compose used.
- The oversize frontend files are split by concern, with no behavior change. `lib/api.ts` is the HTTP core and
  the calls live in `lib/api_<domain>.ts` modules (`api_errors`, `api_projects`, `api_review`, `api_clusters`,
  `api_train` and the rest); `lib/types.ts` keeps the core entities and the rest are `types_<domain>.ts`. The
  review, train, export, clusters, cluster detail and classes pages are thin pages over controllers in
  `lib/<area>/*.svelte.ts` and components in `lib/components/<area>/`; `CropMetaPanel`, `KeymapCard`,
  `TrainForm`, the slot-config parser and the slot gallery controller are split the same way, the e2e
  `conftest.py` is a set of `e2e/harness/` plugins and `capture_docs_screenshots.py` is the `docs_screenshots/`
  package. Oversize test files are split by behavior with the same test cases.

- Every dialog is the one shared `Modal` component: `role="dialog"`, `aria-modal`, an accessible name, focus
  moved in on open and trapped while open, Escape and a click on the dimmed backdrop close it, and focus returns
  to the control that opened it. The 28 hand-rolled dialogs and their `a11y_click_events_have_key_events`
  suppressions are gone; the lightboxes in the card and the embedding plot, and the keymap save and reset
  confirms, now also close on Escape and carry a name. A scan test fails on a hand-rolled dialog.

- Durations read `1h 05m`, `5m 03s` or `42s` on every panel (dashboard ETA, training ETA, auto-label elapsed and
  ETA, bulk ingest); the dashboard no longer prints minutes alone or days, and the training ETA now shows
  seconds. One byte formatter (`/projects` heap size now reads B, KB, MB, GB or TB like the combine pages) and
  one time-of-day formatter replace the copies; the bake-off run list shows its start time as UTC
  (`2026-09-24 21:13:41 UTC`) like the other served timestamps. All formatters live in `src/lib/format/`.

- The three server-event subscriptions (project events, scoped events, pipeline events) share one open,
  backoff and reconnect helper; behavior is unchanged and now covered by tests.

- The `$lib`, `$components`, `$stores` and `$contracts` import aliases are configured in `vite.config.ts` and
  `tsconfig.json` instead of the deprecated SvelteKit `alias` option; the build no longer prints the
  deprecation warning, and CI fails if it comes back.

- `frontend/CLAUDE.md` is cut from 3,104 to about 300 lines of current rules (history lives here).

- Dead code removed: 46 exports no other file used are now file-local, an unused declaration file and the `tslib`
  dev dependency are gone, and `knip` runs in CI to keep it that way.

- `ProjectCounts.images` and `.items` are `number | null` (the API serves `null` when OpenSearch could not be counted); the project list already renders `null` as an em dash.

- Work-package ids and history names are gone from comments, JSDoc and test titles in `src/` and `e2e/`,
  and the route tests under `src/routes/p/[project]/` are named by what they test (for example
  `reviewDismissedPanel`, `clusterPurityDisplay`, `servedTrainingCohorts`) instead of by the pass that
  wrote them. Same 4,575 tests; no rendered change.
- `scripts/capture_docs_screenshots.py` gains states for the sources panel and the bulk-ingest panel (idle, running, failures), the ingest
  policy page, a project with its VLM off, the audit page, a cluster page, the dataset imports list and the prompt-pack,
  region-profile and open-vocabulary lists. `scripts/docs_screenshot_fixtures.py` gains `bulk` and `vlm-off` to set up
  throwaway `cwlife-*` projects for them.

### Fixed

- Review, multi-box tabs: after Tab moved to the next box, the response of the previous accept or reject re-seeded the
  controller and snapped the selection back to box 1, so a fast `R`, `Tab`, `Y` accepted the wrong box. A re-seed of
  the same crop now keeps the selection (clamped to the boxes that remain).
- Infinite-scroll grids (`createPager`) stop requesting pages after three in a row add nothing. A server whose
  `total` outruns what it serves (it still counts a crop just moved out of `/clusters/[id]`) made the scroll
  sentinel fetch pages without end.
- Stubbed e2e suite: a key pressed right after a write's response met the app before it had recorded the undo entry
  ("Nothing to undo", then a 45 s `expect_response` timeout), and a boot chunk fetched with `socket hang up` was not
  retried. The tests now wait for the app's own signal (the success toast) and the harness retries both spellings
  of a closed keep-alive socket.
- Stubbed e2e suite: the project-switch test judged "requests to the previous project" by an index into the stub's
  answered list, which counted requests sent before the switch; it now listens to the requests the page sends. The
  harness gains `E2E_CPU_THROTTLE` to run the suite as a slow runner would, and its docstring states the stability
  rules.

- `/clusters` semantic search no longer throws `each_key_duplicate` (and re-requests `/search/text` in a loop)
  when the backend serves one crop id more than once: the first, best-ranked hit per id is kept.
- At 430 px the ignore-reason menu on `/clusters/[id]` opens inside the viewport, the `/train` past-runs table keeps a
  readable Name column (the table scrolls sideways instead), and the `/clusters` error state shows no count.
- A per-box accept or reject key (`y`/`r`) pressed while the previous box write is still in flight is applied in
  order instead of being silently dropped; this was the cause of the intermittent
  `test_per_box_accept_and_reject_keys_patch_immediately_without_removing_from_queue` CI timeout.

### Removed

- Standalone-repository leftovers: the release scripts (`scripts/release.sh`, `scripts/release/`,
  `scripts/release-smoke.sh`), the frontend's own compose files, pre-commit config, diagram and GIF scripts,
  and its own `SECURITY.md` and `CONTRIBUTING.md`. The Cropwright image ships through the repository's
  `scripts/release/build_and_publish.sh`; the README and docs point at the root files.

## [0.5.0] - 2026-10-09

Cropwright joins the OpenProcessor repository as `frontend/` and from here on is versioned with
the monorepo: this release is 0.5.0, not 0.1.2, and requires OpenProcessor 0.5.0 (the earlier
standalone 0.1.x numbering ends at 0.1.1). Its image is built from `frontend/Dockerfile` by the
repository's release flow and pinned in `images.lock`.

### Added

- The served pack schema gains a `detector_hint_min_confidence_pct` row with `kind: "int"` (the optional per-item detector-name hint, 0 = off). The pack editor shows an unknown kind read-only and saves it unchanged, so the value round-trips; `PackSchemaField.kind` lists `int`.
- Gateway mode: with `OP_GATEWAY_SUBPATHS=true` the entrypoint installs
  `nginx-gateway.conf`, which proxies `/grafana/`, `/prometheus/`, `/dashboards/` and
  `/mlflow/` to the compose services (bare service names only, overridable with
  `GRAFANA_UPSTREAM`, `PROMETHEUS_UPSTREAM`, `DASHBOARDS_UPSTREAM`, `MLFLOW_UPSTREAM`),
  with the websocket upgrade for Grafana Live. No ports are hardcoded in the app.
  With the switch off those four paths answer 404 instead of the app shell, and an
  `https` `X-Forwarded-Proto` from an outer TLS proxy is kept on the gateway locations.
- Label confirmation (#119): a VLM label is a suggestion until a human validates it.
  - `/settings` has a **VLM scope** panel (`GET/PUT /vlm/policy`): the four served scopes
    (everything, uncertain only, cluster representatives, off), the knobs each scope reads
    (confidence limit, representatives per cluster, random sample fraction, crops per day),
    a save that sends the revision it read, and Reload / Keep my edits on a 409.
  - A crop card, the item detail and `/review` show the detector's own class and score
    beside a VLM-sourced label, and the label source reads "VLM suggestion",
    "Human-confirmed" or "Auto-validated" (from the served source role and the validated flag).
  - `/review` has a **Detector Disagreements** tab, offered only while `GET /review/tabs`
    serves it (label and description as served); a `?tab=detector_disagreements` link
    against a backend without it falls back to All with the usual notice.
  - `/audit` (new route and nav link): draw a stratified sample of machine-labelled crops
    (`POST /audit/start`), see the served detector and VLM precision per class with Wilson
    95% intervals and the `insufficient_sample` flag, the confusion matrix, the outcome
    counts and the queue of crops waiting for a human label (each links to `/review`).
  - `/export` says how many of the crops are validated and that only validated crops are
    exported; `/dashboard` says labels are not ground truth and names the VLM row "VLM suggestions".
- The item wire carries `detector_class_name`, `detector_class_id` and `detector_confidence` (the detector's own class, kept next to the VLM or human label); `RawCrop` and the test fixtures list them.
- The frontend moved into the OpenProcessor repository with its history (OpenProcessor #85). Its tests read the repository's generated `contracts/` directory instead of a vendored copy, and the Cropwright documentation is a section of the one docs site (`docs-site/`).

### Changed

- The region-profile body and schema no longer carry `ocr_det_model`, `ocr_det_version`,
  `ocr_det_input_size` or `ocr_det_prob_floor` (OpenProcessor #181), and `choices_from`
  no longer offers `ocr_det_models`; fixtures and the contract pin follow. The
  read-only OCR detector list in the vocabulary panel is unchanged.
- Moved to SvelteKit 3 and `@sveltejs/adapter-static` 4 on TypeScript 6. SvelteKit
  options now live in `sveltekit.options.js` and are passed to the `sveltekit()` Vite
  plugin; `svelte.config.js` is gone. `goto` options use `replace` and `reset: false`
  in place of `replaceState` and `keepFocus`.
- `projectHref` and `switchProjectHref` return `string`, which is what SvelteKit 3's
  `resolve()` accepts for a path built at run time.
- `tsconfig.json` extends `$app/tsconfig` (SvelteKit 3 no longer writes `.svelte-kit/tsconfig.json`) and
  lists its sources explicitly, so root config scripts are not type-checked.
- Dropped the `cookie` override; SvelteKit 3 depends on the fixed 2.x line.

### Fixed

- The VLM scope panel's "Cluster representatives" text no longer says the rest keep their cluster membership: a labeled representative moves to its class cluster, and the next members are not picked in its place (#192).
- The VLM scope panel's "Uncertain only" copy says what the selector does: crops whose detector confidence is below the limit or was not recorded (OpenProcessor #195).
- No page overflows horizontally at 430px (#184). The project top bar wraps to two rows below 768px (logo, project switcher, Resources and the API chip on the first, the scrolling nav strip on the second; the wordmark and breadcrumb are hidden below 640px). `/review`'s tab strip and queue counter wrap the same way, so the tabs are no longer squeezed. On `/projects` an `sr-only` table header escaped its scroll container and widened the page; the container is now positioned so it clips it. New stubbed e2e `test_narrow_viewport_430.py` asserts no horizontal overflow at 430px on every project route and `/projects`.
- nginx uses `absolute_redirect off`, so redirects no longer point at container port 8080.
- A long project display name no longer widens `/projects` at 430px (the "Open <name>" header link truncates) and the project switcher is capped at `min(16rem, 55vw)`; the 430px e2e now uses a long display name and slug (#274).
- `scripts/capture_docs_screenshots.py` no longer aborts after the first image when `--out` is outside `frontend/` (it printed the path with `relative_to`, which raised `ValueError`).

## [0.1.1] - TBD

Adopts the OpenProcessor 0.4.1 API.

### Changed

- Promote to Triton runs as a background job: the dialog follows the served
  phases (exporting, loading, building, warming), shows failures with their
  status, and re-opens on a running promote after a reload.
- Cluster cards read `label_agreement` for label quality; `purity` stays
  geometry-only.

### Added

- Proposal denylist (case-insensitive glob patterns) in the prompt-pack editor.
- Model delete shows the served reason when the model is the project's ingest
  detector (or the check is unavailable) and offers a confirmed force retry.

### Fixed

- Boxes line up with the image in the box editor and the crop detail modal.

## [0.1.0] - TBD

First public release. Cropwright takes you from raw images to a trained
detector without labeling one box at a time: clusters and a
vision-language model do the bulk labeling, you confirm at keyboard speed,
and export, training, model comparison and promotion happen in the same
app. It is the human-in-the-loop frontend for
[OpenProcessor](https://github.com/davidamacey/OpenProcessor), has no
database of its own and requires a running OpenProcessor backend.

### Added

- **Dashboard** (`/dashboard`): live dataset stats, recent labels, "Run
  Clustering Now" with stage progress, and an optional per-class VLM
  assist scope when the backend advertises a prompt pack.
- **Ingest** (`/ingest`): chunked browser upload of files and folders with
  pause, resume and cancel, duplicate pre-filtering, per-file results with
  stable error codes, an optional server-path batch mode, ingest status by
  source, a region-drain panel and a clustering handoff gated on the
  served "drained" verdict.
- **Clusters** (`/clusters`, `/clusters/[id]`): cluster cards with served
  cohesion and tiers, semantic and OCR-text search, an Ignored bucket with
  restore, an optional embedding-plot lasso, and a triage grid with
  pointer-based drag and drop, bulk operations, VLM accept/run, AHC refine,
  a strategy bar (sort, score chips, diverse overlay) and undo.
- **Review** (`/review`): keyboard-driven queues (All, Uncertainty, Model
  Disagreements, Classifier Blind Spots, New Class Proposals and a region
  tab), preset chips, served per-tab filters, deep links, served
  empty-queue reasons and a fuzzy class picker.
- **Classes** (`/classes`): add, rename, merge, deprecate and restore
  classes, per-class hotkeys with server-served reserved keys, and bulk
  resolution of VLM-proposed new classes.
- **Export** (`/export`): balance and trainability gaps, test-holdout
  freeze, YOLO export with served image/object/split counts and
  downloadable registry artifacts.
- **Train** (`/train`): preflight, launch, live progress and log, finished
  run results (test-split evaluation, lineage, MLflow link), promote to
  Triton with the served gate report, reproduce, probe predictions and a
  training-cohorts preview.
- **Bake-off** (`/bakeoff`): compare trained and baseline models on
  evaluation datasets in a model × dataset matrix with ranked and
  per-class results (present only when the backend mounts it).
- **Models** (`/models`) and **Settings** (`/settings`): inference-service
  status with gated unload (an optional model that isn't installed reads
  "optional · not installed", not a warning), deployment-wide curation
  defaults, and curation-score computation.
- **Domain-agnostic annotation slots**: the region feature is synthesized
  from the backend's served region profile; deployments can add or
  customize slots with a validated tier-2 `annotation-profiles.json`
  (examples under `examples/annotation-profiles/`).
- **Vendored API contract** (`contracts/openprocessor/`) with tests that
  fail on a backend wire rename.
- **Tests**: vitest unit and component tests, a fail-closed stubbed
  Playwright suite (`npm run test:e2e`), an env-gated read-only live tier
  (`npm run test:live`) and mutation testing (`npm run test:mutation`).
- **Docker image** `davidamacey/cropwright`, multi-arch (`linux/amd64`
  and `linux/arm64`): multi-stage build on Node 26, nginx running as
  non-root (uid 101) on port 8080 from an exact-pinned
  `nginx-unprivileged` base, runtime-configurable API prefix and
  upstream.
- **Pull-based install**: `docker-compose.yml` only pulls the published
  image, so running Cropwright needs just that file and a `.env`, no
  clone. `CROPWRIGHT_TAG` pins a version; building from source uses the
  `docker-compose.build.yml` overlay.
- **Release pipeline** (`scripts/release.sh`): staged and resumable
  (preflight, verify, build, scan, smoke, tag, publish, finish), building
  both architectures, scanning each with Trivy and smoke-testing each
  image before a single multi-arch publish with an SBOM.
- **Documentation site** (`docs-site/`, Docusaurus): quick start, a guide
  per page, configuration, operations and developer docs, interactive
  Archify architecture diagrams, a roadmap, and screenshots plus a
  walkthrough GIF captured from public sample data (COCO val2017 and Open
  Images).

### Security

- **The curation API has no authentication.** Anyone who can reach the
  Cropwright origin can label, ingest, export and train. Run it only on a
  trusted network or behind an authenticating reverse proxy; never expose
  it to the public internet. See [SECURITY.md](../SECURITY.md).
- nginx sends X-Frame-Options, X-Content-Type-Options, Referrer-Policy and
  Permissions-Policy on every response and does not reveal its version.
- The container drops all capabilities and sets `no-new-privileges` in
  the provided compose file; CI workflows run with read-only
  permissions.

[Unreleased]: https://github.com/davidamacey/OpenProcessor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/davidamacey/OpenProcessor/releases/tag/v0.1.0
