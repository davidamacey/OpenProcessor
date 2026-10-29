# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- An axe-core accessibility check in the stubbed e2e suite (`e2e/stubbed/test_a11y_axe.py`) on projects, clusters, cluster detail, review, classes, settings, prompt packs and open-vocab sets; it fails on any violation.

### Fixed

- An unknown URL shows a not-found page instead of the "Something went wrong / Reload" prompt.
- A thumbnail the server cannot serve shows a dimmed empty tile instead of spilling its alt text across the cluster grid.
- On search results the cluster badge no longer runs under the MATCH chip.
- Accessible names for the settings axis selects and the class table's group select, hotkey input and actions header; the bake-off page no longer nests a second `main`; the review page has an `h1`.
- Dim text and the labels on filled buttons and the LIVE badge meet WCAG AA contrast.
- The blocked-start screen no longer repeats its title as the reason.
- A project that is still building opens by itself once it becomes selectable.
- `/projects` no longer probes a combine job id on every load (a logged 404); "Combine projects…" is always offered because the backend always mounts the combine router.
- A crop card in the cluster grid is a container with one select button (the thumbnail) and sibling action buttons, so no interactive control nests inside another; the class list in the sidebar no longer inherits a list-item role on its buttons; every empty table header (scores, prompt packs, region profiles, open-vocab sets, VLM tables, config, deprecated classes) has a visually hidden label.
- The docs screenshot seeding binds a project (`--project`) like the other scripts.

## [0.6.0] - 2026-10-16

### Highlights

Cropwright 0.6.0 adds **Assist…** (send a crop, a box, a selection, a cluster or a sub-cluster to the VLM, review the proposed label and apply it explicitly), a bulk-ingest panel with Preview, a sources panel for S3 buckets and watched folders, detection-only ingest settings, and a tidier, split-by-concern codebase with one accessible dialog component.

### Upgrade notes / behaviour changes

- The container reads `OP_CROPWRIGHT_INGEST_MAX_REQUEST_MB` and `OP_CROPWRIGHT_DATASET_UPLOAD_MAX_MB` (were `CROPWRIGHT_*`); a container started with an old name set exits with the new name in the message. `OP_CROPWRIGHT_IMAGE`, `_BIND_ADDRESS` and `_PORT` are the compose-side names.
- The "Run VLM on cluster" dialog and the cluster page's "Run VLM" are replaced by **Assist…**, which needs a backend with `/assist/runs` (0.6.0).
- `scripts/capture_docs_screenshots.py` requires `--public-projects`.

### Added

- The sub-cluster tabs on the cluster page list the ids the server serves on the cluster card (`subcluster_ids`), so a sub-cluster whose members are not on the loaded page still has its tab. The assist dialog has an **Agreement needed** field (`min_agreement`) for a cluster classify run: the server's default shown, the contract's 0.5 to 1.0 range.

- The audit queue shows, on each drawn crop, the machine label and its source the crop carried when it was drawn (`audit_label_name`, `audit_label_source`, now on the item wire, #166).

- **Assist…** on the cluster page (the cluster or the open sub-cluster, or the selected crops), on one crop's details
  panel, on one box in `/review`, and on Matching items (a filter): a dialog that posts one assist run
  (`POST {prefix}/assist/runs`). It offers what the VLM sees (selected, all members, a seeded representative sample with
  size, seed and spread), the actions (classify, attributes, verify, segment, with the segmenter's verify choice), the
  VLM endpoint (with the external-endpoint acknowledgement) and the prompt pack. Each change re-asks the server for a dry
  run and shows its counts (members sent, estimated VLM calls, strata, what was left out, the endpoint and pack pinned) or
  its refusal verbatim; Start run stays disabled on a refusal. The run is followed to its end, including `waiting_region`
  for a segment run, with the served result counts; Cancel run stops it.

- The newest label proposal for the viewed cluster or sub-cluster is a folded line above the grid and opens a proposal
  card: the proposed class, the agreement and gate, the votes, new-class names (with Add class…), attribute suggestions
  marked "not labels", a stale banner from the served `scope_now`, and **Apply {class} to {N} members**, the explicit
  button, with the server's dry-run count first. The backend's `scope_changed` and `class_not_in_registry` refusals are
  shown as served, and the override is offered only after the refusal.

- Attribute chips (`assist.attributes`, per box `assist.boxes`) on the crop card, the details panel and `/review`, dashed
  and labelled "VLM suggests"; "VLM said: {class}" when the VLM's answer differs from the item's class.

- The VLM scope panel on Settings has a **Region verification** choice (`region_verify`: verify with the VLM, or accept
  without a VLM).

- Wire types `types_assist.ts`, `api_assist.ts` and the contract test `assistContract.test.ts`; the item wire's `assist`
  object; `cluster_subid` on the item filter and the reprocess filter; `assist` and `attributes` in the pack test calls.

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

- The sub-cluster tab asks the server for the sub-cluster (`GET /crops?cluster_subid=`) instead of filtering the loaded
  page, so a sub-cluster larger than a page is shown whole. The tab list is the sub-clusters seen while the whole cluster
  is shown.

- The dashboard's quick action links to the clusters (where Assist runs) instead of opening a "Run VLM on cluster"
  dialog.

- The docs screenshot capture (`scripts/capture_docs_screenshots.py`) now requires `--public-projects` and
  refuses any project outside that list, and refuses pages that list every project while the backend holds
  another project, so a capture can never show private data.

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

### Removed

- Standalone-repository leftovers: the release scripts (`scripts/release.sh`, `scripts/release/`,
  `scripts/release-smoke.sh`), the frontend's own compose files, pre-commit config, diagram and GIF scripts,
  and its own `SECURITY.md` and `CONTRIBUTING.md`. The Cropwright image ships through the repository's
  `scripts/release/build_and_publish.sh`; the README and docs point at the root files.

- `runVlmOnCluster`, `pollAutoLabelJob` and `getAutoLabelJobStatus`, the dashboard's "Run VLM on cluster" modal and the
  cluster page's "Run VLM" button: the backend deleted its per-cluster label route; an assist run replaces it.

### Fixed

- At narrow widths the top navigation wraps onto a second row, so Export, Models, Train, Bake-off and Settings are reachable without scrolling a clipped strip.

- The dashboard "Run VLM Labeling" action becomes a link to the Models settings ("activate a VLM first") when the backend serves the project's VLM source as `off`, instead of a ready primary button.

- Model titles on the Models page wrap instead of truncating.

- Ingest policy cost preview marks its figures as belonging to the previous policy while a new preview is computed (a large project takes seconds), instead of showing stale numbers as current.

- A project with no classes says "No classes yet" on the Classes page and "No classes in this project yet" on Export instead of "No classes match this filter" and "ingest some data first".

- Removed the internal "Plan §B4" reference from the test-holdout texts.

- At 430 px the Settings page selects (long option text) and the Review sort button no longer widen the page, and served filter fields wrap instead of overflowing when "More filters" is open.

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
