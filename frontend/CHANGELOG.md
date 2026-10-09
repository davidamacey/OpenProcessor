# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Fixed

- The VLM scope panel's "Cluster representatives" text no longer says the rest keep their cluster membership: a labeled representative moves to its class cluster, and the next members are not picked in its place (#192).
- No page overflows horizontally at 430px (#184). The project top bar wraps to two rows below 768px (logo, project switcher, Resources and the API chip on the first, the scrolling nav strip on the second; the wordmark and breadcrumb are hidden below 640px). `/review`'s tab strip and queue counter wrap the same way, so the tabs are no longer squeezed. On `/projects` an `sr-only` table header escaped its scroll container and widened the page; the container is now positioned so it clips it. New stubbed e2e `test_narrow_viewport_430.py` asserts no horizontal overflow at 430px on every project route and `/projects`.
- nginx uses `absolute_redirect off`, so redirects no longer point at container port 8080.

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
  it to the public internet. See [SECURITY.md](SECURITY.md).
- nginx sends X-Frame-Options, X-Content-Type-Options, Referrer-Policy and
  Permissions-Policy on every response and does not reveal its version.
- The container drops all capabilities and sets `no-new-privileges` in
  the provided compose file; CI workflows run with read-only
  permissions.

[Unreleased]: https://github.com/davidamacey/OpenProcessor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/davidamacey/OpenProcessor/releases/tag/v0.1.0
