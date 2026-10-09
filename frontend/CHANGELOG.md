# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
