# Cropwright — CLAUDE.md

SvelteKit + TypeScript image-crop annotation web app (product name
**Cropwright**, package name `cropwright`; repo directory is still
`legacy-labeler` pending a physical rename). Generalized via a
capability-model / annotation-slot mechanism (see
`docs/genericization-plan-2026-09-13.md`) so it is no longer
hardcoded to vehicles or license plates. The build ships with no domain
built in: the region slot is synthesized from the backend's served
region profile (`GET {API_PREFIX}/health` `region_profile`, see
"Served region profile" below), and example domain profiles (license
plate, aircraft tail number, defect code) live under `examples/` as
tier-2 JSON, never bundled. A new domain is a backend region profile
plus, optionally, a tier-2 profile, not an app-code edit. Sister project to `legacy_sorter` (v2 Tauri app for
the actual sort UX) and `openprocessor` (server-side inference + OpenSearch

- clustering).

## Purpose

Manage a high-volume labeling workflow over hundreds of thousands of image
crops, with cluster-based assisted labeling, VLM vision suggestions, and a
keyboard-first UX matching the legacy_sorter manual-mode speed budget.

## Architecture

- **Frontend**: SvelteKit 2 + TypeScript + Tailwind CSS + svelte-dnd-action
  (pointer-event drag — HTML5 DnD is broken in Tauri WebView and unreliable in
  some browsers)
- **Backend**: OpenProcessor at `http://localhost:4603/curation/...` — labeler is a
  pure consumer; no own database
- **State**: Svelte 5 runes (`$state`, `$derived`, `$effect`); nothing in
  localStorage that can't be reconstructed by an API call
- **Build**: SvelteKit static adapter → nginx in production Docker container

## Routes

**Every page lives under a project: `/p/[project]/<section>`** (owner
decision: the active project lives in the URL path only, nothing in
localStorage — see "Projects" below). The table lists the sections; the
real URL of `/review` is `/p/<slug>/review`. All of them are shipped and
linked from the top nav (`src/routes/p/[project]/+layout.svelte`, the app
shell) except `/clusters/[id]`, reached via a cluster card. `/` is not a
route in its own right — it redirects to `/p/<served default_slug>/dashboard`
(`src/routes/+page.ts`), and the bare pre-projects paths (`/review?tab=x`,
`/clusters/12`, ...) redirect to the same path under the served default
project with the query string kept (`src/routes/[...legacy]/+page.ts`).
`/p/<slug>` alone lands on `dashboard`. The top bar carries the **project
switcher** (`ProjectSwitcher.svelte`). `/projects` is the one global page
(not under `/p/`). Every in-app link and `goto()` is built by
`projectHref()` (`$lib/projectPaths`) wrapped in `resolve()`, never
hand-assembled. The MVP/post-MVP split from the original design doc is
gone — every route in this table exists and works; nothing here is a
stub.

| Route            | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/dashboard`     | Current pipeline dashboard — live `DatasetStats` (polls every 10s) + `AutoLabelPanel` ("Run Clustering Now" with stage progress), shared with the daemon-fired auto-label run. `AutoLabelPanel` also hosts an optional per-class assist scope (`AssistScopeBar`, absent unless `/methods` advertises a usable `prompt_pack` — see "Curation-strategy selector bar" below) that lets an operator point the VLM-assisted sweep at a single class instead of the whole pool. Below the stats, `DetectionsSummaryPanel` shows the served `GET {API_PREFIX}/detections/summary` (totals, embedding breakdown, per-label table) with an "Embed N detections" action when the summary serves a `suggested_reprocess`; `AutoLabelPanel` has an "Embed missing vectors first" option (`embed_missing`). See "Detector, ingest policy and embedding state".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `/ingest`        | Bring images into the pool. Offers browser upload (files, folders and drag-drop) to `POST {API_PREFIX}/ingest/upload`, chunked to the served per-request cap with bounded concurrency. It shows a per-file result (ingested / duplicate / failed + served reason), supports pause/resume/cancel, and pre-filters already-indexed identifiers via `POST {API_PREFIX}/ingest/path_lookup`. An optional server-path mode uses `POST {API_PREFIX}/ingest/batch` and is shown only when `GET {API_PREFIX}/ingest/config` serves `batch.enabled: true` and at least one `batch.source_roots` entry; `upload.enabled: false` replaces the browser-upload panel with one line (see "Ingest" below). The page also has an ingest status table by source (`GET {API_PREFIX}/ingest/status`), a region-drain panel (`GET {API_PREFIX}/ingest/region_drain`, only with a served region profile), and a clustering handoff that reuses `AutoLabelPanel`. The page renders from the served `GET {API_PREFIX}/ingest/config`: a loading line until it loads, its error if the read fails. Also shows the served detector and ingest-policy summary (`IngestDetectorCard`), the served per-file and total embedding counts of a run, and links to `/settings/ingest-policy`. See "Detector, ingest policy and embedding state".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `/datasets`      | Labeled-dataset import (OpenProcessor W10): the imports list, the `/datasets/import` wizard (preview, name-based class mapping, options, confirm-gated start) and the `/datasets/imports/[id]` job view (served progress, cancel / resume / dry-run-first undo). Reached from `/ingest`; absent when the backend doesn't serve `GET {API_PREFIX}/datasets/formats`. See "Dataset import and Reprocess" below.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `/clusters`      | Cluster grid view, sidebar filter, **strategy bar** (review-sort dropdown + score chips, see below — no cluster-method picker here; that lives on `/settings`). When the class filter is a slot-bound class (the served region profile's `region_class_name`), replaces the cluster grid with that slot's **region gallery** (`SlotGallery`, driven by `createSlotGalleryController(slot)` in `src/routes/p/[project]/clusters/slotGalleryController.svelte.ts`, one controller per slot bound through `slotForClassName`, browsing the slot's `queue.browsePath`, i.e. `{API_PREFIX}/regions`; detector / verified / status / score / text filters, all copy templated over `slot.label`). The unfiltered grid pins one synthetic inventory card per registered slot with a browse endpoint. Also hosts the **embedding-plot** overlay toggle when `viz_projection` is available (see below). An **Ignored** toggle (2026-09-24, logic-moves W7) swaps the grid for the excluded/`cluster_id=-2` bucket with a "Restore selected" action, and an **item-text search** box (`{API_PREFIX}/crops?item_text=`) swaps it for a literal OCR-text search over `item_text_lines` — both mode-swaps mirror the existing dataset-wide semantic search's pattern, and neither is the same endpoint as the semantic (embedding) search box. The shared item-filter bar scopes the cluster grid, and a **Matching items** mode (`?mode=matching`) lists every item the filter matches with Ignore / Restore / Label / Move on all of them (served dry run first); see "Shared item filter and run on selection".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `/clusters/[id]` | Single cluster crop grid + DnD + bulk ops + strategy bar (sort / diverse overlay / score chips scoped to this cluster)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `/review`        | 6 top-level review tabs (2026-09 consolidation, down from 9, plus `new_class_proposals` added 2026-09-24 — see below): **All** / **Uncertainty** / **Model Disagreements** / **Classifier Blind Spots** / **New Class Proposals** / one tab per registered queue-capable slot (in practice the one region tab, present only when the backend serves a region profile and labelled by its `display_name`, `?tab=regions`), each with its own default sort (`review_sorts.py`'s `_TAB_DEFAULTS`) plus the strategy bar's selectable sort/score overlays — the bar's summary chip also shows the server's `sort_applied` next to whatever was requested. The All tab additionally offers a row of **quick-filter preset chips** (VLM mismatches / VLM low-conf / Primary · low-conf) that layer the former Mismatches / VLM Low-Conf / Primary · Low-Conf tabs' exact cohort queries on top of the All view. Class/Source/Conf filter controls (`class_name`/`source`/`conf_min`/`conf_max`) are live against `GET {API_PREFIX}/review/{tab}`. `/review?crop_id=` deep links resolve via `GET {API_PREFIX}/review/{tab}/locate` — jumps straight to the crop's served page/rank, or shows the backend's `reason` when it isn't in the queue. A slot tab carries provenance chips + the region text reading, driven by the active slot's capabilities rather than a hardcoded tab check (see "Slot-generic review tabs" below). An **Imported** tab (W10) appears only when `GET {API_PREFIX}/review/tabs` serves it, and a link can seed `import_id` / `combine_conflict` filters (shown as removable chips; see "Dataset import and Reprocess" below). The shared item-filter bar (class by name, not-class, area band, origin, embedding and review state) sits in the filter row, gated per tab by the served `filters`, and every other served `filter_specs` entry renders by its kind; see "Shared item filter and run on selection".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `/classes`       | Add / rename / merge / **deprecate** / **restore** classes, per-class hotkey binding, and a **Proposals** section (`GET {API_PREFIX}/review/new_class_proposals/summary`) for creating a class from — or mapping onto an existing class — a VLM-proposed term the registry doesn't have yet, bulk-resolving _every_ pending crop proposing that term (`POST {API_PREFIX}/review/new_class_proposals/resolve`), not just the summary's sample thumbnails. Every active row has a **Deprecate** button (`POST {API_PREFIX}/classes/{id}/deprecate`, confirm-gated) — 409 with a structured `class_still_referenced` detail (`{message, item_count, confirmed_label_count}`) offers the existing merge dialog instead, preselecting the class as the merge source. The deprecated-classes table's **Restore** button (OpenProcessor 01324cb, 243f7f2) is real, not the permanently-disabled placeholder it used to be — `POST {API_PREFIX}/classes/{id}/restore`, whose 409 is either a PLAIN STRING detail (a live class already claims the name) shown verbatim, or, for a class merged into another (OpenProcessor 4c125ec, F-56), a structured `class_merged` detail rendered as "Merged into `<served class_name>`; un-merge isn't supported." plus the served message and hint (`classMergedDetail`/`classMergedRestoreText`, `api.ts`). Since OpenProcessor 51b05d7, `GET {API_PREFIX}/classes` serves `merged_into`, so a merged deprecated class shows "merged into `<name>`" instead of a Restore button (the 409 path stays for a stale page). The merge dry-run's `validations_carried_over` reads "N human validations will carry over" (a merge keeps validations; the old `would_unvalidate` key is gone, no shim). The class table renders first; the Proposals list sits below it (and below the deprecated table) in a collapsed `<details>` (F-53), its per-term Hide button is session-only (not persisted, F-58), and the page scrolls as a whole rather than in an inner pane (F-50). A collapsed "Create classes from the detector" panel (`SeedFromDetectorPanel`: dry run, then a confirm) appears when `GET {API_PREFIX}/ingest/config` reports a detector.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `/export`        | Trigger YOLO export, view balance gap, freeze test holdout, download the frozen export's `class_registry.json`/`data.yaml`/`manifest.json` (via `{API_PREFIX}/export/registry/{artifact}`). Since OpenProcessor df01309, `GET {API_PREFIX}/export/status` also serves `image_count`/`class_count`/`group_key`/`split_counts`/`class_split_counts` — the page shows the served train/val/test totals and a collapsible per-class table (any class at 0 train or 0 val highlighted, served numbers only). The freeze modal has no Seed field: `POST {API_PREFIX}/test_holdout/freeze`'s body is `{percent}` only (selection is deterministic, SHA1 of each crop id per class — an extra `seed` key is a 422); the success toast shows the served `selection`/`min_per_class`. Since OpenProcessor 4c9499a (one image + one label file per source image), `ExportStatus` also carries `object_count`/`split_object_counts` (label lines, distinct from `image_count`/`split_counts`) — the page reads "N objects in M images" and separate images:/objects: split badges, never one ambiguous number. An opt-in "Only images whose every object is labeled" checkbox sends `require_fully_labeled_images` on `POST {API_PREFIX}/export/yolo`; the served partial-frame counts (`unlabeled_items_on_exported_images`/`images_with_unlabeled_items`/`images_dropped_not_fully_labeled`) render when present. Every one of these fields is `null` (not `0`) on an export written before it was recorded — rendered via `formatCount()` (`src/lib/formatCount.ts`) as "—". The `ExportStatus` fields are nullable because the served schema types them so; `TestHoldoutFreezeResult`'s `selection`/`percent`/`min_per_class` are required. A collapsed "Only items matching a filter" limits the export (`item_filter`) and shows the served matching count; see "Shared item filter and run on selection".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `/models`        | Triton model registry browser. Lists the project's own models, the base models and, via `GET {API_PREFIX}/models/status?include_other_projects=true`, other projects' models their owners shared, each with the served `project`/`shared`/`class_mapping` — see "Model sharing" below. The VLM is no longer a single external service: since OpenProcessor W9 each registered endpoint is its own `kind: 'vlm'` row (name = endpoint name, served resolved `model`, served `active` chip, status named through the registry's `labels.status`, no Unload) with a link to **Settings → Models** while the registry is served — see "VLM models" below.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `/train`         | Training cockpit — preflight, launch, live progress, log tail, past runs, **Promote**, **Reproduce**, **Training cohorts picker** (class-agnostic `CORE_COHORTS` for every class + the region profile's 5 server-side modes when one is configured — see "Training cohorts" below). The dataset card shows the _current export's own_ `image_count`/`class_count`/`split_counts`/per-class `class_split_counts` (from `GET {API_PREFIX}/export/status`, OpenProcessor df01309) rather than the dataset-wide validated total — the old global number (which double-counted `test_holdout` crops) survives only as a clearly-labelled "(global pool)" line for an explicitly-picked past export version, which `/export/status` does not describe. `AugmentationPanel`'s preset picker is served from `GET {API_PREFIX}/train/augmentation_presets` (id/label/description/orientation-sensitive), defaulting to the served `default`; a failed load shows its error (there is no hardcoded preset list or default id). `/train/start`/`/start_campaign`'s 422 on an unknown `augmentation.preset` (`{detail: {message, field, valid_presets}}`) surfaces `valid_presets` in the toast alongside the message. Since OpenProcessor 4c9499a, the card also shows the export's `object_count`/`split_object_counts` (label lines) alongside `image_count`/`split_counts` — "N objects in M images" plus separate images:/objects: split badges — and its per-class table is objects, not an ambiguous count; a `null` field (an export written before 4c9499a recorded it) renders via `formatCount()` as "—", never 0.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `/bakeoff`       | Model comparison on OpenProcessor #34's v2 wire (F6, `docs/design/bakeoff-v2-ui-plan-2026-09-25.md`). Pick eval datasets (`GET {API_PREFIX}/bakeoff/eval_datasets`: export test splits first with the current one flagged and preselected, external frozen sets grouped by served `group`), models (finished training runs from `/bakeoff/trained_models`, each with the served per-selected-dataset `for_dataset` facts and a train/test overlap warning when the served overlap is non-null and > 0; the profile's `/bakeoff/baseline_models`; an optional custom ref) and a profile (whatever `/bakeoff/profiles` serves, `default_profile` preselected, `default_error` shown). A confirm dialog precedes `POST /bakeoff/run` (typed `run`/`baseline`/`custom` refs; 400/409/422 detail shown verbatim); the job polls `/bakeoff/status/{id}` with progress, per-stage failures and the enqueue-time class mapping. Results: the `/bakeoff/matrix/{id}` model × dataset matrix bolds every served tied winner (`best` is a list), and `/bakeoff/results/{id}?dataset_id=` renders ranked rows plus a per-class table where an uncovered class reads "not covered" and each model's unmapped classes are listed; a 409 (pre-v2 result) shows a note. Previous runs (`/bakeoff/runs`) stay selectable. State in `src/lib/bakeoff/bakeoffController.svelte.ts`, types in `src/lib/types_bakeoff.ts` (pinned to the vendored OpenAPI by `contract/bakeoffContract.test.ts`); nothing computes a metric, mapping, rank or winner client-side. The nav link and page always render (the backend mounts the `{API_PREFIX}/bakeoff/*` router unconditionally); the page runs its discovery reads on mount.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `/settings`      | Deployment-defaults admin page for the shared curation-strategy defaults (`GET,PUT {API_PREFIX}/settings`) — one place to pin the deployment's clustering method, review-queue sort and VLM prompt pack (the latter honored by the always-on VLM labeler and by auto-label runs that don't pick their own), plus, for any axis the server marks not `settable`, a read-only "Not settable on this backend" section (none today: since OpenProcessor f14f4ddc `detection_profile`, `prompt_pack` and `vlm` are activation-backed and settable, and `GET /settings` `defaults` for them come only from the activation records, so a served `off` renders as a disabled option, `unofferedServedValue`). Which axes get a control is decided solely by the server's per-entry `settable` flag on `/methods` (`settableAxes` in `src/lib/curationSettings.ts`). Deployment-wide — see `docs/design/curation-settings-ui-plan-2026-09-21.md` — so it is its own route rather than a `StrategyBar` chip, with an explicit confirm dialog before every save. Also hosts the **Curation scores card** (`ScoresCard.svelte`, G10, 2026-09-24) — per-scorer coverage from `GET {API_PREFIX}/scores/coverage`, confirm-gated "Compute all"/"Compute selected" (`POST {API_PREFIX}/scores/compute {scorers}`, ids always sourced from the served coverage keys), a progress poll of `GET {API_PREFIX}/scores/status` following `EmbeddingPlot`'s rebuild-job pattern, and "Cancel" (`POST {API_PREFIX}/scores/cancel`). A failed coverage read shows its error with a retry; a failed compute (e.g. mistakenness lacking probe predictions) shows the backend's error verbatim. A completed compute reloads coverage and resets `strategiesStore` so `StrategyBar`'s sort/score options pick up the new coverage without a full page reload — see "Curation-strategy selector bar" below. Also hosts the **Keyboard shortcuts** card (`KeymapCard.svelte`, K2, 2026-09-26) — absent, not disabled, until OpenProcessor W2b's `GET/PUT {API_PREFIX}/keymap` route exists; see "Keyboard shortcuts" below. Also links to the **Prompt packs** editor (`/settings/prompt-packs`, see "Prompt-pack editor" below), absent until the backend serves W3, and the **Region profiles** editor (`/settings/region-profiles`, see "Region-profile editor" below), absent until the backend serves W4. Also links to the **VLM models** page (`/settings/models`, see "VLM models" below) and hosts a `vlm` axis dropdown (served endpoints plus `off`, an unacknowledged external entry disabled), both absent until the backend serves W9. Also links to the **Open-vocabulary sets** editor (`/settings/open-vocab`, see "Open-vocabulary sets" below), absent until the backend serves `GET {API_PREFIX}/open_vocab`. Also links to the **Ingest policy** page (`/settings/ingest-policy`, see "Detector, ingest policy and embedding state"). |
| `/projects`      | Global project management (not under `/p/`; `src/routes/projects/+page.svelte`, state in `$lib/projects/projectsAdminController.svelte.ts`). The served list (`GET {API_PREFIX}/projects`, with a Show-archived toggle sending `include_archived=true`), the served shard `capacity` block, and the P3 lifecycle actions: create (`POST /projects`), edit (`PATCH /projects/{slug}` with `expected_revision`; a 409 `revision_conflict` offers a reload that keeps the typed edit), archive / unarchive, copy settings (`POST /projects/{slug}/clone_settings`, axes from the served `limits.cloneable_axes`), and a guarded delete (the served dry run's report and `blocking` reasons first; no confirm field while anything blocks; `confirm` = the typed slug). Every action is gated on served flags only (Open/Edit: `selectable`; Copy settings: `writable`; Archive: the served `archivable`; Unarchive: the served `unarchivable`; Delete: `deletable`; Create disabled only by a served `blocked` capacity). Refusals render the served `detail.message` verbatim; the lifecycle envelope's `warnings` become toasts. Each selectable row also reads its served pipeline-pause flag (`GET {prefix}/pause` through that row's own served `prefix`) and shows a "paused" chip; writable rows get a confirm-gated Pause / Resume pipeline (`POST {prefix}/pause` / `/resume`) once that flag has loaded.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

## Ingest (`/ingest`, 2026-09-24; BA-1..BA-7 adopted 2026-09-25)

Bring images into the pool — the frontend side of
`docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md`. All 13 pieces
of that plan are implemented as of OpenProcessor #36 (backend commit
c5c606f) — see the plan's status section for the full piece-by-piece
record.

- **Served config.** The ingest router is always mounted, so the nav
  link always renders. `/routes/p/[project]/ingest/+page.svelte` fetches
  `GET {API_PREFIX}/ingest/config` (`getIngestConfig()`, `api.ts`) once
  on mount and renders only a loading line until it loads (or its error
  if the read fails) — the upload/status/drain sections, whose child
  components each fire their own GET, mount only after.
  `ingestConfig.ts`'s `resolveIngestConfig(served)` resolves every
  upload/batch/region-drain limit from it; there are no interim client
  constants. `uploadMaxBytes` is the tighter of the nginx proxy's
  `client_max_body_size` and the served `upload.max_bytes_per_request` —
  a chunk sized against only one of the two could still 413 against the
  other.
- **Served on/off switches.** `upload.enabled: false` replaces the
  browser-upload panel (and its caveat) with one line, "Browser uploads
  are disabled on this deployment." `batch.enabled: false` hides the
  server-path panel, the same as an empty `batch.source_roots`
  (`serverPathIngestAvailable`, `ingestConfig.ts`).
- **Upload caveat.** The amber "uploads aren't kept" banner renders only
  when the served `upload.persists_bytes` is `false`.
  `POST {API_PREFIX}/ingest/upload` persists the uploaded bytes
  server-side, content-addressed, so a stock deployment serves
  `persists_bytes: true` and shows no banner at all.
- **Run controller.** `src/lib/ingest/ingestRunController.svelte.ts`
  (`createIngestRun`, same factory convention as
  `clusterController`/`reviewController`) owns the whole
  `prefiltering → uploading ⇄ paused → (done | cancelled | error)`
  state machine: prefilters already-indexed identifiers via
  `POST {API_PREFIX}/ingest/path_lookup`, chunks the selection
  (`src/lib/ingest/uploadPlanner.ts`, respecting an image-count and a
  byte cap), dispatches with bounded concurrency (default 2), and maps
  each chunk's response — 200 (BA-1: keyed by the returned
  `source_identifier`, not `image_path`, since `image_path` is now the
  server-persisted content-addressed path), a
  backend-style 413 (JSON body: halve the chunk and retry once, tagged
  `error_kind: 'too_large'`), an nginx-style 413 (HTML body: stop the
  run, never retry), 422 (fail with the served detail), 503 (auto-pause
  with the served detail). Per-file results live in
  `src/lib/ingest/ingestResults.svelte.ts`, backed by `SvelteMap`
  (`svelte/reactivity`) — plain `$state(new Map())` only makes the
  _binding_ reassignment reactive, not `.set()`/`.delete()` on the same
  instance, which silently broke the Failed/Duplicate/Ingested tab
  counts until a mount test caught it.
- **Stable error codes (BA-7, landed).** Every failed result (upload,
  batch, single-image) now carries `error_kind` alongside its prose
  `error` — `IngestErrorKind` in `types.ts` documents the known codes
  (`empty`/`unservable_path`/`unsupported_type`/`too_large`/
  `decode_failed`/`detector_infer`/`bulk_index`), kept as a documented
  `string` rather than a closed union since a future failure mode may
  still surface a message without extending this list. The upload run
  panel's Failed tab and the server-path batch panel below both render
  filterable error_kind chips
  (`ingestResults.svelte.ts`'s `errorKindCounts()`/`countOfErrorKind()`/
  `page(kind, offset, limit, errorKind)`) over the served detail — no
  client-side prose parsing.
- **Identifiers.** `${identifierPrefix}${relPath}` (default prefix
  `${sourceTag}/`), built from the collectors' normalized
  `relPath` (`src/lib/ingest/fileSource.ts`: backslashes to `/`, a leading
  `./`/`/` stripped, and any path with a `..` segment skipped). `path_lookup` matches this identifier
  exactly (against either `image_path` or, since BA-1, `source_identifier`
  — server-side), which is why the prefix matters: two different folders
  that both contain e.g. `img001.jpg` at their root would otherwise
  collide.
- **Server-path ingest (piece 11, landed).** `IngestBatchPanel.svelte`
  renders on `/ingest` only when the served `batch.enabled` is true and
  `batch.source_roots` is non-empty (`serverPathIngestAvailable`) — lists the roots read-only and
  submits real batches via the pre-existing `POST
{API_PREFIX}/ingest/batch` (that endpoint predates #36; BA-2 is what
  serves `source_roots`/`max_items` for the gate and client-side cap
  check, and the backend's W10 dropped the label-import fields
  (`label_txt_path`/`label_source`/`detect_mismatches` and the response's
  `labels_imported`/`mismatches`/`missed_labels`/`unmatched_detections`);
  the panel sends and shows image paths and a source tag only). One
  synchronous call per submit, not chunked like the upload run
  controller — a server-path batch has no browser-side byte cost, and
  the backend already batches its own detector inference internally.
  Its own failures render through the same error_kind chip pattern as
  the upload run panel.
- **Clustering handoff.** `ClusteringHandoff.svelte` computes a gate
  from the upload run's own state plus the latest served region drain
  and passes it to `AutoLabelPanel`'s new, additive, optional `gate`
  prop (`{blocked, reason} | null`) — every other `AutoLabelPanel`
  caller (the dashboard) passes nothing and is unaffected. **BA-3
  (landed):** the gate now reads the served `drained` verdict
  (`GET {API_PREFIX}/ingest/region_drain`'s `drained`/`stable_for_s`/
  `observed_at` — true once `total_unfinished` has read 0 for
  `region_drain.stable_polls` consecutive polls) instead of a raw
  `total_unfinished === 0` reading. Still **no client-side stability
  window** — the server now computes the one every client used to have
  to invent independently, and the operator reads the served "Worklog
  drained as of HH:MM:SS" note. `RegionDrainPanel.svelte` shows the
  verdict and how long it's held (`stable for Ns`).
- **Deployment-owned upload cap.** `CROPWRIGHT_INGEST_MAX_REQUEST_MB`
  (default 256, `docker-entrypoint.sh`) is substituted into both
  `nginx.conf`'s `client_max_body_size` and
  `window.__CROPWRIGHT_INGEST_MAX_REQUEST_MB__` (`src/app.html`), so
  the client-side chunk planner and the actual proxy limit can never
  drift apart. `nginx.conf` also gained a dedicated
  `^__API_PREFIX__/projects/[^/]+/ingest/` location (declared before the general API
  location, since nginx matches regex locations in declaration order)
  with a 600s `proxy_read_timeout` — a 128-image batch with detector +
  embedding inference can exceed the general API location's 120s.
- **Security.** The curation API still has **no request authentication
  at all** (an explicit owner decision — see `docs/design/
ingest-ui-and-acceptance-plan-2026-09-24.md` §A.6). Anyone who can reach
  the nginx origin can already ingest (including, now, via the
  server-path batch panel above), label, export and train. Expose
  Cropwright (and the OpenProcessor API it proxies) only on a trusted
  network — never the public internet — until the backend adds opt-in
  auth (BA-5's auth half is still open).
- **Plan deviations** (recorded in the plan doc's own status section
  too):
  - the plan's nginx-413 detection ("`ApiError.detail == null`")
    doesn't match `api.ts`'s actual `errorDetail()` — a non-JSON body
    always becomes a non-null string `detail`, never `null`. The
    controller instead keys off `e.body`'s _type_ (`object` for a
    backend 413, `string` for nginx's HTML page), which is what
    `apiFetch` actually produces.
  - the plan's upload-override env var (`CROPWRIGHT_INGEST_UPLOAD`,
    later `PUBLIC_CROPWRIGHT_INGEST_UPLOAD`) is gone: the banner follows
    the served `persists_bytes` alone (#85).
  - the plan sketches the ingest-only 600s `proxy_read_timeout` as a
    _nested_ `location` inside the general API location; nginx doesn't
    reliably support nesting one regex location inside another, so
    `nginx.conf` uses a sibling location matched first instead.
  - the plan's `batch.max_items_per_request` field name doesn't match
    what the backend actually serves (`batch.max_items`) —
    `IngestConfig`/`ResolvedIngestConfig` use the served name.
  - BA-4 (an upload run's own `run_id`/`GET /ingest/status?run_id=`
    scoping) landed server-side but has no frontend surface yet — no UI
    asked for "this run's own counts after a reload" this pass;
    `ingestUpload()` doesn't send the optional `run_id` form field.

## Detector, ingest policy and embedding state (OpenProcessor v0.4.0, 2026-10-03)

The generic detector and selective embedding: what an ingest keeps, which
detections get a vector, and where the operator sees and repairs the
difference (plan: `docs/design/v040-backend-deltas-ui-plan-2026-10-03.md` §7;
contract e817e4f4, see `docs/design/v040-contract-completeness-addendum-2026-10-03.md`).
Nothing here computes a cost, a count or which items need a vector: every
number, list and request is served, and a refusal shows the served
`detail.message` (and each served `reasons[]` / `unknown_names[]`) through
`detectorErrorLines` (`src/lib/api_detector.ts`).

- **Wrappers and types.** `src/lib/api_detector.ts` (`seedFromDetector`,
  `getIngestPolicy`, `putIngestPolicy`, `previewIngestPolicy`,
  `getDetectionsSummary`; imported directly, never re-exported from
  `api.ts`) and `src/lib/types_detector.ts`, pinned key for key by
  `contract/detectorContract.test.ts`. `IngestConfig` carries `detector`
  (nullable or absent: "No detector reported.") and the `policy` echo.
- **Detector card** (`IngestDetectorCard`, on `/ingest`): the served model,
  version, input size, `assigns_class`, `confidence_floor_applies`, a
  collapsed label table (id, name, slug) and a one-line summary of the served
  policy with a link to the policy page.
- **Ingest policy** (`/settings/ingest-policy`, `IngestPolicyEditor` in
  `src/lib/detector/ingestPolicyController.svelte.ts`, form in
  `components/detector/IngestPolicyForm.svelte`): the draft is the served
  policy minus `revision`. Embedding mode (`all` / `selected` / `lazy`) and
  its criteria are always open; the detect filter (classes, excluded classes,
  confidence, area, per-image cap, `class_resolution`) and the per-project
  detector override are collapsed "advanced" sections. Nothing is validated
  on the client (a `selected` mode with no criterion is the server's 422).
  400 ms after any edit the draft goes to `POST /ingest/policy/preview` (the
  previous request aborted); the panel reads "N of M stored detections would
  be embedded, about X MB" and says outright that it describes detections
  already stored and that the policy changes only future ingests;
  `truncated` shows "estimated from S detections". Save is a confirm, then
  `PUT` with `expected_revision` = the revision last read; a 409
  `revision_conflict` offers Reload (drops edits) or Keep my edits (adopts the
  fresh revision); a served `unknown_names` list shows as a warning ("saved
  anyway"); 422 `detector_not_servable` and 503 `detector_unavailable` show
  their served message and reasons. `IngestPolicyCard` on `/settings` links
  to it and shows the served mode.
- **Create classes from the detector** (`SeedFromDetectorPanel`, collapsed on
  `/classes`, absent when no detector is reported): choose labels (none = all;
  `names` is sent only when picked), Preview is a dry run, the served
  `created` / `skipped` / `conflicts` are listed with their reasons, and
  "Create N classes" runs the real call only behind a confirm, then reloads
  the class registry (`SeedFromDetector`, `seedController.svelte.ts`).
- **Embedding state.** `EmbeddingStateBadge` (the `CropCard` chip uses the
  compact wording, the tooltip carries the full text; `CropMetaPanel`'s
  "Embedding" row via `EmbeddingRows`) names `failed` ("No vector: encoder
  failed", warning tone), `deferred` and `not_selected`; `embedded` and `null`
  (written before the field) render no badge. The copy is frontend wording
  keyed by the contract enum (`embeddingCopy.ts`); there are no served
  labels. `DatasetStats` has an "Embedding" card (served embedded / not
  embedded and the `by_state` chips, legacy `unknown` verbatim), absent when
  the served stats carry no `embedding`.
- **Detections summary** (`DetectionsSummaryPanel` on `/dashboard`, below
  `DatasetStats`): the served totals, embedding breakdown and per-label table
  (`labels_truncated` noted). "Embed N detections" appears only when the
  summary serves a `suggested_reprocess` and opens `ReprocessControl` with
  that request exactly as served (dry run first, apply behind a confirm).
- **"Embed them" everywhere.** `UnembeddedBanner` ("N items in scope have no
  vector and are not ranked" / "cannot match a text search") shows under
  semantic search results (`unembedded_in_scope`) and on `/clusters/[id]`
  ordered views (`n_unembedded`; `getCluster` and `getCrops` hand back the
  served `suggested_reprocess`, the banner's action). `EmptyQueueEmbed`
  (`emptyQueueEmbed.ts`) is an empty review queue's action: the served
  `empty_state.suggested_reprocess`, only while `has_unembedded_items` is true
  and the queue's served reason mentions embedding or vectors.
- **Reprocess embed options.** On a batch of chosen crops, ticking the
  `embed` scope shows "Only items without a vector" and "Parts" (the contract
  enum `crop` / `frame` / `region`, `EMBED_PARTS`); `embed` is sent only once
  one was touched, and a served request keeps its own `embed` untouched. The
  one-crop and one-image bodies have no `embed`, so no options there.
  `ReprocessCounts` shows each scope's served `detail` as key/value rows
  (key through `humanizeId`, value verbatim, booleans yes/no). A served
  request's dialog is titled "Reprocess", never a count of zero.
- **Auto-label.** `AutoLabelPanel` has an "Embed missing vectors first"
  checkbox (`embed_missing=true`, sent only when ticked); the served
  `embed_missing` stage reads "embedding items without a vector" and is
  counted first when the job's echoed args asked for it; a `failed` job keeps
  its served error and the stage table, with an errored stage in the error
  tone.
- **Ingest results.** Every image result and the batch summary carry
  `n_embedded`, `n_not_embedded`, `n_embed_failed` and `n_filtered`; the run
  and batch panels show the summed served totals and per-file counts, and a
  "not embedded" chip lists the files with `n_not_embedded > 0`.
- **Tests:** `api_detector.test.ts`, `contract/detectorContract.test.ts`,
  `detector/*.test.ts`, `components/detector/*.test.ts`,
  `components/embedding/*.test.ts`, `components/ingest/IngestDetectorCard.test.ts`,
  `components/settings/IngestPolicyCard.test.ts`,
  `components/AutoLabelPanel.embed.test.ts`,
  `components/SemanticSearchBox.unembedded.test.ts`,
  `routes/p/[project]/settings/ingest-policy/ingestPolicyPage.test.ts`; e2e
  `test_ingest_policy.py`, `test_detector_seed.py`, `test_embedding_state.py`
  and the detector case in `test_ingest.py`.

## Dataset import and Reprocess (`/datasets`, OpenProcessor W10, 2026-09-27)

Import an already-labeled dataset (YOLO, COCO, or an OpenProcessor export)
into the current project, and re-run machine proposals under the lock rule (OpenProcessor `any_domain_plan.md` §7.12 / W10; plan:
`docs/design/w10-import-reprocess-ui-plan-2026-09-27.md`, which also holds the
numbered questions for the backend).
Built against the frozen W10 spec before the backend ships it.

- **Not-yet-deployed gate, not back-compat.** W10's `/health api_features`
  signal was dropped, so `datasetsAvailability`
  (`src/lib/datasets/datasetsAvailability.svelte.ts`) probes
  `GET {API_PREFIX}/datasets/formats` once per project, lazily (only the
  surfaces below call `init()`; never the layout): 404/501 makes every W10
  surface absent, 200 caches the served vocabulary every control renders
  from, anything else shows the error with a Retry. Reset on project
  switch. To remove the gate once W10 ships: drop the 404 branch and
  `available`; keep the vocabulary load.
- **Routes** (`/p/[project]/datasets/...`): `/datasets` redirects to
  `/datasets/imports` (the list, served status chips); `/datasets/import`
  is the wizard; `/datasets/imports/[id]` the job view. `/ingest` links to
  the wizard only when the feature is served. No top-nav entry.
- **Wizard** (`ImportWizard`, `importWizardController.svelte.ts`): a server
  path (the served `ingest/config` `batch.source_roots` listed) or an
  archive upload (`POST /datasets/uploads`, refused client-side above
  `min(served upload.max_bytes, CROPWRIGHT_DATASET_UPLOAD_MAX_MB)`), a
  400 ms-debounced `POST /datasets/preview` of exactly what the operator
  chose (no `project` in any body), the served splits/totals/estimate/
  OP-export facts/issues (`DatasetIssueList`, grouped by served severity
  with the served catalog labels), the class-mapping table
  (`MappingTable.svelte`: by NAME only — the dataset id and source
  registry id are labels, never compared; served suggestion + match label,
  "Use suggestion", served `resolved` target; a row with boxes and no
  `resolved` is highlighted), options that are omitted unless the operator
  set them (server defaults apply), "Start anyway" only while the served
  `force_allowed`, and Start (confirm) with `expected_import_key`. Served
  refusals: `reused` → link, `import_resumable` → Resume, `dataset_changed`
  → re-preview, `class_mapping_incomplete` → the served `unmapped` rows
  highlighted, `import_blocked`/`class_mapping_invalid` → the served issues.
  Every message is the served `detail.message` (`datasetErrorText`,
  `api.ts`).
- **Job view** (`ImportJob`, `importJobController.svelte.ts`): re-reads the
  job after the served `poll_after_s` (null = terminal, stops); a
  `dataset_import.progress|finished` SSE event for this import wakes an
  immediate re-read. Served status label, progress, report counts,
  `waiting_for`, a failed job's served `error.message`, `next_steps`
  (confirm, then run as served), paged issues/entries tables. Cancel /
  Resume / Undo (dry run first: the served `DatasetUndoReport`, including
  "your edits are kept") are confirm-gated; which status offers which is a
  client reading of the spec until the backend serves action flags.
- **Reprocess** (`ReprocessControl.svelte` + `reprocessController.svelte.ts`):
  on `CropMetaPanel` (one crop, `POST /crops/{id}/reprocess`, `dry_run:
false`; the served post-write crop is adopted by `CropDetailModal` and
  `/review`), on `CropCard`'s expanded view and `SlotCard` as "Reprocess
  image…" (`POST /images/{image_id}/reprocess`, `reprocessImage`,
  `api.ts`; the served `items` of that image are mapped like any crop and
  handed to `onreprocessed`; `/clusters/[id]` adopts them by `crop_id` via
  `clusterController.adoptItems`), and the `/clusters/[id]` selection toolbar
  (`POST /reprocess`, served dry run first, then apply; a served
  `ReprocessJobInfo` is followed). The backend serves no Reprocess
  vocabulary (`/datasets/formats` has no `reprocess` block), so the control
  is present whenever `datasetsAvailability.available === true`, its scope
  and region-mode ids are the contract's enums
  (`src/lib/datasets/reprocessVocabulary.ts`, pinned to the vendored
  `ReprocessOneRequest` by `contract/datasetsContract.test.ts`) labelled by
  `humanizeId`, and there is no lock-rule or summary sentence: results show
  the served counts only (`ReprocessCounts`: selected / locked skipped /
  queued / failed / not found, and the breakdown rows; an omitted count
  reads "—"). Question C-1 asks for served labels. On a batch of chosen crops the `embed` scope also offers the served-enum
  embed options (sent only once touched), and `ReprocessCounts` shows each
  scope's served `detail`; see "Detector, ingest policy and embedding state".
- **Proxy.** `nginx.conf` has a `…/projects/*/datasets/uploads` location
  with `client_max_body_size` from `CROPWRIGHT_DATASET_UPLOAD_MAX_MB`
  (default 2048) and a one-hour timeout.
- **Contract.** Types in `src/lib/types_import.ts`, pinned key-for-key to
  the vendored OpenAPI (OpenProcessor f582aa05) by
  `contract/datasetsContract.test.ts`: `/datasets/formats` serves
  `formats: [{format, label}]`, `processing_modes` / `parents_modes` /
  `trust_levels` / `mapping_actions` / `match_kinds` as `{value, label,
description}` and `upload_limits` (no accepted-extension list, so the
  archive input has no `accept` filter). A failed job's `error` is a
  string. The routes are in the vendored OpenAPI and resolve for real in
  `endpointCatalog.test.ts`; the ingest-batch label fields the W10 backend
  422s are gone (see Ingest).
- **W10 leftovers (2026-10-01, Track C).**
  - **`imported` review tab.** `CORE_REVIEW_TABS` has `imported`, shown only
    while `GET /review/tabs` serves an `imported` entry
    (`visibleReviewTabs`, `reviewTabsVocabularyStore.hasEntry`); absent,
    never disabled. A `?tab=imported` link against a backend that does not
    serve it falls back to All with the usual notice once the vocabulary has
    loaded. Its `dataset_split` / `on_negative_frame` filters render through
    the generic served-filter bar. When the served `empty_state.
has_imported_labels` is false (and W10 is served) the empty panel links
    to `/datasets/import`.
  - **URL-seeded `import_id` and `combine_conflict`.** `/review` reads both
    from the URL (`reviewDeepLink`), sends each to `GET /review/{tab}` and
    `/locate` only when that tab's served `filters` list it (nothing is
    guessed before the vocabulary answers, and a link that seeds one holds
    the first queue load until it has), persists them in the URL, clears
    them on a tab click, and shows a removable chip ("Import `<id>`",
    "Combine conflicts only"). No toggle exists when not URL-seeded. The
    import job view's "Review imported labels" links to
    `/review?tab=imported&import_id=<id>` when the `imported` tab is served.
  - **Lock badges.** `CropCard`: a lock glyph when the served
    `label_locked` is true ("Label locked"); `SlotCard` and
    `MultiBoxCanvas`: a lock glyph on a box whose served `locked` is true
    (the review page passes it by box id). No reason text (question C-2).
    `SlotBboxEditor`'s own canvas does not pass `locked`.
  - **Import provenance** (`provenance/ImportProvenanceRows.svelte`, in
    `CropMetaPanel`): label locked, `dataset_split`, `import_ids` (links to
    `/datasets/imports/<id>` while W10 is served, else text),
    `imported_at`, `proposed_by_import`, `on_negative_frame`,
    `import_standalone_region`, `proposal_chain` (chips, verbatim); each row
    only when it carries a value. `Crop.image_id` (wire `image_id`) is now
    carried by `mapRawCrop`; it targets the image Reprocess.
- **Tests:** `api.datasetImport.test.ts`, `datasets/*.test.ts`,
  `components/datasets/*.test.ts`, `contract/datasetsContract.test.ts`,
  `components/provenance/ImportProvenanceRows.test.ts`, the lock and image
  Reprocess cases in `CropCard.test.ts` / `SlotCard.test.ts` /
  `MultiBoxCanvas.test.ts`; e2e `test_dataset_import.py`,
  `test_import_leftovers.py` (conftest serves `/datasets/formats` 404 by
  default).

## Prompt-pack editor (`/settings/prompt-packs`, OpenProcessor W3, 2026-09-27)

Edit, validate, test, save, activate and roll back the project's VLM prompt
packs (OpenProcessor `any_domain_plan.md` §3, §5.1, §7.2, §7.5, §7.6; plan:
`docs/design/w3-pack-editor-ui-plan-2026-09-27.md`, which also holds the
numbered backend questions W3-Q1..Q16). Built against the frozen W3 spec
before the backend ships it.

- **Not-yet-deployed gate, not back-compat.** W3 has no capability
  signal, so `packsAvailability` (`src/lib/packs/packsAvailability.svelte.ts`)
  probes `GET {API_PREFIX}/prompt_packs` once per project, lazily (the
  pack pages and the `/settings` card call `init()`; never the layout):
  404/501 makes every pack surface absent, anything else but 200 shows the
  error with a Retry. Reset on project switch. To remove once W3 ships:
  drop the 404 branch and `available`.
- **Routes:** `/settings/prompt-packs` (the served list and templates, the
  active-pack panel with a confirm-gated Rollback, Clone, confirm-gated
  Delete) and `/settings/prompt-packs/[name]` (the editor). `/settings`
  shows a "Prompt packs" card linking there only when served; its
  `prompt_pack` dropdown is unchanged (it activates the latest revision).
- **Editor** (`packEditorController.svelte.ts`): one field per served
  `/prompt_packs/schema` row, grouped by the served `group` (heading = the
  served call label with that id), with the served help, placeholder and
  reply-key chips; `kind: "map"` fields edit as key/value rows. 400 ms
  after an edit the draft goes to `POST /prompt_packs/validate` with
  `name: null`; every issue renders under the field its served `field` path
  names (`issuesForField`) or at the top. No client validation at all.
  Save is `PUT` with `expected_revision`; a 409 `revision_conflict` offers
  "Reload" or "Keep my edits" (adopts the served `current_revision`).
  Revisions: view read-only, "Restore as new revision" (confirm, `PUT` of
  the old body), "Activate revision k". Read-only packs offer "Clone to
  edit". A `config.changed` event on the `prompt_pack` axis re-reads; a
  dirty draft gets a notice instead of being overwritten.
- **Activation** (`packActive.svelte.ts`, shared by both pages): activate
  and rollback send `expected_active` = the last read's `active`; activate
  pins the revision on screen. A 422 keeps the served report; "Activate
  anyway" (`force: true`) renders only when the report's `force_allowed`.
  `active_conflict` re-reads. The panel shows the served `stale` and
  `applied[].lagging`.
- **Test on a crop** (`packTestController.svelte.ts`, `PackTestPanel`;
  wrappers in `src/lib/api_configTest.ts`, types in `types_configTest.ts`,
  pinned by `contract/configTestContract.test.ts`):
  `POST /prompt_packs/test` is W5, so the panel renders only for served
  `schema.calls[].testable` calls. Sends the draft or the saved revision,
  the call, crop ids, `use_region_box` only when chosen, and the VLM
  selection (`vlmSelection`: `vlm_name` / `vlm_revision` /
  `acknowledge_external`) only when one is set. `TestVlmPicker` (the W9
  `VlmRunPicker`, no-pick option "Active endpoint") sets it:
  `toTestVlmSelection` (`src/lib/configTest/vlmSelection.ts`) sends
  `vlm_name` with `vlm_revision: null`, and `acknowledge_external: true`
  only once the served external warning is ticked; `vlm_draft` is never
  sent. Renders
  the served pack and VLM refs (`name@revision`, `endpoint`, `model`,
  "draft"), latency, top-level parse state, validation, the prompt, raw
  reply and reasoning, and per crop the thumbnail, `box_id`, the served
  `skipped` text or `parsed` JSON and the preview item
  (`components/config/TestPreviewItem.svelte`, shared with the profile
  test: the crop's context with the tested item replaced, drawn by
  `SourceImageOverlay`). `crop_not_found` names the listed ids in the
  input; every other refusal is the served message. A new run aborts the
  one in flight.
- **Contract:** types in `src/lib/types_packs.ts` (the shared §7.1
  models live in `src/lib/types_config.ts`); routes
  resolve for real in `endpointCatalog.test.ts`; `ActiveConfigResponse`'s
  `source`/`activated_at`/`applied[]` are required (served since W2) and
  `AppliedRuntime.vlm` is required-nullable (OpenProcessor d00e8957): `null` renders "not reported" (the worker never reported a VLM axis), a null name renders the axis's `appliedNoneText` ("no VLM"). Refusals render `configErrorText` (the served `message`).
- **Shared with the region-profile editor** (W4, below): the pack
  modules are thin bindings of `src/lib/config/` (`ConfigActive`,
  `ConfigEditor`, `ConfigList`, `ConfigAvailability`, `validationIssues`)
  and the pages use `src/lib/components/config/` (`ConfigActivePanel`,
  `ConfigSavePanel`, `ConfigRevisions`, `ConfigViewingBanner`,
  `ConfigActivateDialog`, `ConfigRestoreDialog`, `ConfigCloneDialog`,
  `ConfigIssueList`, `ConfigGate`).
- **Not yet built:** clone from another project (`from_project`, W3-Q8),
  the `?profile=` validation context (W4), the test panel's VLM endpoint
  picker (W9), create-from-nothing.
- **Tests:** `api.promptPacks.test.ts`, `packs/*.test.ts`,
  `components/packs/*.test.ts`; e2e `test_prompt_packs.py` (conftest serves
  `/prompt_packs` 404 by default).

## Region-profile editor (`/settings/region-profiles`, OpenProcessor W4, 2026-09-27)

Edit, validate, save, activate, roll back and turn off the project's region
profiles, and read the config vocabulary (OpenProcessor `any_domain_plan.md`
§4, §7.3, §7.4, §7.6; plan: `docs/design/w4-profile-editor-ui-plan-2026-09-27.md`,
which also holds the numbered backend questions W4-Q1..Q16 and the spec
conflicts). Built against the frozen W4 spec before the backend ships it,
on the shared config machinery listed under "Prompt-pack editor".

- **Gate.** `profilesAvailability` (`src/lib/profiles/profilesAvailability.svelte.ts`,
  a `ConfigAvailability`) probes `GET {API_PREFIX}/region_profiles` once
  per project, lazily; 404/501 makes every profile surface absent (W4
  serves no capability signal, W4-Q1). Reset on project switch.
- **List** (`profileListController.svelte.ts`): the served profiles and
  templates (`?include_templates=true`), the active profile
  (`ConfigActivePanel` with `PROFILE_ACTIVE_COPY`: Rollback and a
  confirm-gated **Turn off**, `POST /region_profiles/deactivate`; a nameless
  active reads by its served `source`: `off` is "off: region detection is
  off", `env` is "None: no region profile configured" via the copy's
  optional `noneEnvText`; a nameless pack reads a bare "none", since a pack
  is never nameless with source `env`), "Show
  impact" (`GET /region_profiles/active/impact`), Clone, Delete, and a
  collapsed read-only "Models and sources" panel (`ProfileVocabularyPanel`,
  `GET /config/vocabulary`; the vocabulary has no write route).
- **Editor** (`profileEditorController.svelte.ts`): one `ProfileFieldEditor`
  per served schema row, grouped by the served `groups[]`; the control is
  chosen by the served `type` (`string`/`int`/`float`/`bool`/`enum`/
  `string_list`/`int_list`/`float_pair`/`rgb`; anything else edits as JSON);
  `advanced` rows behind a toggle; a row whose `applies_when` is off in the
  saved revision's served `effective` is dimmed, never disabled (W4-Q2).
  `choices_from` resolves through `profileFields.ts`'s `choiceList` (the
  spec's CHOICE_SOURCES table) to the vocabulary's `choice.{id,label}`,
  with the served `empty_choice` first and a stored value the list lacks
  kept and marked. The segmenter group shows the served segmenter cap and
  floor (`SegmenterStatus`). "Include other projects' shared models"
  re-reads the vocabulary with `include_other_projects=true`. Live
  validation posts `{name: null, body}`; "Check the draft for activation"
  posts it with `for_activation=true` and shows that report apart.
- **Activation.** Pins the revision shown; "Activate anyway" only on a
  served `force_allowed`. The response's served `impact` and `validation`
  render (`ProfileImpactPanel`); a served `suggested_reprocess` gets a
  Re-run (W10's `ReprocessFlow` with a `request` target: the served
  request as served, dry run first, apply behind a confirm; only when W10
  serves Reprocess). Every successful activate / rollback / turn-off calls
  `healthStore.poll()`, so `regionProfileStore` raises its existing
  "reload to apply" notice; nothing hot-swaps the region slot.
- **Test on a crop** (`profileTestController.svelte.ts`,
  `ProfileTestPanel`, `ProfileTestLegs`, `components/configTest/`; wrapper
  `testRegionProfile`, `POST /region_profiles/test`, W5): the profile
  schema serves no `testable` flag (question W5-1), so the panel shows
  whenever the editor loads. One crop id, the unsaved draft or the saved
  revision (read-only profiles and a viewed revision offer saved only), a
  segmenter prompt override sent only when non-empty, "Verify with the
  VLM" (`verify: true` only when ticked), and, while verify is on, the same
  `TestVlmPicker` (the selection is sent only with `verify`).
  Renders the served profile ref, "not eligible" (`item_eligible: false`,
  no served reason, question W5-2), validation, the legs (served status,
  reason, time; a candidate table of score / selected / drop reason / mask
  IoU / detector, dropped rows greyed with the reason in the tooltip), the
  candidates over the source image (`SourceImageOverlay`'s `extraShapes`:
  boxes from `bbox_norm`, mask outlines from `mask_polygon`, dropped ones
  dimmed) and in the crop's own frame (`CropFrameShapes`, from the served
  `bbox_in_parent` / `mask_polygon_in_parent`), the preview item under
  "Selection (not verified)" (`selection_accepted`) or "VLM verdicts"
  (`vlm_verdicts`), and the verify block when served. Nothing is projected
  or judged client-side.
- **Gating (v0.4.0).** The profile schema serves a `gating` group (the
  `gate_hit_*` fields that decide when the open-vocabulary pass is gated); the
  schema-driven editor renders it with no code (`profileFields.test.ts` and
  `ProfileFieldEditor.test.ts` cover the group and its rows). The region stage's own pause, resume and
  gate-skipped re-run live on `/ingest`; see "Open-vocabulary sets" below.
- **Not built:** create-from-nothing,
  `validate_segmenter_prompt` (no surface, W4-Q4), `from_project` clone.
- **Contract:** types in `src/lib/types_profiles.ts`; routes
  resolve for real in `endpointCatalog.test.ts`. **Tests:** `api.regionProfiles.test.ts`, `profiles/*.test.ts`,
  `config/configActive.test.ts`, `components/profiles/*.test.ts`,
  `components/config/ConfigActivePanel.test.ts`; conftest serves
  `/region_profiles` 404 by default.

## VLM models (`/settings/models`, OpenProcessor W9, 2026-10-01)

Register VLM endpoints, test them, choose which one a project uses, switch
the local model and read every model choice (OpenProcessor `any_domain_plan.md`
§7.8; contract f582aa05; plan: `docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md`
§3, which also holds the numbered backend questions A-1..A-9). Built on the
shared config machinery listed under "Prompt-pack editor".

- **Global registry, per-project activation.** The registry, schema, validate,
  probe, catalog and local-model routes are GLOBAL (`globalApi()`, one
  registry per deployment: `src/lib/api_vlm.ts`, types `src/lib/types_vlm.ts`);
  only the active read, activate, rollback and deactivate are project-scoped
  (`scoped()`). The registry page says so ("shared by every project;
  activation is per project") and marks this project among each endpoint's
  `active_in` slugs. `api_vlm.ts` is scanned as both a scoped and a global
  file by `endpointCatalog.test.ts`, so every call is written as a literal
  template.
- **Gate.** `vlmAvailability` (`src/lib/vlm/vlmAvailability.svelte.ts`) probes
  `GET {API_PREFIX}/vlm/endpoints` once (404/501 makes every registry surface
  absent: no `/settings` Models card, no link on `/models`, the routes say so,
  no other `/vlm/*` request). Deployment-wide, so it is NOT reset on a project
  switch. The probe is the list read, so its served `labels.status` is kept
  for `/models`. The per-run pickers and the `/settings` dropdown are gated by
  the served `/methods` `vlm` axis instead (`isVlmSelectable`), with no probe.
- **`/settings/models`** (`vlmModelsController.svelte.ts`, a `ConfigList`):
  (1) the project's active endpoint (`ConfigActivePanel` with `VLM_ACTIVE_COPY`,
  Rollback and a confirm-gated Turn off, the served `/health` `vlm` block);
  (2) the endpoints table (`VlmEndpointsTable`): status / locality / source
  through the served `labels`, a red chip with the served `warning` for an
  endpoint that sends crops outside the deployment, the key REFERENCE and
  whether the host has it (`api_key_ref`/`api_key_present`; no key is ever
  read, shown or typed), Activate here, Probe, Clone, and Delete on stored
  rows only (a 409 `in_use` shows the served project slugs); (3) the local
  model (`LocalVlmPanel`): the served catalog facts, Switch behind a confirm
  (`POST /vlm/local/select`; `force` only on the retry after a served
  `vlm_catalog_does_not_fit`), a restart banner with the served reason and the
  copyable command, "Cancel the request", and a re-read of `GET /vlm/local`
  every served `poll_after_s` until it is null. Nothing says a switch
  happened; `serving` flips when the server says so; (4) all model choices
  (`ModelChoicesTable`, `GET /config/vocabulary` `model_choices`, a four-role
  link map in `modelChoiceLinks.ts`). The page wakes on `vlm.changed` (global
  stream; `registry` re-reads the list, `local_vlm` the catalog) and the
  scoped `config.changed axis=vlm` (the active ref).
- **Editor** (`/settings/models/vlm/[name]`, `vlmEndpointEditorController.svelte.ts`)
  and **create** (`/settings/models/new-endpoint`,
  `vlmEndpointCreateController.svelte.ts`): the form is the served schema
  through `vlmFieldAsProfileField` into the region-profile `ProfileFieldEditor`
  (`VlmEndpointForm`); `api_key_ref` is a picker over the served `secret_refs`
  with the empty choice first, `catalog_id` over the served catalog. Live
  validation posts `{name: null, body}` (create: the typed name), "Test
  connection" posts the draft with `probe=true` and shows the served probe,
  "Probe saved" re-reads the doc, a 429 `probe_busy` is the served message,
  `env` endpoints are read-only (Clone to edit; Probe and Activate still
  offered). Save, conflict, revisions, restore, clone are the shared machinery.
- **Acknowledgement.** An activation of an endpoint that sends crops outside
  the deployment shows the served `warning` and an "I understand" checkbox
  (`ConfigActivateDialog`'s optional `ack` prop); `acknowledge_external` is
  sent ONLY when checked, never as `false`; a served 422
  `vlm_external_not_acknowledged` shows its message and the checkbox even
  when the list said not external. The per-run pickers and
  `startAutoLabel`/`runVlmOnCluster` follow the same rule.
- **Per-run selection.** `VlmRunPicker` ("Project default" sends nothing, then
  every served non-disabled entry with its served `endpoint_status_label`; the
  served warning and an ack checkbox only when the entry's `per_run_ack_required`
  is true) is mounted on the assist bar (`AssistScopeBar`; `scope.vlm` /
  `scope.acknowledgeExternal`; a VLM pick forces `run_vlm` like a class or a
  pack; the bar renders when the pack OR the vlm axis is usable) and beside
  "Run VLM" on the dashboard and `/clusters/[id]` (`runVlmOnCluster(id, {vlm,
acknowledgeExternal})`). Run refusals show the served words
  (`vlmRunErrorText`): `unknown_vlm` names the valid ids, the structured
  refusals their own message.
- **`/settings`.** A `vlm` axis (`SETTINGS_AXES`) whose label and blurb prefer
  the served `/methods` `axes[]` entry (`axisCopy`); the dropdown lists the
  served entries (including `off`) with the served status and warning, and an
  external entry with no recorded acknowledgement
  (`sends_images_externally && default_ack_recorded === false`) is disabled
  with a link to Settings → Models. A refused PUT shows the served message
  (and the Models link for an ack refusal) without replacing the page; an
  `unknown_vlm` 422 names the valid ids. A Models card links to the page when
  the registry is served.
- **`/models` and provenance.** One `kind: 'vlm'` row per endpoint (name =
  endpoint name, the served resolved `model`, the served `active` chip, no
  Unload); its status is named through the registry's served `labels.status`
  (raw when unlabeled; `vlmStatusPill.ts`); a "Manage on Settings → Models"
  link only when the registry is served. `VlmProvenanceRows` (mounted in
  `CropMetaPanel`) prints `vlm_endpoint` (`name@revision`), `vlm_model` and
  `vlm_prompt_pack` verbatim, each only when served.
- **Shared-machinery changes** (all additive; packs and profiles unchanged):
  `ConfigActive.activate(..., extra)` spreads `extra` into the body and keeps
  the served `errorDetail`; `ConfigList`/`ConfigEditor` are generic over the
  event type; `ConfigDocBase.active` is optional (a VLM doc serves `active_in`)
  and `validation` may be null; `ConfigActivateDialog` takes the optional `ack`.
- **Questions built by their literal reading** (A-1..A-9 in the plan): this
  project's activation is read from the scoped `GET vlm/endpoints/active`
  beside the served `active_in`; `external_policy` prints as its served id;
  validate sends `name: null` for an existing endpoint; `vlm.changed` is used
  only as a wake-up by `axis`; the role-to-editor link map is client-side; the
  `/health` `vlm` block is read as `{reachable, model?, last_error?, detail?}`;
  a run with no `?vlm=` never sends the acknowledgement; `off` is a served
  entry; a VLM row's `status` is named through `labels.status` when it matches.
- **Not built (out of scope):** the VLM picker on the pack and profile test
  panels (plan §6 I-1, after the W5 panels merge), `vlm_draft`, the `vlm`
  parameter on `label_batch`/`verify_region*` (no surface runs them).
- **Tests:** `api_vlm.test.ts`, `contract/vlmContract.test.ts` (key maps pinned
  to the vendored OpenAPI, strict request bodies through wrappers and
  controllers), `vlm/*.test.ts`, `components/vlm/*.test.ts`,
  `components/config/ConfigActivateDialog.test.ts`,
  `components/provenance/VlmProvenanceRows.test.ts`,
  `routes/p/[project]/{models/modelsVlm,settings/settingsVlm,settings/models/modelsPage}.test.ts`,
  e2e `test_vlm_models.py` and `test_vlm_run_selection.py`; conftest serves
  `GET /vlm/endpoints` 404 by default.

## Open-vocabulary sets (`/settings/open-vocab`, OpenProcessor v0.4.0 SAM 3, 2026-10-03)

Describe things to find in words and have SAM 3 find them (OpenProcessor
v0.4.0; plan: `docs/design/v040-backend-deltas-ui-plan-2026-10-03.md` §6 and
`docs/design/v040-contract-completeness-addendum-2026-10-03.md`; contract
e817e4f4). A set is a named, revisioned list of **targets** (a `prompt`, and
optionally the `class_name` its hits are stored as; an empty class name is
discovery mode, hits stored as unlabeled proposals named by the prompt) with
set-level fields and a `gating` block. One set is active per project. Built on
the shared config machinery listed under "Prompt-pack editor".

- **Gate.** `openVocabAvailability` (`src/lib/openVocab/openVocabAvailability.svelte.ts`,
  a `ConfigAvailability`) probes `GET {API_PREFIX}/open_vocab?include_templates=true`
  once per project (404/501 = every open-vocab surface absent: no `/settings`
  card, the routes say so, nothing else fires). Reset on a project switch.
  `listOpenVocab` always sends `include_templates=true`.
- **List page** (`/settings/open-vocab`, `OpenVocabListState`): the shared
  `ConfigList` plus create-from-nothing (`POST /open_vocab {name, body: {}}`:
  the server fills the defaults), the served templates (clone-only), the active
  set with Rollback and a confirm-gated Turn off (`expected_active` as last
  read; `{name: null, revision: null}` when nothing is active), Clone and a
  Delete at the served revision. A `config.changed axis=open_vocab` event
  re-reads.
- **Editor** (`/settings/open-vocab/[name]`, `OpenVocabEditor`): the shared
  `ConfigEditor` (live served validation posting `{name: null, body}`, Save with
  `expected_revision`, 409 `revision_conflict` Reload / Keep my edits,
  revisions, restore, "Check the draft for activation") plus the structured
  edits the body needs: `addTarget()` builds a target from the served
  `target`-scope schema rows' `default`s (no client defaults), `removeTarget`,
  `moveTarget`, `setTargetField`, `setGatingField`, `setHitRateField`. The
  form is the served schema (`scope` set / target / gating / tier3_hit_rate)
  rendered by `ProfileFieldEditor` through `openVocabFieldAsProfileField`;
  `OpenVocabTargetsTable` is one row per target (basic cells, advanced ones in
  a per-row expander; the class-name cell autocompletes from the project's
  classes but takes any text). A served issue lands under the cell its
  `field` path names (`targets[2].prompt`, `gating.tier3_hit_rate.window`;
  `issuePath`, `unplacedOpenVocabIssues`); the rest show at the top. The served
  `max_enabled_targets` and ceiling are shown as facts; nothing blocks enabling
  more (the validator answers `open_vocab_too_many_targets`).
- **Activation.** The shared `ConfigActivateDialog` pins the shown revision;
  "Activate anyway" appears only on a served `force_allowed`, so a refusal
  that cannot be overridden offers no override. `AppliedRuntime` carries no
  open-vocabulary ref (only `pack` / `profile` / `vlm`), so the active panel's
  applied-runtime table has nothing of its own to show for this axis.
- **Test panel** (`OpenVocabTestPanel`, `openVocabTestController.svelte.ts`):
  `POST /open_vocab/test` runs one unsaved target (any draft row, or the saved
  revision on screen) on one image: a stored crop's (the crop is read and its
  served `image_id` sent, never a crop id) or an upload (`image_base64`, no
  client resize). `gating: {tier2_vlm_precheck: true}` is sent only when the
  pre-check is ticked; `image_max_side` / `dedup_iou` are the draft's own
  values. A new run aborts the one in flight. Renders the served gate
  (`run`, `tier`, `reason`), validation, and every hit (score, selected, dropped
  reason), drawn as served over `SourceImageOverlay` (`extraShapes`) or, for an
  upload, `HitOverlay`; a dropped hit is dimmed. A 502 `segmenter_error` is an
  error banner, never "no hits".
- **Served vocabulary.** `GET /open_vocab/schema` serves
  `vocabulary{statuses, drop_reasons, gate_reasons}`, each `[{value, label}]`.
  The test panel's drop and gate reasons and the re-run buttons below read
  their words from it (`optionLabel`); a value it does not list prints as
  served, never as a guessed label.
- **Segmenter fact.** `GET /open_vocab` serves
  `segmenter{configured, reachable}`; `SegmenterNotice` shows it on both pages
  (ready, configured but not reachable, not configured) and never hides or
  disables the editor (a set can be prepared first). No segmenter state is
  assumed: the notice follows the served fact.
- **Run on existing images** (`OpenVocabRerunPanel`, list page): only while a
  set is active and the backend serves reprocess (W10). Each button opens
  `ReprocessControl` with one request (dry run first, then the apply behind a
  confirm): `{targets: {filter: {all_images: true}}, scopes: ['open_vocab']}`,
  or the same with `filter: {open_vocab_status: [<status>]}` for `pending`,
  `skipped_gate` and `failed` (`done` is not offered), each labelled by the
  served status label and only offered when the vocabulary lists it. The dry
  run's served `detail` and any refusal are the dialog's to show.
- **Item provenance** (`provenance/OpenVocabProvenanceRows.svelte`, in
  `CropMetaPanel`): `source_prompt`, `open_vocab_set@open_vocab_revision`
  (linked to the editor while the gate is open), `region_gate_skip` ("Region
  stage skipped", verbatim) and a "Show matching items" link to
  `/clusters?mode=matching&open_vocab_set=...&source_prompt=...`, each only when
  served. `CropMetaPanel` draws the served `mask_polygon` over the source image
  when the shown crop carries one (list-served crops send `null`; nothing is
  fetched just for the mask). `sourceBadge` gives the class-source role
  `open_vocab` its own sky tone.
- **Region stage** (`ingest/RegionStagePanel.svelte`, `/ingest`, only while a
  region profile is served): `GET /region_stage` (paused, `paused_since`,
  `pipeline_paused`, the worklog counts), confirm-gated Pause and Resume (no
  write before the confirm; the served state is adopted), and "Re-run
  gate-skipped (N)" through `ReprocessControl` with the served `rerun_skipped`
  request, as served. A refusal shows its served message; the region routes'
  409 is `{detail: {error: 'no_active_profile', message}}`
  (`isNoRegionProfileDetail` matches the code).
- **Contract and tests.** Types in `src/lib/types_openVocab.ts` pinned
  key-for-key by `contract/openVocabContract.test.ts` and
  `contract/regionStageContract.test.ts` (including the drop-reason, gate-reason
  and status enums); wrappers `src/lib/api_openVocab.ts` and
  `src/lib/api_regionStage.ts` (scanned by `endpointCatalog.test.ts`).
  Unit and mount tests under `src/lib/openVocab/`,
  `components/openVocab/`, `components/ingest/RegionStagePanel.test.ts`,
  `components/provenance/OpenVocabProvenanceRows.test.ts`,
  `components/CropMetaPanel.openVocabMask.test.ts`,
  `routes/p/[project]/settings/open-vocab/openVocabPage.test.ts`; e2e
  `test_open_vocab.py` and `test_region_stage.py` (conftest serves
  `/open_vocab` and `/region_stage` 404 by default).

## `/review` tab consolidation (2026-09)

A review of all 9 original review-queue tabs against the live
1,000-crop index found three were too big to function as curated
queues — closer to "most of the dataset" than a triaged worklist:
Mismatches (1,000 rows · 97.5% the size of All), VLM Low-Conf
(1,000 · 11% of the dataset), and Primary · Low-Conf (1,000 · 92% of
the _entire_ dataset). A fourth, Outliers, had only 3 live rows and was
functionally identical to the `atypicality` sort already available via
the strategy bar below.

- **Outliers** was retired entirely — no tab, no rendering path. Its
  backend `{API_PREFIX}/review/outliers` query is untouched/unlinked, not
  deleted (out of scope for a frontend-only change).
- **Mismatches / VLM Low-Conf / Primary · Low-Conf** collapsed from
  top-level tabs into **quick-filter preset chips** shown only on the
  `all` tab (`REVIEW_PRESETS` in `src/lib/reviewTabs.ts`). Each chip
  reuses that former tab's exact backend cohort query unchanged — same
  `GET {API_PREFIX}/review/{id}` endpoint, same default sort, same
  `max_rank`/`min_blur_ratio` params for Primary · Low-Conf — just
  triggered from a chip instead of a nav tab. Radio-style: picking a
  second chip swaps the first; clicking the active chip again (or its
  "clear" button) returns to plain All. `resolveEffectiveTab(tab,
preset)` is the single place that decides which queue actually gets
  fetched — every other tab ignores `preset` outright, and clicking any
  nav tab resets it.
- **Uncertainty / Model Disagreements / Classifier Blind Spots
  (`classifier_blind_spots`, `coco_blind_spots` before OpenProcessor
  naming-w2 F7) / the region tab** are unchanged — still real top-level tabs with their existing default
  sorts and keyboard shortcuts.
- **New Class Proposals** (added 2026-09-24, logic-moves W5) is a 6th
  real tab, not a preset — `GET {API_PREFIX}/review/new_class_proposals`
  is a distinct triage workflow (confirm / map-to-existing / create-a-
  class) over crops the VLM flagged as needing a class the registry
  doesn't have yet (`needs_new_class`), shown too big/different a shape
  to fold into the All-tab preset-chip pattern above. Flagged items show
  `needs_new_class_note` inline. `/classes`'s **Proposals** section is a
  separate, aggregate view of the same cohort
  (`GET {API_PREFIX}/review/new_class_proposals/summary` — top proposed
  terms with counts and sample thumbnails, not a paged item queue).
  "Create class & assign" / "Map to existing" (2026-09-24, adopting
  OpenProcessor `2f5cda2`) each dry-run
  `POST {API_PREFIX}/review/new_class_proposals/resolve?dry_run=true`
  first to show the real served `matched` count (every pending item
  proposing the term, server-selected — not the summary's capped
  `sample_crop_ids`) in a confirm dialog, then call it for real. Exactly
  one of `class_id` / `create` is required (422 otherwise; an unknown
  `class_id` is 400, a duplicate `create.class_name` is 409, and a term
  matching more than the backend's per-call cap is 422 with the count —
  all surface as the `ApiError.detail` text in a toast). A successful
  resolve's `updated_ids` are recorded via `undoStore.recordWrites()`
  (the same `Z`-undo ring buffer `bulkLabel`/`moveCropsToCluster` use
  elsewhere) rather than a new bulk-undo button;
  `POST {API_PREFIX}/crops/label/undo_batch` (`undoLabelBatch`, `api.ts`)
  restores each crop to its prior `vlm_new_class_pending` proposal state.
  This replaced the old sample-only flow (`addClass` then `bulkLabel` on
  `sample_crop_ids`) entirely — `bulkLabel` itself is unchanged and
  still used by `/clusters` drag-and-drop. Degrades to an inline error
  banner on a backend failure (observed live: an opensearch aggregation 503) rather than breaking the page.
  - **Flagged terms (DQ-M11, dq-queues cutover 2026-09-24).** The
    summary now also serves `without_term` (pending items with no
    proposed term at all — shown as help text, not a row) and splits
    every named term into `top_terms` (`flag: null` — the only kind
    offered "Create class & assign") and `flagged_terms`
    (`existing_class` / `generic_parent` / `non_object`). `/classes`
    renders `flagged_terms` in a collapsed `<details>` with the served
    reason ("generic parent" / "not an object" / "existing class → map
    to X") and offers **no create action** for any of them —
    `existing_class` alone gets a one-click "Map to `<class>`" using the
    server's own `class_id` (`mapFlaggedTermToClass`, never an
    operator-picked select). This is a server-side fix for the original
    DQ-M11 bug (a one-click create was offered over super-category/junk
    terms like "motorcycle" 89×) — the frontend just renders what's now
    served. `term_rules` (the deployment's generic/non-object term lists)
    renders as help text under the section intro.
- **Served tab/preset labels** (OpenProcessor 1327181 naming sweep, W0
  finding m9): tab and preset _structure/ids_ above are still entirely
  frontend-owned (`REVIEW_TABS`/`REVIEW_PRESETS`), but their displayed
  `label` and tooltip `description` are now overlaid from `GET
{API_PREFIX}/review/tabs` (`reviewTabsVocabularyStore`,
  `$stores/reviewTabsVocabulary.svelte`, loaded once from the root
  layout) when the served vocabulary has an entry for that endpoint id —
  every core tab and slot tab keyed by its `endpointId` (core tabs:
  `id === endpointId`; the region tab: `'regions'`), every preset keyed
  by its own id. Absent or missing an id ⇒ the tab's/preset's existing
  static label, no tooltip — a missing endpoint never breaks the tab bar.
- **Served per-tab filters (dq-queues cutover, 2026-09-24).** The same
  `GET {API_PREFIX}/review/tabs` response now also carries each entry's
  `filters` (the query params that tab honours) and `filter_defaults`
  (values applied when a filter is omitted — `{max_rank: 2}` for the two
  primary-subject tabs, `{}` elsewhere). `reviewTabsVocabularyStore`
  gained `filtersFor`/`filterSupported`/`filterDefault` for this.
  `/review`'s filter bar (Source / Class / Conf / plate-text / subject /
  clarity) renders each control gated on `filterVisible(param)` —
  `reviewTabsVocabularyStore.filterSupported(activeTabEndpointId, param)`
  — instead of unconditionally, and the subject/max_rank toggle's
  "unset" option label reads the served default (`servedMaxRankDefault`
  → `` `Top ${n}` `` when set, else "All ranks") instead of a hardcoded
  "Top 2". `filterSupported` defaults to **visible** when the served
  list has no entry for the tab id (not loaded yet, or an id `GET
{API_PREFIX}/review/tabs` doesn't know, e.g. a tier-2 slot tab) — a
  control never disappears because the vocabulary hasn't loaded.
- **Generic served-enum filter bar (OpenProcessor 3f1a11e adoption).**
  Beyond the fixed `filters`/`filter_defaults` name list above, each
  `GET {API_PREFIX}/review/tabs` entry now also carries `filter_specs` —
  a self-describing list of `{param, kind: 'enum', label, options:
[{value, label}]}`. `/review` renders one `<select>` per entry in
  `activeFilterSpecs` (`reviewTabsVocabularyStore.filterSpecsFor(activeTabEndpointId)`)
  generically — no tab- or param-specific markup in the page, so a
  future spec on any tab (not just the region tab) just works. Picking a value
  calls `setEnumFilter(param, value)`, which writes `enumFilterValues`
  (sent verbatim as its own query param by `_filter()` to both
  `GET {API_PREFIX}/review/{tab}` and its `/locate` route — the backend
  applies its own `filter_defaults` when a param is omitted, so there's
  no client-side default to fall back to) and persists it in the URL the
  same way `?preset=` does. Resets (state + URL) on every tab click — a
  spec is per-tab, so a param from the previous tab never leaks into the
  next. The region tab's `region_status` (`all` / `detected` /
  `verify_rejected`, defaulting to `all`) is the first live spec —
  choosing "Verifier-rejected candidates only" sends
  `region_status=verify_rejected` to `GET {API_PREFIX}/review/regions`.

## Review-queue deep links and filters (2026-09-24 logic-moves W5)

`/review?tab=<urlId>&crop_id=<id>` resolves via
`GET {API_PREFIX}/review/{tab}/locate?crop_id=&page_size=&<filters/sort>`
(`locateInReviewQueue`, `api.ts`) — the backend reports `{in_queue, rank,
page, reason, sort_applied}` for that exact crop under the tab's active
filters/sort, so the page loads `rank`'s page and jumps the
cursor there directly. The queue counter shows the item's position in the
whole served queue (`#N`, from the loaded page's offset — `queuePosition`,
`reviewCopy.ts`; `rank + 1` for the located crop), not its index within the
loaded buffer (F8 D6). `in_queue: false` (already handled, filtered out,
etc.) shows the backend's `reason` in a toast instead of guessing. This
replaced an older approach that paged forward up to 300 items hoping to
find the crop — deleted along with it.

`Crop`/`mapRawCrop` (`api.ts`) now carry `proposed_class_id`/
`proposed_class_name` directly — served on every crop-shaped item, not
just `{API_PREFIX}/review/{tab}` rows — so `getCrop`/`searchCrops`/undo-
restore all return the real served proposal with no client fill-in. The
diverse-selection hydration, semantic-search results and undo-restore
paths that used to hardcode `proposed_class_id: null` (or, for undo,
`crop.class_id`) were deleted for the same reason.

The Class/Source/Conf filter bar sends `class_name` (by name, repeatable; `class_id` was removed from the query in v0.4.0)/`source`/`conf_min`/
`conf_max` to `GET {API_PREFIX}/review/{tab}` — re-enabled unconditionally
once the backend started honoring them (verified live: each changes the
queue total). The control that used to read/write `Crop.hdd_source` (a
field the backend never actually populated) is renamed "Source" and
reads/writes the live `Crop.source` field instead.

`model_disagreements` items carrying `probe_pred_class_id` get an
"Accept model's class" button next to the "Model predicts" row —
`assign()` takes the served id directly, no class-name-to-id lookup
needed (closes G4, tracked since the initial probe-prediction display).
Since OpenProcessor 51b05d7 (F8 D1) the button renders only when the
served `probe_disagreement` is `true` (`probeOpinion`,
`$lib/review/probeOpinion.ts`) — never on a null disagreement — and an
item with `probe_in_scope: false` reads "no opinion (outside the probe's
classes)" instead of showing the probe's out-of-vocabulary prediction.
Since OpenProcessor main 8990ede, the button additionally requires the
served `probe_actionable === true` (server-computed from in-scope +
disagreement + the server's own `OP_PROBE_ACTIONABLE_MIN_CONFIDENCE`
threshold, echoed read-only as `ProbeStatusResponse.actionable_min_confidence`
— no client-side threshold anywhere); a disagreement that isn't
actionable renders as a muted "model unsure: `<predicted class>`" with
no Accept button. class-id-display-audit-2026-09-26: `showAccept` used
to also require a client-side `probe_pred_class_id !== class_id`
comparison — dropped, since it was redundant with (and riskier than)
the served flags above; the served `probe_disagreement`/
`probe_actionable`/non-null `probe_pred_class_id` decide alone.

## Shared item filter and run on selection (OpenProcessor v0.4.0, 2026-10-03)

One filter, three pages (`docs/design/v040-backend-deltas-ui-plan-2026-10-03.md`
section 8, with `v040-contract-completeness-addendum-2026-10-03.md`, which
wins over the plan). **Class identity is by NAME**: `class_id` is gone from
the query of `GET /review/{tab}` (and `/locate`), `/search/text`, `/crops`,
`/regions` and `/clusters`; a class is sent as repeatable `class_name` (and
`exclude_class_name`), spelled as repeated keys by `qs()`. `class_id` stays
only on `/classes/{id}/crops`, `/clusters/representatives`,
`/regions/training_candidates`, `/training_cohorts`, `/viz/projection`,
pipeline start and inside a selection filter body. `/train`'s cohort preview
therefore asks `/crops` and `/review/model_disagreements` by the group's class
name (`runCohortQuery`).

- **State** (`src/lib/itemFilter/itemFilterState.svelte.ts`):
  `ItemFilterState` holds what the operator picked (class names, not-class,
  confidence and area bands, largest N per image, origin, embedding state,
  review status, open-vocabulary set and prompt) and spells it as query
  parameters (`toQuery(allowed?)`: `allowed` drops params a route or a review
  tab does not serve; `withoutOpenVocab` for routes that do not declare the
  pair), as the `ItemFilter` body (`toBody()`: `selection.filter`,
  `export/yolo item_filter`) and as repeatable URL params (`fromUrl`/`toUrl`).
  It counts nothing and validates nothing: a malformed band is the server's 400.
- **Controls** (`components/itemFilter/ServedFilterField.svelte`): one
  control per served `ReviewFilterSpec`, chosen by its `kind` alone (`enum`,
  `multi_enum`, `class_names`, `bool`, `number`, `integer`, `text`) with the
  served label, options, `min`/`max` and `description`. `ItemFilterBar.svelte`
  draws the shared params through it (`itemFilterControls.ts` holds the local
  fallback specs: only the contract's enum value lists; a served spec for the
  same param replaces them) and a removable chip per class name or
  open-vocabulary value. A param the route does not serve is absent, never
  disabled. `/review` draws every other served spec the same way (the page
  keeps its own Source box, conf band, largest-N toggle, blur slider, strategy
  bar and URL-seeded chips, `SELF_DRAWN_PARAMS`), persists the filter in the
  URL and clears it on a tab click.
- **`/clusters`**: the bar above the grid sends the filter to `GET /clusters`
  (the sidebar's `?class=` is looked up in the registry for its name and used
  only when the bar names no class); the semantic search box gets the same
  filter. **Matching items** (`?mode=matching`, `MatchingItemsView`,
  `matchingItems.svelte.ts`) lists `GET /crops` under the filter, plus
  `open_vocab_set` / `source_prompt` (a link carrying either opens this mode
  with those chips), pages with the shared pager, and offers Ignore / Restore /
  Label / Move on all matching items.
- **`/export`**: a collapsed "Only items matching a filter"
  (`ExportItemFilter.svelte`); while the filter names something the line
  "Matching items: N" is the served `total_crops` of
  `GET /stats/dataset?<filter>` (`getMatchingItemCount`), and `exportYolo`
  sends `item_filter` only then.
- **Run on selection** (`selectionActionController.svelte.ts`,
  `SelectionActionDialog.svelte`; wrappers `bulkLabelSelection`,
  `batchExcludeSelection`, `batchUnexcludeSelection` (selects with
  `include_excluded`), `moveSelectionToCluster` in `api.ts`): opening an
  action, or changing the limit / sample / seed (seed only for `random`), runs
  the same write with `dry_run: true` and the dialog states the served
  `selected`; Apply repeats it with `dry_run: false` against the filter as it
  was when the dialog opened. Nothing is counted client-side, and a refusal
  (both or neither of crop ids and filter, an empty filter without a limit,
  too many items) is the served `message`. The served `updated_ids` go to the
  undo stack: label and move through `recordWrites`, ignore and restore through
  new `'exclude'` / `'unexclude'` entries (`undoStore.recordExclusion` /
  `recordUnexclusion`, undone by `batch_unexclude` / `batch_exclude`), so Z
  works (Matching items registers `clusters_search.undo`). The crop-id
  `excludeCrops` / `unexcludeCrops` callers are unchanged and now read
  `updated_ids` too.
- **Region writes** serve `vector_refresh {embedded, pending}`:
  `putRegionBoxes` / `patchRegionBox` return `{crop, vectorRefresh}`,
  `putBatchRegions` / `postBatchBoxState` carry `vectorRefresh`
  (`null` when not served). `multiBoxRegionController.vectorRefresh` keeps the
  last served value and clears it when the next item is seeded;
  `VectorRefreshNotice` (under the box chips on `/review`) reads "N boxes have
  no vector yet" with an "Embed now" Reprocess dialog (`kind: 'request'`:
  `scopes: ['embed']`, `embed.only_missing`, dry run first) when `pending` > 0.
- **Contract tests**: `contract/itemFilterContract.test.ts` (the shared params
  on every list route, no `class_id`, `ReviewFilterSpec` keys and kinds),
  `contract/selectionContract.test.ts` (selection requests, dry-run and
  `updated_ids` responses, `item_filter`, `vector_refresh`). e2e:
  `test_item_filter.py`, `test_selection_actions.py`,
  `test_review_per_tab_filters.py`.

## Curation-strategy selector bar (`StrategyBar.svelte`, 2026-09)

`/clusters`, `/clusters/[id]`, and `/review` all render a `StrategyBar` —
a collapsed-by-default chip row that expands into independent, **stackable**
controls, never a replacement for the production defaults:

- **No cluster-method picker.** An earlier version of this section
  claimed `StrategyBar` included one; it never has. `rg` over non-test
  source finds `cluster_methods` referenced only in `strategies.ts`
  (types/parse) and `strategiesStore.defaultClusterMethodId` —
  a derived value with zero non-test readers. `StrategyBar.svelte`
  renders only the sort dropdown, score chips, and the diverse stepper.
  The first (and, today, only) surface in the app exposing the
  `cluster` axis at all is `/settings` (see above) — it is a
  deployment-wide default picker, not a per-session `StrategyBar`
  control, and was built there deliberately (`docs/design/
curation-settings-ui-plan-2026-09-21.md` §1.5/§2). Production default
  (`ivf`, FAISS IVF-512 + AHC refine, see
  `openprocessor/docs/design/clustering_methods.md`) is untouched by this;
  other methods (`hdbscan`, retired `ahc`-primary) are informational/
  dormant unless explicitly pinned via `/settings`.
- **Review-sort dropdown** — `axis=sort` entries (`review_sorts.py`):
  `recent`, `representativeness`, `atypicality`, `uncertainty_entropy`,
  `mistakenness`, `uniqueness`, `region_score`, `disagreement_entropy_asc`,
  plus each tab's own legacy default (`primary_low_conf_default`,
  `classifier_blind_spots_default`). Never replaces a tab's default — it's an
  additional option layered on top. The bar's collapsed summary chip
  (2026-09-24) also shows the response's served `sort_applied` next to
  the operator's own selection (`formatAppliedSort`,
  `strategyBar.svelte.ts`) whenever it differs — e.g. "sort: Default
  order → atypicality" when no override was picked and the tab fell
  back to its own default, or when a requested sort itself fell back.
  On `/review`'s `all` and `new_class_proposals` tabs — the only two
  with no tuned default of their own (`tabHonorsPinnedSortDefault`,
  `reviewTabs.ts`; see `/settings`'s `sort` axis blurb above) — when the
  deployment's pinned `sort` default has confirmed-zero `/methods`
  coverage (`hasFieldCoverage`) and the operator hasn't picked their own
  override, the chip instead reads "pinned default `<label>` has no
  coverage yet — using `<sort_applied>`" (`formatPinnedSortFallback`,
  `strategyBar.svelte.ts`) — merged with, never doubled up alongside,
  the plain mismatch text above, since both describe the same event.
  (Visual-audit S1's last bullet — that doc's "deferred, BACKEND" status
  was wrong: the coverage signal was already served, just not wired into
  this chip.)
- **Score chips** (`ScoreChip.svelte`) — render inline when a crop carries
  an `axis=score` value (`mistakenness_score`, uniqueness); invisible when
  absent, so an un-backfilled pool renders identically to today.
- **Diverse overlay** (`/clusters/[id]` only) — `order=diverse&k=N`
  pool-scale k-center-greedy selection, gated by `isDiverseOverlayAvailable`
  (`strategies.ts`) on the `diverse` overlay's status being `stable` or
  `experimental`. Never renders against a backend that hasn't shipped or
  enabled it (`OP_SELECT_DIVERSE_ENABLED`).
- **Embedding plot** (`/clusters` only) — 2-d UMAP scatter, **visualization
  only** (never feeds a clustering decision — see `clustering_methods.md`
  §8), colored by the _existing_ `cluster_id`. Gated by
  `isEmbeddingVizAvailable`/`isEmbeddingVizBannerRequired`
  (`strategies.ts`) on the `viz_projection` overlay's status; renders an
  "approximate" banner when the backend flags `requires_banner`. Points
  come from a cached, batch-computed projection (`GET {API_PREFIX}/viz/projection`)
  — never fit on the request path — with an explicit "Rebuild" action
  (`POST {API_PREFIX}/viz/projection/rebuild`, a background job). Click-and-drag
  lassos a set of points; the selected-crop preview renders in a
  side-by-side scrollable column (never stacked below the plot, so a
  large selection can't push the plot off-screen), each thumbnail
  clickable to a full-size enlarged view — the plot alone doesn't show
  enough detail to safely bulk-assign/move a selection blind.

`AssistScopeBar.svelte` (dashboard only, 2026-09-20) reuses this same
collapsed-chip / stackable-controls pattern for a different domain — scoping
a VLM-assisted auto-label run rather than curating a review/cluster queue —
so it is a **second consumer of the pattern, not a fourth control folded
into `StrategyBar`**: `StrategyBar`'s state object is curation
sort/filter-shaped (`toQueryParams()` targets `getReviewQueue`/`getCluster`),
while `AssistScopeBar`'s (`assistScope.svelte.ts`) targets
`startAutoLabel()`'s `class_id`/`prompt_pack` — different domains, same
interaction shape. It is gated by `isScopedAssistAvailable` (`strategies.ts`)
on the `prompt_pack` `/methods` axis, and is absent — not disabled — until a
usable pack is advertised, because `class_id` has no capability signal of its
own. There is deliberately no detection-profile control: region detection is
the backend's startup config (`OP_REGION_PROFILE`), so OpenProcessor rejects a
per-run `detection_profile` with a 422, and `/settings` shows that axis
read-only. See
`docs/design/vlm-scoped-labeling-assist-plan-2026-09-20.md` for the full
contract. Landed on OpenProcessor main (f4551bf): unknown ids 422 with
`{axis, requested, valid_ids}`, resolved values echo in the job's `args`,
and `class_id` scopes only the VLM sweep, not clustering or auto-promote.

Every one of these degrades gracefully to invisible/default when its
backend flag is off or `{API_PREFIX}/methods` fails: `strategiesStore` starts
from, and on a failed load keeps, `EMPTY_METHODS` (`strategies.ts`, nothing
advertised) and records the error, so every optional control stays hidden
and page load never breaks. There is no hardcoded capability list. Flags are OpenProcessor env vars
(`OP_SCORES_ENABLED`, `OP_SELECT_DIVERSE_ENABLED`,
`OP_VIZ_PROJECTION_ENABLED`, `OP_SEMANTIC_SEARCH_ENABLED`), all default
off.

### Served empty-queue reasons (OpenProcessor #36 item 9, 2026-09-25)

`GET {API_PREFIX}/review/{tab}` now serves `empty_reason` directly when
`total === 0` — a plain-English cause ("no probe predictions — run a
probe") distinct from and more direct than `sort_fallback_reason`
(the ordering-degraded-to note above). `PaginatedResponse.empty_reason`
plumbs it into `/review`'s empty-queue panel (`emptyQueueMessage`,
`$lib/review/reviewCopy.ts`), which shows it ahead of
`sort_fallback_reason` when both are set. `GET {API_PREFIX}/review/tabs`
also gained a top-level `empty_state` (`has_probe_predictions`/
`has_item_scores`), read by the same `getReviewTabs()` call
(`reviewTabsVocabularyStore.emptyState`). When the served reason mentions a probe or a score and the
matching `empty_state` flag is false, the panel adds a direct link ("Run
a probe on /train" / "Compute scores on /settings") instead of leaving
the operator to guess where to go — verified live: the Uncertainty
queue's empty panel links to `/train` on this deployment (no probe has
ever run).

## Cluster purity (DQ-M2, dq-queues cutover 2026-09-24)

`GET {API_PREFIX}/clusters`' `purity` used to be tautological for a class
cluster — computed from members' own `class_id`, so `cluster_id ===
class_id` made it read ~1.0 ("pure") by construction regardless of
actual visual/embedding coherence. It's now **nearest-centroid geometry
purity**: the share of the `purity_n` members the cluster-geometry pass
measured for this cluster whose _nearest cluster centroid_ is this
cluster's own — independent of the labels that placed them.
`purity_basis` names the method (`'nearest_centroid'` today, served
rather than hardcoded); `purity_n` is how many members were measured
(purity is noisy at low n). The **old** label-based number survives as
`label_purity` (largest-class share among labelled members — still 1.0
for a class cluster by construction), alongside `labelled_share`
(fraction of members with any label at all). `promotable` is gated on
`label_purity`, not the new `purity`.

`purity_tier` (the pure/mixed/noisy badge/border color) is **unchanged**
by this — still server-banded against `purity_thresholds`, still the
single signal driving `/clusters`' card border and badge.

- **Displayed as "cohesion" (F-37, 2026-09-25).** "purity 19% · noisy" on
  a class cluster whose labels were 90% right read as "bad labels", so the
  UI calls the served `purity` cohesion: `/clusters` cards show
  `"<purity_tier text> · cohesion NN% · n=NNN"` (`cohesionText`,
  `$lib/clusters/clusterCardText.ts`), `/clusters/[id]`'s header shows
  `"· cohesion NN% · n=NNN"`, and both tooltips start with
  `COHESION_TOOLTIP` ("share of measured members whose nearest cluster
  centre is this one") plus label agreement (`label_purity`) and labelled
  share. The sort options read "cohesion asc/desc". Values and tier bands
  are still the served ones; only the word changed.
- All four fields (`purity_n`, `purity_basis`, `label_purity`,
  `labelled_share`) are served on every `RawCluster` (nullable when
  nothing was measured) and render gated on non-null.

## Training UI — `/train` (Phase 2 of the training pipeline)

Full cockpit for the training pipeline. "Classes to train" defaults to the
item classes whose served `trainable_gap` (`GET {API_PREFIX}/classes`: the
shortfall against the per-class hard minimum) is 0, and offers "Exclude
classes without enough data" while any selected class is short
(`$lib/trainClassSelection.ts`, V-4) — no client threshold. The form stays
mounted (disabled) during a run so its choices survive it, past runs render
above the cohorts section, and the live log strips terminal escapes
(`cleanLogLine`, `$lib/logText.ts`) (F-63). The overall eval figure is
labelled "trainer eval (Ultralytics val)"; `/bakeoff` results label theirs
with the result's own served `thresholds` (`protocolText`,
`$lib/bakeoff/view.ts`), since the two protocols differ (V-5).
The form auto-runs `{API_PREFIX}/train/preflight`
on a 350ms debounce and renders the report inline. Status polls every 5s,
log every 2s, both stop on terminal state. Multi-size campaigns get a
size-chip swap in the submit row (auto-promote-best + `stop_when` threshold).

The preflight panel (`TrainForm.svelte`) renders every check generically
by `name`/`severity`/`message` — no per-check-id branching, so OpenProcessor
df01309's three new checks (`export_splits_nonempty`,
`export_class_split_coverage` — blocks per class, thresholds served as
`min_train_per_class`/`min_val_per_class` — and `augmentation_preset`)
needed no component change; each check's `message` already names the
offending classes/splits in prose, and its `detail` object (when present)
renders in a collapsible JSON block alongside it.

Past-runs table actions:

- **Promote ↑** — opens `PromoteModal` (Triton model name, max_batch_size,
  fp16, overwrite). The name defaults to the run's own Triton-safe job id
  (`defaultTritonName`, `$lib/promote.ts`) — no version suffix. Every
  promote-blocking 422 serves `detail: {message, failures: [{code,
message, class_name?}], force_allowed, override?, thresholds?}`
  (`promoteGateDetail`); the modal renders the message, each failure and
  the override hint, and offers a "Promote anyway" checkbox (sends
  `force: true`) only when the server says `force_allowed` (F-64). The
  success toast adds "the first prediction will be slow while its engine
  builds" only when the response serves
  `cold_start_expected_on_first_inference: true` (OpenProcessor ffb88b8;
  `promoteSuccessMessage`, `$lib/promote.ts`).
- **Reproduce** — fetches `{API_PREFIX}/train/manifest/{job_id}` and submits a
  fresh job with the same `spec`/`lineage`. Phase 6 polish — design §15.4.

After a successful promote, the user re-runs `{API_PREFIX}/pipeline/auto_label` and
the **Model Disagreements** tab on `/review` surfaces validated crops
where the new model and the human label diverge — high-signal candidates
for the next training cycle.

### Finished-run results (`RunResults.svelte`, 2026-09-24; metrics/lineage

renamed by OpenProcessor #34 W1, 2026-09-25)

A live train smoke found the backend already serves test-split
evaluation and full lineage for a finished run but `/train` rendered
neither. Every terminal past-run row (`finished`/`failed`/`cancelled`/
`skipped`/`lost`) gets a collapsed "Results" section, `RunResults.svelte`,
rendered as its own table row right under that run — everything except
lineage comes straight off the already-loaded `TrainJobStatus`
(`last_epoch_metric`/`best_checkpoint_metric`/`eval`/`mlflow_run_*`/
`checkpoint_sha256`/`error`), so it renders instantly on open; the
manifest (`GET {API_PREFIX}/train/manifest/{job_id}`, for lineage) is
fetched lazily, only the first time the section is opened.

- **Val vs. test labelling.** The trainer writes `eval.split` on every
  eval block (`'test'` when the frozen-holdout pass ran, `'val'` when it
  fell back); `TrainEval.split` is required. `eval`'s overall figures
  (`map50`/`map50_95`/`precision`/`recall`), the run's **headline
  number**, and its `per_class` table are both labelled by that served
  `split` (`evalSplitLabel`, `src/lib/trainResults.ts`).
- **`eval.head`** (OpenProcessor #34 W1) names the detection head that
  test-split pass scored (e.g. `'end2end'` when YOLO26's NMS-free
  one-to-one head was explicitly forced to match what's actually
  served) — rendered as a small labelled fact next to the overall
  figures when served.
- **Training-time metrics renamed (#34 W1) — `best_metric`/`last_metric`
  are gone, no fallback shim.** `last_epoch_metric` is the true last
  TRAINING epoch's own metrics; `best_checkpoint_metric` is the best
  checkpoint's (best.pt) own re-validation, which Ultralytics performs
  once, after training ends — each one coherent `{epoch, map50,
map50_95}` row, never a per-key max spanning different epochs. Neither
  is the headline number (that's `eval.map50`/`eval.split` above) —
  they render under "Metrics — training epochs", each captioned
  `(epoch N)` when the run carries one. `null` on a run whose
  status.json predates these fields renders "—", never an incorrectly
  back-filled guess (a real bug in the W1 rollout: a pre-fix backend
  back-filled `best_checkpoint_metric` from the wrong, test-split eval
  numbers — fixed server-side, not papered over client-side).
  `TrainProgress.svelte`'s live-run chips are "Last epoch mAP50/
  mAP50-95" (from `last_epoch_metric` — `best_checkpoint_metric` is only
  populated once, at the very end of a run, so it's useless as a live
  progress signal). `CampaignCard.svelte` and the past-runs table's
  mAP50 column (`trainRunsTable.ts`'s `bestMapDisplay`) both show the
  run's own `eval.map50`/`eval.split`, never a training-time metric.
- **MLflow.** A non-null `mlflow_run_url` renders as a link; a null url
  with a non-null `mlflow_run_id` renders the id as copyable text; a
  null id while the run is still active (non-terminal state) shows
  "pending" rather than "—". TODO: the backend is being asked to serve
  `mlflow_run_url` as `null` unless `OP_MLFLOW_PUBLIC_URL` is set — never
  the docker-internal hostname the live fixture still carries today
  (`http://op-mlflow:5000/...`); this view already renders any non-null
  value as a link on the assumption that contract lands.
- **Confusion matrix.** `eval.confusion_matrix_path` (a server
  filesystem path) renders as text only — never an `<img>`. TODO: once
  the backend serves `eval.confusion_matrix_url`
  (`GET /train/artifacts/{job_id}/{name}`), an `<img src>` renders from
  that URL only.
- **Lineage** (manifest, lazy): `export_dir`, `dataset_sha`,
  `dataset_version_tag`, `frozen_test_sha`, `test_label_sha` (the three
  export-identity fields added by #34 W1), `include_classes`,
  `training_seed`, `code_versions.{api_sha, trainer_sha,
trainer_image_id}`, and a class-remap table (new id → original
  id → name) built from the served `class_remap.new_to_original`/
  `names` — no client-side remap math.
- **`/bakeoff`'s model picker** (`TrainedModel`, `types_bakeoff.ts`)
  shows the served `trainer_map50`/`trainer_map50_split` (the run's
  `eval.map50`), labelled as the trainer's own number so it is not
  confused with the comparison's metrics.
- Thin frontend throughout — `formatMetric`/`formatScalar`/
  `metricEpochLabel` (`src/lib/trainResults.ts`) turn a missing value
  into "—"/`null`, never a false 0, same pattern as the existing
  `formatCount()`.
- New types on `types_train.ts`: `TrainEval`/`TrainEvalPerClass`/
  `TrainEvalSplit`/`TrainEpochMetric`, `TrainManifest` and its
  `Lineage`/`ClassRemap`/`CodeVersions`/`Results` sub-types;
  `getTrainManifest` (`api.ts`) is now typed `Promise<TrainManifest>`
  (was `Record<string, unknown>`).
- Tests: `trainResults.test.ts`, `trainRunsTable.test.ts`,
  `RunResults.test.ts` (mount-based, against the real served fixture
  JSON from the live `2026-09-24T23-47-55_yolo26n` run —
  `src/lib/test/fixtures/trainRun.ts`, hand-corrected to the post-W1-fix
  shape for that run's `last_epoch_metric`/`best_checkpoint_metric`
  since it predates both — see that file's doc comment; plus a
  hand-constructed `trainStatusFixtureW1` for the epoch-labelled-metrics
  and `eval.head` rendering paths no live pre-fix run can exercise yet),
  `e2e/stubbed/test_train_results.py`. Verified against the real
  deployed backend at `:5184` via a temporary `vite preview` proxy
  (reverted before commit, never shipped) — that backend still predates
  the #34 W1 fix (both runs it serves show the buggy
  `best_checkpoint_metric` back-fill), so the fixtures above are
  hand-corrected to the fixed shape rather than mirroring the live
  response verbatim.

### Probe control (`ProbeControl.svelte`, OpenProcessor #36 item 8, 2026-09-25)

Every finished run's `RunResults` panel embeds `ProbeControl` above the
metrics section — "Run probe predictions" starts a probe pass
(`POST {API_PREFIX}/probe/run {job_id}`) from that run's exported
checkpoint over the crop pool, populating `probe_pred_*`, the one
prerequisite the Uncertainty and Model-disagreements review queues need
(see "Served empty-queue reasons" below). Same idempotent job-poll shape
as `ScoresCard`/`EmbeddingPlot`: confirm dialog, adopt-in-flight-on-mount
(`GET {API_PREFIX}/probe/status`), explicit Cancel, the served
result/error rendered verbatim.

`$lib/probe.ts`'s `canRunProbe` gates the button on `state === 'finished'
&& checkpoint_path != null` — **not** `checkpoint_sha256`, which a live
check found `GET {API_PREFIX}/train/status/{job_id}` never serves at the
top level (only inside the run's manifest, `results.checkpoint_sha256`,
which this control doesn't fetch — it renders instantly off the
already-loaded status like the rest of `RunResults`). Gating on the sha
hid the button for every real finished run on the live deployment; fixed
before commit and verified live via the temporary preview-proxy
screenshot pass. `GET {API_PREFIX}/probe/status` is a single current/last-job
singleton, not scoped per training run — a probe started for a
_different_ run still polls as "running" here, so the control detects
that via the response's `train_job_id` and shows "already running for
another run" instead of a misleading enabled/disabled button with no
explanation. The effect that adopts an in-flight job only fires when
`canRunProbe` is true, so a `ProbeControl` instance for a non-qualifying
past run makes zero `/probe/status` requests.

### Training cohorts (2026-09-24 logic-moves W6; originally P2.12-P2.14,

docs/genericization-plan-2026-09-13.md §9.2/§9.3)

The **Training cohorts** section on `/train`, grouped by class, now
sources its cohort _definitions_ from the backend:
`GET {API_PREFIX}/training_cohorts?class_id=` returns
`{cohorts:[{id,label,description,cutoffs,endpoint,params,row_kind}]}`,
already resolved for the requested class (`params` carries a real
`class_id`, not a template) — 4 class-agnostic cohorts
(`validated`/`needs_labeling`/`low_confidence`/`model_disagreements`, on
`GET {API_PREFIX}/crops` / `GET {API_PREFIX}/review/model_disagreements`)
plus, only when the backend has a region profile configured,
the region profile's 5 region-training-candidate modes
(`detector_blind_spots`/`low_conf_correct`/`disagreement`/
`human_corrected`/`false_positives`, on
`GET {API_PREFIX}/regions/training_candidates?mode=`). A cohort's numeric
thresholds (e.g. `low_confidence`'s `classifier_conf_lt`) are the
server's, not a client constant.

- `/train/+page.svelte`'s `loadGroupCohorts()` calls
  `getTrainingCohorts(classId)` and maps each served cohort into the
  page's local `CohortSpec` shape (`row_kind: 'region'` → `rowKind:
'slot'`, rendered via `SlotCard`, click jumps to the owning slot's
  review queue; `row_kind: 'crop'` → `rowKind: 'crop'`, rendered via
  `CropCard`, click jumps to the All review tab). Cohort definitions load
  lazily per class group (`IntersectionObserver`, same trigger as the
  existing lazy counts) rather than firing an N-classes request storm on
  mount.
- **Tier-2 fallback:** a slot registered only via
  `annotation-profiles.json` (`parseSlotConfig.ts`, no backend region
  profile at all) still gets a working cohort picker — `loadGroupCohorts`
  merges in that slot's own `capabilities.trainingCohorts.cohorts`
  (declared cohorts only, not `CORE_COHORTS` or a tier-2 `predicate`
  cohort) for any id the server's response didn't already send, via the
  pre-existing `cohortsForClass()` (`src/lib/annotations/cohorts.ts`,
  `predicateCohortsAvailable=false`). The server wins on an id collision
  — with a backend region profile the server already sends all 5 region
  modes, so a hand-declared copy (e.g. in
  `examples/annotation-profiles/license-plate.json`) never actually
  fires, but is kept (not deleted) because `cohorts.ts`'s `CORE_COHORTS`/
  `derivedCohorts`/`cohortsForClass` and this fallback path are exactly
  what a genuinely backend-unknown tier-2 slot needs — see
  `cohorts.test.ts`, `secondSlotIntegration.test.ts` and
  `exampleProfile.test.ts` for that mechanism's own coverage,
  independent of this page.
- `runCohortQuery()` dispatches a cohort's `endpoint`/`params` (served
  verbatim, or the tier-2 fallback's compiled equivalent) to whichever
  existing `api.ts` function already answers that endpoint shape
  (`getTrainingCandidates` / `getCrops` / `getReviewQueue`), keyed
  structurally by `cohortEndpointKind()` — no new endpoint, no new
  response shape.
- Cohorts default to **all non-deprecated classes** rather than syncing
  to `TrainForm`'s own class-subset selection — that tighter coupling
  (lifting `selectedClasses` into the page) is not done; the cohort
  picker is a curation-preview surface only and never filters the
  actual training run (`TrainJobSpec.include_classes`/`single_cls` are
  untouched, and no cohort id is ever sent to `{API_PREFIX}/train/start`).

## Keyboard shortcuts

Class assignment is per-class `hotkey_letter`, configured on `/classes` (or in
the `~` overlay) and routed through `dropOnClassStore` by the layout-level
keydown listener in `src/routes/p/[project]/+layout.svelte`. There is no `1-9, 0` top-N
scheme — it was removed; one binding scheme means no "what does this key do
here?" friction. On `/clusters/[id]` a class letter labels the current
selection (or the just-dragged set); on `/review` it labels the current item.

### Action-id keymap (`keymapStore`, K1 of `docs/design/configurable-keyboard-shortcuts-plan-2026-09-26.md`)

Every other shortcut is a named **action**, not a hardcoded key:
`<context>.<verb>` ids such as `review.queue.discard`, `review.region.back`,
`box_edit.nudge_up`, `cluster.undo` (46 ids, contexts `global` / `review` /
`review.queue` / `review.region` / `box_edit` / `cluster` /
`clusters_search` / `region_gallery`). They are declared once, with their
default keys and overlay labels, in `src/lib/keymapFallback.ts`
(`FALLBACK_KEYMAP`, the same shape the backend's `GET {prefix}/keymap` will
serve) and read through `keymapStore` (`src/lib/stores/keymap.svelte.ts`):
`keysFor(id)`, `label(id, {region})`, `glyph(id)` / `compactGlyph(id)`
(`⇧N`, `↵`, `⌫`), `actionFor(context, combo)` (walks the context's
`includes`), `actionsForContext(context)`.

- **Registration by id.** Pages call
  `keyboardStore.registerAction(id, handler, scope)`. The registration
  stores the id; dispatch resolves its keys through `keymapStore` at
  keypress time, so a rebind applies with no re-registration. The legacy
  `register(combo, …)` form remains for combo-level callers and tests. The
  overlay toggle/close are the `global.shortcuts_overlay` /
  `global.close_overlay` actions (the old backtick layout aliases, including
  `e.code === 'Backquote'`, still match while the toggle is bound to
  backtick).
- **Slot keys.** `buildSlotKeymap` entries carry their `actionId`. Confirm
  and next (scan) and save and cancel (edit) come from the store. The
  slot's own verbs come from its `queue.keymap`, which for the served
  region slot is derived from the store's `review.region.*` keys
  (`servedRegionSlot.ts`), and which a tier-2 slot may still declare
  itself.
- **One box-edit handler.** `/review`'s edit mode (`BboxCanvas.handleKey`,
  via the page's single forwarded window listener) and the `SlotBboxEditor`
  modal both resolve keys through `runBoxEditKey` (`src/lib/boxEditKeys.ts`).
- **Printed keys.** Every hint strip, toast, button badge/title and
  overlay row prints `keymapStore.glyph(id)`, never a literal.
  `src/lib/keymap.literals.scan.test.ts` fails on a literal `<kbd>`,
  "Press X" or "(X)" outside the keymap's own files (the class picker's
  fixed listbox hint is the one allow-listed exception).
- **Locked keys.** Esc, Enter and the four arrows: an action whose default
  has one keeps it, and no other action may take one. Esc-only actions
  are not modifiable. The store enforces this on every document.
  Fixed form, modal and a11y keys (Enter submits, Esc cancels, ↑↓ in a
  listbox, Tab) are not actions and are not routed through the keymap.
- **W8 per-box actions** (`review.region.accept_box` / `reject_box`,
  `box_edit.next_box`) are `available: true`, registered by the
  multi-box region surface (see "W8 multi-box regions" below).
- **K2 (2026-09-26): the served keymap is live.** `loadKeymap()`
  (`src/lib/stores/keymap.svelte.ts`) reads the scoped `GET {prefix}
/keymap` once from the root layout's `load()` and hands the result to
  `keymapStore.setDocument(doc, 'served')`; a 404/501 (a pre-W2b
  backend) sets `keymapAvailability.available = false` and the store
  stays on `FALLBACK_KEYMAP` silently. A `config.changed axis=keymap`
  SSE frame (subscribed in `+layout.svelte`) refetches and applies live
  — dispatch always resolves an action's keys through the store at
  keypress time, so a rebind needs no reload. `/settings#keyboard`'s
  "Keyboard shortcuts" card (`KeymapCard.svelte`) is the editor —
  absent, not disabled, on the fallback: per-context tables, key
  capture (add/remove, up to `grammar.max_combos_per_action`), locked
  actions read-only, a "custom keys" badge off the served `is_default`,
  a debounced `POST /keymap/validate` rendering the server's own
  errors/warnings verbatim (never a client-computed collision), and
  `PUT`'s 409 `revision_conflict` / 409 `class_hotkey_conflict` (offers
  "Unbind these class keys and save", which retries with
  `unbind_conflicting_class_hotkeys: true`) / 422 `validation_failed`
  all rendered per their wire shape. "Reset this action" / "Reset all"
  round-trip `POST /keymap/reset`. The `~` overlay prints one row per
  ACTION (not per key — a multi-key action used to print twice) and
  links to the editor. Plan §5.3's guard is live too:
  `+layout.svelte`'s class-hotkey listener defers to
  `keyboardStore.hasActiveBinding(key)` first, so a registered action
  always wins a same-key collision with a class hotkey. `setClassHotkey`
  renders the backend's structured 422 `hotkey_reserved`/409
  `hotkey_taken` details (naming the owning action(s)/class) instead of
  a generic string. The four `/keymap*` routes are vendored and resolve in
  `endpointCatalog.test.ts` like every other route.
- **K2b (2026-09-26): per-context overrides.** Plan §0 decision 4 — a
  verb rebind applies on every page by default, with a per-context
  override available. `KeymapCard.svelte` now has two sections: a
  **Verb groups** list (one row per served `group` id shared by 2+
  modifiable actions across contexts — `undo`, `confirm`, `discard`,
  `skip`, `prev`, `next`, `select_all`, `ignore`, `nudge`), where
  editing the row's keys writes every member action id at once, and a
  **"Customize per page" disclosure** under each group listing every
  member by its own context label with its own key chips — editing one
  there writes only that action id and detaches it. Detachment is
  computed each render (does this member's draft differ from the other
  members' shared value), never a stored flag, so a "differs from the
  group" marker + "reset to group" also surface a pre-existing
  server-side per-context override for free. Locked keys stay locked in
  both views; a group-level capture rejects any `grammar.locked_keys`
  combo outright rather than checking per-member locked-key sets. The
  per-context tables below the Verb groups section now hold only
  ungrouped actions and locked/non-modifiable actions (the whole
  `cancel` group has no modifiable members, so it never appears in Verb
  groups at all). Writing is unchanged: the same `overrides` action-id
  map, the same `PUT`/`validate`/`reset` calls — a group edit just
  happens to populate more than one key.

Reserved single-char action keys (`g n d z x u a m /`, plus `b f e` from the
region slot's keymap — server-served today as `/abdefgmnuxz`)
cannot be bound to a class — `setClassHotkey` (`src/lib/classHotkey.ts`)
rejects them, validating against `reservedHotkeyLetters()`. As of the
2026-09-24 OpenProcessor `logic-moves` cutover, the base set is no longer a
hand-maintained frontend constant: `GET {API_PREFIX}/classes` serves its own
`reserved_hotkeys` field (`classesStore.reservedHotkeys`), and
`reservedHotkeyLetters(registry)` is that served set (or, once K2 serves a
keymap, `keymapStore.reserved`; it's `null` until then, so the classes
field stays authoritative) **∪ every
single-character combo any registered queue-capable slot's
`QueueCapability.keymap` declares** (P2.8c, closing Finding C.2
structurally). The registry union is redundant against the server's set
today — verified live, the server's `/abdefgmnuxz` already includes
the region slot's `d`/`f`/`e`/`b` — but is what keeps a _second_,
backend-unaware slot's letters safe without a human re-auditing every class
hotkey, since a tier-2 deployment profile can register a slot the backend
has never heard of. `setClassHotkey` shows the server's 400/409/422 detail
text verbatim when a bind is rejected server-side (a race, or a rule the
client-side check above doesn't know about yet). A slot tab already
suppresses class-drop registration entirely
(`isSlotSuppressedTab`/`isSlotTab`), so all of this is defense in depth, not
a fix for a live collision — it's what keeps a bound letter from firing two
handlers on the same keypress. `/classes` shows a banner for any class whose
bound hotkey predates its reservation — live example: `bmw` is bound to `b`,
which is now reserved; the binding is kept, not auto-cleared.

The tables below are the **default** keys (`FALLBACK_KEYMAP`), which is
what every deployment runs today.

Global:

| Key            | Action                                       |
| -------------- | -------------------------------------------- |
| `` ` `` / `~`  | Toggle keyboard shortcut overlay             |
| `Esc`          | Close the overlay                            |
| _class letter_ | Assign that class (selection / current item) |

`/clusters/[id]`:

| Key           | Action                                                                                                                                                                                                                                                                            |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Enter`       | Confirm selected to the chosen class + advance                                                                                                                                                                                                                                    |
| `Shift+Enter` | Accept all VLM suggestions on the page                                                                                                                                                                                                                                            |
| `G`           | Accept the VLM suggestion for selected                                                                                                                                                                                                                                            |
| `N`           | Skip + advance                                                                                                                                                                                                                                                                    |
| `Shift+N`     | Flag selected as needing a new class (curator review)                                                                                                                                                                                                                             |
| `D`           | Discard selected (`POST {API_PREFIX}/crops/{id}/discard`, or `discard_batch` for a multi-select)                                                                                                                                                                                  |
| `Z`           | Undo last action, including a discard or a Reject-VLM (M6/V1, `POST {API_PREFIX}/crops/{id}/vlm_dismiss/undo`) — one press reverts the whole action (bulk label, move, discard, VLM-accept-all, or a dismiss), however many crops it touched, in the order they actually happened |
| `X`           | Ignore selected (exclude from training + clustering)                                                                                                                                                                                                                              |
| `U`           | Undo last ignore                                                                                                                                                                                                                                                                  |
| `A`           | Select all on page                                                                                                                                                                                                                                                                |
| `←` / `→`     | Move the selection by one crop (not page nav)                                                                                                                                                                                                                                     |
| `M`           | Move selected to another cluster…                                                                                                                                                                                                                                                 |
| `Esc`         | Clear drag capture / close picker / clear selection                                                                                                                                                                                                                               |

`/review`:

| Key             | Action                                                                                                                                                                                                                                                                                                               |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Enter`         | Confirm proposed + advance; opens the class picker instead when there's no proposal (plates tab: confirm plate)                                                                                                                                                                                                      |
| `/`             | Open the fuzzy-search class picker (all non-deprecated classes, not just the top-10 quick-assign row). Not offered on the plates tab.                                                                                                                                                                                |
| `D`             | Discard — dismiss from every review queue. Reversible via the "Dismissed" panel (`POST {API_PREFIX}/crops/{id}/review_undismiss`), not via Z (plates tab: reject — no plate visible)                                                                                                                                 |
| `N`             | Skip                                                                                                                                                                                                                                                                                                                 |
| `Z`             | Undo last (one action, however many crops it touched). On the plates/slot tabs, undoes the most recent confirm/reject/FP/box-edit server-side (M6, `POST {API_PREFIX}/crops/{id}/region/undo`) — distinct from `←`/`B` step-back below, which only re-queues the crop locally without touching what the server saved |
| `←` / `→`       | Previous / next item (plates tab `←` / `B`: step back — re-opens the last confirmed item locally so it can be re-edited; does not by itself undo the server write, use Z for that)                                                                                                                                   |
| `F`             | Plates tab: mark false positive (box kept)                                                                                                                                                                                                                                                                           |
| `E`             | Plates tab: enter bbox edit mode                                                                                                                                                                                                                                                                                     |
| `Enter` / `Esc` | Plates tab, edit mode: save bbox / cancel edit                                                                                                                                                                                                                                                                       |

### Slot-generic review tabs (P2.8b/P2.8c, docs/genericization-plan-2026-09-13.md §9.5;

panel body generalized by C6, docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §6)

The ~12 hardcoded tab-id call sites that used to gate the region-tab-only
behavior above are gone. `review/+page.svelte` derives one value,
`activeSlot` (`REVIEW_TABS.find((t) => t.id === tab)?.slot`), and every
site above reads from its capabilities instead: the keymap comes from
`buildSlotKeymap(activeSlot, editMode, handlers)` (`src/lib/review/
slotKeymap.ts`, reading `activeSlot.capabilities.queue.keymap` — no
second hand-maintained copy), the bbox canvas mounts when
`activeSlot.capabilities.subBox` exists, the false-positive action/hotkey
only renders when `activeSlot.capabilities.lifecycle?.falsePositiveState`
is set, and the text-filter input reads its label/placeholder from
`activeSlot.capabilities.queue.textFilter`. The internal tab id is the
structural `slot:${key}` template (`ReviewTab` = `CoreReviewTab |
SlotReviewTab`). The served region slot's `urlId` (the `?tab=`
bookmark value) is the backend's own tab id, `regions`
(`tabFromUrlId('regions')` → `'slot:<profile name>'`, in
`src/lib/reviewTabs.ts`); a tier-2 override may declare its own, e.g.
the license-plate example keeps `plates`.

The inline review panel body (score / status / detector / OCR text /
rejection reason / confirm-reject-FP-back buttons) is now generic too —
this was Finding D, a real gap through 2026-09-21: the tab shell, keymap
and gating were already slot-generic, but the panel body underneath
still read the old per-domain item fields and a directly-imported
built-in slot regardless of the active tab. It now reads every
value through `slotOf(current, activeSlot)` (`src/lib/annotations/
cropSlots.ts`, off `Crop.slots` — populated by `mapRawCrop` via
`mapCropSlots`/`readSlot`), and every label/status-vocabulary/copy
string through `src/lib/review/slotPanel.ts`'s `humanWritableStates` /
`statusClearsBox` / `statusWantsRejectionReason` / `panelLabels`. Writes
go through `setSlotBox`/`patchSlotMeta` (`api.ts`), which target the
active slot's own declared `endpoints`/wire field names — never a
hardcoded `/crops/{id}/region` or `region_status` literal, and render
the server's own returned item rather than a client-computed post-write
state. A second queue-capable slot registered in `registeredSlots.ts`
now gets a fully working review tab — shell, keymap, hint strip, AND
inline panel body — with zero further edits to `review/+page.svelte`,
proved by `src/lib/annotations/secondSlotIntegration.test.ts`.

Since 2026-09-24 (logic-moves W2), `slotPanel.ts`'s three functions
above take an optional third/second `served` argument — the deployment's
region-status vocabulary from `GET {API_PREFIX}/regions/statuses`
(`regionStatusesStore`, loaded once in the root layout). `review/+page.svelte`
passes `regionStatusesStore.list`; when it's loaded, the status dropdown,
clear-box and rejection-reason behavior all read the server's own
`human_writable`/`clears_box`/`wants_reason` flags. A slot's own
`capabilities.lifecycle.states` is kept only as the fallback for when
the endpoint is absent or hasn't loaded yet. Box edits (`saveBboxAndExit`/
`confirmSlot`) send the box exactly as drawn (crop-local, parent frame)
with `setSlotBox(..., 'parent')` — the server does the projection into
its own stored frame, so there is no client-side `projectFromParent` on
the write path (read-side projection is unchanged, and prefers a served
`region_bbox_in_parent` when present — see "Plate provenance + OCR"
below).

The class picker (`src/lib/classPicker.ts`) is a fuzzy-search combobox over
every non-deprecated class — the top-10 quick-assign row under the crop
(`quickAssignClasses`, `src/lib/classPicker.ts`) only ever surfaces the most-validated
classes, leaving the long tail (including brand-new, zero-sample classes)
reachable only via `/classes` without it. `/` is reserved
(`RESERVED_HOTKEY_LETTERS` in `src/lib/classHotkey.ts`) so a class can never
be bound to it.

`Esc` cannot cancel an in-progress pointer drag — `svelte-dnd-action` only
Escape-cancels keyboard (aria) drags. It clears the captured multi-drag set
and restores the grid layout; the drag itself ends on pointer release.

### Served item-vs-region class kind (OpenProcessor #36 item 1, X2/R1, 2026-09-25)

`GET {API_PREFIX}/classes`/`/stats/classes` now serve `kind` (`'item'` |
`'region'`) and `trainable`/`trainable_gap` directly on each class
(`RegistryClass`/`StatsSummary.per_class`) — `sample_count`/
`validated_count` no longer include region counts (X2). `$lib/
classVisibility.ts`'s `isSlotBoundClass` reads the served `kind` alone
(`RegistryClass.kind` is required) — the R1 fix (the region class
topping the item-class picker and quick-assign row) is backed by the
server's own classification, not a client heuristic keyed off the slot
registry. `/export`'s per-class table (`exportDatasetRows.ts`) shows the
served `trainable`/`trainable_gap` and the served per-bucket holdout
`deficient` flag; there is no client-side validated-minus-holdout math.
Its Gap column is the served `trainable_gap` — the shortfall against the
served per-class hard minimum (`thresholds.block_below`), not against the
aug target — and its tooltips say so (`gapCellTitle`). `/dashboard`'s
class-balance bars are sized by the same served `trainable`.

## Data integrity

- Every label change → immediate API call. No "save" button.
- Optimistic UI with error rollback toast on API failure.
- Audit trail lives server-side in the items index (`{label_source,
label_validated, class_source, updated_at}`, plus `class_id_history`).
- Bulk ops show a confirmation dialog with affected count.
- Test-set crops (`test_holdout=true`) are filtered out at the API level —
  the UI never receives them. Don't try to bypass.

## `/models` unload gating (2026-09-25 follow-up to OpenProcessor #36 item 5)

`GET {API_PREFIX}/models/status` serves `unloadable: boolean` on every
entry (`false` for every external-service entry — the segmenter, the
VLM — and `true` for Triton models) and widens `is_region_protected` to
cover the ingest primary proposer/secondary classifier and the OCR
det/rec pair, not just the region detector — every one of these
hard-blocks `DELETE {API_PREFIX}/models/{name}` server-side (403, no
`force` override). `$lib/modelUnload.ts`'s `unloadButtonState` reads
`unloadable` FIRST (`false` hides the button outright, ahead of
`is_region_protected`; `unloadable`/`optional` are required). A
region-protected-but-otherwise-unloadable model still shows no button,
but `/models` now renders a "protected: in use by the pipeline" chip in
that case (`showsProtectedChip`) instead of nothing. The segmenter can
also serve `status: 'not_configured'`, with null inference/exec/latency
fields — rendered "—" by the existing `fmtCount`/`fmtMs`, never `0`.
Live, this leaves only the CLIP/PE image-encoder models with an Unload
button.

Since OpenProcessor ba88751 each entry also serves `optional` (true only
for the region profile's detector when a segmenter covers the same job)
and a `not_installed` status (optional and absent from the Triton
repository entirely). `$lib/modelStatus.ts`'s `modelStatusPill` renders
it as a neutral "optional · not installed" pill rather than a warning,
and the protected chip is dropped for a model that isn't installed
(`isInstalled`). An installed-but-unloaded optional model still reads
"not ready". Covered by `src/routes/p/[project]/models/modelStatus.test.ts`.

## W8 multi-box regions (lockstep branch `feat/w8-multibox-lockstep`)

A region item can carry an **unbounded list of boxes**, not one — the
owner's binding rule (approved 2026-09-26): "a region is a list per item;
one element is not a special case." No client cap; the only limit is the
served `region_profile.limits.max_boxes_per_write` (a request-size guard,
not a labeling rule). Built against, and merged only alongside, the
backend's W8 wave (`openprocessor/docs/design/
openprocessor_internal/any_domain_plan.md` §7.7); see
`docs/design/w8-multibox-frontend-plan-2026-09-26.md` for the full wire
model, write-path table and the current gap list.

**No backward compatibility (owner decision, 2026-09-26): the single-box
scalar region fields are gone, not additive.** `REGION_SUB_BOX` declares
only `listField: 'region_boxes'` — no `bboxField`/`scoreField`/
`candidateBboxField`/etc. `readSlot` never runs the legacy scalar-box
block for a capability that declares `listField`; `region_bbox_norm`,
`region_candidate_*`, and every other pre-W8 per-box scalar key are gone
from the wire and from this codebase's reads. A `SubBoxCapability` declares exactly one of `listField` or
`bboxField` (a read-only scalar box for a tier-2 slot, e.g.
`aircraftTailNumberSlot`; `parseSlotConfig` rejects both/neither). `setBox`/`clearBox` are gone from
`REGION_ENDPOINTS` (`PUT /crops/{id}/region` is a removed 410 route).

- `SlotData.subBoxes: SlotBox[]`, populated by `readSlot` from
  `SubBoxCapability.listField` — `src/lib/annotations/types.ts`/
  `readSlot.ts`/`servedRegionSlot.ts`. `SlotCard.svelte`,
  `CropMetaPanel.svelte` and `SourceImageOverlay.svelte` all render every
  box in `subBoxes` (state-styled: accepted=green, proposed=amber,
  rejected/false_positive=dashed zinc) — none of them read the region
  slot's `subBox` (singular) anymore.
- Pure box-editing/write-body logic in `src/lib/annotations/multiBox.ts`,
  including the owner-decided **Enter confirms only `proposed` boxes**
  rule — a whole-set confirm never overrides a per-box decision;
  `rejected`/`false_positive` boxes are left exactly as they are.
- `api.ts`: `putRegionBoxes` (`PUT /crops/{id}/regions`), `putBatchRegions`
  (`PUT /crops/batch_regions`), `patchRegionBox` (`PATCH /crops/{id}/
regions/{box_id}` — the per-box accept/reject keys, `y`/`r`), and
  `postBatchBoxState` (`POST /regions/batch_box_state`, region-cluster
  triage, called from the gallery's open-cluster toolbar).
- `MultiBoxCanvas.svelte` — select/add/delete/Tab-cycle/arrow-nudge over
  an unbounded box list; the only box canvas (`BboxCanvas.svelte` is
  deleted). Its keys resolve through `keymapStore.actionFor('box_edit', ...)`,
  so a rebound delete/nudge key applies.
- `/review`'s region tab is fully wired to `MultiBoxCanvas` +
  `multiBoxRegionController.svelte.ts` (new controller, following the
  `reviewController.svelte.ts` extraction convention), in both scan and
  edit mode — `y`/`r` PATCH the selected box immediately without
  advancing the queue; Enter confirms proposed boxes and flushes any
  pending geometry edit in the same write; the on-screen Confirm/Save-
  bbox buttons (not just the keyboard path) branch on `isMultiBoxSlot`.
- `keymapFallback.ts`'s `review.region.accept_box`/`reject_box`/
  `box_edit.next_box` are `available: true` on this branch.
- `SlotGallery.svelte` shows the served `total_rows` (box count) beside
  the item count when they differ, and region cluster cards show
  `box_count` beside `size`; the `has_rejected_box` region-status filter
  option needs no frontend code (it's one more value in the existing
  served-enum `region_status` filter). Bulk triage from an open cluster
  bucket (`gallery.selectedCluster != null`) goes through
  `applyBoxState()`/`POST /regions/batch_box_state` (per-box targets from
  each row's `region_box_id`) — never the item-level `batch_status`,
  which would flip every sibling box; outside a cluster the toolbar is
  unchanged.
- `SlotBboxEditor.svelte` (the `CropCard` pencil ✎) is multi-box too —
  it reuses `MultiBoxCanvas`/`multiBoxRegionController` (a new
  `saveEdits()` method: a plain `PUT .../regions` with no `region_status`,
  since this modal has no confirm concept) rather than a second
  implementation, and opens only for a multi-box slot.
- The served `region_profile.limits.max_boxes_per_write`
  (`ServedRegionProfile.limits`, `SubBoxCapability.maxBoxesPerWrite`)
  gates Add in both `MultiBoxCanvas` instances (`/review`,
  `SlotBboxEditor`) — never a client-guessed cap (a tier-2 profile replacing the
  served region slot inherits the served limit, `withServedLimits`).
- `GET /regions/statuses`' `box_states` vocabulary
  (`regionStatusesStore.boxStates`/`boxStateInfo`/`boxStateByRole`)
  supplies box labels and the `dashed` flag when loaded. Each entry also
  serves `tone` (`'accepted' | 'proposed' | 'rejected' | 'neutral'`,
  2026-09-26 backend follow-up) — `regionStatusesStore.boxStateTone(state)`
  resolves it (falling back to `'neutral'` on a pre-tone backend or an
  unrecognized value), and `toneRingRgb`/`toneBorderClass`/`toneChipClass`
  (`regionStatuses.svelte.ts`) map a tone to the theme's colors for every
  ring/chip surface (`/review`, `SlotBboxEditor.svelte`,
  `SourceImageOverlay.svelte`, `CropMetaPanel.svelte`) — never a client
  role→color guess when a tone is served.
- **Contract: real wire (OpenProcessor f582aa05).** Every W8 route, the
  `region_boxes` element keys (pinned in `wireKeys.test.ts` against the
  vendored `RegionTestCandidate` schema) and `region_profile.limits` are
  vendored; no pending-backend allow-list remains. W8's URLs are
  project-scoped (`scoped()`), and its e2e tests open `/p/default/...`.
- **Optimistic concurrency.** Every item-level box write sends
  `expected_region_revision` (the item's served `region_revision`,
  tracked by `multiBoxRegionController`); a 409 `region_conflict`
  (`regionConflictDetail`, `api.ts`) carries the current item, which the
  controller adopts (`onitem`) so the retry carries the fresh revision.
  The summary fields `region_count`/`region_rejected_count`/
  `region_max_score`/`region_set_complete` render as served (an incomplete
  set shows a chip). Per-box detector, verdict, lock, cluster and text
  come off each `region_boxes[]` element (`SlotBox`); per-box text is
  written with `PATCH .../regions/{box_id}` — there is no item-level
  `region_text` any more. The region thumbnail route requires `box_id`
  (`SlotCard`, from the box's served `thumbnail_url` when present).
  `PUT /crops/batch_regions` takes no `frame` key. The gallery's "Box
  state" filter sends `box_state`, and a served `rows_truncated` shows a
  chip. A tier-2 scalar-box slot (`bboxField`) is now read-only (display,
  `CropCard` rings, overlay): the backend has no single-box write route,
  so `setSlotBox`, `BboxCanvas` and the legacy review edit path are gone.

- **`vector_refresh` (OpenProcessor e817e4f4).** Every region write serves
  `vector_refresh {embedded, pending}`; the wrappers return it
  (`putRegionBoxes` / `patchRegionBox` as `{crop, vectorRefresh}`),
  `multiBoxRegionController.vectorRefresh` keeps the last one and `/review`
  shows "N boxes have no vector yet" with "Embed now" while `pending` > 0
  (see "Shared item filter and run on selection").

### Model sharing (projects P2, OpenProcessor be20dc40, 2026-09-27)

`docs/design/projects-p2-sharing-pause-ui-plan-2026-09-27.md`. A promoted
model belongs to one project; its owner may share it, and its classes
reach another project by name only. Every `/models/status` entry serves
`project` (the owner slug, `null` for a base model or an external
service), `shared` and `class_mapping` (`{mapped_count, unmapped}`, `null`
for a model with no class list). `ModelSharingInfo.svelte`
(`$components/models/`) renders, per card: another project's model gets a
"from `<display name>`" chip and "shared"; the active project's own model
shows its served sharing state and the owner-only "Share with other
projects" / "Stop sharing" button; any served `class_mapping` shows "N
classes map", the served unmapped names and a lazy "Class mapping" table
(`GET {API_PREFIX}/models/{name}/class_mapping`: model class name ->
project class name and the served match label; class ids are never
rendered). `ShareModelDialog.svelte` confirms, then
`PUT {API_PREFIX}/models/{name}/sharing` with the served sharing revision as
`expected_revision` (`$lib/models/modelSharingController.svelte.ts`);
refusals show the served message, a 409 `revision_conflict` reloads the
list so the retry carries the fresh revision, and a 409 `in_use` lists each
served `used_by` project with its profile (typed `ModelInUseDetail`; those whose ACTIVE detection profile uses the model) and
offers a confirm-gated "Unshare anyway" that retries with `?force=true`
(arming it shows a warning and sends nothing); a 503
`config_store_unavailable` (the server could not read every project, only
when unsharing; the backend confirmed force is intended there) shows the
served message verbatim and offers the same two-step "Unshare anyway" plus a
plain "Retry". A forced success toast names the served `used_by`
`{project, profile}`. The unshare copy says the server refuses while another
project's active profile uses the model. The listing serves `owned` and `sharing_revision` (non-null only
when owned): `sharingRole`/`canToggleSharing` (`$lib/modelSharing.ts`)
read the served `owned` flag, the toggle is absent without a served
revision (a model with no stamp yet), and a foreign project's model is
served `unloadable: false`, so it shows no Unload. VLM rows serve
`kind: 'vlm'` with `active`/`active_in`, shown as an "active" chip.
`/bakeoff` is unchanged: its lists don't carry other
projects' models (BA-P2-3). Tests: `modelSharing.test.ts`,
`routes/p/[project]/models/modelsSharing.test.ts` (mount),
`contract/modelsContract.test.ts`, e2e `test_models_sharing.py`.

## Plate provenance + OCR (Wave 1 + Wave 2b, 2026-05-11)

Every plate-bearing crop now carries detector provenance — which
model produced the bbox, plus the VLM-read plate text. These flow
through `mapRawCrop` (`src/lib/api.ts`) onto the `Crop` type and
render via the shared chip components.

**New fields on `Crop` / `ReviewItem`:**

- `region_detector` (`'lpr_nanov11_640'` / `'sam3'` /
  `'paddleocr_det_trt'` / `'human'`)
- `region_detector_version`, `region_bbox_frame` (always `'source'`)
- `region_bbox_in_parent` — the region box already projected into the
  parent crop's frame, server-computed. The region slot declares it
  as `subBox.bboxInParentField` (`REGION_SUB_BOX`,
  `src/lib/annotations/servedRegionSlot.ts`); `readSlot` renders from it directly
  when present, falling back to the client-side projection only when
  it isn't (2026-09-24, logic-moves W2).
- `region_detector_chain` — string array, every step of the cascade:
  `['lpr_nanov11_640:miss', 'sam3:hit', 'sam3:vlm_verify_ok']`
- `region_verifier` (`'gemma-4-e4b'` / `'human'`), `region_verified_at`
- `region_text` + `region_text_source` + `region_text_confidence` —
  the VLM reads the plate during verify in the same round-trip
- `region_rejection_reason` — set when the candidate was rejected: a
  verifier verdict (`region_visible_elsewhere`), an automatic geometry
  gate (`sanity_reject:<gate>`), or no verdict at all
  (`verifier_no_verdict` — needs human review, not a rejection). Labeled
  by `GET {API_PREFIX}/regions/vocabulary`'s `rejection_reasons`
  (openprocessor fix #29 / OpenProcessor 3f1a11e) — see
  `regionVocabularyStore.rejectionReasonLabel`/`rejectionReasonKind`
  below.
- `region_text_engine_version`, `region_text_vlm`, `region_text_ocr`,
  `region_text_disagreement` (2026-09-24, logic-moves W8): `region_text`
  is the backend's _chosen_ reading; `region_text_vlm`/`region_text_ocr`
  are the two readers' own candidates, and `region_text_disagreement` is
  true when they differ. `TextCapability` carries these as
  `vlmValueField`/`ocrValueField`/`disagreementField`;
  `SlotData.text.{vlmValue,ocrValue,disagreement}` is populated by
  `readSlot`. `SlotCard` and `/review`'s inline slot panel both render a
  "readers disagree" badge with the two candidate values in its tooltip;
  `CropMetaPanel` additionally shows `region_text_engine_version`.
  `ProvenanceChip`'s muted-tag pattern now includes the
  `accepted_unverified` chain event (a step the backend accepted without
  a verification pass), rendered the same as a miss/reject.
- `item_text_lines` (`ItemTextLine[]`: `text`, `confidence`, `box_norm`,
  `rel_height`) — OCR text on the _item_ crop (not a slot's sub-box).
  `box_norm` is already normalized in the item-crop frame, so
  `CropMetaPanel` draws it directly as a percentage-positioned overlay
  over the item thumbnail, toggled by a "show boxes" button.
- `class_excluded`, `excluded_reason`, `excluded_at` — the "Ignored"
  bucket (`POST {API_PREFIX}/crops/batch_exclude`/`batch_unexclude`,
  `cluster_id=-2`). `/clusters` has an "Ignored" toggle that lists
  excluded crops (`GET {API_PREFIX}/crops?cluster_id=-2&include_excluded=true`)
  with a "Restore selected" action; `CropMetaPanel` shows the reason/
  timestamp as a banner.
- `source` replaces the dead `hdd_source` Crop field (`mapRawCrop` never
  populated `hdd_source` — the backend only ever emitted `source`).
  `/review`'s source-image panel reads `current.source`. The OpenProcessor
  1327181 naming sweep (F9) removed the `?hdd_source=` query param
  outright — `CropFilter.source`/`?source=` is now the only spelling
  everywhere, including `/review`'s diverse-selection scope's term
  filters (`termFilters()`), which used to send the same value under the
  now-dead param name.

There is no shape-plausibility warning: the ⚠ badge (and the client-side
envelope check that computed it — `shapeGate.ts`, `PLATE_SHAPE_ENVELOPE`,
`SlotData.subBox.shapeWarning`) was deleted 2026-09-24 (logic-moves W2)
— the backend never served this flag, and OpenProcessor's own geometry
guard (`is_plausible_region_bbox`) is a different, server-side-only
check with no client mirror.

### Rejected candidates, auto-confirm, text choice (dq-region, 2026-09-24)

OpenProcessor `main` 22a3e65 ("region verdict integrity") changed three
things about the region wire shape, all adopted here:

- **`verify_rejected` carries no `region_bbox_norm` at all.** The box
  the detector proposed and the verifier rejected lives in
  `region_candidate_bbox_norm`/`_score`/`_detector`/
  `_detector_version`/`_source` (+ `_bbox_in_parent`) instead, kept for
  human review and reversal — `region_rejection_reason` is set
  (`region_visible_elsewhere` / `sanity_reject:<gate>` /
  `verifier_no_verdict`). `SubBoxCapability` gained
  `candidateBboxField`/`candidateBboxInParentField`/
  `candidateScoreField`/`candidateDetectorField`/
  `candidateDetectorVersionField`/`candidateSourceField`;
  `readSlot`/`SlotData.subBox.candidate` populate it the same
  server-projection-preferred way as the main box. **Confirming (or
  marking false-positive on) a verify_rejected item promotes the
  candidate into the region box server-side** (`region_writes.py`'s
  `candidate_promotion`/`human_status_fields`) — the frontend never
  computes that promotion; `/review`'s `_seedSlotFromCurrent` just seeds
  `editedSlotBox` from the candidate's parent-frame box when the main
  box is absent, so an unchanged Confirm still goes through the existing
  boxUnchanged → status-only-PATCH path (see B2 above). The candidate
  box renders **dashed** (`BboxCanvas`'s `dashed` prop) with a hint
  styled/worded by the served rejection **kind** (OpenProcessor 3f1a11e
  adoption, see below) — "rejected candidate · confirm to accept" (red/
  amber) for a `model_verdict`/`automatic` reason, "candidate · needs
  review" (neutral zinc) for `needs_human` — since `verifier_no_verdict`
  means "needs human", not "wrong box" (`region_bbox_correct === false`,
  surfaced as a "model: box wrong" chip on the Validation row, is the
  actual "model said wrong box" signal). `SlotCard` shows a matching
  kind-colored "candidate" badge; `CropMetaPanel` shows a Candidate row
  plus the same kind-styled Rejection/"Needs review" row.
- **`region_validated` is human-only; `region_auto_confirmed` is new.**
  `region_verified` keeps its prior, distinct meaning ("a verification
  pass ran", human or the VLM verifier) — it no longer doubles as "human
  accepted this". `LifecycleCapability` gained `validatedField`/
  `autoConfirmedField`; `/review`'s Status row, `SlotCard` and
  `CropMetaPanel` all render a "human validated" / "auto-confirmed
  (unreviewed)" badge off `lifecycle.validated`/`lifecycle.autoConfirmed`
  rather than `lifecycle.verified`.
- **`region_text_choice`/`region_text_vlm_invalid`.** `region_text` is
  still the backend's chosen reading; `region_text_choice`
  (`readers_agree`/`vlm_preferred`/`vlm_only`/`ocr_only`/`ocr_mode`/
  `vlm_invalid`/`no_valid_reading`/`human`) says why it won, and
  `region_text_vlm_invalid` (`placeholder`/`no_reading`/`sequence`/
  `charset`/`too_short`/`too_long`/`format`) says why the VLM's own
  reading was rejected as not text, when it was. `TextCapability` gained
  `choiceField`/`invalidReasonField`; `SlotData.text.choice`/
  `.invalidReason` render next to the plate text on `/review` and
  `CropMetaPanel`.

`regionVocabularyStore` (`GET {API_PREFIX}/regions/vocabulary`) gained
`textChoices`/`textRules` (the served `region_text_choice` id list and
the active profile's region-text validity rules) and label helpers —
`textChoiceLabel`/`invalidReasonLabel` are still a titlecase-id
placeholder (the backend doesn't serve real labels for those two
vocabularies yet). `rejectionReasonLabel`/`rejectionReasonKind` are NOT
placeholders as of OpenProcessor 3f1a11e (openprocessor fix #29): the
endpoint's new `rejection_reasons` list (`{id, label, kind, match,
label_template}`) is resolved exact-match-first, then longest-prefix
(`label_template`'s `{detail}` filled from the rest of the stored
value), falling back to the raw stored value verbatim — never
titlecased — only when nothing in the served vocabulary matches (an
older free-text human reason). `rejectionReasonKind` returns the served
`model_verdict`/`automatic`/`needs_human` kind, or `null` when
unmatched — every rejection-styled surface (`/review`'s inline panel,
`SlotCard`, `CropMetaPanel`) keys its color/wording off this, not off
"a rejection reason is present." Centralizing the lookup here, rather
than inlining the raw id at each call site, is what lets a served label
slot in later without touching `/review`/`SlotCard`/`CropMetaPanel`.

A slot-tab review item's per-item `reason` (the same key the Mismatches
preset's cohort query carries) is generic across every `GET
{API_PREFIX}/review/{tab}` row and, for a region, always reads "verifier
rejected this candidate (…) — needs human review" regardless of the
actual `region_rejection_reason` kind — wrong wording for a
`needs_human` item. `/review`'s `currentSlotRejectionReason` derived
value (keyed off `slotOf(current, activeSlot)?.lifecycle?.rejectionReason`)
takes priority over `current.reason` in the Reason row whenever it's
set; the generic `reason` string still renders as before on any tab/row
without one (core tabs, e.g. the Mismatches preset).

`GET {API_PREFIX}/regions` also gained a `status=` filter (400 on an
unknown value, `GET {API_PREFIX}/regions/statuses` for the vocabulary)
— the `/clusters` region gallery's `SlotGallery` renders it as a
`<select>`, matching the existing Detector filter's served-vocabulary
pattern (`slotGalleryController`'s `statusFilter`).

**New shared components** (renamed off the license-plate-specific names
during the genericization pass — see `docs/genericization-plan-2026-09-13.md`):

- `src/lib/components/ProvenanceChip.svelte` (formerly `DetectorChip.svelte`)
  — color-coded chip. Since the OpenProcessor 1327181 naming sweep (W0,
  finding m9) both the label and the color are backend-driven: the label
  comes from `regionVocabularyStore` (`GET {API_PREFIX}/regions/vocabulary`,
  `$stores/regionVocabulary.svelte`), and the chip COLOR comes from that
  vocabulary entry's served `role` (`detector | segmenter | ocr |
verifier | human | classifier | proposal`) via `paletteForRole`
  (`src/lib/annotations/detectorRegistry.ts` — `detector`=blue,
  `segmenter`=purple, `ocr`=amber, `verifier`=teal, `human`=emerald,
  `classifier`=indigo, `proposal`=sky, unknown/absent role=neutral zinc).
  The old hardcoded per-model-id label/palette tables
  (`builtinDetectors.ts`) are gone; that file now only keeps the
  `mutedTagPattern` (outcome/tag muting is genuine per-deployment logic,
  not a naming table). Accepts a `raw="lpr_nanov11_640:miss"` chain entry
  directly (still parsed client-side); an id not in the served vocabulary
  renders verbatim with the neutral chip. `SlotGallery.svelte`'s
  plate-gallery Detector filter `<select>` similarly renders
  `regionVocabularyStore.filterableDetectors` instead of a hardcoded
  option list.
- `src/lib/components/SlotCard.svelte` (formerly `PlateCard.svelte`) —
  128px annotation-slot thumbnail (via `{API_PREFIX}/crops/{id}/region_thumbnail`),
  parent class chip, score, provenance chip strip, slot text inline (plus
  the "readers disagree" badge above). Parameterized via `readSlot`
  (`src/lib/annotations/readSlot.ts`) against a `SlotSpec`
  (`registeredSlots.ts`) rather than hardcoded plate fields, so a second
  registered slot renders through the same component.
- `src/lib/components/CropMetaPanel.svelte` — the item-detail panel used
  by `CropDetailModal` (`/clusters`, `/clusters/[id]`) and, since
  2026-09-24 (logic-moves W7/W9), embedded behind a collapsed "Details"
  disclosure on `/review` too (same component, not a re-implementation).
  Beyond the per-slot capability summary described above, it lazily
  fetches (on open, keyed to `crop.id`) the label-write history
  (`getCropHistory`) and the source image's metadata + sibling crops
  (`getCropContext`), and renders `item_text_lines` with the optional box
  overlay. Since K6 (`docs/design/k6-frontend-overlay-plan-2026-09-24.md`)
  its "Source image" section also embeds `SourceImageOverlay` (see
  below), passing its own already-fetched context down instead of
  double-fetching.

**New API helpers:**

- `getRegions(browsePath, params)` — `{API_PREFIX}/regions` paginated browse
  with detector/verified/score/status/text filters. The `/regions`
  wrappers all use the `Region` noun (`RegionBrowseItem`,
  `getRegionClusters`, `batchRegionStatus(slot, ids, status)`, …).
- `getTrainingCandidates(mode, params)` — `{API_PREFIX}/regions/training_candidates`
  with 5 cohort modes.
- `getCropHistory(cropId)` — `GET {API_PREFIX}/crops/{id}/history`, the
  item's class-write history oldest-first (`{crop_id, entries}`).
- `getCropContext(cropId)` — `GET {API_PREFIX}/crops/{id}/context`, the
  shared source image's metadata plus every item cropped from it
  (siblings, including the requested crop, mapped through `mapRawCrop`
  like any other crop list). The image itself is served separately by
  `getSourceImageScaled` (`{API_PREFIX}/crops/{id}/image`).

## Client-side source-image overlay (K6, 2026-09-24)

OpenProcessor removed its server-side burn-in of boxes/labels on
`GET {API_PREFIX}/crops/{id}/image` — the endpoint now always serves a
clean image (optionally `?max_dim=N`). Cropwright draws every box/label
itself from `getCropContext`'s `items`, via
`src/lib/components/SourceImageOverlay.svelte`
(`docs/design/k6-frontend-overlay-plan-2026-09-24.md`):

- Each item's own box (`bbox_norm`, already source-image-normalized)
  renders labelled (emerald), proposed (amber, "`<name>` (proposed)"),
  or unlabeled (zinc) — colors/label text only, no slot involved.
- For an item carrying evidence for a registered slot
  (`subBoxSlotFor`/`slotOf`, same mechanism `CropMetaPanel`/`SlotCard`
  use), the slot's region box renders solid in the slot's own
  `capabilities.subBox.ring.confirmed` color, and a verify-rejected
  candidate box (dq-region) renders dashed in `ring.proposed` —
  projected out of the parent-local frame `readSlot` always produces
  (`SlotData.subBox.parent`/`.candidate.parent`) back into the source
  image's absolute frame via `projectFromParent(box, itemSourceXyxy,
'source')` (`readSlot.ts`) — no new wire field, no new projection math.
- The requested crop renders with a thicker ring; every sibling is
  dimmed and, when the caller passes `onselect`, clickable.
- A "hide/show boxes" toggle removes the overlay layer so the raw image
  can be inspected; hover shows a native tooltip (label/score).
- Fetches `getCropContext` itself (a small module-level cache keyed by
  crop id, so the cluster modal / lightbox / review panel don't each
  refetch the same crop's context within a session) unless a caller
  that already has the response (`CropMetaPanel`) passes it via the
  `context` prop.

Wired into every full-source-image surface: `/review`'s source panel,
`CropDetailModal` (`/clusters`, `/clusters/[id]`), `CropCard`'s expanded
lightbox, and `CropMetaPanel`'s "Source image" section. `api.ts`'s old
`getSourceImageWithBbox` (server-overlay-era name, with a cache-busting
`cacheKey` param the burn-in needed and the client-drawn overlay
doesn't) is retired; `getSourceImageScaled(cropId, maxDim)` is its
replacement — same downscaled-image URL, honest naming.
`SlotBboxEditor`/`BboxCanvas` are unaffected (they only ever draw the
crop's own thumbnail via `getThumbUrl`, never the full source image, so
never depended on the burn-in).

## Served region profile (OpenProcessor naming-w2, 2026-09-25)

The backend serves at most one region profile, on
`GET {API_PREFIX}/health` (and `GET {API_PREFIX}/regions/vocabulary`) as
`region_profile: {name, display_name, display_name_singular,
region_class_name, text_reader, reads_text, text_hint_enabled} | null`
(`display_name_singular` added 2026-09-25, OpenProcessor #36 item 10;
`reads_text`/`text_hint_enabled` added 2026-09-26, OpenProcessor W1
"text-free region mode", 5cbd7ee4 — see below). **It is the only
gate for region features.** With `null`, every
region route (`/regions`, `/crops/{id}/region*`, region undo, the VLM
verify routes, `/regions/clusters`) answers 409, so the UI renders no
region surface at all and calls none of them.

- **Text-free profiles (OpenProcessor W1, 2026-09-26).** A profile can
  detect/segment a region without ever reading text off it —
  `text_reader: 'none'`, `reads_text: false`. Whether the synthesized
  region slot gets a text capability (`SlotCard`'s text value,
  `/review`'s inline text row/edit, `CropMetaPanel`'s text section, the
  region browse text filter) is gated on the served `reads_text` alone
  (`servedRegionSlot.ts`; `ServedRegionProfile.reads_text` is required).
  On a text-free profile,
  `/regions/vocabulary` also serves `text_rules: null`/
  `text_choices: []` and drops the `ocr` actor and (when the profile is
  segmenter-only) the `detector` entry — `regionVocabularyStore` already
  renders both as empty/absent cleanly. A `region_meta` PATCH carrying
  `region_text` against a text-free profile 422s
  `{"detail":{"error":"region_text_disabled"}}`, surfaced through the
  existing generic `ApiError`/toast path (no special-casing needed — the
  detail's `error` string already becomes the toast text).

- `regionProfileStore` / `loadRegionProfile()`
  (`src/lib/stores/regionProfile.svelte.ts`) reads the prefixed
  `{API_PREFIX}/health` once in the root layout's `load()` (bounded at
  2 s per try, 3 tries with a short backoff, alongside the tier-2 load).
  Only a successful read seeds: a served `region_profile: null` is "not
  configured", but a timeout or
  network error leaves the store `unknown` (F-78) — still no region route
  is called, `/review` shows "loading region profile…", and the first
  successful `healthStore` poll seeds it and installs the slot. The root
  layout keys its content on `regionProfileStore.seedVersion`, so that
  late seed re-mounts the page with the region tab, no reload. Only a
  successful read that differs from an earlier successful one (or a
  region route's 409 after seeding) raises the one sticky "reload to
  apply" notice. Tabs are never hot-swapped otherwise. A `?tab=` that
  resolves to no tab falls back to All with a visible notice
  (`unavailableTabMessage`, `reviewTabs.ts`), naming the missing region
  profile for the region tab id.
- `regionSlotFromServedProfile()` (`src/lib/annotations/servedRegionSlot.ts`)
  builds the region `SlotSpec`: `key` = profile `name`, `bind.className`
  = `region_class_name` (falls back to `name`), tab label / plural noun /
  stats panel title = `display_name` (falls back to "Regions"), singular
  label/title = `display_name_singular` (falls back to "region"/"Region"
  — item 10, 2026-09-25), used everywhere a singular-context string is
  built generically ("Confirm `<Region>`", "`<Region>` score", "Edit
  `<region>`") so a deployment reads "Confirm Plate" rather than the
  generic "Confirm Region", `?tab=` id and review endpoint = `regions`, browse path `/regions`,
  text capability only when the profile serves `reads_text: true`, no client
  cohorts (served by `/training_cohorts`), and a `single_class` dataset
  export whose `profileName`/`datasetKind` is the profile `name` (so the
  export's output root, e.g. `/exports/license_plate/`, is unchanged).
  **This file is the only place region wire names live**
  (`REGION_WIRE_CAPABILITIES`, `REGION_ENDPOINTS`).
- With no profile: no region review tab, no `/clusters` region gallery
  or pinned inventory card, no `CropCard` ✎, no region section in
  `CropMetaPanel`, no detections panel on `/dashboard`, no region drain
  panel on `/ingest`, no region keymap letters in the reserved hotkey
  set, and the root layout skips `regionStatusesStore`/
  `regionVocabularyStore`. Proved by the no-profile unit tests
  (`registeredSlots.noProfile.test.ts` and the no-profile cases in
  `CropCard`/`CropMetaPanel.visualAudit`/`DatasetStats`/`reviewTabs`/
  `classHotkey`/`classPicker` tests) and `e2e/stubbed/
test_no_region_profile.py` (every route mounts with zero requests to
  a region route).
- A region route's 409 `no_active_profile` (`{detail: {error, message}}`)
  becomes `RegionProfileUnavailableError` in `apiFetch`
  (`src/lib/regionProfileUnavailable.ts`); `toastStore` drops any toast
  carrying its message, so a call site's generic "X failed: …" toast
  never shows it raw.

## Deployment annotation profiles (tier 2, 2026-09-20)

A deployment configures its own annotation slots by placing
`annotation-profiles.json` next to the built `index.html` — either in
`static/` before a build, or bind-mounted over
`/usr/share/nginx/html/annotation-profiles.json` in the running
container. It is fetched once, in the root layout's `load()`, parsed by
`src/lib/annotations/config/parseSlotConfig.ts`, and merged over the
built-in and served slots per-key REPLACE. **Absent or malformed ⇒ no
deployment slots, with a console warning and one toast — never a
crash.** The file is untrusted operator input: template paths, Tailwind
ring classes, regex flags and hotkeys all validate against closed
allow-lists in `src/lib/annotations/config/allowLists.ts`. See
`static/annotation-profiles.example.json` (customizes a served
`pallet_label` profile) and `examples/annotation-profiles/` (license
plate, aircraft tail number, defect code; `examples/README.md`) for
worked examples, `docs/annotation-slots-contract-draft.md` §4 for the
schema, and `docs/design/tier2-annotation-profile-config-plan-2026-09-20.md`
for the full design.

**Region-profile rule** (`applyRegionProfileRule`,
`registeredSlots.ts`): a tier-2 slot that declares any region route
(browse path, endpoints, thumbnail or cohort path under `/regions` or
`/crops/{id}/region*`) is kept only when its `key` equals the served
profile's `name`, in which case it replaces the synthesized slot
wholesale (how a deployment restores domain copy, a keymap or
hand-declared cohorts — e.g. mount `examples/annotation-profiles/
license-plate.json` on a backend running the `license_plate` profile).
Any other region slot, and every region slot when the backend has no
profile, is dropped with a warning. A slot that touches no region route
is kept either way.

`registeredSlots.ts` composes, in order, `builtinSlots` (tier 1, empty),
the served region slot, and the tier-2 slots, through
`resolveSlotRegistry`'s merge-by-replace. It still exports the live
bindings (`registeredSlots`, `slotRegistry`, `slotRegistryWarnings`)
every consumer reads.

## Domain-neutral source (2026-09-24, completed 2026-09-25)

Cropwright must work for any data domain
(`docs/design/domain-neutral-audit-2026-09-24.md`). All 11 steps of that
audit are done; its status section records the deviations.

- **Naming:** `api.ts` wrappers of `/regions` endpoints use `Region`
  (`getRegions`, `RegionBrowseItem`, `batchRegionStatus`); the view layer
  uses `Slot` or no prefix (`SlotGallery`, `slotGalleryController`,
  `pager`/`sel`/`statusFilter` inside it). No rename ever crosses the
  wire — paths and JSON keys are the backend's.
- **No profile in product code.** `src/lib/annotations/profiles/` holds
  only `builtinDetectors.ts`; domain profiles live under `examples/`.
  `createSlotGalleryController(slot)` reads browse path, lifecycle
  states and slot key from its `SlotSpec` (served `/regions/statuses`
  first). The permanent false-positive bucket is identified solely by the
  served `cluster_kind === 'false_positive'` on the selected region
  cluster's own card (`GET {API_PREFIX}/regions/clusters`) — never by a
  client-side id constant; a selected cluster whose id happens to collide
  with a past reserved id is unaffected. `CropCard` picks its sub-box slot by the
  crop's region evidence (`subBoxSlotFor`, `cropSlots.ts`) and renders ✎
  only when there is one. `SlotCard` / `SlotBboxEditor` require their
  `slot` prop.
- **UI copy** names the region by `slot.label` (singular/plural/title),
  the served tab label, or `datasetExportSpec.blurb` — never a fixed noun.
- **Tests use the neutral fixture domain:** widgets with a tag region
  (`TAG-001`, `tag_detector_v1`). `src/lib/test/fixtures/regionSlot.ts`
  exports `WIDGET_TAG_PROFILE` (the served profile), `widgetTagServedSlot`
  (the production synthesis, untouched) and `widgetTagSlot` (that slot
  with tier-2-style domain copy and cohorts). Tests that need the live
  registry call `installServedRegionProfile(WIDGET_TAG_PROFILE)` /
  `resetDeploymentSlots()`. The aircraft and defect demo slots are test
  fixtures (`src/lib/test/fixtures/*Slot.ts`); `roundTrip.test.ts` pins
  the aircraft one to its `examples/` JSON, and `exampleProfile.test.ts`
  parses every `examples/annotation-profiles/*.json`
  (`src/lib/test/fixtures/exampleProfiles.ts`). e2e: `conftest.py`
  serves `REGION_PROFILE` (widget_tag) on `/health` by default;
  `e2e/fixtures/wire.py` derives `REGION_CLASS`/`REGION_TAB_URL_ID`
  (`regions`)/`REGION_TAB_LABEL` from it. The live tier reads the class
  and label from the live `/health` (`live_region_profile`).
- **Ratchet:** `src/lib/domainNeutral.scan.test.ts` fails on any plate
  noun, `LPR`/`lpr_`, or car/
  vehicle-domain word (`vehicle`, `sedan`, `suv(s)`, `motorcycle`, `bmw`,
  `audi`, `brand_b`, `porsche`, `subaru`, `pickup`, `coupe`, `sidecar`,
  `classic_car`, `sports_car`, `dumptruck` — the audit §8 sweep,
  2026-09-25) anywhere under `src/` (code, strings, comments, tests). Its
  allow-list is this file, `profiles.falsification.test.ts` (the one test
  that exercises `examples/`), and `test/fixtures/trainRun.ts` (a real
  served fixture captured verbatim from a live train run — its class
  names are that deployment's own registry, not domain fiction). Fix the
  file; don't widen the allow-list. `Gemma`/`SAM3` are deliberately NOT
  scanned — nearly every remaining occurrence is a served model/detector
  id or its served vocabulary label (approved content per owner
  direction), and a bare-word scan would flag those fixtures for no real
  signal; the audit doc's §8 entry has the full reasoning.
- **Private-origin leak gate (F10, 2026-09-25).** The company name in
  every spelling, the retired `/curation` prefix and `op_` names,
  `openprocessor`, private host paths/IPs, sibling project names, private
  dataset numbers/class names and personal emails are NOT in
  `domainNeutral.scan.test.ts` (that file ships publicly, so it can't
  spell them out). They live in the private-only
  `scripts/oss-export/leak-patterns.txt`, enforced by
  `scripts/oss-export/leak-scan.sh --tree .` over every file a public
  export would ship (tracked files minus `exclude.txt`, with `overlay/`
  in place of the files it replaces) — CI's `export-leak-gate` job. See
  "Public export" below.

## Documentation site (`docs-site/`)

A standalone Docusaurus 3 site, not part of the SvelteKit app — its own
`package.json`/`node_modules`/build, excluded from this project's root
`npm run lint`/`check`/`test` (see `.prettierignore`, `eslint.config.js`).
Build/serve it from inside that directory:

```bash
cd docs-site
npm install
npm start      # http://localhost:3000/cropwright/
npm run build  # -> build/, fails on any broken link/anchor
```

- **`docs-site/site.config.ts`** is the single place every
  project-specific value lives (title, repo, URLs, nav/footer links,
  sibling cross-links) — `docusaurus.config.ts` and every landing-page/
  roadmap/architecture component read from it, never hardcode copy, so
  the whole directory is designed to be cloned for a sibling project
  (see `docs-site/TEMPLATE.md` for the exact clone checklist and
  build-time assumptions table).
- Content lives in `docs-site/docs/**` (getting-started, user-guide,
  configuration, operations, developer-guide, faq) and
  `docs-site/src/data/*.json` (`features.json`, `workflow.json`,
  `screenshots.json`, `roadmap.json`, `architecture-diagrams.json` — the
  `/architecture` page's tab/group metadata, not diagram content).
- **Architecture diagrams** are hand-authored [Archify](https://github.com/tt-a1i/archify)
  specs under `docs-site/architecture-diagrams/specs/*.json`
  (architecture/workflow/sequence types), built from real repo evidence
  (`src/routes/`, `src/lib/api.ts`, the controllers, the slot registry,
  `nginx.conf`, `docker-compose.yml`) — not Mermaid, and not app code.
  `scripts/generate-architecture-diagrams.sh` validates each spec at
  showcase quality and renders it to `docs-site/static/architecture/
<name>.html`, embedded as an iframe by `docs-site/src/pages/
architecture.tsx` (tabbed: System / Workflows / Sequences, per
  `docs-site/src/data/architecture-diagrams.json`). See
  `docs-site/architecture-diagrams/README.md` for the diagram list and
  regeneration instructions. Treat a stale diagram like any other doc:
  edit the spec and re-run the generator when the code it describes
  changes — never hand-edit the rendered HTML.
- **Screenshots are never captured from the shared dev stack** — only
  from a Cropwright instance pointed at a public-sample-data
  OpenProcessor backend (COCO val2017 / Open Images plates). The 1600px
  captures are committed under `docs-site/static/img/screenshots/`
  (image credits on the screenshots page); a `<Screenshot>` slot with no
  file renders a "pending" placeholder. `scripts/capture_docs_screenshots.py`
  requires an explicit `--base-url`, reads its routes from
  `docs-site/src/data/screenshot_routes.json`, and aborts every request
  except GET/HEAD and the side-effect-free `/train/preflight`. See
  `docs-site/docs/developer-guide/screenshots.md`.
- Hosted locally as the `cropwright-docs` container on :5185
  (`docker build -t cropwright-docs:local docs-site && docker run -d
--name cropwright-docs -p 5185:8080 cropwright-docs:local`), served
  under the same `/cropwright/` base path GitHub Pages uses.
- Deploys to GitHub Pages via `.github/workflows/docs.yml` (build on
  every PR touching `docs-site/**`, deploy on push to `main`/`master`) —
  won't actually publish until Pages/Actions are enabled on the public
  repo.

## Development

```bash
npm install
npm run dev    # http://localhost:5173 (uses Vite default)
npm run check  # svelte-check + tsc
npm run build  # SvelteKit → /build (static)
```

### Component tests + the `/review` controller

`vite.config.ts` sets `resolve: process.env.VITEST ? { conditions:
['browser'] } : undefined`, which makes vitest resolve Svelte 5's real
browser build under jsdom (no new dependency). This means a `.test.ts`
can mount a component directly:

```ts
import { mount, unmount, flushSync } from 'svelte';
import MyComponent from './MyComponent.svelte';

const target = document.createElement('div');
document.body.appendChild(target);
const instance = mount(MyComponent, { target, props: {...} });
flushSync();
// assert on target.textContent / target.querySelector(...)
unmount(instance);
```

Prefer this over a source-text regex scan whenever the behavior is
actually renderable — `CropCard.test.ts` / `DatasetStats.test.ts` /
`SlotCard.test.ts` / `TrainForm.gpuPicker.test.ts` /
`StrategyBar.appliedSort.test.ts` are the worked examples
(docs/design/test-audit-2026-09-24.md recommendation 7). A source scan
is still the right tool for something a mount can't reach — absence of
dead code, a call site's exact wiring the test would otherwise have to
drive a full user flow to observe — see `TrainForm.test.ts`'s remaining
`it()`s, each with a one-line reason comment, and
`$lib/testing/sourceScan.ts`'s `normalize`/`extractFunction`/
`extractBalanced` helpers, which keep a scan robust to reformatting
(no `\n {2}\}`-anchored regex that breaks on a prettier re-wrap).

`/review`'s queue actions (assign/discard/skip/undo) live in
`src/lib/review/reviewController.svelte.ts`, not inline in
`+page.svelte` — the page still owns `queue`/`cursor`/`handledIds`
(shared with slot-tab actions and arrow-key nav) and hands them to the
controller by reference/accessor, following the existing
`slotGalleryController.svelte.ts` factory-function convention. Test
new queue-action behavior against the controller directly
(`reviewController.test.ts`), not by mounting the whole page.

`/clusters/[id]`'s equivalent actions (assign / drop-on-class / accept-
reject-VLM / accept-all-VLM-on-page / discard / undo / ignore-unignore /
move) live in `src/lib/clusters/clusterController.svelte.ts`, same
convention — the page owns `cropPager`/`sel`/`dragIds`/the grid's
drag-local override and hands them to the controller by
reference/accessor. The stale-fetch-race guard (`excludedCropIds`) is
its own factory, `createExclusionGuard()`, exported alongside the
controller: the page creates it first and wires `cropPager`'s `accept`
option to it before the rest of the controller exists (which itself
needs `cropPager` as an input) — this sidesteps a forward-declared
`let controller` that `svelte-check` would flag as a non-reactive
`$state` update. Test new cluster-action behavior against the
controller directly (`clusterController.test.ts`).

Every new test in either category should be verified to fail against a
mutated copy of the code it covers (edit a scratch copy, confirm red,
restore byte-for-byte — never `git checkout`/`stash`/`restore`) before
being trusted; `npm run test:mutation -- --mutate <file>` does this at
scale for a file already in `stryker.config.json`'s `mutate` list.

### End-to-end tests (`e2e/`)

`npm run test:e2e` — creates/reuses a per-project venv at `e2e/.venv/`
(never installed on the host), installs `e2e/requirements.txt` and a
chromium browser if missing (`scripts/run-e2e.mjs`), then runs
`e2e/stubbed/` (`npm run build` + `vite preview`, driven by pytest +
Playwright). The runner builds and serves once, then runs the suite in
parallel with pytest-xdist (`--dist loadfile`; workers default to half the
CPUs, capped at 6; override with `E2E_WORKERS`), which takes about 40 s
instead of about 160 s. Tests must wait on real conditions such as a
selector or `page.expect_request`, not fixed sleeps, because a sleep that
works serially flakes under parallel load. Covers the flows a plain `npm test` (jsdom, no real
browser) can't: keyboard-driven `/review` assign/undo, `/clusters/[id]`
drag/hotkey/discard, the `/settings` and dashboard assist-scope
wire-composition round trips, tier-2 annotation-profile loading, and
that every top-level route still mounts. Never runs against a live
OpenProcessor backend — `e2e/conftest.py`'s `Stub` fixture routes every
`{API_PREFIX}` request itself.

The runner and the `app_url` fixture start `vite preview` in its own
process group (`scripts/lib/previewServer.mjs`, `start_new_session` in
conftest) and stop the whole group (SIGTERM, then SIGKILL), on normal exit,
test failure, SIGINT/SIGTERM/SIGHUP and uncaught errors, so no server is
left running; killing only an `npx` wrapper used to orphan one per run.
`src/lib/testing/previewServer.test.ts` pins the group stop.

**Fail-closed, on purpose** (see docs/design/test-audit-2026-09-24.md
recommendation 5 — this replaced the old `scripts/playwright_*.py`
runbooks, which drifted to a stale `/curation/**` prefix for weeks because
their catch-all stub silently answered `200 {}`): a request under
`{API_PREFIX}` that no test registered gets `501` and is recorded in
`stub.unhandled`; every test asserts at teardown that it is empty and
that at least one stub actually fired. (The separate retired-prefix
intercept, `stub.op_hits`, was removed in F10: the fail-closed 501 is
the real guard.) An API-prefix
change or a route rename fails the test outright instead of the page
silently rendering empty. CI runs this in the `e2e-stubbed` job on every
push/PR; a `py_compile` pre-commit hook (and CI step) gates
`scripts/*.py` and `e2e/**/*.py` syntax separately from this suite.

**The app's own files are loaded by the harness, not by Chromium.** The
`stub` fixture routes every app-origin request through `route.fetch()`
(`serve_app_through_harness`). Chromium fails requests queued for a socket
with `net::ERR_NETWORK_CHANGED` whenever the host's network changes (any
docker container starting or stopping), which used to break the SPA's
chunk loading at boot and fail a random test on its first element. Keep
the routing; `test_harness_serves_app.py` fails without it, and
`scripts/repro_chromium_network_change.py` reproduces the abort (it
disturbs other browsers on the host, so run it only when idle). A failed
test writes a screenshot, console, request log with timings, the stub's
handled/unhandled lists and the DOM to `artifacts_local/e2e-failures/
<test id>/`; read those first when a test flakes.

#### Live read-only tier (`e2e/live/`)

`npm run test:live` — same `e2e/.venv`/chromium setup as `npm run
test:e2e` (`scripts/run-e2e.mjs e2e/live`, sharing the venv-provisioning
logic rather than a second copy of it), but drives the **real, currently
deployed** build against a **real** OpenProcessor backend instead of the
stubbed in-browser fixtures — the one class of bug `e2e/stubbed/`
structurally cannot see: the frontend and the live backend's own wire
shape/data actually disagreeing. Point it at the deployment with
`CROPWRIGHT_LIVE_URL=http://localhost:5184 npm run test:live` (or any
other reachable Cropwright nginx origin) — unset, `_require_live_url`
(`e2e/live/conftest.py`, session-scoped + autouse) skips every test in
the tier before a browser is ever launched, so this never runs in CI or
under plain `npm run test:e2e`. A session-scoped `live_url` fixture
preflights the GLOBAL `GET {API_PREFIX}/health` (never a project-scoped
one — health and `/projects` are the only routes that stay unscoped
post-cutover, see "Projects — /p/[project] routes" above), skipping with
a clear message on anything short of a clean 200.

**Project-aware (2026-09-27, following the projects cutover).** A
session-scoped `live_project` fixture reads the GLOBAL `GET
{API_PREFIX}/projects` once, resolves the served `default_slug` against
that response's `projects` list, and exposes `{slug, prefix}` — every
direct API read in this tier goes through `api_get(live_url,
live_project, path)` (built from the served `prefix`, never assembled
client-side), and every page navigation goes through
`page_path(live_project, path)` (`/p/<slug><path>`). `live_region_profile`
now reads the region profile from the default project's own scoped
`{prefix}/health` (per-project, matching
`src/lib/stores/regionProfile.svelte.ts`), not the global one. The
write guard's `_READ_ONLY_POST_SUFFIXES` matches `/train/preflight` by
path suffix (rather than a fixed absolute path) since the scoped prefix
varies per project.

**Hard read-only, structurally, not by test discipline.** Every test
uses the `guarded_page` fixture: it routes every `**/curation/**`
request through a handler that lets GET/HEAD through untouched and
`route.abort()`s anything else, recording the attempt — and its
teardown asserts the recorded list is empty. A test that merely
_attempts_ a write (even one the guard successfully blocked) fails; see
the fixture's own docstring for the disposable proof-of-guard runbook
(a scratch test firing a `fetch(..., {method: 'POST'})` via
`page.evaluate`, run once to confirm both the abort and the teardown
failure, then deleted — never a permanent test). The same fixture also
fails a test on any `pageerror`, launches chromium with
`args=["--disable-gpu"]` (inherited from `e2e/conftest.py`'s
`browser_type_launch_args` — it hangs without this flag on this
machine, see that fixture's docstring), sets an explicit 1280×720
viewport, and explicit navigation/action timeouts. Tests use
`wait_until="domcontentloaded"` plus a concrete `wait_for_selector`/
`wait_for_function`, never `wait_until="load"` or a fixed sleep.
A failing test's own screenshot is captured via pytest-playwright's
`--screenshot=only-on-failure --output=...` flags (`scripts/run-e2e.mjs`
when the target is `e2e/live`), landing directly under
`artifacts_local/cw-live/live-tier/` (gitignored). pytest-playwright
wipes that directory at the start of every session, so it only ever
holds the latest run's failure shots; the always-on review screenshots
below live in a sibling directory for exactly that reason.

**Full-page route screenshots (2026-09-24 visual-review follow-up),
always, not just on failure.** `test_route_sweep.py` additionally saves
one full-page PNG per route at each of two viewports — desktop
(1600×1000) and narrow (800×1000) — under
`artifacts_local/cw-live/live-shots/<run-timestamp>/<route-slug>-
<width>.png` (`screenshot_run_dir`, a session-scoped fixture in
`e2e/live/conftest.py` that timestamps one directory per test-session
run). **These screenshots are not self-checking — a human (or an agent
acting on the user's behalf) MUST actually open a representative sample
of them with an image-reading tool after every live-tier run and look at
them.** The sweep's assertions (no `pageerror`, no bad response, no
`NaN`/`undefined`, no broken image, no narrow-viewport overflow) catch
what's mechanically checkable; they do not catch a confusion-matrix image
rendering at full panel width, a wrapped nav link, a clipped status chip,
or any other layout regression that only shows up to the eye. Treat a
live-tier run that skipped this review step as incomplete.

While at the narrow (800px) viewport, the sweep also asserts
`document.documentElement.scrollWidth <= window.innerWidth + 1` — no
horizontal page overflow. (This caught and drove the fix for the top
nav wrapping "Bake-off" onto two lines and clipping the "API OK" chip
past the viewport edge at ≤800px — `src/routes/p/[project]/+layout.svelte`'s primary
nav is now its own horizontally-scrolling strip, `overflow-x-auto
whitespace-nowrap`, with every link `shrink-0` and the status chip
pinned `shrink-0` so it's never squeezed.)

Three test modules, 26 tests total against this deployment's live
dataset:

- **`test_route_sweep.py`** — every project-scoped top-level route mounts
  under `/p/<slug>/...` (`live_project`'s resolved default-project slug):
  `/dashboard`,
  `/ingest`, `/clusters` (plain and `?class=<served region_class_name>`), `/review`
  with each tab's `?tab=<urlId>` (`all`/`uncertainty`/
  `model_disagreements`/`classifier_blind_spots`/`new_class_proposals`/
  `regions`), `/classes`, `/export`, `/train`, `/models`, `/bakeoff`,
  `/settings` — plus the one GLOBAL page, `/projects` (never under
  `/p/<slug>/...`), swept by its own
  `test_global_route_mounts_cleanly`. `/ingest` fails against any backend that predates
  `docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md` (this
  currently deployed build 404s it) — that's expected until the next
  deploy, not a bug. Each asserts:
  no `pageerror`; no `**/curation/**` response >= 400 outside a small,
  explicit, documented allow-list (`ALLOWED_4XX_5XX` in
  `e2e/live/conftest.py` — one entry today, `GET .../keymap` 404, for a
  deployment that predates OpenProcessor W2b, see "Keyboard shortcuts"
  above; every other endpoint this deployment serves on mount came back
  200/204 in manual verification);
  no literal `"NaN"`/`"undefined"` in the rendered body text; every
  `<img>` whose bounding box intersects the 1280×720 viewport finishes
  loading (`naturalWidth > 0`) — an offscreen lazy image is allowed to
  still be pending; no horizontal overflow at the narrow 800px viewport
  (see above). Also saves the always-on desktop/narrow screenshots
  described above.
- **`test_data_agreement.py`** — the UI shows what the API serves, every
  direct read scoped to the default project's own served `prefix` via
  `api_get(live_url, live_project, path)`:
  the dashboard's "Clusters (total now)" vs `GET {prefix}/stats/
dataset` `clusters.cluster_count`; `/review?tab=regions`'s queue-counter
  total vs `GET {prefix}/review/regions` `total`, both unfiltered
  and with `?region_status=verify_rejected`; every `filter_specs` entry
  `GET {prefix}/review/tabs` serves for the `regions` tab renders a
  `<select>` with exactly the served option labels; a served
  `rejection_reasons` label (`GET {prefix}/regions/vocabulary`,
  resolved exact-then-longest-prefix, mirroring
  `regionVocabularyStore.rejectionReasonLabel`) actually renders for a
  live `verify_rejected` item — skipped, not failed, when the live
  cohort is momentarily empty. A concurrently-writing actor (another
  agent's UI-write smoke test against the same deployment, expected per
  this tier's own operating assumption) can move a count between the
  API read and the UI read: `agrees_with_retry` (conftest.py) re-reads
  the API once on a mismatch and accepts either value, and
  `wait_for_stable_text` polls a value until it stops changing (the
  queue counter briefly renders an unfiltered total before a
  URL-seeded `?region_status=` filter's `filter_specs` finish loading —
  see "Served per-tab filters" above) rather than racing a single read.
  Also: `test_ingest_status_agrees` (the `/ingest` status table's total
  vs `GET {prefix}/ingest/status`) and `test_region_drain_agrees`
  (the region-drain panel's `total_unfinished` vs
  `GET {prefix}/ingest/region_drain`, via `wait_for_stable_text`
  since the value moves during a live cascade) — both fail against a
  pre-ingest backend the same way `test_route_sweep.py`'s `/ingest`
  case does.
- **`test_deep_link.py`** — takes the first crop id off `GET
{prefix}/review/regions?region_status=verify_rejected&page_size=1`
  (skips if none), opens `/p/<slug>/review?tab=regions&region_status=
verify_rejected&crop_id=<id>`, and asserts it actually lands: "Locating
  crop…" clears, the queue counter reports a real `#position · N loaded`
  position (the served queue position, F8 D6), and no "not in this review queue" toast appears.

### Mutation testing

`npm run test:mutation` (Stryker, `stryker.config.json`) runs mutation
testing against 10 pure, high-value modules — `api.ts` (data mapping +
wire params), `stores/undo.svelte.ts`, `datasetStats.ts`,
`autoLabelRunVlm.ts`, `sourceBadge.ts`, `reviewTabs.ts`,
`curationSettings.ts`, `strategies.ts`, `classPicker.ts`,
`annotations/readSlot.ts` — the files `docs/design/
test-audit-2026-09-24.md` flagged as most exposed to "the suite passes
but doesn't actually test the behavior." It answers a different
question than `npm test`: not "does every assertion pass" but "if I
break this line on purpose, does some test actually notice." Takes
about 15-20 minutes locally; not run in pre-commit or the per-push CI
`check` job (too slow) — it runs on a schedule and `workflow_dispatch`
via `.github/workflows/mutation.yml`. `thresholds.break` in the config
ratchets up only when a real pass raises the score; don't lower it to
make a red run go green. Run it locally after touching wire-mapping
logic in `api.ts` or store logic in `stores/*.svelte.ts`, or whenever
CI's scheduled run goes red, to see the exact surviving mutants
(`reports/mutation/index.html`).

The production build runs in a non-root
`nginxinc/nginx-unprivileged:1.31.2-alpine3.23` container (uid 101, nginx on
container port 8080; F10 D13), host port 5184 (`CROPWRIGHT_PORT`, mapped
to 8080). `docker-compose.yml` is pull-only (`image:
davidamacey/cropwright:${CROPWRIGHT_TAG:-latest}`, no `build:` —
needs no repo checkout, see README's "Quick start"); building from
source is the `docker-compose.build.yml` overlay (`docker compose -f
docker-compose.yml -f docker-compose.build.yml up -d --build`, tagged
`cropwright-dev:local`, never `docker-compose.override.yml` — that
auto-loads and would make a plain clone silently build instead of
pull). **The currently deployed `cropwright` container on this host was
started from a checkout with the old `build: .`-in-`docker-compose.yml`
shape and keeps running unmodified** — its next
`docker compose up -d --build` must use the override explicitly
(`docker compose -f docker-compose.yml -f docker-compose.build.yml up -d
--build`) now that the base file is pull-only, since `latest` isn't
published on Docker Hub yet and a bare `docker compose up -d --build`
against the new base file has no `build:` to run. Port conflicts:
5174=example-app-backend, 5180/5181=example-app-opensearch,
5183=example-app-docs.

## Connecting to OpenProcessor

The backend is OpenProcessor's public `main` (the pre-cutover private stack
serving `{API_PREFIX}/*` is retired). Three runtime env vars, all substituted at
container start by `docker-entrypoint.sh`, so one image fits any deployment:

- `PUBLIC_API_PREFIX` (default `/curation`) — must equal the API's
  `OP_API_PREFIX`; the API builds some URLs (region thumbnails) from its
  own prefix, and nginx proxies only this one prefix.
- `API_UPSTREAM` (default `http://op-api:8000`) — where nginx proxies
  `{PUBLIC_API_PREFIX}/*`, by container name over the API's docker network
  (`OP_DOCKER_NETWORK`, default `openprocessor_triton_net`). No CORS, works
  from any LAN client.
- `PUBLIC_TRITON_API_URL` (default empty = relative URLs via the proxy) —
  set only when the browser must call an API on a different origin.

All API calls flow through `src/lib/api.ts` with retry + AbortController
for in-flight cancellation. `apiFetch` retries a 5xx up to 3 times
(250/500/1000ms backoff); since OpenProcessor 3cd4ca87, a 503 carrying a
`Retry-After` header (seconds — the shape the backend's Triton-outage
handler sends, `Retry-After: 5`, in place of the old bare 500/silent-200
on an inference-backend outage) replaces that attempt's fixed delay
instead, clamped to `MAX_RETRY_AFTER_MS` (5s) so it can't stall the UI
past the existing retry budget or add an extra attempt. Every caller
still just sees the eventual `ApiError` with the served `detail` string.

### Projects — `/p/[project]` routes, switcher, `/projects` (2026-09-26)

**OWNER DECISION: a fresh build, no backward compatibility.** The
backend's projects cutover removes the old unscoped `{API_PREFIX}/...`
alias entirely — there is no `default`-prefix fallback anywhere in this
build.

- **GLOBAL routes** (never project-scoped): `{API_PREFIX}/projects` (the
  list and every lifecycle write), the global `{API_PREFIX}/health` and
  the global `{API_PREFIX}/events`.
- **Everything else** lives ONLY under a project's own served `prefix`:
  `{API_PREFIX}/projects/{project}/...`
  (`docs/design/any-domain-rev3-and-projects-contract-review-2026-09-26.md`
  §7).

**The active project lives in the URL path only** (`/p/<slug>/...`,
owner decision) — reconstructable from the URL, so nothing is persisted.

- `projectsStore` (`src/lib/stores/projects.svelte.ts`): the root
  layout's `load()` reads `GET {globalApi()}/projects` once (bounded
  retries). `src/routes/p/[project]/+layout.ts` then `resolve()`s the
  slug — a listed project, or `GET {globalApi()}/projects/{slug}` for one
  the default list doesn't carry (an archived project opened by link) —
  and `select()`s it: `setScopedPrefix(<served prefix>)` (never
  assembled client-side) and, on a change of project, every registered
  reset hook. Only then does it load the scoped region profile and
  keymap. An unknown slug, or one the server marks not `selectable`
  (`building`/`failed`/`deleting`), renders `ProjectUnavailable.svelte`
  ("not found" / "not available", with links to `/projects` and the
  served default project) and fires no scoped call. A failed project-list
  load renders the root layout's full blocking error state
  (`data-testid="projects-blocking-error"`).
- Every scoped call builds its URL through `scoped()` (`api.ts`), which
  throws `ProjectNotSelectedError` until a project is selected (fails
  closed). `globalApi()` is the only way to reach the global routes.
- **Stale responses are dropped.** `apiFetch` remembers the scoped prefix
  a request was built for; a scoped response that lands after the active
  project changed rejects as an `AbortError` (which every caller already
  ignores), so the previous project's data never renders in the next.
  Global calls pass `{ global: true }` and are never dropped.
- **Per-project caches reset themselves.** Each store registers its own
  hook with `onProjectChange()` (`$lib/projectChange`, dependency-free):
  the undo ring buffer, the source-overlay context cache, `classesStore`,
  `healthStore`'s scoped half, `classSourcesStore`,
  `regionProfileStore` (a different project's profile is not a "change":
  no reload notice — the notice still fires for the SAME project),
  `regionStatusesStore`, `regionVocabularyStore`,
  `reviewTabsVocabularyStore`, `strategiesStore`,
  `curationSettingsStore` and the keymap.
  Load-once stores carry a generation counter so a load started for the
  previous project never lands. The shell's content is keyed on the slug
  too: SvelteKit reuses a page instance across `/p/a/review` →
  `/p/b/review`, so page-local state would otherwise survive a switch.
- **Links.** `projectHref(path)` (`$lib/projectPaths`) builds
  `/p/<active slug><path>`; call sites wrap it in SvelteKit's `resolve()`
  (`svelte/no-navigation-without-resolve`). `switchProjectHref()` keeps
  the section and ordinary query params and drops ids that don't carry
  across projects (`/clusters/<id>`, `crop_id`, `class`);
  `legacyRedirectTarget()` maps `/` and the bare old paths.
- **Switcher** (`ProjectSwitcher.svelte`, top bar): the served
  `selectable` projects (plus the active one), each with its
  `display_name`, slug and a served status label (`labels.status`) for
  anything not `active`; a "custom keys" badge when the active project's
  served keymap has `is_default: false`; a "paused" chip when the active
  project's served pipeline-pause flag is true; "Manage projects…" links to
  `/projects`. A project whose served `writable` is false (archived) shows
  a read-only banner under the top bar. The global `project.*` event
  stream re-reads the list.
- **Pipeline pause** (P2, §5.1). `projectPauseStore`
  (`$stores/projectPause.svelte.ts`) holds each project's served
  `GET {prefix}/pause` `paused` by slug; the `/p/[project]` layout reads the
  active project's after `select()` (not awaited), `/projects` takes each
  row's from the list's `paused`, and `PauseProjectDialog.svelte` POSTs `/pause` or
  `/resume`. Every call goes through `projectPrefix(project)` (`api.ts`),
  the project's own served `prefix` — never a slug path. `ProjectSummary.paused`
  is the project's own flag, so `/projects` reads the chip and the
  Pause/Resume buttons off the list (no per-row `GET /pause`); the pause
  state's served `paused_by`/`reason` (a global GPU-training claim shows
  there) render in the switcher tooltip. The global `project.paused`/
  `project.resumed` events (`projectEvents.ts`) update the store and
  re-read the list.
- **`/projects`** — see the Routes table. Wire types in
  `src/lib/types_projects.ts`, pinned key-for-key to the vendored OpenAPI
  by `src/lib/contract/projectsContract.test.ts`: `capacity` is the typed
  `ProjectCapacityWire` (served `shards_after_create`, and `limit_source`:
  `heap` or `cluster_max_shards_per_node`, are carried; the served
  `message` already states both facts, so there is no extra copy), and
  `DeleteDryRunResponse` is pinned too. Only its `referenced_by` rows are
  served as bare objects; `project` and `profile` are the keys read.
  OpenProcessor v0.4.0 (fce17771) additions: an "Embedded" column
  (`counts.items_embedded`, via `formatCount`, "—" when null) and, in
  `DeleteProjectDialog`, the dry run's `referenced_by` listed ("`<project>`
  uses a shared model (profile `<profile>`)"; blocking is still decided by
  the served `blocking` alone), a 409 `in_use` showing the served message
  plus its `used_by` (else `projects`), and a 503 `config_store_unavailable`
  showing the served message with a Retry (re-runs the dry run, or resends
  the same confirm). No force option is offered for the delete. Failed
  lifecycle results carry the served `ProjectErrorDetail` as `detail`.
- **Tests:** `projectPaths.test.ts`, `stores/projects.svelte.test.ts`,
  `stores/projectSwitchResets.test.ts`, `api.projectLifecycle.test.ts`
  (wrappers + stale guard), `projects/projectsAdminController.test.ts`,
  `components/ProjectSwitcher.test.ts`, `routes/projectRouting.test.ts`
  (redirect and slug-resolution loads), `routes/projects/projectsPage.test.ts`
  (mount); e2e `test_project_scoping.py` (prefix boundaries, redirects,
  not-found pages), `test_project_switch.py` (two projects: requests move
  to the other prefix, undo resets), `test_projects_crud.py` and
  `test_projects_pause.py`; `stores/projectPause.svelte.test.ts`.
  `src/lib/test/setup.ts` selects a test project whose prefix is
  `API_PREFIX` (slug `default`) before every unit test.

### Combine projects (P4, 2026-10-01)

`docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md` §4. Merge several
projects into a NEW one: `/projects/combine` (wizard) and
`/projects/combine/[job_id]` (job view), both GLOBAL like `/projects`
(the target does not exist yet). Wrappers `src/lib/api_combine.ts`
(imported directly, never re-exported from `api.ts`), types
`src/lib/types_combine.ts`; the preview/job shapes the contract leaves as
bare objects (`sources`, `target`, `dedup`, `report`, `next_steps`) are
typed as the backend builds them, every field optional.

- **Gate.** `combineAvailability` (`src/lib/combine/`) probes
  `GET {globalApi()}/projects/combine/__probe__` once (no served flag, no
  job list): a 404 whose `detail.error` is `combine_not_found` means the
  router answered (available); a plain 404 or 501 means it is not mounted
  (absent: no "Combine projects…" button, the routes say so, nothing else
  fires). Global, not reset on a project switch. Replace the probe the
  day the backend serves a real signal.
- **Wizard** (`combineWizardController.svelte.ts`). Sources come from the
  served list (`selectable` and `active`), ordered by priority with
  up/down buttons (first wins a conflict and donates a duplicate's image);
  each has a `label_states` select. The preview re-runs 400 ms after any
  change with the previous request aborted. Every option (`dedup`,
  `dedup_iou`, `holdout`, `settings_from`, a source's `label_states`)
  stays out of the body until touched. The mapping step is
  `CombineMappingTable.svelte`, deliberately not the W10 `MappingTable`:
  combine maps into classes the request itself creates, by name (a
  `create` row defines one, a `map` row picks one of the form's own
  `create` names; no registry, no `class_id`). Untouched rows are filled
  from the served `suggested_mapping` after each preview (one re-preview
  follows if that changed the body); a touched row is never overwritten;
  "Reset to suggestions" re-copies. Action labels come from the first
  source's `GET {its prefix}/datasets/formats` `mapping_actions` (raw ids
  when that 404s). Start (confirm listing the served target counts) sends
  the request plus `expected_preview_sha` and is disabled while a preview
  is in flight, the request changed since it was served, or `ok` is
  false. 409 `preview_stale` shows the message and re-previews; 422
  `combine_invalid` shows the message and the served report; 409
  `project_busy` lists the served `jobs`.
- **Job view** (`combineJobController.svelte.ts`). The backend serves no
  `poll_after_s` or actions, so the view re-reads every 2 s while
  `queued`/`running`, stops at any other status, and a global
  `combine.progress` event with this `job_id` wakes an immediate re-read.
  Cancel while queued/running, Resume while interrupted/cancelled
  (confirm-gated, refusal text verbatim). Completed: Open project,
  "Review flagged conflicts" (`/p/<target>/review?tab=all&combine_conflict=true`),
  and one confirm-gated button per served `next_steps` entry, run as served
  against the TARGET project's own prefix (`runCombineNextStep`, body-less).
  The served response stays on the job view as a "Last next step" block
  (`CombineStepResult.svelte`: action label, `status` chip, scalar rows,
  nested/long values collapsed; `CombineJob.lastStep`, per job id) with a
  neutral "Ran <action>" toast; a refusal shows the served detail instead.
  The buttons stay disabled, with "Target project: <served status>" beside
  them (`CombineNextSteps.svelte`), until the controller's own poll of
  `GET /projects/{target}` (`CombineJob.targetStatus`, same 2 s interval,
  stops at any status but `building` or on unmount) reads `active`; a 409
  `project_building` still shows its served message and re-reads the status.
  Failed: the served error and report plus "Undo combine".
- **`/projects`.** A project the server says came from a combine
  (`origin.kind === 'combine'` with a `job_id`) links to its job (how a job
  is found again after a reload) and its delete is labelled "Undo combine"
  (`DeleteProjectDialog`'s `title` prop; the guarded dry run and typed slug
  are unchanged, since undoing a combine is deleting the target).
- **Item provenance.** `provenance/CombineOriginRows.svelte` (mounted in
  `CropMetaPanel`): origin project/item/image/split, an amber "Conflict
  between sources" chip with `combine_conflict_origins`, and
  `combine_merged_origins`, only when `origin_project` is non-null.
- **v0.4.0 preview warnings.** `embedding_model_mismatch` and
  `region_profiles_differ` are served as ordinary preview `warnings` and
  render verbatim through `CombineIssueList` (served message and code; no
  client wording, no code change); the preview stays startable.
- **Recluster step.** The served `next_steps[0]` is
  `POST /cluster/umap/rebuild` (project-relative, OpenProcessor f14f4ddc);
  `endpointCatalog.test.ts`'s `runCombineNextStep` override is anchored to
  that real route and the button runs whatever path is served.
- **Tests.** `api_combine.test.ts`, `contract/combineContract.test.ts`
  (interface keys and strict bodies), `combine/*.test.ts` (availability,
  wizard, job), mount tests under `components/combine/` and
  `provenance/CombineOriginRows.test.ts`, `routes/projects/projectsPage.test.ts`
  (extended); e2e `test_combine_projects.py`.

## API contract

The frontend's picture of the backend's wire format is not hand-copied —
it is vendored from OpenProcessor's own generated contract files and
checked against them by tests, so a backend rename fails a frontend test
instead of silently rendering blanks or 404ing.

- **Vendored snapshot:** `contracts/openprocessor/` (`json/item_wire.json`,
  `ts/{itemWire,classSources,regionStatus}.ts`, `openapi/curation.json`,
  `SOURCE.md` recording the backend commit it came from). Generated on
  OpenProcessor's side (`contracts/README.md` there); copied here
  verbatim, never hand-edited.
- **Sync / check:** `npm run contract:sync` refreshes the snapshot from a
  local OpenProcessor checkout (`OPENPROCESSOR_REPO`, default
  `../OpenProcessor` — on this host set
  `OPENPROCESSOR_REPO=/data/repos/openprocessor`, or the check silently
  skips; `OPENPROCESSOR_REF`, default `main`) via
  `git -C $OPENPROCESSOR_REPO show $OPENPROCESSOR_REF:contracts/...`.
  `SOURCE.md` records the public repo URL (`OPENPROCESSOR_URL`) and the
  sha, never the local checkout path.
  `npm run contract:check` diffs the vendored copy against that ref and
  exits non-zero on drift; it exits 0 with a "skipping" message when the
  backend repo isn't present (CI), so it's harmless there. Wired into
  CI's `check` job and a local pre-commit hook gated on `contracts/` or
  `src/lib/api.ts` changing.
- **Tests read the vendored files**, not a hand-typed copy — under
  `src/lib/contract/`:
  - `wireKeys.test.ts` — `api.ts`'s `RAW_CROP_KEYS` (compile-time
    exhaustive against `RawCrop`) and every built-in slot's `*Field`
    wire references vs the backend's item/region wire keys, with an
    explicit `KNOWN_STALE` allow-list for any field the frontend reads
    that the backend doesn't (currently empty).
  - `regionStatus.test.ts` — slot lifecycle states/human-writable
    statuses vs the backend's `RegionStatus` enum.
  - `classSources.test.ts` — `sourceBadge` handles every backend
    `class_source` role.
  - `endpointCatalog.test.ts` — every `${API_PREFIX}/...` call site
    (mechanically extracted by `src/lib/contract/apiCallScanner.ts` from
    `api.ts`, `sse.ts`, `SlotCard.svelte`, `export/+page.svelte` — not a
    hand-maintained list, and a new file with a call site the scanner
    doesn't cover fails the "no other file references API_PREFIX"
    completeness guard; a scoped route of a specific project, built with
    `${projectPrefix(project)}`, is scanned the same way) resolves to a real path+method in the vendored
    OpenAPI spec, with every mechanically-resolvable query param checked
    against that operation's declared parameters. A handful of
    runtime-composed paths (a slot's own declared `endpoints`/
    `extras.datasetExport` paths) are anchored `MANUAL_OVERRIDES` in the
    test, not silently skipped.

## Chunk-load recovery and new-version handling

A failed module chunk (flaky network, a browser network-change abort, or a
deploy that replaced the hashed chunks under an open tab) used to render a
bare "500 Internal Error" until a manual reload. `src/hooks.client.ts`
(`handleError` plus a `vite:preloadError` listener) detects it
(`isChunkLoadError`, `src/lib/chunkRecovery.ts`) and does one
`location.reload()`, guarded by a timestamp in `sessionStorage`
(`cropwright.chunkReloadAt`, 30 s window; per tab, reconstructible, unreadable
storage fails closed so it can never loop). A repeat inside the window falls
through to `src/routes/+error.svelte`: neutral headline, a Reload button and
the thrown message in a collapsed Details block. `kit.version.pollInterval`
is 60 s, so the `updated` store flips after a deploy: the root layout shows a
small "A new version is available" Reload banner, and SvelteKit itself does a
full-page load on the next failed navigation when `updated` is true.
`nginx.conf` serves `/_app/version.json` with `no-cache` (like `index.html`)
so the poll cannot be answered from the browser cache. Tests:
`chunkRecovery.test.ts`, `e2e/stubbed/test_chunk_recovery.py`.

## Deployment

The API serves all images and crop thumbnails — the labeler does NOT mount
NAS volumes. This avoids volume duplication and keeps NAS path knowledge in
one place. Image URLs look like `{API_PREFIX}/crops/{id}/thumbnail`.

**Security: the curation API has no request authentication (owner
decision, 2026-09-24).** Every write — label, discard, export, freeze,
train, ingest — is reachable by anyone who can reach the nginx origin,
with no login and no token. Never expose Cropwright (or the
OpenProcessor API it proxies) on the public internet; run it only on a
trusted LAN/VPN until the backend adds opt-in auth (tracked as backend
ask BA-5 in `docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md`).

**Releasing** is local (`./scripts/release.sh`), not a GitHub Actions
workflow, so the arm64 image is built and smoke-tested natively.
Multi-arch
(`linux/amd64`+`linux/arm64`) via a multi-arch buildx builder with a
remote node that builds arm64 natively, no QEMU (`CROPWRIGHT_BUILDER`,
default `cropwright-multiarch`); the arm64 leg is smoke-tested over its
remote docker context (`CROPWRIGHT_REMOTE_ARM64_CONTEXT`, default
`remote-arm64`) since it can't run on an amd64 host. See README's
"Releasing" section for the stage list and usage; there is no
`.github/workflows/release.yml`.

## Style

- Dark theme by default (photographers work in dim environments).
- No emoji. No gradients. Apple system colors.
- Tailwind `bg-zinc-950` base, accent via CSS variables for easy retheme.

## Public export (F10, `scripts/oss-export/`)

Cropwright is published as a separate public repo
(`davidamacey/OpenProcessor`, fresh history, MIT, Copyright example-org LLC)
built from a filtered export of this tree —
`docs/design/cropwright-oss-export-plan-2026-09-25.md`. Scrubs land here
as ordinary forward commits; the exporter is private-only and never
ships:

- `scripts/oss-export/export.sh <SHA> <EXPORT_DIR>` — `git archive` the
  sha, delete every path in `exclude.txt` (each must exist, or the export
  fails), copy `overlay/` over the result, strip the private
  `export-leak-gate` CI job and the `master` trigger, run prettier and the
  leak gate. `EXPORT_DIR` must be empty and outside any git work tree.
- `overlay/` holds the public `CLAUDE.md`, a fresh `CHANGELOG.md`
  (curated `[0.1.0]`), `docs/README.md` and the `docs/design/README.md`
  stub. **Keep the public `CLAUDE.md` in step** with this one when a
  route, mechanism or test tier changes (until the public repo becomes
  the upstream and this repo is archived — D5-A).
- `leak-scan.sh <EXPORT_DIR>` / `--tree .` — `leak-patterns.txt` (zero
  tolerance, no allow-list), email/IPv4 sweeps, excluded-path presence,
  lockfile registries and gitleaks (container image).
- Public screenshots come only from the fresh-start public-data run
  (COCO val2017, Open Images plates) — the docs-site captures above.

## Documentation & changelog discipline

This is enforced (CI, see `.github/workflows/ci.yml`'s `changelog` job), not
just a convention — don't rely on remembering it:

- **Every push/PR that changes `src/` or `scripts/` must also touch
  `CHANGELOG.md`.** Add the entry under `## [Unreleased]` in Keep a
  Changelog format. For a multi-commit batch of work, one consolidated
  entry in the batch's last commit is fine — CI checks the whole
  push/PR diff, not each commit individually.
- **Escape hatch:** if a change genuinely has no user-facing or
  architectural effect (a pure test-only fixture tweak, a typo fix),
  include `[skip-changelog]` anywhere in the last commit message of the
  push. Don't reach for this to avoid writing a real entry — it's for
  the rare case where "what changed" truly isn't answerable in
  changelog terms.
- **Architecture-level changes** (a new route, a renamed core
  mechanism, a capability added to the slot model, a cross-repo contract
  change) should also update this file (`CLAUDE.md`) in the same
  push/PR — CI does not check this one mechanically, it's a human
  (or agent) judgment call, but it's the reason this file has stayed
  accurate through several large refactors instead of rotting.
- Planning passes for non-trivial work leave a dated plan doc behind in
  `docs/design/` (see existing files there for the naming convention) —
  this is how the reasoning behind a change stays discoverable after
  the fact, not just the diff.

## Known constraints

- HTML5 DnD doesn't work reliably across all browsers/webviews. Use
  `svelte-dnd-action` (pointer-based) only.
- Don't fetch full-resolution NAS images in grids. Always use the thumbnail
  endpoint (128×128 LRU cached server-side).
- Test-holdout crops MUST never be relabeled by the VLM or via cluster
  auto-suggest. The OpenProcessor `{API_PREFIX}/` endpoints filter; UI is the second line.
