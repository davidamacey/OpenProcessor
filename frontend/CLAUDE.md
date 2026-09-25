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

Manage a high-volume labeling workflow over hundreds of thousands of vehicle
crops, with cluster-based assisted labeling, Gemma 4 vision suggestions, and a
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

All routes below are shipped and linked from the top nav (`+layout.svelte`)
except `/clusters/[id]`, reached via a cluster card. `/` is not a route in
its own right — it's a bare 307 redirect to `/dashboard` (`src/routes/
+page.ts`), and the top-bar logo/"Dashboard" link both point at
`/dashboard` directly, not `/`. The MVP/post-MVP split from the original
design doc is gone — every route in this table exists and works; nothing
here is a stub.

| Route            | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/dashboard`     | Current pipeline dashboard — live `DatasetStats` (polls every 10s) + `AutoLabelPanel` ("Run Clustering Now" with stage progress), shared with the daemon-fired auto-label run. `AutoLabelPanel` also hosts an optional per-class assist scope (`AssistScopeBar`, absent unless `/methods` advertises a usable `prompt_pack` — see "Curation-strategy selector bar" below) that lets an operator point the VLM-assisted sweep at a single class instead of the whole pool.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `/ingest`        | Bring images into the pool. Offers browser upload (files, folders and drag-drop) to `POST {API_PREFIX}/ingest/upload`, chunked to the served per-request cap with bounded concurrency. It shows a per-file result (ingested / duplicate / failed + served reason), supports pause/resume/cancel, and pre-filters already-indexed identifiers via `POST {API_PREFIX}/ingest/path_lookup`. An optional server-path mode uses `POST {API_PREFIX}/ingest/batch` and is shown only when the backend advertises at least one `batch.source_roots` entry via `GET {API_PREFIX}/ingest/config` (see "Ingest" below). The page also has an ingest status table by source (`GET {API_PREFIX}/ingest/status`), a region-drain panel (`GET {API_PREFIX}/ingest/region_drain`, only with a served region profile), and a clustering handoff that reuses `AutoLabelPanel`. The route is gated by `ingestAvailability`: when the backend lacks the ingest router, the page is absent, not disabled.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `/clusters`      | Cluster grid view, sidebar filter, **strategy bar** (review-sort dropdown + score chips, see below — no cluster-method picker here; that lives on `/settings`). When the class filter is a slot-bound class (the served region profile's `region_class_name`), replaces the cluster grid with that slot's **region gallery** (`SlotGallery`, driven by `createSlotGalleryController(slot)` in `src/routes/clusters/slotGalleryController.svelte.ts`, one controller per slot bound through `slotForClassName`, browsing the slot's `queue.browsePath`, i.e. `{API_PREFIX}/regions`; detector / verified / status / score / text filters, all copy templated over `slot.label`). The unfiltered grid pins one synthetic inventory card per registered slot with a browse endpoint. Also hosts the **embedding-plot** overlay toggle when `viz_projection` is available (see below). An **Ignored** toggle (2026-09-24, logic-moves W7) swaps the grid for the excluded/`cluster_id=-2` bucket with a "Restore selected" action, and an **item-text search** box (`{API_PREFIX}/crops?item_text=`) swaps it for a literal OCR-text search over `item_text_lines` — both mode-swaps mirror the existing dataset-wide semantic search's pattern, and neither is the same endpoint as the semantic (embedding) search box.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `/clusters/[id]` | Single cluster crop grid + DnD + bulk ops + strategy bar (sort / diverse overlay / score chips scoped to this cluster)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `/review`        | 6 top-level review tabs (2026-09 consolidation, down from 9, plus `new_class_proposals` added 2026-09-24 — see below): **All** / **Uncertainty** / **Model Disagreements** / **Classifier Blind Spots** / **New Class Proposals** / one tab per registered queue-capable slot (in practice the one region tab, present only when the backend serves a region profile and labelled by its `display_name`, `?tab=regions`), each with its own default sort (`review_sorts.py`'s `_TAB_DEFAULTS`) plus the strategy bar's selectable sort/score overlays — the bar's summary chip also shows the server's `sort_applied` next to whatever was requested. The All tab additionally offers a row of **quick-filter preset chips** (VLM mismatches / VLM low-conf / Primary · low-conf) that layer the former Mismatches / Gemma Low-Conf / Primary · Low-Conf tabs' exact cohort queries on top of the All view. Class/Source/Conf filter controls (`class_id`/`source`/`conf_min`/`conf_max`) are live against `GET {API_PREFIX}/review/{tab}`. `/review?crop_id=` deep links resolve via `GET {API_PREFIX}/review/{tab}/locate` — jumps straight to the crop's served page/rank, or shows the backend's `reason` when it isn't in the queue. A slot tab carries provenance chips + the region text reading, driven by the active slot's capabilities rather than a hardcoded tab check (see "Slot-generic review tabs" below).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `/classes`       | Add / rename / merge / **deprecate** / **restore** classes, per-class hotkey binding, and a **Proposals** section (`GET {API_PREFIX}/review/new_class_proposals/summary`) for creating a class from — or mapping onto an existing class — a VLM-proposed term the registry doesn't have yet, bulk-resolving _every_ pending crop proposing that term (`POST {API_PREFIX}/review/new_class_proposals/resolve`), not just the summary's sample thumbnails. Every active row has a **Deprecate** button (`POST {API_PREFIX}/classes/{id}/deprecate`, confirm-gated) — 409 with a structured `class_still_referenced` detail (`{message, item_count, confirmed_label_count}`) offers the existing merge dialog instead, preselecting the class as the merge source. The deprecated-classes table's **Restore** button (OpenProcessor 698d1da, cf3c87a) is real, not the permanently-disabled placeholder it used to be — `POST {API_PREFIX}/classes/{id}/restore`, whose 409 is a PLAIN STRING detail (a live class already claims the name) shown verbatim.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `/export`        | Trigger YOLO export, view balance gap, freeze test holdout, download the frozen export's `class_registry.json`/`data.yaml`/`manifest.json` (via `{API_PREFIX}/export/registry/{artifact}`). Since OpenProcessor 6c77deb, `GET {API_PREFIX}/export/status` also serves `image_count`/`class_count`/`group_key`/`split_counts`/`class_split_counts` — the page shows the served train/val/test totals and a collapsible per-class table (any class at 0 train or 0 val highlighted, served numbers only). The freeze modal has no Seed field: `POST {API_PREFIX}/test_holdout/freeze`'s body is `{percent}` only (selection is deterministic, SHA1 of each crop id per class — an extra `seed` key is a 422); the success toast shows the served `selection`/`min_per_class` when present. Since OpenProcessor d5343cb (one image + one label file per source image), `ExportStatus` also carries `object_count`/`split_object_counts` (label lines, distinct from `image_count`/`split_counts`) — the page reads "N objects in M images" and separate images:/objects: split badges, never one ambiguous number. An opt-in "Only images whose every object is labeled" checkbox sends `require_fully_labeled_images` on `POST {API_PREFIX}/export/yolo`; the served partial-frame counts (`unlabeled_items_on_exported_images`/`images_with_unlabeled_items`/`images_dropped_not_fully_labeled`) render when present. Every one of these fields is `null` (not `0`) on an export written before it was recorded — rendered via `formatCount()` (`src/lib/formatCount.ts`) as "—". All of the above is optional/nullable on `ExportStatus`/`TestHoldoutFreezeResult` so a pre-6c77deb/pre-d5343cb backend (missing every new field) renders exactly as before.                                                                                                                                                                                                                 |
| `/models`        | Triton model registry browser                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `/train`         | Training cockpit — preflight, launch, live progress, log tail, past runs, **Promote**, **Reproduce**, **Training cohorts picker** (class-agnostic `CORE_COHORTS` for every class + the region profile's 5 server-side modes when one is configured — see "Training cohorts" below). The dataset card shows the _current export's own_ `image_count`/`class_count`/`split_counts`/per-class `class_split_counts` (from `GET {API_PREFIX}/export/status`, OpenProcessor 6c77deb) rather than the dataset-wide validated total — the old global number (which double-counted `test_holdout` crops) survives only as a clearly-labelled "(global pool)" fallback for a pre-6c77deb backend or an explicitly-picked past export version. `AugmentationPanel`'s preset picker is served from `GET {API_PREFIX}/train/augmentation_presets` (id/label/description/orientation-sensitive), defaulting to the served `default`; a pre-6c77deb backend 404s that endpoint and the panel falls back to a read-only display of the current value. `/train/start`/`/start_campaign`'s 422 on an unknown `augmentation.preset` (`{detail: {message, field, valid_presets}}`) surfaces `valid_presets` in the toast alongside the message. Since OpenProcessor d5343cb, the card also shows the export's `object_count`/`split_object_counts` (label lines) alongside `image_count`/`split_counts` — "N objects in M images" plus separate images:/objects: split badges — and its per-class table is objects, not an ambiguous count; a `null` field (an export written before d5343cb recorded it) renders via `formatCount()` as "—", never 0.                                                                                                                                                                                                                                                                                                                                           |
| `/bakeoff`       | Model comparison on OpenProcessor #34's v2 wire (F6, `docs/design/bakeoff-v2-ui-plan-2026-09-25.md`). Pick eval datasets (`GET {API_PREFIX}/bakeoff/eval_datasets`: export test splits first with the current one flagged and preselected, external frozen sets grouped by served `group`), models (finished training runs from `/bakeoff/trained_models`, each with the served per-selected-dataset `for_dataset` facts and a train/test overlap warning when the served overlap is non-null and > 0; the profile's `/bakeoff/baseline_models`; an optional custom ref) and a profile (whatever `/bakeoff/profiles` serves, `default_profile` preselected, `default_error` shown). A confirm dialog precedes `POST /bakeoff/run` (typed `run`/`baseline`/`custom` refs; 400/409/422 detail shown verbatim); the job polls `/bakeoff/status/{id}` with progress, per-stage failures and the enqueue-time class mapping. Results: the `/bakeoff/matrix/{id}` model × dataset matrix bolds every served tied winner (`best` is a list), and `/bakeoff/results/{id}?dataset_id=` renders ranked rows plus a per-class table where an uncovered class reads "not covered" and each model's unmapped classes are listed; a 409 (pre-v2 result) shows a note. Previous runs (`/bakeoff/runs`) stay selectable. State in `src/lib/bakeoff/bakeoffController.svelte.ts`, types in `src/lib/types_bakeoff.ts` (pinned to the vendored OpenAPI by `contract/bakeoffContract.test.ts`); nothing computes a metric, mapping, rank or winner client-side. Gated on backend availability via a one-shot probe of `GET {API_PREFIX}/bakeoff/runs` (`src/lib/bakeoffAvailability.svelte.ts`) — absent, not disabled: the nav link and page body don't render at all when the backend's `{API_PREFIX}/bakeoff/*` router isn't mounted, and no discovery request fires unconditionally on mount. The probe itself would move to an `evaluation` axis on `/methods` once the backend ships one. |
| `/settings`      | Deployment-defaults admin page for the shared curation-strategy defaults (`GET,PUT {API_PREFIX}/settings`) — one place to pin the deployment's clustering method, review-queue sort and VLM prompt pack (the latter honored by the always-on VLM labeler and by auto-label runs that don't pick their own), plus a read-only "Set by the backend's startup config" section for `detection_profile` (the backend picks it from startup config; nothing that runs reads a shared or per-run selection). Which axes get a control is decided solely by the server's per-entry `settable` flag on `/methods` (`settableAxes` in `src/lib/curationSettings.ts`). Deployment-wide — see `docs/design/curation-settings-ui-plan-2026-09-21.md` — so it is its own route rather than a `StrategyBar` chip, with an explicit confirm dialog before every save. Also hosts the **Curation scores card** (`ScoresCard.svelte`, G10, 2026-09-24) — per-scorer coverage from `GET {API_PREFIX}/scores/coverage`, confirm-gated "Compute all"/"Compute selected" (`POST {API_PREFIX}/scores/compute {scorers}`, ids always sourced from the served coverage keys), a progress poll of `GET {API_PREFIX}/scores/status` following `EmbeddingPlot`'s rebuild-job pattern, and "Cancel" (`POST {API_PREFIX}/scores/cancel`). Absent, not broken, when `/scores/coverage` 404s; a failed compute (e.g. mistakenness lacking probe predictions) shows the backend's error verbatim. A completed compute reloads coverage and resets `strategiesStore` so `StrategyBar`'s sort/score options pick up the new coverage without a full page reload — see "Curation-strategy selector bar" below.                                                                                                                                                                                                                                                                                                   |

## Ingest (`/ingest`, 2026-09-24; BA-1..BA-7 adopted 2026-09-25)

Bring images into the pool — the frontend side of
`docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md`. All 13 pieces
of that plan are implemented as of OpenProcessor #36 (backend commit
c676d2b) — see the plan's status section for the full piece-by-piece
record.

- **Gating.** `src/lib/ingest/ingestAvailability.svelte.ts` probes
  `GET {API_PREFIX}/ingest/status` once (modelled on
  `bakeoffAvailability.svelte.ts`): a 404/501 hides the nav link and the
  route body entirely; any other failure leaves the optimistic value
  alone. The **page itself** (not just the nav link) additionally
  renders neither the absence copy nor the upload UI while `available`
  is still `null` — deliberately not the nav link's optimistic
  behavior, because the upload/status/drain child components each fire
  their own GET on mount, and a genuinely-absent backend must fire zero
  ingest requests (caught live by `e2e/stubbed/test_ingest.py`'s
  `test_ingest_absent`, which found this as a real bug during
  implementation). The availability probe deliberately still targets
  `/ingest/status`, not `/ingest/config` — `/routes/ingest/+page.svelte`
  fetches `getIngestConfig()` itself, once, only after `available`
  resolves `true`, so a config fetch never fires against a backend that
  lacks the router.
- **Served config (BA-2, landed).** `GET {API_PREFIX}/ingest/config` is
  real and wired up (`getIngestConfig()`, `api.ts`) — `ingestConfig.ts`'s
  `resolveIngestConfig(served)` resolves every upload/batch/region-drain
  limit from it, falling back to the documented interim constants only
  while the fetch is in flight or against a pre-BA-2 backend (404 →
  `null`). `uploadMaxBytes` is the tighter of the nginx proxy's
  `client_max_body_size` and the served `upload.max_bytes_per_request` —
  a chunk sized against only one of the two could still 413 against the
  other.
- **Upload caveat.** The amber "uploaded images can't be browsed" banner
  now renders only when `upload.persists_bytes` is `false`, or when it's
  unknown (`null` — a pre-BA-2 backend, or the served config hasn't
  loaded yet) and the operator hasn't set
  `PUBLIC_CROPWRIGHT_INGEST_UPLOAD=1`. BA-1 landed
  (`POST {API_PREFIX}/ingest/upload` persists the uploaded bytes
  server-side, content-addressed) so a deployment on c676d2b+ serves
  `persists_bytes: true` and shows no banner at all — the served truth,
  not a hardcoded caveat.
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
  server-persisted content-addressed path; falls back to `image_path`
  for a pre-BA-1 backend that doesn't echo `source_identifier`), a
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
  `${sourceTag}/`), computed by `src/lib/ingest/fileSource.ts`'s
  `makeIdentifier` — normalizes backslashes, strips a leading `./`/`/`,
  and rejects any `..` segment. `path_lookup` matches this identifier
  exactly (against either `image_path` or, since BA-1, `source_identifier`
  — server-side), which is why the prefix matters: two different folders
  that both contain e.g. `img001.jpg` at their root would otherwise
  collide.
- **Server-path ingest (piece 11, landed).** `IngestBatchPanel.svelte`
  renders on `/ingest` only when the served `batch.source_roots` is
  non-empty (`config.batchSourceRoots`) — lists the roots read-only and
  submits real batches via the pre-existing `POST
{API_PREFIX}/ingest/batch` (that endpoint predates #36; BA-2 is what
  serves `source_roots`/`max_items` for the gate and client-side cap
  check, and BA-5 is what guards a submitted `label_txt_path` against
  the same configured roots as the image path server-side). One
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
  `^__API_PREFIX__/ingest/` location (declared before the general API
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
  auth (BA-5's auth half is still open; only the `label_txt_path` root
  guard half of BA-5 landed with #36).
- **Plan deviations** (recorded in the plan doc's own status section
  too):
  - the plan's nginx-413 detection ("`ApiError.detail == null`")
    doesn't match `api.ts`'s actual `errorDetail()` — a non-JSON body
    always becomes a non-null string `detail`, never `null`. The
    controller instead keys off `e.body`'s _type_ (`object` for a
    backend 413, `string` for nginx's HTML page), which is what
    `apiFetch` actually produces.
  - the plan's upload-override env var (`CROPWRIGHT_INGEST_UPLOAD`)
    isn't exposed to the client bundle — `vite.config.ts`'s `envPrefix`
    only forwards `VITE_`/`PUBLIC_`-prefixed vars. Uses
    `PUBLIC_CROPWRIGHT_INGEST_UPLOAD` instead, matching every other
    client-visible env var in this codebase.
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

## `/review` tab consolidation (2026-09)

A review of all 9 original review-queue tabs against the live
1,000-crop index found three were too big to function as curated
queues — closer to "most of the dataset" than a triaged worklist:
Mismatches (1,000 rows · 97.5% the size of All), Gemma Low-Conf
(1,000 · 11% of the dataset), and Primary · Low-Conf (1,000 · 92% of
the _entire_ dataset). A fourth, Outliers, had only 3 live rows and was
functionally identical to the `atypicality` sort already available via
the strategy bar below.

- **Outliers** was retired entirely — no tab, no rendering path. Its
  backend `{API_PREFIX}/review/outliers` query is untouched/unlinked, not
  deleted (out of scope for a frontend-only change).
- **Mismatches / Gemma Low-Conf / Primary · Low-Conf** collapsed from
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
  OpenProcessor `af3a580`) each dry-run
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
  "Top 2". `filterSupported` defaults to **visible** when a tab's
  `filters` is absent/unknown (older backend, or an id `GET
{API_PREFIX}/review/tabs` doesn't know) — a control never disappears
  because the vocabulary endpoint is stale or hasn't loaded yet.
- **Generic served-enum filter bar (OpenProcessor 840beb8 adoption).**
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
filters/sort, so the page loads pages up to `rank`'s page and jumps the
cursor there directly. `in_queue: false` (already handled, filtered out,
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

The Class/Source/Conf filter bar sends `class_id`/`source`/`conf_min`/
`conf_max` to `GET {API_PREFIX}/review/{tab}` — re-enabled unconditionally
once the backend started honoring them (verified live: each changes the
queue total). The control that used to read/write `Crop.hdd_source` (a
field the backend never actually populated) is renamed "Source" and
reads/writes the live `Crop.source` field instead.

`model_disagreements` items carrying `probe_pred_class_id` get an
"Accept model's class" button next to the "Model predicts" row —
`assign()` takes the served id directly, no class-name-to-id lookup
needed (closes G4, tracked since the initial probe-prediction display).

## Curation-strategy selector bar (`StrategyBar.svelte`, 2026-09)

`/clusters`, `/clusters/[id]`, and `/review` all render a `StrategyBar` —
a collapsed-by-default chip row that expands into independent, **stackable**
controls, never a replacement for the production defaults:

- **No cluster-method picker.** An earlier version of this section
  claimed `StrategyBar` included one; it never has. `rg` over non-test
  source finds `cluster_methods` referenced only in `strategies.ts`
  (types/parse/fallback) and `strategiesStore.defaultClusterMethodId` —
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
contract. Landed on OpenProcessor main (80dd097): unknown ids 422 with
`{axis, requested, valid_ids}`, resolved values echo in the job's `args`,
and `class_id` scopes only the VLM sweep, not clustering or auto-promote.

Every one of these degrades gracefully to invisible/default when its
backend flag is off or `{API_PREFIX}/methods` fails: `strategiesStore` falls back
to `FALLBACK_METHODS` (`strategies.ts`) — the hardcoded stable-only list
matching what's always been implemented — so a missing endpoint never
breaks page load. Flags are OpenProcessor env vars
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
`has_item_scores`) — a sibling read, `getReviewEmptyState()`
(`reviewTabsVocabularyStore.emptyState`, kept separate from
`getReviewTabsVocabulary()` so the tabs array's own shape/tests are
untouched). When the served reason mentions a probe or a score and the
matching `empty_state` flag is false, the panel adds a direct link ("Run
a probe on /train" / "Compute scores on /settings") instead of leaving
the operator to guess where to go — verified live: the Uncertainty
queue's empty panel links to `/train` on this deployment (no probe has
ever run). Absent/malformed on either field renders exactly as before.

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

- `/clusters` cards show `"<purity_tier text> NN% · n=NNN"` next to the
  size chip, with `purity_basis`/`label_purity`/`labelled_share` in the
  chip's tooltip. The legend strip's tooltip also names the basis now.
- `/clusters/[id]`'s header gained a `"purity (nearest-centroid) NN% ·
n=NNN"` line it previously lacked entirely (label_purity/
  labelled_share in its tooltip).
- All four new fields (`purity_n`, `purity_basis`, `label_purity`,
  `labelled_share`) are optional on `RawCluster`/`Cluster` and render
  gated on non-null — an older backend that doesn't serve them shows the
  same purity number as before, just without the n/basis/label-purity
  detail.

## Training UI — `/train` (Phase 2 of the training pipeline)

Full cockpit for the training pipeline. The form auto-runs `{API_PREFIX}/train/preflight`
on a 350ms debounce and renders the report inline. Status polls every 5s,
log every 2s, both stop on terminal state. Multi-size campaigns get a
size-chip swap in the submit row (auto-promote-best + `stop_when` threshold).

The preflight panel (`TrainForm.svelte`) renders every check generically
by `name`/`severity`/`message` — no per-check-id branching, so OpenProcessor
6c77deb's three new checks (`export_splits_nonempty`,
`export_class_split_coverage` — blocks per class, thresholds served as
`min_train_per_class`/`min_val_per_class` — and `augmentation_preset`)
needed no component change; each check's `message` already names the
offending classes/splits in prose, and its `detail` object (when present)
renders in a collapsible JSON block alongside it.

Past-runs table actions:

- **Promote ↑** — opens `PromoteModal` (Triton model name, max_batch_size,
  fp16, overwrite, force-bypass-gate). 422 with the gate report renders
  inline; `force=true` bypasses for known-good experimental runs.
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

- **Val vs. test labelling — resolved.** `eval`'s overall figures
  (`map50`/`map50_95`/`precision`/`recall`) and its `per_class` table
  are labelled independently by `evalOverallLabel`/`evalPerClassLabel`
  (`src/lib/trainResults.ts`), never assumed to both mean "test". The
  eval-split cutover (OpenProcessor e9aac68) has landed: the backend
  serves `eval.split` (`'test'` when the frozen-holdout pass ran, `'val'`
  when it fell back), and this is the run's **headline number** —
  labelled directly by `split`, not the pre-cutover guess. A run whose
  `eval` predates the cutover (no `split` served) still falls back to
  the old guess (overall = last val epoch, per-class = test) — see
  `types_train.ts`'s `TrainEval` doc comment.
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
trainer_image_id}` (`trainer_sha` replaces the old `trainer_image`
  field name — a rolling-deploy backend may still echo that legacy key
  too, but it's never read), and a class-remap table (new id → original
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
keydown listener in `src/routes/+layout.svelte`. There is no `1-9, 0` top-N
scheme — it was removed; one binding scheme means no "what does this key do
here?" friction. On `/clusters/[id]` a class letter labels the current
selection (or the just-dragged set); on `/review` it labels the current item.

Reserved single-char action keys (`g n d z x u a m /`, plus `b f e` from the
region slot's keymap — server-served today as `/abdefgmnuxz`)
cannot be bound to a class — `setClassHotkey` (`src/lib/classHotkey.ts`)
rejects them, validating against `reservedHotkeyLetters()`. As of the
2026-09-24 OpenProcessor `logic-moves` cutover, the base set is no longer a
hand-maintained frontend constant: `GET {API_PREFIX}/classes` serves its own
`reserved_hotkeys` field (`classesStore.reservedHotkeys`), and
`reservedHotkeyLetters(registry)` is that served set **∪ every
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
(`classesStore.topNForCluster(0, 10)`) only ever surfaces the most-validated
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
classVisibility.ts`'s `isSlotBoundClass` reads the served `kind` first;
falls back to the slot registry only when a class isn't tagged (an older
backend) — the R1 fix (the region class topping the item-class picker
and quick-assign row) is now backed by the server's own classification,
not a client heuristic keyed off the slot registry alone. `/export`'s
per-class table (`exportDatasetRows.ts`) prefers the served `trainable`/
`trainable_gap` over the client-side validated-minus-holdout math, kept
only as the fallback for an older export/backend.

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
`unloadable` FIRST (`=== false` hides the button outright, ahead of
`kind`/`is_region_protected`); a backend that predates the field
(`undefined`) falls back to the prior `kind !== 'triton'` rule. A
region-protected-but-otherwise-unloadable model still shows no button,
but `/models` now renders a "protected: in use by the pipeline" chip in
that case (`showsProtectedChip`) instead of nothing. The segmenter can
also serve `status: 'not_configured'`, with null inference/exec/latency
fields — rendered "—" by the existing `fmtCount`/`fmtMs`, never `0`.
Live, this leaves only the CLIP/PE image-encoder models with an Unload
button.

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
  (openprocessor fix #29 / OpenProcessor 840beb8) — see
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

OpenProcessor `main` f7171cc ("region verdict integrity") changed three
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
  styled/worded by the served rejection **kind** (OpenProcessor 840beb8
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
placeholders as of OpenProcessor 840beb8 (openprocessor fix #29): the
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
  `getSourceImageScaled`/`getSourceImageFull` (`{API_PREFIX}/crops/{id}/image`).

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
region_class_name, text_reader} | null` (`display_name_singular` added
2026-09-25, OpenProcessor #36 item 10 — see below). **It is the only
gate for region features.** With `null`, every
region route (`/regions`, `/crops/{id}/region*`, region undo, the VLM
verify routes, `/regions/clusters`) answers 409, so the UI renders no
region surface at all and calls none of them.

- `regionProfileStore` / `loadRegionProfile()`
  (`src/lib/stores/regionProfile.svelte.ts`) reads the prefixed
  `{API_PREFIX}/health` once in the root layout's `load()` (bounded at
  2 s, alongside the tier-2 load). A failure, a timeout, or a backend
  that predates the field counts as not configured (fail closed). The
  UI is built from that one reading; if a later `healthStore` poll (or
  a region route's 409) disagrees, one sticky "reload to apply" notice
  shows. Tabs are never hot-swapped.
- `regionSlotFromServedProfile()` (`src/lib/annotations/servedRegionSlot.ts`)
  builds the region `SlotSpec`: `key` = profile `name`, `bind.className`
  = `region_class_name` (falls back to `name`), tab label / plural noun /
  stats panel title = `display_name` (falls back to "Regions"), singular
  label/title = `display_name_singular` (falls back to "region"/"Region"
  — item 10, 2026-09-25), used everywhere a singular-context string is
  built generically ("Confirm `<Region>`", "`<Region>` score", "Edit
  `<region>`") so a deployment reads "Confirm Plate" rather than the
  generic "Confirm Region", `?tab=` id and review endpoint = `regions`, browse path `/regions`,
  text capability only when `text_reader` is non-empty, no client
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
- A region route's 409 `no region profile is configured` becomes
  `RegionProfileUnavailableError` in `apiFetch`
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
  first). The false-positive bucket id is one constant,
  `FALSE_POSITIVE_REGION_CLUSTER_ID` (TODO: the served
  `cluster_kind === 'false_positive'`, which OpenProcessor now serves on
  `/regions/clusters` cards). `CropCard` picks its sub-box slot by the
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
  noun, `LPR`/`lpr_`, `legacy`, `/curation` or private dataset number
  anywhere under `src/` (code, strings, comments, tests). Its allow-list
  is only itself and `profiles.falsification.test.ts` (the one test
  that exercises `examples/`). Fix the file; don't widen the allow-list.
  Vehicle/Gemma/SAM3 literals are a separate sweep (audit §8) and not
  scanned yet.

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
Playwright). Covers the flows a plain `npm test` (jsdom, no real
browser) can't: keyboard-driven `/review` assign/undo, `/clusters/[id]`
drag/hotkey/discard, the `/settings` and dashboard assist-scope
wire-composition round trips, tier-2 annotation-profile loading, and
that every top-level route still mounts. Never runs against a live
OpenProcessor backend — `e2e/conftest.py`'s `Stub` fixture routes every
`{API_PREFIX}` request itself.

**Fail-closed, on purpose** (see docs/design/test-audit-2026-09-24.md
recommendation 5 — this replaced the old `scripts/playwright_*.py`
runbooks, which drifted to a stale `/curation/**` prefix for weeks because
their catch-all stub silently answered `200 {}`): a request under
`{API_PREFIX}` that no test registered gets `501` and is recorded in
`stub.unhandled`; a request to the retired `/curation/` prefix is intercepted
and recorded in `stub.op_hits`; every test asserts at teardown that both
are empty and that at least one stub actually fired. An API-prefix
change or a route rename fails the test outright instead of the page
silently rendering empty. CI runs this in the `e2e-stubbed` job on every
push/PR; a `py_compile` pre-commit hook (and CI step) gates
`scripts/*.py` and `e2e/**/*.py` syntax separately from this suite.

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
preflights `GET {API_PREFIX}/health`, skipping with a clear message on
anything short of a clean 200.

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
past the viewport edge at ≤800px — `src/routes/+layout.svelte`'s primary
nav is now its own horizontally-scrolling strip, `overflow-x-auto
whitespace-nowrap`, with every link `shrink-0` and the status chip
pinned `shrink-0` so it's never squeezed.)

Three test modules, 24 tests total against this deployment's live
dataset:

- **`test_route_sweep.py`** — every top-level route mounts: `/dashboard`,
  `/ingest`, `/clusters` (plain and `?class=<served region_class_name>`), `/review`
  with each tab's `?tab=<urlId>` (`all`/`uncertainty`/
  `model_disagreements`/`classifier_blind_spots`/`new_class_proposals`/
  `regions`), `/classes`, `/export`, `/train`, `/models`, `/bakeoff`,
  `/settings`. `/ingest` fails against any backend that predates
  `docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md` (this
  currently deployed build 404s it) — that's expected until the next
  deploy, not a bug. Each asserts:
  no `pageerror`; no `**/curation/**` response >= 400 outside a small,
  explicit, documented allow-list (`ALLOWED_4XX_5XX` in
  `e2e/live/conftest.py` — empty today, since every endpoint this
  deployment serves on mount came back 200/204 in manual verification);
  no literal `"NaN"`/`"undefined"` in the rendered body text; every
  `<img>` whose bounding box intersects the 1280×720 viewport finishes
  loading (`naturalWidth > 0`) — an offscreen lazy image is allowed to
  still be pending; no horizontal overflow at the narrow 800px viewport
  (see above). Also saves the always-on desktop/narrow screenshots
  described above.
- **`test_data_agreement.py`** — the UI shows what the API serves:
  the dashboard's "Clusters (total now)" vs `GET {API_PREFIX}/stats/
dataset` `clusters.cluster_count`; `/review?tab=regions`'s queue-counter
  total vs `GET {API_PREFIX}/review/regions` `total`, both unfiltered
  and with `?region_status=verify_rejected`; every `filter_specs` entry
  `GET {API_PREFIX}/review/tabs` serves for the `regions` tab renders a
  `<select>` with exactly the served option labels; a served
  `rejection_reasons` label (`GET {API_PREFIX}/regions/vocabulary`,
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
  vs `GET {API_PREFIX}/ingest/status`) and `test_region_drain_agrees`
  (the region-drain panel's `total_unfinished` vs
  `GET {API_PREFIX}/ingest/region_drain`, via `wait_for_stable_text`
  since the value moves during a live cascade) — both fail against a
  pre-ingest backend the same way `test_route_sweep.py`'s `/ingest`
  case does.
- **`test_deep_link.py`** — takes the first crop id off `GET
{API_PREFIX}/review/regions?region_status=verify_rejected&page_size=1`
  (skips if none), opens `/review?tab=regions&region_status=
verify_rejected&crop_id=<id>`, and asserts it actually lands: "Locating
  crop…" clears, the queue counter reports a real `rank / loaded`
  position, and no "not in this review queue" toast appears.

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

The production build runs in an `nginx:alpine` container defined by this
repo's `docker-compose.yml` (`docker compose up -d --build`), host port
5184 (`CROPWRIGHT_PORT`). Port conflicts: 5174=example-app-backend,
5180/5181=example-app-opensearch, 5183=example-app-docs.

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
for in-flight cancellation.

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
  `../openprocessor`; `OPENPROCESSOR_REF`, default `main`) via
  `git -C $OPENPROCESSOR_REPO show $OPENPROCESSOR_REF:contracts/...`.
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
    completeness guard) resolves to a real path+method in the vendored
    OpenAPI spec, with every mechanically-resolvable query param checked
    against that operation's declared parameters. A handful of
    runtime-composed paths (a slot's own declared `endpoints`/
    `extras.datasetExport` paths) are anchored `MANUAL_OVERRIDES` in the
    test, not silently skipped.

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

## Style

- Dark theme by default (photographers work in dim environments).
- No emoji. No gradients. Apple system colors.
- Tailwind `bg-zinc-950` base, accent via CSS variables for easy retheme.

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
  auto-suggest. The openprocessor `{API_PREFIX}/` endpoints filter; UI is the second line.
