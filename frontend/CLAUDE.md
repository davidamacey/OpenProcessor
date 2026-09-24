# Cropwright — CLAUDE.md

SvelteKit + TypeScript image-crop annotation web app (product name
**Cropwright**, package name `cropwright`; repo directory is still
`legacy-labeler` pending a physical rename). Generalized via a
capability-model / annotation-slot mechanism (see
`docs/genericization-plan-2026-09-13.md`) so it is no longer
hardcoded to vehicles or license plates — this deployment is
currently configured for a vehicle + license-plate dataset construction via
`src/lib/annotations/registeredSlots.ts` and `profiles/licensePlate.ts`,
but a new domain is added by registering a new slot profile, not by
editing app code. Sister project to `legacy_sorter` (v2 Tauri app for
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
except `/` and `/clusters/[id]`, which are reached via the logo / a cluster
card respectively. The MVP/post-MVP split from the original design doc is
gone — every route in this table exists and works; nothing here is a stub.

| Route                        | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/` (legacy, logo link only) | Older stats + recent-crops + quick Gemma-cluster-run page. Superseded by `/dashboard` for nav purposes but still reachable; not deleted since it's a working page, just not the primary entry point.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `/dashboard`                 | Current pipeline dashboard — live `DatasetStats` (polls every 10s) + `AutoLabelPanel` ("Run Clustering Now" with stage progress), shared with the daemon-fired auto-label run. `AutoLabelPanel` also hosts an optional per-class assist scope (`AssistScopeBar`, absent unless `/methods` advertises a usable `prompt_pack` — see "Curation-strategy selector bar" below) that lets an operator point the VLM-assisted sweep at a single class instead of the whole pool.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `/clusters`                  | Cluster grid view, sidebar filter, **strategy bar** (review-sort dropdown + score chips, see below — no cluster-method picker here; that lives on `/settings`). When `class=license_plate` is selected, replaces the cluster grid with a **plate-thumbnail grid** backed by `{API_PREFIX}/regions` (detector / verified / score / plate-text filters; click → jump to the license_plate slot's review tab). Also hosts the **embedding-plot** overlay toggle when `viz_projection` is available (see below). An **Ignored** toggle (2026-09-24, logic-moves W7) swaps the grid for the excluded/`cluster_id=-2` bucket with a "Restore selected" action, and an **item-text search** box (`{API_PREFIX}/crops?item_text=`) swaps it for a literal OCR-text search over `item_text_lines` — both mode-swaps mirror the existing dataset-wide semantic search's pattern, and neither is the same endpoint as the semantic (embedding) search box.                                                                                                                                                                                                                                                                                                                                                                                       |
| `/clusters/[id]`             | Single cluster crop grid + DnD + bulk ops + strategy bar (sort / diverse overlay / score chips scoped to this cluster)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `/review`                    | 6 top-level review tabs (2026-09 consolidation, down from 9, plus `new_class_proposals` added 2026-09-24 — see below): **All** / **Uncertainty** / **Model Disagreements** / **COCO Blind Spots** / **New Class Proposals** / one tab per registered queue-capable slot (today: **Plates**, for `license_plate`), each with its own default sort (`review_sorts.py`'s `_TAB_DEFAULTS`) plus the strategy bar's selectable sort/score overlays — the bar's summary chip also shows the server's `sort_applied` next to whatever was requested. The All tab additionally offers a row of **quick-filter preset chips** (VLM mismatches / VLM low-conf / Primary · low-conf) that layer the former Mismatches / Gemma Low-Conf / Primary · Low-Conf tabs' exact cohort queries on top of the All view. Class/Source/Conf filter controls (`class_id`/`source`/`conf_min`/`conf_max`) are live against `GET {API_PREFIX}/review/{tab}`. `/review?crop_id=` deep links resolve via `GET {API_PREFIX}/review/{tab}/locate` — jumps straight to the crop's served page/rank, or shows the backend's `reason` when it isn't in the queue. A slot tab (Plates today) carries provenance chips + VLM-read plate text, driven by the active slot's capabilities rather than a hardcoded `'plates'` check (see "Slot-generic review tabs" below). |
| `/classes`                   | Add / rename / merge classes, per-class hotkey binding, and a **Proposals** section (`GET {API_PREFIX}/review/new_class_proposals/summary`) for creating a class from — or mapping onto an existing class — a VLM-proposed term the registry doesn't have yet, bulk-resolving _every_ pending crop proposing that term (`POST {API_PREFIX}/review/new_class_proposals/resolve`), not just the summary's sample thumbnails.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `/export`                    | Trigger YOLO export, view balance gap, freeze test holdout, download the frozen export's `class_registry.json`/`data.yaml`/`manifest.json` (via `{API_PREFIX}/export/registry/{artifact}`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `/models`                    | Triton model registry browser                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `/train`                     | Training cockpit — preflight, launch, live progress, log tail, past runs, **Promote**, **Reproduce**, **Training cohorts picker** (class-agnostic `CORE_COHORTS` for every class + `license_plate`'s 5 hand-tuned server-side modes — see "Training cohorts" below)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `/bakeoff`                   | LPR model × frozen-dataset bake-off cockpit — scores every selected model against every selected dataset in the on-demand `curation-evaluator` container, renders a model × dataset matrix (best cell per dataset bolded). Gated on backend availability via a one-shot probe of `GET {API_PREFIX}/bakeoff/runs` (`src/lib/bakeoffAvailability.svelte.ts`) — absent, not disabled: the nav link and page body don't render at all when the backend's `{API_PREFIX}/bakeoff/*` router isn't mounted, and no discovery request fires unconditionally on mount. Provisional until the backend ships an `evaluation` axis on `/methods`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `/settings`                  | Deployment-defaults admin page for the shared curation-strategy defaults (`GET,PUT {API_PREFIX}/settings`) — one place to pin the deployment's clustering method, review-queue sort and VLM prompt pack (the latter honored by the always-on VLM labeler and by auto-label runs that don't pick their own), plus a read-only "Set by the backend's startup config" section for `detection_profile` (the backend picks it from startup config; nothing that runs reads a shared or per-run selection). Which axes get a control is decided solely by the server's per-entry `settable` flag on `/methods` (`settableAxes` in `src/lib/curationSettings.ts`). Deployment-wide — see `docs/design/curation-settings-ui-plan-2026-09-21.md` — so it is its own route rather than a `StrategyBar` chip, with an explicit confirm dialog before every save.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

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
- **Uncertainty / Model Disagreements / COCO Blind Spots / Plates**
  are unchanged — still real top-level tabs with their existing default
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
  `coco_blind_spots_default`). Never replaces a tab's default — it's an
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

## Training UI — `/train` (Phase 2 of the training pipeline)

Full cockpit for the training pipeline. The form auto-runs `{API_PREFIX}/train/preflight`
on a 350ms debounce and renders the report inline. Status polls every 5s,
log every 2s, both stop on terminal state. Multi-size campaigns get a
size-chip swap in the submit row (auto-promote-best + `stop_when` threshold).

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
`license_plate`'s 5 region-training-candidate modes
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
  — for `license_plate` today the server already sends all 5 region
  modes, so its hand-declared copy in `licensePlate.ts` never actually
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
`license_plate` slot's keymap — server-served today as `/abdefgmnuxz`)
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
`license_plate`'s `d`/`f`/`e`/`b` — but is what keeps a _second_,
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

| Key           | Action                                                                                                                                                      |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Enter`       | Confirm selected to the chosen class + advance                                                                                                              |
| `Shift+Enter` | Accept all VLM suggestions on the page                                                                                                                      |
| `G`           | Accept the VLM suggestion for selected                                                                                                                      |
| `N`           | Skip + advance                                                                                                                                              |
| `Shift+N`     | Flag selected as needing a new class (curator review)                                                                                                       |
| `D`           | Discard selected (`POST {API_PREFIX}/crops/{id}/discard`, or `discard_batch` for a multi-select)                                                            |
| `Z`           | Undo last label action, including a discard — one press reverts the whole action (bulk label, move, discard, VLM-accept-all), however many crops it touched |
| `X`           | Ignore selected (exclude from training + clustering)                                                                                                        |
| `U`           | Undo last ignore                                                                                                                                            |
| `A`           | Select all on page                                                                                                                                          |
| `←` / `→`     | Move the selection by one crop (not page nav)                                                                                                               |
| `M`           | Move selected to another cluster…                                                                                                                           |
| `Esc`         | Clear drag capture / close picker / clear selection                                                                                                         |

`/review`:

| Key             | Action                                                                                                                                                                               |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Enter`         | Confirm proposed + advance; opens the class picker instead when there's no proposal (plates tab: confirm plate)                                                                      |
| `/`             | Open the fuzzy-search class picker (all non-deprecated classes, not just the top-10 quick-assign row). Not offered on the plates tab.                                                |
| `D`             | Discard — dismiss from every review queue. Reversible via the "Dismissed" panel (`POST {API_PREFIX}/crops/{id}/review_undismiss`), not via Z (plates tab: reject — no plate visible) |
| `N`             | Skip                                                                                                                                                                                 |
| `Z`             | Undo last (one action, however many crops it touched)                                                                                                                                |
| `←` / `→`       | Previous / next item (plates tab `←` / `B`: step back)                                                                                                                               |
| `F`             | Plates tab: mark false positive (box kept)                                                                                                                                           |
| `E`             | Plates tab: enter bbox edit mode                                                                                                                                                     |
| `Enter` / `Esc` | Plates tab, edit mode: save bbox / cancel edit                                                                                                                                       |

### Slot-generic review tabs (P2.8b/P2.8c, docs/genericization-plan-2026-09-13.md §9.5;

panel body generalized by C6, docs/design/slot-generic-crop-mapping-plan-2026-09-21.md §6)

The ~12 `tab === 'plates'` call sites that used to gate the Plates-tab-only
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
SlotReviewTab`), not `'plates'` — `'plates'` survives only as the
`license_plate` slot's `urlId` bookmark value (`tabFromUrlId('plates')`
→ `'slot:license_plate'`, in `src/lib/reviewTabs.ts`).

The inline review panel body (score / status / detector / OCR text /
rejection reason / confirm-reject-FP-back buttons) is now generic too —
this was Finding D, a real gap through 2026-09-21: the tab shell, keymap
and gating were already slot-generic, but the panel body underneath
still read `current.plate_*` fields and a directly-imported
`licensePlateSlot` regardless of the active tab. It now reads every
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

## Data integrity

- Every label change → immediate API call. No "save" button.
- Optimistic UI with error rollback toast on API failure.
- Audit trail lives server-side in the items index (`{label_source,
label_validated, class_source, updated_at}`, plus `class_id_history`).
- Bulk ops show a confirmation dialog with affected count.
- Test-set crops (`test_holdout=true`) are filtered out at the API level —
  the UI never receives them. Don't try to bypass.

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
  parent crop's frame, server-computed. `licensePlateSlot` declares it
  as `subBox.bboxInParentField`; `readSlot` renders from it directly
  when present, falling back to the client-side projection only when
  it isn't (2026-09-24, logic-moves W2).
- `region_detector_chain` — string array, every step of the cascade:
  `['lpr_nanov11_640:miss', 'sam3:hit', 'sam3:vlm_verify_ok']`
- `region_verifier` (`'gemma-4-e4b'` / `'human'`), `region_verified_at`
- `region_text` + `region_text_source` + `region_text_confidence` —
  the VLM reads the plate during verify in the same round-trip
- `region_rejection_reason` — set when the server-side sanity gate
  rejected the candidate
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
  `/review`'s source-image panel reads `current.source`.

There is no shape-plausibility warning: the ⚠ badge (and the client-side
envelope check that computed it — `shapeGate.ts`, `PLATE_SHAPE_ENVELOPE`,
`SlotData.subBox.shapeWarning`) was deleted 2026-09-24 (logic-moves W2)
— the backend never served this flag, and OpenProcessor's own geometry
guard (`is_plausible_region_bbox`) is a different, server-side-only
check with no client mirror.

**New shared components** (renamed off the license-plate-specific names
during the genericization pass — see `docs/genericization-plan-2026-09-13.md`):

- `src/lib/components/ProvenanceChip.svelte` (formerly `DetectorChip.svelte`)
  — color-coded chip (LPR=blue, SAM3=purple, Paddle=amber, Human=green,
  Gemma=teal, v6=rose, YOLO11=sky). Accepts a `raw="lpr_nanov11_640:miss"`
  chain entry directly. Miss/reject tags get a muted variant.
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
  (`getCropImage`), and renders `item_text_lines` with the optional box
  overlay.

**New API helpers:**

- `getPlates(params)` — `{API_PREFIX}/regions` paginated browse with
  detector/verified/score/text filters .
- `getTrainingCandidates(mode, params)` — `{API_PREFIX}/regions/training_candidates`
  with 5 cohort modes.
- `getCropHistory(cropId)` — `GET {API_PREFIX}/crops/{id}/history`, the
  item's class-write history oldest-first (`{crop_id, entries}`).
- `getCropImage(cropId)` — `GET {API_PREFIX}/crops/{id}/image`, the
  shared source image's metadata plus every item cropped from it
  (siblings, mapped through `mapRawCrop` like any other crop list).

## Deployment annotation profiles (tier 2, 2026-09-20)

A deployment configures its own annotation slots by placing
`annotation-profiles.json` next to the built `index.html` — either in
`static/` before a build, or bind-mounted over
`/usr/share/nginx/html/annotation-profiles.json` in the running
container. It is fetched once, in the root layout's `load()`, parsed by
`src/lib/annotations/config/parseSlotConfig.ts`, and merged over the
built-in profiles per-key REPLACE. **Absent or malformed ⇒ built-in
slots only, with a console warning and one toast — never a crash.** The
file is untrusted operator input: template paths, Tailwind ring classes,
regex flags and hotkeys all validate against closed allow-lists in
`src/lib/annotations/config/allowLists.ts`. See
`static/annotation-profiles.example.json` for a worked example and
`docs/annotation-slots-contract-draft.md` §4 for the schema, and
`docs/design/tier2-annotation-profile-config-plan-2026-09-20.md` for the
full design.

`registeredSlots.ts` (referenced earlier in this file, and by
`docs/genericization-plan-2026-09-13.md`) is the **tier-1**, build-time
registration point — "register a new domain by adding a line there" is
still true for a developer with a checkout. It is no longer the _only_
registration point: a deployment operator with no checkout at all
registers a domain via the JSON file above instead, through the same
`resolveSlotRegistry`/merge-by-replace mechanism.

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
`plateGalleryController.svelte.ts` factory-function convention. Test
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
- The `/classes` "Restore" button (deprecated classes table) is
  intentionally disabled — no backend support exists. `deprecated` is
  only ever set `True` (via `POST {API_PREFIX}/classes/merge`); there's no
  un-deprecate endpoint in openprocessor. Restoring also wouldn't reverse
  a prior merge's bulk crop relabel — that's a separate design decision
  (real "undo merge" vs. just un-hiding an empty class), not wired up
  end-to-end yet on either side of the API boundary.
