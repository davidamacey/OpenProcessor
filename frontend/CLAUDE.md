# Cropwright — CLAUDE.md

SvelteKit + TypeScript image-crop annotation web app (product name
**Cropwright**, package name `cropwright`; repo directory is still
`legacy-labeler` pending a physical rename). Generalized via a
capability-model / annotation-slot mechanism (see
`docs/genericization-plan-2026-09-13.md`) so it is no longer
hardcoded to vehicles or license plates — this deployment is
currently configured for legacy v7 vehicle dataset construction via
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
- **Backend**: openprocessor at `http://localhost:4603/op/...` — labeler is a
  pure consumer; no own database
- **State**: Svelte 5 runes (`$state`, `$derived`, `$effect`); nothing in
  localStorage that can't be reconstructed by an API call
- **Build**: SvelteKit static adapter → nginx in production Docker container

## Routes

All routes below are shipped and linked from the top nav (`+layout.svelte`)
except `/` and `/clusters/[id]`, which are reached via the logo / a cluster
card respectively. The MVP/post-MVP split from the original design doc is
gone — every route in this table exists and works; nothing here is a stub.

| Route                        | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/` (legacy, logo link only) | Older stats + recent-crops + quick Gemma-cluster-run page. Superseded by `/dashboard` for nav purposes but still reachable; not deleted since it's a working page, just not the primary entry point.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `/dashboard`                 | Current pipeline dashboard — live `DatasetStats` (polls every 10s) + `AutoLabelPanel` ("Run Clustering Now" with stage progress), shared with the daemon-fired auto-label run. `AutoLabelPanel` also hosts an optional per-class assist scope (`AssistScopeBar`, absent unless `/methods` advertises the `detection_profile`/`prompt_pack` axes — see "Curation-strategy selector bar" below) that lets an operator point the VLM-assisted sweep at a single class instead of the whole pool.                                                                                                                                                                                                                                                                                                                                                               |
| `/clusters`                  | Cluster grid view, sidebar filter, **strategy bar** (review-sort dropdown + score chips, see below — no cluster-method picker here; that lives on `/settings`). When `class=license_plate` is selected, replaces the cluster grid with a **plate-thumbnail grid** backed by `/curation/plates` (detector / verified / score / plate-text filters; click → jump to the license_plate slot's review tab). Also hosts the **embedding-plot** overlay toggle when `viz_projection` is available (see below).                                                                                                                                                                                                                                                                                                                                                          |
| `/clusters/[id]`             | Single cluster crop grid + DnD + bulk ops + strategy bar (sort / diverse overlay / score chips scoped to this cluster)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `/review`                    | 5 top-level review tabs (2026-09 consolidation, down from 9 — see below): **All** / **Uncertainty** / **Model Disagreements** / **COCO Blind Spots** / one tab per registered queue-capable slot (today: **Plates**, for `license_plate`), each with its own default sort (`review_sorts.py`'s `_TAB_DEFAULTS`) plus the strategy bar's selectable sort/score overlays. The All tab additionally offers a row of **quick-filter preset chips** (Gemma mismatches / Gemma low-conf / Primary · low-conf) that layer the former Mismatches / Gemma Low-Conf / Primary · Low-Conf tabs' exact cohort queries on top of the All view. A slot tab (Plates today) carries provenance chips + Gemma-OCR'd plate text + ⚠ shape warnings, driven by the active slot's capabilities rather than a hardcoded `'plates'` check (see "Slot-generic review tabs" below). |
| `/classes`                   | Add / rename / merge classes, per-class hotkey binding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `/export`                    | Trigger YOLO export, view balance gap, freeze test holdout, download the frozen export's `class_registry.json`/`data.yaml`/`manifest.json` (via `/curation/export/registry/{artifact}`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `/models`                    | Triton model registry browser                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `/train`                     | Training cockpit — preflight, launch, live progress, log tail, past runs, **Promote**, **Reproduce**, **Training cohorts picker** (class-agnostic `CORE_COHORTS` for every class + `license_plate`'s 5 hand-tuned server-side modes — see "Training cohorts" below)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `/bakeoff`                   | LPR model × frozen-dataset bake-off cockpit — scores every selected model against every selected dataset in the on-demand `legacy-evaluator` container, renders a model × dataset matrix (best cell per dataset bolded). Gated on backend availability via a one-shot probe of `GET {API_PREFIX}/bakeoff/runs` (`src/lib/bakeoffAvailability.svelte.ts`) — absent, not disabled: the nav link and page body don't render at all when the backend's `/curation/bakeoff/*` router isn't mounted, and no discovery request fires unconditionally on mount. Provisional until the backend ships an `evaluation` axis on `/methods`.                                                                                                                                                                                                                                  |
| `/settings`                  | Deployment-defaults admin page for the shared curation-strategy defaults (`GET,PUT {API_PREFIX}/settings`) — one place to pin the deployment's clustering method and review-queue sort, plus a read-only "Advertised but not yet wired" section for the `detection_profile`/`prompt_pack` axes (see "Curation-strategy selector bar" below for why those two are display-only). Deployment-wide and, for `sort`, irreversible — see `docs/design/curation-settings-ui-plan-2026-09-21.md` — so it is its own route rather than a `StrategyBar` chip, with an explicit confirm dialog before every save.                                                                                                                                                                                                                                                     |

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
  backend `/curation/review/outliers` query is untouched/unlinked, not
  deleted (out of scope for a frontend-only change).
- **Mismatches / Gemma Low-Conf / Primary · Low-Conf** collapsed from
  top-level tabs into **quick-filter preset chips** shown only on the
  `all` tab (`REVIEW_PRESETS` in `src/lib/reviewTabs.ts`). Each chip
  reuses that former tab's exact backend cohort query unchanged — same
  `GET /curation/review/{id}` endpoint, same default sort, same
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
  `mistakenness`, `uniqueness`, `plate_score`, `disagreement_entropy_asc`,
  plus each tab's own legacy default (`primary_low_conf_default`,
  `coco_blind_spots_default`). Never replaces a tab's default — it's an
  additional option layered on top.
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
  come from a cached, batch-computed projection (`GET /curation/viz/projection`)
  — never fit on the request path — with an explicit "Rebuild" action
  (`POST /curation/viz/projection/rebuild`, a background job). Click-and-drag
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
`startAutoLabel()`'s `class_id`/`detection_profile`/`prompt_pack` — different
domains, same interaction shape. It reads two new `/methods` axes,
`detection_profile` and `prompt_pack` (parsed into
`detection_profiles`/`prompt_packs`, gated by
`isDetectionProfileAvailable`/`isPromptPackAvailable`/`isScopedAssistAvailable`
in `strategies.ts`), and is absent — not disabled — until at least one is
advertised, because an unknown query param on the pipeline start call is
silently dropped server-side rather than 404ing. See
`docs/design/vlm-scoped-labeling-assist-plan-2026-09-20.md` for the full
contract (agreed with the backend session, not yet landed there).

Every one of these degrades gracefully to invisible/default when its
backend flag is off or `/curation/methods` fails: `strategiesStore` falls back
to `FALLBACK_METHODS` (`strategies.ts`) — the hardcoded stable-only list
matching what's always been implemented — so a missing endpoint never
breaks page load. Flags live in openprocessor's `.env.legacy.example`
(`OP_SCORES_ENABLED`, `OP_SCORES_SHADOW`, `OP_SELECT_DIVERSE_ENABLED`,
`OP_VIZ_PROJECTION_ENABLED`), all default off.

## Training UI — `/train` (Phase 2 of legacy_train_pipeline)

Full cockpit for the training pipeline. The form auto-runs `/curation/train/preflight`
on a 350ms debounce and renders the report inline. Status polls every 5s,
log every 2s, both stop on terminal state. Multi-size campaigns get a
size-chip swap in the submit row (auto-promote-best + `stop_when` threshold).

Past-runs table actions:

- **Promote ↑** — opens `PromoteModal` (Triton model name, max_batch_size,
  fp16, overwrite, force-bypass-gate). 422 with the gate report renders
  inline; `force=true` bypasses for known-good experimental runs.
- **Reproduce** — fetches `/curation/train/manifest/{job_id}` and submits a
  fresh job with the same `spec`/`lineage`. Phase 6 polish — design §15.4.

After a successful promote, the user re-runs `/curation/pipeline/auto_label` and
the **Model Disagreements** tab on `/review` surfaces validated crops
where the new model and the human label diverge — high-signal candidates
for the next training cycle.

### Training cohorts (P2.12-P2.14, docs/genericization-plan-2026-09-13.md §9.2/§9.3)

The former "Plate training cohorts" panel (hardcoded to `license_plate`'s
4 backend modes) is now a generic **Training cohorts** section, grouped
by class. Mechanism lives in `src/lib/annotations/cohorts.ts`:

- **`CORE_COHORTS`** — 4 class-agnostic cohorts (`validated` /
  `needs_labeling` / `low_confidence` / `model_disagreements`) every
  class gets for free, riding entirely on the already class-agnostic
  `GET /curation/crops` and `GET /curation/review/model_disagreements` — zero
  backend change.
- **`SlotSpec.capabilities.trainingCohorts`** (optional) — a slot's own
  hand-declared cohorts, which replace any derived id of the same name.
  `licensePlateSlot` declares its 5 backend `/curation/plates/
training_candidates` modes here (`lpr_blind_spots` /
  `lpr_low_conf_correct` / `disagreement` / `human_corrected` /
  `false_positives` — the 5th mode was previously typed but
  unreachable from the UI; it's live now).
- **`derivedCohorts()`** — capability ⇒ cohort rules (subBox ⇒
  `blind_spots`, subBox+scoreField ⇒ `low_conf`, provenance.chainField
  ⇒ `disagreement`, lifecycle.falsePositiveState ⇒ `false_positives`).
  Tier-2 `predicate` queries, gated behind a `predicateCohortsAvailable`
  flag that is hardcoded `false` today (no backend support exists yet —
  see H5 in the plan's §9.4) — so only `CORE_COHORTS` + a slot's own
  declared cohorts ever render in this deployment.
- `/train/+page.svelte`'s `runCohortQuery()` dispatches a cohort's
  compiled tier-1 endpoint query to whichever existing `api.ts`
  function already answers it (`getTrainingCandidates` /
  `getCrops` / `getReviewQueue`) — no new endpoint, no new response
  shape. The preview grid is `rowKind`-aware (`SlotCard` for a slot
  cohort, `CropCard` for a class-agnostic one). Counts load lazily
  per class group (`IntersectionObserver`) rather than firing an
  N-classes × M-cohorts request storm on mount.
- Cohorts default to **all non-deprecated classes** rather than syncing
  to `TrainForm`'s own class-subset selection — that tighter coupling
  (lifting `selectedClasses` into the page) is not done; the cohort
  picker is a curation-preview surface only and never filters the
  actual training run (`TrainJobSpec.include_classes`/`single_cls` are
  untouched, and no cohort id is ever sent to `/curation/train/start`).

## Keyboard shortcuts

Class assignment is per-class `hotkey_letter`, configured on `/classes` (or in
the `~` overlay) and routed through `dropOnClassStore` by the layout-level
keydown listener in `src/routes/+layout.svelte`. There is no `1-9, 0` top-N
scheme — it was removed; one binding scheme means no "what does this key do
here?" friction. On `/clusters/[id]` a class letter labels the current
selection (or the just-dragged set); on `/review` it labels the current item.

Reserved single-char action keys (`g n d z x u a m /`) cannot be bound to a
class — `setClassHotkey` (`src/lib/classHotkey.ts`) rejects them, validating
against `reservedHotkeyLetters()` rather than the bare `RESERVED_HOTKEY_LETTERS`
constant. `reservedHotkeyLetters(registry)` is that constant **∪ every
single-character combo any registered queue-capable slot's
`QueueCapability.keymap` declares** (P2.8c, closing Finding C.2
structurally) — today that adds `d`/`f`/`e`/`b` from `license_plate`'s
keymap (`d` was already reserved). A slot tab already suppresses class-drop
registration entirely (`isSlotSuppressedTab`/`isSlotTab`), so this is
defense in depth, not a fix for a live collision — it's what keeps a
_second_ capable slot's letters safe without a human re-auditing every
class hotkey. Both window keydown listeners fire on the same keypress, so
a class bound to `d` would be assigned _and_ the selection discarded.

Global:

| Key            | Action                                       |
| -------------- | -------------------------------------------- |
| `` ` `` / `~`  | Toggle keyboard shortcut overlay             |
| `Esc`          | Close the overlay                            |
| _class letter_ | Assign that class (selection / current item) |

`/clusters/[id]`:

| Key           | Action                                                |
| ------------- | ----------------------------------------------------- |
| `Enter`       | Confirm selected to the chosen class + advance        |
| `Shift+Enter` | Accept all Gemma suggestions on the page              |
| `G`           | Accept Gemma suggestion for selected                  |
| `N`           | Skip + advance                                        |
| `Shift+N`     | Flag selected as needing a new class (curator review) |
| `D`           | Discard (unlabel) selected                            |
| `Z`           | Undo last label action                                |
| `X`           | Ignore selected (exclude from training + clustering)  |
| `U`           | Undo last ignore                                      |
| `A`           | Select all on page                                    |
| `←` / `→`     | Move the selection by one crop (not page nav)         |
| `M`           | Move selected to another cluster…                     |
| `Esc`         | Clear drag capture / close picker / clear selection   |

`/review`:

| Key             | Action                                                                                                                                |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `Enter`         | Confirm proposed + advance; opens the class picker instead when there's no proposal (plates tab: confirm plate)                       |
| `/`             | Open the fuzzy-search class picker (all non-deprecated classes, not just the top-10 quick-assign row). Not offered on the plates tab. |
| `D`             | Discard — dismiss from every review queue, **permanent** (plates tab: reject — no plate visible)                                      |
| `N`             | Skip                                                                                                                                  |
| `Z`             | Undo last                                                                                                                             |
| `←` / `→`       | Previous / next item (plates tab `←` / `B`: step back)                                                                                |
| `F`             | Plates tab: mark false positive (box kept)                                                                                            |
| `E`             | Plates tab: enter bbox edit mode                                                                                                      |
| `Enter` / `Esc` | Plates tab, edit mode: save bbox / cancel edit                                                                                        |

### Slot-generic review tabs (P2.8b/P2.8c, docs/genericization-plan-2026-09-13.md §9.5)

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
→ `'slot:license_plate'`, in `src/lib/reviewTabs.ts`). A second
queue-capable slot registered in `registeredSlots.ts` gets a fully
working review tab (keymap, hint strip, inline review panel gating)
with zero further edits to `review/+page.svelte` — proved by
`src/lib/annotations/secondSlotIntegration.test.ts`.

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
- Audit trail lives server-side in `op_vehicle_crops.{label_source,
label_validated, class_source, updated_at}`.
- Bulk ops show a confirmation dialog with affected count.
- Test-set crops (`test_holdout=true`) are filtered out at the API level —
  the UI never receives them. Don't try to bypass.

## Plate provenance + OCR (Wave 1 + Wave 2b, 2026-05-11)

Every plate-bearing crop now carries detector provenance — which
model produced the bbox, plus the Gemma-read plate text. These flow
through `mapRawCrop` (`src/lib/api.ts`) onto the `OpCrop` type and
render via the shared chip components.

**New fields on `OpCrop` / `ReviewItem`:**

- `plate_detector` (`'lpr_nanov11_640'` / `'sam3'` /
  `'paddleocr_det_trt'` / `'human'`)
- `plate_detector_version`, `plate_bbox_frame` (always `'source'`)
- `plate_detector_chain` — string array, every step of the cascade:
  `['lpr_nanov11_640:miss', 'sam3:hit', 'sam3:gemma_verify_ok']`
- `plate_verifier` (`'gemma-4-e4b'` / `'human'`), `plate_verified_at`
- `plate_text` + `plate_text_source` + `plate_text_confidence` —
  Gemma reads the plate during verify in the same round-trip
- `plate_rejection_reason` — set when the server-side sanity gate
  rejected the candidate
- `plate_shape_warning` — client-side computed via the same envelope
  as `is_plausible_plate_bbox` in `plate_detect.py`

**New shared components** (renamed off the license-plate-specific names
during the genericization pass — see `docs/genericization-plan-2026-09-13.md`):

- `src/lib/components/ProvenanceChip.svelte` (formerly `DetectorChip.svelte`)
  — color-coded chip (LPR=blue, SAM3=purple, Paddle=amber, Human=green,
  Gemma=teal, v6=rose, YOLO11=sky). Accepts a `raw="lpr_nanov11_640:miss"`
  chain entry directly. Miss/reject tags get a muted variant.
- `src/lib/components/SlotCard.svelte` (formerly `PlateCard.svelte`) —
  128px annotation-slot thumbnail (via `{API_PREFIX}/crops/{id}/region_thumbnail`),
  parent class chip, score, provenance chip strip, slot text inline,
  ⚠ shape warning. Parameterized via `readSlot` (`src/lib/annotations/
readSlot.ts`) against a `SlotSpec` (`registeredSlots.ts`) rather than
  hardcoded plate fields, so a second registered slot renders through
  the same component.

**New API helpers:**

- `getPlates(params)` — `/curation/plates` paginated browse with
  detector/verified/score/text filters.
- `getTrainingCandidates(mode, params)` — `/curation/plates/training_candidates`
  with 4 cohort modes.

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

The production build is consumed by a `nginx:alpine` container declared in
`openprocessor/docker-compose.legacy.yml` on port 5184 (host). Port conflicts:
5174=example-app-backend, 5180/5181=example-app-opensearch, 5183=example-app-docs.

## Connecting to openprocessor

`PUBLIC_TRITON_API_URL` env var (default: empty string in production Docker).
When empty, the nginx container proxies `/curation/*` and `/clusters/*` to
`http://op-api:4603` — no CORS, no hardcoded IPs, works on any LAN client.
Set `PUBLIC_TRITON_API_URL=http://<host>:4603` only when pointing at a remote
openprocessor on a different machine. All API calls flow through `src/lib/api.ts`
with retry + AbortController for in-flight cancellation.

## Deployment

openprocessor serves all images and crop thumbnails — the labeler does NOT mount
NAS volumes. This avoids volume duplication and keeps NAS path knowledge in
one place. Image URLs look like `${PUBLIC_TRITON_API_URL}/curation/crops/{id}/thumbnail`.

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
- Test-holdout crops MUST never be relabeled by Gemma or via cluster
  auto-suggest. The openprocessor `/curation/` endpoints filter; UI is the second line.
- The `/classes` "Restore" button (deprecated classes table) is
  intentionally disabled — no backend support exists. `deprecated` is
  only ever set `True` (via `POST /curation/classes/merge`); there's no
  un-deprecate endpoint in openprocessor. Restoring also wouldn't reverse
  a prior merge's bulk crop relabel — that's a separate design decision
  (real "undo merge" vs. just un-hiding an empty class), not wired up
  end-to-end yet on either side of the API boundary.
