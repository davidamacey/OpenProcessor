# legacy-labeler — CLAUDE.md

SvelteKit + TypeScript labeling web app for legacy v7 vehicle dataset
construction. Sister project to `legacy_sorter` (v2 Tauri app for the actual
sort UX) and `openprocessor` (server-side inference + OpenSearch + clustering).

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

## Routes (post-MVP)

| Route            | Purpose                                                                                                                                                                                                                                              | MVP?     |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `/`              | Dashboard (class balance, ingestion stats)                                                                                                                                                                                                           | yes      |
| `/clusters`      | Cluster grid view, sidebar filter. When `class=license_plate` is selected, replaces the cluster grid with a **plate-thumbnail grid** backed by `/curation/plates` (detector / verified / score / plate-text filters; click → jump to `/review?tab=plates`) | yes      |
| `/clusters/[id]` | Single cluster crop grid + DnD + bulk ops                                                                                                                                                                                                            | yes      |
| `/review`        | Mismatch / Gemma low-conf / Outlier / Uncertainty / **Model Disagreements** / **Plates** review queues. Plates tab carries provenance chips + Gemma-OCR'd plate text + ⚠ shape warnings                                                              | yes      |
| `/classes`       | Add / rename / merge classes                                                                                                                                                                                                                         | post-MVP |
| `/export`        | Trigger YOLO export, view balance gap                                                                                                                                                                                                                | post-MVP |
| `/models`        | Triton model registry browser                                                                                                                                                                                                                        | post-MVP |
| `/train`         | Training cockpit — preflight, launch, live progress, log tail, past runs, **Promote**, **Reproduce**, **Plate training cohorts picker** (4 modes: lpr_blind_spots / lpr_low_conf_correct / disagreement / human_corrected)                           | post-MVP |

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

## Keyboard shortcuts

Class assignment is per-class `hotkey_letter`, configured on `/classes` (or in
the `~` overlay) and routed through `dropOnClassStore` by the layout-level
keydown listener in `src/routes/+layout.svelte`. There is no `1-9, 0` top-N
scheme — it was removed; one binding scheme means no "what does this key do
here?" friction. On `/clusters/[id]` a class letter labels the current
selection (or the just-dragged set); on `/review` it labels the current item.

Reserved single-char action keys (`g n d z x u a m`) cannot be bound to a
class — `setClassHotkey` (`src/lib/classHotkey.ts`) rejects them. Both window
keydown listeners fire on the same keypress, so a class bound to `d` would be
assigned _and_ the selection discarded.

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

| Key             | Action                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------ |
| `Enter`         | Confirm proposed + advance (plates tab: confirm plate)                                           |
| `D`             | Discard — dismiss from every review queue, **permanent** (plates tab: reject — no plate visible) |
| `N`             | Skip                                                                                             |
| `Z`             | Undo last                                                                                        |
| `←` / `→`       | Previous / next item (plates tab `←` / `B`: step back)                                           |
| `F`             | Plates tab: mark false positive (box kept)                                                       |
| `E`             | Plates tab: enter bbox edit mode                                                                 |
| `Enter` / `Esc` | Plates tab, edit mode: save bbox / cancel edit                                                   |

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

**New shared components:**

- `src/lib/components/DetectorChip.svelte` — color-coded chip
  (LPR=blue, SAM3=purple, Paddle=amber, Human=green, Gemma=teal,
  v6=rose, YOLO11=sky). Accepts a `raw="lpr_nanov11_640:miss"`
  chain entry directly. Miss/reject tags get a muted variant.
- `src/lib/components/PlateCard.svelte` — 128px plate thumbnail
  (via `/curation/crops/{id}/plate_thumbnail`), parent class chip, score,
  detector chip strip, plate text inline, ⚠ shape warning.

**New API helpers:**

- `getPlates(params)` — `/curation/plates` paginated browse with
  detector/verified/score/text filters.
- `getTrainingCandidates(mode, params)` — `/curation/plates/training_candidates`
  with 4 cohort modes.

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

## Known constraints

- HTML5 DnD doesn't work reliably across all browsers/webviews. Use
  `svelte-dnd-action` (pointer-based) only.
- Don't fetch full-resolution NAS images in grids. Always use the thumbnail
  endpoint (128×128 LRU cached server-side).
- Test-holdout crops MUST never be relabeled by Gemma or via cluster
  auto-suggest. The openprocessor `/curation/` endpoints filter; UI is the second line.
