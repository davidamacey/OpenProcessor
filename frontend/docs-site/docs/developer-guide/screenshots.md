---
sidebar_position: 4
title: Screenshots
---

# Capturing documentation screenshots

Every screenshot in these docs (the homepage showcase and any
`<Screenshot>` slot in a doc page) must come from a Cropwright instance
pointed at an OpenProcessor backend holding **public sample data only** —
COCO val2017 (`make sample-coco` in the OpenProcessor checkout) or the
Open Images "Vehicle registration plate" subset (`make sample-plates`).

**Never** capture from, or commit an image sourced from, a real deployment
— its imagery, class names, and counts are not public data.

## Image credits

The screenshots in these docs show photographs from the
[COCO](https://cocodataset.org) val2017 set and the
[Open Images](https://storage.googleapis.com/openimages/web/index.html)
"Vehicle registration plate" subset. COCO images are from Flickr under
their owners' Creative Commons licenses; Open Images images are listed by
their authors as CC BY 2.0. Annotations and boxes shown are produced by
Cropwright and OpenProcessor, not by either dataset.

The state screenshots (projects, combine, prompt packs, reprocess, models)
show only photographs from COCO: T.-Y. Lin et al., "Microsoft COCO: Common
Objects in Context", ECCV 2014 (https://cocodataset.org). COCO annotations
are CC BY 4.0; each image remains under its Flickr owner's Creative Commons
license (per-image license and author are in the COCO annotation files).

## Capture script

`scripts/capture_docs_screenshots.py` (repo root) drives a real browser
against a running Cropwright instance and writes full-page PNGs at 1600px
and 800px wide into `docs-site/static/img/screenshots/`. Only the 1600px
images are committed; the 800px ones are for checking the narrow layout.
The capture is read-only: every request other than GET/HEAD is aborted,
except the small explicit allow-list below.

```bash
# once, to provision the Playwright venv used by the e2e suite
npm run test:e2e

e2e/.venv/bin/python scripts/capture_docs_screenshots.py \
  --base-url http://localhost:<port-of-your-public-data-instance> \
  --project <public-sample-project-slug>
```

`--project` prefixes the per-project routes with `/p/<slug>` (the bare paths
would otherwise open the default, empty, project). `--only <name>...` captures
just those routes or states; the scripted states (a dialog opened, a crop
selected, a pack test run) live in the script's `STATES` and are captured at
1600px only. A state only opens and looks: nothing is saved or confirmed.

### Allow-listed side-effect-free calls

Besides GET/HEAD the guard (`READ_ONLY_CALLS` in the script) lets through
exactly these calls, each of which writes nothing server-side:

| Call | Why |
| --- | --- |
| `POST .../train/preflight` | preflight report |
| `DELETE /curation/projects/{project}?dry_run=true` (never with `confirm`) | the delete dialog's dry-run report |
| `POST /projects/combine/preview` | combine preview report |
| `POST .../reprocess` with `dry_run: true` | Reprocess "Check what would run" |
| `POST .../prompt_packs/validate` | live pack validation |
| `POST .../prompt_packs/test` | "Test on a crop" (runs the VLM, "Nothing is written") |
| `POST .../region_profiles/validate`, `.../keymap/validate`, `.../vlm/endpoints/validate` | validation reports |
| `POST .../ingest/policy/preview` | the ingest policy cost preview (a report over stored detections) |
| `POST .../region_profiles/validate_segmenter_prompt` | "Check segmenter prompt" (a text-only check) |
| `POST .../open_vocab/validate`, `.../open_vocab/test` | open-vocabulary validation and "Test a target" (runs the segmenter on one stored image, writes nothing) |

### Fixtures that need a write

A few slots show a state that only exists after a write (a paused project, a
combine job, an editable prompt pack or region profile, a running job that
blocks a delete). `scripts/docs_screenshot_fixtures.py setup|teardown` creates
and removes throwaway projects whose slug starts with `cwlife-`; every write
asserts that prefix first, and `teardown` waits for the asynchronous delete.
The browser stays read-only throughout.

The exact list of routes/states to capture is **not hardcoded in the
script** — it reads `docs-site/src/data/screenshot_routes.json`. Add a route there
when a doc page gains a new `<Screenshot name="...">` slot.

## Preconditions

1. `<repo>/OpenProcessor` (or another reachable checkout) running with
   `make sample-coco` (and, for the region example, `make sample-plates`)
   already ingested through `/ingest`.
2. Cropwright pointed at that backend and reachable at the URL passed to
   `--base-url`.
3. Enough clusters/reviewed items exist that the pages being captured
   aren't showing empty states (unless an empty state is the thing being
   documented).

Run the script, review each PNG for anything that leaked (hostnames,
paths, unexpected text), then commit them under
`docs-site/static/img/screenshots/`.

## Slots for the project, import, configuration and VLM pages

The import, multi-box and region-profile-test slots come from the `sample-coco-import` and `sample-coco-vehicles` projects (`IMPORT_PREVIEW_PATH` names a server path under an allowed source root, e.g. the project's uploaded archive). The region-profile test reads "not eligible" because the sample items carry no vehicle parent class; the candidates are still drawn. Every slot has its image; a slot added later renders as "pending" until it is captured. Capture them against a backend holding public sample
data (never from a real deployment). Rows marked **route** are in
`screenshot_routes.json` and the script captures them as-is. Rows marked
**state** need the page put into a particular state first (a dialog open, an
item selected) and, because the script aborts every non-GET request, either
a manual capture or a small extension that drives the clicks read-only.

| File (`-1600.png`) | Page | Kind | What it must show |
| --- | --- | --- | --- |
| `projects` | `/projects` | route | List, capacity banner, row actions, a paused chip |
| `projects-delete-dry-run` | `/projects` | state | Delete dialog showing the dry-run report and a blocking reason |
| `projects-copy-settings` | `/projects` | state | Copy settings dialog with the cloneable groups |
| `combine-wizard` | `/projects/combine` | route | Ordered sources, class mapping, preview |
| `combine-job` | `/projects/combine/<job>` | state | A running or completed job with next steps |
| `import-wizard` | `/datasets/import` | state | A previewed import with the class-mapping table |
| `import-job` | `/datasets/imports/<id>` | state | A job with progress, report and the Undo dry run |
| `reprocess-dialog` | an item's Details | state | Scopes and the locked-and-skipped counts |
| `review-imported` | `/review?tab=imported&import_id=<id>` | state | Imported tab with an import chip |
| `review-regions-multibox` | `/review?tab=regions` | state | One item with several boxes in different states |
| `box-editor` | `/review?tab=regions` (edit mode) | state | A selected box, Add box and the "N / max" counter |
| `prompt-pack-editor` | `/settings/prompt-packs/<name>` | state | Grouped fields with a validation issue |
| `prompt-pack-test` | same, Test on a crop | state | Prompt, raw reply and parsed answer |
| `region-profile-editor` | `/settings/region-profiles/<name>` | state | Typed fields and model pickers |
| `region-profile-test` | same, Test on a crop | state | Candidates drawn over the source image |
| `settings-models` | `/settings/models` | route | Active endpoint, registry table, local model panel |
| `vlm-endpoint-editor` | `/settings/models/vlm/<name>` | state | A key reference and "host has it", no key value |
| `vlm-run-picker` | the dashboard assist bar | state | Per-run VLM picker with its options listed (project default, served endpoints, off) |
| `settings-keymap` | `/settings#keyboard` | route | The keyboard shortcut editor with verb groups |
| `models-sharing` | `/models` | state | The owner's model marked "Shared with other projects" with Stop sharing |
| `models-unshare-force` | `/models` | state | The 409 in-use list (project and profile) and the Unshare anyway button, never armed |

The routes in `screenshot_routes.json` use bare paths, which redirect to the
default project; the script follows the redirect. Check each capture for
anything private: project slugs, host names, secret reference names and file
paths must be sample-data values.

### Notes on the sharing and picker slots

The three slots come from the throwaway `model-demo-owner` /
`model-demo-consumer` projects (a shared model used by the consumer's active
profile). `models-unshare-force` is the one capture that sends a write: the
plain unshare PUT, which the server must refuse with 409 `in_use`. The script
allows only that URL, only without `force`, and restores the share and fails
if the answer is anything but 409; "Unshare anyway" is never armed.
`vlm-run-picker` shows the select's options inline (display only) and starts
no run. Only the built-in endpoint exists, so the external-images
acknowledgement is not shown. The consumer's own `/models` view (the "from
project" chip and class mapping) is not captured: the backend lists the shared
model twice there and the page stays on "Loading...".

`vlm-endpoint-editor` shows the built-in endpoint's key reference with no key
value; the "host has it" state needs a key file on the host, which the sample
stack does not have.

## Slots for the v0.4.0 features

These sixteen slots show the final v0.4.0 backend. They are scripted states
(`--only <name>`) in `scripts/capture_docs_screenshots.py`, 1600px only (none of
these layouts is narrow-sensitive). Two environment variables pick the projects:
`IMPORT_PROJECT=sample-coco-import-v2` and `LIFE_PROJECT=<the throwaway cwlife- project>`.

| File (`-1600.png`) | Project | What it shows |
| --- | --- | --- |
| `resources-menu` | `sample-coco-2k-v3` | The open Resources menu: bundled docs plus the served service links |
| `wheels-inventory-card` | throwaway | The pinned Wheels inventory card on `/clusters` |
| `region-gallery-boxes` | throwaway | The gallery toolbar's "N / M boxes listed (K items)" |
| `import-job-actions` | `sample-coco-import-v2` | A completed import: Undo offered, Cancel and Resume with the served reason |
| `reprocess-served-scopes` | `sample-coco-import-v2` | The Reprocess dialog's served scope labels and the dry-run counts (dry run only, never applied) |
| `locked-item-badge` | `sample-coco-import-v2` | The lock glyph on imported items |
| `region-profile-testable` | throwaway | "Test on a crop": each leg's served status and reason, candidates drawn |
| `segmenter-prompt-check` | throwaway | The result of "Check segmenter prompt" |
| `clone-from-project` | `sample-coco-2k-v3` | The clone dialog's "Copy from another project" list (never submitted) |
| `ingest-policy-preview` | throwaway | The ingest policy editor in `selected` mode with the served cost preview |
| `open-vocab-editor` | throwaway | The open-vocabulary editor with three targets |
| `open-vocab-test` | throwaway | "Test a target": hits, scores and dropped reasons |
| `region-stage-panel` | throwaway | The Region stage panel (cropped to its card) |
| `dashboard-embedding` | throwaway | The dashboard Embedding card with a `not_selected` count |
| `crop-embedding-row` | throwaway | A crop's detail panel with the Embedding row |
| `run-results-confusion` | `model-demo-owner` | A finished run's Results with the confusion matrix image |

Notes on how they were made:

- **Throwaway project.** The wheel profile, the open-vocabulary set, the ingest
  policy and the un-embedded detections only exist after writes, so they live in
  one project `cwlife-<epoch>-x` created for the capture and deleted afterwards
  (dry run first, then `confirm=<slug>`). It holds 72 COCO vehicle photos
  uploaded through `/ingest/upload` (the bytes were read from the public
  `sample-coco-vehicles-v2` project), a `vehicle_wheel` profile cloned from the
  served template and activated, the detector's classes seeded, an ingest policy
  in `selected` mode for `car` and `truck`, and an open-vocabulary set
  `vehicle_parts` (wheel, side mirror, headlight). All writes went to that
  project only; the browser stayed read-only.
- **Clone picker.** The native select is shown open by sizing it to its options.
  Before the shot the capture removes, from that page only, the options for projects
  that are not part of the public sample set; the served list is unchanged.
- **Lock tooltip.** The lock glyph's tooltip is a native `title`, which a page
  screenshot cannot contain; the caption says so. The tooltip text is the served
  lock rule from `GET /config/vocabulary`.
- **Not captured.** The Resources menu's "not configured" row and "not running"
  note: the sample stack serves every link as configured and reachable, and
  faking a served state in the browser would not be a real capture.
- Images that show a whole panel (`region-stage-panel`, `open-vocab-test`,
  `region-profile-testable`, `segmenter-prompt-check`) are element crops so that
  no server path or throwaway slug appears in them.
