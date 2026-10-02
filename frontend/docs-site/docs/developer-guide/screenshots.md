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

## Capture script

`scripts/capture_docs_screenshots.py` (repo root) drives a real browser
against a running Cropwright instance and writes full-page PNGs at 1600px
and 800px wide into `docs-site/static/img/screenshots/`. Only the 1600px
images are committed; the 800px ones are for checking the narrow layout.
The capture is read-only: every request other than GET/HEAD (and the
side-effect-free `/train/preflight` report) is aborted.

```bash
# once, to provision the Playwright venv used by the e2e suite
npm run test:e2e

e2e/.venv/bin/python scripts/capture_docs_screenshots.py \
  --base-url http://localhost:<port-of-your-public-data-instance>
```

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

## Pending slots for the project, import, configuration and VLM pages

These `<Screenshot>` slots are in the docs but have no image yet, so they
render as "pending". Capture them against a backend holding public sample
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
| `import-wizard` | `/datasets/import` | route | A previewed import with the class-mapping table |
| `import-job` | `/datasets/imports/<id>` | state | A job with progress, report and the Undo dry run |
| `reprocess-dialog` | an item's Details | state | Scopes and the locked-and-skipped counts |
| `review-imported` | `/review?tab=imported` | route | Imported tab with an import chip |
| `review-regions-multibox` | `/review?tab=regions` | state | One item with several boxes in different states |
| `box-editor` | `/review?tab=regions` (edit mode) | state | A selected box, Add box and the "N / max" counter |
| `prompt-pack-editor` | `/settings/prompt-packs/<name>` | state | Grouped fields with a validation issue |
| `prompt-pack-test` | same, Test on a crop | state | Prompt, raw reply and parsed answer |
| `region-profile-editor` | `/settings/region-profiles/<name>` | state | Typed fields and model pickers |
| `region-profile-test` | same, Test on a crop | state | Candidates drawn over the source image |
| `settings-models` | `/settings/models` | route | Active endpoint, registry table, local model panel |
| `vlm-endpoint-editor` | `/settings/models/vlm/<name>` | state | A key reference and "host has it", no key value |
| `vlm-run-picker` | the dashboard assist bar | state | Per-run picker with the external-images acknowledgement |
| `settings-keymap` | `/settings#keyboard` | route | The keyboard shortcut editor with verb groups |
| `models-sharing` | `/models` | state | Another project's shared model and its class mapping |
| `models-unshare-force` | `/models` | state | The in-use list and Unshare anyway |

The routes in `screenshot_routes.json` use bare paths, which redirect to the
default project; the script follows the redirect. Check each capture for
anything private: project slugs, host names, secret reference names and file
paths must be sample-data values.
