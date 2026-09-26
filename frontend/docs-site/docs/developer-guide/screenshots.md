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
