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
— its imagery, class names, and counts are not public data. This is why no
screenshot ships with this initial docs-site build: the shared development
stack currently holds non-public imagery.

## Capture script

`scripts/capture_docs_screenshots.py` (repo root) drives a real browser
against a running Cropwright instance and writes full-page PNGs at 1600px
and 800px wide into `docs-site/static/img/screenshots/`.

```bash
# once, to provision the Playwright venv used by the e2e suite
npm run test:e2e

e2e/.venv/bin/python scripts/capture_docs_screenshots.py \
  --base-url http://localhost:5184
# or: CROPWRIGHT_URL=http://localhost:5184 e2e/.venv/bin/python scripts/capture_docs_screenshots.py
```

The exact list of routes/states to capture is **not hardcoded in the
script** — it reads `docs-site/src/data/screenshots.json` (the same file
`ScreenshotShowcase` renders from) plus each doc page's own `<Screenshot
name="...">` references, so adding a new screenshot slot to a doc page is
the only step needed before the next capture run picks it up.

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
