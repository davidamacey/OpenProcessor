# OpenProcessor docs site

Docusaurus 3 site (npm only; Node version in `.nvmrc`).

```bash
make docs-build          # from the repo root: npm ci + docusaurus build
cd docs-site && npm ci && npm start   # live-reloading dev server
```

`onBrokenLinks` and `onBrokenAnchors` are `throw`, so a broken link or anchor fails the
build. CI (`.github/workflows/docs.yml`) runs the same build on every PR touching
`docs-site/`. `node_modules/`, `build/` and `.docusaurus/` are gitignored.

Screenshots: the Cropwright captures were taken read-only against the dev frontend backed
by the public COCO sample projects (`sample-coco-2k-v3`, `sample-coco-vehicles`,
`sample-coco-import-v2`) using Cropwright's `scripts/capture_docs_screenshots.py` (its read-only
request guard aborts every non-GET call except dry-run reports) plus a few extra single-page
captures (project switcher, shortcuts overlay, settings pages) at 1600 px, quantized to 256
colours to stay under 500 KB. Inspect every PNG for private project names or paths before
committing. Outstanding placeholders use `> Screenshot pending: ...`; see
`docs/developer-guide/screenshots.mdx`.
