# OpenProcessor docs site

Docusaurus 3 site (npm only; Node version in `.nvmrc`).

```bash
make docs-build          # from the repo root: npm ci + docusaurus build
cd docs-site && npm ci && npm start   # live-reloading dev server
```

`onBrokenLinks` and `onBrokenAnchors` are `throw`, so a broken link or anchor fails the
build. CI (`.github/workflows/docs.yml`) runs the same build on every PR touching
`docs-site/`. `node_modules/`, `build/` and `.docusaurus/` are gitignored.

Screenshots: pages carry `> Screenshot pending: ...` placeholders until real public-data
(COCO / Open Images) captures exist; see `docs/developer-guide/screenshots.mdx`.
