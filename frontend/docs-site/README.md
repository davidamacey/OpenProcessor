# Cropwright documentation site

Docusaurus 3 site for Cropwright, modelled on a sister project's `docs-site/`.
Deploys to GitHub Pages at `https://davidamacey.github.io/cropwright/`.

## Development

```bash
npm install
npm start     # http://localhost:3000/cropwright/
```

## Build

```bash
npm run build   # -> build/, fails on any broken link/anchor
npm run serve   # serve the production build locally
```

## Structure

- `docs/` — the doc pages themselves (getting-started, user-guide,
  configuration, operations, developer-guide, faq), ordered by `sidebars.ts`.
- `src/pages/index.tsx` — the marketing/landing homepage.
- `src/pages/architecture.tsx` — tabbed, Archify-rendered architecture
  diagrams (system, workflows, sequences). Specs live in
  `architecture-diagrams/specs/`; see that directory's `README.md` and
  `scripts/generate-architecture-diagrams.sh`.
- `src/pages/roadmap.tsx` + `src/data/roadmap.json` — the hand-maintained
  public roadmap. Update the JSON when scope changes; don't hand-edit the
  page for content.
- `src/components/Screenshot` — `<Screenshot name="..." alt="..." />`,
  used throughout the docs; renders a "pending" placeholder until the named
  file exists under `static/img/screenshots/`. See
  `docs/developer-guide/screenshots.md` for how those are captured.

## Deployment

`.github/workflows/docs.yml` builds on every push touching `docs-site/**`
and deploys to GitHub Pages on `main`. It won't actually run until CI
minutes/Pages are enabled on the public repo — that's expected for now.
