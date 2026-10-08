# Cloning this docs-site for a sibling project (e.g. OpenProcessor)

This `docs-site/` is built so a sibling project can reuse the whole
framework — config, landing page, roadmap, architecture-diagram
machinery, screenshot handling, deploy workflow — by editing **only** a
short, explicit list of files. Nothing project-specific should be
hardcoded inside a component (`src/components/**`) or the Docusaurus
config plumbing (`docusaurus.config.ts`, `sidebars.ts`).

## Copy unchanged

These files/directories are generic and should not need a single edit in
a cloned sibling site:

- `docusaurus.config.ts` — reads everything from `site.config.ts`.
- `sidebars.ts` — reads the `docs/**` file tree; only the *category
  labels/order* might change if the sibling's doc set has different
  top-level sections (edit this one file's `items` lists if so — it is
  structural, not copy).
- `src/css/custom.css`, `src/theme/MDXComponents.tsx`.
- `src/components/Hero`, `FeatureGrid`, `HowItWorks`, `QuickStart`,
  `ScreenshotShowcase`, `Screenshot`, `Lightbox`, `RoadmapView` — all
  data-driven, read from `site.config.ts` and `src/data/*.json`.
  `Screenshot` opens its image in the shared `Lightbox` (Esc/click-outside
  close, arrow keys page a gallery, keyboard focusable); the component is
  identical in every sibling site, so copy it, don't fork it.
  `ScreenshotShowcase` takes an optional `items` list for a second gallery.
- Local preview: serve the build with the provided nginx `Dockerfile` (or
  `python3 -m http.server` from a directory containing it at the baseUrl
  path). `docusaurus serve` mishandles the baseUrl for the embedded
  architecture iframes.
- `src/pages/index.tsx`, `src/pages/architecture.tsx`,
  `src/pages/roadmap.tsx` — assemble the components above; carry no copy.
- `Dockerfile`, `nginx.conf`, `.dockerignore`, `.gitignore`, `.nvmrc`,
  `tsconfig.json`, `package.json` (bump `name`/`version` only if you care).
- `.github/workflows/docs.yml` (repo root) — parameterized by the
  triggering repo's own `github.repository`/branch; see "Build-time
  assumptions" below for the one manual check.

## Edit for the new project

1. **`site.config.ts`** — the single site-identity module. Change:
   - `title`, `tagline`, `favicon`, `logo`
   - `organizationName`, `projectName`, `url`, `baseUrl` (GitHub Pages
     target — see "Build-time assumptions" below)
   - `githubRepo`, `editUrlBase`
   - `siblingProjects` (cross-links — point back at Cropwright and any
     other sibling)
   - `announcementBar` (drop or rewrite — Cropwright's is the "no auth"
     warning, which may not apply to every sibling)
   - `navbar.items`, `footerLinkGroups`
2. **`src/data/features.json`** — the landing-page feature grid. One
   `{title, icon, desc}` object per card; `icon` is currently unused
   (text-only cards) but kept as a stable key for a future icon swap.
3. **`src/data/workflow.json`** — the "How it works" step list
   (`{step, route, desc}`), conceptually related to but not generated from
   the `labeling-loop` architecture diagram (see item 6 below) — keep the
   two in sync by hand, there's no generator linking them.
4. **`src/data/screenshots.json`** — the landing-page showcase list
   (`{name, alt, caption}`); `name` must match a file under
   `static/img/screenshots/`.
5. **`src/data/roadmap.json`** — hand-maintained; see its own `note`
   field.
6. **`architecture-diagrams/specs/*.json`** — hand-authored Archify
   specs (architecture/workflow/sequence), built from real repo evidence,
   not app code. Rendered to `static/architecture/<name>.html` via
   `scripts/generate-architecture-diagrams.sh` (uses the Archify Claude
   Code skill — see `architecture-diagrams/README.md`). Group/diagram
   metadata for the `/architecture` page's tabs lives in
   `src/data/architecture-diagrams.json` (`{id, label, diagrams:
   [{id, title, description, height}]}`) — edit that data file, not
   `src/pages/architecture.tsx`, when adding or reordering a diagram.
7. **`docs/**`** — the actual doc content. Directory names under `docs/`
   drive `sidebars.ts`'s `items` paths, so keep them in sync if you rename
   a folder.
8. **`static/img/**`** — favicon/logo files, plus
   `static/img/screenshots/*` (see below).
9. **`scripts/capture_docs_screenshots.py`** (repo root, not under
   `docs-site/`) — update the `ROUTES` it reads from
   `docs-site/src/data/screenshot_routes.json` (see that file) to the new
   project's own route list; the capture mechanics (viewport widths,
   output path, CLI flags) are already generic.

## Steps to stand up a sibling site

1. Copy the whole `docs-site/` directory into the sibling repo.
2. Edit `site.config.ts` (step 1 above) for the sibling's identity.
3. Replace every file listed under "Edit for the new project" above.
4. Delete or replace `static/img/screenshots/*` — do not carry over
   Cropwright's screenshots.
5. `npm install && npm run build` — must pass with zero broken
   links/anchors (`onBrokenLinks`/`onBrokenAnchors` stay `'throw'`).
6. Copy `.github/workflows/docs.yml` into the sibling repo's own
   `.github/workflows/`, adjusting only the `paths:` filters if the
   sibling keeps `docs-site/` at a different path.
7. Run the screenshot capture script (see `docs/developer-guide/
   screenshots.md` in the copied docs for the public-data-only rule) once
   a public-sample-data backend is available.

## Build-time assumptions

Every value below is tied to *this specific deployment* (Cropwright on
GitHub Pages under `davidamacey/OpenProcessor`). A sibling site changes each
one at the location named — never by patching `docusaurus.config.ts`,
`sidebars.ts`, or a component directly.

| Assumption | Lives at | Sibling-site value |
| --- | --- | --- |
| Site `url` / `baseUrl` | `site.config.ts`: `url`, `baseUrl` | e.g. `https://davidamacey.github.io` / `/OpenProcessor/` |
| Org/repo name (GitHub Pages project settings) | `site.config.ts`: `organizationName`, `projectName` | e.g. `davidamacey` / `OpenProcessor` |
| Docs "Edit this page" target | `site.config.ts`: `editUrlBase` | e.g. `https://github.com/davidamacey/OpenProcessor/tree/main/docs-site/` |
| GitHub repo link (navbar/footer/hero) | `site.config.ts`: `githubRepo` | e.g. `https://github.com/davidamacey/OpenProcessor` |
| Pages deploy workflow's repo/branch | `.github/workflows/docs.yml` (repo root) — triggers off `push` to the *checked-out* repo's default branch via `github.ref`; no hardcoded repo name in the workflow itself, but confirm the sibling repo's default branch is actually `main` (the workflow's `if:` gate names it explicitly) | Edit the `if: github.ref == 'refs/heads/main'` line only if the sibling uses a different default branch |
| Node version | `docs-site/.nvmrc` (`20`) and the workflow's `node-version-file: docs-site/.nvmrc` | Bump both together if the sibling wants a newer Node; keep them equal — nothing enforces that automatically here (Cropwright has no cross-repo version-consistency test) |
| Architecture diagrams | `architecture-diagrams/specs/*.json` (Archify specs) + `scripts/generate-architecture-diagrams.sh` (repo root) + `src/data/architecture-diagrams.json` (page tab metadata) | Re-author the specs against the sibling's own routes/controllers/wire contract — never copy Cropwright's specs verbatim; regenerate `static/architecture/*.html` before shipping |
| Broken-link policy | `docusaurus.config.ts`: `onBrokenLinks: 'throw'`, `onBrokenAnchors: 'throw'` | Keep `'throw'` in any sibling site — do not weaken to `'warn'` to unblock a build; fix the link instead |
| Screenshot source of truth | `docs/developer-guide/screenshots.md` (rule: public-sample-data backend only, never a real deployment) + `scripts/capture_docs_screenshots.py` (repo root) + `src/data/screenshots.json` (what the landing page shows) | Rewrite the doc page's specific sample-data instructions for the sibling's own "how to get public sample data" story; keep the "never a real deployment" rule verbatim |
