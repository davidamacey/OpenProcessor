# Cropwright health pass, 2026-10-29

A quality pass over the Cropwright frontend (`frontend/`, branch
`test/v070-cropwright-health`) against a live backend: every automated check, then
the main workflows in a real browser at 1600 px and 430 px on public data.

Host load was high throughout (load average 27 to 62 while checks ran), so every
timing below is noisy. Functional and visual results are not affected.

## Setup

- Backend: a running test stack (core API, Triton, OpenSearch, curation workers) with
  the stock Cropwright container in front of it. Cropwright was driven through that
  container, so the live pages ran the build that shipped with the stack, not this
  branch. Fixes were checked on a local `vite build` + `vite preview` of this branch
  pointed at the same API.
- Data: public COCO images only. A throwaway project `fe-walk` was created from the
  UI, filled with 10 COCO photos through the browser upload, clustered, labeled,
  and deleted from the UI at the end. Other projects on the stack were not opened.
- Tools: Playwright 1.63 (headless chromium), axe-core 4 injected into each page,
  stubbed e2e suite with 4 workers.

## Checks

| Check | Result |
| --- | --- |
| `npm ci` | pass (npm audit advisories only) |
| `npm run check` (svelte-kit sync + svelte-check) | pass, 0 errors, 0 warnings |
| `npm run lint` (prettier + eslint) | pass |
| `npx knip` | pass, no findings |
| `npx vitest run` | pass before and after: 460 files / 4866 tests, then 465 files / 4881 tests |
| `npm run build` | pass |
| `npm run test:e2e` (stubbed pytest + Playwright) | pass, 297 passed, before and after the fixes |
| `npm run test:live` against the live stack (project `fe-walk`) | 18 passed, 8 skipped (the region-profile routes: the stack serves no region profile) |
| axe-core, 13 pages, desktop | before: 201 contrast nodes, 5 critical `select-name`, 2 `label-title-only`, bake-off landmark errors, no `h1` on review. After: contrast clean on the pages checked, `select-name` and landmarks clean, `h1` present; only the items in #380 remain |
| Horizontal overflow at 430 px, 18 routes | none |
| Console errors and failed requests across 18 routes x 2 widths | none from the app. Seen: expected 404s from the combine availability probe (issue 382), `ERR_NETWORK_CHANGED` once from the host network, and 404 thumbnails on two older projects (see below) |

Page timings (navigation to network idle, load average 27 to 44, so noisy): every
route measured 2.6 to 3.4 s, almost all of it the wait for idle. No page was
visibly slow.

## Workflows walked

Projects list and create (including the blocked "building" state right after
create), ingest upload of 10 files, dashboard stats and recluster (a second start
shows the served 409 as a toast), clusters list, cluster detail, semantic text
search, class create, bulk assign of 60 crops (immediate, undoable, toast shown),
review label then undo, review Confirm disabled with the served reason when there is
no proposal, export and train empty states, models, bake-off, dataset imports, every
settings page, 404 project, 404 route, API down. Region editing was not available:
the stack serves no region profile, so the region tabs and routes are absent (the
stubbed e2e suite covers them).

## Defects found and fixed

Each fix has a unit or component test that failed first (the colour-token and
project-poll tests were seen failing against the old code).

| Defect | Fix commit subject |
| --- | --- |
| An unknown URL showed "Something went wrong / Reloading may help" | show a not-found page for unknown URLs instead of the reload prompt |
| A cluster thumbnail the server could not serve spilled the literal alt text "thumb" across the grid | stop cluster thumbnails spilling alt text when an image fails |
| On search results the cluster badge ran under the MATCH chip | keep the cluster badge clear of the match chip on search result cards |
| Settings axis selects had no accessible name (axe critical) | name each settings axis select for assistive tech |
| Class table group select and hotkey input unnamed, empty header cell, bake-off page nested a second `main`, review page had no `h1` | name class-table controls, drop nested main in bake-off, add review h1 |
| Dim text (`zinc-500`/`zinc-600`) failed AA contrast, 201 nodes (3.7:1 and 2.6:1) | lift dim zinc text to WCAG AA contrast |
| White labels on the primary and danger buttons and the LIVE badge failed AA (3.6, 3.4, 2.3:1) | give filled buttons and the live badge AA label contrast |
| Blocked-start screen repeated its own title as the reason | stop the blocked-start screen repeating its title as the reason |
| A project still building showed "not available" until a manual reload | open a building project by itself once it becomes selectable |

## Left, filed as issues (milestone v0.7.0, label bug)

- #380 cluster detail crop cards nest interactive controls (axe `nested-interactive`,
  `aria-allowed-role`, remaining empty table headers).
- #381 small UI inconsistencies: `60 / 3 listed` after a bulk assign, clipped search
  placeholder, blank `Proposed` row, two timestamp formats.
- #382 `/projects` logs a 404 console error on every load (the combine probe).

Remaining axe findings after the fixes: the three items in #380. Contrast is clean
on the pages checked.

## Environment notes (not frontend defects)

- On two older projects of the stack every crop thumbnail returned 404
  (`image not found` for a path under the sample-images mount that no longer exists
  in the API container). The frontend now degrades to a dimmed empty tile instead of
  alt text; the files themselves need restoring on the stack.
- Playwright `full_page` screenshots capture only the viewport on this app, because
  the shell scrolls inside its own container. Evidence shots of long pages used a
  tall viewport (1400 to 2000 px) instead. The first screenshots of `default`
  routes were taken at 900 px high and show only the first screen.

## Screenshot index

Stored under `frontend/artifacts_local/health/` (gitignored), `d_` = 1600 px,
`m_` = 430 px:

- `{d,m}_fe-walk_<route>.png`: dashboard, ingest, clusters, clusters_10000, review,
  classes, audit, export, models, train, bakeoff, datasets_imports and the five
  settings pages, on the public project.
- `{d,m}_default_<route>.png`: the same routes on the stock project, first screen
  only.
- `d_fe-walk_ingest_uploaded.png`, `d_fe-walk_dashboard_after_recluster.png`,
  `{d,m}_fe-walk_dashboard_409.png`, `d_fe-walk_clusters_search.png`,
  `d_fe-walk_cluster_assign_dialog.png`, `{d,m}_fe-walk_review_confirm.png`.
- Empty and error states: `{d,m}_404project.png`, `{d,m}_404route.png`,
  `{d,m}_api_down.png`, `d_g_projects.png`, `d_g_projects_new_modal.png`.
- After the fixes (local build): `{d,m}_fixed_search.png`, `{d,m}_fixed_dashboard.png`,
  `{d,m}_fixed_classes.png`, `{d,m}_fixed_404route.png`.
- axe results: `axe_before.json`, `axe_after.json`.
