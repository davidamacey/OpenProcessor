# Cropwright: CLAUDE.md

SvelteKit + TypeScript app for labeling image crops at scale (product name
**Cropwright**, package `cropwright`), the human-in-the-loop frontend for
[OpenProcessor](https://github.com/davidamacey/OpenProcessor). It is a pure consumer
of the OpenProcessor curation API: no database of its own, no authentication.

This file holds the conventions, layout, commands and rules an agent needs. What a
page does is in the code and its tests; what changed and when is in `CHANGELOG.md`.
Do not add per-feature history here.

## Architecture

- **Stack:** SvelteKit 3 (static adapter) + TypeScript + Tailwind CSS + `svelte-dnd-action`
  (pointer-event drag; HTML5 DnD is unreliable in some browsers and webviews). Built to
  static files, served by nginx in the production container.
- **Backend:** OpenProcessor at `{API_PREFIX}` (default `/curation`). All state is
  reconstructed from API calls. Nothing goes in `localStorage` that an API call cannot
  rebuild; `sessionStorage` holds only per-tab view state.
- **State:** Svelte 5 runes (`$state`, `$derived`, `$effect`). Controllers are
  `*.svelte.ts` modules; pages stay thin.
- **Domain-agnostic:** the build ships no domain. The region slot is synthesized from the
  backend's served region profile (`GET {prefix}/health` `region_profile`). Example domain
  profiles live under `examples/` as tier-2 JSON, never bundled. A new domain is a backend
  region profile, optionally plus a tier-2 profile, not an app-code edit.
- **Served facts, not client guesses:** the backend serves decisions, vocabularies, labels,
  counts and gating flags; the frontend renders them. Nothing computes a metric, rank,
  mapping, winner, class kind or capability client-side. A control whose backend feature is
  not served is absent, not disabled. A failed read shows its error; never fabricate a
  default list or id.

## Layout

| Path                                    | What is there                                                                                                              |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `src/routes/`                           | Pages. Every page is `p/[project]/<section>` except `/projects` and the redirects in `+page.ts` and `[...legacy]/+page.ts` |
| `src/routes/p/[project]/+layout.svelte` | App shell: top nav, project switcher, status chip, the one global keydown listener                                         |
| `src/lib/api*.ts`                       | Typed API wrappers (`api.ts` plus `api_<area>.ts`); the only files that build `${API_PREFIX}` URLs                         |
| `src/lib/stores/`                       | Rune stores (projects, health, keymap, region profile, vocabularies)                                                       |
| `src/lib/components/`                   | Shared components (`Modal.svelte`, `StrategyBar`, `SlotCard`, ...)                                                         |
| `src/lib/format/`                       | The only home of duration, byte, count and time formatters                                                                 |
| `src/lib/annotations/`                  | Slot model: capabilities, served-slot synthesis, cohorts                                                                   |
| `src/lib/contract/`                     | Tests that pin the frontend to `contracts/`                                                                                |
| `src/lib/sse.ts`                        | `openStream()`, the one SSE open/backoff/reconnect helper                                                                  |
| `e2e/`                                  | Playwright via pytest: `stubbed/` (CI) and `live/` (read-only, manual)                                                     |
| `examples/`                             | Tier-2 annotation profiles (JSON)                                                                                          |
| `scripts/`                              | `run-e2e.mjs`, `mutation.mjs`, docs screenshot capture, `lib/previewServer.mjs`                                            |
| `sveltekit.options.js`                  | Adapter, preprocess, `$lib`/`$contracts` aliases, version polling                                                          |

Aliases: `$lib`, `$components`, `$stores`, `$contracts` (`../contracts`, the backend's
generated wire contract). Use them, never `../../` paths. 2-space indent, `npm` only,
Prettier formats.

## Commands

```bash
cd frontend
npm ci
npm run dev        # http://localhost:5173; set PUBLIC_TRITON_API_URL in .env if not proxied
npm run check      # svelte-kit sync + svelte-check
npm run lint       # prettier --check + eslint
npx vitest run     # unit and component tests (jsdom)
npm run build      # static build to build/
npm run test:e2e   # stubbed Playwright suite (own venv at e2e/.venv, auto-provisioned)
npm run test:live  # live read-only tier; needs CROPWRIGHT_LIVE_URL, skipped otherwise
npm run test:mutation   # Stryker, 15-20 minutes
```

Run one stubbed test: `e2e/.venv/bin/python -m pytest e2e/stubbed/<file> -n 2`.
Before reporting UI work done, open the built page in a browser (`DISPLAY=:11`) at desktop
and 430 px width and look at it; type checks do not prove a feature works.

## Routes

Every page lives under a project: `/p/[project]/<section>`. The active project is in the
URL path only. `/` redirects to `/p/<served default_slug>/dashboard`; bare legacy paths
redirect to the same path under the default project. Links and `goto()` are built with
`projectHref()` (`$lib/projectPaths`) wrapped in `resolve()`, never hand-assembled.

| Route                                                                                   | Purpose                                                                                                                                          |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `/dashboard`                                                                            | Dataset stats (polled), `AutoLabelPanel` (clustering run), detections summary                                                                    |
| `/ingest`                                                                               | Browser upload (chunked, pause/resume/cancel), server-path ingest panels gated by `GET /ingest/config`, status by source, region drain           |
| `/datasets`                                                                             | Labeled-dataset import wizard, imports list, job view; absent without `GET /datasets/formats`                                                    |
| `/clusters`, `/clusters/[id]`                                                           | Cluster grid, slot region gallery, strategy bar, semantic and item-text search, matching-items mode; one cluster with drag-and-drop and bulk ops |
| `/review`                                                                               | Review tabs (served by `GET /review/tabs`), filter bar, deep links via `/locate`, region tab with provenance                                     |
| `/classes`                                                                              | Add, rename, merge, deprecate, restore classes; hotkeys; new-class proposals                                                                     |
| `/audit`                                                                                | Label confirmation (VLM scope)                                                                                                                   |
| `/export`                                                                               | YOLO export, balance, test-holdout freeze, registry downloads                                                                                    |
| `/models`                                                                               | Model registry browser, shared models from other projects, VLM endpoint rows                                                                     |
| `/train`                                                                                | Training cockpit: preflight, launch, progress, runs, promote, reproduce, cohorts                                                                 |
| `/bakeoff`                                                                              | Model comparison on the served bake-off wire                                                                                                     |
| `/settings`, `/settings/{prompt-packs,region-profiles,models,open-vocab,ingest-policy}` | Deployment defaults, scores, keymap editor and the config-store editors; each sub-page absent until the backend serves it                        |
| `/projects`                                                                             | The one global page: project lifecycle, capacity, pipeline pause                                                                                 |

## Rules

### Projects

- No backward compatibility: there is no unscoped `{API_PREFIX}/...` alias and no
  `default` fallback.
- GLOBAL routes (never project-scoped): `{API_PREFIX}/projects` (list and every lifecycle
  write), global `{API_PREFIX}/health`, global `{API_PREFIX}/events`. Everything else is
  under the project's own **served** `prefix`, never assembled client-side.
- `projectsStore` is read in the root `load()`; `p/[project]/+layout.ts` resolves the slug
  (an archived project opened by link is fetched by slug), calls `setScopedPrefix(<served
prefix>)` and runs every registered reset hook on a project change before loading the
  scoped region profile and vocabularies.
- Combine projects and model sharing show served refusals and `class_mapping` verbatim.
  Class identity is the name, never an index.

### Data integrity

- Every label change is an immediate API call. No save button.
- Optimistic UI with an error-rollback toast on API failure.
- Bulk operations show a confirmation dialog with the affected count (served dry run first
  where the backend offers one).
- The audit trail lives server-side (`label_source`, `label_validated`, `class_source`,
  `updated_at`, `class_id_history`).
- Test-holdout crops (`test_holdout=true`) never reach the UI and must never be relabeled
  by the VLM or cluster auto-suggest. The backend filters; the UI is the second line.
- Refusals render the served `detail.message` verbatim.

### Health split

`GET {API_PREFIX}/health` (global) carries deployment facts; the scoped
`{prefix}/health` carries the project's region profile and `api_features`. The root layout
reads the global one once (bounded retries); `healthStore.poll()` keeps the scoped facts
fresh, and a successful poll seeds the region slot. The top-bar status chip polls it.

### Served per-tab filters

`GET {prefix}/review/tabs` serves each tab's `filters`, `filter_defaults` and
`filter_specs`. `/review` renders each control through `filterVisible(param)`
(`reviewTabsVocabularyStore.filterSupported`), which defaults to visible when the tab id is
unknown (not loaded, or a tier-2 slot tab). Option labels that name a default read the served
default. Every other served `filter_specs` entry renders by its kind. When a UI test races a
value that settles after the filter specs load, wait for it to stabilise rather than reading once.

### Keyboard shortcuts

- Class assignment is the per-class `hotkey_letter` (set on `/classes` or in the `~`
  overlay), routed through `dropOnClassStore` by the layout-level keydown listener. There
  is no `1-9, 0` scheme.
- Every other shortcut is a named action `<context>.<verb>` (`review.queue.discard`,
  `box_edit.nudge_up`, `cluster.undo`, ...), declared once with its defaults in
  `src/lib/keymapFallback.ts` and read through `keymapStore`
  (`keysFor`, `label`, `glyph`, `actionFor`). Pages call
  `keyboardStore.registerAction(id, handler, scope)`; dispatch resolves keys at keypress
  time, so a rebind needs no re-registration.
- **Hard constraint for every new control:** no new global keydown binding. A new key is
  a new action id in the fallback keymap; component tests assert no `window`/`document`
  keydown listener is added (`AssistScopeBar`, `StrategyBar`, `EmbeddingPlot`,
  `SemanticSearchBox`).
- Printed keys always come from `keymapStore.glyph(id)`. `keymap.literals.scan.test.ts`
  fails on a literal `<kbd>`, "Press X" or "(X)" outside the keymap's own files.
- Locked keys: Esc, Enter and the four arrows. Fixed form and a11y keys (Enter submits,
  Esc cancels, listbox arrows, Tab) are not actions.
- One box-edit handler: `MultiBoxCanvas.handleKey` resolves through
  `keymapStore.actionFor('box_edit', ...)`; the page and `SlotBboxEditor` forward to it.
- The served keymap (`GET/PUT {prefix}/keymap`) is loaded once from the root layout; a
  404/501 leaves the store on `FALLBACK_KEYMAP` and hides the editor card. A
  `config.changed axis=keymap` SSE frame refetches and applies live.

### StrategyBar

`/clusters`, `/clusters/[id]` and `/review` render a collapsed `StrategyBar`: independent,
stackable controls (review-sort dropdown, score chips, diverse stepper) that never replace
the production defaults. There is no cluster-method picker and no detection-profile
control on it; the deployment-wide defaults (cluster method, sort, prompt pack, VLM) live
on `/settings`, where the control shown for each axis is decided only by the served
per-entry `settable` flag on `/methods`. An empty queue shows the served reason.

### Slots and regions

- Region work is slot-generic: UI copy names the region by `slot.label`, the served tab
  label or the dataset-export blurb, never a fixed noun. A tier-2 profile may add
  capabilities; the served region slot is built by `servedRegionSlot.ts`.
- A permanent false-positive bucket is recognised only by the served
  `cluster_kind === 'false_positive'`, never by an id constant.
- Regions are lists of boxes. Per-box actions come from the multi-box surface; a stale box
  state is refused by the backend and shown as served.
- `api.ts` wrappers of `/regions` endpoints use `Region`; the view layer uses `Slot` or
  no prefix. No rename crosses the wire: paths and JSON keys are the backend's.

### Domain neutrality

`src/lib/domainNeutral.scan.test.ts` fails on any plate noun, `LPR`/`lpr_` or vehicle-domain
word anywhere under `src/` (code, strings, comments, tests). Fix the file; do not widen the
allow-list. Tests use the neutral widget domain (`WIDGET_TAG_PROFILE`, `widgetTagSlot` in
`src/lib/test/fixtures/regionSlot.ts`; `installServedRegionProfile(...)` /
`resetDeploymentSlots()` for the live registry). `exampleProfile.test.ts` parses every
`examples/annotation-profiles/*.json`. the e2e stub (`e2e/harness/stub.py`) serves the widget profile on `/health`.

### Shared helpers (one home each)

- Formatters: `src/lib/format/` (`duration.ts`, `bytes.ts`, `count.ts`, `date.ts`, `latency.ts`, `time.ts`).
  `format.scan.test.ts` fails on a second definition of `formatDuration|fmtBytes|formatTime|fmtEta`.
- Dialogs: `lib/components/Modal.svelte` (role `dialog`, `aria-modal`, Escape closes, focus
  trap and restore, backdrop click). No page hand-rolls a backdrop; `modal.scan.test.ts`
  fails on `svelte-ignore a11y_click_events_have_key_events` outside `Modal.svelte`.
- Server streams: `openStream()` in `src/lib/sse.ts`.
- Counts that may be `null` (an artifact written before the field existed) render through
  `formatCount()` as an em dash, never 0.

### Styling

Dark theme by default. No emoji, no gradients. Tailwind `bg-zinc-950` base with accent
through CSS variables. Never fetch full-resolution images in grids; use
`{prefix}/crops/{id}/thumbnail`. The API serves all images; Cropwright mounts no volumes.

### Security

The curation API has no request authentication. Every write is reachable by anyone who can
reach the nginx origin. Never expose Cropwright or the API it proxies to the public
internet; run it on a trusted LAN/VPN or behind an authenticating reverse proxy.
See the repository `SECURITY.md`.

## API contract

The tests read the backend's generated files in the repository's `contracts/` directory
(`$contracts`), so a backend rename fails a frontend test instead of rendering blanks.
Never hand-edit `contracts/`; regenerate with `make contracts` at the repository root.
Under `src/lib/contract/`:

- `wireKeys.test.ts`: `RAW_CROP_KEYS` and every built-in slot's `*Field` references vs the
  backend item and region wire keys (`KNOWN_STALE` allow-list, empty today).
- `regionStatus.test.ts`: slot lifecycle states vs the backend `RegionStatus` enum.
- `classSources.test.ts`: `sourceBadge` handles every backend `class_source`.
- `endpointCatalog.test.ts`: every `${API_PREFIX}/...` call site (extracted mechanically by
  `apiCallScanner.ts`) resolves to a real path and method in the OpenAPI spec, with query
  params checked. A file that references `API_PREFIX` outside the scanned set fails the
  completeness guard. Runtime-composed paths are anchored `MANUAL_OVERRIDES`.
- `bakeoffContract.test.ts` pins `types_bakeoff.ts` to the OpenAPI spec.

## Testing

- **Unit and component:** vitest + jsdom. `vite.config.ts` sets
  `resolve.conditions: ['browser']` under `VITEST`, so a `.test.ts` mounts Svelte 5
  components directly. Page logic lives in controllers (`*.svelte.ts`) and is tested there.
  Write the test first and watch it fail. Do not skip or relax assertions.
- **Stubbed e2e (`e2e/stubbed/`, CI job `e2e-stubbed`):** `npm run build` + `vite preview`,
  driven by pytest + Playwright, never against a real backend. It is fail-closed: a request
  under `{API_PREFIX}` that no test registered gets 501 and is recorded in `stub.unhandled`;
  every test asserts at teardown that it is empty and that at least one stub fired.
  - The `stub` fixture serves the app's own files through the harness (`route.fetch()`),
    because Chromium aborts queued requests with `net::ERR_NETWORK_CHANGED` whenever the
    host network changes. Keep it; `test_harness_serves_app.py` fails without it.
  - No fixed sleeps (tests run in parallel): wait on a real selector, response or state.
    The host is often loaded, so give builds and first navigations generous timeouts.
  - The preview server runs in its own process group (`scripts/lib/previewServer.mjs`)
    and the whole group is stopped on exit; never leave one running.
  - A failed test writes a screenshot, console, request log, stub lists and DOM to
    `artifacts_local/e2e-failures/<test id>/`; read those first when a test flakes.
  - A `py_compile` hook and CI step gate `scripts/*.py` and `e2e/**/*.py` syntax.
- **Live read-only tier (`e2e/live/`):** drives the real deployed build against a real
  backend (`CROPWRIGHT_LIVE_URL=... npm run test:live`; skipped when unset, never in CI).
  Structurally read-only: the `guarded_page` fixture aborts any non-GET/HEAD request under
  the API prefix and its teardown fails if one was attempted. It is project-aware
  (`live_project` resolves the served default slug; reads go through
  `api_get(live_url, live_project, path)` and navigation through `page_path(...)`), and it
  preflights the global `/health`. Tests assert no page error, no unexpected 4xx/5xx
  (`ALLOWED_4XX_5XX` is explicit), no `NaN`/`undefined` text, visible images load, no
  horizontal overflow at 800 px, and that the UI agrees with the API. Its screenshots are
  not a substitute for a human looking at them.
- **Mutation testing:** `npm run test:mutation` checks that the tests notice a broken line.
- **Visual check:** a fresh-start run on public datasets only (COCO, license-checked Open
  Images); never a private dataset or deployment detail in a screenshot.
- GPU 1 is reserved for other projects; do not target it in training-form tests or fixtures.

## Chunk-load recovery

A failed module chunk (flaky network, a deploy that replaced hashed chunks) is detected by
`isChunkLoadError` (`src/lib/chunkRecovery.ts`) in `hooks.client.ts` (`handleError` and
`vite:preloadError`). It does one `location.reload()` guarded by a `sessionStorage`
timestamp (`cropwright.chunkReloadAt`, 30 s; unreadable storage fails closed so it cannot
loop). A repeat falls through to `+error.svelte`. `kit.version.pollInterval` is 60 s, the
root layout shows a "new version available" banner, and `nginx.conf` serves
`/_app/version.json` with `no-cache`. Tests: `chunkRecovery.test.ts`,
`e2e/stubbed/test_chunk_recovery.py`.

## Deployment

The image is built from `frontend/Dockerfile` by the root compose `cropwright` profile
(`docker-compose.dev.yml` builds from source) and published by the repository's
`scripts/release/build_and_publish.sh`. nginx proxies `PUBLIC_API_PREFIX/*` to
`API_UPSTREAM`, runs as a non-root user and listens on 8080 inside the container. In
gateway mode it proxies the sub-paths in `OP_GATEWAY_SUBPATHS`. Docs are served by the
repository `docs-site/` (Cropwright section in `docs-site/docs/cropwright/`).

## Documentation and changelog discipline

Enforced by CI (`.github/workflows/frontend.yml`, `changelog` job):

- A push or PR that changes `frontend/src/` or `frontend/scripts/` must also touch
  `frontend/CHANGELOG.md` with an entry under `## [Unreleased]` (Keep a Changelog). One
  consolidated entry per batch is fine.
- `[skip-changelog]` in the last commit message is for a change with no user-facing or
  architectural effect (a test-only fixture tweak, a typo). Do not use it to avoid a real entry.
- An architecture-level change (new route, renamed core mechanism, new slot capability,
  cross-repo contract change) also updates this file in the same push. Keep this file under
  400 lines (a repository guard enforces it): rules and orientation, never history.
- A non-trivial change leaves a plan under the repository's `docs/design/`.
- Code comments say why, not what. No wave, round or work-package ids.

## Known constraints

- Use `svelte-dnd-action` (pointer-based) only; never HTML5 DnD.
- Never fetch full-resolution images in grids.
- The frontend never holds a copy of the wire contract; it reads `contracts/`.
- Do not split large files here unless the task is a split; see `CONTRIBUTING.md`
  "Guards and baselines" for the file-size ratchet.
