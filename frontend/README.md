# Cropwright

Web-based, domain-agnostic image-crop annotation app. In this
deployment it's configured for a vehicle + license-plate dataset, driven
entirely by data (classes, annotation-slot profiles) rather than
hardcoded assumptions — see `docs/genericization-plan-2026-09-13.md`
for the capability model that makes that possible. Companion to
OpenProcessor (server-side inference, OpenSearch, clustering) and
`legacy_sorter` v2 (the desktop sort UX).

Built with **SvelteKit 2 + Svelte 5 runes + TypeScript strict + Tailwind v4**.
Pointer-event drag-and-drop via `svelte-dnd-action`. Dark theme, Apple
system colors, keyboard-first UX matching the legacy_sorter manual mode.

## Screenshots

![Demo](docs/screenshots/demo.gif)

Full walkthrough of every route with explanations: **[docs/FEATURES.md](docs/FEATURES.md)**
(full doc index, including design/research docs: **[docs/README.md](docs/README.md)**)

## Quick start

```bash
# 1. Install deps
cd /data/repos/legacy-labeler
npm install

# 2. Point it at the OpenProcessor API (default fine if running locally on 4603)
cp .env.example .env
# edit if your OpenProcessor backend lives elsewhere

# 3. Dev server (hot reload)
npm run dev    # http://localhost:5173

# 4. Production build
npm run build
npm run preview
```

The Docker production image is built and run from this repo's own
`docker-compose.yml` (`docker compose up -d --build`), host port 5184
(`CROPWRIGHT_PORT`) — see "Production deployment" below.

## Routes

See `CLAUDE.md`'s Routes table for the authoritative, up-to-date list —
it's kept current as routes change and isn't duplicated here to avoid
drift. Highlights: `/dashboard` (pipeline health), `/clusters` +
`/clusters/[id]` (cluster-based triage, drag-and-drop, strategy bar with
an embedding-plot lasso-select tool), `/review` (5 consolidated tabs +
quick-filter presets, fuzzy class picker), `/classes`, `/export` (YOLO
export, test-holdout freeze, frozen-artifact downloads), `/train`
(training cockpit, promote/reproduce), `/models`, `/bakeoff`.

## Keyboard shortcuts

See `CLAUDE.md`'s Keyboard shortcuts section for the authoritative
table. There is **no** `1-9, 0` top-N quick-assign scheme — it was
removed in favor of one binding scheme (per-class `hotkey_letter`,
configured on `/classes` or the `` ` `` overlay) so there's no "what
does this key do here?" ambiguity. Reserved single-char action keys
(`g n d z x u a m /`) can't be bound to a class — both the client and
the server reject that.

## Configuration

Env vars (`.env` or via Vite `--define`):

| Var                     | Default     | Purpose                                                                                                                                                                                                                               |
| ----------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PUBLIC_TRITON_API_URL` | _(empty)_   | Override the OpenProcessor API base URL. Leave empty in Docker — nginx proxies `PUBLIC_API_PREFIX` same-origin to `API_UPSTREAM`, so no CORS and no hostname hardcoding. Set to `http://localhost:4603` for local `npm run dev` only. |
| `PUBLIC_API_PREFIX`     | `/curation` | Path prefix the backend serves its curation endpoints under. Must equal the backend's `OP_API_PREFIX`.                                                                                                                                |

The app is a pure SPA consumer of the OpenProcessor API — there is **no**
local database. State is reconstructed from API calls; `localStorage` only
caches transient UI state (sidebar collapse, last-seen cluster ID).

### Configuring annotation slots for your own dataset (no code change)

Cropwright ships configured for one domain (this deployment's
vehicle/license-plate dataset), but a deployment can register its own annotation slot — a
sub-bbox, a text field, a review queue, whatever your class needs — by
dropping an `annotation-profiles.json` file next to the built app, with
no fork and no rebuild required to iterate on it:

- **Before a build**: place the file at `static/annotation-profiles.json`
  in the checkout, then `npm run build` / `docker build` as usual.
- **Against a running container, no rebuild**: bind-mount the file over
  `/usr/share/nginx/html/annotation-profiles.json`.

See `static/annotation-profiles.example.json` for a fully worked example
(a pallet-shipping-label slot) and
`docs/annotation-slots-contract-draft.md` for the schema. A missing or
invalid file is always silently ignored — the app falls back to its
built-in slots and never crashes on a bad config.

## Data integrity

- Every label change is an immediate API call with optimistic UI.
  Drag-drop into a class is also optimistic — the crop disappears from
  the grid on release; if the backend reports an OCC conflict the UI
  re-fetches to show the true state.
- Backend writes use `refresh=True` so the next queue / cluster fetch
  sees the change immediately (no ~1 s OpenSearch refresh race).
- Multi-crop writes (`PUT {API_PREFIX}/crops/batch_label`, `POST {API_PREFIX}/crops/move`)
  parallelize their per-crop OCC updates via `asyncio.gather` and retry
  up to 5 times against a `(0.05, 0.15, 0.45)s` backoff, so a worker
  racing the human almost never produces a conflict the operator can
  see.
- Failures roll back via snapshot revert + a toast.
- The 50-entry undo ring per page restores the prior state on `Z` —
  for an undo of a class label that means `DELETE {API_PREFIX}/crops/{id}/label`
  (resets to model-suggested), for an undo of a discard it restores
  the local list (the backend dismissal flag is left in place; a
  follow-up `review_undismiss` endpoint will land alongside the
  re-show workflow).
- Bulk operations (multi-select label, multi-crop move) always show
  a confirm dialog with the affected count before firing.
- Test-set crops (`test_holdout=true`) are filtered out of every
  labeling queue at the API level — the UI never receives them.
- `review_queue` now uses `track_total_hits=true`, so the "X of N" count
  in the page header is the real total rather than capped at 10,000.

## Architecture

```
src/
  app.css            # Tailwind v4 + Apple system colors as CSS vars
  app.html           # SvelteKit shell
  routes/            # File-based routing (SvelteKit static adapter)
    +layout.svelte   # Top bar, sidebar, toast container, shortcut overlay
    +layout.ts
    +page.svelte                # /
    clusters/+page.svelte       # /clusters
    clusters/[id]/+page.svelte  # /clusters/[id]
    review/+page.svelte         # /review
    classes/+page.svelte        # /classes
    export/+page.svelte         # /export
  lib/
    api.ts           # Typed fetch wrapper, retry on 5xx, AbortSignal
    types.ts         # RegistryClass, Cluster, Crop, StatsSummary, ApiHealth
    stores/
      classes.svelte.ts   # 30s auto-refresh + topNForCluster()
      keyboard.svelte.ts  # Page-scoped shortcut registry
      health.svelte.ts    # 15s {API_PREFIX}/health poll → top-bar dot
      toast.svelte.ts     # Optimistic-UI rollback toasts
      undo.svelte.ts      # Per-page 50-entry undo ring
    components/
      Toast.svelte
      ClassSidebar.svelte         # left-rail class list + drop-target rows
      CropCard.svelte             # grid card; ⓘ button opens CropDetailModal
      CropDetailModal.svelte      # source image + crop + CropMetaPanel
      CropMetaPanel.svelte        # read-only provenance grid (class, plate chain,
                                  # detector/verifier chips, Gemma confidence, …)
      CutLine.svelte
      DetectorChip.svelte         # provenance chip for lpr / sam3 / gemma / human
      PlateBboxCanvas.svelte      # in-place plate bbox editor on /review/plates
      PlateEditor.svelte          # full PlateBboxCanvas modal (M on a cluster crop)
      ShortcutOverlay.svelte      # ` overlay; also an inline hotkey editor
```

All stores use **Svelte 5 runes** (`$state`, `$derived`, `$effect`)
exclusively — no Svelte 4 `writable()`/`readable()`.

## Conventions

- **Indentation**: 2 spaces. Prettier handles it.
- **TypeScript**: strict mode. `$lib`, `$components`, `$stores` aliases
  (configured in `svelte.config.js`).
- **Imports**: stdlib → vendor → `$lib/*` → relative.
- **No emoji**. No gradients. No purple/neon. Apple system colors only.
- **Dark theme** is the default and only theme.
- HTML5 drag-and-drop is broken in Tauri WebView and unreliable in some
  browsers — `svelte-dnd-action` (pointer-events) is the only DnD path.
- Don't fetch full-resolution NAS images in grids. Always use the
  thumbnail endpoint (`{API_PREFIX}/crops/{id}/thumbnail`, 128×128 LRU-cached
  server-side).

## Production deployment

The `Dockerfile` produces a static SPA served by `nginx:alpine` on port 80.
This repo's own `docker-compose.yml` (`docker compose up -d --build`) exposes
it on host port **5184** (`CROPWRIGHT_PORT`), and joins the OpenProcessor
API's docker network (`OP_DOCKER_NETWORK`, default
`openprocessor_triton_net`) so nginx can proxy by container name.

**nginx proxy (LAN-transparent API routing)**
The nginx config inside the container routes `PUBLIC_API_PREFIX` (default
`/curation`) to `API_UPSTREAM` (default `http://op-api:8000`) via
`proxy_pass`. This means:

- The browser always makes same-origin requests — no CORS issues.
- `PUBLIC_TRITON_API_URL` defaults to `''` so all API paths are relative URLs.
- The app works identically whether accessed from `localhost` or any LAN IP.

`docker-entrypoint.sh` can still inject a `PUBLIC_TRITON_API_URL` value at
container start (replacing the `__RUNTIME__` placeholder in built JS) if you
need to point the frontend at an OpenProcessor API on a different origin
instead of the co-located one. For the standard Docker Compose stack leave
it empty.

## Limits

- **No offline mode.** The OpenProcessor API must be reachable for the
  labeler to do anything useful. Without it the dashboard shows red
  health, all fetches surface "API unavailable" empty states.
- **No mobile support** by design (labeling sessions are keyboard-
  driven on a real keyboard).
- **No multi-user permissions.** Whoever loads the URL has full edit
  rights. Front the deployment with auth (oauth2-proxy / nginx
  basic auth) before exposing it beyond the LAN.
- **Deprecated-class restore is not implemented**, backend or frontend.
  The `/classes` "Restore" button is disabled — there is no backend
  endpoint that un-deprecates a class (`deprecated` is only ever set to
  `True`, via merge). Restoring a class also wouldn't automatically
  revert crops that a prior merge already bulk-relabeled to the target
  class — that needs its own design decision, not just a toggle.

## Troubleshooting

| Symptom                                                             | Likely cause                                                                                                                                                                  |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Red health dot in top bar                                           | OpenProcessor API not reachable at `PUBLIC_TRITON_API_URL`; or `{API_PREFIX}/health` returns 5xx                                                                              |
| Empty cluster grid                                                  | OpenSearch not running, or the crops index not yet populated by ingest                                                                                                        |
| Thumbnails 404                                                      | Crop document missing `bbox_norm`; or the NAS path isn't mounted into the API container                                                                                       |
| Hotkeys do nothing                                                  | A modal is open; or the focused element is an `<input>` (the keyboard store ignores typing)                                                                                   |
| Drag-drop appears to move but the crop is still there after refresh | Stale `dist/` bundle — rebuild and hard-reload (⌘⇧R / Ctrl+Shift+R). If `PUT {API_PREFIX}/crops/batch_label` never shows up in DevTools Network on drop, the bundle is stale. |
| Discard reappears on tab switch                                     | Should not happen anymore (the backend writes `review_dismissed_at` with `refresh=True` and the queue must_not's it). If it does, hard-reload — your bundle predates the fix. |
| Counters in /review header showing 10,000 when there should be more | Should not happen anymore (`track_total_hits=true` is on by default). If you see it, the deployed API predates the fix — restart it.                                          |
| Build fails with `Cannot find module 'svelte-dnd-action'`           | `npm install` not run, or node_modules stale (`rm -rf node_modules && npm install`)                                                                                           |

## Repos this depends on

- The OpenProcessor API — curation endpoints, OpenSearch, clustering,
  Triton models. Separate repo; not part of this checkout.

## License & attribution

[AGPL-3.0-or-later](LICENSE). Renamed to **Cropwright** and being
generalized toward open-source release — see
[GH issue #1](https://github.com/example-org/openprocessor/issues/1)
for the genericization plan and `docs/genericization-plan-2026-09-13.md`
for how the mechanism work landed. Logic and UX patterns originally
derive from the original vehicle + license-plate dataset plan at
`~/.claude/plans/we-need-a-full-compressed-manatee.md`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev workflow and
[SECURITY.md](SECURITY.md) for reporting vulnerabilities.
