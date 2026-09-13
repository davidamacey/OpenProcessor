# legacy-labeler

Web-based labeling app for the legacy v7 vehicle dataset. Companion
to `openprocessor` (server-side inference, OpenSearch, clustering) and
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

# 2. Point it at openprocessor (default fine if running locally on 4603)
cp .env.example .env
# edit if your openprocessor lives elsewhere

# 3. Dev server (hot reload)
npm run dev    # http://localhost:5173

# 4. Production build
npm run build
npm run preview    # serves the static build on http://localhost:5181
```

The Docker production image is built and run from the parent
`/data/repos/openprocessor/docker-compose.legacy.yml` overlay (see
that repo's `scripts/legacy/RUNBOOK.md`).

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

| Var                     | Default   | Purpose                                                                                                                                                                                                                               |
| ----------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PUBLIC_TRITON_API_URL` | _(empty)_ | Override the openprocessor base URL. Leave empty in Docker — nginx proxies `/curation/*` and `/clusters/*` to `op-api:4603` same-origin, so no CORS and no hostname hardcoding. Set to `http://localhost:4603` for local `npm run dev` only. |

The app is a pure SPA consumer of `openprocessor` — there is **no** local
database. State is reconstructed from API calls; `localStorage` only
caches transient UI state (sidebar collapse, last-seen cluster ID).

## Data integrity

- Every label change is an immediate API call with optimistic UI.
  Drag-drop into a class is also optimistic — the crop disappears from
  the grid on release; if the backend reports an OCC conflict the UI
  re-fetches to show the true state.
- Backend writes use `refresh=True` so the next queue / cluster fetch
  sees the change immediately (no ~1 s OpenSearch refresh race).
- Multi-crop writes (`PUT /curation/crops/batch_label`, `POST /curation/crops/move`)
  parallelize their per-crop OCC updates via `asyncio.gather` and retry
  up to 5 times against a `(0.05, 0.15, 0.45)s` backoff, so a worker
  racing the human almost never produces a conflict the operator can
  see.
- Failures roll back via snapshot revert + a toast.
- The 50-entry undo ring per page restores the prior state on `Z` —
  for an undo of a class label that means `DELETE /curation/crops/{id}/label`
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
    types.ts         # OpClass, OpCluster, OpCrop, OpStats, OpHealth
    stores/
      classes.svelte.ts   # 30s auto-refresh + topNForCluster()
      keyboard.svelte.ts  # Page-scoped shortcut registry
      health.svelte.ts    # 15s /curation/health poll → top-bar dot
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
  thumbnail endpoint (`/curation/crops/{id}/thumbnail`, 128×128 LRU-cached
  server-side).

## Production deployment

The `Dockerfile` produces a static SPA served by `nginx:alpine` on port 80.
The host port mapping in `docker-compose.legacy.yml` exposes it on **5184**.

**nginx proxy (LAN-transparent API routing)**
The nginx config inside the container routes `/curation/*` and `/clusters/*` paths
to `op-api:4603` via `proxy_pass`. This means:

- The browser always makes same-origin requests (`/curation/...`) — no CORS issues.
- `PUBLIC_TRITON_API_URL` defaults to `''` so all API paths are relative URLs.
- The app works identically whether accessed from `localhost` or any LAN IP.

`docker-entrypoint.sh` can still inject a `PUBLIC_TRITON_API_URL` value at
container start (replacing the `__RUNTIME__` placeholder in built JS) if you
need to point the frontend at a remote openprocessor instead of the co-located
one. For the standard Docker Compose stack leave it empty.

## Sharing the GPUs with another app

legacy training/inference holds GPU 0 (~40 GB SAM3), GPU 1 (~8 GB Triton),
and GPU 2 (~46 GB vLLM-Gemma). When another GPU-heavy app on the same host
needs the cards (example-app, model training, etc.), park the legacy ML
stack and keep human labeling running.

From the **openprocessor repo**:

```bash
make gpu-free       # stop Triton + SAM3 + vLLM + workers (labeler stays up)
make gpu-legacy    # bring the ML stack back when you're done
make gpu-status     # show containers + per-GPU memory usage
```

What stays up: `opensearch`, `op-api`, `legacy-labeler` — the entire
human-labeling workflow (`/review`, `/clusters`, `/classes`, drag-and-drop,
hotkeys) keeps working because none of those endpoints touch the GPU.

What you lose while in `gpu-free` mode:

- Pipeline ingest (no Triton, so no fresh detections).
- Auto-label / Gemma re-classification (no vLLM).
- Plate cascade (no SAM3, no LPR Triton model).
- Training and cluster-refresh background workers.

In-flight crops stay in OpenSearch with `plate_status='pending_*'` and resume
processing automatically when you `make gpu-legacy`.

## Limits

- **No offline mode.** openprocessor must be reachable for the labeler to
  do anything useful. Without it the dashboard shows red health, all
  fetches surface "API unavailable" empty states.
- **No mobile support** by design (labeling sessions are keyboard-
  driven on a real keyboard).
- **No multi-user permissions.** Whoever loads the URL has full edit
  rights. Front the deployment with auth (oauth2-proxy / nginx
  basic auth) before exposing it beyond the LAN.
- **Deprecated-class restore is not implemented**, backend or frontend.
  The `/classes` "Restore" button is disabled — there is no `openprocessor`
  endpoint that un-deprecates a class (`deprecated` is only ever set to
  `True`, via merge). Restoring a class also wouldn't automatically
  revert crops that a prior merge already bulk-relabeled to the target
  class — that needs its own design decision, not just a toggle.

## Troubleshooting

| Symptom                                                             | Likely cause                                                                                                                                                                   |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Red health dot in top bar                                           | openprocessor not reachable at `PUBLIC_TRITON_API_URL`; or `/curation/health` returns 5xx                                                                                               |
| Empty cluster grid                                                  | OpenSearch not running, or `op_vehicle_crops` not yet populated by ingest                                                                                                      |
| Thumbnails 404                                                      | `op_vehicle_crops` doc missing `bbox_norm`; or NAS path not mounted into op-api container                                                                                    |
| Hotkeys do nothing                                                  | A modal is open; or the focused element is an `<input>` (the keyboard store ignores typing)                                                                                    |
| Drag-drop appears to move but the crop is still there after refresh | Stale `dist/` bundle — `make ml-resume` then hard-reload (⌘⇧R / Ctrl+Shift+R). If `PUT /curation/crops/batch_label` never shows up in DevTools Network on drop, the bundle is stale. |
| Discard reappears on tab switch                                     | Should not happen anymore (the backend writes `review_dismissed_at` with `refresh=True` and the queue must_not's it). If it does, hard-reload — your bundle predates the fix.  |
| Counters in /review header showing 10,000 when there should be more | Should not happen anymore (`track_total_hits=true` is on by default). If you see it, the deployed `op-api` predates the fix — `docker compose restart op-api`.             |
| Build fails with `Cannot find module 'svelte-dnd-action'`           | `npm install` not run, or node_modules stale (`rm -rf node_modules && npm install`)                                                                                            |

## Repos this depends on

- `/data/repos/openprocessor/` — the `/curation/*` API + OpenSearch + Triton
  models. Private repo: do not push.
- `/data/repos/run_openwebui/` — Gemma 4 E4B vLLM server (port
  8012); the labeler doesn't talk to it directly, but openprocessor does.

See `/data/repos/openprocessor/scripts/legacy/RUNBOOK.md` for the
end-to-end pipeline.

## License & attribution

[AGPL-3.0-or-later](LICENSE). The repository is currently **private**
(not yet publicly released) — see
[GH issue #1](https://github.com/example-org/openprocessor/issues/1)
for the plan to genericize and open-source it. Logic and UX patterns
derive from the legacy v7 plan at
`~/.claude/plans/we-need-a-full-compressed-manatee.md`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev workflow and
[SECURITY.md](SECURITY.md) for reporting vulnerabilities.
