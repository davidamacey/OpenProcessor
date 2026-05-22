# legacy-labeler

Web-based labeling app for the legacy v7 vehicle dataset. Companion
to `openprocessor` (server-side inference, OpenSearch, clustering) and
`legacy_sorter` v2 (the desktop sort UX).

Built with **SvelteKit 2 + Svelte 5 runes + TypeScript strict + Tailwind v4**.
Pointer-event drag-and-drop via `svelte-dnd-action`. Dark theme, Apple
system colors, keyboard-first UX matching the legacy_sorter manual mode.

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

| Route | Purpose |
|---|---|
| `/` | Dashboard — class balance bar chart, ingestion progress, recent activity, quick actions |
| `/clusters` | Grid of clusters; sort by purity / size / dominant class; click into one. Selecting `class=license_plate` replaces the cluster grid with a **plate browse view** — paginated plate thumbnails filtered by detector / verified / score / plate-text |
| `/clusters/[id]` | Per-cluster paginated crop grid + DnD-to-other-cluster + bulk label/Gemma/AHC + similarity cut-line + sub-cluster tabs |
| `/review` | Review queues: Mismatches / Gemma low-conf / Outliers / Uncertainty / Model Disagreements / **Plates**. Plate rows show detector chips, the cascade chain, Gemma-read plate text, and a ⚠ shape-warning when the bbox envelope fails |
| `/classes` | Add/rename/regroup/merge classes; sync to OpenSearch; adequacy badges |
| `/export` | YOLO export status + augmentation gap table + Test Holdout freeze + downloads |
| `/train` | Training cockpit + **plate training cohort picker** (lpr_blind_spots / lpr_low_conf_correct / disagreement / human_corrected) |

## Keyboard shortcuts

| Key | Action |
|---|---|
| `1-9, 0` | Assign top-10 most-frequent classes for the current cluster |
| `Enter` | Confirm selected + advance to next unvalidated |
| `G` | Accept Gemma suggestion for selected |
| `N` | Skip (defer to review queue) |
| `D` | Discard (mark for delete) |
| `Z` | Undo last action (50-entry ring) |
| `A` | Select all on page |
| `Shift+Enter` | Confirm all Gemma suggestions on page |
| `M` | Open inline cluster picker (move selected to a different cluster) |
| `←` / `→` | Page navigation |
| `~` | Toggle keyboard-shortcut overlay |
| `Esc` | Cancel drag / picker / dismiss overlay |

## Configuration

Env vars (`.env` or via Vite `--define`):

| Var | Default | Purpose |
|---|---|---|
| `PUBLIC_TRITON_API_URL` | *(empty)* | Override the openprocessor base URL. Leave empty in Docker — nginx proxies `/curation/*` and `/clusters/*` to `op-api:4603` same-origin, so no CORS and no hostname hardcoding. Set to `http://localhost:4603` for local `npm run dev` only. |

The app is a pure SPA consumer of `openprocessor` — there is **no** local
database. State is reconstructed from API calls; `localStorage` only
caches transient UI state (sidebar collapse, last-seen cluster ID).

## Data integrity

- Every label change is an immediate API call with optimistic UI.
- Failures roll back via snapshot revert + a toast.
- The 50-entry undo ring per page calls `DELETE /curation/crops/{id}/label`
  on `Z` to restore the model-suggested label.
- Bulk operations (multi-select label, multi-crop move) always show
  a confirm dialog with the affected count before firing.
- Test-set crops (`test_holdout=true`) are filtered out of every
  labeling queue at the API level — the UI never receives them.

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
      ClassSidebar.svelte
      CropCard.svelte
      CutLine.svelte
      ShortcutOverlay.svelte
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

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Red health dot in top bar | openprocessor not reachable at `PUBLIC_TRITON_API_URL`; or `/curation/health` returns 5xx |
| Empty cluster grid | OpenSearch not running, or `op_vehicle_crops` not yet populated by ingest |
| Thumbnails 404 | `op_vehicle_crops` doc missing `bbox_norm`; or NAS path not mounted into op-api container |
| Hotkeys do nothing | A modal is open; or the focused element is an `<input>` (the keyboard store ignores typing) |
| Drag-drop doesn't trigger | Browser DnD vs. svelte-dnd-action confusion — refresh the page; check console for the `start_drag` event |
| Build fails with `Cannot find module 'svelte-dnd-action'` | `npm install` not run, or node_modules stale (`rm -rf node_modules && npm install`) |

## Repos this depends on

- `/data/repos/openprocessor/` — the `/curation/*` API + OpenSearch + Triton
  models. Private repo: do not push.
- `/data/repos/run_openwebui/` — Gemma 4 E4B vLLM server (port
  8012); the labeler doesn't talk to it directly, but openprocessor does.

See `/data/repos/openprocessor/scripts/legacy/RUNBOOK.md` for the
end-to-end pipeline.

## License & attribution

Internal. Not for public release. Logic and UX patterns derive from
the legacy v7 plan at
`~/.claude/plans/we-need-a-full-compressed-manatee.md`.
