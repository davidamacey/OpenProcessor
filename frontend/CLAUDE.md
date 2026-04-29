# legacy-labeler — CLAUDE.md

SvelteKit + TypeScript labeling web app for legacy v7 vehicle dataset
construction. Sister project to `legacy_sorter` (v2 Tauri app for the actual
sort UX) and `openprocessor` (server-side inference + OpenSearch + clustering).

## Purpose

Manage a high-volume labeling workflow over hundreds of thousands of vehicle
crops, with cluster-based assisted labeling, Gemma 4 vision suggestions, and a
keyboard-first UX matching the legacy_sorter manual-mode speed budget.

## Architecture

- **Frontend**: SvelteKit 2 + TypeScript + Tailwind CSS + svelte-dnd-action
  (pointer-event drag — HTML5 DnD is broken in Tauri WebView and unreliable in
  some browsers)
- **Backend**: openprocessor at `http://localhost:4603/op/...` — labeler is a
  pure consumer; no own database
- **State**: Svelte 5 runes (`$state`, `$derived`, `$effect`); nothing in
  localStorage that can't be reconstructed by an API call
- **Build**: SvelteKit static adapter → nginx in production Docker container

## Routes (post-MVP)

| Route | Purpose | MVP? |
|---|---|---|
| `/` | Dashboard (class balance, ingestion stats) | yes |
| `/clusters` | Cluster grid view, sidebar filter | yes |
| `/clusters/[id]` | Single cluster crop grid + DnD + bulk ops | yes |
| `/review` | Mismatch / Gemma low-conf / Outlier / Uncertainty review queues | yes |
| `/classes` | Add / rename / merge classes | post-MVP |
| `/export` | Trigger YOLO export, view balance gap | post-MVP |

## Keyboard shortcuts (every page)

| Key | Action |
|---|---|
| `1-9, 0` | Assign top-10 most frequent classes for context |
| `Enter` | Confirm selected + advance to next unvalidated |
| `G` | Accept Gemma suggestion for selected |
| `N` | Skip (defer to review queue) |
| `D` | Discard (mark for delete) |
| `Z` | Undo last action |
| `A` | Select all on page |
| `Shift+Enter` | Confirm all Gemma suggestions on page |
| `←/→` | Page navigation |
| `~` | Toggle keyboard shortcut overlay |
| `Esc` | Cancel in-progress drag |

## Data integrity

- Every label change → immediate API call. No "save" button.
- Optimistic UI with error rollback toast on API failure.
- Audit trail lives server-side in `op_vehicle_crops.{label_source,
  label_validated, class_source, updated_at}`.
- Bulk ops show a confirmation dialog with affected count.
- Test-set crops (`test_holdout=true`) are filtered out at the API level —
  the UI never receives them. Don't try to bypass.

## Development

```bash
npm install
npm run dev    # http://localhost:5173 (uses Vite default)
npm run check  # svelte-check + tsc
npm run build  # SvelteKit → /build (static)
```

The production build is consumed by a `nginx:alpine` container declared in
`openprocessor/docker-compose.legacy.yml` on port 5174 (host).

## Connecting to openprocessor

`PUBLIC_TRITON_API_URL` env var (default `http://localhost:4603`). All API
calls flow through `src/lib/api.ts` with retry + AbortController for in-flight
cancellation.

## Deployment

openprocessor serves all images and crop thumbnails — the labeler does NOT mount
NAS volumes. This avoids volume duplication and keeps NAS path knowledge in
one place. Image URLs look like `${PUBLIC_TRITON_API_URL}/curation/crops/{id}/thumbnail`.

## Style

- Dark theme by default (photographers work in dim environments).
- No emoji. No gradients. Apple system colors.
- Tailwind `bg-zinc-950` base, accent via CSS variables for easy retheme.

## Known constraints

- HTML5 DnD doesn't work reliably across all browsers/webviews. Use
  `svelte-dnd-action` (pointer-based) only.
- Don't fetch full-resolution NAS images in grids. Always use the thumbnail
  endpoint (128×128 LRU cached server-side).
- Test-holdout crops MUST never be relabeled by Gemma or via cluster
  auto-suggest. The openprocessor `/curation/` endpoints filter; UI is the second line.
