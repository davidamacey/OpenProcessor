---
sidebar_position: 3
title: Architecture overview
---

# Architecture overview

Cropwright is a **thin frontend**. It displays and sorts; the backend owns
every decision (clustering, scoring, model training) and serves the result
as data. See the [architecture diagrams](./architecture-diagrams.mdx) for the full picture.

## Frontend

- **SvelteKit 3 + Svelte 5 runes + TypeScript strict + Tailwind v4.**
- Pointer-event drag-and-drop via `svelte-dnd-action` (HTML5 drag-and-drop
  is unreliable across browsers/webviews and is not used).
- State lives in Svelte 5 runes (`$state`/`$derived`/`$effect`) — nothing in
  `localStorage` that can't be reconstructed by an API call.
- Built as a static site (SvelteKit's static adapter) and served by nginx in
  production.

## One API prefix, one proxy

The browser only ever talks to Cropwright's own nginx origin. nginx proxies
everything under one path prefix (`PUBLIC_API_PREFIX`, default `/curation`)
to the OpenProcessor API container by Docker network name — no CORS, and no
backend hostname is ever exposed to the browser.

```
Browser  --same-origin-->  nginx (Cropwright, :8080 non-root)  --docker network-->  OpenProcessor API
```

## Projects and the config store

One deployment hosts many **projects**. The active project is the URL
(`/p/<project>/...`), and every project-scoped request goes to that project's
own served prefix under the one API prefix. Switching project drops
everything cached for the previous one (classes, stores, undo history), and
any response that arrives late for the old project is discarded.

What a project is configured with, its prompt pack, region profile, active
VLM endpoint and keymap, is stored by the backend as revisioned,
activate-able documents ("the config store"); Cropwright is an editor and
viewer for them. The VLM endpoint registry is shared by every project, with
the active choice per project. See
[Runtime settings](../configuration/runtime-settings.md).

## The served region profile

Cropwright ships with **no domain built in**. At most one "region" concept
(a sub-box within an item crop — a license plate on a vehicle, a tag on a
part, a defect zone) is active at a time, and it's entirely defined by what
the backend serves on `GET {API_PREFIX}/health` as `region_profile`. With no
profile served, every region-specific surface (a review tab, a gallery, a
sub-box editor) simply doesn't render — absent, not disabled.

A region item can carry any number of boxes; see
[Multi-box regions](../user-guide/multi-box-regions.md).

A deployment can add further, backend-unaware annotation slots via a small
JSON file (`annotation-profiles.json`) placed next to the built app. See
[Annotation profiles](../configuration/annotation-profiles.md).

## Everything degrades gracefully

Every optional feature (score chips, the embedding-plot lasso tool,
semantic search, the region surfaces, the single-class export) is gated on
what the backend serves at runtime — its own `/methods` capability list or
the `/health` region profile — never a Cropwright build flag. A backend
that hasn't enabled a feature simply doesn't get a page section for it.

## Absent, not disabled

Newer backend features (dataset import, prompt-pack and region-profile
editors, the VLM registry, project combine, model sharing, the keymap) are
each probed once and shown only when served, so one Cropwright build works
against backends at different versions.

## No database, no auth, of its own

Cropwright keeps no state beyond what the current page needs. It also
implements **no authentication** — see [Security](../operations/security.md)
before deploying anywhere but a trusted network.
