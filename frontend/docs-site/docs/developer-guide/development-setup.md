---
sidebar_position: 1
title: Development setup
---

# Development setup

```bash
npm install
cp .env.example .env   # point PUBLIC_TRITON_API_URL at your backend
npm run dev              # http://localhost:5173
```

A running OpenProcessor backend (OpenSearch + the curation API, served
under `PUBLIC_API_PREFIX`) is required for the app to do anything useful —
there is no local database or mock-data mode.

## Building the Docker image from source

`docker-compose.yml` only pulls the published image. To run your own
build instead, add the build overlay (it tags the result
`cropwright-dev:local`, so it never overwrites a pulled image):

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build
```

## Before opening a PR

```bash
npm run check   # svelte-check + tsc, strict mode, must be 0 errors
npm run lint     # prettier + eslint
npm test         # vitest
npm run build    # production build must succeed
```

CI runs the same four, plus the stubbed Playwright suite
(`npm run test:e2e`) and a gitleaks secret scan.

## Conventions

See `CLAUDE.md` in the repo for the authoritative reference on
architecture (SvelteKit 2, Svelte 5 runes only), the route table, keyboard
shortcut reservations, and style (2-space indent, dark theme only, no
emoji, no gradients, Apple system colors only). If you change a route, a
keyboard shortcut, or add a feature flag, update `CLAUDE.md`'s tables in
the same PR.
