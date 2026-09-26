---
sidebar_position: 3
title: API contract vendoring
---

# API contract vendoring

Cropwright's picture of the backend's wire format is vendored from
OpenProcessor's own generated contract files, not hand-copied.

- **Vendored snapshot**: `contracts/openprocessor/` (`json/item_wire.json`,
  `ts/*.ts`, `openapi/curation.json`, `SOURCE.md` recording the backend
  commit it came from).
- **Sync**: `npm run contract:sync` refreshes the snapshot from a local
  OpenProcessor checkout (`OPENPROCESSOR_REPO`, default `../OpenProcessor`).
- **Check**: `npm run contract:check` diffs the vendored copy against that
  ref and fails on drift; it's a no-op (exit 0) when the backend repo isn't
  present, so it's harmless in CI.
- **Tests read the vendored files** directly — `wireKeys.test.ts`,
  `regionStatus.test.ts`, `classSources.test.ts`, `endpointCatalog.test.ts`
  (the last one mechanically extracts every `${API_PREFIX}/...` call site
  from `api.ts`/`sse.ts`/a couple of components and resolves it against the
  vendored OpenAPI spec).

A backend rename fails a frontend test instead of silently rendering
blanks or 404ing.
