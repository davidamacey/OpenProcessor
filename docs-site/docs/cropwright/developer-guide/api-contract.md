---
sidebar_position: 3
title: API contract vendoring
---

# API contract vendoring

Cropwright's picture of the backend's wire format is vendored from
OpenProcessor's own generated contract files, not hand-copied.

- **One copy**: the backend generates `contracts/` at the repository root (`json/item_wire.json`,
  `ts/*.ts`, `openapi/curation.json`) with `make contracts`, and a pre-commit hook rejects a stale
  copy. There is no vendored snapshot and no sync step.
- **Tests read it directly**: the frontend's contract tests import the files through the
  `$contracts` alias (`../contracts`), so a backend change that breaks the frontend fails a test in
  the same pull request.

A backend rename fails a frontend test instead of silently rendering
blanks or 404ing.
