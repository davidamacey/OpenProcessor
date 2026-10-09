---
sidebar_position: 2
title: Testing
---

# Testing

Four layers, from fastest to slowest.

## Unit / component (`npm test`)

Vitest, with `resolve.conditions: ['browser']` under Vitest so Svelte 5's
real browser build resolves under jsdom — a `.test.ts` can `mount()` a
component directly rather than regex-scanning source. Controllers
(`reviewController`, `clusterController`, `slotGalleryController`,
`ingestRunController`) are tested directly, not by mounting the whole page.

## Stubbed end-to-end (`npm run test:e2e`)

`e2e/stubbed/` — Playwright + pytest against a `vite preview` build, with
every `{API_PREFIX}` request intercepted by a fail-closed stub (an
unregistered request gets a `501` and fails the test, rather than a
silent `200 {}`). Never touches a live backend.

## Live read-only tier (`npm run test:live`)

`e2e/live/` — drives the real deployed build against a **real**
OpenProcessor backend, structurally read-only (every non-GET/HEAD request
is aborted and any attempt fails the test). Skipped entirely unless
`CROPWRIGHT_LIVE_URL` is set — never runs in CI. Also saves full-page
screenshots per route at two viewports; **a human must actually look at
them** after a run — the assertions catch what's mechanically checkable,
not a layout regression.

## Mutation testing (`npm run test:mutation`)

Stryker, against the ~10 highest-value pure modules (wire mapping, undo
store, dataset stats, review tabs, `curationSettings.ts`, `strategies.ts`,
`classPicker.ts`, `readSlot.ts`). Answers "if I break this line, does a
test notice" rather than "does every assertion pass." Runs on a schedule,
not per-push — too slow for that.

## New test discipline

Verify a new test actually fails against a mutated copy of the code it
covers before trusting it (edit a scratch copy, confirm red, restore
byte-for-byte).
