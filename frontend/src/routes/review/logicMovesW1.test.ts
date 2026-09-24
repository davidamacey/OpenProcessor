/**
 * W1 (docs/design/logic-moves-adoption-plan-2026-09-24.md) — the
 * un-dismiss panel, reversing discard()'s reviewDismissCrop via the
 * backend's new POST {API_PREFIX}/crops/{id}/review_undismiss.
 *
 * This repo has no component-mount harness (see slotReviewCharacterization
 * .test.ts's doc comment); pinned via the same static source-scan
 * convention.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe("W1: un-dismiss panel reverses discard()'s reviewDismissCrop", () => {
  it('toggleDismissedPanel loads {API_PREFIX}/crops?review_dismissed=true', () => {
    const fn = src.match(/async function toggleDismissedPanel\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(/getCrops\(\{\s*review_dismissed: true/);
  });

  it('undismiss calls reviewUndismissCrop and drops the item from the local list', () => {
    const fn = src.match(/async function undismiss\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).toBeDefined();
    expect(fn).toMatch(/await reviewUndismissCrop\(crop\.id\)/);
  });
});
