/**
 * W7 (Ignored bucket) / W8 (item-text search) on `/clusters`
 * (docs/design/logic-moves-adoption-plan-2026-09-24.md). No
 * `@testing-library/svelte` in this repo (see `searchMode.test.ts`'s
 * header comment for the established convention this file follows), so
 * this is a static source scan of the load-bearing wiring:
 *
 *  - the Ignored-bucket grid branch calls `unexcludeCrops` on restore,
 *    not `excludeCrops` (which would re-ignore instead of restoring)
 *  - the Ignored-bucket fetch passes `include_excluded: true` — omitting
 *    it would return an empty bucket forever, since the default crop
 *    browse filters `class_excluded` out server-side
 *  - the item-text search fetch uses the `item_text` param, and its 400
 *    branch renders an inline hint rather than a toast
 *  - both new mode branches render before the `showEmbeddingViz` branch,
 *    matching the existing `searchModeActive` precedent
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.resolve(here, '+page.svelte'), 'utf-8');

describe('/clusters Ignored bucket (W7)', () => {
  it('restoreIgnored calls unexcludeCrops, not excludeCrops', () => {
    const fnStart = src.indexOf('async function restoreIgnored');
    expect(fnStart).toBeGreaterThan(-1);
    const fnEnd = src.indexOf('\n  }', fnStart);
    const body = src.slice(fnStart, fnEnd);
    expect(body).toMatch(/unexcludeCrops\(ids\)/);
    expect(body).not.toMatch(/(?<!un)excludeCrops\(ids/);
  });

  it('loadIgnored requests include_excluded so the bucket is not permanently empty', () => {
    const fnStart = src.indexOf('async function loadIgnored');
    expect(fnStart).toBeGreaterThan(-1);
    const fnEnd = src.indexOf('\n  }', fnStart);
    const body = src.slice(fnStart, fnEnd);
    expect(body).toMatch(/include_excluded:\s*true/);
    expect(body).toMatch(/cluster_id:\s*-2/);
  });

  it('the ignoredModeActive grid branch renders before showEmbeddingViz (searchModeActive precedent)', () => {
    const ignoredBranch = src.indexOf('{:else if ignoredModeActive}');
    const embeddingBranch = src.indexOf('{:else if showEmbeddingViz}');
    expect(ignoredBranch).toBeGreaterThan(-1);
    expect(embeddingBranch).toBeGreaterThan(-1);
    expect(ignoredBranch).toBeLessThan(embeddingBranch);
  });
});

describe('/clusters item-text search (W8)', () => {
  it('runItemTextSearch sends item_text and handles a 400 as an inline hint, not a toast', () => {
    const fnStart = src.indexOf('async function runItemTextSearch');
    expect(fnStart).toBeGreaterThan(-1);
    const fnEnd = src.indexOf('\n  }', fnStart);
    const body = src.slice(fnStart, fnEnd);
    expect(body).toMatch(/item_text:\s*q/);
    expect(body).toMatch(/e\.status === 400/);
    expect(body).toMatch(/itemTextError\s*=/);
    // A 400 must NOT also fire a toast — that's the non-400 branch's job.
    const status400Branch = body.slice(
      body.indexOf('e.status === 400'),
      body.indexOf('} else {'),
    );
    expect(status400Branch).toMatch(/itemTextError\s*=/);
    expect(status400Branch).not.toMatch(/toastStore\.error/);
  });

  it('the itemTextModeActive grid branch renders before showEmbeddingViz', () => {
    const itemTextBranch = src.indexOf('{:else if itemTextModeActive}');
    const embeddingBranch = src.indexOf('{:else if showEmbeddingViz}');
    expect(itemTextBranch).toBeGreaterThan(-1);
    expect(embeddingBranch).toBeGreaterThan(-1);
    expect(itemTextBranch).toBeLessThan(embeddingBranch);
  });
});
