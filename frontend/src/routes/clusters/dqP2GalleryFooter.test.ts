/**
 * DQ-p2 (docs/design/data-quality-pass-2026-09-24.md §7 FRONTEND item 7):
 * the /clusters status bar always read off `clusterPager` (the cluster-
 * grid pager) — "102 / 102 all loaded" under a much larger slot-gallery
 * grid, because the slot-gallery view (`isSlotFilter`) renders
 * `slotGallery.pager.items`, a wholly different pager, but the
 * footer never switched to match it.
 *
 * Same static source-scan convention as the other clusters/+page.svelte
 * regression tests — no mount harness for a page this size.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('DQ-p2: the status bar reads the pager that actually backs the visible grid', () => {
  it('branches on slotGallery to read its pager in gallery mode', () => {
    const idx = src.indexOf('DQ-p2 (docs/design/data-quality-pass-2026-09-24.md)');
    expect(idx).toBeGreaterThan(-1);
    const slice = src.slice(idx, idx + 1200);
    expect(slice).toMatch(/\{#if slotGallery\}/);
    expect(slice).toMatch(
      /\{slotGallery\.pager\.items\.length\} \/ \{slotGallery\.pager\.total\}/,
    );
    expect(slice).toMatch(/clusterPager\.total/);
  });
});
