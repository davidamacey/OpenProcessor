/**
 * DQ-m9 (docs/design/data-quality-pass-2026-09-24.md §7 FRONTEND item 7):
 * the same "Total" label was used for two different served numbers —
 * `/classes`' `sample_count` (the class-cluster bucket size) and
 * `/export`'s per-class `stats/classes` count (every crop with that
 * class_id). They can disagree wildly (a region-bound class: thousands vs
 * 0) because regions are sub-boxes on other items, never a crop's own
 * class_id.
 * Fixed by labeling each column for what it actually counts.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('DQ-m9: /classes labels its Total column as the cluster bucket size', () => {
  it('the column header says "Total (in cluster)", not a bare "Total"', () => {
    expect(src).toMatch(/Total \(in cluster\)/);
    expect(src).not.toMatch(/<th class="px-3 py-2 text-right font-medium">Total<\/th>/);
  });

  it("carries a tooltip distinguishing it from /export's Total column", () => {
    const idx = src.indexOf('Total (in cluster)');
    const before = src.slice(Math.max(0, idx - 400), idx);
    expect(before).toMatch(/title="Class-cluster size \(cluster_size\)/);
  });
});
