/**
 * The same "Total" label used to name two different served numbers:
 * `/classes`' `sample_count` (the class-cluster bucket size) and
 * `/export`'s per-class `stats/classes` count (every crop with that
 * class_id). They can disagree wildly (a region-bound class: thousands vs
 * 0) because regions are sub-boxes on other items, never a crop's own
 * class_id. Each column is labeled for what it counts; this is the /classes
 * half (see ../export/exportTotalColumnLabel.test.ts).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const LIB = path.resolve(here, '../../../../lib');
const src = readFileSync(
  path.join(LIB, 'components/classes/ClassRegistryTable.svelte'),
  'utf-8',
);

describe('the /classes Total column is labeled as the cluster bucket size', () => {
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
