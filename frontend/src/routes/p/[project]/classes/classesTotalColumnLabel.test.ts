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

describe('the /classes item-count column is labeled as the served item count', () => {
  it('the column header says "Items", not the cluster bucket size', () => {
    expect(src).toMatch(/>\s*Items\s*</);
    expect(src).not.toMatch(/in cluster/i);
    expect(src).not.toMatch(/cluster_size/);
  });

  it("carries a tooltip distinguishing it from /export's Total column", () => {
    const idx = src.indexOf('Items\n');
    const before = src.slice(Math.max(0, idx - 400), idx);
    expect(before).toMatch(/title="Items carrying this class/);
  });
});
