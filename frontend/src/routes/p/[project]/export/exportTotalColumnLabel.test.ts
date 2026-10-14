/**
 * The /export half of the Total-column labeling fix: /export's per-class
 * count is every crop with that class_id, not the /classes cluster bucket
 * size. See src/routes/p/[project]/classes/classesTotalColumnLabel.test.ts
 * for the /classes half.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(
  path.resolve(here, '../../../../lib/components/export/ExportClassTable.svelte'),
  'utf-8',
);

describe('the /export Total column is labeled as the labelled (class_id) count', () => {
  it('the column header says "Total (labelled)", not a bare "Total"', () => {
    expect(src).toMatch(/>Total \(labelled\)<\/th/);
  });

  it('carries a tooltip distinguishing it from the sidebar/classes Total column', () => {
    const idx = src.indexOf('Total (labelled)');
    const before = src.slice(Math.max(0, idx - 400), idx);
    expect(before).toMatch(/title="Crops with this class_id/);
  });
});
