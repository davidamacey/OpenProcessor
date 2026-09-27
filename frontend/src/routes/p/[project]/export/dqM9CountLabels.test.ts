/**
 * DQ-m9 (docs/design/data-quality-pass-2026-09-24.md §7 FRONTEND item 7)
 * — the /export half. See src/routes/classes/dqM9CountLabels.test.ts for
 * the /classes half of the same fix.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('DQ-m9: /export labels its Total column as the labelled (class_id) count', () => {
  it('the column header says "Total (labelled)", not a bare "Total"', () => {
    expect(src).toMatch(/>Total \(labelled\)<\/th/);
  });

  it('carries a tooltip distinguishing it from the sidebar/classes Total column', () => {
    const idx = src.indexOf('Total (labelled)');
    const before = src.slice(Math.max(0, idx - 400), idx);
    expect(before).toMatch(/title="Crops with this class_id/);
  });
});
