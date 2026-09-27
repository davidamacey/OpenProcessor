/**
 * DQ-m11 (docs/design/data-quality-pass-2026-09-24.md §7 FRONTEND item 7):
 * the class-balance strip used a plain `sort(validated_count desc)`,
 * which degenerated to the server's alphabetical per_class order whenever
 * every class tied at 0 validated (the live state throughout the audit),
 * hiding `widget_d`/`widget_b` behind small early-alphabet classes once sliced
 * to 30. See src/lib/dashboard/classBalance.test.ts for
 * sortClassBalance()'s full behavior coverage; this pins that the page
 * actually uses it.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');
// Visual audit D2 moved the chart into ClassBalanceChart /
// buildClassBalance(); the tiebreak must still go through sortClassBalance.
const balanceSrc = readFileSync(
  path.join(here, '../../../../lib/dashboard/classBalance.ts'),
  'utf-8',
);

describe('DQ-m11: the dashboard class-balance strip breaks validated_count ties by total count', () => {
  it('the page renders the shared chart over the served per_class rows', () => {
    expect(src).toMatch(/<ClassBalanceChart\s+rows=\{legacyStats\.per_class\}/);
    expect(src).not.toMatch(
      /\.sort\(\(a, b\) => b\.validated_count - a\.validated_count\)/,
    );
  });

  it('buildClassBalance orders rows with the shared, unit-tested sortClassBalance', () => {
    expect(balanceSrc).toMatch(/const sorted = sortClassBalance\(rows\);/);
  });
});
