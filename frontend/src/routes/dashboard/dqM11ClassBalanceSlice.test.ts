/**
 * DQ-m11 (docs/design/data-quality-pass-2026-09-24.md §7 FRONTEND item 7):
 * the class-balance strip used a plain `sort(validated_count desc)`,
 * which degenerated to the server's alphabetical per_class order whenever
 * every class tied at 0 validated (the live state throughout the audit),
 * hiding `pickup`/`suv` behind small early-alphabet classes once sliced
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

describe('DQ-m11: the dashboard class-balance strip breaks validated_count ties by total count', () => {
  it('balance uses sortClassBalance(), not a raw validated_count-only sort', () => {
    expect(src).toMatch(/return sortClassBalance\(legacyStats\.per_class\)/);
    expect(src).not.toMatch(
      /\.sort\(\(a, b\) => b\.validated_count - a\.validated_count\)/,
    );
  });

  it('imports the shared, unit-tested helper rather than reimplementing the tiebreak inline', () => {
    expect(src).toMatch(
      /import \{ sortClassBalance \} from '\$lib\/dashboard\/classBalance';/,
    );
  });
});
