/**
 * Regression tests for DQ-M4 (docs/design/data-quality-pass-2026-09-24.md
 * §7 FRONTEND item 2): at the default sort ("purity asc"), representatives
 * used to be requested via a server-order (`_count desc`) `offset`/`limit`
 * window that never lined up with the client-side sorted display order —
 * 4 of the first 8 cards rendered blank until scrolling forced the window
 * far enough to cover them.
 *
 * Same static source-scan convention as interactivePassFixes.test.ts — no
 * `@testing-library/svelte` mount harness for a page this size. The pure
 * selection logic (idsNeedingRepresentatives) has its own full behavior
 * suite in src/lib/clusters/displayOrderRepresentatives.test.ts; these
 * tests pin that the page actually wires it in display order rather than
 * server order.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

function fn(name: string): string {
  const m = src.match(new RegExp(`async function ${name}\\([\\s\\S]*?\\n {2}\\}`));
  expect(m, `function ${name} not found`).toBeDefined();
  return m![0];
}

describe('DQ-M4: representatives are windowed over gridItems (display order), not the raw server list', () => {
  it('loadMoreRepresentatives windows idsNeedingRepresentatives(gridItems, ...), not clusterPager.items', () => {
    const body = fn('loadMoreRepresentatives');
    expect(body).toMatch(/idsNeedingRepresentatives\(gridItems, windowStart, pageSize\)/);
    expect(body).not.toMatch(/idsNeedingRepresentatives\(clusterPager\.items/);
  });

  it('fetches each missing card individually by cluster_id, not a stale offset/limit window', () => {
    const body = fn('loadMoreRepresentatives');
    expect(body).toMatch(/getClusters\(\{ cluster_id: id, representatives_limit: 1 \}\)/);
    expect(body).not.toMatch(/representatives_offset:\s*repsOffset/);
  });

  it('the main card-list fetch no longer requests a representatives window at all (superseded by display-order fetch)', () => {
    const clusterQueryFn = src.match(
      /function clusterQuery\(page: number\): ClusterFilter \{[\s\S]*?\n {2}\}/,
    )?.[0];
    expect(clusterQueryFn).toBeDefined();
    expect(clusterQueryFn).toMatch(/representatives_offset:\s*0,/);
    expect(clusterQueryFn).toMatch(/representatives_limit:\s*0,/);
  });

  it('loadFirst fetches the first display-order window before the operator ever sees the grid', () => {
    const body = fn('loadFirst');
    expect(body).toMatch(/dispOffset = 0;/);
    expect(body).toMatch(/await loadMoreRepresentatives\(\);/);
  });

  it('sort/unlabeledOnly changes reset dispOffset and re-cover the new first window, without refetching the card list', () => {
    const idx = src.indexOf('void sort;\n    void unlabeledOnly;');
    expect(
      idx,
      'the sort/unlabeledOnly display-order effect was not found',
    ).toBeGreaterThan(-1);
    const slice = src.slice(idx, idx + 200);
    expect(slice).toMatch(/dispOffset = 0;/);
    expect(slice).toMatch(/void loadMoreRepresentatives\(\);/);
    expect(slice).not.toMatch(/loadFirst\(\)/);
  });
});
