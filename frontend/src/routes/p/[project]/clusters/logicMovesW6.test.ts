/**
 * W6 (docs/design/logic-moves-adoption-plan-2026-09-24.md) — cluster
 * cards render the served purity_tier/promotable instead of recomputing a
 * 0.8/0.6 purity threshold client-side. Static source scan (no
 * component-mount harness in this repo — see
 * clusters/[id]/clusterMoveRace.test.ts's doc comment).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('W6: cluster cards use the served purity_tier/promotable, not a client threshold', () => {
  it('borderColor/purityBadge branch on c.purity_tier, not a numeric cutoff', () => {
    const borderFn = src.match(/function borderColor\(c: Cluster\)[\s\S]*?\n {2}\}/)?.[0];
    const badgeFn = src.match(/function purityBadge\(c: Cluster\)[\s\S]*?\n {2}\}/)?.[0];
    expect(borderFn).not.toBeUndefined();
    expect(badgeFn).not.toBeUndefined();
    expect(borderFn).toMatch(/c\.purity_tier === 'pure'/);
    expect(borderFn).toMatch(/c\.purity_tier === 'mixed'/);
    expect(badgeFn).toMatch(/c\.purity_tier === 'pure'/);
    expect(badgeFn).toMatch(/c\.purity_tier === 'mixed'/);
    // The old thresholds — must not survive anywhere in either function.
    expect(borderFn).not.toMatch(/0\.8/);
    expect(borderFn).not.toMatch(/0\.6/);
    expect(badgeFn).not.toMatch(/0\.8/);
    expect(badgeFn).not.toMatch(/0\.6/);
  });

  it('the card renders a promotable badge from c.promotable', () => {
    expect(src).toMatch(/\{#if c\.promotable\}/);
  });
});
