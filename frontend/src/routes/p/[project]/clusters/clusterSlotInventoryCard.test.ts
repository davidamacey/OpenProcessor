/**
 * The synthetic slot inventory card on /clusters invents nothing and never
 * hides a real cluster. Static source scan (no component-mount harness; see
 * clusters/[id]/clusterMoveRace.test.ts's doc comment).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const LIB = path.resolve(here, '../../../../lib');
const pageSrc = readFileSync(path.join(here, '+page.svelte'), 'utf-8');
const inventorySrc = readFileSync(
  path.join(LIB, 'clusters/slotInventory.svelte.ts'),
  'utf-8',
);
const listSrc = readFileSync(
  path.join(LIB, 'clusters/clusterListController.svelte.ts'),
  'utf-8',
);
const gridSrc = readFileSync(
  path.join(LIB, 'components/clusters/ClustersGrid.svelte'),
  'utf-8',
);
const cardSrc = readFileSync(
  path.join(LIB, 'components/clusters/ClusterGridCard.svelte'),
  'utf-8',
);

describe('the synthetic slot inventory card invents nothing and never hides a real cluster', () => {
  it('buildSlotInventoryCard builds a fully-typed Cluster (no `as Cluster` cast hiding missing fields)', () => {
    const fn = inventorySrc.match(
      /async function buildSlotInventoryCard\([\s\S]*?\n {2}\}/,
    )?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).not.toMatch(/as Cluster/);
    expect(fn).toMatch(/isSlotCard: true/);
    expect(fn).toMatch(/purity_tier: null/);
  });

  it('re-runs whenever classesStore.classes changes, not just once after the first loadFirst (the ~1-in-8 race)', () => {
    expect(pageSrc).toMatch(
      /slotInventory\.slotInventoryCards\.length === 0 &&\s*\n\s*classesStore\.classes\.length > 0/,
    );
  });

  it('never drops the real cluster sharing the slot class’s id from the grid', () => {
    const gridItemsFn = listSrc.match(
      /const gridItems = \$derived\.by<Cluster\[\]>\(\(\) => \{[\s\S]*?\n {2}\}\);/,
    )?.[0];
    expect(gridItemsFn).not.toBeUndefined();
    // The old bug filtered the real cluster out via
    // `sorted.filter((c) => c.id !== slotInventoryCard!.id)`.
    expect(gridItemsFn).not.toMatch(/filter\(\(c\) => c\.id !== slotInventoryCard/);
    expect(gridItemsFn).toMatch(
      /return \[\.\.\.slotInventory\.slotInventoryCards, \.\.\.cards\];/,
    );
  });

  it('keys the #each off isSlotCard so the synthetic card can never collide with a real cluster’s key', () => {
    expect(gridSrc).toMatch(
      /\{#each list\.gridItems as c \(c\.isSlotCard \? `slot-\$\{c\.id\}` : c\.id\)\}/,
    );
  });

  it('purityBadge renders no badge at all for the synthetic card (never an invented "noisy 0%")', () => {
    const fn = cardSrc.match(/function purityBadge\(c: Cluster\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/if \(c\.isSlotCard\) return null;/);
  });

  it('the subtitle row skips the invented dominant_pct "· 0%" for the synthetic card too', () => {
    expect(cardSrc).toMatch(/\{#if c\.isSlotCard\}/);
    expect(cardSrc.match(/\{#if c\.isSlotCard\}[\s\S]{0,80}/)?.[0]).not.toMatch(
      /dominant_pct/,
    );
  });
});
