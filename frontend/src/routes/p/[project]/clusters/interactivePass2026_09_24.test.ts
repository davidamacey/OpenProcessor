/**
 * Source-scan coverage for the /clusters frontend fixes in
 * docs/design/interactive-pass-2026-09-24.md §6 FRONTEND (M4, m22, m18).
 * No component-mount harness exists for this route (see
 * clusters/[id]/clusterMoveRace.test.ts's doc comment, and
 * logicMovesW6.test.ts in this directory for the established
 * static-source-scan convention this file follows).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = readFileSync(path.join(here, '+page.svelte'), 'utf-8');

describe('M4: the synthetic slot inventory card invents nothing and never hides a real cluster', () => {
  it('buildSlotInventoryCard builds a fully-typed Cluster (no `as Cluster` cast hiding missing fields)', () => {
    const fn = src.match(/async function buildSlotInventoryCard\([\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).not.toMatch(/as Cluster/);
    expect(fn).toMatch(/isSlotCard: true/);
    expect(fn).toMatch(/purity_tier: null/);
  });

  it('re-runs whenever classesStore.classes changes, not just once after the first loadFirst (the ~1-in-8 race)', () => {
    expect(src).toMatch(
      /slotInventoryCards\.length === 0 &&\s*\n\s*classesStore\.classes\.length > 0/,
    );
  });

  it('never drops the real cluster sharing the slot class’s id from the grid', () => {
    const gridItemsFn = src.match(
      /const gridItems = \$derived\.by<Cluster\[\]>\(\(\) => \{[\s\S]*?\n {2}\}\);/,
    )?.[0];
    expect(gridItemsFn).not.toBeUndefined();
    // The old bug filtered the real cluster out via
    // `sorted.filter((c) => c.id !== slotInventoryCard!.id)`.
    expect(gridItemsFn).not.toMatch(/filter\(\(c\) => c\.id !== slotInventoryCard/);
    expect(gridItemsFn).toMatch(/return \[\.\.\.slotInventoryCards, \.\.\.sorted\];/);
  });

  it('keys the #each off isSlotCard so the synthetic card can never collide with a real cluster’s key', () => {
    expect(src).toMatch(
      /\{#each gridItems as c \(c\.isSlotCard \? `slot-\$\{c\.id\}` : c\.id\)\}/,
    );
  });

  it('purityBadge renders no badge at all for the synthetic card (never an invented "noisy 0%")', () => {
    const fn = src.match(/function purityBadge\(c: Cluster\)[\s\S]*?\n {2}\}/)?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/if \(c\.isSlotCard\) return null;/);
  });

  it('the subtitle row skips the invented dominant_pct "· 0%" for the synthetic card too', () => {
    expect(src).toMatch(/\{#if c\.isSlotCard\}/);
    expect(src.match(/\{#if c\.isSlotCard\}[\s\S]{0,80}/)?.[0]).not.toMatch(
      /dominant_pct/,
    );
  });
});

describe('m22: a name-form ?class= deep link resolves against the loaded registry', () => {
  it('classFilter falls back to the normalized active-class name lookup when the param is not numeric', () => {
    const fn = src.match(
      /const classFilter = \$derived\.by\(\(\) => \{[\s\S]*?\n {2}\}\);/,
    )?.[0];
    expect(fn).not.toBeUndefined();
    expect(fn).toMatch(/findActiveClassByName\(classesStore\.classes, v\)/);
  });
});

describe('m18: item-text search shows only the error, not the error AND the empty state', () => {
  it('the empty-state copy is gated on !itemTextError', () => {
    const block = src.match(
      /\{#if itemTextLoading[\s\S]*?No crops matched that text\.[\s\S]*?\{:else\}/,
    )?.[0];
    expect(block).not.toBeUndefined();
    expect(block).toMatch(/\{:else if itemTextError\}/);
  });
});
