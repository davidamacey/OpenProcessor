/**
 * DQ-M2 fix (dq-queues cutover, 2026-09-24): `/clusters` and
 * `/clusters/[id]` now render the served nearest-centroid geometry
 * purity (`purity`, `purity_basis`, `purity_n`) alongside the old
 * label-based `label_purity`/`labelled_share`, instead of showing only
 * the (tautological, always-1.0-for-a-class-cluster) label-based number.
 * `purity_tier` stays the pure/mixed/noisy badge — unchanged by this.
 *
 * Static source scan (no component-mount harness for these pages — see
 * clusters/[id]/clusterMoveRace.test.ts's doc comment).
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const here = path.dirname(fileURLToPath(import.meta.url));
const gridSrc = readFileSync(path.join(here, '+page.svelte'), 'utf-8');
const detailSrc = readFileSync(path.join(here, '[id]/+page.svelte'), 'utf-8');

describe('/clusters cards show purity_n/purity_basis and label_purity/labelled_share', () => {
  it('the purity chip shows purity_n next to the percentage', () => {
    const idx = gridSrc.indexOf('{pb.text}');
    expect(idx).toBeGreaterThan(-1);
    const block = gridSrc.slice(idx, idx + 300);
    expect(block).toMatch(/c\.purity_n/);
  });

  it('the purity chip tooltip carries purity_basis, label_purity and labelled_share', () => {
    const idx = gridSrc.indexOf('title="{c.purity_basis');
    expect(idx).toBeGreaterThan(-1);
    const block = gridSrc.slice(idx, idx + 400);
    expect(block).toMatch(/c\.purity_n/);
    expect(block).toMatch(/c\.label_purity/);
    expect(block).toMatch(/c\.labelled_share/);
  });

  it('purity_tier is still what drives the badge color/text (unchanged by DQ-M2)', () => {
    expect(gridSrc).toMatch(/c\.purity_tier === 'pure'/);
    expect(gridSrc).toMatch(/c\.purity_tier === 'mixed'/);
  });
});

describe('/clusters/[id] header shows purity with basis/n and label_purity/labelled_share', () => {
  it('renders cluster.purity gated on non-null, with purity_basis and purity_n', () => {
    const idx = detailSrc.indexOf('{#if cluster.purity != null}');
    expect(idx).toBeGreaterThan(-1);
    const block = detailSrc.slice(idx, idx + 700);
    expect(block).toMatch(/cluster\.purity_basis/);
    expect(block).toMatch(/cluster\.purity_n/);
  });

  it('carries label_purity/labelled_share in the tooltip', () => {
    const idx = detailSrc.indexOf('{#if cluster.purity != null}');
    const block = detailSrc.slice(idx, idx + 700);
    expect(block).toMatch(/cluster\.label_purity/);
    expect(block).toMatch(/cluster\.labelled_share/);
  });
});
