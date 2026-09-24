import { describe, expect, it } from 'vitest';
import { computeCutLine } from './cutLine';

function crops(...vals: Array<boolean | null>): { cluster_is_core: boolean | null }[] {
  return vals.map((v) => ({ cluster_is_core: v }));
}

describe('computeCutLine (DQ-M3 frontend half)', () => {
  it('draws a clean boundary when the order really is core-first', () => {
    const r = computeCutLine(crops(true, true, true, false, false));
    expect(r).toEqual({ index: 3, visible: true });
  });

  it('hides the line when most members have a null cluster_is_core (class-cluster repro)', () => {
    // 4 of 5 null, mirrors the design doc's ~85% null rate on class
    // clusters.
    const r = computeCutLine(crops(null, null, null, null, true));
    expect(r.visible).toBe(false);
  });

  it('hides the line when a later crop is core again — order is not core-first (candidate-cluster repro)', () => {
    // Mirrors #10000: first non-core at index 1, core crops resume later.
    // Every member has cluster_is_core set (not null), so the
    // mostly-null guard alone would NOT catch this — the order check is
    // the part that does.
    const r = computeCutLine(crops(true, false, false, true, true));
    expect(r.visible).toBe(false);
  });

  it('hides the line when every loaded crop is core (no boundary to draw)', () => {
    const r = computeCutLine(crops(true, true, true));
    expect(r).toEqual({ index: 3, visible: false });
  });

  it('hides the line when the very first crop is already non-core (index 0, nothing "core" to mark off)', () => {
    const r = computeCutLine(crops(false, false, true));
    // Half-null threshold not tripped (0 nulls), but a 0 boundary is
    // degenerate — nothing precedes it to call "the core section".
    expect(r.visible).toBe(false);
    expect(r.index).toBe(0);
  });

  it('hides the line on an empty crop list', () => {
    expect(computeCutLine([])).toEqual({ index: 0, visible: false });
  });

  it('treats undefined the same as null for the mostly-null check', () => {
    const r = computeCutLine([
      { cluster_is_core: undefined },
      { cluster_is_core: undefined },
      { cluster_is_core: undefined },
      { cluster_is_core: true },
    ]);
    expect(r.visible).toBe(false);
  });

  it('a minority-null cluster with a real core-first order still draws the line', () => {
    // 1 of 6 null (well under the 50% threshold) — the rest are a clean
    // core-first order.
    const r = computeCutLine(crops(true, true, null, false, false, false));
    expect(r).toEqual({ index: 2, visible: true });
  });
});
