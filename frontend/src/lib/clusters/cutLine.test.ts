import { describe, expect, it } from 'vitest';
import { computeCutLine } from './cutLine';

function crops(...vals: Array<boolean | null | undefined>) {
  return vals.map((v) => ({ cluster_is_core: v }));
}

describe('computeCutLine (order=core_first)', () => {
  it('cuts at the first item the server serves as cluster_is_core === false', () => {
    expect(computeCutLine(crops(true, true, true, false, false), true)).toEqual({
      index: 3,
      visible: true,
    });
  });

  it('draws the line even when most members have a null cluster_is_core (no client null-share heuristic)', () => {
    expect(computeCutLine(crops(true, null, null, null, false), true)).toEqual({
      index: 4,
      visible: true,
    });
  });

  it('skips null and undefined members when looking for the first false', () => {
    expect(computeCutLine(crops(true, undefined, null, false), true)).toEqual({
      index: 3,
      visible: true,
    });
  });

  it('hides the line when no item is served as non-core', () => {
    expect(computeCutLine(crops(true, true, null), true)).toEqual({
      index: 0,
      visible: false,
    });
  });

  it('hides the line when the very first item is already non-core (no core section)', () => {
    expect(computeCutLine(crops(false, true), true)).toEqual({
      index: 0,
      visible: false,
    });
  });

  it('hides the line for any other order (the cut only means something in core_first order)', () => {
    expect(computeCutLine(crops(true, false), false)).toEqual({
      index: 0,
      visible: false,
    });
  });

  it('hides the line on an empty crop list', () => {
    expect(computeCutLine([], true)).toEqual({ index: 0, visible: false });
  });
});
