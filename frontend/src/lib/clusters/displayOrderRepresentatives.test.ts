import { describe, expect, it } from 'vitest';
import {
  idsNeedingRepresentatives,
  type DisplayCardLike,
} from './displayOrderRepresentatives';

function card(id: number, reps: string[] = [], isSlotCard = false): DisplayCardLike {
  return { id, representative_crop_ids: reps, isSlotCard };
}

describe('idsNeedingRepresentatives (DQ-M4)', () => {
  it('returns ids in the requested window that have no representatives yet', () => {
    const list = [card(1), card(2, ['c1']), card(3), card(4)];
    expect(idsNeedingRepresentatives(list, 0, 3)).toEqual([1, 3]);
  });

  it('reproduces the DQ-M4 repro: purity-asc display order differs from size-desc fetch order', () => {
    // Server (size-desc) order: 10, 20, 30, 40 — the old windowed offset
    // filled representatives for these first. The operator's chosen sort
    // (purity asc) displays a different order first.
    const displayOrder = [card(40), card(20, ['c']), card(30), card(10, ['c'])];
    // Only 40 and 30 are missing in the NEW display order's first window —
    // this is exactly what should be requested, not [10, 20] (the stale
    // server-order window).
    expect(idsNeedingRepresentatives(displayOrder, 0, 4)).toEqual([40, 30]);
  });

  it('never asks for the synthetic widget_tag card (no real cluster_id to query)', () => {
    const list = [card(1, [], true), card(2)];
    expect(idsNeedingRepresentatives(list, 0, 2)).toEqual([2]);
  });

  it('respects windowStart/windowSize — scrolled-past cards are not re-requested', () => {
    const list = [card(1), card(2), card(3), card(4), card(5)];
    expect(idsNeedingRepresentatives(list, 2, 2)).toEqual([3, 4]);
  });

  it('returns an empty array once every card in the window already has representatives', () => {
    const list = [card(1, ['a']), card(2, ['b'])];
    expect(idsNeedingRepresentatives(list, 0, 2)).toEqual([]);
  });
});
