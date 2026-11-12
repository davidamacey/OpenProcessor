import { describe, expect, it } from 'vitest';
import {
  NO_SELECTION,
  pressSelectAll,
  selectionKindFor,
  toItemSelection,
  type WorkspaceSelection,
} from './selection';

const FILTER = { class_names: ['widget'], review_tab: 'to_label' };

describe('toItemSelection', () => {
  it('sends nothing for no selection', () => {
    expect(toItemSelection(NO_SELECTION, FILTER)).toBeNull();
  });

  it('sends explicit crop ids for items', () => {
    expect(toItemSelection({ kind: 'items', cropIds: ['a', 'b'] }, FILTER)).toEqual({
      crop_ids: ['a', 'b'],
    });
  });

  it('sends the filter plus image_ids for images', () => {
    expect(toItemSelection({ kind: 'images', imageIds: ['i1'] }, FILTER)).toEqual({
      filter: { ...FILTER, image_ids: ['i1'] },
    });
  });

  it('sends the filter plus cluster_uids for clusters', () => {
    expect(
      toItemSelection({ kind: 'clusters', clusterUids: ['u1', 'u2'] }, FILTER),
    ).toEqual({
      filter: { ...FILTER, cluster_uids: ['u1', 'u2'] },
    });
  });

  it('sends the filter alone for all matching', () => {
    expect(toItemSelection({ kind: 'matching' }, FILTER)).toEqual({ filter: FILTER });
  });

  it('never carries include_test or an exclusion list', () => {
    const kinds: WorkspaceSelection[] = [
      { kind: 'items', cropIds: ['a'] },
      { kind: 'images', imageIds: ['i'] },
      { kind: 'clusters', clusterUids: ['u'] },
      { kind: 'matching' },
    ];
    for (const sel of kinds) {
      expect(Object.keys(toItemSelection(sel, FILTER)!)).not.toContain('include_test');
    }
  });

  it('does not alias the caller filter', () => {
    const f = { class_names: ['widget'] };
    const out = toItemSelection({ kind: 'matching' }, f);
    expect(out!.filter).not.toBe(f);
  });
});

describe('selectionKindFor', () => {
  it.each([
    ['gallery', undefined, 'images'],
    ['map', undefined, 'images'],
    ['clusters', undefined, 'clusters'],
    ['detections', undefined, 'items'],
    ['review', undefined, 'items'],
    ['table', 'images', 'images'],
    ['table', 'detections', 'items'],
    ['table', undefined, 'items'],
  ] as const)('%s %s -> %s', (view, rows, kind) => {
    expect(selectionKindFor(view, rows)).toBe(kind);
  });
});

describe('pressSelectAll', () => {
  const page = ['a', 'b', 'c'];

  it('first press selects the loaded page', () => {
    const r = pressSelectAll(NO_SELECTION, 'items', page, 40);
    expect(r.selection).toEqual({ kind: 'items', cropIds: page });
    expect(r.pending).toBeNull();
  });

  it('a second select-all on a full page asks for all matching', () => {
    const r = pressSelectAll({ kind: 'items', cropIds: page }, 'items', page, 40);
    expect(r.pending).toEqual({ total: 40 });
    expect(r.selection).toEqual({ kind: 'items', cropIds: page });
  });

  it('does not ask when the page is everything that matches', () => {
    const r = pressSelectAll({ kind: 'items', cropIds: page }, 'items', page, 3);
    expect(r.pending).toBeNull();
  });

  it('replaces a selection of another kind', () => {
    const r = pressSelectAll({ kind: 'images', imageIds: page }, 'items', page, 40);
    expect(r.selection).toEqual({ kind: 'items', cropIds: page });
    expect(r.pending).toBeNull();
  });
});
