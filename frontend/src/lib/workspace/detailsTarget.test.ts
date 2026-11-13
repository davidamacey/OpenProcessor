import { describe, expect, it } from 'vitest';
import { detailsTarget } from './detailsTarget';
import type { WorkspaceSelection } from './selection';

const NONE: WorkspaceSelection = { kind: 'none' };
const items = (...cropIds: string[]): WorkspaceSelection => ({ kind: 'items', cropIds });
const item = { kind: 'item', id: 'c1' } as const;

describe('detailsTarget', () => {
  it('shows the focused detection, even with one card selected', () => {
    expect(detailsTarget('detections', item, items('c1'), null)).toEqual({
      kind: 'item',
      id: 'c1',
    });
  });

  it('shows the selection once two or more are selected, whatever has focus', () => {
    expect(detailsTarget('detections', item, items('c1', 'c2'), null)).toEqual({
      kind: 'selection',
      id: null,
    });
    expect(
      detailsTarget('gallery', { kind: 'image', id: 'i1' }, { kind: 'matching' }, null),
    ).toEqual({ kind: 'selection', id: null });
  });

  it('keeps the item beside the Review pane, which does not show the selection', () => {
    expect(detailsTarget('review', item, items('c1', 'c2'), null)).toEqual({
      kind: 'review',
      id: 'c1',
    });
  });

  it('shows a lone selection when nothing has focus, else the open cluster, else nothing', () => {
    expect(detailsTarget('detections', null, items('c1'), null)?.kind).toBe('selection');
    expect(detailsTarget('clusters', null, NONE, { kind: 'cluster', id: 'u1' })).toEqual({
      kind: 'cluster',
      id: 'u1',
    });
    expect(detailsTarget('clusters', null, NONE, null)).toBeNull();
  });

  it('ignores a focus the view cannot show', () => {
    expect(detailsTarget('clusters', { kind: 'image', id: 'i1' }, NONE, null)).toBeNull();
  });
});
