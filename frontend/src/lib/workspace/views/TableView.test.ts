/**
 * The Table view mounted over a stubbed table API: opening it over an item selection made in
 * Detections keeps that selection (the view used to mirror its own empty selection back and clear
 * it), and an images selection is not carried into the detection rows.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync, mount, unmount } from 'svelte';
import { ItemFilterState } from '$lib/itemFilter/itemFilterState.svelte';
import type { WorkspaceController } from '$lib/workspace/workspaceController.svelte';
import type { WorkspaceSelection } from '$lib/workspace/selection';

vi.mock('$lib/api_table', () => ({
  getTableColumns: vi.fn().mockResolvedValue({ columns: [], defaults: [] }),
  getTable: vi.fn().mockResolvedValue({
    rows: [],
    total: 0,
    next_cursor: null,
    columns: [],
    sort: null,
    sort_options: [],
    hidden: { test_holdout: 0, excluded: 0 },
    empty_message: null,
  }),
}));
vi.mock('$components/workspace/BulkBar.svelte', () => ({ default: () => {} }));
vi.mock('$components/table/DataTable.svelte', () => ({ default: () => {} }));
vi.mock('$components/table/ColumnPicker.svelte', () => ({ default: () => {} }));
vi.mock('$components/workspace/DuplicatesControl.svelte', () => ({ default: () => {} }));
vi.mock('$components/workspace/ViewMoreMenu.svelte', () => ({ default: () => {} }));
vi.mock('$components/itemFilter/HiddenItemsNote.svelte', () => ({ default: () => {} }));

const { default: TableView } = await import('./TableView.svelte');

let target: HTMLDivElement;
let instance: ReturnType<typeof mount> | undefined;

function fakeWs(selection: WorkspaceSelection, rows: 'detections' | 'images') {
  const ws = {
    filter: new ItemFilterState(),
    selection,
    url: { rows, columns: null, sort: {}, nearDup: null },
    itemQuery: () => ({}),
    patchUrl: vi.fn(),
    registerPage: vi.fn(() => () => {}),
    select: vi.fn((next: WorkspaceSelection) => {
      ws.selection = next;
    }),
    setFocus: vi.fn(),
    setView: vi.fn(),
    openReview: vi.fn(),
    setNearDup: vi.fn(),
    matching: null,
  };
  return ws;
}

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
});
afterEach(() => {
  if (instance) unmount(instance);
  instance = undefined;
  target.remove();
});

describe('TableView selection', () => {
  it('keeps the items selected in another view', () => {
    const ws = fakeWs({ kind: 'items', cropIds: ['c1', 'c2'] }, 'detections');
    instance = mount(TableView, {
      target,
      props: { ws: ws as unknown as WorkspaceController },
    });
    flushSync();
    expect(ws.selection).toEqual({ kind: 'items', cropIds: ['c1', 'c2'] });
  });

  it('does not carry an images selection into detection rows', () => {
    const ws = fakeWs({ kind: 'images', imageIds: ['i1'] }, 'detections');
    instance = mount(TableView, {
      target,
      props: { ws: ws as unknown as WorkspaceController },
    });
    flushSync();
    expect(ws.selection.kind).toBe('none');
  });
});
