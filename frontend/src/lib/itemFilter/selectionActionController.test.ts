import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return {
    ...actual,
    bulkLabelSelection: vi.fn(),
    batchExcludeSelection: vi.fn(),
    batchUnexcludeSelection: vi.fn(),
    moveSelectionToCluster: vi.fn(),
  };
});

import {
  ApiError,
  batchExcludeSelection,
  batchUnexcludeSelection,
  bulkLabelSelection,
  moveSelectionToCluster,
} from '$lib/api';
import { undoStore } from '$stores/undo.svelte';
import { createSelectionActionController } from './selectionActionController.svelte';
import type { ItemFilter } from '$lib/types_itemFilter';

const FILTER: ItemFilter = { class_names: ['widget'], origin: ['sam3'] };

function make(filter: ItemFilter = FILTER) {
  return createSelectionActionController(() => filter);
}

beforeEach(() => {
  undoStore.clear();
});
afterEach(() => {
  vi.clearAllMocks();
});

describe('selectionActionController', () => {
  it('opening an action runs a dry run first and never a real write', async () => {
    vi.mocked(batchExcludeSelection).mockResolvedValue({ dry_run: true, selected: 40 });
    const c = make();
    await c.open('exclude');
    expect(batchExcludeSelection).toHaveBeenCalledTimes(1);
    expect(batchExcludeSelection).toHaveBeenCalledWith(
      { filter: FILTER },
      'ignore',
      true,
      expect.anything(),
    );
    expect(c.selected).toBe(40);
    expect(c.confirmed).toBe(false);
  });

  it('confirm repeats the same selection with dry_run false, and records exclude ids for Z', async () => {
    vi.mocked(batchExcludeSelection)
      .mockResolvedValueOnce({ dry_run: true, selected: 2 })
      .mockResolvedValueOnce({ excluded: 2, updated_ids: ['a', 'b'], errors: 0 });
    const c = make();
    await c.open('exclude');
    await c.confirm();
    expect(batchExcludeSelection).toHaveBeenLastCalledWith(
      { filter: FILTER },
      'ignore',
      false,
      expect.anything(),
    );
    expect(undoStore.stack).toHaveLength(1);
    expect(undoStore.stack[0]).toMatchObject({ crop_ids: ['a', 'b'], kind: 'exclude' });
    expect(c.result?.summary).toContain('2');
  });

  it('limit, sample and seed are sent only when set, and a change re-runs the dry run', async () => {
    vi.mocked(batchExcludeSelection).mockResolvedValue({ dry_run: true, selected: 10 });
    const c = make();
    await c.open('exclude');
    expect(vi.mocked(batchExcludeSelection).mock.calls[0]![0]).toEqual({
      filter: FILTER,
    });
    c.limit = 10;
    c.sample = 'random';
    c.seed = 7;
    await c.refresh();
    expect(vi.mocked(batchExcludeSelection).mock.calls[1]![0]).toEqual({
      filter: FILTER,
      limit: 10,
      sample: 'random',
      seed: 7,
    });
  });

  it('seed is dropped when the sample is not random', async () => {
    vi.mocked(batchExcludeSelection).mockResolvedValue({ dry_run: true, selected: 10 });
    const c = make();
    await c.open('exclude');
    c.limit = 5;
    c.sample = 'largest';
    c.seed = 7;
    await c.refresh();
    expect(vi.mocked(batchExcludeSelection).mock.calls[1]![0]).toEqual({
      filter: FILTER,
      limit: 5,
      sample: 'largest',
    });
  });

  it('label records the served updated_ids as one label undo entry', async () => {
    vi.mocked(bulkLabelSelection)
      .mockResolvedValueOnce({ dry_run: true, selected: 3 })
      .mockResolvedValueOnce({ updated: 3, updated_ids: ['a', 'b', 'c'], conflicts: [] });
    const c = make();
    c.classId = 5;
    await c.open('label');
    expect(bulkLabelSelection).toHaveBeenCalledWith(
      { filter: FILTER },
      5,
      true,
      expect.anything(),
    );
    await c.confirm();
    expect(undoStore.stack).toHaveLength(1);
    expect(undoStore.stack[0]).toMatchObject({
      crop_ids: ['a', 'b', 'c'],
      kind: 'label',
    });
  });

  it('move and unexclude record their served ids too', async () => {
    vi.mocked(moveSelectionToCluster)
      .mockResolvedValueOnce({ dry_run: true, selected: 1 })
      .mockResolvedValueOnce({ updated: 1, updated_ids: ['m'], conflicts: [] });
    const mv = make();
    mv.clusterId = 9;
    await mv.open('move');
    await mv.confirm();
    expect(moveSelectionToCluster).toHaveBeenLastCalledWith(
      { filter: FILTER },
      9,
      false,
      expect.anything(),
    );
    expect(undoStore.stack.at(-1)).toMatchObject({ crop_ids: ['m'], kind: 'label' });

    vi.mocked(batchUnexcludeSelection)
      .mockResolvedValueOnce({ dry_run: true, selected: 1 })
      .mockResolvedValueOnce({ unexcluded: 1, updated_ids: ['u'], errors: 0 });
    const un = make();
    await un.open('unexclude');
    await un.confirm();
    expect(undoStore.stack.at(-1)).toMatchObject({ crop_ids: ['u'], kind: 'unexclude' });
  });

  it('a served refusal is shown verbatim and nothing is written', async () => {
    vi.mocked(batchExcludeSelection).mockRejectedValue(
      new ApiError(422, '/x', { detail: 'empty filter needs a limit' }),
    );
    const c = make({});
    await c.open('exclude');
    expect(c.error).toContain('empty filter needs a limit');
    expect(c.selected).toBeNull();
    await c.confirm();
    expect(batchExcludeSelection).toHaveBeenCalledTimes(1);
  });

  it('a structured refusal is worded by its served message, not its error code', async () => {
    vi.mocked(batchExcludeSelection).mockRejectedValue(
      new ApiError(422, '/x', {
        detail: {
          error: 'selection_too_large',
          message: 'matches more than 20000 items',
        },
      }),
    );
    const c = make();
    await c.open('exclude');
    expect(c.error).toBe('matches more than 20000 items');
  });

  it('label with no class picked does not call the server', async () => {
    const c = make();
    await c.open('label');
    expect(bulkLabelSelection).not.toHaveBeenCalled();
    expect(c.selected).toBeNull();
  });

  it('a zero-item dry run cannot be confirmed', async () => {
    vi.mocked(batchExcludeSelection).mockResolvedValue({ dry_run: true, selected: 0 });
    const c = make();
    await c.open('exclude');
    expect(c.canConfirm).toBe(false);
    await c.confirm();
    expect(batchExcludeSelection).toHaveBeenCalledTimes(1);
  });

  it('the filter is snapshotted when the dialog opens, not re-read at confirm', async () => {
    vi.mocked(batchExcludeSelection)
      .mockResolvedValueOnce({ dry_run: true, selected: 2 })
      .mockResolvedValueOnce({ excluded: 2, updated_ids: ['a', 'b'], errors: 0 });
    let live: ItemFilter = FILTER;
    const c = createSelectionActionController(() => live);
    await c.open('exclude');
    live = { class_names: ['other'] };
    await c.confirm();
    expect(vi.mocked(batchExcludeSelection).mock.calls[1]![0]).toEqual({
      filter: FILTER,
    });
  });

  it('cancel closes without writing', async () => {
    vi.mocked(batchExcludeSelection).mockResolvedValue({ dry_run: true, selected: 2 });
    const c = make();
    await c.open('exclude');
    c.cancel();
    expect(c.action).toBeNull();
    expect(batchExcludeSelection).toHaveBeenCalledTimes(1);
  });
});
