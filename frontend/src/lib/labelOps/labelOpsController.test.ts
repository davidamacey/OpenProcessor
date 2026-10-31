/** Dry run -> confirm sequencing, paging and error handling of the label-ops controller. */
import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api_errors';
import { LabelOpsController, type LabelOpsDeps } from './labelOpsController.svelte';
import type { LabelOp, LabelOpUndoResponse } from '$lib/types_labelOps';

function op(id: string, over: Partial<LabelOp> = {}): LabelOp {
  return {
    op_id: id,
    kind: 'label',
    at: '2026-10-30T10:00:00+00:00',
    finished_at: '2026-10-30T10:00:05+00:00',
    class_name: 'widget_a',
    target_cluster_id: null,
    target_kind: null,
    label_source: 'human',
    target: 'selection',
    selected: 10,
    count: 10,
    conflicts: 0,
    current: 10,
    status: 'written',
    undone: false,
    undoable: true,
    last_undo: null,
    ...over,
  };
}

function undoResult(
  dry: boolean,
  over: Partial<LabelOpUndoResponse> = {},
): LabelOpUndoResponse {
  return {
    op_id: 'a',
    dry_run: dry,
    status: 'written',
    members_known: true,
    matched: 10,
    restored: 8,
    already_undone: 0,
    kept_changed: 1,
    kept_holdout: 0,
    kept_stale_class: 0,
    conflicts: 1,
    conflict_ids: ['c1'],
    ...over,
  };
}

function make(over: Partial<LabelOpsDeps> = {}) {
  const deps: LabelOpsDeps = {
    list: vi.fn().mockResolvedValue({
      items: [op('a')],
      limit: 20,
      retention: 200,
      next_after: null,
    }),
    undo: vi.fn((_id: string, dry: boolean) => Promise.resolve(undoResult(dry))),
    ...over,
  };
  return { deps, ctl: new LabelOpsController(deps) };
}

describe('LabelOpsController', () => {
  it('loads the list and the served retention', async () => {
    const { ctl } = make();
    await ctl.load();
    expect(ctl.items.map((o) => o.op_id)).toEqual(['a']);
    expect(ctl.retention).toBe(200);
    expect(ctl.loading).toBe(false);
  });

  it('keeps a failed read as an error, not an empty list', async () => {
    const { ctl } = make({
      list: vi.fn().mockRejectedValue(new ApiError(500, '/x', { detail: 'index down' })),
    });
    await ctl.load();
    expect(ctl.loadError).toBe('index down');
  });

  it('loads the first page without a cursor and follows next_after', async () => {
    const list = vi.fn().mockResolvedValue({
      items: [op('a')],
      limit: 20,
      retention: 200,
      next_after: 'a',
    });
    const { ctl } = make({ list });
    const signal = new AbortController().signal;
    await ctl.load(signal);
    expect(list).toHaveBeenCalledWith({ limit: 20 }, signal);
    expect(ctl.nextAfter).toBe('a');
    expect(ctl.canLoadOlder).toBe(true);
  });

  it('loadOlder sends the cursor, appends, and stops at the end', async () => {
    const full = Array.from({ length: 20 }, (_, i) => op(`o${i}`));
    const list = vi
      .fn()
      .mockResolvedValueOnce({
        items: full,
        limit: 20,
        retention: 200,
        next_after: 'o19',
      })
      .mockResolvedValueOnce({
        items: [op('p0')],
        limit: 20,
        retention: 200,
        next_after: null,
      });
    const { ctl } = make({ list });
    await ctl.load();
    await ctl.loadOlder();
    expect(list).toHaveBeenLastCalledWith({ limit: 20, after: 'o19' });
    expect(ctl.items).toHaveLength(21);
    expect(ctl.canLoadOlder).toBe(false);
    await ctl.loadOlder();
    expect(list).toHaveBeenCalledTimes(2);
  });

  it('a failed loadOlder keeps the list and cursor, and a retry succeeds', async () => {
    const list = vi
      .fn()
      .mockResolvedValueOnce({
        items: [op('a')],
        limit: 20,
        retention: 200,
        next_after: 'a',
      })
      .mockRejectedValueOnce(new ApiError(500, '/x', { detail: 'index down' }))
      .mockResolvedValueOnce({
        items: [op('b')],
        limit: 20,
        retention: 200,
        next_after: null,
      });
    const { ctl } = make({ list });
    await ctl.load();
    await ctl.loadOlder();
    expect(ctl.olderError).toBe('index down');
    expect(ctl.items.map((o) => o.op_id)).toEqual(['a']);
    expect(ctl.nextAfter).toBe('a');
    await ctl.loadOlder();
    expect(ctl.olderError).toBeNull();
    expect(ctl.items.map((o) => o.op_id)).toEqual(['a', 'b']);
  });

  it('drops an older page that lands after a reload', async () => {
    let release!: (v: unknown) => void;
    const slow = new Promise((r) => (release = r));
    const list = vi
      .fn()
      .mockResolvedValueOnce({
        items: [op('a')],
        limit: 20,
        retention: 200,
        next_after: 'a',
      })
      .mockReturnValueOnce(slow)
      .mockResolvedValueOnce({
        items: [op('n')],
        limit: 20,
        retention: 200,
        next_after: null,
      });
    const { ctl } = make({ list });
    await ctl.load();
    const pending = ctl.loadOlder();
    await ctl.load();
    release({ items: [op('stale')], limit: 20, retention: 200, next_after: 'stale' });
    await pending;
    expect(ctl.items.map((o) => o.op_id)).toEqual(['n']);
    expect(ctl.nextAfter).toBeNull();
    expect(ctl.loadingOlder).toBe(false);
  });

  it('after an undo the shown pages are re-served', async () => {
    const page1 = Array.from({ length: 20 }, (_, i) => op(`o${i}`));
    const list = vi
      .fn()
      .mockResolvedValueOnce({
        items: page1,
        limit: 20,
        retention: 200,
        next_after: 'o19',
      })
      .mockResolvedValueOnce({
        items: [op('p0')],
        limit: 20,
        retention: 200,
        next_after: null,
      })
      .mockResolvedValueOnce({
        items: page1,
        limit: 20,
        retention: 200,
        next_after: 'o19',
      })
      .mockResolvedValueOnce({
        items: [op('p0')],
        limit: 20,
        retention: 200,
        next_after: null,
      });
    const { ctl } = make({ list });
    await ctl.load();
    await ctl.loadOlder();
    ctl.open('o0');
    await ctl.startUndo();
    await ctl.confirmUndo();
    expect(list).toHaveBeenNthCalledWith(3, { limit: 20 });
    expect(list).toHaveBeenNthCalledWith(4, { limit: 20, after: 'o19' });
    expect(ctl.items).toHaveLength(21);
    expect(ctl.canLoadOlder).toBe(false);
  });

  it('the first undo call is always a dry run and writes nothing', async () => {
    const { ctl, deps } = make();
    await ctl.load();
    ctl.open('a');
    await ctl.startUndo();
    expect(deps.undo).toHaveBeenCalledTimes(1);
    expect(deps.undo).toHaveBeenCalledWith('a', true);
    expect(ctl.phase).toBe('ready');
    expect(ctl.result?.conflict_ids).toEqual(['c1']);
  });

  it('confirm is refused before a dry run was shown', async () => {
    const { ctl, deps } = make();
    await ctl.load();
    ctl.open('a');
    await ctl.confirmUndo();
    expect(deps.undo).not.toHaveBeenCalled();
    expect(ctl.phase).toBe('idle');
  });

  it('confirm writes with dry_run false and refreshes the list', async () => {
    const { ctl, deps } = make();
    await ctl.load();
    ctl.open('a');
    await ctl.startUndo();
    await ctl.confirmUndo();
    expect(deps.undo).toHaveBeenLastCalledWith('a', false);
    expect(ctl.phase).toBe('done');
    expect(ctl.result?.dry_run).toBe(false);
    expect(deps.list).toHaveBeenCalledTimes(2);
  });

  it('cancelling the dry run discards it without writing', async () => {
    const { ctl, deps } = make();
    await ctl.load();
    ctl.open('a');
    await ctl.startUndo();
    ctl.cancelUndo();
    expect(ctl.phase).toBe('idle');
    expect(ctl.result).toBeNull();
    expect(deps.undo).toHaveBeenCalledTimes(1);
  });

  it('shows a served refusal and returns to idle', async () => {
    const refusal = new ApiError(409, '/x', {
      detail: { error: 'label_op_busy', message: 'another undo is running' },
    });
    const { ctl } = make({ undo: vi.fn().mockRejectedValue(refusal) });
    await ctl.load();
    ctl.open('a');
    await ctl.startUndo();
    expect(ctl.undoError).toBe('another undo is running');
    expect(ctl.phase).toBe('idle');
  });

  it('opening another operation clears the previous dry run', async () => {
    const { ctl } = make({
      list: vi.fn().mockResolvedValue({
        items: [op('a'), op('b')],
        limit: 20,
        retention: 5,
        next_after: null,
      }),
    });
    await ctl.load();
    ctl.open('a');
    await ctl.startUndo();
    ctl.open('b');
    expect(ctl.result).toBeNull();
    expect(ctl.phase).toBe('idle');
    expect(ctl.selected?.op_id).toBe('b');
  });
});
