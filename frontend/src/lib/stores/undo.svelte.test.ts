/**
 * Unit tests for the UndoStore ring buffer.
 *
 * The store is a module singleton, so every test clears it first.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { undoStore } from './undo.svelte';
import { toastStore } from './toast.svelte';
import { API_PREFIX } from '$lib/api';
import type { UndoEntry } from '$lib/types';

function entry(...ids: string[]): UndoEntry {
  return { crop_ids: ids, at: 0 };
}

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

describe('undoStore', () => {
  beforeEach(() => {
    undoStore.clear();
  });

  it('pops in LIFO order', () => {
    undoStore.push(entry('a'));
    undoStore.push(entry('b'));
    undoStore.push(entry('c'));
    expect(undoStore.pop()?.crop_ids).toEqual(['c']);
    expect(undoStore.pop()?.crop_ids).toEqual(['b']);
    expect(undoStore.pop()?.crop_ids).toEqual(['a']);
  });

  it('caps at 50 entries, dropping the oldest', () => {
    for (let i = 0; i < 51; i++) undoStore.push(entry(`e${i}`));
    expect(undoStore.stack).toHaveLength(50);
    expect(undoStore.stack[0]!.crop_ids).toEqual(['e1']);
    expect(undoStore.stack[49]!.crop_ids).toEqual(['e50']);
  });

  it('returns undefined when popping an empty stack', () => {
    expect(undoStore.pop()).toBeUndefined();
  });

  it('remove() drops exactly the given entries by identity', () => {
    const a = entry('a');
    const b = entry('b');
    const c = entry('c');
    undoStore.push(a);
    undoStore.push(b);
    undoStore.push(c);
    undoStore.remove([a, c]);
    expect(undoStore.stack.map((e) => e.crop_ids)).toEqual([['b']]);
  });

  it('remove() ignores entries that are no longer on the stack', () => {
    const a = entry('a');
    undoStore.push(a);
    undoStore.remove([entry('a')]); // structurally equal but a different object
    expect(undoStore.stack).toHaveLength(1);
  });

  it('recordWrites() pushes exactly ONE entry holding every served id, however many crops the write touched', () => {
    undoStore.recordWrites(['a', 'b', 'c']);
    expect(undoStore.stack).toHaveLength(1);
    expect(undoStore.stack[0]!.crop_ids).toEqual(['a', 'b', 'c']);
  });

  it('recordWrites() pushes one entry for a single-crop write too', () => {
    undoStore.recordWrites(['a']);
    expect(undoStore.stack).toHaveLength(1);
    expect(undoStore.stack[0]!.crop_ids).toEqual(['a']);
  });

  it('recordWrites() pushes nothing for an empty served list (e.g. every id conflicted)', () => {
    undoStore.recordWrites([]);
    expect(undoStore.stack).toHaveLength(0);
  });
});

describe('undoStore.undoLast — single-crop entry routes to the single undo endpoint', () => {
  beforeEach(() => {
    undoStore.clear();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs the single-crop undo route and returns the server's restored item", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse(200, { crop_id: 'c1', class_id: 3, class_name: 'sedan' }),
      );
    vi.stubGlobal('fetch', fetchMock);
    undoStore.recordWrites(['c1']);
    const toastsBefore = toastStore.toasts.length;

    const crops = await undoStore.undoLast();

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe(`${API_PREFIX}/crops/c1/label/undo`);
    expect(init.method).toBe('POST');
    expect(crops).toHaveLength(1);
    expect(crops[0]?.id).toBe('c1');
    expect(crops[0]?.class_id).toBe(3);
    expect(undoStore.stack).toHaveLength(0);
    // Confirms the success branch actually ran (not just that the request
    // succeeded) — a mutant that empties that branch's block still returns
    // the crop but never surfaces a 'success' toast.
    expect(toastStore.toasts.length).toBe(toastsBefore + 1);
    expect(toastStore.toasts.at(-1)?.kind).toBe('success');
  });

  it('409 (nothing left to undo) returns [] and does not re-push', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(409, { detail: 'nothing to undo' })),
    );
    undoStore.recordWrites(['c1']);
    const toastsBefore = toastStore.toasts.length;

    expect(await undoStore.undoLast()).toEqual([]);

    expect(undoStore.stack).toHaveLength(0);
    // The 409 branch, specifically, must run — distinguishes it from the
    // generic-failure branch below, which re-pushes and toasts 'error'.
    expect(toastStore.toasts.length).toBe(toastsBefore + 1);
    expect(toastStore.toasts.at(-1)?.kind).toBe('info');
  });

  it('any other failure re-pushes the entry so Z stays retryable', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(404, { detail: 'unknown crop' })),
    );
    undoStore.recordWrites(['c1']);
    const toastsBefore = toastStore.toasts.length;

    expect(await undoStore.undoLast()).toEqual([]);

    expect(undoStore.stack.map((e) => e.crop_ids)).toEqual([['c1']]);
    expect(toastStore.toasts.length).toBe(toastsBefore + 1);
    expect(toastStore.toasts.at(-1)?.kind).toBe('error');
  });

  it('an empty stack makes no request and leaves the stack empty', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    expect(await undoStore.undoLast()).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
    // Guards the early-return itself: without it, `entry` is undefined and
    // the catch block's re-push (`this.push(entry)`) would put an
    // undefined entry onto the stack instead of leaving it empty.
    expect(undoStore.stack).toHaveLength(0);
  });
});

describe('undoStore.undoLast — multi-crop entry routes to the batch undo endpoint', () => {
  beforeEach(() => {
    undoStore.clear();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('POSTs undo_batch with every crop id from the one recorded entry and returns every restored item', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(200, {
        items: [
          { crop_id: 'c1', class_id: 1, class_name: 'sedan' },
          { crop_id: 'c2', class_id: 1, class_name: 'sedan' },
          { crop_id: 'c3', class_id: 1, class_name: 'sedan' },
        ],
        undone: 3,
        nothing_to_undo: [],
        conflicts: [],
        not_found: [],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    undoStore.recordWrites(['c1', 'c2', 'c3']);
    const toastsBefore = toastStore.toasts.length;

    const crops = await undoStore.undoLast();

    // Exactly one request for the whole bulk write.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe(`${API_PREFIX}/crops/label/undo_batch`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ crop_ids: ['c1', 'c2', 'c3'] });
    expect(crops.map((c) => c.id)).toEqual(['c1', 'c2', 'c3']);
    expect(undoStore.stack).toHaveLength(0);
    expect(toastStore.toasts.length).toBe(toastsBefore + 1);
    expect(toastStore.toasts.at(-1)?.kind).toBe('success');
    expect(toastStore.toasts.at(-1)?.text).toMatch(/Reverted 3\./);
  });

  it('surfaces nothing_to_undo and conflicts counts in the toast when non-zero', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse(200, {
          items: [{ crop_id: 'c1', class_id: 1, class_name: 'sedan' }],
          undone: 1,
          nothing_to_undo: ['c2'],
          conflicts: ['c3'],
          not_found: [],
        }),
      ),
    );
    undoStore.recordWrites(['c1', 'c2', 'c3']);

    const crops = await undoStore.undoLast();

    expect(crops.map((c) => c.id)).toEqual(['c1']);
    const toast = toastStore.toasts.at(-1);
    expect(toast?.text).toMatch(/Reverted 1\./);
    expect(toast?.text).toMatch(/1 nothing to undo/);
    expect(toast?.text).toMatch(/1 conflict\(s\)/);
  });

  it('409 (nothing in the batch had anything to undo) returns [] and does not re-push', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(409, { detail: 'nothing to undo' })),
    );
    undoStore.recordWrites(['c1', 'c2']);
    const toastsBefore = toastStore.toasts.length;

    expect(await undoStore.undoLast()).toEqual([]);

    expect(undoStore.stack).toHaveLength(0);
    expect(toastStore.toasts.length).toBe(toastsBefore + 1);
    expect(toastStore.toasts.at(-1)?.kind).toBe('info');
  });

  it('a transport/5xx failure re-pushes the whole entry so Z stays retryable', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(500, { detail: 'boom' })),
    );
    undoStore.recordWrites(['c1', 'c2']);
    const toastsBefore = toastStore.toasts.length;

    expect(await undoStore.undoLast()).toEqual([]);

    expect(undoStore.stack).toHaveLength(1);
    expect(undoStore.stack[0]!.crop_ids).toEqual(['c1', 'c2']);
    expect(toastStore.toasts.length).toBe(toastsBefore + 1);
    expect(toastStore.toasts.at(-1)?.kind).toBe('error');
  });
});
