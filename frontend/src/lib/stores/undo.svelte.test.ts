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

function entry(id: string): UndoEntry {
  return { crop_id: id, at: 0 };
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
    expect(undoStore.pop()?.crop_id).toBe('c');
    expect(undoStore.pop()?.crop_id).toBe('b');
    expect(undoStore.pop()?.crop_id).toBe('a');
  });

  it('caps at 50 entries, dropping the oldest', () => {
    for (let i = 0; i < 51; i++) undoStore.push(entry(`e${i}`));
    expect(undoStore.stack).toHaveLength(50);
    expect(undoStore.stack[0]!.crop_id).toBe('e1');
    expect(undoStore.stack[49]!.crop_id).toBe('e50');
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
    expect(undoStore.stack.map((e) => e.crop_id)).toEqual(['b']);
  });

  it('remove() ignores entries that are no longer on the stack', () => {
    const a = entry('a');
    undoStore.push(a);
    undoStore.remove([entry('a')]); // structurally equal but a different object
    expect(undoStore.stack).toHaveLength(1);
  });

  it('recordWrites() pushes one entry per crop, skipping conflicted crops', () => {
    undoStore.recordWrites(['a', 'b', 'c'], [{ crop_id: 'b' }]);
    expect(undoStore.stack.map((e) => e.crop_id)).toEqual(['a', 'c']);
  });

  it('recordWrites() with no conflicts argument pushes every crop (default is empty, not a sentinel)', () => {
    undoStore.recordWrites(['x', 'y']);
    expect(undoStore.stack.map((e) => e.crop_id)).toEqual(['x', 'y']);
  });
});

describe('undoStore.undoLast', () => {
  beforeEach(() => {
    undoStore.clear();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs the backend undo route and returns the server's restored item", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse(200, { crop_id: 'c1', class_id: 3, class_name: 'sedan' }),
      );
    vi.stubGlobal('fetch', fetchMock);
    undoStore.recordWrites(['c1']);
    const toastsBefore = toastStore.toasts.length;

    const crop = await undoStore.undoLast();

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe(`${API_PREFIX}/crops/c1/label/undo`);
    expect(init.method).toBe('POST');
    expect(crop?.id).toBe('c1');
    expect(crop?.class_id).toBe(3);
    expect(undoStore.stack).toHaveLength(0);
    // Confirms the success branch actually ran (not just that the request
    // succeeded) — a mutant that empties that branch's block still returns
    // the crop but never surfaces a 'success' toast.
    expect(toastStore.toasts.length).toBe(toastsBefore + 1);
    expect(toastStore.toasts.at(-1)?.kind).toBe('success');
  });

  it('409 (nothing left to undo) returns null and does not re-push', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(409, { detail: 'nothing to undo' })),
    );
    undoStore.recordWrites(['c1']);
    const toastsBefore = toastStore.toasts.length;

    expect(await undoStore.undoLast()).toBeNull();

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

    expect(await undoStore.undoLast()).toBeNull();

    expect(undoStore.stack.map((e) => e.crop_id)).toEqual(['c1']);
    expect(toastStore.toasts.length).toBe(toastsBefore + 1);
    expect(toastStore.toasts.at(-1)?.kind).toBe('error');
  });

  it('an empty stack makes no request and leaves the stack empty', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    expect(await undoStore.undoLast()).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
    // Guards the early-return itself: without it, `entry` is undefined and
    // the catch block's re-push (`this.push(entry)`) would put an
    // undefined entry onto the stack instead of leaving it empty.
    expect(undoStore.stack).toHaveLength(0);
  });
});
