/**
 * Behavior tests for the queue-action controller extracted out of
 * review/+page.svelte (docs/design/test-audit-2026-09-24.md P1-4). These
 * close the three mutations the audit found surviving under the old
 * source-scan-only suite: a failed assign() leaving the item gone, a
 * failed discard() leaving the item gone, and skip() not advancing.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { createReviewQueueController } from './reviewController.svelte';
import { createPager } from '$lib/pager.svelte';
import type { ReviewItem } from '$lib/types';

vi.mock('$lib/api', async () => {
  const actual = await vi.importActual<typeof import('$lib/api')>('$lib/api');
  return {
    ...actual,
    putCropLabel: vi.fn(),
    reviewDismissCrop: vi.fn(),
    undoCropLabel: vi.fn(),
    undoLabelBatch: vi.fn(),
  };
});

import { putCropLabel, reviewDismissCrop, undoCropLabel, undoLabelBatch } from '$lib/api';
import { classesStore } from '$stores/classes.svelte';
import { undoStore } from '$stores/undo.svelte';
import { toastStore } from '$stores/toast.svelte';

function item(id: string): ReviewItem {
  return {
    id,
    source_image_path: '/img.jpg',
    bbox_norm: { cx: 0.5, cy: 0.5, w: 0.4, h: 0.4 },
    class_id: null,
    class_name: null,
    class_source: null,
    label_source: 'model',
    label_validated: false,
    class_validated: false,
    label_confidence: null,
    cluster_id: null,
    similarity_to_centroid: null,
    cluster_subid: null,
    test_holdout: false,
    updated_at: '',
    reason: 'test',
  } as ReviewItem;
}

function setup(items: ReviewItem[]) {
  const queue = createPager<ReviewItem>({
    fetchPage: async () => ({ items: [], total: 0 }),
    keyOf: (i) => i.id,
  });
  queue.items = items;
  queue.total = items.length;
  const handledIds = new Set<string>();
  let cursor = 0;
  const maybePrefetch = vi.fn();
  const controller = createReviewQueueController({
    queue,
    handledIds,
    getCursor: () => cursor,
    setCursor: (v) => {
      cursor = v;
    },
    maybePrefetch,
  });
  return {
    queue,
    handledIds,
    controller,
    getCursor: () => cursor,
  };
}

afterEach(() => {
  vi.clearAllMocks();
  undoStore.clear();
});

describe('assign', () => {
  it('removes the item, advances past it, and records the write on success', async () => {
    vi.mocked(putCropLabel).mockResolvedValue({} as never);
    vi.spyOn(classesStore, 'byId').mockReturnValue({ id: 3, name: 'widget_a' } as never);
    const successSpy = vi
      .spyOn(toastStore, 'success')
      .mockImplementation(() => 'toast-id');
    const recordWritesSpy = vi.spyOn(undoStore, 'recordWrites');
    const a = item('a');
    const b = item('b');
    const { queue, controller, handledIds } = setup([a, b]);

    await controller.assign(a, 3);

    expect(queue.items.map((i) => i.id)).toEqual(['b']);
    expect(queue.total).toBe(1);
    expect(handledIds.has('a')).toBe(true);
    expect(putCropLabel).toHaveBeenCalledWith('a', 3);
    expect(successSpy).toHaveBeenCalledWith('Labeled "widget_a".');
    expect(recordWritesSpy).toHaveBeenCalledWith(['a']);
  });

  it('falls back to the raw class id in the toast when the class is not in the catalog', async () => {
    vi.mocked(putCropLabel).mockResolvedValue({} as never);
    vi.spyOn(classesStore, 'byId').mockReturnValue(undefined);
    const successSpy = vi
      .spyOn(toastStore, 'success')
      .mockImplementation(() => 'toast-id');
    const a = item('a');
    const { controller } = setup([a]);

    await controller.assign(a, 999);

    expect(successSpy).toHaveBeenCalledWith('Labeled "999".');
  });

  it('restores the item at its original index and cursor when the write fails', async () => {
    vi.mocked(putCropLabel).mockRejectedValue(new Error('network down'));
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'toast-id');
    const a = item('a');
    const b = item('b');
    const c = item('c');
    const { queue, controller, handledIds, getCursor } = setup([a, b, c]);

    await controller.assign(b, 5);

    expect(queue.items.map((i) => i.id)).toEqual(['a', 'b', 'c']);
    expect(queue.total).toBe(3);
    expect(handledIds.has('b')).toBe(false);
    expect(getCursor()).toBe(0);
    expect(errorSpy).toHaveBeenCalledWith('Label failed: network down');
  });
});

describe('removeFromQueue', () => {
  it('claims the id, calls maybePrefetch, and clamps the cursor to the new max index', () => {
    const a = item('a');
    const b = item('b');
    const c = item('c');
    const queue = createPager<ReviewItem>({
      fetchPage: async () => ({ items: [], total: 0 }),
      keyOf: (i) => i.id,
    });
    queue.items = [a, b, c];
    queue.total = 3;
    const handledIds = new Set<string>();
    let cursor = 2; // pointing at 'c', the last item
    const maybePrefetch = vi.fn();
    const controller = createReviewQueueController({
      queue,
      handledIds,
      getCursor: () => cursor,
      setCursor: (v) => {
        cursor = v;
      },
      maybePrefetch,
    });

    controller.removeFromQueue(c);

    expect(queue.items.map((i) => i.id)).toEqual(['a', 'b']);
    expect(handledIds.has('c')).toBe(true);
    expect(maybePrefetch).toHaveBeenCalledTimes(1);
    // Cursor was at 2 (now past the end of a 2-item queue) -> clamps to 1.
    expect(cursor).toBe(1);
  });

  it('uses the found index (including index 0) rather than always falling back to the cursor', () => {
    const a = item('a');
    const b = item('b');
    const queue = createPager<ReviewItem>({
      fetchPage: async () => ({ items: [], total: 0 }),
      keyOf: (i) => i.id,
    });
    queue.items = [a, b];
    queue.total = 2;
    let cursor = 1; // cursor points at 'b', but we remove 'a' (index 0)
    const controller = createReviewQueueController({
      queue,
      handledIds: new Set(),
      getCursor: () => cursor,
      setCursor: (v) => {
        cursor = v;
      },
      maybePrefetch: () => {},
    });

    const restore = controller.removeFromQueue(a);
    restore();

    // Restored at index 0 (where 'a' actually was), not at the cursor (1).
    expect(queue.items.map((i) => i.id)).toEqual(['a', 'b']);
  });

  it('restore() puts the cursor back to its exact pre-removal value', () => {
    const a = item('a');
    const b = item('b');
    const c = item('c');
    const queue = createPager<ReviewItem>({
      fetchPage: async () => ({ items: [], total: 0 }),
      keyOf: (i) => i.id,
    });
    queue.items = [a, b, c];
    queue.total = 3;
    let cursor = 2;
    const controller = createReviewQueueController({
      queue,
      handledIds: new Set(),
      getCursor: () => cursor,
      setCursor: (v) => {
        cursor = v;
      },
      maybePrefetch: () => {},
    });

    const restore = controller.removeFromQueue(a);
    cursor = 0; // simulate the operator moving around while the request is in flight
    restore();

    expect(cursor).toBe(2);
  });

  it('falls back to the current cursor as the restore index when the item is not in the queue', () => {
    const a = item('a');
    const b = item('b');
    const stray = item('stray');
    const queue = createPager<ReviewItem>({
      fetchPage: async () => ({ items: [], total: 0 }),
      keyOf: (i) => i.id,
    });
    // Two items (not one) so a `found` value of -1 (leaking through
    // instead of falling back to getCursor()) produces a distinguishably
    // different splice point rather than coincidentally matching.
    queue.items = [a, b];
    queue.total = 2;
    const handledIds = new Set<string>();
    let cursor = 0;
    const controller = createReviewQueueController({
      queue,
      handledIds,
      getCursor: () => cursor,
      setCursor: (v) => {
        cursor = v;
      },
      maybePrefetch: () => {},
    });

    const restore = controller.removeFromQueue(stray);
    // total still drops even though the item was never actually present.
    expect(queue.total).toBe(1);
    restore();
    // Restored at the fallback index (the cursor at claim time, 0), ahead
    // of 'a' — proves `found >= 0 ? found : getCursor()` chose the
    // getCursor() branch rather than leaking the -1 "not found" sentinel
    // through as the splice index.
    expect(queue.items.map((i) => i.id)).toEqual(['stray', 'a', 'b']);
  });
});

describe('discard', () => {
  it('dismisses the item and toasts success', async () => {
    vi.mocked(reviewDismissCrop).mockResolvedValue(undefined as never);
    const successSpy = vi
      .spyOn(toastStore, 'success')
      .mockImplementation(() => 'toast-id');
    const a = item('a');
    const { queue, controller } = setup([a]);

    await controller.discard(a);

    expect(queue.items).toEqual([]);
    expect(reviewDismissCrop).toHaveBeenCalledWith('a');
    expect(successSpy).toHaveBeenCalledWith('Dismissed from review (permanent).');
  });

  it('restores the item and toasts the error message when the dismiss call fails', async () => {
    vi.mocked(reviewDismissCrop).mockRejectedValue(new Error('boom'));
    const errorSpy = vi.spyOn(toastStore, 'error').mockImplementation(() => 'toast-id');
    const a = item('a');
    const b = item('b');
    const { queue, controller } = setup([a, b]);

    await controller.discard(a);

    expect(queue.items.map((i) => i.id)).toEqual(['a', 'b']);
    expect(queue.total).toBe(2);
    expect(errorSpy).toHaveBeenCalledWith('Discard failed: boom');
  });
});

describe('skip', () => {
  it('advances the cursor by one and calls maybePrefetch, without touching the queue', () => {
    const a = item('a');
    const b = item('b');
    const c = item('c');
    const queue = createPager<ReviewItem>({
      fetchPage: async () => ({ items: [], total: 0 }),
      keyOf: (i) => i.id,
    });
    queue.items = [a, b, c];
    queue.total = 3;
    let cursor = 0;
    const maybePrefetch = vi.fn();
    const controller = createReviewQueueController({
      queue,
      handledIds: new Set(),
      getCursor: () => cursor,
      setCursor: (v) => {
        cursor = v;
      },
      maybePrefetch,
    });

    controller.skip();

    expect(cursor).toBe(1);
    expect(maybePrefetch).toHaveBeenCalledTimes(1);
    expect(queue.items.map((i) => i.id)).toEqual(['a', 'b', 'c']);
  });

  it('does not advance past the last item', () => {
    const { controller, getCursor } = setup([item('a')]);

    controller.skip();
    controller.skip();

    expect(getCursor()).toBe(0);
  });
});

// Stryker flags `if (crops.length === 0) return;` (reviewController
// .svelte.ts:103) as a surviving mutant when replaced with `if (false)
// return;`. This is an accepted equivalent mutant, not a coverage gap:
// with `crops` empty, `for (const crop of crops)` below is a no-op
// either way, so removing the early-return guard cannot produce any
// observable difference. Left in the source as a documented fast path,
// not because a test can distinguish it.
describe('undoLast', () => {
  it('re-inserts the restored item at the cursor and un-marks it handled', async () => {
    vi.mocked(undoCropLabel).mockResolvedValue(item('a'));
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'toast-id');
    const a = item('a');
    const b = item('b');
    const c = item('c');
    const { queue, controller, handledIds } = setup([b, c]);
    handledIds.add('a');
    undoStore.recordWrites(['a']);

    await controller.undoLast();

    expect(queue.items.map((i) => i.id)).toEqual(['a', 'b', 'c']);
    expect(handledIds.has('a')).toBe(false);
    expect(queue.total).toBe(3);
    void a; // constructed only to document the pre-undo shape above
  });

  it('is a no-op when there is nothing to undo', async () => {
    const { queue, controller } = setup([item('a')]);
    vi.spyOn(toastStore, 'info').mockImplementation(() => 'toast-id');

    await controller.undoLast();

    expect(queue.items.map((i) => i.id)).toEqual(['a']);
  });

  it('m2 (2026-09-24 interactive pass): restores the item’s own served reason after assign+undo, never an invented "restored by undo" string', async () => {
    vi.mocked(putCropLabel).mockResolvedValue({} as never);
    vi.spyOn(classesStore, 'byId').mockReturnValue({ id: 3, name: 'widget_a' } as never);
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'toast-id');
    const a = { ...item('a'), reason: 'mistakenness_score high' };
    const b = item('b');
    const { controller, queue } = setup([a, b]);

    await controller.assign(a, 3);
    // undoCropLabel's mock resolves a bare Crop shape with no `reason` at
    // all — exactly what the real API returns (ReviewItem's `reason` is
    // a queue-only field). The controller must recover 'a's original
    // reason from its own cache, not from this response.
    vi.mocked(undoCropLabel).mockResolvedValue({ id: 'a' } as never);
    undoStore.recordWrites(['a']);

    await controller.undoLast();

    const restored = queue.items.find((i) => i.id === 'a');
    expect(restored?.reason).toBe('mistakenness_score high');
  });

  it('m2: falls back to null (never a fabricated string) when the removed-item cache has no entry for the restored id', async () => {
    vi.mocked(undoCropLabel).mockResolvedValue({ id: 'z' } as never);
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'toast-id');
    const a = item('a');
    const { controller, queue } = setup([a]);
    // 'z' was never removed via this controller (e.g. undone from a
    // different session/page) — no cache entry exists for it.
    undoStore.recordWrites(['z']);

    await controller.undoLast();

    const restored = queue.items.find((i) => i.id === 'z');
    expect(restored?.reason).toBeNull();
  });

  it('inserts each restored item at the cursor, not always at the end', async () => {
    vi.mocked(undoCropLabel).mockResolvedValue(item('z'));
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'toast-id');
    const a = item('a');
    const b = item('b');
    const { queue, controller } = setup([a, b]);
    undoStore.recordWrites(['z']);

    await controller.undoLast();

    // cursor starts at 0 -> restored 'z' lands BEFORE 'a', not appended
    // after 'b'. Kills the `without` (no-op filter) and `at` mutants.
    expect(queue.items.map((i) => i.id)).toEqual(['z', 'a', 'b']);
  });

  it('inserts at a non-zero cursor without duplicating items after it', async () => {
    vi.mocked(undoCropLabel).mockResolvedValue(item('z'));
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'toast-id');
    const a = item('a');
    const b = item('b');
    const c = item('c');
    const queue = createPager<ReviewItem>({
      fetchPage: async () => ({ items: [], total: 0 }),
      keyOf: (i) => i.id,
    });
    queue.items = [a, b, c];
    queue.total = 3;
    let cursor = 1;
    const controller = createReviewQueueController({
      queue,
      handledIds: new Set(),
      getCursor: () => cursor,
      setCursor: (v) => {
        cursor = v;
      },
      maybePrefetch: () => {},
    });
    undoStore.recordWrites(['z']);

    await controller.undoLast();

    // Inserted between 'a' and 'b' (at=1), and cursor moved to that index —
    // NOT ['a', 'z', 'a', 'b', 'c'], which is what dropping the
    // `.slice(at)` on the tail half would produce.
    expect(queue.items.map((i) => i.id)).toEqual(['a', 'z', 'b', 'c']);
    expect(cursor).toBe(1);
  });

  it('moves the cursor to the clamped insert index, not just wherever it already was', async () => {
    vi.mocked(undoCropLabel).mockResolvedValue(item('z'));
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'toast-id');
    const a = item('a');
    const b = item('b');
    const queue = createPager<ReviewItem>({
      fetchPage: async () => ({ items: [], total: 0 }),
      keyOf: (i) => i.id,
    });
    queue.items = [a, b];
    queue.total = 2;
    let cursor = 5; // stale/out-of-range cursor
    const controller = createReviewQueueController({
      queue,
      handledIds: new Set(),
      getCursor: () => cursor,
      setCursor: (v) => {
        cursor = v;
      },
      maybePrefetch: () => {},
    });
    undoStore.recordWrites(['z']);

    await controller.undoLast();

    // at = Math.min(cursor=5, without.length=2) = 2, distinct from the
    // pre-call cursor (5) — a dropped `setCursor(at)` call would leave
    // cursor at 5 instead.
    expect(cursor).toBe(2);
  });

  it('increments total only when the restored id was actually missing from the queue', async () => {
    vi.mocked(undoLabelBatch).mockResolvedValue({
      undone: 2,
      nothing_to_undo: [],
      conflicts: [],
      items: [item('a'), item('c')],
    } as never);
    vi.spyOn(toastStore, 'success').mockImplementation(() => 'toast-id');
    // 'a' is still present in the queue (never actually removed this
    // session); 'c' was removed earlier and is absent.
    const a = item('a');
    const b = item('b');
    const { queue, controller } = setup([a, b]);
    undoStore.recordWrites(['a', 'c']);

    await controller.undoLast();

    expect(queue.total).toBe(3); // started at 2, +1 only for 'c'
    expect(queue.items.map((i) => i.id).sort()).toEqual(['a', 'b', 'c']);
  });
});
