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

import { putCropLabel, reviewDismissCrop, undoCropLabel } from '$lib/api';
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
    vi.spyOn(classesStore, 'byId').mockReturnValue({ id: 3, name: 'sedan' } as never);
    const successSpy = vi
      .spyOn(toastStore, 'success')
      .mockImplementation(() => 'toast-id');
    const a = item('a');
    const b = item('b');
    const { queue, controller, handledIds } = setup([a, b]);

    await controller.assign(a, 3);

    expect(queue.items.map((i) => i.id)).toEqual(['b']);
    expect(queue.total).toBe(1);
    expect(handledIds.has('a')).toBe(true);
    expect(putCropLabel).toHaveBeenCalledWith('a', 3);
    expect(successSpy).toHaveBeenCalledWith(expect.stringContaining('sedan'));
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
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining('network down'));
  });
});

describe('discard', () => {
  it('dismisses the item on success', async () => {
    vi.mocked(reviewDismissCrop).mockResolvedValue(undefined as never);
    const a = item('a');
    const { queue, controller } = setup([a]);

    await controller.discard(a);

    expect(queue.items).toEqual([]);
    expect(reviewDismissCrop).toHaveBeenCalledWith('a');
  });

  it('restores the item when the dismiss call fails', async () => {
    vi.mocked(reviewDismissCrop).mockRejectedValue(new Error('boom'));
    vi.spyOn(toastStore, 'error').mockImplementation(() => 'toast-id');
    const a = item('a');
    const b = item('b');
    const { queue, controller } = setup([a, b]);

    await controller.discard(a);

    expect(queue.items.map((i) => i.id)).toEqual(['a', 'b']);
    expect(queue.total).toBe(2);
  });
});

describe('skip', () => {
  it('advances the cursor by one without touching the queue', () => {
    const { controller, queue, getCursor } = setup([item('a'), item('b'), item('c')]);

    controller.skip();

    expect(getCursor()).toBe(1);
    expect(queue.items.map((i) => i.id)).toEqual(['a', 'b', 'c']);
  });

  it('does not advance past the last item', () => {
    const { controller, getCursor } = setup([item('a')]);

    controller.skip();
    controller.skip();

    expect(getCursor()).toBe(0);
  });
});

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
});
