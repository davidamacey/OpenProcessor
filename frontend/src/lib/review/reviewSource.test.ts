import { beforeEach, describe, expect, it, vi } from 'vitest';
import { reviewSource } from '$lib/review/reviewSource.svelte';

const getCrops = vi.hoisted(() => vi.fn());
const locateCrop = vi.hoisted(() => vi.fn());
const getReviewQueue = vi.hoisted(() => vi.fn());
const locateInReviewQueue = vi.hoisted(() => vi.fn());

vi.mock('$lib/api_cropRead', () => ({ getCrops, locateCrop }));
vi.mock('$lib/api_review', () => ({ getReviewQueue, locateInReviewQueue }));

const page = { items: [{ id: 'a' }], next_cursor: 'c2', total: 9, limit: 30 };

beforeEach(() => {
  vi.resetAllMocks();
  getReviewQueue.mockResolvedValue(page);
  getCrops.mockResolvedValue(page);
});

describe('reviewSource', () => {
  it('uses the queue route when a queue is set', async () => {
    const source = reviewSource({
      queue: 'to_label',
      filter: { class_name: ['car'] },
      sort: 'confidence:asc',
    });
    await source.page('c1', 30);
    expect(source.kind).toBe('queue');
    expect(getReviewQueue).toHaveBeenCalledWith(
      'to_label',
      'c1',
      30,
      { class_name: ['car'], sort: 'confidence:asc' },
      undefined,
    );
    expect(getCrops).not.toHaveBeenCalled();
  });

  it('uses crops with the detections sort otherwise', async () => {
    const source = reviewSource({
      queue: null,
      filter: { class_name: ['car'] },
      sort: 'detector_confidence:desc',
    });
    const served = await source.page(null, 30);
    expect(source.kind).toBe('crops');
    expect(getCrops).toHaveBeenCalledWith(
      { class_name: ['car'], sort: 'detector_confidence:desc', limit: 30 },
      null,
      undefined,
    );
    expect(getReviewQueue).not.toHaveBeenCalled();
    expect(served.items).toEqual([{ id: 'a', reason: null }]);
  });

  it('takes the served default order when no sort is chosen', async () => {
    await reviewSource({ queue: 'audit', filter: {}, sort: null }).page(null, 5);
    expect(getReviewQueue.mock.calls[0]?.[3]).toEqual({});
  });

  it('locates the focused item with one request on a queue', async () => {
    locateInReviewQueue.mockResolvedValue({
      in_queue: true,
      index: 11,
      total: 412,
      cursor_before: 'cb',
      cursor_after: 'ca',
      reason: null,
    });
    const at = await reviewSource({ queue: 'to_label', filter: {}, sort: null }).locate(
      'x1',
    );
    expect(locateInReviewQueue).toHaveBeenCalledTimes(1);
    expect(locateCrop).not.toHaveBeenCalled();
    expect(at).toEqual({
      found: true,
      index: 11,
      total: 412,
      cursorBefore: 'cb',
      reason: null,
      reasonMessage: null,
    });
  });

  it('locates the focused item with one request on crops and keeps the served sentence', async () => {
    locateCrop.mockResolvedValue({
      in_view: false,
      index: null,
      total: 40,
      cursor_before: null,
      cursor_after: null,
      reason: 'filtered_out',
      reason_message: 'It does not match the filter.',
    });
    const at = await reviewSource({
      queue: null,
      filter: { class_name: ['car'] },
      sort: 'updated_at:desc',
    }).locate('x1');
    expect(locateCrop).toHaveBeenCalledTimes(1);
    expect(locateCrop).toHaveBeenCalledWith(
      'x1',
      { class_name: ['car'], sort: 'updated_at:desc' },
      undefined,
    );
    expect(locateInReviewQueue).not.toHaveBeenCalled();
    expect(at.found).toBe(false);
    expect(at.reason).toBe('filtered_out');
    expect(at.reasonMessage).toBe('It does not match the filter.');
  });
});
