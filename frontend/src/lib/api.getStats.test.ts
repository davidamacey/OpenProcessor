/**
 * G1: getStats() must use Promise.allSettled so a /stats/dataset failure
 * still returns per_class from /stats/classes — /export renders its class
 * table off StatsSummary.per_class even when the dataset rollup 503s
 * (live: op_items' region_status mapping isn't aggregatable).
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, getStats } from './api';

const ok = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('getStats', () => {
  it('still returns per_class when /stats/dataset 503s', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/stats/dataset')) {
        return Promise.resolve(
          new Response(JSON.stringify({ error: 'boom' }), { status: 503 }),
        );
      }
      return Promise.resolve(
        ok({
          classes: [{ class_id: 1, class_name: 'sedan', count: 10, validated_count: 4 }],
        }),
      );
    });
    vi.stubGlobal('fetch', fetchMock);

    const result = await getStats();

    expect(result.per_class).toEqual([
      { class_id: 1, class_name: 'sedan', count: 10, validated_count: 4 },
    ]);
    expect(result.total_crops).toBe(0);
    expect(result.dataset_error).toMatch(/503/);
  }, 10_000);

  it('still returns dataset totals when /stats/classes fails', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/stats/classes')) {
        return Promise.resolve(new Response('boom', { status: 500 }));
      }
      return Promise.resolve(
        ok({ total_crops: 422, validated: 297, test_holdout: 12, by_source: [] }),
      );
    });
    vi.stubGlobal('fetch', fetchMock);

    const result = await getStats();

    expect(result.total_crops).toBe(422);
    expect(result.dataset_error).toBeNull();
    expect(result.per_class).toEqual([]);
  }, 10_000);

  it('merges both on a healthy backend', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes(`${API_PREFIX}/stats/dataset`)) {
        return Promise.resolve(
          ok({
            total_crops: 5,
            validated: 2,
            test_holdout: 0,
            by_source: [{ key: 'a', doc_count: 5 }],
          }),
        );
      }
      return Promise.resolve(ok({ classes: [] }));
    });
    vi.stubGlobal('fetch', fetchMock);

    const result = await getStats();
    expect(result.total_crops).toBe(5);
    expect(result.ingestion.images_processed).toBe(5);
  });

  // W4 (docs/design/logic-moves-adoption-plan-2026-09-24.md §1.7): thresholds
  // are served on `/stats/classes` too, and each class row carries
  // server-computed adequacy/aug_target/aug_gap — `/export` and the
  // dashboard read these directly rather than clamping or deriving a tier.
  it('passes adequacy/aug_target/aug_gap per class and the top-level thresholds through untouched', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes(`${API_PREFIX}/stats/dataset`)) {
        return Promise.resolve(new Response('boom', { status: 503 }));
      }
      return Promise.resolve(
        ok({
          classes: [
            {
              class_id: 8,
              class_name: 'bmw',
              count: 8,
              validated_count: 8,
              adequacy: 'block',
              aug_target: 500,
              aug_gap: 492,
            },
          ],
          thresholds: {
            block_below: 20,
            warn_below: 500,
            min_test_per_class: 5,
            aug_target_min: 500,
            aug_target_max: 3000,
          },
        }),
      );
    });
    vi.stubGlobal('fetch', fetchMock);

    const result = await getStats();

    expect(result.per_class[0]).toMatchObject({
      class_id: 8,
      adequacy: 'block',
      aug_target: 500,
      aug_gap: 492,
    });
    expect(result.thresholds).toEqual({
      block_below: 20,
      warn_below: 500,
      min_test_per_class: 5,
      aug_target_min: 500,
      aug_target_max: 3000,
    });
  });
});
