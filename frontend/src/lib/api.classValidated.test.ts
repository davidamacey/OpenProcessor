/**
 * G2: `label_validated` on the wire is `class_validated OR
 * region_validated` (wire.py:105) — 125/422 live crops have
 * label_validated=true with class_validated=false (e.g. crop
 * `67fd954c…`, class_source=v6_model, region_status=no_region_visible).
 * mapRawCrop must carry `class_validated` through as its own field so
 * class-label UI can tell the difference.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { getCrop } from './api';

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('mapRawCrop class_validated', () => {
  it('carries class_validated independently of the OR-combined label_validated', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        crop_id: '67fd954c',
        image_path: '/img/1.jpg',
        bbox_norm: [0, 0, 0.4, 0.2],
        class_source: 'v6_model',
        label_validated: true,
        class_validated: false,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const crop = await getCrop('67fd954c');

    expect(crop.label_validated).toBe(true);
    expect(crop.class_validated).toBe(false);
  });

  it('defaults class_validated to false when the wire omits it', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        crop_id: 'c2',
        image_path: '/img/2.jpg',
        bbox_norm: [0, 0, 0.4, 0.2],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const crop = await getCrop('c2');
    expect(crop.class_validated).toBe(false);
  });
});
