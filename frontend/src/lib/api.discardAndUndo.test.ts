/**
 * W1 (docs/design/logic-moves-adoption-plan-2026-09-24.md) — discard,
 * VLM dismiss, review un-dismiss and the region-statuses vocabulary.
 * Each is a thin wrapper: verify the request shape and that the
 * response is unwrapped/mapped, not passed through raw.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  ApiError,
  bulkLabel,
  discardCrop,
  discardCropsBatch,
  moveCropsToCluster,
  vlmDismissCrop,
  reviewUndismissCrop,
  API_PREFIX,
} from './api';

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function rawItem(id: string): Record<string, unknown> {
  return { crop_id: id, image_path: `/img/${id}.jpg`, bbox_norm: [0, 0, 1, 1] };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('bulkLabel / moveCropsToCluster: updated_ids passes through verbatim', () => {
  it('bulkLabel returns the served updated_ids, not a locally-derived list', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        updated: 1,
        updated_ids: ['a'],
        conflicts: [{ crop_id: 'b', current_source: 'vlm' }],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const res = await bulkLabel(['a', 'b'], 3);
    expect(res.updated_ids).toEqual(['a']);
    expect(res.conflicts).toEqual([{ crop_id: 'b', current_source: 'vlm' }]);
  });

  it('moveCropsToCluster returns the served updated_ids', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ updated: 2, updated_ids: ['a', 'b'], conflicts: [] }),
      );
    vi.stubGlobal('fetch', fetchMock);
    const res = await moveCropsToCluster(['a', 'b'], 42);
    expect(res.updated_ids).toEqual(['a', 'b']);
  });
});

describe('discardCrop', () => {
  it('POSTs {clear_class, dismiss_from_review} and returns the mapped item', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(rawItem('c1')));
    vi.stubGlobal('fetch', fetchMock);

    const crop = await discardCrop('c1', {
      clear_class: true,
      dismiss_from_review: false,
    });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/c1/discard`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({
      clear_class: true,
      dismiss_from_review: false,
    });
    expect(crop.id).toBe('c1');
  });

  it('defaults to an empty body when no options are passed', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(rawItem('c1')));
    vi.stubGlobal('fetch', fetchMock);
    await discardCrop('c1');
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({});
  });
});

describe('discardCropsBatch', () => {
  it('POSTs crop_ids + opts and maps every returned item', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        items: [rawItem('a'), rawItem('b')],
        discarded: 2,
        conflicts: [],
        not_found: [],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await discardCropsBatch(['a', 'b'], { clear_class: true });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/discard_batch`);
    expect(JSON.parse(init.body)).toEqual({ crop_ids: ['a', 'b'], clear_class: true });
    expect(res.discarded).toBe(2);
    expect(res.items.map((c) => c.id)).toEqual(['a', 'b']);
  });
});

describe('vlmDismissCrop', () => {
  it('POSTs to vlm_dismiss and returns the mapped item', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(rawItem('c1')));
    vi.stubGlobal('fetch', fetchMock);
    const crop = await vlmDismissCrop('c1');
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/c1/vlm_dismiss`);
    expect(init.method).toBe('POST');
    expect(crop.id).toBe('c1');
  });

  it('a 409 (no suggestion) rejects with an ApiError the caller can branch on', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ detail: 'no VLM suggestion' }, 409));
    vi.stubGlobal('fetch', fetchMock);
    await expect(vlmDismissCrop('c1')).rejects.toThrow(ApiError);
    try {
      await vlmDismissCrop('c1');
    } catch (e) {
      expect((e as ApiError).status).toBe(409);
    }
  });
});

describe('reviewUndismissCrop', () => {
  it('POSTs to review_undismiss and returns the mapped item', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(rawItem('c1')));
    vi.stubGlobal('fetch', fetchMock);
    const crop = await reviewUndismissCrop('c1');
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/crops/c1/review_undismiss`);
    expect(init.method).toBe('POST');
    expect(crop.id).toBe('c1');
  });
});

describe('strict batch bodies (OpenProcessor d72cc63): an empty list is never sent', () => {
  it('discardCropsBatch / undo batches / ingestBatch reject [] without calling fetch', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const { undoLabelBatch, undoCropRegionBatch, ingestBatch } = await import('./api');
    await expect(discardCropsBatch([])).rejects.toThrow(/no request sent/);
    await expect(undoLabelBatch([])).rejects.toThrow(/no request sent/);
    await expect(undoCropRegionBatch([])).rejects.toThrow(/no request sent/);
    await expect(ingestBatch({ items: [] })).rejects.toThrow(/no request sent/);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
