/**
 * Tests for the `/ingest` API wrappers
 * (docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §B.2/§B.3):
 * URLs, methods, the multipart request shape, and error surfacing.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  API_PREFIX,
  ApiError,
  getIngestStatus,
  getRegionDrain,
  ingestBatch,
  ingestPathLookup,
  ingestUpload,
} from './api';

const jsonResponse = (body: unknown, init: ResponseInit = {}) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
    ...init,
  });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('getIngestStatus', () => {
  it('GETs {API_PREFIX}/ingest/status', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ total: 3, by_source: [], by_day: [] }));
    vi.stubGlobal('fetch', fetchMock);
    const result = await getIngestStatus();
    expect(result.total).toBe(3);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${API_PREFIX}/ingest/status`);
    expect(init.method ?? 'GET').toBe('GET');
  });
});

describe('getRegionDrain', () => {
  it('GETs {API_PREFIX}/ingest/region_drain', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        pending_detection: 1,
        pending_verification: 2,
        total_unfinished: 3,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const result = await getRegionDrain();
    expect(result.total_unfinished).toBe(3);
    const [url] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${API_PREFIX}/ingest/region_drain`);
  });

  it('surfaces a served 503 detail', async () => {
    const fetchMock = vi
      .fn()
      .mockImplementation(
        async () =>
          new Response(JSON.stringify({ detail: 'opensearch outage' }), { status: 503 }),
      );
    vi.stubGlobal('fetch', fetchMock);
    await expect(getRegionDrain()).rejects.toMatchObject({ detail: 'opensearch outage' });
  });
});

describe('ingestPathLookup', () => {
  it('POSTs { image_paths } in identifier order', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ known_paths: {} }));
    vi.stubGlobal('fetch', fetchMock);
    await ingestPathLookup(['a/1.jpg', 'a/2.jpg']);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${API_PREFIX}/ingest/path_lookup`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({
      image_paths: ['a/1.jpg', 'a/2.jpg'],
    });
  });
});

describe('ingestUpload', () => {
  it('sends a multipart body with images repeated, image_paths as JSON, and source', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        status: 'success',
        summary: {
          successful: 1,
          duplicates: 0,
          failed: 0,
          mismatches: 0,
          missed_labels: 0,
          unmatched_detections: 0,
          labels_imported: 0,
          crops_indexed: 1,
        },
        results: [],
        disagreements: [],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const f1 = new File(['a'], 'a.jpg', { type: 'image/jpeg' });
    const f2 = new File(['b'], 'b.jpg', { type: 'image/jpeg' });
    await ingestUpload({
      files: [f1, f2],
      identifiers: ['src/a.jpg', 'src/b.jpg'],
      source: 'src',
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${API_PREFIX}/ingest/upload`);
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
    const fd = init.body as FormData;
    const images = fd.getAll('images');
    expect(images).toHaveLength(2);
    expect((images[0] as File).name).toBe('a.jpg');
    expect((images[1] as File).name).toBe('b.jpg');
    expect(JSON.parse(fd.get('image_paths') as string)).toEqual([
      'src/a.jpg',
      'src/b.jpg',
    ]);
    expect(fd.get('source')).toBe('src');
  });

  it('never sets a JSON Content-Type header for a multipart body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        status: 'success',
        summary: {
          successful: 0,
          duplicates: 0,
          failed: 0,
          mismatches: 0,
          missed_labels: 0,
          unmatched_detections: 0,
          labels_imported: 0,
          crops_indexed: 0,
        },
        results: [],
        disagreements: [],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    await ingestUpload({ files: [], identifiers: [], source: 'x' });
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(headers['Content-Type']).toBeUndefined();
  });

  it('surfaces a backend 413 detail on an oversized chunk', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'too many images' }), { status: 413 }),
      );
    vi.stubGlobal('fetch', fetchMock);
    await expect(
      ingestUpload({ files: [], identifiers: [], source: 'x' }),
    ).rejects.toMatchObject({ status: 413, detail: 'too many images' });
  });

  it('surfaces a 422 validation detail', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'image_paths length mismatch' }), {
        status: 422,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    await expect(
      ingestUpload({ files: [], identifiers: [], source: 'x' }),
    ).rejects.toBeInstanceOf(ApiError);
  });

  it('surfaces a 503 (detector not configured) detail', async () => {
    const fetchMock = vi.fn().mockImplementation(
      async () =>
        new Response(JSON.stringify({ detail: 'detector not configured' }), {
          status: 503,
        }),
    );
    vi.stubGlobal('fetch', fetchMock);
    await expect(
      ingestUpload({ files: [], identifiers: [], source: 'x' }),
    ).rejects.toMatchObject({ detail: 'detector not configured' });
  });
});

describe('ingestBatch', () => {
  it('POSTs the batch items request body verbatim', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        status: 'success',
        summary: {
          successful: 0,
          duplicates: 0,
          failed: 0,
          mismatches: 0,
          missed_labels: 0,
          unmatched_detections: 0,
          labels_imported: 0,
          crops_indexed: 0,
        },
        results: [],
        disagreements: [],
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    await ingestBatch({ items: [{ path: '/data/a.jpg', source: 'nas' }] });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${API_PREFIX}/ingest/batch`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({
      items: [{ path: '/data/a.jpg', source: 'nas' }],
    });
  });
});
