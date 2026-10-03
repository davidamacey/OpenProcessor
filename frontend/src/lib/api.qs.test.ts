/**
 * `qs()`: FastAPI list query params (`class_name`, `origin`,
 * `embedding_state`, `review_status`, ...) are read from repeated keys, so
 * an array value must be sent as one `k=v` per element, never as a
 * comma-joined `String(array)`. Also pinned through `getCrops`, the first
 * route whose filter carries list params.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { getCrops, qs } from '$lib/api';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('qs', () => {
  it('sends an array value as repeated keys', () => {
    expect(qs({ a: ['x', 'y'] })).toBe('?a=x&a=y');
  });

  it('omits an empty array', () => {
    expect(qs({ a: [] })).toBe('');
    expect(qs({ a: [], b: 2 })).toBe('?b=2');
  });

  it('keeps scalars and skips null / undefined', () => {
    expect(qs({ a: 1, b: null, c: undefined })).toBe('?a=1');
    expect(qs({ a: false, b: 'p q' })).toBe('?a=false&b=p+q');
  });

  it('keeps repeats in order beside scalars', () => {
    expect(qs({ page: 1, class_name: ['a b', 'c'], conf_min: 0.5 })).toBe(
      '?page=1&class_name=a+b&class_name=c&conf_min=0.5',
    );
  });
});

describe('getCrops sends list filters as repeated query keys', () => {
  it('class_name / origin / review_status repeat', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ crops: [], total: 0, page: 1, page_size: 50 }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    await getCrops({
      class_name: ['widget', 'gadget'],
      origin: ['detector', 'human'],
      review_status: [],
      embedding_state: ['failed'],
    });
    const url = new URL(String(fetchMock.mock.calls[0]![0]), 'http://x');
    expect(url.searchParams.getAll('class_name')).toEqual(['widget', 'gadget']);
    expect(url.searchParams.getAll('origin')).toEqual(['detector', 'human']);
    expect(url.searchParams.getAll('embedding_state')).toEqual(['failed']);
    expect(url.searchParams.has('review_status')).toBe(false);
  });
});
