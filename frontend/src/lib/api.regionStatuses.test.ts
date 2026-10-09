/**
 * W2 (docs/design/logic-moves-adoption-plan-2026-09-24.md) — the
 * deployment's region-status vocabulary, served whole by
 * `GET {API_PREFIX}/regions/statuses` and consumed by
 * `$stores/regionStatuses.svelte`.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { getRegionStatuses, API_PREFIX } from './api';

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('getRegionStatuses', () => {
  it('GETs {API_PREFIX}/regions/statuses and returns the response verbatim', async () => {
    const payload = {
      statuses: [
        {
          value: 'detected',
          label: 'detected',
          role: 'positive',
          terminal: true,
          human_writable: true,
          clears_box: false,
          wants_reason: false,
        },
      ],
      confirm_status: 'detected',
      reject_status: 'no_region_visible',
      false_positive_status: 'false_positive',
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal('fetch', fetchMock);
    const res = await getRegionStatuses();
    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/regions/statuses`);
    expect(res).toEqual(payload);
  });
});
