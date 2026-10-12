import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX } from './api';
import {
  cancelBulkIngest,
  getBulkIngest,
  getBulkIngestConfig,
  getBulkIngestFailures,
  listBulkIngests,
  startBulkIngest,
} from './api_bulkIngest';

const json = (body: unknown, init: ResponseInit = {}) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
    ...init,
  });

function stub(body: unknown = {}) {
  const fetchMock = vi.fn().mockImplementation(async () => json(body));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

afterEach(() => vi.unstubAllGlobals());

describe('bulk ingest wrappers', () => {
  it('start POSTs the request body to /ingest/bulk', async () => {
    const f = stub({ run_id: 'r' });
    await startBulkIngest({ folder: '/data/source/a', source: 's', batch_size: 8 });
    const [url, init] = f.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${API_PREFIX}/ingest/bulk`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({
      folder: '/data/source/a',
      source: 's',
      batch_size: 8,
    });
  });

  it('reads the config, the list, one run and its failures', async () => {
    const f = stub({});
    await getBulkIngestConfig();
    await listBulkIngests({ page: 2, page_size: 5 });
    await getBulkIngest('bulk_x');
    await getBulkIngestFailures('bulk_x', { offset: 20, limit: 20 });
    const urls = f.mock.calls.map((c) => c[0]);
    expect(urls).toEqual([
      `${API_PREFIX}/ingest/bulk/config`,
      `${API_PREFIX}/ingest/bulk?page=2&page_size=5`,
      `${API_PREFIX}/ingest/bulk/bulk_x`,
      `${API_PREFIX}/ingest/bulk/bulk_x/failures?offset=20&limit=20`,
    ]);
  });

  it('cancel POSTs to the run', async () => {
    const f = stub({});
    await cancelBulkIngest('bulk_x');
    const [url, init] = f.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${API_PREFIX}/ingest/bulk/bulk_x/cancel`);
    expect(init.method).toBe('POST');
  });

  it('keeps the served error code on a refused start', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        json(
          {
            detail: {
              error: 'bulk_ingest_busy',
              message: 'a bulk ingest is already running',
            },
          },
          { status: 409 },
        ),
      ),
    );
    await expect(startBulkIngest({ folder: '/x' })).rejects.toMatchObject({
      status: 409,
    });
  });
});
