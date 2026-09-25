/**
 * #36 item 8 — POST {API_PREFIX}/probe/run, GET {API_PREFIX}/probe/status,
 * POST {API_PREFIX}/probe/cancel.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { API_PREFIX, cancelProbe, getProbeStatus, runProbe } from './api';

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('runProbe', () => {
  it('POSTs {job_id} to {API_PREFIX}/probe/run and returns the served status verbatim', async () => {
    const payload = { status: 'running', job_id: 'probe-1', train_job_id: 'train-1' };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal('fetch', fetchMock);

    const res = await runProbe('train-1');

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/probe/run`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ job_id: 'train-1' });
    expect(res).toEqual(payload);
  });

  it('forwards architecture/gpu/resume when given', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ status: 'running' }));
    vi.stubGlobal('fetch', fetchMock);

    await runProbe('train-1', { architecture: 'yolo26', gpu: '0', resume: true });

    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body as string)).toEqual({
      job_id: 'train-1',
      architecture: 'yolo26',
      gpu: '0',
      resume: true,
    });
  });
});

describe('getProbeStatus', () => {
  it('GETs {API_PREFIX}/probe/status', async () => {
    const payload = { status: 'idle' };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal('fetch', fetchMock);

    const res = await getProbeStatus();

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/probe/status`);
    expect(res).toEqual(payload);
  });
});

describe('cancelProbe', () => {
  it('POSTs {API_PREFIX}/probe/cancel', async () => {
    const payload = { status: 'cancelled' };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal('fetch', fetchMock);

    const res = await cancelProbe();

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_PREFIX}/probe/cancel`);
    expect(init.method).toBe('POST');
    expect(res).toEqual(payload);
  });
});
